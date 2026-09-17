import os
from pathlib import Path
import sys
if (_package_root := str(Path(__file__).absolute().parents[2])) not in sys.path:
    sys.path.insert(0, _package_root)

import asyncio
import json
import socket
import struct
import time
import zlib
from collections import deque

import click
import cv2
import numpy as np
import torch

from moge.model.v3 import MoGeModel

try:
    import utils3d_moge as utils3d
except ImportError:
    import utils3d

DEFAULT_PRETRAINED = r'D:\MoGe3\checkpoints\moge-3-vitl\model.pt'

# 接收侧上限：单帧最大 256MB，请求图片最大 64MB
MAX_FRAME_BYTES = 256 * 1024 * 1024
MAX_IMAGE_BYTES = 64 * 1024 * 1024

LENGTH_PREFIX = struct.Struct('<I')

# 结果数组的发送顺序（mask 为 uint8 0/1，depth/points 中无效像素为 inf，normal 中无效像素为 0）
ARRAY_ORDER = ['depth', 'points', 'normal', 'mask']


class ProtocolError(Exception):
    """请求不符合协议时抛出。"""


async def read_frame(reader: asyncio.StreamReader):
    """接收一帧：4 字节小端长度 + payload。连接在帧边界处正常关闭时返回 None。"""
    try:
        header = await reader.readexactly(LENGTH_PREFIX.size)
    except asyncio.IncompleteReadError as e:
        if len(e.partial) == 0:
            return None
        raise ConnectionError('connection closed while reading frame header')
    (length,) = LENGTH_PREFIX.unpack(header)
    if length > MAX_FRAME_BYTES:
        raise ProtocolError(f'frame too large: {length} bytes (max {MAX_FRAME_BYTES})')
    try:
        payload = await reader.readexactly(length)
    except asyncio.IncompleteReadError:
        raise ConnectionError('connection closed while reading frame payload')
    return payload


async def send_frame(writer: asyncio.StreamWriter, payload: bytes):
    writer.write(LENGTH_PREFIX.pack(len(payload)) + payload)
    await writer.drain()


async def send_json(writer: asyncio.StreamWriter, obj: dict):
    await send_frame(writer, json.dumps(obj).encode('utf-8'))


async def try_send_json(writer: asyncio.StreamWriter, obj: dict):
    try:
        await send_json(writer, obj)
    except (ConnectionError, OSError):
        pass


class InferRequest:
    """一个待推理请求：图片 CPU 张量 + 参数 + 结果 future。"""

    __slots__ = ('image', 'resolution_level', 'refine_steps', 'fov_x', 'future', 'enqueued_at', 'queue_ms')

    def __init__(self, image: torch.Tensor, resolution_level: int, refine_steps: int, fov_x):
        self.image = image                # CPU tensor (3, H, W) float32 RGB [0,1]
        self.resolution_level = resolution_level
        self.refine_steps = refine_steps
        self.fov_x = fov_x                # float 度数或 None
        self.future: asyncio.Future = None
        self.enqueued_at = 0.0
        self.queue_ms = 0.0


class MoGeService:
    """异步推理服务：asyncio 单事件循环 + 单推理 worker。

    - 每个连接一个协程，收发/解码互不阻塞
    - 推理请求进入全局队列，由唯一 worker 串行取出
    - 动态微批处理：并发请求在收集窗口内按 (分辨率, 参数, 是否带 fov) 分组，
      合成一个 batch 一次前向，提升 GPU 利用率与总吞吐
    - 显存不足 (CUDA OOM) 时自动降级为逐张推理
    """

    def __init__(self, model: MoGeModel, refine_steps: int, resolution_level: int, use_fp16: bool,
                 max_image_px: int, max_batch_size: int, batch_window_ms: int):
        self.model = model
        self.default_resolution_level = resolution_level
        self.default_refine_steps = refine_steps
        self.use_fp16 = use_fp16
        self.max_image_px = max_image_px
        self.max_batch_size = max_batch_size
        self.batch_window_ms = batch_window_ms
        self.queue: asyncio.Queue = None
        self.deferred = deque()   # 与正在收集的批 key 不匹配的请求，下一轮优先取出
        self.active_conns = 0     # 当前活跃连接数（仅 >1 时启用批收集窗口，避免单流加延迟）

    # ---------------- 连接层 ----------------

    async def handle_connection(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        self.active_conns += 1
        try:
            while True:
                header_frame = await read_frame(reader)
                if header_frame is None:
                    break
                try:
                    await self.process_request(reader, writer, header_frame)
                except (ProtocolError, ConnectionError, OSError, asyncio.IncompleteReadError) as e:
                    await try_send_json(writer, {'type': 'error', 'stage': 'request', 'message': str(e)})
                    break
                except Exception as e:
                    await try_send_json(writer, {'type': 'error', 'stage': 'inference',
                                                 'message': f'{type(e).__name__}: {e}'})
                    break
        except (ConnectionError, OSError, asyncio.IncompleteReadError):
            pass
        finally:
            self.active_conns -= 1
            writer.close()
            try:
                await writer.wait_closed()
            except OSError:
                pass

    async def process_request(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter, header_frame: bytes):
        t_start = time.perf_counter()

        header = json.loads(header_frame)
        if not isinstance(header, dict):
            raise ProtocolError('request header must be a JSON object')
        image_bytes = header.get('image_bytes')
        if not isinstance(image_bytes, int) or image_bytes <= 0:
            raise ProtocolError('missing or invalid "image_bytes" in request header')
        if image_bytes > MAX_IMAGE_BYTES:
            raise ProtocolError(f'image too large: {image_bytes} bytes (max {MAX_IMAGE_BYTES})')
        params = header.get('params') or {}
        if not isinstance(params, dict):
            raise ProtocolError('"params" must be a JSON object')

        # 读取图片 payload
        image_payload = await read_frame(reader)
        if len(image_payload) != image_bytes:
            raise ProtocolError(f'image payload length mismatch: header says {image_bytes}, got {len(image_payload)}')

        # 解码（线程池，不阻塞事件循环）
        image = await asyncio.to_thread(self._decode_image, image_payload)
        height, width = image.shape[:2]
        if height * width > self.max_image_px:
            raise ProtocolError(f'image resolution too high: {width}x{height} (max {self.max_image_px} pixels)')

        await send_json(writer, {'type': 'status', 'stage': 'received', 'width': width, 'height': height})

        # 合并请求参数与服务端默认值
        resolution_level = params.get('resolution_level', self.default_resolution_level)
        refine_steps = params.get('refine_steps', self.default_refine_steps)
        fov_x = params.get('fov_x', None)
        if not isinstance(resolution_level, int) or not (0 <= resolution_level <= 9):
            raise ProtocolError(f'invalid resolution_level: {resolution_level!r} (expected int 0-9)')
        if not isinstance(refine_steps, int) or refine_steps < 0:
            raise ProtocolError(f'invalid refine_steps: {refine_steps!r} (expected int >= 0)')
        if fov_x is not None:
            if not isinstance(fov_x, (int, float)) or not (1 < fov_x < 179):
                raise ProtocolError(f'invalid fov_x: {fov_x!r} (expected degrees in (1, 179))')
            fov_x = float(fov_x)

        await send_json(writer, {
            'type': 'status', 'stage': 'inferring',
            'params': {'resolution_level': resolution_level, 'refine_steps': refine_steps, 'fov_x': fov_x},
        })

        # 入队等待推理结果
        image_tensor = await asyncio.to_thread(self._to_tensor, image)
        req = InferRequest(image_tensor, resolution_level, refine_steps, fov_x)
        req.future = asyncio.get_running_loop().create_future()
        req.enqueued_at = time.perf_counter()
        await self.queue.put(req)
        result = await req.future  # 推理异常会在此抛出

        # 发送结果数组（浮点数组 zlib 压缩率仅 ~10% 且 CPU 开销大，直接传原始字节；mask 压缩收益高）
        for _, meta, payload in result['arrays']:
            await send_json(writer, meta)
            await send_frame(writer, payload)

        done = result['done']
        done['elapsed_ms'] = round((time.perf_counter() - t_start) * 1000, 1)
        await send_json(writer, done)

    @staticmethod
    def _decode_image(payload: bytes):
        encoded = np.frombuffer(payload, dtype=np.uint8)
        image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if image is None:
            raise ProtocolError('failed to decode image (supported formats: jpg, png)')
        return image

    @staticmethod
    def _to_tensor(image_bgr: np.ndarray) -> torch.Tensor:
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        return torch.from_numpy(image_rgb).permute(2, 0, 1).float().div_(255)

    # ---------------- 推理层 ----------------

    async def inference_worker(self):
        while True:
            # 优先取暂存的异类请求，否则等待新请求
            if self.deferred:
                req = self.deferred.popleft()
            else:
                req = await self.queue.get()
            key = self._batch_key(req)
            batch = [req]

            # 批收集窗口：多连接并发时等一小段时间让请求到齐；单连接不等待，避免增加延迟
            if self.batch_window_ms > 0 and self.active_conns >= 2:
                await asyncio.sleep(self.batch_window_ms / 1000)

            # 收集同 key 的请求：先轮转检查一遍 deferred（固定次数，防止自旋），再排空 queue。
            # 异类请求放回 deferred 队尾，下一轮优先取出，不会被饿死。
            for _ in range(len(self.deferred)):
                if len(batch) >= self.max_batch_size:
                    break
                cand = self.deferred.popleft()
                if self._batch_key(cand) == key:
                    batch.append(cand)
                else:
                    self.deferred.append(cand)
            while len(batch) < self.max_batch_size:
                try:
                    nxt = self.queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
                if self._batch_key(nxt) == key:
                    batch.append(nxt)
                else:
                    self.deferred.append(nxt)

            t_pickup = time.perf_counter()
            for r in batch:
                r.queue_ms = (t_pickup - r.enqueued_at) * 1000

            t_infer = time.perf_counter()
            try:
                results = await asyncio.to_thread(self._run_batch, batch)
            except Exception as e:
                for r in batch:
                    if not r.future.done():
                        r.future.set_exception(e)
                continue
            infer_ms = (time.perf_counter() - t_infer) * 1000

            for r, result in zip(batch, results):
                done = result['done']
                done['infer_ms'] = round(infer_ms, 1)        # 本批整体推理耗时（批内所有请求共享）
                done['queue_ms'] = round(r.queue_ms, 1)      # 排队等待耗时
                done['batch_size'] = len(batch)              # 本请求所在批的大小
                if not r.future.done():
                    r.future.set_result(result)

    @staticmethod
    def _batch_key(req: InferRequest):
        # 同批要求：分辨率与推理参数一致；fov 需全有或全无（混合时拆批）
        h, w = req.image.shape[-2], req.image.shape[-1]
        return (h, w, req.resolution_level, req.refine_steps, req.fov_x is not None)

    def _infer_single(self, req: InferRequest):
        """OOM 降级路径：单张推理。"""
        fov = torch.tensor([req.fov_x], device='cuda') if req.fov_x is not None else None
        with torch.inference_mode():
            return self.model.infer(
                req.image.unsqueeze(0).cuda(),
                resolution_level=req.resolution_level, refine_steps=req.refine_steps,
                fov_x=fov, use_fp16=self.use_fp16,
            )

    def _run_batch(self, batch):
        """在线程池中执行：组 batch -> GPU 推理 -> 拆分并序列化每个请求的结果。"""
        images = torch.stack([r.image for r in batch]).cuda()
        if batch[0].fov_x is not None:
            fov = torch.tensor([r.fov_x for r in batch], device=images.device)
        else:
            fov = None
        rl, rs = batch[0].resolution_level, batch[0].refine_steps
        try:
            with torch.inference_mode():
                output = self.model.infer(images, resolution_level=rl, refine_steps=rs, fov_x=fov, use_fp16=self.use_fp16)
        except torch.cuda.OutOfMemoryError:
            if len(batch) == 1:
                raise
            # 显存不足：清理缓存后降级为逐张推理
            torch.cuda.empty_cache()
            output = None
            results = []
            for r in batch:
                results.extend(self._pack_result(self._infer_single(r), [r]))
            return results
        return self._pack_result(output, batch)

    def _pack_result(self, output: dict, batch) -> list:
        """把批量输出 (B, ...) 拆分为每个请求的结果（numpy 数组 + 待发送字节）。"""
        B = len(batch)
        results = []
        for i, req in enumerate(batch):
            item = {k: v[i] for k, v in output.items() if isinstance(v, torch.Tensor) and v.shape[0] == B}
            arrays = {}
            for name in ARRAY_ORDER:
                if name in item:
                    arr = item[name].detach().cpu().numpy()
                    arrays[name] = arr.astype(np.uint8) if name == 'mask' else arr.astype(np.float32, copy=False)
            intrinsics = item['intrinsics'].detach().cpu().numpy()
            fov_x_rad, fov_y_rad = utils3d.np.intrinsics_to_fov(intrinsics)

            payload_arrays = []
            for name in ARRAY_ORDER:
                if name not in arrays:
                    continue
                arr = arrays[name]
                if name == 'mask':
                    compression, payload = 'zlib', zlib.compress(arr.tobytes(), level=1)
                else:
                    compression, payload = 'none', arr.tobytes()
                meta = {
                    'type': 'array', 'name': name,
                    'dtype': str(arr.dtype), 'shape': list(arr.shape),
                    'compression': compression, 'bytes': len(payload),
                }
                payload_arrays.append((name, meta, payload))

            h, w = req.image.shape[-2], req.image.shape[-1]
            done = {
                'type': 'done',
                'width': w, 'height': h,
                'intrinsics': intrinsics.tolist(),
                'fov_x': round(float(np.rad2deg(fov_x_rad)), 2),
                'fov_y': round(float(np.rad2deg(fov_y_rad)), 2),
            }
            results.append({'arrays': payload_arrays, 'done': done})
        return results


async def _serve(service: MoGeService, host: str, port: int):
    service.queue = asyncio.Queue()
    server = await asyncio.start_server(service.handle_connection, host, port)
    click.echo(f'Listening on tcp://{host}:{port} ... (Ctrl+C to stop)')
    worker = asyncio.create_task(service.inference_worker())
    async with server:
        try:
            await server.serve_forever()
        finally:
            worker.cancel()


@click.command(help='Start an async TCP inference server with dynamic micro-batching. Clients send an image and receive status updates, then depth/points/normal/mask arrays and camera intrinsics.')
@click.option('--pretrained', type=str, default=DEFAULT_PRETRAINED, show_default=True, help='Local checkpoint path or HuggingFace model ID.')
@click.option('--host', type=str, default='0.0.0.0', show_default=True, help='Address to bind.')
@click.option('--port', type=int, default=8317, show_default=True, help='Port to bind.')
@click.option('--refine-steps', type=int, default=3, show_default=True, help='Default number of sparse 3D refinement steps (overridable per request).')
@click.option('--resolution-level', type=int, default=9, show_default=True, help='Default inference resolution level 0-9 (overridable per request).')
@click.option('--fp16/--no-fp16', default=True, show_default=True, help='Use fp16 mixed precision for faster inference.')
@click.option('--max-image-px', type=int, default=8192 * 8192, show_default=True, help='Maximum input image size in total pixels.')
@click.option('--max-batch-size', type=int, default=2, show_default=True, help='Maximum requests merged into one GPU forward pass.')
@click.option('--batch-window-ms', type=int, default=25, show_default=True, help='Collection window (ms) to gather concurrent requests into one batch (skipped for single connections).')
def main(pretrained: str, host: str, port: int, refine_steps: int, resolution_level: int, fp16: bool,
         max_image_px: int, max_batch_size: int, batch_window_ms: int):
    if Path(pretrained).exists():
        click.echo(f'Loading model from {pretrained} ...')
    else:
        click.echo(f'{pretrained} not found locally; trying as HuggingFace model ID ...')
    model = MoGeModel.from_pretrained(pretrained).cuda().eval()

    if torch.cuda.is_available():
        click.echo(f'Device: {torch.cuda.get_device_name(0)} | VRAM reserved after load: {torch.cuda.memory_reserved() / 1024**3:.2f} GB')
    else:
        click.echo('WARNING: CUDA not available, running on CPU (very slow).')
    click.echo(f'Defaults: resolution_level={resolution_level}, refine_steps={refine_steps}, fp16={fp16}, '
               f'max_batch_size={max_batch_size}, batch_window_ms={batch_window_ms}')

    service = MoGeService(model, refine_steps, resolution_level, fp16, max_image_px, max_batch_size, batch_window_ms)

    try:
        asyncio.run(_serve(service, host, port))
    except KeyboardInterrupt:
        click.echo('\nShutting down.')


if __name__ == '__main__':
    main()
