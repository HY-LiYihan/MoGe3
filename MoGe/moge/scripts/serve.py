import os
from pathlib import Path
import sys
if (_package_root := str(Path(__file__).absolute().parents[2])) not in sys.path:
    sys.path.insert(0, _package_root)

import json
import math
import socket
import struct
import threading
import time
import zlib

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


def recv_exact(sock: socket.socket, n: int, allow_eof: bool = False):
    """精确读取 n 字节。allow_eof=True 时在帧边界处遇到 EOF 返回 None。"""
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            if allow_eof and len(buf) == 0:
                return None
            raise ConnectionError(f'connection closed by peer ({len(buf)}/{n} bytes received)')
        buf.extend(chunk)
    return bytes(buf)


def recv_frame(sock: socket.socket, allow_eof: bool = False):
    """接收一帧：4 字节小端长度 + payload。"""
    header = recv_exact(sock, LENGTH_PREFIX.size, allow_eof=allow_eof)
    if header is None:
        return None
    (length,) = LENGTH_PREFIX.unpack(header)
    if length > MAX_FRAME_BYTES:
        raise ProtocolError(f'frame too large: {length} bytes (max {MAX_FRAME_BYTES})')
    return recv_exact(sock, length)


def send_frame(sock: socket.socket, payload: bytes):
    sock.sendall(LENGTH_PREFIX.pack(len(payload)) + payload)


def send_json(sock: socket.socket, obj: dict):
    send_frame(sock, json.dumps(obj).encode('utf-8'))


class MoGeService:
    """单模型推理服务：每连接一个线程，GPU 推理用全局锁串行化。"""

    def __init__(self, model: MoGeModel, refine_steps: int, resolution_level: int, use_fp16: bool, max_image_px: int):
        self.model = model
        self.default_resolution_level = resolution_level
        self.default_refine_steps = refine_steps
        self.use_fp16 = use_fp16
        self.max_image_px = max_image_px
        self.infer_lock = threading.Lock()

    def serve_connection(self, conn: socket.socket, addr):
        """长连接：循环处理请求，客户端断开或出错时退出。"""
        try:
            while True:
                header_frame = recv_frame(conn, allow_eof=True)
                if header_frame is None:
                    break
                try:
                    header = json.loads(header_frame)
                    self.process_request(conn, header)
                except (ProtocolError, ConnectionError, OSError) as e:
                    self._try_send_json(conn, {'type': 'error', 'stage': 'request', 'message': str(e)})
                    break
                except Exception as e:
                    self._try_send_json(conn, {'type': 'error', 'stage': 'inference', 'message': f'{type(e).__name__}: {e}'})
                    break
        except OSError:
            pass
        finally:
            try:
                conn.close()
            except OSError:
                pass

    @staticmethod
    def _try_send_json(conn: socket.socket, obj: dict):
        try:
            send_json(conn, obj)
        except OSError:
            pass

    def process_request(self, conn: socket.socket, header: dict):
        t_start = time.perf_counter()

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
        image_payload = recv_frame(conn)
        if len(image_payload) != image_bytes:
            raise ProtocolError(f'image payload length mismatch: header says {image_bytes}, got {len(image_payload)}')

        # 解码
        encoded = np.frombuffer(image_payload, dtype=np.uint8)
        image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if image is None:
            raise ProtocolError('failed to decode image (supported formats: jpg, png)')
        height, width = image.shape[:2]
        if height * width > self.max_image_px:
            raise ProtocolError(f'image resolution too high: {width}x{height} (max {self.max_image_px} pixels)')

        send_json(conn, {'type': 'status', 'stage': 'received', 'width': width, 'height': height})

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

        send_json(conn, {
            'type': 'status', 'stage': 'inferring',
            'params': {'resolution_level': resolution_level, 'refine_steps': refine_steps, 'fov_x': fov_x},
        })

        # 推理（(3, H, W) float32，[0,1]，RGB）
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_tensor = torch.from_numpy(image_rgb).permute(2, 0, 1).float().div_(255).cuda()

        with self.infer_lock:
            t_infer = time.perf_counter()
            with torch.inference_mode():
                output = self.model.infer(
                    image_tensor,
                    resolution_level=resolution_level,
                    refine_steps=refine_steps,
                    fov_x=fov_x,
                    use_fp16=self.use_fp16,
                )
            infer_ms = (time.perf_counter() - t_infer) * 1000

        # 数组输出
        arrays = {
            'depth': output['depth'].detach().cpu().numpy().astype(np.float32, copy=False),
            'points': output['points'].detach().cpu().numpy().astype(np.float32, copy=False),
            'normal': output['normal'].detach().cpu().numpy().astype(np.float32, copy=False) if 'normal' in output else None,
            'mask': output['mask'].detach().cpu().numpy().astype(np.uint8),
        }
        for name in ARRAY_ORDER:
            arr = arrays[name]
            if arr is None:
                continue
            # 浮点数组 zlib 压缩率仅 ~10% 且 CPU 开销大，直接传原始字节；mask 压缩收益高
            if name == 'mask':
                compression, payload = 'zlib', zlib.compress(arr.tobytes(), level=1)
            else:
                compression, payload = 'none', arr.tobytes()
            send_json(conn, {
                'type': 'array', 'name': name,
                'dtype': str(arr.dtype), 'shape': list(arr.shape),
                'compression': compression, 'bytes': len(payload),
            })
            send_frame(conn, payload)

        # 相机参数
        intrinsics = output['intrinsics'].detach().cpu().numpy()
        fov_x_rad, fov_y_rad = utils3d.np.intrinsics_to_fov(intrinsics)

        elapsed_ms = (time.perf_counter() - t_start) * 1000
        send_json(conn, {
            'type': 'done',
            'width': width, 'height': height,
            'intrinsics': intrinsics.tolist(),
            'fov_x': round(float(np.rad2deg(fov_x_rad)), 2),
            'fov_y': round(float(np.rad2deg(fov_y_rad)), 2),
            'elapsed_ms': round(elapsed_ms, 1),
            'infer_ms': round(infer_ms, 1),
        })


@click.command(help='Start a TCP inference server. Clients send an image and receive status updates, then depth/points/normal/mask arrays and camera intrinsics.')
@click.option('--pretrained', type=str, default=DEFAULT_PRETRAINED, show_default=True, help='Local checkpoint path or HuggingFace model ID.')
@click.option('--host', type=str, default='0.0.0.0', show_default=True, help='Address to bind.')
@click.option('--port', type=int, default=8317, show_default=True, help='Port to bind.')
@click.option('--refine-steps', type=int, default=3, show_default=True, help='Default number of sparse 3D refinement steps (overridable per request).')
@click.option('--resolution-level', type=int, default=9, show_default=True, help='Default inference resolution level 0-9 (overridable per request).')
@click.option('--fp16/--no-fp16', default=True, show_default=True, help='Use fp16 mixed precision for faster inference.')
@click.option('--max-image-px', type=int, default=8192 * 8192, show_default=True, help='Maximum input image size in total pixels.')
def main(pretrained: str, host: str, port: int, refine_steps: int, resolution_level: int, fp16: bool, max_image_px: int):
    if Path(pretrained).exists():
        click.echo(f'Loading model from {pretrained} ...')
    else:
        click.echo(f'{pretrained} not found locally; trying as HuggingFace model ID ...')
    model = MoGeModel.from_pretrained(pretrained).cuda().eval()

    if torch.cuda.is_available():
        click.echo(f'Device: {torch.cuda.get_device_name(0)} | VRAM reserved after load: {torch.cuda.memory_reserved() / 1024**3:.2f} GB')
    else:
        click.echo('WARNING: CUDA not available, running on CPU (very slow).')
    click.echo(f'Defaults: resolution_level={resolution_level}, refine_steps={refine_steps}, fp16={fp16}')

    service = MoGeService(model, refine_steps, resolution_level, fp16, max_image_px)

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((host, port))
    srv.listen(8)
    srv.settimeout(0.5)  # Windows 下让 Ctrl+C 能打断 accept() 阻塞
    click.echo(f'Listening on tcp://{host}:{port} ... (Ctrl+C to stop)')

    try:
        while True:
            try:
                conn, addr = srv.accept()
            except socket.timeout:
                continue
            threading.Thread(target=service.serve_connection, args=(conn, addr), daemon=True).start()
    except KeyboardInterrupt:
        click.echo('\nShutting down.')
    finally:
        srv.close()


if __name__ == '__main__':
    main()
