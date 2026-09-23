r"""统一多模型 TCP 推理服务 (VGGT-Omega + SAM3, 同端口)

用法:
    python serve.py [--vggt-checkpoint <pt>] [--sam3-checkpoint <pt>]
                    [--host 0.0.0.0] [--port 8317] [--conf 0.25]

单端口同时提供两类推理, 请求 JSON 头新增可选字段 model:
    "model": "vggt" (默认, 向后兼容旧客户端) | "sam3"

  vggt (mode: images / video, 详见 vggt-omega/serve.py):
    {"type":"infer","model":"vggt","mode":"images","images_bytes":[n1,...],
     "params":{"resolution":512,...}}
    {"type":"infer","model":"vggt","mode":"video","video_bytes":M,"params":{...}}
  sam3 (mode: text / reference, 详见 sam3/serve.py):
    {"type":"infer","model":"sam3","mode":"text","image_bytes":N,"text":["robotic arm"]}
    {"type":"infer","model":"sam3","mode":"reference","ref_image_bytes":N1,
     "ref_mask_bytes":N2,"images_bytes":[...]}

GPU 前向全局共享一把锁: VGGT 与 SAM3 不会同时前向, 避免显存叠加。
VGGT 启动即加载 (~4.3GB); SAM3 两模型懒加载 (首个对应请求时构建, 平时零占用)。
"""

import argparse
import asyncio
import importlib.util
import logging
import struct
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("unified")

ROOT = Path(__file__).resolve().parent
MAX_PAYLOAD = 512 * 1024 * 1024


def _load_module(name: str, path: Path):
    """按路径加载子服务模块 (vggt-omega 与 sam3 的脚本同名 serve.py, 需别名加载)。"""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


vggt_mod = _load_module("vggt_serve_impl", ROOT / "vggt-omega" / "serve.py")
sam3_mod = _load_module("sam3_serve_impl", ROOT / "sam3" / "serve.py")
# 两模块的 send_json 字节格式一致 (4 字节小端长度 + JSON)
send_json = vggt_mod.send_json

DEFAULT_SAM3_CHECKPOINT = str(ROOT / "checkpoints" / "sam3" / "sam3.pt")


def _payload_len(req: dict) -> int:
    """按 model/mode 计算二进制负载长度 (不合法即抛 ValueError)。"""
    model = req.get("model", "vggt")
    if model == "vggt":
        mode = req.get("mode", "images")
        if mode == "images":
            return sum(int(n) for n in req["images_bytes"])
        if mode == "video":
            return int(req["video_bytes"])
        raise ValueError(f"vggt mode 须为 images|video: {mode}")
    if model == "sam3":
        mode = req.get("mode")
        if mode == "text":
            return int(req["image_bytes"])
        if mode == "reference":
            return (int(req["ref_image_bytes"]) + int(req["ref_mask_bytes"])
                    + sum(int(x) for x in req.get("images_bytes", [])))
        raise ValueError(f"sam3 mode 须为 text|reference: {mode}")
    raise ValueError(f"未知 model: {model} (可选 vggt / sam3)")


class UnifiedService:
    def __init__(self, vggt_ckpt: str, sam3_ckpt: str, conf: float):
        self.vggt = vggt_mod.VGGTService(vggt_ckpt)              # 启动即加载
        self.sam3 = sam3_mod.Sam3Service(sam3_ckpt, conf=conf)   # 懒加载
        # 共享 GPU 锁: 两模型前向互斥 (合计显存 ~8GB, 并行前向会溢出)
        self.gpu_lock = asyncio.Lock()
        self.vggt.infer_lock = self.gpu_lock
        self.sam3.infer_lock = self.gpu_lock

    async def dispatch(self, reader, writer, req: dict, payload: bytes):
        model = req.get("model", "vggt")
        if model == "vggt":
            await send_json(writer, {"type": "status", "stage": "received",
                                     "payload_bytes": len(payload)})
            await self.vggt.handle_request(reader, writer, req, payload)
        else:  # sam3 (handle_request 自发 received/inferring)
            await self.sam3.handle_request(reader, writer, req, payload)

    async def handle_conn(self, reader, writer):
        peer = writer.get_extra_info("peername")
        log.info("连接: %s", peer)
        recv_task = None  # 复用读取任务, 避免 wait_for 取消时丢失已读字节
        try:
            while True:
                if recv_task is None:
                    recv_task = asyncio.create_task(vggt_mod.recv_msg(reader))
                done, _ = await asyncio.wait({recv_task}, timeout=0.5)
                if not done:
                    continue  # 空闲轮询 (保持 Ctrl+C 响应), 读取任务保持挂起
                try:
                    req = recv_task.result()
                except asyncio.IncompleteReadError:
                    break  # 客户端已断开
                except ValueError as e:  # 含 json.JSONDecodeError
                    recv_task = None
                    await send_json(writer, {"type": "error", "stage": "request",
                                             "message": f"非法 JSON: {e}"})
                    continue
                recv_task = None
                if req.get("type") == "quit":
                    await send_json(writer, {"type": "bye"})
                    break
                if req.get("type") != "infer":
                    await send_json(writer, {"type": "error", "stage": "request",
                                             "message": f"未知 type: {req.get('type')}"})
                    continue
                try:
                    payload_len = _payload_len(req)
                    if payload_len > MAX_PAYLOAD:
                        raise ValueError(f"负载超限 {payload_len} > {MAX_PAYLOAD}")
                    (length,) = struct.unpack("<I", await reader.readexactly(4))
                    if length != payload_len:
                        raise ValueError("二进制负载帧长度与 JSON 声明不符")
                    payload = await reader.readexactly(payload_len) if payload_len else b""
                    await self.dispatch(reader, writer, req, payload)
                except (KeyError, ValueError, TypeError, RuntimeError) as e:
                    log.warning("请求失败(%s): %s", peer, e)
                    await send_json(writer, {"type": "error", "stage": "request",
                                             "message": str(e)})
                except Exception as e:  # 推理异常: 回错误但保持连接
                    log.warning("推理失败(%s): %s: %s", peer, type(e).__name__, e)
                    await send_json(writer, {"type": "error", "stage": "infer",
                                             "message": f"{type(e).__name__}: {e}"})
        except (ConnectionResetError, BrokenPipeError, asyncio.IncompleteReadError):
            pass
        finally:
            if recv_task and not recv_task.done():
                recv_task.cancel()
            writer.close()
            log.info("断开: %s", peer)

    async def serve(self, host: str, port: int):
        server = await asyncio.start_server(self.handle_conn, host, port)
        log.info("统一服务监听 %s:%d (vggt=images/video, sam3=text/reference; "
                 "请求头 model 字段区分, 默认 vggt)", host, port)
        log.info("sam3 checkpoint: %s (懒加载)", self.sam3.checkpoint)
        async with server:
            await server.serve_forever()


def main():
    ap = argparse.ArgumentParser(description="统一多模型 TCP 推理服务 (VGGT-Omega + SAM3 同端口)")
    ap.add_argument("--vggt-checkpoint", default=vggt_mod.DEFAULT_CHECKPOINT)
    ap.add_argument("--sam3-checkpoint", default=DEFAULT_SAM3_CHECKPOINT)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8317)
    ap.add_argument("--conf", type=float, default=0.25, help="SAM3 text 模式置信度阈值")
    args = ap.parse_args()

    svc = UnifiedService(args.vggt_checkpoint, args.sam3_checkpoint, args.conf)
    try:
        asyncio.run(svc.serve(args.host, args.port))
    except KeyboardInterrupt:
        log.info("服务已停止")


if __name__ == "__main__":
    main()
