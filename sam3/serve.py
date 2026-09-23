"""SAM3 分割 TCP 服务 (默认端口 8318)

基于 Ultralytics (ultralytics==8.4.160+, 权重 sam3.pt 需手动下载)。

两种模式:
  1. text      文本提示切 mask: 给一张图 + 名词短语, 分割出所有匹配实例
  2. reference 参考图 mask 切新图: 给参考图 + 其二值 mask + 目标图序列,
               用 SAM3 视频跟踪器把 mask 传播到目标图 (帧0=mask提示, 后续帧自动跟踪)

协议与 vggt-omega/serve.py 一致: 4 字节小端长度 + payload (JSON 或原始字节)。
模型懒加载: 首个请求时才构建, 平时显存零占用 (两种模式各自独立实例)。
"""

import argparse
import asyncio
import json
import logging
import shutil
import struct
import tempfile
import time

import cv2
import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("sam3")

MAX_PAYLOAD = 256 * 1024 * 1024
MAX_FRAMES = 32  # reference 模式目标图上限


def decode_image(buf):
    """字节流 → BGR ndarray (jpg/png)。失败抛 ValueError。"""
    img = cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("图片解码失败 (仅支持 jpg/png)")
    return img


# ---------------- 协议 ----------------

async def send_msg(writer, payload: bytes):
    writer.write(struct.pack("<I", len(payload)) + payload)
    await writer.drain()


async def send_json(writer, obj):
    await send_msg(writer, json.dumps(obj, ensure_ascii=False).encode("utf-8"))


async def recv_msg(reader):
    """读一帧消息 (None = EOF)。"""
    header = await reader.readexactly(4)
    (length,) = struct.unpack("<I", header)
    if length > MAX_PAYLOAD:
        raise ValueError(f"消息超限 {length}")
    return await reader.readexactly(length) if length else b""


async def send_array(writer, name, arr):
    """发送数组: 元数据 JSON + 原始字节 (小端, C 连续)。"""
    arr = np.ascontiguousarray(arr)
    await send_json(writer, {
        "type": "array", "name": name, "dtype": arr.dtype.name,
        "shape": list(arr.shape), "bytes": arr.nbytes,
    })
    await send_msg(writer, arr.tobytes())


# ---------------- 服务 ----------------

class Sam3Service:
    def __init__(self, checkpoint, conf=0.25):
        self.checkpoint = checkpoint
        self.conf = conf
        self.semantic = None  # SAM3SemanticPredictor (text 模式), 懒加载
        self.video = None     # SAM3VideoPredictor  (reference 模式), 懒加载
        self.infer_lock = asyncio.Lock()  # GPU 推理全局串行

    def _overrides(self):
        return {
            "task": "segment", "mode": "predict", "model": self.checkpoint,
            "quantize": 16,  # fp16
            "conf": self.conf,
            "save": False, "verbose": False, "retina_masks": True,
        }

    def _get_semantic(self):
        if self.semantic is None:
            from ultralytics.models.sam import SAM3SemanticPredictor
            t0 = time.perf_counter()
            self.semantic = SAM3SemanticPredictor(overrides=self._overrides())
            self.semantic.setup_model()
            log.info("SAM3SemanticPredictor 就绪 %.1fs (显存 %.2f GB)",
                     time.perf_counter() - t0, torch.cuda.memory_allocated() / 2**30)
        return self.semantic

    def _get_video(self):
        if self.video is None:
            from ultralytics.models.sam import SAM3VideoPredictor
            t0 = time.perf_counter()
            self.video = SAM3VideoPredictor(overrides=self._overrides())
            self.video.setup_model()
            log.info("SAM3VideoPredictor 就绪 %.1fs (显存 %.2f GB)",
                     time.perf_counter() - t0, torch.cuda.memory_allocated() / 2**30)
        return self.video

    # ---- 模式 1: text ----

    def _run_text(self, img, texts, conf):
        p = self._get_semantic()
        p.args.conf = conf
        p.set_image(img)  # BGR ndarray, 内存图
        results = p(text=texts)  # 注: 勿调 reset_prompts(), 会破坏 set_classes 状态缓存
        r = results[0]
        if r.masks is None:  # 无匹配
            h, w = img.shape[:2]
            return {"masks": np.zeros((0, h, w), np.uint8),
                    "boxes": np.zeros((0, 4), np.float32),
                    "scores": np.zeros((0,), np.float32)}
        return {
            "masks": r.masks.data.cpu().numpy().astype(np.uint8),  # (K,H,W)
            "boxes": r.boxes.xyxy.cpu().numpy().astype(np.float32),  # (K,4)
            "scores": r.boxes.conf.cpu().numpy().astype(np.float32),  # (K,)
        }

    # ---- 模式 2: reference (视频跟踪传播) ----

    def _run_reference(self, ref_img, ref_mask, target_imgs):
        """参考图 mask → 外接框提示帧0, 视频跟踪器传播到目标图。"""
        vp = self._get_video()
        tmp = tempfile.mkdtemp(prefix="sam3svc_")
        try:
            h, w = ref_img.shape[:2]
            video_path = f"{tmp}/seq.mp4"
            vw = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"mp4v"), 10, (w, h))
            vw.write(ref_img)
            for t in target_imgs:
                vw.write(t)
            vw.release()
            ys, xs = np.nonzero(ref_mask)
            bbox = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]
            vp.inference_state = {}  # 重置跟踪状态
            results = list(vp(source=video_path, bboxes=[bbox], stream=True))
            masks, scores = [], []
            for r in results[1:]:  # 帧0 是参考图本身, 只回传目标图
                if r.masks is None:
                    masks.append(np.zeros((h, w), np.uint8))
                    scores.append(0.0)
                else:
                    m = r.masks.data.cpu().numpy()
                    masks.append(m[0].astype(np.uint8) if m.ndim == 3 else m.astype(np.uint8))
                    scores.append(float(r.boxes.conf.mean()) if r.boxes is not None and len(r.boxes) else 0.0)
            return {"masks": np.stack(masks) if masks else np.zeros((0, h, w), np.uint8),
                    "scores": np.array(scores, np.float32)}
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    # ---- 请求入口 ----

    async def handle_request(self, reader, writer, req, payload):
        peer = writer.get_extra_info("peername")
        mode = req.get("mode")
        await send_json(writer, {"type": "status", "stage": "received", "bytes": len(payload)})
        await send_json(writer, {"type": "status", "stage": "inferring", "mode": mode})

        t0 = time.perf_counter()
        async with self.infer_lock:
            if mode == "text":
                img = decode_image(payload)
                texts = req.get("text") or []
                if isinstance(texts, str):
                    texts = [texts]
                if not texts:
                    raise ValueError("text 模式需要 text 字段 (名词短语列表)")
                conf = float(req.get("conf", self.conf))
                out = await asyncio.to_thread(self._run_text, img, texts, conf)
                k = out["masks"].shape[0]
                log.info("推理完成(%s): text %d个概念 %d个实例 | %.2fs", peer, len(texts), k,
                         time.perf_counter() - t0)
                await send_array(writer, "masks", out["masks"])
                await send_array(writer, "boxes", out["boxes"])
                await send_array(writer, "scores", out["scores"])
                await send_json(writer, {
                    "type": "done", "mode": "text", "k": int(k),
                    "prompts": texts, "height": int(img.shape[0]), "width": int(img.shape[1]),
                    "conf_thresh": conf,
                    "infer_ms": round((time.perf_counter() - t0) * 1000, 1),
                })
            elif mode == "reference":
                n_ref = int(req["ref_image_bytes"])
                n_mask = int(req["ref_mask_bytes"])
                sizes = [int(x) for x in req.get("images_bytes", [])]
                if not 0 < n_ref <= MAX_PAYLOAD or not 0 < n_mask <= MAX_PAYLOAD:
                    raise ValueError("ref_image_bytes / ref_mask_bytes 非法")
                if not sizes or len(sizes) > MAX_FRAMES:
                    raise ValueError(f"images_bytes 需为 1~{MAX_FRAMES} 张目标图")
                off = n_ref + n_mask
                if off + sum(sizes) != len(payload):
                    raise ValueError("二进制负载长度与 JSON 声明不符")
                ref_img = await asyncio.to_thread(decode_image, payload[:n_ref])
                ref_mask = await asyncio.to_thread(
                    cv2.imdecode, np.frombuffer(payload[n_ref:n_ref + n_mask], np.uint8),
                    cv2.IMREAD_GRAYSCALE)
                if ref_mask is None:
                    raise ValueError("参考 mask 解码失败 (需 PNG)")
                if ref_mask.shape != ref_img.shape[:2]:
                    raise ValueError(f"参考 mask 尺寸 {ref_mask.shape} 与参考图 {ref_img.shape[:2]} 不符")
                ref_mask = (ref_mask > 127).astype(np.uint8)
                if ref_mask.sum() == 0:
                    raise ValueError("参考 mask 全空")
                pos = off
                target_imgs = []
                for s in sizes:
                    target_imgs.append(await asyncio.to_thread(decode_image, payload[pos:pos + s]))
                    pos += s
                for i, t in enumerate(target_imgs):  # 视频跟踪要求全序列同分辨率
                    if t.shape != ref_img.shape:
                        raise ValueError(f"目标图 {i} 分辨率 {t.shape[:2]} 与参考图 {ref_img.shape[:2]} 不一致")
                out = await asyncio.to_thread(self._run_reference, ref_img, ref_mask, target_imgs)
                n = out["masks"].shape[0]
                areas = [int(m.sum()) for m in out["masks"]]
                log.info("推理完成(%s): reference %d目标图 面积%s | %.2fs", peer, n, areas,
                         time.perf_counter() - t0)
                await send_array(writer, "masks", out["masks"])  # (N,H,W)
                await send_json(writer, {
                    "type": "done", "mode": "reference", "frames": int(n),
                    "height": int(ref_img.shape[0]), "width": int(ref_img.shape[1]),
                    "scores": [round(float(s), 4) for s in out["scores"]],
                    "mask_areas": areas,
                    "infer_ms": round((time.perf_counter() - t0) * 1000, 1),
                })
            else:
                raise ValueError(f"未知 mode: {mode} (可选 text / reference)")


async def handle_conn(service, reader, writer):
    peer = writer.get_extra_info("peername")
    log.info("连接: %s", peer)
    try:
        while True:
            req_raw = await recv_msg(reader)
            if req_raw is None:
                break
            try:
                req = json.loads(req_raw)
                if req.get("type") == "quit":
                    await send_json(writer, {"type": "bye"})
                    break
                if req.get("type") != "infer":
                    await send_json(writer, {"type": "error", "stage": "request",
                                              "message": f"未知 type: {req.get('type')}"})
                    continue
                # 读二进制负载
                if req.get("mode") == "text":
                    payload_len = int(req["image_bytes"])
                    if payload_len > MAX_PAYLOAD:
                        raise ValueError(f"负载超限 {payload_len}")
                else:
                    payload_len = None  # reference 模式在 handler 内校验
                if payload_len is not None:
                    (length,) = struct.unpack("<I", await reader.readexactly(4))
                    if length != payload_len:
                        raise ValueError("二进制负载帧长度与 JSON 声明不符")
                    payload = await reader.readexactly(payload_len)
                else:
                    payload = b""
                    sizes = [int(x) for x in req.get("images_bytes", [])]
                    need = int(req["ref_image_bytes"]) + int(req["ref_mask_bytes"]) + sum(sizes)
                    if need > MAX_PAYLOAD:
                        raise ValueError(f"负载超限 {need}")
                    (length,) = struct.unpack("<I", await reader.readexactly(4))
                    if length != need:
                        raise ValueError("二进制负载帧长度与 JSON 声明不符")
                    payload = await reader.readexactly(need)
                await service.handle_request(reader, writer, req, payload)
            except (KeyError, ValueError, TypeError) as e:
                await send_json(writer, {"type": "error", "stage": "request", "message": str(e)})
            except Exception as e:  # 推理/解码异常: 回错误但保持连接
                log.warning("请求失败(%s): %s: %s", peer, type(e).__name__, e)
                await send_json(writer, {"type": "error", "stage": "infer",
                                          "message": f"{type(e).__name__}: {e}"})
    except (asyncio.IncompleteReadError, ConnectionResetError):
        pass  # 客户端断开
    finally:
        writer.close()
        await writer.wait_closed()
        log.info("断开: %s", peer)


async def main():
    ap = argparse.ArgumentParser(description="SAM3 TCP 分割服务")
    ap.add_argument("--checkpoint", default=r"D:\MoGe3\checkpoints\sam3\sam3.pt")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8318)
    ap.add_argument("--conf", type=float, default=0.25, help="text 模式置信度阈值")
    args = ap.parse_args()

    service = Sam3Service(args.checkpoint, conf=args.conf)
    server = await asyncio.start_server(lambda r, w: handle_conn(service, r, w), args.host, args.port)
    log.info("SAM3 服务监听 %s:%d (text/reference 两种模式, 模型懒加载)",
             args.host, args.port)
    log.info("checkpoint: %s", args.checkpoint)
    async with server:
        await server.serve_forever()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
