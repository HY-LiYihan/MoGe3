"""SAM3 TCP 服务测试: text / reference 两种模式 + 并发。

用法: python test_service.py [--host 127.0.0.1] [--port 8318]
"""
import argparse
import json
import socket
import struct
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

VIDEO = r"D:\MoGe3\vggt-omega\examples\desert_road.mp4"


# ---- 协议 ----

def send_msg(sock, payload):
    sock.sendall(struct.pack("<I", len(payload)) + payload)


def recv_msg(sock):
    header = b""
    while len(header) < 4:
        chunk = sock.recv(4 - len(header))
        if not chunk:
            return None
        header += chunk
    (length,) = struct.unpack("<I", header)
    data = b""
    while len(data) < length:
        chunk = sock.recv(length - len(data))
        if not chunk:
            raise ConnectionError("连接中断")
        data += chunk
    return data


def recv_json(sock):
    m = recv_msg(sock)
    return json.loads(m) if m else None


def call_text(host, port, jpg_bytes, texts, conf=0.25):
    req = {"type": "infer", "mode": "text", "image_bytes": len(jpg_bytes),
           "text": texts, "conf": conf}
    t0 = time.perf_counter()
    with socket.create_connection((host, port), timeout=300) as s:
        send_msg(s, json.dumps(req).encode())
        send_msg(s, jpg_bytes)
        arrs, done = {}, None
        while True:
            msg = recv_json(s)
            if msg is None:
                break
            if msg["type"] == "array":
                raw = recv_msg(s)
                assert len(raw) == msg["bytes"], "数组长度不符"
                arrs[msg["name"]] = np.frombuffer(raw, dtype=msg["dtype"]).reshape(msg["shape"])
            elif msg["type"] == "done":
                done = msg
                break
            elif msg["type"] == "error":
                raise RuntimeError(f"服务错误: {msg}")
        return {"arrays": arrs, "done": done, "client_ms": (time.perf_counter() - t0) * 1000}


def call_reference(host, port, ref_jpg, mask_png, target_jpgs):
    payload = ref_jpg + mask_png + b"".join(target_jpgs)
    req = {"type": "infer", "mode": "reference",
           "ref_image_bytes": len(ref_jpg), "ref_mask_bytes": len(mask_png),
           "images_bytes": [len(t) for t in target_jpgs]}
    t0 = time.perf_counter()
    with socket.create_connection((host, port), timeout=300) as s:
        send_msg(s, json.dumps(req).encode())
        send_msg(s, payload)
        arrs, done = {}, None
        while True:
            msg = recv_json(s)
            if msg is None:
                break
            if msg["type"] == "array":
                raw = recv_msg(s)
                arrs[msg["name"]] = np.frombuffer(raw, dtype=msg["dtype"]).reshape(msg["shape"])
            elif msg["type"] == "done":
                done = msg
                break
            elif msg["type"] == "error":
                raise RuntimeError(f"服务错误: {msg}")
        return {"arrays": arrs, "done": done, "client_ms": (time.perf_counter() - t0) * 1000}


def grab_frames(video, n=3):
    cap = cv2.VideoCapture(video)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idxs = [int(total * i / n) for i in range(n)]
    frames, i = [], 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if i in idxs:
            frames.append(f)
            if len(frames) == n:
                break
        i += 1
    cap.release()
    return frames


def to_jpg(img, quality=92):
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes()


def mask_to_png(mask):
    ok, buf = cv2.imencode(".png", (mask * 255).astype(np.uint8))
    return buf.tobytes()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8318)
    args = ap.parse_args()

    frames = grab_frames(VIDEO, 3)
    jpgs = [to_jpg(f) for f in frames]
    print(f"抽帧 {len(frames)} 张 {frames[0].shape}\n")

    # ---- 模式 1: text ----
    print("===== text 模式 =====")
    r = call_text(args.host, args.port, jpgs[0], ["sky", "road"])
    d = r["done"]
    print(f"实例数 {d['k']} | conf {[round(x,3) for x in r['arrays']['scores'].tolist()]} "
          f"| 耗时 {r['client_ms']:.0f}ms")
    print(f"masks {r['arrays']['masks'].shape}, boxes {r['arrays']['boxes'].shape}")
    assert d["k"] == 2 and d["height"] == 720 and d["width"] == 1280, "text 模式结果异常"
    sky_mask = r["arrays"]["masks"][0]  # 概念 0 = sky

    # ---- 模式 2: reference ----
    print("\n===== reference 模式 =====")
    r2 = call_reference(args.host, args.port, jpgs[0], mask_to_png(sky_mask), jpgs[1:])
    d2 = r2["done"]
    print(f"目标图 {d2['frames']} 张 | 面积 {[f'{a/ (d2['height']*d2['width'])*100:.1f}%' for a in d2['mask_areas']]} "
          f"| scores {d2['scores']} | 耗时 {r2['client_ms']:.0f}ms")
    m = r2["arrays"]["masks"]
    assert m.shape == (2, 720, 1280), f"mask 形状异常 {m.shape}"
    iou0 = (m[0] & sky_mask).sum() / max((m[0] | sky_mask).sum(), 1)
    print(f"帧1 与参考 mask IoU: {iou0:.3f} (同物体应较高)")
    assert iou0 > 0.5, "传播 mask 与参考差异过大"

    # ---- 错误处理 ----
    print("\n===== 错误处理 =====")
    with socket.create_connection((args.host, args.port), timeout=30) as s:
        send_msg(s, json.dumps({"type": "infer", "mode": "text", "image_bytes": 4, "text": ["x"]}).encode())
        send_msg(s, b"BAD!")
        while True:
            msg = recv_json(s)
            if msg["type"] == "error":
                break
        print(f"坏图 → {msg['type']}({msg['stage']}): {msg['message'][:40]}")
        assert msg["type"] == "error"

    # ---- 并发 ----
    print("\n===== 并发 2 连接 =====")
    with ThreadPoolExecutor(2) as ex:
        futs = [ex.submit(call_text, args.host, args.port, jpgs[i % 3], ["road"]) for i in range(2)]
        rs = [f.result() for f in futs]
    print(f"2 连接全部成功: {[round(x['client_ms']) for x in rs]}ms | "
          f"k={[x['done']['k'] for x in rs]}")

    print("\n全部测试通过 ✓")


if __name__ == "__main__":
    main()
