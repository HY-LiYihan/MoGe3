r"""统一服务 (VGGT + SAM3 同端口 8317) 集成测试

阶段:
  1. sam3/text   : 5 张机械臂图 × "robotic arm", 断言每张检出 >=1 实例, 叠加可视化
  2. vggt 回归   : 旧协议 (不带 model 字段, 向后兼容) + 新协议 (model=vggt) 各跑一次
  3. sam3 组合   : text 切第 1 帧 mask → reference 传播到后续帧 (验证文档推荐的组合用法)

用法: python test_unified_service.py [--host 127.0.0.1] [--port 8317]
"""
import argparse
import json
import socket
import struct
import sys
import time
from pathlib import Path

import cv2
import numpy as np

IMG_DIR = Path(r"D:\MoGe3\outputs\test_robot_arm")
VIS_DIR = IMG_DIR / "vis"


def recv_exact(s: socket.socket, n: int) -> bytes:
    buf = b""
    while len(buf) < n:
        chunk = s.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("连接被关闭")
        buf += chunk
    return buf


def rpc(s: socket.socket, header: dict, payload: bytes = b""):
    """发一次请求, 收齐 arrays + done; error 则抛 RuntimeError。"""
    for data in (json.dumps(header).encode(), payload):
        s.sendall(struct.pack("<I", len(data)) + data)
    arrays, statuses = {}, []
    while True:
        (ln,) = struct.unpack("<I", recv_exact(s, 4))
        msg = json.loads(recv_exact(s, ln))
        t = msg["type"]
        if t == "status":
            statuses.append(msg.get("stage"))
        elif t == "array":
            if "bytes" in msg:  # sam3 格式: 数组是独立的长度前缀帧
                (n,) = struct.unpack("<I", recv_exact(s, 4))
                buf = recv_exact(s, n)
            else:  # vggt 格式: 裸字节紧跟 JSON 帧
                n = int(np.prod(msg["shape"])) * np.dtype(msg["dtype"]).itemsize
                buf = recv_exact(s, n)
            arrays[msg["name"]] = np.frombuffer(buf, dtype=msg["dtype"]).reshape(msg["shape"])
        elif t == "done":
            return arrays, msg
        elif t == "error":
            raise RuntimeError(f"[{msg['stage']}] {msg['message']}")


def call(host, port, header, payload=b""):
    with socket.create_connection((host, port), timeout=300) as s:
        s.settimeout(300)
        return rpc(s, header, payload)


def find_images():
    imgs = sorted(list(IMG_DIR.glob("*.png")) + list(IMG_DIR.glob("*.jpg")))
    if len(imgs) != 5:
        sys.exit(f"预期 5 张机械臂图, 实际 {len(imgs)}: {IMG_DIR}")
    return imgs


def sample_video(path: Path, k: int):
    """取视频中段 k 帧 (跳过片头淡入, 首帧作参考图更稳)。"""
    cap = cv2.VideoCapture(str(path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    picks = sorted({int(t * total) for t in np.linspace(0.15, 0.85, k)})
    frames, i = [], 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i in picks:
            frames.append(frame)
        i += 1
    cap.release()
    return frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8317)
    args = ap.parse_args()
    VIS_DIR.mkdir(parents=True, exist_ok=True)

    # ---- 阶段 1: sam3/text, 5 张机械臂图 ----
    print("===== 阶段 1: sam3/text × 5 张机械臂图 =====")
    imgs = find_images()
    with socket.create_connection((args.host, args.port), timeout=300) as s:
        s.settimeout(300)
        for p in imgs:
            data = p.read_bytes()
            t0 = time.perf_counter()
            arrays, done = rpc(s, {
                "type": "infer", "model": "sam3", "mode": "text",
                "image_bytes": len(data), "text": ["robotic arm"], "conf": 0.25,
            }, data)
            dt = (time.perf_counter() - t0) * 1000
            k = done["k"]
            assert k >= 1, f"{p.name}: 未检出机械臂"
            assert arrays["masks"].shape == (k, done["height"], done["width"])
            assert arrays["boxes"].shape == (k, 4)
            assert arrays["scores"].shape == (k,)
            print(f"  {p.name}: K={k} scores={np.round(arrays['scores'], 3).tolist()} "
                  f"boxes={[np.round(b).astype(int).tolist() for b in arrays['boxes']]} | {dt:.0f}ms")
            # 可视化: mask 轮廓叠加
            img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            vis = img.copy()
            for m in arrays["masks"]:
                vis[m > 0] = (vis[m > 0] * 0.4 + np.array([0, 200, 0]) * 0.6).astype(np.uint8)
            cv2.imwrite(str(VIS_DIR / f"{p.stem}_vis.png"), vis)
    print(f"  可视化已存: {VIS_DIR}")

    # ---- 阶段 2: vggt 回归 (旧协议 + 新协议) ----
    print("\n===== 阶段 2: vggt 回归 =====")
    two = [p.read_bytes() for p in imgs[:2]]
    arrays, done = call(args.host, args.port, {  # 旧协议: 不带 model 字段
        "type": "infer", "mode": "images", "images_bytes": [len(b) for b in two],
        "params": {"resolution": 512},
    }, b"".join(two))
    n, h, w = done["frames"], done["height"], done["width"]
    assert n == 2 and arrays["depth"].shape == (2, h, w), "vggt 旧协议回归失败"
    assert arrays["extrinsics"].shape == (2, 3, 4) and arrays["intrinsics"].shape == (2, 3, 3)
    print(f"  旧协议 (无 model 字段): {n}帧 {w}x{h} 深度范围 "
          f"{arrays['depth'].min():.1f}~{arrays['depth'].max():.1f} | {done['infer_ms']}ms")
    arrays, done = call(args.host, args.port, {  # 新协议: model=vggt
        "type": "infer", "model": "vggt", "mode": "images",
        "images_bytes": [len(two[0])], "params": {"resolution": 512},
    }, two[0])
    assert done["frames"] == 1 and arrays["depth"].shape == (1, done["height"], done["width"])
    print(f"  新协议 (model=vggt): 单帧 {done['width']}x{done['height']} OK | {done['infer_ms']}ms")

    # ---- 阶段 3: sam3 组合用法 (text 切 mask → reference 传播) ----
    print("\n===== 阶段 3: sam3/reference (视频帧传播) =====")
    video = next(Path(r"D:\MoGe3\测试场景").glob("*.mp4"))
    frames = sample_video(video, 4)
    ref, targets = frames[0], frames[1:3]
    ok, ref_arr = cv2.imencode(".jpg", ref, [cv2.IMWRITE_JPEG_QUALITY, 92])
    ref_jpg = ref_arr.tobytes()
    # 3a. text 切参考帧 (概念候选兜底, 取第一个检出的)
    ref_mask, used = None, None
    for concept in ["road", "sky", "car", "building", "tree", "person"]:
        arrays, done = call(args.host, args.port, {
            "type": "infer", "model": "sam3", "mode": "text",
            "image_bytes": len(ref_jpg), "text": [concept], "conf": 0.25,
        }, ref_jpg)
        if done["k"] >= 1:
            ref_mask, used = arrays["masks"][0], concept
            print(f"  text 切参考帧 '{concept}': K={done['k']} "
                  f"scores={np.round(arrays['scores'], 3).tolist()}")
            break
    assert ref_mask is not None, "参考帧所有候选概念均未检出"
    # 3b. reference 传播
    ok, mask_arr = cv2.imencode(".png", (ref_mask * 255).astype(np.uint8))
    mask_png = mask_arr.tobytes()
    tg_jpgs = []
    for t in targets:
        ok, jpg = cv2.imencode(".jpg", t, [cv2.IMWRITE_JPEG_QUALITY, 92])
        tg_jpgs.append(jpg.tobytes())
    payload = ref_jpg + mask_png + b"".join(tg_jpgs)
    arrays, done = call(args.host, args.port, {
        "type": "infer", "model": "sam3", "mode": "reference",
        "ref_image_bytes": len(ref_jpg), "ref_mask_bytes": len(mask_png),
        "images_bytes": [len(j) for j in tg_jpgs],
    }, payload)
    n = done["frames"]
    assert arrays["masks"].shape == (n, done["height"], done["width"]) and n == len(targets)
    print(f"  reference 传播 {n} 帧: scores={done['scores']} 面积={done['mask_areas']}")

    print("\n全部通过 ✓ (sam3/text ×5, vggt 新旧协议, sam3/reference)")


if __name__ == "__main__":
    main()
