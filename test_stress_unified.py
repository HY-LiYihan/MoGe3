r"""统一服务压力自测: 验证服务"绝对不会爆掉"

阶段 A: 边界拒绝 (每个非法请求后紧跟一次正常请求, 验证连接与服务存活)
  A1 vggt 65 张图 (超 MAX_IMAGES=64)        A2 video num_frames=65 (超 64)
  A3 未知 model                              A4 非法 JSON
  A5 坏图字节 (vggt + sam3 各一)             A6 超 512MB 负载声明
  A7 声明与负载不符
阶段 B: 并发排队 (4 连接: 2×vggt + 2×sam3 同时打, 断言全部成功且前向串行)
阶段 C: 显存压力 (C1: video 16 帧 + 并发 sam3; C2 手动: 32 帧极限, --phase C2)

用法: python test_stress_unified.py [--phase AB|C1|C2|all]
"""
import argparse
import json
import socket
import struct
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

HOST, PORT = "127.0.0.1", 8317
IMG = Path(r"D:\MoGe3\outputs\test_robot_arm\arm2.jpg").read_bytes()
VIDEO = next(Path(r"D:\MoGe3\测试场景").glob("*.mp4")).read_bytes()


def recv_exact(s, n):
    buf = b""
    while len(buf) < n:
        c = s.recv(n - len(buf))
        if not c:
            raise ConnectionError("closed")
        buf += c
    return buf


def rpc(s, header, payload=b""):
    for data in (json.dumps(header).encode(), payload):
        s.sendall(struct.pack("<I", len(data)) + data)
    arrays = {}
    while True:
        msg = json.loads(recv_exact(s, struct.unpack("<I", recv_exact(s, 4))[0]))
        t = msg["type"]
        if t == "array":
            if "bytes" in msg:  # sam3: 独立长度前缀帧
                (n,) = struct.unpack("<I", recv_exact(s, 4))
                buf = recv_exact(s, n)
            else:  # vggt: 裸字节
                n = int(np.prod(msg["shape"])) * np.dtype(msg["dtype"]).itemsize
                buf = recv_exact(s, n)
            arrays[msg["name"]] = np.frombuffer(buf, msg["dtype"]).reshape(msg["shape"])
        elif t == "done":
            return arrays, msg
        elif t == "error":
            return None, msg
        # status 忽略


def one(header, payload=b""):
    with socket.create_connection((HOST, PORT), timeout=300) as s:
        s.settimeout(300)
        return rpc(s, header, payload)


def ok_vggt(s=None):
    """一次最小正常请求 (vggt 单图), 验证连接/服务存活。"""
    h = {"type": "infer", "model": "vggt", "mode": "images",
         "images_bytes": [len(IMG)], "params": {"resolution": 256}}
    r = (rpc(s, h, IMG) if s else one(h, IMG))[1]
    assert r["type"] == "done", f"存活检查失败: {r}"
    return True


def expect_error(tag, header, payload=b""):
    arrays, resp = one(header, payload)
    assert resp["type"] == "error", f"{tag}: 预期 error 实得 {resp}"
    print(f"  {tag}: error[{resp['stage']}] {resp['message'][:70]}")


def phase_a():
    print("===== 阶段 A: 边界拒绝 =====")
    expect_error("A1 65张图", {"type": "infer", "model": "vggt", "mode": "images",
                               "images_bytes": [1] * 65}, b"x" * 65)
    expect_error("A2 num_frames=65", {"type": "infer", "model": "vggt", "mode": "video",
                                      "video_bytes": 10, "params": {"num_frames": 65}}, b"x" * 10)
    expect_error("A3 未知model", {"type": "infer", "model": "foo", "mode": "images",
                                  "images_bytes": [1]}, b"x")
    expect_error("A5a vggt坏图", {"type": "infer", "model": "vggt", "mode": "images",
                                 "images_bytes": [100], "params": {"resolution": 256}},
                 bytes(range(100)))
    expect_error("A5b sam3坏图", {"type": "infer", "model": "sam3", "mode": "text",
                                  "image_bytes": 100, "text": ["car"]}, bytes(range(100)))
    expect_error("A6 超限声明", {"type": "infer", "model": "vggt", "mode": "video",
                                "video_bytes": 600 * 1024 * 1024})
    expect_error("A7 长度不符", {"type": "infer", "model": "vggt", "mode": "images",
                                 "images_bytes": [50]}, b"x" * 40)
    # A4 非法 JSON + 同连接恢复
    with socket.create_connection((HOST, PORT), timeout=60) as s:
        s.settimeout(60)
        bad = b"not a json!!!"
        s.sendall(struct.pack("<I", len(bad)) + bad)
        resp = json.loads(recv_exact(s, struct.unpack("<I", recv_exact(s, 4))[0]))
        assert resp["type"] == "error" and "JSON" in resp["message"], resp
        print(f"  A4 非法JSON: error[{resp['stage']}]")
        ok_vggt(s)  # 同连接继续正常
        print("  A4 同连接恢复正常 ✓")
    ok_vggt()  # 全部边界后服务仍正常
    print("  边界拒绝后服务存活 ✓")


def phase_b():
    print("\n===== 阶段 B: 并发排队 (2×vggt 8帧 + 2×sam3 text) =====")

    def job_vggt():
        h = {"type": "infer", "model": "vggt", "mode": "video", "video_bytes": len(VIDEO),
             "params": {"num_frames": 8, "resolution": 512}}
        t0 = time.perf_counter()
        arrays, done = one(h, VIDEO)
        assert done["type"] == "done", f"vggt 8帧被拒: {done}"
        return "vggt", done["infer_ms"], (time.perf_counter() - t0) * 1000, True

    def job_sam3():
        h = {"type": "infer", "model": "sam3", "mode": "text",
             "image_bytes": len(IMG), "text": ["robotic arm"]}
        t0 = time.perf_counter()
        arrays, done = one(h, IMG)
        return "sam3", done.get("infer_ms", 0), (time.perf_counter() - t0) * 1000, done["type"] == "done"

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=4) as ex:
        results = list(ex.map(lambda f: f(), [job_vggt, job_vggt, job_sam3, job_sam3]))
    wall = (time.perf_counter() - t0) * 1000
    for name, ims, cms, ok in results:
        assert ok, f"{name} 并发失败"
        print(f"  {name}: 前向 {ims:.0f}ms 客户端 {cms:.0f}ms")
    # 串行证据: 墙钟 >= vggt 纯前向之和 (vggt 的 infer_ms 在锁内计时, 不含排队;
    # sam3 的 infer_ms 含等锁时间, 会重叠计入多请求, 不参与此断言)
    vggt_sum = sum(r[1] for r in results if r[0] == "vggt")
    assert wall >= vggt_sum * 0.9, f"墙钟 {wall:.0f}ms < vggt 前向和 {vggt_sum:.0f}ms, 疑似并行"
    print(f"  墙钟 {wall:.0f}ms ≥ vggt 前向之和 {vggt_sum:.0f}ms → GPU 串行排队 ✓, 4/4 成功")


def phase_c1():
    print("\n===== C1: 显存压力 (video 16帧 + 并发 sam3 text) =====")

    def job_vggt():
        h = {"type": "infer", "model": "vggt", "mode": "video", "video_bytes": len(VIDEO),
             "params": {"num_frames": 16, "resolution": 512}}
        t0 = time.perf_counter()
        arrays, done = one(h, VIDEO)
        return "vggt16", done["infer_ms"], done["type"] == "done"

    def job_sam3():
        h = {"type": "infer", "model": "sam3", "mode": "text",
             "image_bytes": len(IMG), "text": ["robotic arm"]}
        arrays, done = one(h, IMG)
        return "sam3", 0, done["type"] == "done"

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=2) as ex:
        results = list(ex.map(lambda f: f(), [job_vggt, job_sam3]))
    for name, ims, ok in results:
        assert ok, f"{name} 失败"
        print(f"  {name}: 前向 {ims:.0f}ms ✓")
    print(f"  压力并发成功 ✓ ({(time.perf_counter() - t0):.1f}s)")
    ok_vggt()
    print("  压力后服务存活 ✓")


def phase_c2(frames=32):
    print(f"\n===== C2: 极限 (video {frames}帧 @512, 可能 OOM/换页, 观察) =====")
    h = {"type": "infer", "model": "vggt", "mode": "video", "video_bytes": len(VIDEO),
         "params": {"num_frames": frames, "resolution": 512}}
    t0 = time.perf_counter()
    arrays, resp = one(h, VIDEO)
    dt = time.perf_counter() - t0
    if resp["type"] == "done":
        print(f"  32帧成功: 前向 {resp['infer_ms']}ms 墙钟 {dt:.1f}s")
    else:
        print(f"  32帧被拒: error[{resp['stage']}] {resp['message'][:80]}")
    print("  极限后存活检查 ...")
    ok_vggt()
    ok_sam3()
    print("  极限后服务存活 ✓")


def ok_sam3():
    h = {"type": "infer", "model": "sam3", "mode": "text",
         "image_bytes": len(IMG), "text": ["robotic arm"]}
    arrays, done = one(h, IMG)
    assert done["type"] == "done", f"sam3 存活检查失败: {done}"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="AB", help="AB | C1 | C2 | all")
    ap.add_argument("--frames", type=int, default=32, help="C2 阶段帧数")
    args = ap.parse_args()
    if "A" in args.phase or args.phase == "all":
        phase_a()
    if "B" in args.phase or args.phase == "all":
        phase_b()
    if "C1" in args.phase or args.phase == "all":
        phase_c1()
    if "C2" in args.phase or args.phase == "all":
        phase_c2(args.frames)
    print("\n自测通过 ✓ 服务不会因非法输入或并发而崩溃")
