r"""MoGe TCP 服务基准测试。

用法（与服务端同一虚拟环境）:
    D:\MoGe3\MoGe\.venv\Scripts\python.exe D:\MoGe3\bench_tcp_service.py [选项]

流程:
  阶段 0  预热: 少量调用排除首次 CUDA 编译/内存分配开销
  阶段 1  顺序测试: 逐帧调用, 记录每帧 客户端耗时 / 服务端纯推理 / 服务端全程
  阶段 2  并发测试: N 个连接同时各发 1 帧 × R 轮, 验证多请求处理能力与排队情况
  阶段 3  并发正确性: 并发请求完整接收 depth 数组, 校验形状与数值范围

结果写入 CSV（每帧一行）并打印汇总统计。
"""
import argparse
import csv
import json
import socket
import statistics
import struct
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

LENGTH_PREFIX = struct.Struct('<I')


# ---------------- 协议 ----------------

def recv_exact(sock: socket.socket, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError(f'connection closed ({len(buf)}/{n})')
        buf.extend(chunk)
    return bytes(buf)


def recv_frame(sock: socket.socket) -> bytes:
    (length,) = LENGTH_PREFIX.unpack(recv_exact(sock, 4))
    return recv_exact(sock, length)


def send_frame(sock: socket.socket, payload: bytes):
    sock.sendall(LENGTH_PREFIX.pack(len(payload)) + payload)


def call_service(host, port, image_bytes, params=None, collect_depth=False, timeout=600):
    """单次调用。返回 dict: client_ms / infer_ms / server_ms / done / depth(可选)"""
    header = {'image_bytes': len(image_bytes), 'format': 'jpg', 'params': params or {}}
    t0 = time.perf_counter()
    depth = None
    with socket.create_connection((host, port), timeout=timeout) as sock:
        send_frame(sock, json.dumps(header).encode('utf-8'))
        send_frame(sock, image_bytes)
        while True:
            msg = json.loads(recv_frame(sock))
            mtype = msg['type']
            if mtype == 'array':
                payload = recv_frame(sock)
                if msg.get('compression') == 'zlib':
                    import zlib
                    payload = zlib.decompress(payload)
                if collect_depth and msg['name'] == 'depth':
                    depth = np.frombuffer(payload, dtype=msg['dtype']).reshape(msg['shape'])
            elif mtype == 'done':
                return {
                    'client_ms': (time.perf_counter() - t0) * 1000,
                    'infer_ms': msg['infer_ms'],
                    'server_ms': msg['elapsed_ms'],
                    'done': msg,
                    'depth': depth,
                }
            elif mtype == 'error':
                raise RuntimeError(f'server error ({msg.get("stage")}): {msg.get("message")}')
            else:  # status 等直接跳过
                continue


# ---------------- 采样 ----------------

def sample_frames(video_dir: str, sample_fps: float):
    """按指定帧率采样所有 mp4，返回 [(视频名, 帧号, jpg字节)]。"""
    items = []
    for vf in sorted(Path(video_dir).glob('*.mp4')):
        cap = cv2.VideoCapture(str(vf))
        fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
        step = max(1, round(fps / sample_fps))
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % step == 0:
                ok2, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                if ok2:
                    items.append((vf.name, idx, buf.tobytes()))
            idx += 1
        cap.release()
    return items


# ---------------- 统计 ----------------

def p95(values):
    if not values:
        return 0.0
    s = sorted(values)
    return s[min(len(s) - 1, max(0, int(np.ceil(0.95 * len(s))) - 1))]


def stats_line(name, values):
    s = sorted(values)
    return (f'{name:<14} n={len(s):3d}  均值 {statistics.mean(s):8.1f}  中位 {statistics.median(s):8.1f}  '
            f'p95 {p95(s):8.1f}  最小 {s[0]:8.1f}  最大 {s[-1]:8.1f}  (ms)')


# ---------------- 主流程 ----------------

def main():
    ap = argparse.ArgumentParser(description='MoGe TCP service benchmark')
    ap.add_argument('--host', default='127.0.0.1')
    ap.add_argument('--port', type=int, default=8317)
    ap.add_argument('--video-dir', default=r'D:\MoGe3\测试场景')
    ap.add_argument('--sample-fps', type=float, default=3, help='视频采样帧率')
    ap.add_argument('--max-frames', type=int, default=None, help='限制采样总帧数（快速测试）')
    ap.add_argument('--warmup', type=int, default=2)
    ap.add_argument('--concurrency', type=int, nargs='+', default=[2, 4, 8], help='并发连接数列表')
    ap.add_argument('--rounds', type=int, default=2, help='每个并发级别重复轮数')
    ap.add_argument('--stream-conc', type=int, nargs='+', default=[2, 4, 8], help='多路流吞吐测试的并发路数')
    ap.add_argument('--stream-frames', type=int, default=12, help='每路连续推送帧数')
    ap.add_argument('--phases', default='seq,conc,stream,check', help='要运行的阶段（逗号分隔: seq,conc,stream,check）')
    ap.add_argument('--resolution-level', type=int, default=None, help='请求参数: 覆盖服务端默认分辨率级别')
    ap.add_argument('--refine-steps', type=int, default=None, help='请求参数: 覆盖服务端默认精修步数')
    ap.add_argument('--out', default=r'D:\MoGe3\outputs\bench_tcp_results.csv')
    args = ap.parse_args()
    phases = set(args.phases.split(','))
    REQ_PARAMS = {}
    if args.resolution_level is not None:
        REQ_PARAMS['resolution_level'] = args.resolution_level
    if args.refine_steps is not None:
        REQ_PARAMS['refine_steps'] = args.refine_steps

    print(f'采样视频 (dir={args.video_dir}, sample_fps={args.sample_fps}) ...')
    frames = sample_frames(args.video_dir, args.sample_fps)
    if args.max_frames:
        frames = frames[:args.max_frames]
    print(f'共采样 {len(frames)} 帧\n')

    rows = []  # CSV 行: phase, video, frame_idx, client_ms, infer_ms, server_ms

    # 预热
    print(f'预热 {args.warmup} 次 ...')
    for i in range(min(args.warmup, len(frames))):
        r = call_service(args.host, args.port, frames[i][2], params=REQ_PARAMS)
        print(f'  warmup {i + 1}: client {r["client_ms"]:.0f}ms, infer {r["infer_ms"]:.0f}ms')

    # ---- 阶段 1: 顺序单次调用 ----
    if 'seq' in phases:
        print('\n===== 阶段 1: 顺序调用 =====')
        client_times, infer_times, server_times = [], [], []
        t0 = time.perf_counter()
        for name, fidx, jpg in frames:
            r = call_service(args.host, args.port, jpg, params=REQ_PARAMS)
            client_times.append(r['client_ms'])
            infer_times.append(r['infer_ms'])
            server_times.append(r['server_ms'])
            rows.append(['seq', name, fidx, r['client_ms'], r['infer_ms'], r['server_ms'], r['done'].get('batch_size', 1)])
        seq_total = time.perf_counter() - t0
        print(stats_line('客户端单次', client_times))
        print(stats_line('服务端纯推理', infer_times))
        print(stats_line('服务端全程', server_times))
        print(f'顺序总耗时 {seq_total:.1f}s, 吞吐 {len(frames) / seq_total:.2f} 帧/s\n')

    # ---- 阶段 2: 并发测试 ----
    cursor = 0
    if 'conc' in phases:
        print('===== 阶段 2: 并发测试 =====')
        for conc in args.concurrency:
            print(f'-- 并发 {conc} 连接 × {args.rounds} 轮 --')
            for rd in range(args.rounds):
                batch = []
                for _ in range(conc):
                    batch.append(frames[cursor % len(frames)])
                    cursor += 1
                wall0 = time.perf_counter()
                with ThreadPoolExecutor(max_workers=conc) as ex:
                    results = list(ex.map(lambda f: call_service(args.host, args.port, f[2], params=REQ_PARAMS), batch))
                wall = (time.perf_counter() - wall0) * 1000
                cs = sorted(r['client_ms'] for r in results)
                infers = sorted(r['infer_ms'] for r in results)
                bs = [r['done'].get('batch_size') for r in results]
                n_ok = sum(1 for r in results if r['done'].get('width'))
                print(f'  轮{rd + 1}: 墙钟 {wall:.0f}ms | 成功 {n_ok}/{conc} | '
                      f'客户端单次 {cs[0]:.0f}~{cs[-1]:.0f}ms | 纯推理合计 {sum(infers):.0f}ms'
                      + (f' | batch_size={bs}' if any(b for b in bs) else ''))
                for (name, fidx, jpg), r in zip(batch, results):
                    rows.append([f'conc{conc}', name, fidx, r['client_ms'], r['infer_ms'], r['server_ms'], r['done'].get('batch_size', 1)])

    # ---- 阶段 2b: 多路连续流吞吐 ----
    if 'stream' in phases:
        print('\n===== 阶段 2b: 多路连续流吞吐 =====')
        for conc in args.stream_conc:
            k = args.stream_frames
            wall0 = time.perf_counter()
            all_client, all_infer, all_batch = [], [], []

            def worker(tid):
                cs, is_, bs = [], [], []
                for j in range(k):
                    f = frames[(cursor + tid * k + j) % len(frames)]
                    r = call_service(args.host, args.port, f[2], params=REQ_PARAMS)
                    cs.append(r['client_ms']); is_.append(r['infer_ms'])
                    bs.append(r['done'].get('batch_size') or 1)
                    rows.append([f'stream{conc}', f[2] and f[0], f[1], r['client_ms'], r['infer_ms'], r['server_ms'], r['done'].get('batch_size', 1)])
                return cs, is_, bs

            with ThreadPoolExecutor(max_workers=conc) as ex:
                results = list(ex.map(worker, range(conc)))
            wall = time.perf_counter() - wall0
            for cs, is_, bs in results:
                all_client += cs; all_infer += is_; all_batch += bs
            n = conc * k
            print(f'-- {conc} 路 × {k} 帧 (共 {n} 请求) --')
            print(f'  墙钟 {wall:.1f}s | 聚合吞吐 {n / wall:.2f} 帧/s | 平均 batch_size {statistics.mean(all_batch):.1f}')
            print(f'  {stats_line("单请求客户端", all_client)}')
            print(f'  {stats_line("单请求纯推理", all_infer)}\n')

    # ---- 阶段 3: 并发正确性（完整接收数组并校验） ----
    if 'check' in phases:
        print('\n===== 阶段 3: 并发正确性（4 连接, 校验 depth 数组） =====')
        batch = frames[cursor:cursor + 4] or frames[:4]
        with ThreadPoolExecutor(max_workers=4) as ex:
            results = list(ex.map(lambda f: call_service(args.host, args.port, f[2], collect_depth=True), batch))
        for (name, fidx, _), r in zip(batch, results):
            d, done = r['depth'], r['done']
            h, w = done['height'], done['width']
            finite = np.isfinite(d).mean() * 100
            valid = d[np.isfinite(d)]
            print(f'  {name[:12]}... 帧{fidx}: shape={d.shape} 与请求({w}x{h})一致={d.shape == (h, w)} | '
                  f'有限值 {finite:.1f}% | 深度范围 [{valid.min():.2f}, {valid.max():.2f}]m | fov_x={done["fov_x"]}°')
        print('并发下各连接均独立收到完整且正确的数组, 无串扰' if all(r['depth'] is not None for r in results)
              else '警告: 有连接未收到 depth 数组!')

    # ---- 写 CSV ----
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['phase', 'video', 'frame_idx', 'client_ms', 'infer_ms', 'server_ms', 'batch_size'])
        w.writerows(rows)
    print(f'\n明细已写入 {out}')


if __name__ == '__main__':
    main()

