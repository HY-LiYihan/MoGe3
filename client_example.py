r"""MoGe-3 TCP 推理服务示例客户端。

用法（使用服务端同一套环境）:
    D:\MoGe3\MoGe\.venv\Scripts\python.exe D:\MoGe3\client_example.py --image 图片路径 [--host 127.0.0.1] [--port 8317]

协议: 所有消息 = 4 字节小端 uint32 长度 + payload。
请求: JSON 头 {"image_bytes": N, "format": "jpg", "params": {...}} + 图片原始字节。
响应: status(received) -> status(inferring) -> [array 元信息 + 数组]*4 -> done（浮点数组为原始字节，mask 为 zlib 压缩）。
"""
import argparse
import json
import socket
import struct
import zlib
from pathlib import Path

import numpy as np

LENGTH_PREFIX = struct.Struct('<I')


def recv_exact(sock: socket.socket, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError(f'connection closed by peer ({len(buf)}/{n} bytes received)')
        buf.extend(chunk)
    return bytes(buf)


def recv_frame(sock: socket.socket) -> bytes:
    (length,) = LENGTH_PREFIX.unpack(recv_exact(sock, 4))
    return recv_exact(sock, length)


def send_frame(sock: socket.socket, payload: bytes):
    sock.sendall(LENGTH_PREFIX.pack(len(payload)) + payload)


def request(host: str, port: int, image_path: str, params: dict = None, timeout: float = 120):
    """发送一张图片，返回 (arrays dict, done dict)。出错时抛 RuntimeError。"""
    image_bytes = Path(image_path).read_bytes()
    header = {
        'image_bytes': len(image_bytes),
        'format': Path(image_path).suffix.lstrip('.').lower() or 'jpg',
        'params': params or {},
    }
    arrays, result = {}, None
    with socket.create_connection((host, port), timeout=timeout) as sock:
        send_frame(sock, json.dumps(header).encode('utf-8'))
        send_frame(sock, image_bytes)
        while True:
            msg = json.loads(recv_frame(sock))
            mtype = msg['type']
            if mtype == 'status':
                extra = {k: v for k, v in msg.items() if k not in ('type', 'stage')}
                print(f'[status] {msg["stage"]} {extra if extra else ""}')
            elif mtype == 'array':
                payload = recv_frame(sock)
                if msg.get('compression', 'zlib') == 'zlib':
                    payload = zlib.decompress(payload)
                arr = np.frombuffer(payload, dtype=msg['dtype']).reshape(msg['shape'])
                arrays[msg['name']] = arr
                print(f'[array]  {msg["name"]}: shape={list(arr.shape)}, dtype={arr.dtype}')
            elif mtype == 'done':
                result = msg
                break
            elif mtype == 'error':
                raise RuntimeError(f'server error at stage "{msg.get("stage")}": {msg.get("message")}')
            else:
                raise RuntimeError(f'unknown message type: {mtype}')
    return arrays, result


def main():
    parser = argparse.ArgumentParser(description='MoGe-3 TCP client example')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8317)
    parser.add_argument('--image', required=True)
    parser.add_argument('--resolution-level', type=int, default=None, help='可选：覆盖服务端默认值 (0-9)')
    parser.add_argument('--refine-steps', type=int, default=None, help='可选：覆盖服务端默认精修步数')
    parser.add_argument('--fov-x', type=float, default=None, help='可选：已知相机水平 FOV（度），传入可提升精度')
    args = parser.parse_args()

    params = {}
    if args.resolution_level is not None:
        params['resolution_level'] = args.resolution_level
    if args.refine_steps is not None:
        params['refine_steps'] = args.refine_steps
    if args.fov_x is not None:
        params['fov_x'] = args.fov_x

    arrays, result = request(args.host, args.port, args.image, params)

    depth, mask = arrays['depth'], arrays['mask'] > 0
    valid = depth[mask]
    print('\n===== 结果摘要 =====')
    print(f'分辨率: {result["width"]}x{result["height"]}')
    print(f'fov_x: {result["fov_x"]}°  fov_y: {result["fov_y"]}°')
    print(f'intrinsics (归一化): {result["intrinsics"]}')
    print(f'有效像素: {mask.mean() * 100:.1f}%')
    print(f'depth 范围 (有效像素): [{valid.min():.3f}, {valid.max():.3f}] m')
    print(f'服务端耗时: 总 {result["elapsed_ms"]} ms（推理 {result["infer_ms"]} ms）')


if __name__ == '__main__':
    main()
