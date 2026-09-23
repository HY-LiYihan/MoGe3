"""离线验证 SAM3 两条链路 (不经 TCP): text 模式 + reference 视频传播模式。

用法: python test_local.py [视频路径]
"""
import sys
import time
import tempfile
import shutil
from pathlib import Path

import cv2
import numpy as np
import torch

CKPT = r"D:\MoGe3\checkpoints\sam3\sam3.pt"


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


def main():
    video = sys.argv[1] if len(sys.argv) > 1 else r"D:\MoGe3\vggt-omega\examples\desert_road.mp4"
    frames = grab_frames(video, 3)
    print(f"抽帧 {len(frames)} 张 {frames[0].shape}")

    from ultralytics.models.sam import SAM3SemanticPredictor, SAM3VideoPredictor
    ov = {"task": "segment", "mode": "predict", "model": CKPT, "quantize": 16,
          "conf": 0.25, "save": False, "verbose": False, "retina_masks": True}

    # ---- 链路 1: text ----
    t0 = time.perf_counter()
    sp = SAM3SemanticPredictor(overrides=ov)
    sp.setup_model()
    print(f"\n[text] predictor 构建+权重加载 {time.perf_counter()-t0:.1f}s, "
          f"显存 {torch.cuda.memory_allocated()/2**30:.2f} GB")

    t0 = time.perf_counter()
    sp.set_image(frames[0])
    results = sp(text=["sky", "road"])
    t_text = time.perf_counter() - t0
    r = results[0]
    print(f"[text] 'sky','road' → 耗时 {t_text:.2f}s")
    print(f"[text] masks: {tuple(r.masks.data.shape)}, boxes: {tuple(r.boxes.xyxy.shape)}, "
          f"conf: {[round(float(c),3) for c in r.boxes.conf]}")
    sky_masks = r.masks.data.cpu().numpy().astype(np.uint8)
    sky_idx = list(r.boxes.cls.cpu().numpy()).index(0)  # 类 0 = sky
    sky_mask = sky_masks[sky_idx]
    print(f"[text] sky 面积占比: {sky_mask.mean()*100:.1f}%")

    # ---- 链路 2: reference (视频跟踪传播) ----
    t0 = time.perf_counter()
    vp = SAM3VideoPredictor(overrides=ov)
    vp.setup_model()
    print(f"\n[ref] predictor 构建完成 {time.perf_counter()-t0:.1f}s, "
          f"显存 {torch.cuda.memory_allocated()/2**30:.2f} GB")

    tmp = tempfile.mkdtemp(prefix="sam3_local_")
    try:
        video_path = f"{tmp}/seq.mp4"
        h, w = frames[0].shape[:2]
        vw = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"mp4v"), 10, (w, h))
        for f in frames:
            vw.write(f)
        vw.release()
        vp.inference_state = {}
        t0 = time.perf_counter()
        ys, xs = np.nonzero(sky_mask)
        bbox = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]
        results = list(vp(source=video_path, bboxes=[bbox], stream=True))
        t_ref = time.perf_counter() - t0
        print(f"[ref] 3 帧传播耗时 {t_ref:.2f}s ({t_ref/3*1000:.0f}ms/帧)")
        for i, rr in enumerate(results):
            if rr.masks is None:
                print(f"[ref] 帧{i}: 无 mask")
                continue
            m = rr.masks.data.cpu().numpy()
            m = m[0] if m.ndim == 3 else m
            print(f"[ref] 帧{i}: mask shape {m.shape}, 面积占比 {m.mean()*100:.1f}%")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print(f"\n峰值显存: {torch.cuda.max_memory_allocated()/2**30:.2f} GB")


if __name__ == "__main__":
    main()
