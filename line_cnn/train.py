#!/usr/bin/env python3
"""Train LineNet: python train.py --data DIR/real.npz --out runs/x [options]"""
import argparse
import json
import os
import random
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from common import to_tensor_np
from evaluate import evaluate
from model import LineNet

# Time blocks held out from training, plus a margin against adjacent-frame leakage.
TEST_BLOCKS = [(90, 140), (385, 435), (445, 475), (690, 715), (240, 300), (585, 630)]
MARGIN = 10


def is_excluded(i):
    return any(a - MARGIN <= i <= b + MARGIN for a, b in TEST_BLOCKS)


def augment(img, line, mat, rng, dark_aug):
    if rng.random() < 0.5:   # mirror labels too
        img, line, mat = img[:, ::-1], line[:, ::-1], mat[:, ::-1]
    h, w = img.shape[:2]
    M = np.float32([[1, 0, rng.integers(-24, 25)], [0, 1, rng.integers(-8, 9)]])
    img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    line = cv2.warpAffine(line, M, (w, h), flags=cv2.INTER_NEAREST)
    mat = cv2.warpAffine(mat, M, (w, h), flags=cv2.INTER_NEAREST)
    x = img.astype(np.float32)
    x = x * rng.uniform(0.7, 1.25) + rng.uniform(-18, 18)
    if rng.random() < 0.3:
        x = cv2.GaussianBlur(x, (0, 0), rng.uniform(0.5, 1.5))
    x = x + rng.normal(0, rng.uniform(0, 6), x.shape)
    img = np.clip(x, 0, 255).astype(np.uint8)
    if dark_aug and rng.random() < dark_aug:
        # Make a lit frame tunnel-like: strong darkening plus compression artifacts.
        g = rng.uniform(2.0, 5.0)
        img = (255 * (img / 255.0) ** g * rng.uniform(0.4, 1.0)).astype(np.uint8)
        if rng.random() < 0.6:
            _, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(15, 50))])
            img = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    return img, line, mat


def seg_loss(logit, target, mask, kind, pos_weight):
    """kind: bce = plain BCE, focal_dice = for sparse line pixels."""
    if kind == "bce":
        l = F.binary_cross_entropy_with_logits(logit, target, reduction="none")
        return (l * mask).sum() / mask.sum().clamp(min=1)
    p = torch.sigmoid(logit)
    ce = F.binary_cross_entropy_with_logits(
        logit, target, reduction="none", pos_weight=torch.tensor(pos_weight))
    pt = p * target + (1 - p) * (1 - target)
    focal = (ce * (1 - pt) ** 2 * mask).sum() / mask.sum().clamp(min=1)
    inter = (p * target * mask).sum((1, 2))
    den = (p * mask).sum((1, 2)) + (target * mask).sum((1, 2))
    dice = 1 - ((2 * inter + 1) / (den + 1)).mean()
    return focal + dice


def load_batch(d, ids, rng, a):
    xs, ls, ms, mm, lw = [], [], [], [], []
    for i in ids:
        im, li, ma = augment(d["img"][i], d["line"][i], d["mat"][i], rng, a.dark_aug)
        xs.append(to_tensor_np(im, a.gamma)[0])
        ls.append(li)
        ms.append(ma)
        # Tunnel frames have no overlay, so no mat loss.
        mm.append(1.0 if d["lit"][i] else 0.0)
        lw.append(1.0 if d["lit"][i] else a.dark_weight)
    x = torch.from_numpy(np.stack(xs)).float()
    li = torch.from_numpy(np.stack(ls).astype(np.float32))
    ma = torch.from_numpy(np.stack(ms).astype(np.float32))
    return x, li, ma, torch.tensor(mm), torch.tensor(lw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--base", type=int, default=16)
    ap.add_argument("--loss", default="bce", choices=["bce", "focal_dice"])
    ap.add_argument("--pos-weight", type=float, default=8.0)
    ap.add_argument("--gamma", action="store_true", help="adaptive input gamma")
    ap.add_argument("--dark-aug", type=float, default=0.0, help="probability of turning a lit frame tunnel-like")
    ap.add_argument("--dark-weight", type=float, default=1.0, help="loss weight of weak tunnel labels")
    ap.add_argument("--no-dark", action="store_true", help="exclude weakly labeled tunnel frames")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)

    z = np.load(a.data)
    d = {k: z[k] for k in z.files}
    ids = [int(i) for i in d["idx"] if not is_excluded(int(i)) and (d["lit"][i] or not a.no_dark)]
    print(f"train frames {len(ids)} (lit {sum(d['lit'][i] for i in ids)})")

    net = LineNet(a.base)
    n_params = sum(p.numel() for p in net.parameters())
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    steps = a.epochs * (len(ids) // a.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=steps)

    t0 = time.time()
    for ep in range(a.epochs):
        net.train()
        rng.shuffle(ids)
        tot = 0.0
        for b in range(len(ids) // a.batch):
            x, li, ma, mm, lw = load_batch(d, ids[b * a.batch:(b + 1) * a.batch], rng, a)
            out = net(x)
            mask = lw[:, None, None] * torch.ones_like(li)
            l_line = seg_loss(out[:, 0], li, mask, a.loss, a.pos_weight)
            m_mask = mm[:, None, None] * torch.ones_like(ma)
            l_mat = F.binary_cross_entropy_with_logits(out[:, 1], ma, weight=m_mask)
            loss = l_line + 0.3 * l_mat
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            tot += loss.item()
        if (ep + 1) % 10 == 0 or ep == a.epochs - 1:
            print(f"ep {ep + 1}/{a.epochs} loss {tot / max(len(ids) // a.batch, 1):.4f} t={time.time() - t0:.0f}s", flush=True)

    torch.save({"state": net.state_dict(), "base": a.base, "gamma": a.gamma}, os.path.join(a.out, "linenet.pt"))

    net.eval()
    frames = {int(i): d["img"][int(i)] for i in d["idx"]}

    @torch.no_grad()
    def predict(i, bgr, thr=0.5):
        x = torch.from_numpy(to_tensor_np(bgr, a.gamma)).float()
        return (torch.sigmoid(net(x))[0, 0].numpy() > thr).astype(np.uint8)

    m = evaluate(predict, frames)
    m.update({"params": n_params, "config": vars(a)})
    with open(os.path.join(a.out, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(m, f, indent=1, ensure_ascii=False)
    print("RESULT lit", m["lit"], "dark", m["dark"], "params", n_params)
    print("per_frame", m["per_frame"])


if __name__ == "__main__":
    main()
