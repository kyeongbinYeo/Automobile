#!/usr/bin/env python3
"""Fine-tune the decoder heads of the official pretrained TwinLiteNet on the weak labels.

python train_twinlite.py --data REAL.npz --repo ../third_party/TwinLiteNet --out runs/tw
The encoder stays frozen. Dilated convolutions are very slow on CPU (about 0.8 s per image), so
the encoder features are computed once for a fixed set of augmented copies and the decoders train
on those cached features. Same frame split and metric as train.py, so results compare with LineNet.
Training and inference use the 320x120 crop size.
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from common import IN_H, IN_W, adaptive_gamma
from evaluate import evaluate
from perception import TwinLiteNetBackend
from train import augment, is_excluded


def to_input(bgr, gamma):
    if gamma:
        bgr = adaptive_gamma(bgr)
    return np.ascontiguousarray(bgr[:, :, ::-1].transpose(2, 0, 1)).astype(np.float32) / 255.0


def score(backend, frames, thr=0.5):
    cache = {}

    def predict(i, bgr):
        if i not in cache:
            cache[i] = backend.predict(bgr)
        return (cache[i][0] > thr).astype(np.uint8)
    r = evaluate(predict, frames)
    return {k: r[k] for k in ("lit", "dark")}, r["per_frame"]


def build_cache(net, d, ids, copies, rng, dark_aug):
    """Encoder features, labels and lit flags for `copies` versions of every training frame."""
    feats, lines, mats, lit = [], [], [], []
    batch = []
    t0 = time.time()
    net.eval()

    def flush():
        with torch.no_grad():
            feats.append(net.encoder(torch.from_numpy(np.stack(batch))))
        batch.clear()

    for n, i in enumerate(ids):
        for k in range(copies):
            if k == 0:
                im, li, ma = d["img"][i], d["line"][i], d["mat"][i]
            else:
                im, li, ma = augment(d["img"][i], d["line"][i], d["mat"][i], rng, dark_aug)
            batch.append(to_input(im, True))
            lines.append(li)
            mats.append(ma)
            lit.append(bool(d["lit"][i]))
            if len(batch) == 8:
                flush()
        if n % 40 == 0:
            print(f"  encoder features {n}/{len(ids)} frames, {time.time() - t0:.0f}s", flush=True)
    if batch:
        flush()
    return (torch.cat(feats), torch.from_numpy(np.stack(lines).astype(np.int64)),
            torch.from_numpy(np.stack(mats).astype(np.int64)), torch.tensor(lit))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--copies", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--pos-weight", type=float, default=8.0)
    ap.add_argument("--dark-aug", type=float, default=0.25)
    ap.add_argument("--dark-weight", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    torch.set_num_threads(12)
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)

    z = np.load(a.data)
    d = {k: z[k] for k in z.files}
    frames = {int(i): d["img"][int(i)] for i in d["idx"]}
    ids = [int(i) for i in d["idx"] if not is_excluded(int(i))]
    weights = os.path.join(a.repo, "pretrained", "best.pth")
    result = {}

    backend = TwinLiteNetBackend.from_repo(a.repo, weights, size=(IN_W, IN_H))
    result["zero_shot_320x120"], _ = score(backend, frames)
    print("zero-shot 320x120", result["zero_shot_320x120"], flush=True)
    net = backend.model

    t0 = time.time()
    feats, line, mat, lit = build_cache(net, d, ids, a.copies, rng, a.dark_aug)
    print(f"cached {len(feats)} samples in {time.time() - t0:.0f}s", flush=True)

    heads = {"da": (net.up_1_1, net.up_2_1, net.classifier_1), "ll": (net.up_1_2, net.up_2_2, net.classifier_2)}
    params = [p for h in heads.values() for m in h for p in m.parameters()]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=1e-4)
    n = len(feats)
    steps = a.epochs * (n // a.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=steps)
    cw = torch.tensor([1.0, a.pos_weight])

    def run(head, f):
        for m in heads[head]:
            f = m(f)
        return f

    t0 = time.time()
    for ep in range(a.epochs):
        for h in heads.values():
            for m in h:
                m.train()
        perm = torch.randperm(n)
        tot = 0.0
        for b in range(n // a.batch):
            idx = perm[b * a.batch:(b + 1) * a.batch]
            f, li, ma, lt = feats[idx], line[idx], mat[idx], lit[idx]
            ll, da = run("ll", f), run("da", f)
            w = torch.where(lt, 1.0, a.dark_weight)[:, None, None]
            ce_l = (F.cross_entropy(ll, li, weight=cw, reduction="none") * w).mean()
            p = torch.softmax(ll, 1)[:, 1]
            inter = (p * li).sum((1, 2))
            dice = 1 - ((2 * inter + 1) / (p.sum((1, 2)) + li.sum((1, 2)) + 1)).mean()
            ce_d = (F.cross_entropy(da, ma, reduction="none") * lt[:, None, None]).mean()
            loss = ce_l + dice + 0.3 * ce_d
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item()
        if (ep + 1) % 10 == 0 or ep == 0:
            print(f"ep {ep + 1}/{a.epochs} loss {tot / max(n // a.batch, 1):.4f} t={time.time() - t0:.0f}s", flush=True)

    torch.save(net.state_dict(), os.path.join(a.out, "twinlite_ft.pt"))
    net.eval()
    ft = TwinLiteNetBackend(net, size=(IN_W, IN_H), use_gamma=True)
    result["fine_tuned_decoder_320x120"], per = score(ft, frames)
    result["per_frame"] = per
    result["train_samples"] = n
    result["config"] = vars(a)
    print("fine-tuned decoder 320x120", result["fine_tuned_decoder_320x120"], flush=True)
    print("per-frame (P, R):", per, flush=True)
    with open(os.path.join(a.out, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
