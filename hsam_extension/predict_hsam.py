"""
用訓練好的 H-SAM 預測 BUSI，存下每張影像的預測遮罩與逐張 Dice，並可計算 Ψ（同 psi-weighted-ukan 的 predict_busi.py）
  推論：與官方 utils.test_single_volume 相同（縮放 order 3 → stage 3 = 兩個 decoder 的輸出平均 → argmax → 縮回原尺寸 order 0）
        input_size = img_size = 512（官方 test.py 的 input_size 預設 224 是給 Synapse 的；BUSI 訓練時直接從原圖縮到 512，所以測試也一樣）
        不給任何 prompt（H-SAM 原本就不用 prompt）
  逐張 Dice：在原始解析度計算（同官方評估）
  Ψ（--psi）：沿用 psi-weighted-nnunet/psi/run_all_psi.py 的函式，只用預測遮罩，不用標註
    Ψ1 凸包比、Ψ6 連通元件、Ψ7 拓撲環：由預測遮罩計算
    Ψ8 旋轉一致性：影像旋轉 15°（邊緣補影像最小值）→ 同一模型預測 → 轉回（最近鄰）→ 1 − Dice(原預測, 轉回的預測)
    Ψ2/3/4/5/9/10：BUSI 無定義，標為 NaN（同 psi-weighted-ukan）
輸出：<out>/percase_last.csv（case, split, dice, iou, pred_pixels, gt_pixels[, psi…]）、<out>/pred_last/<split>/<影像>.png
用法（在 H-SAM 資料夾）：
  python predict_hsam.py --lora_ckpt outputs/busi_HSAM_base/.../epoch_249.pth --data <資料> --out outputs/busi_HSAM_base --psi
"""
import argparse
import importlib.util
import os

import cv2
import h5py
import numpy as np
import pandas as pd
import torch
from einops import repeat
from scipy.ndimage import zoom

from sam_lora_image_encoder import LoRA_Sam
from segment_anything import sam_model_registry

ap = argparse.ArgumentParser()
ap.add_argument('--lora_ckpt', required=True)
ap.add_argument('--data', required=True, help='prep_busi_hsam.py 的輸出資料夾')
ap.add_argument('--out', required=True)
ap.add_argument('--splits', default='train,test')
ap.add_argument('--ckpt', default='checkpoints/sam_vit_b_01ec64.pth')
ap.add_argument('--vit_name', default='vit_b')
ap.add_argument('--img_size', type=int, default=512)
ap.add_argument('--num_classes', type=int, default=1)
ap.add_argument('--rank', type=int, default=5)
ap.add_argument('--stage', type=int, default=3)
ap.add_argument('--psi', action='store_true')
ap.add_argument('--psi_lib', default=os.path.expanduser('~/桌面/論文/psi-weighted-nnunet/psi/run_all_psi.py'))
ap.add_argument('--angle', type=float, default=15.0)
ap.add_argument('--device', default='cuda')
args = ap.parse_args()

sam, _ = sam_model_registry[args.vit_name](image_size=args.img_size, num_classes=args.num_classes,
                                           checkpoint=args.ckpt, pixel_mean=[0, 0, 0], pixel_std=[1, 1, 1])
net = LoRA_Sam(sam, args.rank).to(args.device)
net.load_lora_parameters(args.lora_ckpt)
net.eval()
multimask_output = args.num_classes > 1                               # 同 test.py
print(f'模型：{args.lora_ckpt}')

if args.psi:
    spec = importlib.util.spec_from_file_location('psi_lib', args.psi_lib)
    P = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(P)


@torch.no_grad()
def predict(img):
    """img：(H, W) float [0, 1] → (H, W) bool；步驟同 utils.test_single_volume（3D 分支，單一切片）"""
    x, y = img.shape
    s = zoom(img, (args.img_size / x, args.img_size / y), order=3)
    inputs = torch.from_numpy(s).unsqueeze(0).unsqueeze(0).float().to(args.device)
    inputs = repeat(inputs, 'b c h w -> b (repeat c) h w', repeat=3)
    o1, o2, _, _ = net(inputs, multimask_output, args.img_size, gt=None, mode='test')
    m = (o1['masks'] + o2['masks']) / 2 if args.stage == 3 else o2['masks']
    out = torch.argmax(torch.softmax(m, dim=1), dim=1).squeeze(0).cpu().numpy()
    oh, ow = out.shape
    if (x, y) != (oh, ow):
        out = zoom(out, (x / oh, y / ow), order=0)
    return out == 1


def rotate(a, ang, border, interp):
    h, w = a.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, 1.0)
    return cv2.warpAffine(a, M, (w, h), flags=interp, borderMode=cv2.BORDER_CONSTANT, borderValue=border)


def dice(a, b):
    s = a.sum() + b.sum()
    return 1.0 if s == 0 else 2.0 * (a & b).sum() / s


rows = []
for sp in args.splits.split(','):
    cases = [l.strip('\n') for l in open(os.path.join(args.data, 'lists', f'{sp}_vol.txt')) if l.strip()]
    od = os.path.join(args.out, 'pred_last', sp)
    os.makedirs(od, exist_ok=True)
    for k, c in enumerate(cases):
        with h5py.File(os.path.join(args.data, 'vol', c + '.npy.h5'), 'r') as f:
            img, g = f['image'][0], f['label'][0] > 0
        p = predict(img)
        cv2.imwrite(os.path.join(od, c + '.png'), p.astype(np.uint8) * 255)
        inter, uni = (p & g).sum(), (p | g).sum()
        r = dict(case=c, split=sp, dice=dice(p, g), iou=1.0 if uni == 0 else inter / uni,
                 pred_pixels=int(p.sum()), gt_pixels=int(g.sum()))
        if args.psi:
            ir = rotate(img, args.angle, float(img.min()), cv2.INTER_LINEAR)
            pb = rotate(predict(ir).astype(np.uint8), -args.angle, 0, cv2.INTER_NEAREST) > 0
            r.update(psi1_convex=P.psi1_convex(p), psi2_shadow=np.nan, psi3_inclusion=np.nan,
                     psi4_adjacency=np.nan, psi5_exclusion=np.nan, psi6_cc=P.psi6_cc(p),
                     psi7_genus=P.psi7_genus(p), psi8_rotation=1.0 - dice(p, pb),
                     psi9_nesting=np.nan, psi10_centroid=np.nan)
        rows.append(r)
        if (k + 1) % 50 == 0:
            print(f'  {sp} {k + 1}/{len(cases)}', flush=True)
d = pd.DataFrame(rows)
out = os.path.join(args.out, 'percase_last.csv')
if os.path.exists(out):                                           # 保留其他 split 已算好的列
    old = pd.read_csv(out)
    d = pd.concat([old[~old.split.isin(d.split.unique())], d], ignore_index=True)
d.to_csv(out, index=False)
for sp, g in d.groupby('split'):
    print(f'{sp:<5} {len(g):>4} 張　逐張 Dice 平均 {g.dice.mean():.4f}　IoU 平均 {g.iou.mean():.4f}')
print(f'→ {out}')
