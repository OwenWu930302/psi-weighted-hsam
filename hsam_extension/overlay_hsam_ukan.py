"""
低分影像的疊合圖：U-KAN 與 H-SAM 在同一張影像上的預測（作法同 psi-discussion 的 overlay_low.py）
  影像 = low_dice_BUSI_wide.csv 的 16 張（H-SAM base 或 U-KAN seed2981 測試集 Dice < 0.5）
  每張一張圖（2 列 × 5）：原圖、標註、U-KAN seed2981 / base2 / shift │ H-SAM base / ctrl / psi1w3n / psi6w3n / psi8w3n
  全部縮到 256×256（影像雙線性、遮罩最近鄰，與 U-KAN 相同），綠 = 標註、紅 = 預測、黃字 = 256×256 的 Dice
  另存 low_dice_overview.jpg（16 張一覽：原圖、U-KAN seed2981、U-KAN shift、H-SAM base、H-SAM psi8w3n）
  與 hard_cases_BUSI.csv（每張影像在 8 個模型的 Dice、幾個模型 < 0.5、幾個 < 0.1）
用法（H-SAM 資料夾，hsam 環境）：
  python overlay_hsam_ukan.py --repo ~/桌面/論文/psi-weighted-hsam
"""
import argparse
import os

import cv2
import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument('--repo', required=True, help='psi-weighted-hsam（讀 low_dice_BUSI_wide.csv，寫入 results/examples、results/analysis）')
ap.add_argument('--busi', default=os.path.expanduser('~/桌面/BUSI/U-KAN/Seg_UKAN/inputs/busi'))
ap.add_argument('--ukan_out', default=os.path.expanduser('~/桌面/BUSI/U-KAN/Seg_UKAN/outputs'))
ap.add_argument('--hsam_out', default='../outputs')
args = ap.parse_args()
R = os.path.expanduser(args.repo)
N = 256
MODELS = [  # (顯示名稱, 預測資料夾)
    ('U-KAN seed2981 (best)', os.path.join(args.ukan_out, 'busi_UKAN_seed2981', 'pred_best', 'val')),
    ('U-KAN base2', os.path.join(args.ukan_out, 'busi_UKAN_base2', 'pred_last', 'val')),
    ('U-KAN shift', os.path.join(args.ukan_out, 'busi_UKAN_shift', 'pred_last', 'val')),
] + [(f'H-SAM {m}', os.path.join(args.hsam_out, f'busi_HSAM_{m}', 'pred_last', 'test'))
     for m in ('base', 'ctrl', 'psi1w3n', 'psi6w3n', 'psi8w3n')]
KEY = {lab: lab.replace(' (best)', '').replace('U-KAN ', 'ukan_').replace('H-SAM ', 'hsam_') for lab, _ in MODELS}   # CSV 欄位名稱
for lab, d in MODELS:
    print(('✓ ' if os.path.isdir(d) else '✗ 找不到 ') + f'{lab}：{d}')
W = pd.read_csv(os.path.join(R, 'results', 'analysis', 'low_dice_BUSI_wide.csv'), encoding='utf-8-sig').set_index('case')
EX = os.path.join(R, 'results', 'examples', 'low_dice')
os.makedirs(EX, exist_ok=True)


def tile(img, gt, pr, text, n=N):
    o = cv2.resize(img, (n, n)) if img.shape[0] != n else img.copy()
    for m, col in [(gt, (0, 255, 0)), (pr, (0, 0, 255))]:
        if m is not None:
            m = cv2.resize(m.astype(np.uint8), (n, n), interpolation=cv2.INTER_NEAREST) if m.shape[0] != n else m
            cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(o, cs, -1, col, 2)
    cv2.rectangle(o, (0, 0), (n, 22), (0, 0, 0), -1)
    cv2.putText(o, text, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45 * n / N, (0, 255, 255), 1, cv2.LINE_AA)
    return o


rows, over = [], []
for c in W.index:
    img = cv2.resize(cv2.imread(os.path.join(args.busi, 'images', c + '.png')), (N, N), interpolation=cv2.INTER_LINEAR)
    gt = cv2.resize(cv2.imread(os.path.join(args.busi, 'masks', '0', c + '_mask.png'), cv2.IMREAD_GRAYSCALE), (N, N),
                    interpolation=cv2.INTER_NEAREST) > 127
    tiles = [tile(img, None, None, c), tile(img, gt, None, 'ground truth')]
    r = dict(case=c, hsam_low=W.hsam_low[c], ukan_low=W.ukan_low[c], nine=W.nine[c])
    P = {}
    for lab, d in MODELS:
        f = os.path.join(d, c + '.png')
        if not os.path.exists(f):
            tiles.append(np.zeros_like(img))
            r[KEY[lab]] = np.nan
            continue
        pr = cv2.resize(cv2.imread(f, cv2.IMREAD_GRAYSCALE), (N, N), interpolation=cv2.INTER_NEAREST) > 127
        dc = 2 * (pr & gt).sum() / max(pr.sum() + gt.sum(), 1)
        r[KEY[lab]], P[lab] = dc, (pr, dc)
        tiles.append(tile(img, gt, pr, f'{lab}  {dc:.2f}'))
    v = np.array([r[KEY[lab]] for lab, _ in MODELS], float)
    r.update(n_models=int(np.isfinite(v).sum()), n_below_05=int((v < 0.5).sum()), n_below_01=int((v < 0.1).sum()))
    rows.append(r)
    cv2.imwrite(os.path.join(EX, c + '.png'), np.vstack([np.hstack(tiles[:5]), np.hstack(tiles[5:10])]))
    n = 192
    t = [tile(img, gt, None, c, n)]
    for lab in ('U-KAN seed2981 (best)', 'U-KAN shift', 'H-SAM base', 'H-SAM psi8w3n'):
        t.append(tile(img, gt, P[lab][0], f'{lab.replace(" (best)", "")} {P[lab][1]:.2f}', n) if lab in P
                 else np.zeros((n, n, 3), np.uint8))
    over.append(np.hstack(t))
    print(f'→ {EX}/{c}.png')
cv2.imwrite(os.path.join(R, 'results', 'examples', 'low_dice_overview.jpg'), np.vstack(over), [cv2.IMWRITE_JPEG_QUALITY, 85])
H = pd.DataFrame(rows).sort_values(['n_below_05', 'n_below_01'], ascending=False)
H.to_csv(os.path.join(R, 'results', 'analysis', 'hard_cases_BUSI.csv'), index=False, encoding='utf-8-sig', float_format='%.4f')
pd.set_option('display.width', 250)
print(H[['case', 'hsam_low', 'ukan_low', 'nine', 'n_models', 'n_below_05', 'n_below_01']].to_string(index=False))
print(f'→ {R}/results/examples/low_dice_overview.jpg、results/analysis/hard_cases_BUSI.csv')
