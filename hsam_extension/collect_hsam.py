"""
把 H-SAM 的 BUSI 結果整理進 GitHub 倉庫（psi-weighted-hsam），格式同 psi-weighted-ukan
  results/percase/busi_HSAM_<模型>_last.csv     每張影像的 Dice（原始解析度，官方評估）＋ dice256（縮到 256×256，與 U-KAN 相同，只用來和 U-KAN 比較）
  results/training_logs/busi_HSAM_<模型>_log.txt、_config.txt    官方 trainer 的逐迭代紀錄與設定
  weight_lists/    Ψ 篩選結果與加權清單（由 baseline 的訓練集預測產生）
  hsam_extension/  我們的程式（patch、預測、篩選、訓練腳本、整理、分析）
  分析表格與圖由 analyze_hsam.py 產生（results/analysis/）
用法：python collect_hsam.py --base ~/桌面/HSAM --repo ~/桌面/論文/psi-weighted-hsam --masks ~/桌面/BUSI/U-KAN/Seg_UKAN/inputs/busi/masks/0
"""
import argparse
import glob
import os
import shutil

import cv2
import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument('--base', required=True, help='HSAM 資料夾（含 H-SAM/、outputs/、psi_lists/）')
ap.add_argument('--repo', required=True)
ap.add_argument('--masks', required=True, help='U-KAN 的 inputs/busi/masks/0')
args = ap.parse_args()
B, R = os.path.expanduser(args.base), os.path.expanduser(args.repo)
MODELS = [  # (名稱, loss, 加權清單)
    ('base', 'official (batch Dice)', 'none'),
    ('ctrl', 'per-case, w = 1', 'none'),
    ('psi1w3n', 'per-case, x3 / n', 'psi1_convex'),
    ('psi6w3n', 'per-case, x3 / n', 'psi6_cc'),
    ('psi8w3n', 'per-case, x3 / n', 'psi8_rotation'),
]
for d in ('results/percase', 'results/training_logs', 'results/analysis', 'weight_lists', 'hsam_extension'):
    os.makedirs(os.path.join(R, d), exist_ok=True)


def dice(a, b):
    s = a.sum() + b.sum()
    return 1.0 if s == 0 else 2.0 * (a & b).sum() / s


M = {}
for m, _, _ in MODELS:
    o = os.path.join(B, 'outputs', f'busi_HSAM_{m}')
    f = os.path.join(o, 'percase_last.csv')
    if not os.path.exists(f):
        print(f'略過 {m}：沒有 {f}')
        continue
    d = pd.read_csv(f)
    d256 = []
    for c, sp in zip(d.case, d.split):
        p = cv2.imread(os.path.join(o, 'pred_last', sp, c + '.png'), cv2.IMREAD_GRAYSCALE) > 127
        g = cv2.imread(os.path.join(os.path.expanduser(args.masks), c + '_mask.png'), cv2.IMREAD_GRAYSCALE) > 127
        r = lambda a: cv2.resize(a.astype(np.uint8), (256, 256), interpolation=cv2.INTER_NEAREST) > 0
        d256.append(dice(r(p), r(g)))
    d.insert(d.columns.get_loc('dice') + 1, 'dice256', d256)
    d.round(6).to_csv(os.path.join(R, 'results/percase', f'busi_HSAM_{m}_last.csv'), index=False)
    M[m] = d.set_index('case')
    snap = sorted(glob.glob(os.path.join(o, '*', 'log.txt')))
    if snap:
        shutil.copy(snap[0], os.path.join(R, 'results/training_logs', f'busi_HSAM_{m}_log.txt'))
        cfg = os.path.join(os.path.dirname(snap[0]), 'config.txt')
        if os.path.exists(cfg):
            shutil.copy(cfg, os.path.join(R, 'results/training_logs', f'busi_HSAM_{m}_config.txt'))
for f in glob.glob(os.path.join(B, 'psi_lists', '*.csv')):
    shutil.copy(f, os.path.join(R, 'weight_lists'))
H = os.path.join(B, 'H-SAM')
for f in ('prep_busi_hsam.py', 'patch_hsam_weighted.py', 'patch_hsam_accum.py', 'predict_hsam.py',
          'screen_hsam.py', 'memtest_hsam.py', 'run_hsam_busi.sh', 'collect_hsam.py', 'analyze_hsam.py'):
    if os.path.exists(os.path.join(H, f)):
        shutil.copy(os.path.join(H, f), os.path.join(R, 'hsam_extension'))

print(f'已整理 {len(M)} 個模型 → {R}；接著在倉庫根目錄執行 python hsam_extension/analyze_hsam.py')
