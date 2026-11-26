"""
把 H-SAM 的 BUSI 結果整理進 GitHub 倉庫（psi-weighted-hsam），格式同 psi-weighted-busi
  results/percase/busi_HSAM_<模型>_last.csv     每張影像的 Dice（原始解析度，官方評估）＋ dice256（縮到 256×256，與 U-KAN 相同，只用來和 U-KAN 比較）
  results/training_logs/busi_HSAM_<模型>_log.txt、_config.txt    官方 trainer 的逐迭代紀錄與設定
  weight_lists/    Ψ 篩選結果與加權清單（由 baseline 的訓練集預測產生）
  hsam_extension/  我們的程式（patch、預測、篩選、訓練腳本）
  results/analysis/
    hsam_BUSI_models.csv    每個模型一列：測試集、U-KAN 低分 15 張、那 9 張、其他 115 張、訓練集、差距
    hsam_BUSI_summary.csv   成對比較（逐張 Wilcoxon）：ctrl vs base、各專家 vs ctrl、各專家 vs base
    hsam_BUSI_percase.csv   每張影像一列、每個模型一欄（測試集 + 訓練集）
  組別沿用 psi-weighted-busi 的定義（由 U-KAN 的 baseline seed2981 定義，和 H-SAM 是不同模型）：
    low_dice 15 = seed2981 best 的測試集 Dice < 0.5；nine 9 = moved_busi_dice02.csv；other 115 = 測試集其餘
用法：python collect_hsam.py --base ~/桌面/HSAM --repo ~/桌面/論文/psi-weighted-hsam --busi_repo ~/桌面/論文/psi-weighted-busi \\
        --masks ~/桌面/BUSI/U-KAN/Seg_UKAN/inputs/busi/masks/0
"""
import argparse
import glob
import os
import shutil

import cv2
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ap = argparse.ArgumentParser()
ap.add_argument('--base', required=True, help='HSAM 資料夾（含 H-SAM/、outputs/、psi_lists/）')
ap.add_argument('--repo', required=True)
ap.add_argument('--busi_repo', required=True)
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
          'screen_hsam.py', 'memtest_hsam.py', 'run_hsam_busi.sh', 'collect_hsam.py'):
    if os.path.exists(os.path.join(H, f)):
        shutil.copy(os.path.join(H, f), os.path.join(R, 'hsam_extension'))

ub = pd.read_csv(os.path.join(os.path.expanduser(args.busi_repo), 'results/percase/busi_UKAN_seed2981_best.csv'))
low = pd.Index(ub[(ub.split == 'val') & (ub.dice < 0.5)].case)
nine = pd.Index(pd.read_csv(os.path.join(os.path.expanduser(args.busi_repo), 'weight_lists/moved_busi_dice02.csv')).case)
any_m = next(iter(M.values()))
te, tr = any_m.index[any_m.split == 'test'], any_m.index[any_m.split == 'train']
G = [('all', te), ('other', te.difference(low)), ('low_dice', low), ('nine', nine), ('train', tr)]
nw = {}
for m, _, L in MODELS:
    f = os.path.join(B, 'psi_lists', f'wcases_busi_{L}.csv')
    nw[m] = len(pd.read_csv(f)) if os.path.exists(f) else 0

rows = []
for m, loss, L in MODELS:
    if m not in M:
        continue
    d = M[m]
    r = dict(model=f'busi_HSAM_{m}', loss=loss, psi_weight=L, n_weighted=nw[m], n_test=len(te))
    for g, idx in G:
        r[f'test_{g}' if g not in ('train', 'all') else ('test_all' if g == 'all' else 'train_all')] = d.dice.reindex(idx).mean()
    r['gap'] = r['train_all'] - r['test_all']
    r['test_all_dice256'] = d.dice256.reindex(te).mean()
    rows.append(r)
pd.DataFrame(rows).rename(columns={'test_low_dice': 'test_low_dice', 'test_nine': 'nine'}).round(4).to_csv(
    os.path.join(R, 'results/analysis/hsam_BUSI_models.csv'), index=False)

S = []
COMPS = [('ctrl_vs_base', 'base', 'ctrl')] + [(f'{m}_vs_ctrl', 'ctrl', m) for m, _, L in MODELS if L != 'none'] + \
        [(f'{m}_vs_base', 'base', m) for m, _, L in MODELS if L != 'none']
for comp, a, b in COMPS:
    if a not in M or b not in M:
        continue
    for g, idx in G:
        x, y = M[a].dice.reindex(idx), M[b].dice.reindex(idx)
        k = x.notna() & y.notna()
        x, y = x[k], y[k]
        dd = y - x
        try:
            p = wilcoxon(y, x).pvalue if (dd != 0).any() else 1.0
        except ValueError:
            p = np.nan
        S.append(dict(comparison=comp, control=f'busi_HSAM_{a}', model=f'busi_HSAM_{b}', group=g, n=int(k.sum()),
                      before_control=x.mean(), after_model=y.mean(), delta=dd.mean(),
                      n_up=int((dd > 1e-6).sum()), n_down=int((dd < -1e-6).sum()), wilcoxon_p=f'{p:.2g}'))
pd.DataFrame(S).round(4).to_csv(os.path.join(R, 'results/analysis/hsam_BUSI_summary.csv'), index=False)

W = pd.DataFrame(index=any_m.index)
W['split'] = any_m.split
W['low_dice'] = ['yes' if c in low else 'no' for c in W.index]
W['nine'] = ['yes' if c in nine else 'no' for c in W.index]
for m in M:
    W[m] = M[m].dice
W = W.sort_values(['split', 'low_dice', 'nine'], ascending=[True, False, False])
W.round(4).rename_axis('case').to_csv(os.path.join(R, 'results/analysis/hsam_BUSI_percase.csv'))

pd.set_option('display.width', 200)
print(pd.read_csv(os.path.join(R, 'results/analysis/hsam_BUSI_models.csv')).to_string(index=False))
print()
s = pd.DataFrame(S)
print(s[s.group.isin(['all', 'low_dice', 'nine'])][['comparison', 'group', 'n', 'before_control', 'after_model', 'delta', 'n_up', 'n_down', 'wilcoxon_p']].round(4).to_string(index=False))
print(f'\n→ {R}')