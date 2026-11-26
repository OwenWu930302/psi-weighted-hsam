"""
U-KAN 與 H-SAM 對照組的測試集 Dice 對照（130 張，都在 256×256 計算）
  Ψ 實驗是「對照組 → 專家」的比較，所以這裡比兩邊的對照組：
    U-KAN 對照組 = base2（官方設定，last；psi-weighted-ukan 的 control）
    H-SAM 對照組 = ctrl（逐張 loss、權重 1，last；本倉庫的 control）
  CSV 另附 U-KAN shift（+ 平移增強）與 H-SAM base（官方 loss，Ψ 由它產生）供參考
輸出：results/analysis/compare_UKAN_HSAM_BUSI.csv（每張一列）、compare_UKAN_HSAM_BUSI.png（左：Dice 分布；右：逐張散佈圖）
用法（倉庫根目錄）：python hsam_extension/compare_ukan_hsam.py --ukan_repo ../psi-weighted-ukan --disc ../psi-discussion
"""
import argparse
import os

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use('Agg')
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument('--repo', default='.')
ap.add_argument('--ukan_repo', default='../psi-weighted-ukan')
ap.add_argument('--disc', default='../psi-discussion')
args = ap.parse_args()
OUT = os.path.join(args.repo, 'results', 'analysis')
rd = lambda f: pd.read_csv(f, encoding='utf-8-sig').set_index('case')
ub = rd(os.path.join(args.ukan_repo, 'results/percase/busi_UKAN_base2_last.csv'))
us = rd(os.path.join(args.disc, 'busi/results/percase/busi_UKAN_shift_last.csv'))
hs = rd(os.path.join(args.repo, 'results/percase/busi_HSAM_ctrl_last.csv'))
hb = rd(os.path.join(args.repo, 'results/percase/busi_HSAM_base_last.csv'))
te = hs.index[hs.split == 'test']
assert set(te) == set(ub.index[ub.split == 'val']), '測試集不同'
D = pd.DataFrame({'ukan_base2': ub.dice.reindex(te), 'hsam_ctrl': hs.dice256.reindex(te),
                  'ukan_shift': us.dice.reindex(te), 'hsam_base': hb.dice256.reindex(te)})
D['hsam_ctrl_minus_ukan_base2'] = D.hsam_ctrl - D.ukan_base2
D = D.sort_values('hsam_ctrl_minus_ukan_base2').rename_axis('case')
D.to_csv(os.path.join(OUT, 'compare_UKAN_HSAM_BUSI.csv'), encoding='utf-8-sig', float_format='%.4f')

C_U, C_H = '#257de0', '#f47230'          # 類別色第 1、2 格（已通過色盲檢查）
INK, INK2, GRID = '#191919', '#666666', '#e7e7e7'
plt.rcParams.update({'font.size': 9, 'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2, 'ytick.color': INK2})
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={'width_ratios': [1.35, 1]})
bins = np.arange(0, 1.0001, 0.1)
cu, _ = np.histogram(D.ukan_base2, bins)
ch, _ = np.histogram(D.hsam_ctrl, bins)
x = bins[:-1] + 0.05
w = 0.042
for off, cnt, col, lab, m in [(-w / 2 - 0.001, cu, C_U, 'U-KAN control (base2)', D.ukan_base2.mean()),
                              (w / 2 + 0.001, ch, C_H, 'H-SAM control (ctrl)', D.hsam_ctrl.mean())]:
    a1.bar(x + off, cnt, width=w, color=col, label=f'{lab}  (mean {m:.3f})', zorder=3)
    for xi, c in zip(x + off, cnt):
        if c:
            a1.text(xi, c + 0.8, int(c), ha='center', va='bottom', fontsize=7.5, color=INK2)
a1.axvline(0.5, color=INK2, lw=1, ls='--', zorder=2)
a1.set_xticks(bins)
a1.set_xlim(0, 1)
a1.set_xlabel('Dice (256×256)')
a1.set_ylabel('test images')
a1.grid(axis='y', color=GRID, lw=0.8, zorder=0)
a1.legend(frameon=False, loc='upper left')
nu, nh = int((D.ukan_base2 < 0.5).sum()), int((D.hsam_ctrl < 0.5).sum())
a1.set_title(f'BUSI test set (130), control models: Dice < 0.5  U-KAN {nu}, H-SAM {nh}', fontsize=10, color=INK, loc='left')
for s in ('top', 'right'):
    a1.spines[s].set_visible(False)

hard = ['benign (247)', 'benign (116)', 'malignant (106)', 'malignant (105)']
rest = D.index.difference(hard)
a2.plot([0, 1], [0, 1], color=INK2, lw=1, ls='--', zorder=1)
a2.axhline(0.5, color=GRID, lw=0.8, zorder=0)
a2.axvline(0.5, color=GRID, lw=0.8, zorder=0)
a2.scatter(D.ukan_base2[rest], D.hsam_ctrl[rest], s=22, color=C_H, edgecolor='white', linewidth=1, zorder=3, label='test image')
a2.scatter(D.ukan_base2[hard], D.hsam_ctrl[hard], s=40, marker='X', color=INK, edgecolor='white', linewidth=1, zorder=4,
           label='all 8 models < 0.1 (4 images)')
up, dn = int((D.hsam_ctrl_minus_ukan_base2 > 0).sum()), int((D.hsam_ctrl_minus_ukan_base2 < 0).sum())
a2.set_xlim(-0.02, 1.02)
a2.set_ylim(-0.02, 1.02)
a2.set_aspect('equal')
a2.set_xlabel('U-KAN control (base2) Dice')
a2.set_ylabel('H-SAM control (ctrl) Dice')
a2.set_title(f'Per image: H-SAM better {up}, U-KAN better {dn}', fontsize=10, color=INK, loc='left')
a2.legend(frameon=False, loc='lower right', fontsize=8)
for s in ('top', 'right'):
    a2.spines[s].set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'compare_UKAN_HSAM_BUSI.png'), dpi=150, facecolor='white')
print(D[['ukan_base2', 'hsam_ctrl', 'ukan_shift', 'hsam_base']].describe().round(4))
print(f'Dice < 0.5：U-KAN base2 {nu}、H-SAM ctrl {nh}；H-SAM ctrl 較好 {up} 張、較差 {dn} 張')
