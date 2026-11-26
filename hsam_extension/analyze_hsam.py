"""
BUSI（H-SAM）：對照組（逐張 loss、權重 1）→ 專家（Ψ ×3），作法同 psi-weighted-busi 的 analyze_busi.py
  --psi_from：計算 Ψ、產生加權清單、定義低分組的 baseline（busi_HSAM_base，官方 loss）
  --control ：對照組（busi_HSAM_ctrl）；與 --psi_from 是不同模型，避免「用對照組自己的低分挑影像，重訓後自然回升」
  全部用最後一個 epoch（last）

訓練集：被加權的影像本身有沒有變好
  weighted_cases_BUSI.csv / _all.csv / _summary.csv / _compare.csv
測試集（130 張）：
  prominent：Ψ 超過訓練清單的門檻（只用 baseline 的預測，不看標註）
  low_dice ：baseline 測試集 Dice < 0.5（用標註，僅供分析）
  hit = prominent 且 low_dice；false_pick = prominent 但不是 low_dice；coverage = hit / low_dice；precision = hit / prominent
  test_BUSI_selection.csv、test_BUSI_summary.csv（all / prominent / low_dice / false_pick）、test_BUSI_percase.csv
  test_BUSI_dice_hist.png：baseline 測試集 Dice 分布
每個模型的分數與成對比較：
  models_BUSI.csv          每個模型一列：測試集、其他、low_dice（H-SAM base 定義）、U-KAN 的低分 15 張、那 9 張、訓練集、差距、256×256 的測試集
  models_BUSI_compare.csv  ctrl vs base、各專家 vs ctrl、各專家 vs base（逐張 Wilcoxon）
低分影像（像 psi-discussion 的 low_dice）：
  low_dice_BUSI_wide.csv    每張低分影像一列、每個 H-SAM 模型一欄，並附 U-KAN seed2981 的 Dice、是否在 U-KAN 的低分 15 張 / 那 9 張
  low_dice_BUSI_summary.csv 每個模型在 H-SAM 低分組、與 U-KAN 共同的低分、只有 H-SAM 低分 三組的平均
檔案名稱、欄位、編碼（utf-8-sig）、小數位數皆比照 psi-weighted-ukan 的 results/analysis
用法（倉庫根目錄）：python hsam_extension/analyze_hsam.py --ukan_repo ../psi-weighted-ukan
"""
import argparse
import os

import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, wilcoxon

matplotlib.use('Agg')
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument('--repo', default='.')
ap.add_argument('--ukan_repo', default='../psi-weighted-ukan', help='psi-weighted-ukan（原 psi-weighted-busi）')
ap.add_argument('--psi_from', default='busi_HSAM_base')
ap.add_argument('--control', default='busi_HSAM_ctrl')
ap.add_argument('--low_t', default=0.5, type=float)
args = ap.parse_args()
R = args.repo
OUT = os.path.join(R, 'results', 'analysis')
os.makedirs(OUT, exist_ok=True)
MODELS = ['base', 'ctrl', 'psi1w3n', 'psi6w3n', 'psi8w3n']


def load(name):
    return pd.read_csv(os.path.join(R, 'results', 'percase', f'{name}_last.csv')).set_index('case')


def wil(x):
    x = np.asarray(x, float)
    return float(wilcoxon(x).pvalue) if len(x) > 1 and np.any(x != 0) else np.nan


def change(g):
    return dict(n=len(g), before_control=g.control.mean(), after_expert=g.expert.mean(), delta=g.delta.mean(),
                n_up=int((g.delta > 0).sum()), n_down=int((g.delta < 0).sum()), wilcoxon_p=wil(g.delta))


PSI, CTL = load(args.psi_from), load(args.control)
sel = pd.read_csv(os.path.join(R, 'weight_lists', 'proxy_selection_busi.csv'))
sel = sel[sel.list_file.notna() & (sel.list_file.astype(str) != '')]
EXPERTS = []
for _, r in sel.iterrows():
    wl = pd.read_csv(os.path.join(R, 'weight_lists', os.path.basename(r.list_file)))
    col = r.proxy
    if r.rule == 'top30':
        t = wl[col].min(); f = (lambda t: lambda v: v >= t)(t); desc = f'>= {t:.6g}'
    elif r.rule == 'bottom30':
        t = wl[col].max(); f = (lambda t: lambda v: v <= t)(t); desc = f'<= {t:.6g}'
    else:
        f = lambda v: v > 0; desc = '> 0'
    EXPERTS.append(dict(psi=col, model=f'busi_HSAM_{col.split("_")[0]}w3n', wset=set(wl.case.astype(str)), sel=f, rule=desc))

TEST = PSI[PSI.split == 'test']
LOW = TEST.dice < args.low_t

# ---------- 測試集 Dice 分布 ----------
fig, ax = plt.subplots(figsize=(7, 3.6))
cnt, _, bars = ax.hist(TEST.dice, bins=np.arange(0, 1.0001, 0.05), color='#7a9cc6', edgecolor='white')
for b, c in zip(bars, cnt):
    if b.get_x() < args.low_t:
        b.set_color('#d9534f')
    if c:
        ax.text(b.get_x() + 0.025, c + 0.5, int(c), ha='center', fontsize=8)
ax.axvline(args.low_t, color='k', ls='--', lw=1)
ax.text(args.low_t - 0.01, cnt.max() * 0.9, f'low_dice: Dice < {args.low_t}\n{int(LOW.sum())} images '
        f'({int((TEST.dice < 0.1).sum())} below 0.1)', ha='right', fontsize=9)
ax.set_xlabel('Dice'); ax.set_ylabel('images'); ax.set_xlim(0, 1)
ax.set_title(f'BUSI test set ({len(TEST)} images): Dice of {args.psi_from} (last)', fontsize=10)
fig.tight_layout(); fig.savefig(os.path.join(OUT, 'test_BUSI_dice_hist.png'), dpi=150); plt.close(fig)

# ---------- Ψ 在測試集挑到幾張 ----------
SELROWS, PROM = [], {}
for e in EXPERTS:
    p = TEST[e['psi']].apply(lambda v: bool(not pd.isna(v) and e['sel'](v)))
    PROM[e['model']] = p
    hit, n_low, n_p = int((p & LOW).sum()), int(LOW.sum()), int(p.sum())
    SELROWS.append(dict(expert=e['model'], psi=e['psi'], threshold=e['rule'], n_test=len(TEST), n_prominent=n_p,
                        n_low_dice=n_low, n_hit=hit, n_false_pick=n_p - hit, n_missed=n_low - hit,
                        coverage=hit / max(n_low, 1), precision=hit / max(n_p, 1)))

TR, TRS, TRC, TE, SUM = [], [], [], [], []
for e in EXPERTS:
    ex = load(e['model'])
    cs = CTL[CTL.split == 'train'].index.intersection(ex.index)
    t = pd.DataFrame({'control': CTL.loc[cs, 'dice'], 'expert': ex.loc[cs, 'dice']})
    t['delta'] = t.expert - t.control
    t['weighted'] = [c in e['wset'] for c in t.index]
    t['psi_value'] = PSI.loc[t.index, e['psi']]
    t = t.sort_values(['weighted', 'psi_value'], ascending=False)
    for c, r in t.iterrows():
        TR.append(dict(checkpoint='last', expert=e['model'], case=c, weighted='yes' if r.weighted else 'no', psi_value=r.psi_value,
                       before_control=r.control, after_expert=r.expert, delta=r.delta))
    W, U = t[t.weighted], t[~t.weighted]
    TRS.append(dict(checkpoint='last', expert=e['model'], **change(W)))
    TRC.append(dict(checkpoint='last', expert=e['model'], n_weighted=len(W), delta_weighted=W.delta.mean(), n_unweighted=len(U),
                    delta_unweighted=U.delta.mean(), weighted_minus_unweighted=W.delta.mean() - U.delta.mean(),
                    mannwhitney_p=float(mannwhitneyu(W.delta, U.delta).pvalue)))
    cs = TEST.index.intersection(ex.index)
    g = pd.DataFrame({'control': CTL.loc[cs, 'dice'], 'expert': ex.loc[cs, 'dice']})
    g['delta'] = g.expert - g.control
    p, low = PROM[e['model']].loc[cs], LOW.loc[cs]
    for c, r in g.iterrows():
        TE.append(dict(checkpoint='last', expert=e['model'], case=c, psi_value=TEST.loc[c, e['psi']], prominent='yes' if p[c] else 'no',
                       low_dice='yes' if low[c] else 'no', baseline_dice=TEST.loc[c, 'dice'],
                       before_control=r.control, after_expert=r.expert, delta=r.delta))
    for name, m in [('all', p | ~p), ('prominent', p), ('low_dice', low), ('false_pick', p & ~low)]:
        SUM.append(dict(checkpoint='last', expert=e['model'], group=name, **change(g[m])))

TRW = [{k: v for k, v in r.items() if k != 'weighted'} for r in TR if r['weighted'] == 'yes']
def save(name, d, index=False):
    d.to_csv(os.path.join(OUT, f'{name}.csv'), index=index, encoding='utf-8-sig', float_format='%.4f')


for name, rows in [('weighted_cases_BUSI', TRW), ('weighted_cases_BUSI_all', TR), ('weighted_cases_BUSI_summary', TRS),
                   ('weighted_cases_BUSI_compare', TRC), ('test_BUSI_selection', SELROWS),
                   ('test_BUSI_summary', SUM), ('test_BUSI_percase', TE)]:
    save(name, pd.DataFrame(rows))

# ---------- 低分影像：每個模型的 Dice，和 U-KAN 的低分比較 ----------
ub = pd.read_csv(os.path.join(args.ukan_repo, 'results', 'percase', 'busi_UKAN_seed2981_best.csv')).set_index('case')
u_low = set(ub[(ub.split == 'val') & (ub.dice < 0.5)].index)
nine = set(pd.read_csv(os.path.join(args.ukan_repo, 'weight_lists', 'moved_busi_dice02.csv')).case)
M = {m: load(f'busi_HSAM_{m}').dice for m in MODELS}
cases = sorted(set(TEST.index[LOW]) | u_low, key=lambda c: (TEST.dice[c] >= args.low_t, M['base'][c]))
W = pd.DataFrame(index=pd.Index(cases, name='case'))
W['hsam_low'] = ['yes' if TEST.dice[c] < args.low_t else 'no' for c in cases]
W['ukan_low'] = ['yes' if c in u_low else 'no' for c in cases]
W['nine'] = ['yes' if c in nine else 'no' for c in cases]
W['ukan_seed2981'] = ub.dice.reindex(cases)
for m in MODELS:
    W[m] = M[m].reindex(cases)
save('low_dice_BUSI_wide', W, index=True)
S = []
for gname, msk in [('hsam_low', W.hsam_low == 'yes'), ('both_low', (W.hsam_low == 'yes') & (W.ukan_low == 'yes')),
                   ('hsam_only', (W.hsam_low == 'yes') & (W.ukan_low == 'no')),
                   ('ukan_only', (W.hsam_low == 'no') & (W.ukan_low == 'yes')), ('nine', W.nine == 'yes')]:
    r = dict(group=gname, n=int(msk.sum()), ukan_seed2981=W.ukan_seed2981[msk].mean())
    for m in MODELS:
        r[m] = W[m][msk].mean()
    S.append(r)
save('low_dice_BUSI_summary', pd.DataFrame(S))

# ---------- 每個模型的分數與成對比較 ----------
LOSS = dict(base=('official (batch Dice)', 'none'), ctrl=('per-case, w = 1', 'none'), psi1w3n=('per-case, x3 / n', 'psi1_convex'),
            psi6w3n=('per-case, x3 / n', 'psi6_cc'), psi8w3n=('per-case, x3 / n', 'psi8_rotation'))
te, tr = TEST.index, PSI.index[PSI.split == 'train']
low = te[LOW.reindex(te).values]
G = [('all', te), ('other', te.difference(low)), ('low_dice', low), ('ukan_low_dice', pd.Index(sorted(u_low))),
     ('nine', pd.Index(sorted(nine))), ('train', tr)]
D = {m: load(f'busi_HSAM_{m}') for m in MODELS}
MR = []
for m in MODELS:
    L = LOSS[m][1]
    wf = os.path.join(R, 'weight_lists', f'wcases_busi_{L}.csv')
    r = dict(checkpoint='last', model=f'busi_HSAM_{m}', loss=LOSS[m][0], psi_weight=L,
             n_weighted=len(pd.read_csv(wf)) if os.path.exists(wf) else 0)
    for g, idx in G:
        r['train_all' if g == 'train' else f'test_{g}'] = D[m].dice.reindex(idx).mean()
    r['gap'] = r['train_all'] - r['test_all']
    r['test_all_dice256'] = D[m].dice256.reindex(te).mean()
    MR.append(r)
save('models_BUSI', pd.DataFrame(MR))
CR = []
for comp, a, b in [('ctrl_vs_base', 'base', 'ctrl')] + [(f'{m}_vs_ctrl', 'ctrl', m) for m in MODELS[2:]] + \
                  [(f'{m}_vs_base', 'base', m) for m in MODELS[2:]]:
    for g, idx in G:
        x, y = D[a].dice.reindex(idx), D[b].dice.reindex(idx)
        dd = y - x
        CR.append(dict(checkpoint='last', comparison=comp, control=f'busi_HSAM_{a}', model=f'busi_HSAM_{b}', group=g,
                       n=len(idx), before_control=x.mean(), after_model=y.mean(), delta=dd.mean(),
                       n_up=int((dd > 0).sum()), n_down=int((dd < 0).sum()), wilcoxon_p=wil(dd)))
save('models_BUSI_compare', pd.DataFrame(CR))

pd.set_option('display.width', 200)
for title, rows in [('models (last)', MR), ('training set: weighted vs unweighted', TRC), ('test set: selection', SELROWS),
                    ('test set: control -> expert', SUM), ('low-Dice images', S)]:
    print(f'\n[{title}]')
    print(pd.DataFrame(rows).to_string(index=False, float_format='%.4f'))
print(f'\nlow_dice = {args.psi_from} test Dice < {args.low_t}；輸出：{OUT}')
