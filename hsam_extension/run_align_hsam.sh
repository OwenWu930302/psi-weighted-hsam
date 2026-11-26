#!/bin/bash
# 梯度對齊（failure_diagnostics.py 第 2 部分）補做 Ψ1、Ψ6 的加權清單（run_diag_hsam.sh 只用了 Ψ8 的清單）
#   每個 Ψ：用該 Ψ 的加權清單當 grads_flag（權重 3），其餘訓練影像當對照，其他設定與 run_diag_hsam.sh 相同
#   結果：results/diagnostics/base_last_psi1_convex/、base_last_psi6_cc/（Ψ8 清單為 base_last_psi8_rotation/，由 run_diag_hsam.sh 產生）
# 用法（H-SAM 資料夾、hsam 環境）：bash run_align_hsam.sh > run_align_hsam.log 2>&1
cd "$(dirname "$0")"
BASE=${BASE:-$(cd .. && pwd)}
REPO=${REPO:-$HOME/桌面/論文/psi-weighted-hsam}
CODE=${CODE:-$HOME/桌面/論文/psi-discussion/busi/code}
MLIST=${MLIST:-$HOME/桌面/論文/psi-weighted-ukan/weight_lists/moved_busi_dice02.csv}
PSI=$BASE/psi_lists
OUT=$BASE/outputs/diag
FD=$CODE/failure_diagnostics.py
NC=$CODE/name_cases.py
tau() { python -c "import pandas as pd; print(repr(float(pd.read_csv('$PSI/wcases_busi_$1.csv', encoding='utf-8-sig')['$1'].min()) - 1e-7))"; }
TAUS="--tau psi1_convex=$(tau psi1_convex) --tau psi6_cc=0 --tau psi8_rotation=$(tau psi8_rotation)"
step() { echo; echo "===== $1　$(date '+%m/%d %H:%M')"; }
for c in psi1_convex psi6_cc; do
  step "base：梯度對齊，加權清單 = $c"
  python diag_npz_hsam.py --model base --base "$BASE" --features --grads --flag_list $PSI/wcases_busi_$c.csv --diag $FD \
    --out $OUT/busi_HSAM_base_last_$c.npz || exit 1
  python $FD --npz $OUT/busi_HSAM_base_last_$c.npz --out-dir $OUT/base_last_$c --delta 0.5 $TAUS \
    --shape-features psi1_convex psi6_cc > $OUT/base_last_$c.txt || exit 1
  python $NC $OUT/busi_HSAM_base_last_$c.npz $OUT/base_last_$c/per_case.csv $MLIST || exit 1
  mkdir -p $REPO/results/diagnostics/base_last_$c
  cp $OUT/base_last_$c/per_case.csv $OUT/base_last_$c/summary.json $REPO/results/diagnostics/base_last_$c/
done
cp run_align_hsam.sh $REPO/hsam_extension/
(cd $REPO && git add -A && git status --short)
step "全部完成"
for d in base_last base_last_psi1_convex base_last_psi6_cc; do
  python -c "import json;s=json.load(open('$OUT/$d/summary.json'));a=s['alignment_on_failures'];c=s['control_alignment_on_failures'];print('$d'.replace('base_last', 'base_last_psi8_rotation') if '$d' == 'base_last' else '$d', '失敗 rho_G', round(a['rho_G'],3), '對照 rho_G', round(c['rho_G'],3), 'A>0 比例', round(s['frac_failures_with_A_positive'],3))"
done