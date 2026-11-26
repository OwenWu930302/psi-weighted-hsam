#!/bin/bash
# H-SAM 的失敗診斷（教授的 failure_diagnostics.py，作法同 psi-discussion 的 run_diag.sh）＋ 低分影像疊合圖
#   base（Ψ 與加權清單都由它而來）：候選排名 recall@k + 失敗類型、Ψ 的 AUROC / 門檻、梯度對齊
#   ctrl、psi1w3n、psi6w3n、psi8w3n：只做候選排名 + 失敗類型
#   Ψ 門檻 = 加權清單的最小值（failure_diagnostics 用「>」，所以減 1e-7）；Ψ6 為 > 0
#   形狀特徵（只由遮罩算）：psi1_convex、psi6_cc
#   疊合圖：U-KAN 與 H-SAM 在 16 張低分影像上的預測（overlay_hsam_ukan.py）
# 結果複製到 psi-weighted-hsam：results/diagnostics/<模型>_last/、results/examples/、results/analysis/hard_cases_BUSI.csv
# 用法（H-SAM 資料夾、hsam 環境）：bash run_diag_hsam.sh > run_diag_hsam.log 2>&1
cd "$(dirname "$0")"
BASE=${BASE:-$(cd .. && pwd)}
REPO=${REPO:-$HOME/桌面/論文/psi-weighted-hsam}
CODE=${CODE:-$HOME/桌面/論文/psi-discussion/busi/code}
MLIST=${MLIST:-$HOME/桌面/論文/psi-weighted-ukan/weight_lists/moved_busi_dice02.csv}
BUSI=${BUSI:-$HOME/桌面/BUSI/U-KAN/Seg_UKAN/inputs/busi}
UKAN_OUT=${UKAN_OUT:-$HOME/桌面/BUSI/U-KAN/Seg_UKAN/outputs}
EXTRA=${EXTRA:-}                      # 傳給 diag_npz_hsam.py 的其他參數（測試用）
PSI=$BASE/psi_lists
OUT=$BASE/outputs/diag
FD=$CODE/failure_diagnostics.py
NC=$CODE/name_cases.py
for f in diag_npz_hsam.py overlay_hsam_ukan.py $FD $NC $MLIST $PSI/wcases_busi_psi1_convex.csv $PSI/wcases_busi_psi8_rotation.csv \
         $BASE/outputs/busi_HSAM_base/percase_last.csv $REPO/results/analysis/low_dice_BUSI_wide.csv; do
  [ -e "$f" ] || { echo "✗ 找不到 $f"; exit 1; }
done
step() { echo; echo "===== $1　$(date '+%m/%d %H:%M')"; }
tau() { python -c "import pandas as pd; print(repr(float(pd.read_csv('$PSI/wcases_busi_$1.csv', encoding='utf-8-sig')['$1'].min()) - 1e-7))"; }
TAUS="--tau psi1_convex=$(tau psi1_convex) --tau psi6_cc=0 --tau psi8_rotation=$(tau psi8_rotation)"
echo "Ψ 門檻：$TAUS"
mkdir -p $OUT

step "0. failure_diagnostics.py 自我測試"
python $FD --selftest | tail -n 1 | grep -q PASSED || { echo "自我測試失敗"; exit 1; }
echo "PASSED"

step "1. base：機率圖、Ψ、梯度"
python diag_npz_hsam.py --model base --base "$BASE" --features --grads --diag $FD $EXTRA || exit 1
python $FD --npz $OUT/busi_HSAM_base_last.npz --out-dir $OUT/base_last --delta 0.5 $TAUS \
  --shape-features psi1_convex psi6_cc > $OUT/base_last.txt || exit 1
python $NC $OUT/busi_HSAM_base_last.npz $OUT/base_last/per_case.csv $MLIST || exit 1

for m in ctrl psi1w3n psi6w3n psi8w3n; do
  step "2. $m：機率圖"
  python diag_npz_hsam.py --model $m --base "$BASE" $EXTRA || exit 1
  python $FD --npz $OUT/busi_HSAM_${m}_last.npz --out-dir $OUT/${m}_last --delta 0.5 > $OUT/${m}_last.txt || exit 1
  python $NC $OUT/busi_HSAM_${m}_last.npz $OUT/${m}_last/per_case.csv $MLIST || exit 1
done

step "3. 低分影像疊合圖（U-KAN 與 H-SAM）"
python overlay_hsam_ukan.py --repo "$REPO" --busi "$BUSI" --ukan_out "$UKAN_OUT" --hsam_out "$BASE/outputs" || exit 1

step "4. 複製到 $REPO"
mkdir -p $REPO/results/diagnostics $REPO/hsam_extension
for m in base ctrl psi1w3n psi6w3n psi8w3n; do
  mkdir -p $REPO/results/diagnostics/${m}_last
  cp $OUT/${m}_last/per_case.csv $OUT/${m}_last/summary.json $REPO/results/diagnostics/${m}_last/
done
python $FD --selftest > $REPO/results/diagnostics/selftest.txt
cp diag_npz_hsam.py overlay_hsam_ukan.py run_diag_hsam.sh $REPO/hsam_extension/
(cd $REPO && git add -A && git status --short)

step "全部完成"
for m in base ctrl psi1w3n psi6w3n psi8w3n; do
  echo "--- $m"; python -c "import json;s=json.load(open('$OUT/${m}_last/summary.json'));print('失敗',s['n_failures'],'/',s['n_cases'],' 類型',s['failure_types'],' recall@k（失敗）',s['recall_at_k_on_failures'])"
done
python -c "import json;s=json.load(open('$OUT/base_last/summary.json'));[print(k, s.get(k)) for k in ('alignment_on_failures','control_alignment_on_failures','frac_failures_with_A_positive')]"
