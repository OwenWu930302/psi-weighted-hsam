#!/bin/bash
# H-SAM（CVPR 2024）在 BUSI 上的 Ψ 加權實驗，流程同 psi-weighted-busi / psi-weighted-nnunet
#   1. baseline（官方 loss）→ 預測訓練集 + 測試集，計算 Ψ
#   2. 用 baseline 的訓練集預測篩選 Ψ，產生 Ψ1、Ψ6、Ψ8 的加權清單（前 30%；整數值 > 0）
#   3. 對照組 ctrl（逐張 loss、權重全為 1）
#   4. 專家 psi1w3n、psi6w3n、psi8w3n（清單影像 loss × 3、÷ n）
#   5. 每個模型預測訓練集 + 測試集（最後一個 epoch）
# 訓練設定：官方 README 的單一類別（PROMISE12）指令；--dataset Synapse 只是程式內的名稱（官方 train.py 只接受這個）
#   batch 16 以梯度累積分 2 份計算（--accum 2，RTX 4080 記憶體不足）
# 可中斷重跑：已有 epoch_249.pth 的模型略過訓練；中斷的模型會刪掉該模型資料夾後從頭訓練
# 用法（在 H-SAM 資料夾）：nohup bash run_hsam_busi.sh > run_hsam_busi.log 2>&1 &
# 進度：grep -a "=====" run_hsam_busi.log | tail -n 3
cd "$(dirname "$0")"
BASE=${BASE:-$(cd .. && pwd)}
DATA=${DATA:-$BASE/data/busi}
OUT=${OUT:-$BASE/outputs}
PSI=${PSI:-$BASE/psi_lists}
SAMCK=${SAMCK:-checkpoints/sam_vit_b_01ec64.pth}
PSI_LIB=${PSI_LIB:-$HOME/桌面/論文/psi-weighted-nnunet/psi/run_all_psi.py}
EPOCHS=${EPOCHS:-250}
BATCH=${BATCH:-8}                      # 每張卡 8，官方預設 n_gpu 2 → 實際 batch 16（同官方指令）
ACCUM=${ACCUM:-2}                      # 梯度累積：batch 16 分成 2 份 forward（RTX 4080 放不下 16），見 patch_hsam_accum.py
DEVICE=${DEVICE:-cuda}
EXTRA=${EXTRA:-}
FEATS="psi1_convex psi6_cc psi8_rotation"
LAST=$((EPOCHS - 1))
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
step() { echo; echo "===== $*　$(date '+%m/%d %H:%M')"; }
mkdir -p "$OUT" "$PSI"

for f in train.py trainer.py predict_hsam.py screen_hsam.py "$SAMCK" "$PSI_LIB" "$DATA/lists/train.txt" "$DATA/lists/test_vol.txt"; do
  [ -e "$f" ] || { echo "✗ 找不到 $f"; exit 1; }
done
grep -q "Ψ加權" trainer.py || { echo "✗ trainer.py 尚未修改，請先執行 python patch_hsam_weighted.py"; exit 1; }
[ "$ACCUM" -le 1 ] || grep -q "梯度累積" trainer.py || { echo "✗ trainer.py 尚未加入梯度累積，請先執行 python patch_hsam_accum.py"; exit 1; }

ckpt() { ls "$OUT/$1"/*/epoch_$LAST.pth 2>/dev/null | head -n 1; }

train() {   # train <名稱> [額外參數]
  local n=$1; shift
  step "訓練 $n $*"
  if [ -n "$(ckpt $n)" ]; then echo "已完成，略過"; return 0; fi
  rm -rf "$OUT/$n"
  python train.py --root_path "$DATA/train_npz" --output "$OUT/$n" --split train --batch_size $BATCH --base_lr 0.0026 \
    --img_size 512 --warmup --dice_param 0.9 --AdamW --max_epochs $EPOCHS --stop_epoch $EPOCHS --num_classes 1 \
    --dataset Synapse --list_dir "$DATA/lists" --ckpt "$SAMCK" --accum $ACCUM $EXTRA "$@" > "$OUT/${n}_train.log" 2>&1 || { echo "✗ 訓練失敗，見 $OUT/${n}_train.log"; tail -n 5 "$OUT/${n}_train.log"; exit 1; }
  grep -a "iteration" "$OUT/${n}_train.log" | tail -n 1
}

predict() {   # predict <名稱> [--psi]
  local n=$1; shift
  step "預測 $n（訓練集 + 測試集，epoch $LAST）"
  if [ -f "$OUT/$n/percase_last.csv" ]; then echo "已完成，略過"; return 0; fi
  python predict_hsam.py --lora_ckpt "$(ckpt $n)" --data "$DATA" --out "$OUT/$n" --ckpt "$SAMCK" --device $DEVICE \
    --psi_lib "$PSI_LIB" "$@" || { echo "✗ 預測失敗"; exit 1; }
}

train busi_HSAM_base
predict busi_HSAM_base --psi

step "篩選 Ψ，產生加權清單"
python screen_hsam.py "$OUT/busi_HSAM_base/percase_last.csv" "$PSI" $(echo $FEATS | tr ' ' ',') || exit 1

train busi_HSAM_ctrl --per_case_loss
predict busi_HSAM_ctrl

for c in $FEATS; do
  L="$PSI/wcases_busi_$c.csv"
  n=busi_HSAM_${c%%_*}w3n
  if [ ! -f "$L" ]; then step "$n：沒有清單（Ψ 恆定或清單過大），略過"; continue; fi
  train $n --wcases "$L" --wfactor 3
  predict $n
done

step "結果（最後一個 epoch，逐張 Dice 平均）"
python - "$OUT" <<'EOF'
import glob, os, sys
import pandas as pd
for f in sorted(glob.glob(os.path.join(sys.argv[1], 'busi_HSAM_*', 'percase_last.csv'))):
    d = pd.read_csv(f)
    s = '　'.join(f'{sp} {g.dice.mean():.4f}（{len(g)}）' for sp, g in d.groupby('split'))
    print(f'{os.path.basename(os.path.dirname(f)):<22}{s}')
EOF
step "全部完成"
