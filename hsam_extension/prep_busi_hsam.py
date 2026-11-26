"""
把 U-KAN 用的 BUSI（inputs/busi）轉成 H-SAM 的資料格式，切分與 U-KAN 完全相同
  來源：<busi>/images/<名稱>.png、<busi>/masks/0/<名稱>_mask.png（psi-weighted-busi 的 prep_busi.py 產生，647 張）
  切分：影像名稱排序 → train_test_split(test_size=0.2, random_state=2981)，訓練 517 / 測試 130（同 U-KAN、psi-weighted-busi）
  影像：灰階、原始解析度、/255 → float32 [0, 1]（H-SAM 的 Synapse 資料同為 [0, 1]，訓練時由 RandomGenerator 縮放到 img_size）
  標註：> 127 → 1，其餘 0（uint8）
輸出（H-SAM 的 Synapse 格式）：
  <out>/train_npz/<名稱>.npz          image (H, W)、label (H, W)          訓練用（split = train）
  <out>/vol/<名稱>.npy.h5             image (1, H, W)、label (1, H, W)    預測用（訓練集 + 測試集，H-SAM test.py 的格式）
  <out>/lists/train.txt               訓練 517 張（train.py --list_dir 讀這個）
  <out>/lists/train_vol.txt、test_vol.txt
用法：python prep_busi_hsam.py <Seg_UKAN/inputs/busi> <輸出資料夾>
"""
import glob
import os
import sys

import cv2
import h5py
import numpy as np
from sklearn.model_selection import train_test_split

if len(sys.argv) != 3:
    sys.exit(__doc__)
SRC, OUT = sys.argv[1], sys.argv[2]
ids = sorted(os.path.splitext(os.path.basename(p))[0] for p in glob.glob(os.path.join(SRC, 'images', '*.png')))
if len(ids) != 647:
    sys.exit(f'預期 647 張，找到 {len(ids)} 張：請確認第一個參數是 inputs/busi')
tr, te = train_test_split(ids, test_size=0.2, random_state=2981)          # 與 U-KAN train.py 相同
for d in ('train_npz', 'vol', 'lists'):
    os.makedirs(os.path.join(OUT, d), exist_ok=True)

for k, i in enumerate(ids):
    img = cv2.imread(os.path.join(SRC, 'images', i + '.png'), cv2.IMREAD_GRAYSCALE)
    gt = cv2.imread(os.path.join(SRC, 'masks', '0', i + '_mask.png'), cv2.IMREAD_GRAYSCALE)
    if img is None or gt is None or img.shape != gt.shape:
        sys.exit(f'✗ {i}：影像或遮罩讀取失敗或大小不同')
    x = (img.astype(np.float32) / 255.0)
    y = (gt > 127).astype(np.uint8)
    if i in tr:
        np.savez(os.path.join(OUT, 'train_npz', i + '.npz'), image=x, label=y)
    with h5py.File(os.path.join(OUT, 'vol', i + '.npy.h5'), 'w') as f:
        f.create_dataset('image', data=x[None])
        f.create_dataset('label', data=y[None])
    if (k + 1) % 100 == 0:
        print(f'  {k + 1}/{len(ids)}', flush=True)

for name, lst in (('train.txt', tr), ('train_vol.txt', tr), ('test_vol.txt', te)):
    with open(os.path.join(OUT, 'lists', name), 'w') as f:
        f.writelines(c + '\n' for c in lst)
print(f'訓練 {len(tr)} 張、測試 {len(te)} 張 → {OUT}')
