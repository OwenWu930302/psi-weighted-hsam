"""
為教授的 failure_diagnostics.py 準備 H-SAM 的 .npz（作法同 psi-discussion 的 make_diag_npz.py）：BUSI 測試集 130 張
  probs (N,256,256)：H-SAM 推論（同 predict_hsam.py：縮到 512、stage 3 = 兩個 decoder 平均、softmax）取腫瘤類機率，
                     再縮到 256×256（雙線性），與 U-KAN 的診斷相同大小
  gts：標註縮到 256×256（最近鄰）；preds = probs ≥ 0.5（兩類 softmax 的 argmax）
  cases：影像名稱（failure_diagnostics 不讀；name_cases.py 用來把編號換回名稱）
  --features：features (N,3) = psi1_convex、psi6_cc、psi8_rotation，取自 baseline 的 percase_last.csv（不看標註）
  --grads：每張影像的梯度（failure_diagnostics.extract_grads，batch 1）
      loss = 官方訓練 loss：(1 − w)·L(decoder1, 128×128 標註) + w·L(decoder2, 512×512 標註)，L = 0.1·CE + 0.9·Dice，
             w = 0.6^(0.99^249)（最後一個 epoch 的權重）；推論模式（mode='test'，不加訓練時的隨機雜訊）
      參數 = 兩個 decoder 的輸出層：產生遮罩的 hypernetwork MLP 最後一層（對應 U-KAN 的輸出層 final）
      grads_eval = 測試集 130 張；grads_flag = Ψ8 加權清單（weights_flag = 3）；grads_ctrl = 其餘訓練影像（對照）
用法（H-SAM 資料夾，hsam 環境）：
  python diag_npz_hsam.py --model base --features --grads
  python diag_npz_hsam.py --model ctrl
"""
import argparse
import glob
import importlib.util
import os
import re
import sys

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat
from scipy.ndimage import zoom

from sam_lora_image_encoder import LoRA_Sam
from segment_anything import sam_model_registry
from utils import DiceLoss

ap = argparse.ArgumentParser()
ap.add_argument('--model', required=True, help='base、ctrl、psi1w3n、psi6w3n、psi8w3n')
ap.add_argument('--base', default='..', help='HSAM 資料夾（含 outputs/、data/、psi_lists/）')
ap.add_argument('--data', default=None, help='預設 <base>/data/busi')
ap.add_argument('--ckpt', default='checkpoints/sam_vit_b_01ec64.pth')
ap.add_argument('--epoch', type=int, default=249)
ap.add_argument('--features', action='store_true')
ap.add_argument('--grads', action='store_true')
ap.add_argument('--flag_list', default=None, help='預設 <base>/psi_lists/wcases_busi_psi8_rotation.csv')
ap.add_argument('--wfactor', type=float, default=3.0)
ap.add_argument('--diag', default=os.path.expanduser('~/桌面/論文/psi-discussion/busi/code/failure_diagnostics.py'))
ap.add_argument('--out', default=None)
ap.add_argument('--device', default='cuda')
args = ap.parse_args()
FEATS = ['psi1_convex', 'psi6_cc', 'psi8_rotation']
S, LOWS = 512, 128
B = args.base
DATA = args.data or os.path.join(B, 'data', 'busi')
O = os.path.join(B, 'outputs', f'busi_HSAM_{args.model}')
ck = sorted(glob.glob(os.path.join(O, '*', f'epoch_{args.epoch}.pth')))
ck or sys.exit(f'找不到 {O}/*/epoch_{args.epoch}.pth')

sam, _ = sam_model_registry['vit_b'](image_size=S, num_classes=1, checkpoint=args.ckpt, pixel_mean=[0, 0, 0], pixel_std=[1, 1, 1])
net = LoRA_Sam(sam, 5).to(args.device)
net.load_lora_parameters(ck[0])
net.eval()
print(f'模型：{ck[0]}')


class Wrap(nn.Module):   # extract_grads 用 model(x) 呼叫；推論模式，不給 prompt
    def __init__(self, n):
        super().__init__()
        self.n = n

    def forward(self, x):
        return self.n(x, False, S, gt=None, mode='test')


model = Wrap(net)
lists = lambda sp: [l.strip('\n') for l in open(os.path.join(DATA, 'lists', f'{sp}_vol.txt')) if l.strip()]
te, tr = lists('test'), lists('train')


def load(c):
    with h5py.File(os.path.join(DATA, 'vol', c + '.npy.h5'), 'r') as f:
        img, lab = f['image'][0], f['label'][0]
    x, y = img.shape
    xi = torch.from_numpy(zoom(img, (S / x, S / y), order=3)).float()[None, None]
    xi = repeat(xi, 'b c h w -> b (repeat c) h w', repeat=3)
    l512 = zoom(lab, (S / x, S / y), order=0)                                     # 同 RandomGenerator
    return xi, torch.from_numpy(l512.astype(np.int64))[None], lab


out = {'cases': np.array(te)}
probs, gts = [], []
with torch.no_grad():
    for c in te:
        xi, _, lab = load(c)
        o1, o2, _, _ = model(xi.to(args.device))
        p = torch.softmax((o1['masks'] + o2['masks']) / 2, dim=1)[0, 1]             # stage 3，同 predict_hsam.py
        p = F.interpolate(p[None, None], size=(256, 256), mode='bilinear', align_corners=False)[0, 0]
        probs.append(p.cpu().numpy().astype(np.float32))
        gts.append(zoom(lab, (256 / lab.shape[0], 256 / lab.shape[1]), order=0) > 0)
out['probs'], out['gts'] = np.stack(probs), np.stack(gts)
out['preds'] = out['probs'] >= 0.5
dd = [2 * (p & g).sum() / max(p.sum() + g.sum(), 1) for p, g in zip(out['preds'], out['gts'])]
print(f'測試集 {len(te)} 張，256×256 平均 Dice {np.mean(dd):.4f}，Dice < 0.5 有 {int((np.array(dd) < 0.5).sum())} 張')

if args.features:
    f = pd.read_csv(os.path.join(B, 'outputs', 'busi_HSAM_base', 'percase_last.csv')).set_index('case')
    out['features'] = f.loc[te, FEATS].to_numpy(float)
    out['feature_names'] = np.array(FEATS)
    print(f'Ψ 特徵：{FEATS}（busi_HSAM_base/percase_last.csv）')

if args.grads:
    spec = importlib.util.spec_from_file_location('fd', args.diag)
    fd = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fd)
    fl_path = args.flag_list or os.path.join(B, 'psi_lists', 'wcases_busi_psi8_rotation.csv')
    flag = set(pd.read_csv(fl_path, encoding='utf-8-sig').iloc[:, 0].astype(str))
    fl, ct = [c for c in tr if c in flag], [c for c in tr if c not in flag]
    # 兩個 decoder 的 hypernetwork MLP 最後一層
    pat = re.compile(r'(.*output_hypernetworks_mlps2?\.\d+\.layers\.)(\d+)\.')
    last = {}
    for n, p in model.named_parameters():
        m = pat.match(n)
        if p.requires_grad and m:
            last[m.group(1)] = max(last.get(m.group(1), -1), int(m.group(2)))
    pick = lambda n: bool(pat.match(n)) and int(pat.match(n).group(2)) == last[pat.match(n).group(1)]
    names = [n for n, p in model.named_parameters() if p.requires_grad and pick(n)]
    print(f'梯度參數：{len(names)} 個張量，{sum(p.numel() for n, p in model.named_parameters() if n in names)} 維')
    ce, dl = nn.CrossEntropyLoss(), DiceLoss(2)
    W2 = 0.6 ** (0.990 ** args.epoch)                                             # 同 trainer.py 的 weight

    def lossf(o, y):
        o1, o2, _, _ = o
        ylow = F.interpolate(y[:, None].float(), size=o1['low_res_logits'].shape[-2:], mode='nearest')[:, 0].long()
        L = lambda lg, t: 0.1 * ce(lg, t) + 0.9 * dl(lg, t, softmax=True)          # 同 calc_loss（dice_param 0.9）
        return (1 - W2) * L(o1['low_res_logits'], ylow) + W2 * L(o2['low_res_logits'], y)

    def loader(lst):
        for c in lst:
            xi, y, _ = load(c)
            yield xi, y

    grads = lambda lst: fd.extract_grads(model, lossf, loader(lst), device=args.device, param_filter=pick)
    out['grads_eval'] = grads(te)
    out['grads_flag'] = grads(fl)
    out['weights_flag'] = np.full(len(fl), args.wfactor)
    out['grads_ctrl'] = grads(ct)
    print(f'梯度：測試 {len(te)}、加權清單 {len(fl)}、對照 {len(ct)}')

path = args.out or os.path.join(B, 'outputs', 'diag', f'busi_HSAM_{args.model}_last.npz')
os.makedirs(os.path.dirname(path), exist_ok=True)
np.savez_compressed(path, **out)
print(f'→ {path}')
