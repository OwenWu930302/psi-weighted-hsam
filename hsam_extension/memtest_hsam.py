"""
訓練前的顯示卡記憶體與速度測試：用和正式訓練相同的模型與 batch（官方 PROMISE12 設定：vit_b、512、batch 8 × n_gpu 2 = 16）
跑幾次 forward + backward，印出記憶體峰值與每次迭代秒數，並估計一個模型（250 epoch）要多久
不讀資料、不存任何檔案；隨機影像與隨機圓形標註
用法（在 H-SAM 資料夾）：python memtest_hsam.py            或指定 batch：python memtest_hsam.py --batch 16
"""
import argparse
import time

import torch
import torch.nn.functional as F

from sam_lora_image_encoder import LoRA_Sam
from segment_anything import sam_model_registry

ap = argparse.ArgumentParser()
ap.add_argument('--batch', type=int, default=16)
ap.add_argument('--img_size', type=int, default=512)
ap.add_argument('--iters', type=int, default=6)
ap.add_argument('--n_train', type=int, default=517)
ap.add_argument('--epochs', type=int, default=250)
ap.add_argument('--ckpt', default='checkpoints/sam_vit_b_01ec64.pth')
args = ap.parse_args()

sam, emb = sam_model_registry['vit_b'](image_size=args.img_size, num_classes=1, checkpoint=args.ckpt,
                                       pixel_mean=[0, 0, 0], pixel_std=[1, 1, 1])
net = LoRA_Sam(sam, 5).cuda().train()
opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, net.parameters()), lr=1e-4, betas=(0.9, 0.999), weight_decay=0.1)
low = emb * 4
yy, xx = torch.meshgrid(torch.arange(args.img_size), torch.arange(args.img_size), indexing='ij')
lab = ((yy - args.img_size / 2) ** 2 + (xx - args.img_size / 2) ** 2 < (args.img_size / 5) ** 2).long()
lab = lab[None].repeat(args.batch, 1, 1).cuda()
lab_low = F.interpolate(lab[:, None].float(), size=(low, low), mode='nearest')[:, 0].long()
x = torch.rand(args.batch, 3, args.img_size, args.img_size).cuda()
torch.cuda.reset_peak_memory_stats()
ts = []
for i in range(args.iters):
    torch.cuda.synchronize()
    t = time.time()
    o1, o2, _, _ = net(x, False, args.img_size, gt=lab_low)
    loss = F.cross_entropy(o1['low_res_logits'], lab_low) + F.cross_entropy(o2['low_res_logits'], lab)
    opt.zero_grad()
    loss.backward()
    opt.step()
    torch.cuda.synchronize()
    ts.append(time.time() - t)
    print(f'  iter {i + 1}: {ts[-1]:.2f} s', flush=True)
peak = torch.cuda.max_memory_allocated() / 2 ** 30
total = torch.cuda.get_device_properties(0).total_memory / 2 ** 30
sec = sum(ts[2:]) / max(len(ts) - 2, 1)                                # 前兩次含初始化，不計
iters = -(-args.n_train // args.batch) * args.epochs
print(f'batch {args.batch}、{args.img_size}×{args.img_size}：記憶體峰值 {peak:.1f} / {total:.1f} GB，每次迭代 {sec:.2f} 秒')
print(f'估計一個模型（{args.epochs} epoch × {-(-args.n_train // args.batch)} 次迭代）約 {iters * sec / 3600:.1f} 小時（不含資料讀取）')
