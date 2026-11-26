"""
在官方 H-SAM（CVPR 2024，github.com/Cccccczh404/H-SAM，commit 5bdf491）的 train.py、trainer.py 加入 Ψ 加權
  --wcases <csv>      加權清單（第一欄 case = 影像名稱），清單影像 loss × --wfactor
  --wfactor 3         清單影像的權重（其餘為 1）
  --per_case_loss     對照組：與專家相同的逐張 loss，但權重全為 1
  loss：L = (1/n) Σ w_i · L_i，L_i = (1 − dice_param)·CE_i + dice_param·Dice_i（逐張計算，n = batch 張數）
        官方的 Dice 是整個 batch 一起算，逐張計算後即使權重全為 1 也和官方不同，所以另有對照組（同 psi-weighted-nnunet）
  不給 --wcases、--per_case_loss 時，走官方原本的程式，與原版完全相同
  修改處皆標 [Ψ加權]；每個 epoch 印出本輪加權樣本數
用法（在 H-SAM 資料夾）：python patch_hsam_weighted.py          會先備份成 train.py.orig、trainer.py.orig；重複執行會略過
"""
import os
import shutil
import sys

TAG = '[Ψ加權]'


def patch(fn, pairs):
    s = open(fn, encoding='utf-8').read()
    if TAG in s:
        print(f'{fn}：已修改過，略過')
        return
    if not os.path.exists(fn + '.orig'):
        shutil.copy(fn, fn + '.orig')
    for a, b in pairs:
        if s.count(a) != 1:
            sys.exit(f'✗ {fn}：找不到要修改的位置（或不只一處），請確認是官方 H-SAM commit 5bdf491\n{a}')
        s = s.replace(a, b)
    open(fn, 'w', encoding='utf-8').write(s)
    print(f'{fn}：已修改（原檔 {fn}.orig）')


patch('train.py', [(
    "parser.add_argument('--dice_param', type=float, default=0.9)\n",
    "parser.add_argument('--dice_param', type=float, default=0.9)\n"
    "parser.add_argument('--wcases', type=str, default=None, help='加權清單 csv（第一欄 case）')   # [Ψ加權]\n"
    "parser.add_argument('--wfactor', type=float, default=3.0, help='清單影像的 loss 權重')   # [Ψ加權]\n"
    "parser.add_argument('--per_case_loss', action='store_true', help='對照組：逐張 loss、權重全為 1')   # [Ψ加權]\n",
)])

patch('trainer.py', [
    ("class AddGaussianNoise(object):\n",
     "def calc_loss_per_case(outputs, label_batch, n_classes, dice_weight, w):   # [Ψ加權] 逐張計算，再加權平均（÷ n）\n"
     "    low_res_logits = outputs['low_res_logits']\n"
     "    ce = F.cross_entropy(low_res_logits, label_batch[:].long(), reduction='none').flatten(1).mean(1)   # 每張的 CE（像素平均）\n"
     "    prob = torch.softmax(low_res_logits, dim=1)\n"
     "    dice = 0.0\n"
     "    for i in range(n_classes):                                   # 與 utils.DiceLoss 相同的公式，只是每張分開算\n"
     "        s, t = prob[:, i], (label_batch == i).float()\n"
     "        inter = (s * t).flatten(1).sum(1)\n"
     "        y_sum, z_sum = (t * t).flatten(1).sum(1), (s * s).flatten(1).sum(1)\n"
     "        dice = dice + (1 - (2 * inter + 1e-5) / (z_sum + y_sum + 1e-5))\n"
     "    dice = dice / n_classes\n"
     "    loss_i = (1 - dice_weight) * ce + dice_weight * dice\n"
     "    return (w * loss_i).mean(), ce.mean(), dice.mean()\n\n\n"
     "class AddGaussianNoise(object):\n"),
    ("    best_performance = 0.0\n",
     "    best_performance = 0.0\n"
     "    WSET = {}   # [Ψ加權] 影像名稱 → 權重\n"
     "    if args.wcases:\n"
     "        import csv\n"
     "        with open(args.wcases, encoding='utf-8-sig') as f:\n"
     "            WSET = {r[0]: args.wfactor for k, r in enumerate(csv.reader(f)) if k > 0 and r}\n"
     "        logging.info(f'[Ψ加權] 清單 {args.wcases}：{len(WSET)} 張 × {args.wfactor}')\n"
     "    PER_CASE = bool(args.wcases) or args.per_case_loss   # [Ψ加權] 不給選項時走官方原本的 loss\n"),
    ("    for epoch_num in iterator:\n",
     "    for epoch_num in iterator:\n"
     "        n_w, n_all = 0, 0   # [Ψ加權]\n"),
    ("            loss1, loss_ce1, loss_dice1 = calc_loss(outputs1, low_res_label_batch, ce_loss, dice_loss, dice_weight=args.dice_param)\n"
     "            loss2, loss_ce2, loss_dice2 = calc_loss(outputs2, label_batch, ce_loss, dice_loss, dice_weight=args.dice_param)\n",
     "            if PER_CASE:   # [Ψ加權]\n"
     "                w = torch.tensor([WSET.get(c, 1.0) for c in sampled_batch['case_name']], dtype=torch.float32, device=image_batch.device)\n"
     "                n_w, n_all = n_w + int((w != 1).sum()), n_all + len(w)\n"
     "                loss1, loss_ce1, loss_dice1 = calc_loss_per_case(outputs1, low_res_label_batch, dice_loss.n_classes, args.dice_param, w)\n"
     "                loss2, loss_ce2, loss_dice2 = calc_loss_per_case(outputs2, label_batch, dice_loss.n_classes, args.dice_param, w)\n"
     "            else:\n"
     "                loss1, loss_ce1, loss_dice1 = calc_loss(outputs1, low_res_label_batch, ce_loss, dice_loss, dice_weight=args.dice_param)\n"
     "                loss2, loss_ce2, loss_dice2 = calc_loss(outputs2, label_batch, ce_loss, dice_loss, dice_weight=args.dice_param)\n"),
    ("        save_interval = 50 # int(max_epoch/6)\n",
     "        if PER_CASE:   # [Ψ加權]\n"
     "            logging.info(f'[Ψ加權] epoch {epoch_num} 本輪加權樣本 {n_w}/{n_all}')\n"
     "        save_interval = 50 # int(max_epoch/6)\n"),
])
