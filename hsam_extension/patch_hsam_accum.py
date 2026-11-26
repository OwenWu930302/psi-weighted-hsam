"""
在 H-SAM 的 train.py、trainer.py 加入梯度累積 --accum（先執行 patch_hsam_weighted.py，再執行本檔）
  RTX 4080（16 GB）放不下官方的 batch 16（8 × n_gpu 2），所以把同一個 batch 16 分成 --accum 份依序 forward / backward，
  累積梯度後才更新一次：資料順序、資料增強、batch 大小（16）、學習率與迭代次數都和官方相同
  逐張 loss（對照組、Ψ 專家）：每份乘上（張數 / 16）後相加 = 整個 batch 的 (1/n) Σ w_i·L_i，公式與一次算完相同
  官方 loss（baseline）：官方的 Dice 是整個 batch 一起算，分份後變成每份各自算
  另外 H-SAM 訓練時會在第二個 decoder 的輸入加隨機雜訊，強度 = 0.2 ×（該次 forward 的 embedding 最大值 − 最小值），
  分份後強度改由每份各自計算，雜訊本身是隨機的；除這兩點外與一次算完相同（已在 CPU 上逐項比對 loss）
  模型只有 LayerNorm / InstanceNorm（沒有 BatchNorm），分份不影響正規化
  --accum 1（預設）時走官方原本的程式；修改處皆標 [梯度累積]
用法（在 H-SAM 資料夾）：python patch_hsam_accum.py
"""
import sys

TAG = '[梯度累積]'


def prepare(fn, pairs):
    s = open(fn, encoding='utf-8').read()
    if TAG in s:
        return None
    if '[Ψ加權]' not in s:
        sys.exit(f'✗ {fn}：請先執行 python patch_hsam_weighted.py')
    for a, b in pairs:
        if s.count(a) != 1:
            sys.exit(f'✗ {fn}：找不到要修改的位置（兩個檔案都沒有改動）\n{a}')
        s = s.replace(a, b)
    return s


TRAIN = [(
    "parser.add_argument('--per_case_loss', action='store_true', help='對照組：逐張 loss、權重全為 1')   # [Ψ加權]\n",
    "parser.add_argument('--per_case_loss', action='store_true', help='對照組：逐張 loss、權重全為 1')   # [Ψ加權]\n"
    "parser.add_argument('--accum', type=int, default=1, help='同一個 batch 分成幾份 forward/backward（顯示卡記憶體不足時）')   # [梯度累積]\n",
)]

OLD = """            outputs1,outputs2, attn1, attn2 = model(image_batch, multimask_output, args.img_size, gt=low_res_label_batch)
            if PER_CASE:   # [Ψ加權]
                w = torch.tensor([WSET.get(c, 1.0) for c in sampled_batch['case_name']], dtype=torch.float32, device=image_batch.device)
                n_w, n_all = n_w + int((w != 1).sum()), n_all + len(w)
                loss1, loss_ce1, loss_dice1 = calc_loss_per_case(outputs1, low_res_label_batch, dice_loss.n_classes, args.dice_param, w)
                loss2, loss_ce2, loss_dice2 = calc_loss_per_case(outputs2, label_batch, dice_loss.n_classes, args.dice_param, w)
            else:
                loss1, loss_ce1, loss_dice1 = calc_loss(outputs1, low_res_label_batch, ce_loss, dice_loss, dice_weight=args.dice_param)
                loss2, loss_ce2, loss_dice2 = calc_loss(outputs2, label_batch, ce_loss, dice_loss, dice_weight=args.dice_param)
            weight = 0.6**(0.990**epoch_num)
            weight_self = 0.2
            loss = (1-weight)*loss1 + (weight)*loss2
<<BLANK12>>
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
"""
OLD = OLD.replace('<<BLANK12>>', ' ' * 12)   # 官方原檔這一行是 12 個空白
NEW = """            if args.accum > 1:   # [梯度累積] 同一個 batch 分成 accum 份，逐份 forward / backward，累積後更新一次
                weight = 0.6**(0.990**epoch_num)
                optimizer.zero_grad()
                n_b = image_batch.shape[0]
                logs = torch.zeros(5)
                for idx in torch.arange(n_b).chunk(args.accum):
                    j = idx.to(image_batch.device)
                    o1, o2, _, _ = model(image_batch[j], multimask_output, args.img_size, gt=low_res_label_batch[j])
                    if PER_CASE:
                        w = torch.tensor([WSET.get(sampled_batch['case_name'][k], 1.0) for k in idx.tolist()], dtype=torch.float32, device=image_batch.device)
                        n_w, n_all = n_w + int((w != 1).sum()), n_all + len(w)
                        l1, c1, d1 = calc_loss_per_case(o1, low_res_label_batch[j], dice_loss.n_classes, args.dice_param, w)
                        l2, c2, d2 = calc_loss_per_case(o2, label_batch[j], dice_loss.n_classes, args.dice_param, w)
                    else:
                        l1, c1, d1 = calc_loss(o1, low_res_label_batch[j], ce_loss, dice_loss, dice_weight=args.dice_param)
                        l2, c2, d2 = calc_loss(o2, label_batch[j], ce_loss, dice_loss, dice_weight=args.dice_param)
                    f = len(idx) / n_b
                    lc = (1-weight)*l1 + (weight)*l2
                    (lc * f).backward()
                    logs += f * torch.stack([torch.as_tensor(v).detach().float().cpu() for v in (lc, c1, d1, c2, d2)])
                optimizer.step()
                loss, loss_ce1, loss_dice1, loss_ce2, loss_dice2 = logs
            else:
""" + ''.join('    ' + l if l.strip() else l for l in OLD.splitlines(True))

new = {fn: prepare(fn, pairs) for fn, pairs in (('train.py', TRAIN), ('trainer.py', [(OLD, NEW)]))}
for fn, txt in new.items():
    if txt is None:
        print(f'{fn}：已修改過，略過')
    else:
        open(fn, 'w', encoding='utf-8').write(txt)
        print(f'{fn}：已加入梯度累積')
