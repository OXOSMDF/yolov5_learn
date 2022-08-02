# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Loss functions
"""

import torch
import torch.nn as nn

from utils.metrics import bbox_iou
from utils.torch_utils import de_parallel

# 标签平滑
def smooth_BCE(eps=0.1):  # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    # return positive, negative label smoothing BCE targets
    return 1.0 - 0.5 * eps, 0.5 * eps


class BCEBlurWithLogitsLoss(nn.Module):
    # BCEwithLogitLoss() with reduced missing label effects.
    def __init__(self, alpha=0.05):
        super().__init__()
        self.loss_fcn = nn.BCEWithLogitsLoss(reduction='none')  # must be nn.BCEWithLogitsLoss()
        self.alpha = alpha

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        pred = torch.sigmoid(pred)  # prob from logits
        dx = pred - true  # reduce only missing label effects
        # dx = (pred - true).abs()  # reduce missing label and false label effects
        alpha_factor = 1 - torch.exp((dx - 1) / (self.alpha + 1e-4))
        loss *= alpha_factor
        return loss.mean()

# 主要是为了解决one—stage目标检测中正负样本比例失衡严重的问题。该损失函数降低了大量简单负样本在训练中所占的权重
class FocalLoss(nn.Module):
    # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super().__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class QFocalLoss(nn.Module):
    # Wraps Quality focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super().__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)

        pred_prob = torch.sigmoid(pred)  # prob from logits
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = torch.abs(true - pred_prob) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss

# 计算损失（分类损失+置信度损失+框坐标回归损失）
class ComputeLoss:
    sort_obj_iou = False

    # Compute losses
    def __init__(self, model, autobalance=False):
        device = next(model.parameters()).device  # get model device
        h = model.hyp  # hyperparameters

        # Define criteria
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['cls_pw']], device=device))
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['obj_pw']], device=device))

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        self.cp, self.cn = smooth_BCE(eps=h.get('label_smoothing', 0.0))  # positive, negative BCE targets

        # Focal loss
        g = h['fl_gamma']  # focal loss gamma
        if g > 0:
            BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)

        m = de_parallel(model).model[-1]  # Detect() module
        self.balance = {3: [4.0, 1.0, 0.4]}.get(m.nl, [4.0, 1.0, 0.25, 0.06, 0.02])  # P3-P7
        self.ssi = list(m.stride).index(16) if autobalance else 0  # stride 16 index
        self.BCEcls, self.BCEobj, self.gr, self.hyp, self.autobalance = BCEcls, BCEobj, 1.0, h, autobalance
        self.na = m.na  # number of anchors
        self.nc = m.nc  # number of classes
        self.nl = m.nl  # number of layers
        self.anchors = m.anchors
        self.device = device

    def __call__(self, p, targets):  # predictions, targets
        # 初始化各部分的损失
        lcls = torch.zeros(1, device=self.device)  # class loss
        lbox = torch.zeros(1, device=self.device)  # box loss
        lobj = torch.zeros(1, device=self.device)  # object loss
        # 获得标签分类，边框，索引，anchors
        tcls, tbox, indices, anchors = self.build_targets(p, targets)  # targets

        # Losses
        # 遍历每个预测的输出
        for i, pi in enumerate(p):  # layer index, layer predictions
            b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
            # 通过indices返回网格的输出
            tobj = torch.zeros(pi.shape[:4], dtype=pi.dtype, device=self.device)  # target obj

            n = b.shape[0]  # number of targets
            if n:
                # pxy, pwh, _, pcls = pi[b, a, gj, gi].tensor_split((2, 4, 5), dim=1)  # faster, requires torch 1.8.0
                # 找到对应网格的输出，去除对应位置的预测值
                pxy, pwh, _, pcls = pi[b, a, gj, gi].split((2, 2, 1, self.nc), 1)  # target-subset of predictions

                # Regression
                # 对输出的xywh做反算，目标框回归公式
                pxy = pxy.sigmoid() * 2 - 0.5
                pwh = (pwh.sigmoid() * 2) ** 2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1)  # predicted box
                # 计算边框损失
                iou = bbox_iou(pbox, tbox[i], CIoU=True).squeeze()  # iou(prediction, target)
                lbox += (1.0 - iou).mean()  # iou loss

                # Objectness
                # 根据model.gr设置objectness的标签值；有目标的conf分支权重
                # 不同的anchor和gt bbox匹配度不一样，预测框和gt bbox 的匹配度也不一样，如果权重设置一样肯定不是最优的
                # 故将预测框和bbox的iuo作为权重乘到conf分支，用于表征预测质量
                iou = iou.detach().clamp(0).type(tobj.dtype)
                if self.sort_obj_iou:
                    j = iou.argsort()
                    b, a, gj, gi, iou = b[j], a[j], gj[j], gi[j], iou[j]
                if self.gr < 1:
                    iou = (1.0 - self.gr) + self.gr * iou
                tobj[b, a, gj, gi] = iou  # iou ratio

                # Classification
                if self.nc > 1:  # cls loss (only if multiple classes)
                    t = torch.full_like(pcls, self.cn, device=self.device)  # targets
                    t[range(n), tcls[i]] = self.cp
                    lcls += self.BCEcls(pcls, t)  # BCE 每个类被单独计算loss

                # Append targets to text file
                # with open('targets.txt', 'a') as file:
                #     [file.write('%11.5g ' * 4 % tuple(x) + '\n') for x in torch.cat((txy[i], twh[i]), 1)]
            # 计算obj loss
            obji = self.BCEobj(pi[..., 4], tobj)
            lobj += obji * self.balance[i]  # obj loss
            if self.autobalance:
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()

        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]
        #损失加权求和 
        lbox *= self.hyp['box']
        lobj *= self.hyp['obj']
        lcls *= self.hyp['cls']
        bs = tobj.shape[0]  # batch size

        return (lbox + lobj + lcls) * bs, torch.cat((lbox, lobj, lcls)).detach()

    def build_targets(self, p, targets):
    # 用于获得在训练时计算的loss函数所需要的目标框，被认为是正样本
    # yolo_v5支持跨网络预测
    # 对于任何一个bbox，三个输出预测特征图都有可能右先验框匹配
    # 该函数的输出的正样本框比传入的targets数目多
    # 具体处理流程:
    # 1.对于任何一层计算当前bbox和目前层anchor的匹配程度，不采用iou，而是shape比例
    # 如果anchor和bbox的宽高差距大于4，则认为是不匹配，此时忽略相应的bbox，即当成背景
    # 2.然后对bbox计算落在网格所有anchors都计算loss（并不是直接和GT框比较loss）
    # 注意此时落在网格不再是一个，而是附近多个，这样就增加了正样本数，可能存在有些bbox
    # 在三个尺度都预测的情况。另外yolo_v5没有conf分支忽略阈值的操作。
        # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
        na, nt = self.na, targets.shape[0]  # number of anchors, targets
        tcls, tbox, indices, anch = [], [], [], []
        gain = torch.ones(7, device=self.device)  # normalized to gridspace gain
        # anchor索引，后面有用，用于表示当前bbox和当前层的哪个anchor匹配
        ai = torch.arange(na, device=self.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)
        # 先repeat targets和当前层anchor个数一样，相当于每个bbox变成三个，然后和三个anchor单独匹配
        targets = torch.cat((targets.repeat(na, 1, 1), ai[..., None]), 2)  # append anchor indices
        # 设置网格中心的偏移量
        g = 0.5  # bias
        # 附近的四个网格
        off = torch.tensor(
            [
                [0, 0],
                [1, 0],
                [0, 1],
                [-1, 0],
                [0, -1],  # j,k,l,m
                # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
            ],
            device=self.device).float() * g  # offsets
        # 对于每个检测层进行处理
        for i in range(self.nl): # 三个尺度的预测特征图输出分支 
            anchors, shape = self.anchors[i], p[i].shape#当前分支的anchor大小（已经处于对应的stride）
            # p是网格的输出
            gain[2:6] = torch.tensor(shape)[[3, 2, 3, 2]]  # xyxy gain

            # Match targets to anchors
            # 将标签框的xywh从基于0~1映射到基于特征图；targets的xywh本身是归一化尺度，故需要变成特征图尺度
            t = targets * gain  # shape(3,n,7)
            # 对每个输出层单独匹配
            # 首先将target wh shape和anchor 的wh计算比例，如果比例过大，则说明匹配度不高，将该bbox过滤，在当前层认为是bg
            if nt:
                # Matches
                # 预测的wh与anchor的wh做匹配，筛选掉比值大于hyp['anchor_t']的，从而更好地回归
                # 作者采用新的wh回归方式 （(wh.sigmoid()*2)**2*anchors[i]）
                # 将标签框与anchor的倍数控制在0~4之间；hyp.scratch.yaml中的超参数anchor_t=4,用于判定anchor与标签框的契合度。
                
                
                # 计算当前target的wh和anchor的wh比例值
                # 如果最大比例大于预设值model.hpy['anchor_t']=4，则当前target和anchor匹配度不高，不强制回归，而把target丢弃
                # 计算比值ratio
                r = t[..., 4:6] / anchors[:, None]  # wh ratio
                # 筛选满足 1/[anchor_t] < targets w/h < [anchor_t]的框
                j = torch.max(r, 1 / r).max(2)[0] < self.hyp['anchor_t']  # compare
                # 筛选过后的t.shape = (M,7),M为筛选过后的数量
                # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
                t = t[j]  # filter ；注意过滤规则没有考虑xy，也就是当前bbox的wh是和所有anchor计算的

                # Offsets
                gxy = t[:, 2:4]  # grid xy ； label的中心点坐标
                # 得到中心点相对于当前特征图的坐标(M,2)
                gxi = gain[[2, 3]] - gxy  # inverse
                # 对于筛选后的bbox，计算其落在哪个网格内，同时找出邻近的网格，将这些网格都认为是负责预测该bbox的网格
                j, k = ((gxy % 1 < g) & (gxy > 1)).T
                l, m = ((gxi % 1 < g) & (gxi > 1)).T
                j = torch.stack((torch.ones_like(j), j, k, l, m))
                t = t.repeat((5, 1, 1))[j]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = targets[0]
                offsets = 0

            # Define
            # 对每个bbox找出对应的正样本anchor，
            # 其中包括b表示当前的bbox属于batch内部的第几张图
            # a表示当前的bbox和当前层的第几个anchor匹配上
            # gi,gj是对应的负责预测该bbox的网格坐标
            # gw是对应的bbox wh，
            # c 是该bbox类别
            bc, gxy, gwh, a = t.chunk(4, 1)  # (image, class), grid xy, grid wh, anchors
            a, (b, c) = a.long().view(-1), bc.long().T  # anchors, image, class
            # 当前坐标落在哪个网格上
            gij = (gxy - offsets).long()
            gi, gj = gij.T  # grid indices

            # Append
            # 添加索引，方便计算损失的时候去除对应位置的输出
            indices.append((b, a, gj.clamp_(0, shape[2] - 1), gi.clamp_(0, shape[3] - 1)))  # image, anchor, grid
            tbox.append(torch.cat((gxy - gij, gwh), 1))  # box 坐标值
            anch.append(anchors[a])  # anchors 尺寸
            tcls.append(c)  # class

        return tcls, tbox, indices, anch
