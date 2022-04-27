# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Loss functions
"""

import torch
import torch.nn as nn

from utils.metrics import bbox_iou
from utils.torch_utils import is_parallel


def smooth_BCE(eps=0.1):  # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    # return positive, negative label smoothing BCE targets
    return 1.0 - 0.5 * eps, 0.5 * eps


class BCEBlurWithLogitsLoss(nn.Module):
    # BCEwithLogitLoss() with reduced missing label effects.
    def __init__(self, alpha=0.05):
        super(BCEBlurWithLogitsLoss, self).__init__()
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


class FocalLoss(nn.Module):
    # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super(FocalLoss, self).__init__()
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
        super(QFocalLoss, self).__init__()
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


class ComputeLoss:
    # Compute losses
    def __init__(self, model, autobalance=False):
        self.sort_obj_iou = False
        device = next(model.parameters()).device  # get model device 获取设备
        h = model.hyp  # hyperparameters 获取超参数

        # Define criteria  定义类别和目标性得分损失函数
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['cls_pw']], device=device))
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['obj_pw']], device=device))

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        self.cp, self.cn = smooth_BCE(eps=h.get('label_smoothing', 0.0))  # positive, negative BCE targets

        # Focal loss
        g = h['fl_gamma']  # focal loss gamma
        if g > 0:
            BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)

        det = model.module.model[-1] if is_parallel(model) else model.model[-1]  # Detect() module
        self.balance = {3: [4.0, 1.0, 0.4]}.get(det.nl, [4.0, 1.0, 0.25, 0.06, .02])  # P3-P7
        self.ssi = list(det.stride).index(16) if autobalance else 0  # stride 16 index
        self.BCEcls, self.BCEobj, self.gr, self.hyp, self.autobalance = BCEcls, BCEobj, 1.0, h, autobalance
        for k in 'na', 'nc', 'nl', 'anchors':
            setattr(self, k, getattr(det, k))

    def __call__(self, p, targets):  # predictions, targets, model
        device = targets.device
        lcls, lbox, lobj = torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device)
        # 获取标签分类，边框，索引，anchors
        tcls, tbox, indices, anchors = self.build_targets(p, targets)  # targets

        # Losses
        for i, pi in enumerate(p):  # layer index, layer predictions
            b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
            tobj = torch.zeros_like(pi[..., 0], device=device)  # target obj

            n = b.shape[0]  # number of targets
            if n:
                ps = pi[b, a, gj, gi]  # prediction subset corresponding to targets

                # Regression
                pxy = ps[:, :2].sigmoid() * 2. - 0.5
                pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1)  # predicted box
                iou = bbox_iou(pbox.T, tbox[i], x1y1x2y2=False, CIoU=True)  # iou(prediction, target)
                lbox += (1.0 - iou).mean()  # iou loss

                # Objectness
                score_iou = iou.detach().clamp(0).type(tobj.dtype)
                if self.sort_obj_iou:
                    sort_id = torch.argsort(score_iou)
                    b, a, gj, gi, score_iou = b[sort_id], a[sort_id], gj[sort_id], gi[sort_id], score_iou[sort_id]
                tobj[b, a, gj, gi] = (1.0 - self.gr) + self.gr * score_iou  # iou ratio

                # Classification
                if self.nc > 1:  # cls loss (only if multiple classes)
                    t = torch.full_like(ps[:, 5:], self.cn, device=device)  # targets
                    t[range(n), tcls[i]] = self.cp
                    lcls += self.BCEcls(ps[:, 5:], t)  # BCE

                # Append targets to text file
                # with open('targets.txt', 'a') as file:
                #     [file.write('%11.5g ' * 4 % tuple(x) + '\n') for x in torch.cat((txy[i], twh[i]), 1)]

            obji = self.BCEobj(pi[..., 4], tobj)
            lobj += obji * self.balance[i]  # obj loss
            if self.autobalance:
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()

        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]
        lbox *= self.hyp['box']
        lobj *= self.hyp['obj']
        lcls *= self.hyp['cls']
        bs = tobj.shape[0]  # batch size

        return (lbox + lobj + lcls) * bs, torch.cat((lbox, lobj, lcls)).detach()

    """
    build_target函数用于获得在训练时计算loss函数所需要的目标框，即被认为是正样本
    与yolov3/4不同，v5支持跨网格预测
    对于任何一个bbox,三个输出预测特征层都可能有先验框anchor匹配
    该函数输出的正样本框比传入的targets(GT框-我们标注的框)数目多
    具体处理过程：
    （1）对于任何一层计算当前bbox和当前层anchor的匹配程度，不采用iou,而是shape比例
        如果anchor和bbox的宽高比差距大于4,则人为不匹配，此时忽略相应的bbox，即当作背景
    （2）然后对bbox计算落在网格所有anchors都记算loss（并不是直接和GT框比较计算loss）
        注意此时落在网格不再是1个，而是附近的多个，这样就增加了正样本数，可能存在有些bbox在三个尺度都预测的情况
        另外，yolov5也没有conf分支忽略阈值（ignore_thresh）的操作，而v3/4有
    """

    def build_targets(self, p, targets):
        """
        p:网络输出 List[torch.tensor*3] p[i].shape = (b,3,h,w,nc+5)
          hw分别为特征图的长宽，b为batch size
        targets:GT框 targets.shape=(nt,6),6=icxywh i表示第i+1张图片，c为类别，坐标为xywh
        """
        # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
        na, nt = self.na, targets.shape[0]  # number of anchors, targets
        tcls, tbox, indices, anch = [], [], [], []  #初始化
        gain = torch.ones(7, device=targets.device)  # normalized to gridspace gain
        # ai.shape=(na,nt) 生成anchor索引
        # anchor索引，后面有用，用于表示当前bbox是和当前层的哪个anchor匹配
        ai = torch.arange(na, device=targets.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)
        # 先repeat,targets和当前层anchor个数一样，相当于每个bbox变成了3个，然后和三个anchor单独匹配
        targets = torch.cat((targets.repeat(na, 1, 1), ai[:, :, None]), 2)  # append anchor indices

        # 设置网格中心偏移量
        g = 0.5  # bias
        # 附近的4个网格
        off = torch.tensor([[0, 0],
                            [1, 0], [0, 1], [-1, 0], [0, -1],  # j,k,l,m
                            # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
                            ], device=targets.device).float() * g  # offsets

        # 对每个检测层进行处理
        for i in range(self.nl):  # 三个尺度的预测特征图输出分支，i代表尺度
            anchors = self.anchors[i]  # 当前分支的anchor大小，（已经除以了当前特征图对应的stride）
            gain[2:6] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]  # xyxy gain

            # Match targets to anchors
            # 将标签框的xywh从基于0-1映射到基于特征图，targets的xywh本身是归一化尺度，故需要变成特征图尺度
            t = targets * gain
            # 对每个输出层单独匹配
            # 首先将target变成anchor尺度，方便计算
            # 然后将target wh shape 和 anchor的wh计算比例，如果比例过大，则说明匹配度不高，将该bbox过滤，在当前层认为是bg
            if nt:
                # Matches
                """
                预测的wh与anchor的wh做匹配，筛选掉比值大于hyp['anchor_t']的，从而更好的回归
                作者采用新的wh回归方式：（wh.sigmoid()*2) ** 2 *anchors[i]
                原来v3/4为anchors[i]*exp(wh)
                将标签框与anchor的倍数控制在0-4之间，hyp.scratch.yaml中的超参数anchor_t=4,用于判定anchors与标签框契合度
                """
                # 计算当前target的wh和anchor的比例值
                # 如果最大比例大于预设值model.hyp['anchor_t']=4,则当前target与anchor匹配度不高，不强制回归，而把target丢弃
                # 计算比值ratio
                r = t[:, :, 4:6] / anchors[:, None]  # wh ratio
                j = torch.max(r, 1. / r).max(2)[0] < self.hyp['anchor_t']  # compare
                # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
                t = t[j]  # filter

                # Offsets
                gxy = t[:, 2:4]  # grid xy
                gxi = gain[[2, 3]] - gxy  # inverse
                """
                把相对于各个网格左上角x<0.5,y<0.5和相对于右下角x<0.5,y<0.5的框提出出来，也就是j k l m
                在选取gij（也就是标签框分配给的网格）的时候对这四个部分的框都做一个偏移（减去上面的offsets）
                也就是下面的gij=(gxy - offsets).long()
                再将这四个部分的框与原始的gxy拼接在一起，一共就是五个部分
                yolov3/4仅仅采用当前网格的anchor进行回归，v4也有解决网格跑偏的措施，即通过对sigmoid限制输出
                yolov5中心点回归从yolov3/4的0-1范围变成0.5-1.5的范围
                中心点回归的公式变为：
                xy.sigmoid()*2-0.5 +cx (其中对原始中心点网格坐标扩展两个邻居像素)
                """
                # 对于筛选后的bbox，计算其落在哪个网格内，同时找出临近的网络，将这些网络都认为是负责预测该bbox的网络
                # 浮点数取模的数学定义：对于两个浮点数a和b， a%b=a-n*b,其中n为不超过a/b的最大整数
                j, k = ((gxy % 1. < g) & (gxy > 1.)).T  # 都是 TURE  FALSE 值
                l, m = ((gxi % 1. < g) & (gxi > 1.)).T
                j = torch.stack((torch.ones_like(j), j, k, l, m))
                # 5是因为预设的off是5个
                t = t.repeat((5, 1, 1))[j]
                # 添加偏移量
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = targets[0]
                offsets = 0

            """
            对每个bbox找出对应的正样本anchor，其中包括b表示当前bbox属于batch内部的第几张图片，
            a表示当前bbox和当前层的第几个anchor匹配上，gi,gj是对应的负责预测该bbox的网格坐标，
            gxy是不考虑offset或者或yolov3/4里面设定的该bbox的负责预测网格中心点坐标xy
            gwh是对应的bbox wh,  c是该bbox的类别
            """
            # Define
            b, c = t[:, :2].long().T  # image, class
            gxy = t[:, 2:4]  # grid xy  中心点回归标签
            gwh = t[:, 4:6]  # grid wh  宽高回归标签
            gij = (gxy - offsets).long() # 当前label落在哪个网格上
            gi, gj = gij.T  # grid xy indices（索引值）：gridx  gridy

            # Append
            a = t[:, 6].long()  # anchor indices  anchor的索引
            # 添加索引，方便计算损失的时候提取除对应位置的输出
            indices.append((b, a, gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1)))  # image, anchor, grid indices
            tbox.append(torch.cat((gxy - gij, gwh), 1))  # box 坐标值
            anch.append(anchors[a])  # anchors 尺寸
            tcls.append(c)  # class

        return tcls, tbox, indices, anch
