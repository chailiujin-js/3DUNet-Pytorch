"""
基于Dice的loss函数，计算时pred和target的shape必须相同，亦即target为onehot编码后的Tensor
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, pred, target):

        # pred = pred.squeeze(dim=1)

        smooth = 1

        dice = 0.
        # dice系数的定义
        for i in range(pred.size(1)):
            dice += 2 * (pred[:,i] * target[:,i]).sum(dim=1).sum(dim=1).sum(dim=1) / (pred[:,i].pow(2).sum(dim=1).sum(dim=1).sum(dim=1) +
                                                target[:,i].pow(2).sum(dim=1).sum(dim=1).sum(dim=1) + smooth)
        # 返回的是dice距离
        dice = dice / pred.size(1)
        return torch.clamp((1 - dice).mean(), 0, 1)

class ELDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):

        smooth = 1

        dice = 0.
        # dice系数的定义
        for i in range(pred.size(1)):
            dice += 2 * (pred[:,i] * target[:,i]).sum(dim=1).sum(dim=1).sum(dim=1) / (pred[:,i].pow(2).sum(dim=1).sum(dim=1).sum(dim=1) +
                                                target[:,i].pow(2).sum(dim=1).sum(dim=1).sum(dim=1) + smooth)

        dice = dice / pred.size(1)
        # 返回的是dice距离
        return torch.clamp((torch.pow(-torch.log(dice + 1e-5), 0.3)).mean(), 0, 2)


class HybridLoss(nn.Module):
    def __init__(self):
        super().__init__()

        self.bce_loss = nn.BCELoss()
        self.bce_weight = 1.0

    def forward(self, pred, target):

        smooth = 1

        dice = 0.
        # dice系数的定义
        for i in range(pred.size(1)):
            dice += 2 * (pred[:,i] * target[:,i]).sum(dim=1).sum(dim=1).sum(dim=1) / (pred[:,i].pow(2).sum(dim=1).sum(dim=1).sum(dim=1) +
                                                target[:,i].pow(2).sum(dim=1).sum(dim=1).sum(dim=1) + smooth)

        dice = dice / pred.size(1)

        # 返回的是dice距离 +　二值化交叉熵损失
        return torch.clamp((1 - dice).mean(), 0, 1) + self.bce_loss(pred, target) * self.bce_weight


class JaccardLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):

        smooth = 1

        # jaccard系数的定义
        jaccard = 0.

        for i in range(pred.size(1)):
            jaccard  += (pred[:,i] * target[:,i]).sum(dim=1).sum(dim=1).sum(dim=1) / (pred[:,i].pow(2).sum(dim=1).sum(dim=1).sum(dim=1) +
                        target[:,i].pow(2).sum(dim=1).sum(dim=1).sum(dim=1) - (pred[:,i] * target[:,i]).sum(dim=1).sum(dim=1).sum(dim=1) + smooth)

        # 返回的是jaccard距离
        jaccard = jaccard / pred.size(1)
        return torch.clamp((1 - jaccard).mean(), 0, 1)

class SSLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):

        smooth = 1

        loss = 0.

        for i in range(pred.size(1)):
            s1 = ((pred[:,i] - target[:,i]).pow(2) * target[:,i]).sum(dim=1).sum(dim=1).sum(dim=1) / (smooth + target[:,i].sum(dim=1).sum(dim=1).sum(dim=1))

            s2 = ((pred[:,i] - target[:,i]).pow(2) * (1 - target[:,i])).sum(dim=1).sum(dim=1).sum(dim=1) / (smooth + (1 - target[:,i]).sum(dim=1).sum(dim=1).sum(dim=1))

            loss += (0.05 * s1 + 0.95 * s2)

        return loss / pred.size(1)

class TverskyLoss(nn.Module):
    
    def __init__(self):
        super().__init__()

    # def forward(self, pred, target):
    #
    #     smooth = 1
    #
    #     dice = 0.
    #
    #     for i in range(pred.size(1)):
    #         target = target.squeeze(0)
    #         print("pred:", pred.size())
    #         print("target:", target.size())
    #
    #
    #         # print(pred.size)
    #         dice += (pred[:,i] * target[:,i]).sum(dim=1).sum(dim=1).sum(dim=1) / ((pred[:,i] * target[:,i]).sum(dim=1).sum(dim=1).sum(dim=1)+
    #                     0.3 * (pred[:,i] * (1 - target[:,i])).sum(dim=1).sum(dim=1).sum(dim=1) + 0.7 * ((1 - pred[:,i]) * target[:,i]).sum(dim=1).sum(dim=1).sum(dim=1) + smooth)
    #
    #
    #     dice = dice / pred.size(1)
    #     return torch.clamp((1 - dice).mean(), 0, 2)

    # def forward(self, pred, target):
    #     smooth = 1
    #     dice = 0.
    #
    #     # 确保 pred 和 target 的尺寸匹配
    #     if pred.size()[1:] != target.size()[1:]:
    #         target = target.squeeze(0)  # 去掉 batch 维度
    #
    #     # 确保 pred 和 target 的通道数一致
    #     if pred.size(1) != target.size(1):
    #         raise ValueError("pred and target must have the same number of channels")
    #
    #     print("pred:", pred.size())
    #     print("target:", target.size())
    #
    #     for i in range(pred.size(1)):
    #         # 计算每个通道的 Tversky 损失
    #         dice += (pred[:, i].squeeze(1) * target[:, i]).sum(dim=1).sum(dim=1) / (
    #                 (pred[:, i].squeeze(1) * target[:, i]).sum(dim=1).sum(dim=1) +
    #                 0.3 * (pred[:, i].squeeze(1) * (1 - target[:, i])).sum(dim=1).sum(dim=1) +
    #                 0.7 * ((1 - pred[:, i].squeeze(1)) * target[:, i]).sum(dim=1).sum(dim=1) + smooth
    #         )
    #
    #     dice = dice / pred.size(1)
    #     return torch.clamp((1 - dice).mean(), 0, 2)

    def forward(self, pred, target):
        smooth = 1
        dice = 0.

        # 确保 pred 和 target 的尺寸匹配
        if pred.size()[1:] != target.size()[1:]:
            target = target.squeeze(0)  # 去掉 batch 维度

        # 确保 pred 和 target 的通道数一致
        if pred.size(1) != target.size(1):
            raise ValueError("pred and target must have the same number of channels")

        for i in range(pred.size(1)):
            # 计算每个通道的 Tversky 损失
            dice += (pred[:, i] * target[:, i]).sum(dim=1).sum(dim=1).sum(dim=1) / (
                    (pred[:, i] * target[:, i]).sum(dim=1).sum(dim=1).sum(dim=1) +
                    0.3 * (pred[:, i] * (1 - target[:, i])).sum(dim=1).sum(dim=1).sum(dim=1) +
                    0.7 * ((1 - pred[:, i]) * target[:, i]).sum(dim=1).sum(dim=1).sum(dim=1) + smooth
            )

        dice = dice / pred.size(1)
        return torch.clamp((1 - dice).mean(), 0, 2)



