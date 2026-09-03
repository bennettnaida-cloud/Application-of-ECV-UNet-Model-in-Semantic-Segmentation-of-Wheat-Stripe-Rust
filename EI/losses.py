from __future__ import print_function, division
import torch.nn.functional as F

def dice_loss(prediction, target):
    """Calculating the dice loss
    Args:
        prediction = predicted image
        target = Targeted image
    Output:
        dice_loss"""

    smooth = 1.0

    i_flat = prediction.view(-1)
    t_flat = target.view(-1)

    intersection = (i_flat * t_flat).sum()

    return 1 - ((2. * intersection + smooth) / (i_flat.sum() + t_flat.sum() + smooth))

def calc_loss(prediction, target, bce_weight=0.5):
    """Calculating the loss and metrics
    Args:
        prediction = predicted image
        target = Targeted image
        metrics = Metrics printed
        bce_weight = 0.5 (default)
    Output:
        loss : dice loss of the epoch """
    bce = F.binary_cross_entropy_with_logits(prediction, target)
    prediction = F.sigmoid(prediction)
    dice = dice_loss(prediction, target)

    loss = bce * bce_weight + dice * (1 - bce_weight)

    return loss

def tversky_loss(prediction, target, alpha=0.7, beta=0.3, smooth=1e-6):
    prediction = F.sigmoid(prediction)
    tp = (prediction * target).sum()
    fp = (prediction * (1 - target)).sum()
    fn = ((1 - prediction) * target).sum()
    return 1 - (tp + smooth) / (tp + alpha*fp + beta*fn + smooth)

def focal_tversky_loss(prediction, target, gamma=0.75, **kwargs):
    t_loss = tversky_loss(prediction, target, **kwargs)
    return (t_loss + 1e-6) ** gamma  # 对高损失区域（难样本）施加指数惩罚

def calc_loss_2(prediction, target,
              tversky_weight=0.7,
              focal_tversky_weight=0.3,
              alpha=0.7):
    """
    组合损失公式：
    Loss = Tversky * tversky_weight + FocalTversky * focal_tversky_weight
    推荐参数：tversky_weight=0.7, focal_tversky_weight=0.3, alpha=0.7
    """
    tversky = tversky_loss(prediction, target, alpha=alpha)
    f_tversky = focal_tversky_loss(prediction, target, alpha=alpha)
    return tversky * tversky_weight + f_tversky * focal_tversky_weight

def threshold_predictions_v(predictions, thr=150):
    thresholded_preds = predictions[:]
   # hist = cv2.calcHist([predictions], [0], None, [2], [0, 2])
   # plt.plot(hist)
   # plt.xlim([0, 2])
   # plt.show()
    low_values_indices = thresholded_preds < thr
    thresholded_preds[low_values_indices] = 0
    low_values_indices = thresholded_preds >= thr
    thresholded_preds[low_values_indices] = 255
    return thresholded_preds


def threshold_predictions_p(predictions, thr=0.01):
    thresholded_preds = predictions[:]
    #hist = cv2.calcHist([predictions], [0], None, [256], [0, 256])
    low_values_indices = thresholded_preds < thr
    thresholded_preds[low_values_indices] = 0
    low_values_indices = thresholded_preds >= thr
    thresholded_preds[low_values_indices] = 1
    return thresholded_preds