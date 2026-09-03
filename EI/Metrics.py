import numpy as np
from scipy import spatial

def dice_coeff(im1, im2, empty_score=1.0):
    im1 = np.asarray(im1).astype(bool)
    im2 = np.asarray(im2).astype(bool)

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch")

    # 若输入已为布尔类型，可注释掉以下两行
    # im1 = im1 > 0.5
    # im2 = im2 > 0.5

    im_sum = im1.sum() + im2.sum()
    if im_sum == 0:
        return empty_score

    intersection = np.logical_and(im1, im2)
    return 2. * intersection.sum() / im_sum

def numeric_score(prediction, groundtruth):
    """ 直接使用布尔逻辑运算，避免依赖 0/1 比较 """
    FP = np.sum(prediction & ~groundtruth).astype(float)
    FN = np.sum(~prediction & groundtruth).astype(float)
    TP = np.sum(prediction & groundtruth).astype(float)
    TN = np.sum(~prediction & ~groundtruth).astype(float)
    return FP, FN, TP, TN

def accuracy_score(prediction, groundtruth):
    """Getting the accuracy of the model"""

    FP, FN, TP, TN = numeric_score(prediction, groundtruth)
    N = FP + FN + TP + TN
    accuracy = np.divide(TP + TN, N)
    return accuracy * 100.0

def iou_coeff(im1, im2, smooth=1e-7):
    im1 = np.asarray(im1).astype(bool)
    im2 = np.asarray(im2).astype(bool)
    intersection = np.logical_and(im1, im2).sum()
    union = np.logical_or(im1, im2).sum()
    return (intersection + smooth) / (union + smooth)

def precision_score(im1, im2):
    im1 = np.asarray(im1).astype(bool)
    im2 = np.asarray(im2).astype(bool)
    true_positives = np.logical_and(im1, im2).sum()
    predicted_positives = im1.sum()
    return true_positives / (predicted_positives + 1e-7)

def recall_score(im1, im2):
    im1 = np.asarray(im1).astype(bool)
    im2 = np.asarray(im2).astype(bool)
    true_positives = np.logical_and(im1, im2).sum()
    actual_positives = im2.sum()
    return true_positives / (actual_positives + 1e-7)

