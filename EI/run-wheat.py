from __future__ import print_function, division
import argparse  # 新增导入argparse模块
import os
import gc
import numpy as np
import pandas as pd
from PIL import Image
import glob
from torch import optim
import torch.utils.data
import torch
import torch.nn.functional as F
import torch.nn
import torchvision
import matplotlib.pyplot as plt
import natsort
from torch.utils.data.sampler import SubsetRandomSampler
from Data_Loader import Images_Dataset, Images_Dataset_folder
from torchinfo import summary
import torchsummary
import shutil
import random
from Models import Unet_dict, NestedUNet, U_Net, R2U_Net, AttU_Net, R2AttU_Net
from model.TransUnet import TransUnet
from model.SegNet import SegNet
from model.Pspnet import Pspnet
from model.Enet import ENet
from model.FCN8s import FCN8s
from model.Unet3plus import UNet3Plus
from model.VIT import VIT
from model.VIT_1 import VIT_1
from model.deeplabv3.deeplabv3 import DeepLab
from model.ECAVUnet import ECAVUnet
from model.ECAUnet import ECAUnet
from model.CBAM_UNet import CBAM_UNet
from model.CBAM_VUNet import CBAM_VUNet
from model.CBAM_ECAUNet import CBAM_ECAUNet
from model.CBAM_ECAVUnet import CBAM_ECAVUnet
from losses import calc_loss, threshold_predictions_v,threshold_predictions_p
from ploting import plot_kernels, LayerActivations, input_images, plot_grad_flow
from Metrics import dice_coeff, accuracy_score, numeric_score, iou_coeff, precision_score,recall_score
import time
from datetime import datetime
from tensorboardX import SummaryWriter
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
base_results_dir = os.path.join(PROJECT_ROOT, 'EI', 'results')
#Checking if GPU is used
#######################################################
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='Index of GPU to use (default: 0)')
args = parser.parse_args()

train_on_gpu = torch.cuda.is_available()

if train_on_gpu:
    if args.gpu < torch.cuda.device_count():
        device = torch.device(f"cuda:{args.gpu}")
        torch.cuda.set_device(device)
        print(f'Using GPU [{args.gpu}] for training')
    else:
        print(f'Warning: GPU [{args.gpu}] not available. Falling back to CPU.')
        device = torch.device("cpu")
        train_on_gpu = False
else:
    print('CUDA is not available. Training on CPU')
    device = torch.device("cpu")
#######################################################
#Setting the basic paramters of the model
#######################################################
batch_size =16
print('batch_size = ' + str(batch_size))
valid_size = 0.2
img_size1 = 256#224#360#720#576
img_size2 = 256#224#640#1280#1024
epoch = 300
patience = 300
print('epoch = ' + str(epoch))
random_seed = random.randint(1, 100)
print('random_seed = ' + str(random_seed))
shuffle = True#随机打乱
valid_loss_min = np.inf
num_workers = min(64, os.cpu_count())
lossT = []
lossL = []
lossL.append(np.inf)
lossT.append(np.inf)
epoch_valid = epoch
n_iter = 1
i_valid = 0
pin_memory = False
if train_on_gpu:
    pin_memory = True
#######################################################
#Setting up the model
#######################################################
model_Inputs = [U_Net,      #0
                DeepLab,    #1
                AttU_Net,   #2
                UNet3Plus,  #3
                FCN8s,      #4
                ENet,       #5
                Pspnet,     #6
                SegNet,     #7
                TransUnet,  #8
                ECAUnet,    #9
                VIT,        #10
                CBAM_UNet,  #11
                ECAVUnet,   #12
                CBAM_VUNet, #13
                CBAM_ECAUNet,#14
                CBAM_ECAVUnet,#15
                NestedUNet, #16
                VIT_1
                 ]
#选择模型，0是U_Net,1是R2U_Net
select=9
model_name = model_Inputs[select].__name__
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
run_dir = os.path.join(base_results_dir, model_name, timestamp)
os.makedirs(run_dir, exist_ok=True)
pred_dir = os.path.join(run_dir, 'pred')
model_save_dir = os.path.join(run_dir, 'saved_models')
plot_dir = os.path.join(run_dir, 'result_plot')  # 新增图表目录
os.makedirs(plot_dir, exist_ok=True)  # 确保目录存在
os.makedirs(pred_dir, exist_ok=True)
os.makedirs(model_save_dir, exist_ok=True)
log_path = os.path.join(run_dir, 'training.log')
tensorboard_dir = os.path.join(run_dir, 'runs')
writer = SummaryWriter(log_dir=tensorboard_dir)
def model_unet(model_input, in_channel=3, out_channel=1):
    model_test = model_input(in_channel, out_channel)
    return model_test

model_test = model_unet(model_Inputs[select], 3, 1)
model_test.to(device)
#######################################################
#Getting the Summary of Model
#######################################################
torchsummary.summary(model_test, input_size=(3, img_size1, img_size2))
#######################################################
#Passing the Dataset of Images and Labels
#######################################################
t_data = os.path.join(PROJECT_ROOT, 'data', 'images')
l_data = os.path.join(PROJECT_ROOT, 'data', 'masks')
_sample_images = natsort.natsorted(glob.glob(os.path.join(t_data, '*.jpg')))
test_image = _sample_images[0] if _sample_images else os.path.join(t_data, 'sample.jpg')
test_label = os.path.join(l_data, os.path.splitext(os.path.basename(test_image))[0] + '.png')
test_folderP = os.path.join(t_data, '*.jpg')
test_folderL = os.path.join(l_data, '*.png')
Training_Data = Images_Dataset_folder(t_data,l_data)
New_folder = run_dir
read_test_folder112 = glob.glob(test_folderP)  # 使用正确变量名
x_sort_test = natsort.natsorted(read_test_folder112)  # 自然排序文件列表
# 新增保存函数（添加在训练循环之前）
def save_results_to_log(results_dict, log_path, overwrite=False):
    """将结果保存到LOG文件，符合标准日志格式"""
    import os
    import time
    # 创建目录（同原CSV逻辑）
    dir_path = os.path.dirname(log_path)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)
    # 处理写入模式
    mode = 'w' if overwrite else 'a'
    # 生成带时间戳的日志行
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    log_line = f"[{timestamp}] " + \
               " | ".join([f"{k}:{v}" for k, v in results_dict.items()]) + "\n"
    # 首次写入时添加标题说明
    if not os.path.exists(log_path) or overwrite:
        header = f"[{timestamp}] Created training log\n" \
                 f"[Format] timestamp | Epoch | Train Loss | Valid Loss | Dice | ...\n"
        with open(log_path, 'w', encoding='utf-8') as f:
            f.write(header)
    # 追加日志内容
    with open(log_path, 'a', encoding='utf-8') as f:
        f.write(log_line)
#######################################################
#Giving a transformation for input data
#######################################################
data_transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((img_size1,img_size2)),
         #   torchvision.transforms.CenterCrop(96),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
# 标签预处理（单通道，无标准化）
label_transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((img_size1,img_size2)),
            torchvision.transforms.ToTensor()
        ])
#######################################################
#Trainging Validation Split
#######################################################
num_train = len(Training_Data)
indices = list(range(num_train))
split = int(np.floor(valid_size * num_train))

if shuffle:
    np.random.seed(random_seed)
    np.random.shuffle(indices)

train_idx, valid_idx = indices[split:], indices[:split]
train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)

train_loader = torch.utils.data.DataLoader(Training_Data, batch_size=batch_size, sampler=train_sampler,
                                           num_workers=num_workers, pin_memory=pin_memory,)
valid_loader = torch.utils.data.DataLoader(Training_Data, batch_size=batch_size, sampler=valid_sampler,
                                           num_workers=num_workers, pin_memory=pin_memory,)
#######################################################
#Using Adam as Optimizer
#######################################################
initial_lr = 0.0001
#opt = torch.optim.Adam(model_test.parameters(), lr=initial_lr)
opt = optim.SGD(model_test.parameters(), lr = initial_lr, momentum=0.99)# try SGD

MAX_STEP = int(1000)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, MAX_STEP, eta_min=1e-5)
#scheduler = optim.lr_scheduler.CosineAnnealingLr(opt, epoch, 1)
#######################################################
#Creating a Folder for every data of the program
#######################################################

#######################################################
#Setting the folder of saving the predictions
#######################################################
read_pred = pred_dir  # 使用创建的pred目录
#######################################################
#Checking if prediction folder exixts
#######################################################
if os.path.exists(read_pred) and os.path.isdir(read_pred):
    shutil.rmtree(read_pred)

try:
    os.mkdir(read_pred)
except OSError:
    print("Creation of the prediction directory '%s' failed of dice loss" % read_pred)
else:
    print("Successfully created the prediction directory '%s' of dice loss" % read_pred)
#######################################################
#checking if the model exists and if true then delete
#######################################################
read_model_path = model_save_dir  # 直接使用创建好的目录
if os.path.exists(read_model_path) and os.path.isdir(read_model_path):
    shutil.rmtree(read_model_path)
    print('Model folder there, so deleted for newer one')

try:
    os.mkdir(read_model_path)
except OSError:
    print("Creation of the model directory '%s' failed" % read_model_path)
else:
    print("Successfully created the model directory '%s' " % read_model_path)

train_losses = []
val_losses = []
dice_scores = []
iou_scores = []
accuracy_scores = []
precision_scores = []
recall_scores = []
f1_scores = []

#######################################################
#Training loop
#######################################################
for i in range(epoch):
    train_loss = 0.0
    valid_loss = 0.0
    total_samples = 0
    total_TP = 0
    total_FP = 0
    total_TN = 0
    total_FN = 0
    since = time.time()

    #######################################################
    # Training Data
    #######################################################
    model_test.train()
    k = 1
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        input_images(x, y, i, n_iter, k, run_dir)
        opt.zero_grad()
        y_pred = model_test(x)
        lossT = calc_loss(y_pred, y)
        train_loss += lossT.item() * x.size(0)
        lossT.backward()
        opt.step()
        x_size = lossT.item() * x.size(0)
        k = 2
        scheduler.step()
        lr = scheduler.get_last_lr()[0]

    #######################################################
    # Validation Step
    #######################################################
    model_path = os.path.join(model_save_dir, f'Unet_epoch_{epoch}_batchsize_{batch_size}.pth')
    model_test.eval()
    with torch.no_grad():
        for x1, y1 in valid_loader:
            x1, y1 = x1.to(device), y1.to(device)
            y_pred1 = model_test(x1)
            lossL = calc_loss(y_pred1, y1)
            valid_loss += lossL.item() * x1.size(0)

            # Convert predictions and labels to numpy arrays
            pred_mask = (torch.sigmoid(y_pred1) > 0.5).cpu().numpy().astype(bool)
            true_mask = y1.cpu().numpy().astype(bool)
            del y_pred1, y1
            torch.cuda.empty_cache()

            # Accumulate confusion matrix components
            for b in range(pred_mask.shape[0]):
                batch_pred = pred_mask[b][0]  # Assuming shape (batch, 1, H, W)
                batch_true = true_mask[b][0]

                # Calculate pixel-level metrics
                FP, FN, TP, TN = numeric_score(batch_pred, batch_true)
                total_TP += TP
                total_FP += FP
                total_TN += TN
                total_FN += FN

            total_samples += x1.size(0)

        # Calculate final metrics using accumulated values
        epsilon = 1e-7
        avg_dice = (2 * total_TP) / (2 * total_TP + total_FP + total_FN + epsilon)
        avg_iou = total_TP / (total_TP + total_FP + total_FN + epsilon)
        avg_accuracy = (total_TP + total_TN) / (total_TP + total_TN + total_FP + total_FN + epsilon) * 100
        avg_precision = total_TP / (total_TP + total_FP + epsilon)
        avg_recall = total_TP / (total_TP + total_FN + epsilon)
        f1 = 2 * (avg_precision * avg_recall) / (avg_precision + avg_recall + epsilon)

        #######################################################
    #Saving the predictions
    #######################################################
        im_tb = Image.open(test_image)
        im_label = Image.open(test_label)
        s_tb = data_transform(im_tb)
        s_label = label_transform(im_label)
        s_label = s_label.detach().numpy()

        pred_tb = model_test(s_tb.unsqueeze(0).to(device)).cpu()
        pred_tb = F.sigmoid(pred_tb)
        pred_tb = pred_tb.detach().numpy()
   #pred_tb = threshold_predictions_v(pred_tb)
        if (i + 1) % 1 == 0:
            save_path = os.path.join(pred_dir, f'img_iteration_{n_iter}_epoch_{i}.png')
            x1 = plt.imsave(save_path, pred_tb[0][0])
        del s_tb, pred_tb  # 删除图像预处理和预测结果 [3](@ref)
        gc.collect()
  #  accuracy = accuracy_score(pred_tb[0][0], s_label)
    #######################################################
    #To write in Tensorboard
    #######################################################
        train_loss = train_loss / len(train_idx)
        valid_loss = valid_loss / len(valid_idx)
        writer.add_scalars('Metrics/main', {
            'Accuracy': avg_accuracy,
            'Dice': avg_dice * 100,
            'IoU': avg_iou * 100
        }, epoch)

        writer.add_scalars('Metrics/breakdown', {
            'Precision': avg_precision * 100,
            'Recall': avg_recall * 100,
            'F1': f1 * 100
        }, epoch)

        writer.add_scalars('Confusion_Matrix', {
            'FP': total_FP,
            'FN': total_FN,
            'TP': total_TP,
            'TN': total_TN
        }, epoch)
        if (i+1) % 1 == 0:
            print('-' * 80)
            print('Epoch: {}/{} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(i + 1, epoch, train_loss,
                                                                                      valid_loss))
            print('Metrics:')
            print(f'Dice: {avg_dice:.4f} \tIoU: {avg_iou:.4f} \tAccuracy: {avg_accuracy:.2f}% \tPrecision: {avg_precision:.4f} \tRecall: {avg_recall:.4f}   \tF1: {f1:.4f}')
            print('-' * 80)

        current_lr = opt.param_groups[0]['lr']  # 获取当前学习率
        epoch_results = {
        'Epoch': i + 1,
        'Train Loss': "{:.4g}".format(train_loss),
        'Valid Loss': "{:.4g}".format(valid_loss),
        'Dice': "{:.4g}".format(avg_dice),
        'IoU': "{:.4g}".format(avg_iou),
        'Accuracy': "{:.4g}".format(avg_accuracy),
        'Precision': "{:.4g}".format(avg_precision),
        'Recall': "{:.4g}".format(avg_recall),
        'F1': "{:.4g}".format(f1),
        'Learning Rate': current_lr
    }
    save_results_to_log(epoch_results, log_path)
    train_losses.append(train_loss)
    val_losses.append(valid_loss)
    dice_scores.append(avg_dice)
    iou_scores.append(avg_iou)
    accuracy_scores.append(avg_accuracy)
    precision_scores.append(avg_precision)
    recall_scores.append(avg_recall)
    f1_scores.append(2 * (avg_precision * avg_recall) / (avg_precision + avg_recall + 1e-7))
    torch.cuda.empty_cache()
    #######################################################
    #Early Stopping
    #######################################################

    if valid_loss <= valid_loss_min and epoch_valid >= i:
        print(f'Validation loss decreased ({valid_loss_min:.6f} --> {valid_loss:.6f}). Saving model...')
        # 修改为动态命名（保存时使用当前epoch i）
        best_model_path = os.path.join(model_save_dir, 'best_model.pth')
        torch.save(model_test.state_dict(), best_model_path)

        gc.collect()  # 清理保存模型时可能产生的临时对象 [3](@ref)
        valid_loss_min = valid_loss  # 仅在损失降低时更新最小值
        counter = 0  # 重置计数器
    else:
        counter += 1

    # 检查早停条件
    if counter >= patience:
        epoch_valid = i + 1  # 记录实际终止epoch ← 核心修复
        print(f"Early stopping triggered at epoch {epoch_valid}!")
        print(f"The min valid_loss is {valid_loss_min}")
        break  # 正确跳出循环
        torch.cuda.empty_cache()  # 防止显存未完全释放导致后续操作异常 [2](@ref)
    # print(accuracy)
        if round(valid_loss, 4) == round(valid_loss_min, 4):
            print(i_valid)
            i_valid = i_valid+1
        valid_loss_min = valid_loss
    #######################################################
    # Extracting the intermediate layers
    #######################################################
    #####################################
    # for kernals
    #####################################
    x1 = torch.nn.ModuleList(model_test.children())
    # x2 = torch.nn.ModuleList(x1[16].children())
     #x3 = torch.nn.ModuleList(x2[0].children())
    #To get filters in the layers
     #plot_kernels(x1.weight.detach().cpu(), 7)
    #####################################
    # for images
    #####################################
    x2 = len(x1)
    dr = LayerActivations(x1[x2-1]) #Getting the last Conv Layer

    img = Image.open(test_image)
    s_tb = data_transform(img)

    pred_tb = model_test(s_tb.unsqueeze(0).to(device)).cpu()
    pred_tb = F.sigmoid(pred_tb)
    pred_tb = pred_tb.detach().numpy()

    # 修改后
    plot_kernels(dr.features, n_iter, 7, run_dir, cmap="rainbow")

    time_elapsed = time.time() - since
    print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    del y_pred, lossT  # 显式删除训练阶段变量 [1](@ref)
    gc.collect()
    torch.cuda.empty_cache()  # 周期性释放显存碎片 [2](@ref
    n_iter += 1

# 训练结束后保存图表
def save_plots(plot_dir, train_losses, val_losses, dice, iou, accuracy, precision, recall, f1):
    """保存训练指标图表到指定目录"""
    os.makedirs(plot_dir, exist_ok=True)
    # 绘制Loss曲线
    plt.figure(figsize=(10,6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(plot_dir, 'loss_plot.png'))
    plt.close()

    # 绘制其他指标
    metrics = [
        ('Dice Coefficient', dice, 'dice_plot.png'),
        ('IoU Score', iou, 'iou_plot.png'),
        ('Accuracy', accuracy, 'accuracy_plot.png'),
        ('Precision', precision, 'precision_plot.png'),
        ('Recall', recall, 'recall_plot.png'),
        ('F1 Score', f1, 'f1_plot.png')
    ]
    for title, data, filename in metrics:
        plt.figure(figsize=(10,6))
        plt.plot(data, label=title)
        plt.title(title)
        plt.xlabel('Epoch')
        plt.ylabel('Value')
        plt.legend()
        plt.savefig(os.path.join(plot_dir, filename))
        plt.close()

# 调用绘图函数
save_plots(plot_dir, train_losses, val_losses, dice_scores, iou_scores,
           accuracy_scores, precision_scores, recall_scores, f1_scores)
gc.collect()
torch.cuda.empty_cache()  # 释放整个epoch累积的显存碎片 [1](@ref)
#######################################################
#closing the tensorboard writer
#######################################################
writer.close()  # 确保数据写入磁盘
#######################################################
#checking if cuda is available
#######################################################
#######################################################
#Loading the model
#######################################################
# 加载最终保存的最佳模型
final_model_path = os.path.join(model_save_dir, 'best_model.pth')
print("模型权重保存路径:", final_model_path)
model_test.load_state_dict(
    torch.load(final_model_path, map_location=device, weights_only=True)
)
model_test.eval()  # 设置为评估模式
#######################################################
#opening the test folder and creating a folder for generated images
#######################################################
read_test_folder112 = os.path.join(run_dir, 'gen_images')
read_test_folder_P_Thres = os.path.join(run_dir, 'pred_threshold')
read_test_folder_L_Thres = os.path.join(run_dir, 'label_threshold')
os.makedirs(read_test_folder_P_Thres, exist_ok=True)
os.makedirs(read_test_folder_L_Thres, exist_ok=True)
if os.path.exists(read_test_folder112):
    shutil.rmtree(read_test_folder112)
os.makedirs(read_test_folder112, exist_ok=True)
#For Prediction Threshold
if os.path.exists(read_test_folder_P_Thres) and os.path.isdir(read_test_folder_P_Thres):
    shutil.rmtree(read_test_folder_P_Thres)

try:
    os.mkdir(read_test_folder_P_Thres)
except OSError:
    print("Creation of the testing directory %s failed" % read_test_folder_P_Thres)
else:
    print("Successfully created the testing directory %s " % read_test_folder_P_Thres)

#For Label Threshold
if os.path.exists(read_test_folder_L_Thres) and os.path.isdir(read_test_folder_L_Thres):
    shutil.rmtree(read_test_folder_L_Thres)

try:
    os.mkdir(read_test_folder_L_Thres)
except OSError:
    print("Creation of the testing directory %s failed" % read_test_folder_L_Thres)
else:
    print("Successfully created the testing directory %s " % read_test_folder_L_Thres)
#######################################################
#saving the images in the files
#######################################################
img_test_no = 0

for i in range(len(x_sort_test)):
    im = Image.open(x_sort_test[i])

    im1 = im
    im_n = np.array(im1)
    im_n_flat = im_n.reshape(-1, 1)

    for j in range(im_n_flat.shape[0]):
        if im_n_flat[j] != 0:
            im_n_flat[j] = 255

    s = data_transform(im)
    pred = model_test(s.unsqueeze(0).cuda()).cpu()
    pred = F.sigmoid(pred)
    pred = pred.detach().numpy()

#    pred = threshold_predictions_p(pred) #Value kept 0.01 as max is 1 and noise is very small.

    if i % 1 == 0:
        img_test_no = img_test_no + 1
    save_path = os.path.join(read_test_folder112,
                             f'im_epoch_{epoch}int_{i}_img_no_{img_test_no}.png')
    x1 = plt.imsave(save_path, pred[0][0])
####################################################
# Calculating the Metrics
####################################################
data_transform = torchvision.transforms.Compose([
    torchvision.transforms.Grayscale(),
    torchvision.transforms.Resize((img_size1, img_size2)),
])
read_test_folderP = glob.glob(os.path.join(run_dir, 'gen_images', '*'))
x_sort_testP = natsort.natsorted(read_test_folderP)

read_test_folderL = glob.glob(test_folderL)
x_sort_testL = natsort.natsorted(read_test_folderL)

# Initialize metrics accumulators
total_tp = 0
total_fp = 0
total_tn = 0
total_fn = 0
for i in range(len(read_test_folderP)):
    # Process predicted image
    x = Image.open(x_sort_testP[i])
    s = data_transform(x)                # 输入图像预处理
    s = np.array(s)                      # 转换为NumPy数组
    s = threshold_predictions_v(s)       # 阈值处理
    pred = (s > 127).astype(np.uint8)    # 转换为二值掩码
    # 确保pred是二维数组（H x W）
    if pred.ndim == 3 and pred.shape[0] == 1:  # 如果形状为 (1, H, W)
        pred = pred.squeeze(0)                 # 转换为 (H, W)
    elif pred.ndim == 3 and pred.shape[2] == 1:  # 如果形状为 (H, W, 1)
        pred = pred.squeeze(-1)

    # 保存预测图像
    if (i + 1) % 1 == 0:
        # 创建预测阈值结果目录
        # 生成保存路径
        pred_threshold_dir=read_test_folder_P_Thres
        save_path = os.path.join(
            pred_threshold_dir,
            f'im_epoch_{epoch}int_{i}_img_no_{img_test_no}.png'
        )
        # 保存图像
        plt.imsave(save_path, pred, cmap='gray')
    # Process label image
    y = Image.open(x_sort_testL[i])
    s2 = data_transform(y)
    s3 = np.array(s2)
    label = (s3 > 127).astype(np.uint8)  # 转换为二值掩码
    # 确保label是二维数组
    if label.ndim == 3 and label.shape[0] == 1:
        label = label.squeeze(0)
    elif label.ndim == 3 and label.shape[2] == 1:
        label = label.squeeze(-1)
    # 保存标签图像（使用label变量而非pred）
    if (i + 1) % 1 == 0:
        # 生成保存路径
        label_threshold_dir=read_test_folder_L_Thres
        save_path = os.path.join(
            label_threshold_dir,
            f'im_epoch_{epoch}int_{i}_img_no_{img_test_no}.png'
        )
        # 保存图像
        plt.imsave(save_path, label, cmap='gray')
    # Calculate confusion matrix components
    tp = np.sum((pred == 1) & (label == 1))
    fp = np.sum((pred == 1) & (label == 0))
    tn = np.sum((pred == 0) & (label == 0))
    fn = np.sum((pred == 0) & (label == 1))
    # Accumulate metrics
    total_tp += tp
    total_fp += fp
    total_tn += tn
    total_fn += fn
epsilon = 1e-7  # Prevent division by zero
# Calculate metrics
dice = (2 * total_tp) / (2 * total_tp + total_fp + total_fn + epsilon)
iou = total_tp / (total_tp + total_fp + total_fn + epsilon)
accuracy = (total_tp + total_tn) / (total_tp + total_tn + total_fp + total_fn + epsilon) * 100
precision = total_tp / (total_tp + total_fp + epsilon)
recall = total_tp / (total_tp + total_fn + epsilon)
f1 = 2 * (precision * recall) / (precision + recall + epsilon)
# Format output with consistent decimal places
print(f'Dice: {dice:.4f}    IoU: {iou:.4f}    Accuracy: {accuracy:.2f}%    '
      f'Precision: {precision:.4f}    Recall: {recall:.4f}    F1: {f1:.4f}')
test_results = {
    f'Dice: {dice:.4f} ...'
}
# 修改后（分离指标字段）
test_results = {
    'Test Dice': dice,
    'Test IoU': iou,
    'Test Accuracy': accuracy,
    'Test Precision': precision,
    'Test Recall': recall,
    'Test F1': f1
}
save_results_to_log(test_results, log_path)
