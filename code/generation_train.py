
import mindspore.context as context
# 设置  mindspore下的 GPU环境
context.set_context(device_id=1, device_target="GPU")

import mindspore as ms
import numpy as np
import mindspore.dataset as ds
from mindspore import nn, ops
import os
import nibabel as nib
import math
import time
from sklearn.metrics import roc_curve, auc
from ssim_loss import SSIM 


dataset_dir= "/media/sdd/gaoxingyu/Project2/Dataset/CN_AD" # 数据集根目录
dataset_dir_test = "/media/sdd/gaoxingyu/Project2/Dataset"
train_batch_size = 4 # 批量大小
test_batch_size = 1
#image_size = [76, 94, 76] # 训练图像空间大小
workers = 1 # 并行线程个数
num_classes = 2 # 分类数量



# 自定义数据集 （训练集和测试集）
class create_dataset_train:
    def __init__(self, dataset_dir, status):  

        if status == 'normal':
            dataset_dir_to = os.path.join(dataset_dir, 'ADNI1_MRI_CN')
        if status == 'disease':
            dataset_dir_to = os.path.join(dataset_dir, 'ADNI1_MRI_AD')
        self.path = dataset_dir_to
        filename_list = os.listdir(self.path)
        train_data = []

        for image_label in filename_list:
            train_data.append(image_label)
        train_data = np.asarray(train_data)
        self.name = train_data

    def __len__(self):
        return len(self.name)
        

    def __getitem__(self, index):
        file_name = self.name[index]                    
        path = os.path.join(self.path, file_name)
        label = np.array(file_name[0]).astype(np.int32) 

        out = nib.load(path).get_fdata()
        data = np.array(out).astype(np.float32) 

        max_val_ = data.max()
        min_val_ = data.min()
        data = (data - min_val_) / (max_val_ - min_val_)

        data = np.expand_dims(data, axis=0)
        return (data, label)

class create_dataset_test:
    def __init__(self, dataset_dir,):  

        dataset_dir_to = os.path.join(dataset_dir, 'ADNI2')

        self.path = dataset_dir_to
        filename_list = os.listdir(self.path)
        train_data = []

        for image_label in filename_list:
            train_data.append(image_label)
        train_data = np.asarray(train_data)
        self.name = train_data

    def __len__(self):
        return len(self.name)
        

    def __getitem__(self, index):
        file_name = self.name[index]                    
        path = os.path.join(self.path, file_name)
        label = np.array(file_name[0]).astype(np.int32) 

        out = nib.load(path).get_fdata()
        data = np.array(out).astype(np.float32) 


        max_val_ = data.max()
        min_val_ = data.min()
        data = (data - min_val_) / (max_val_ - min_val_)

        data = np.expand_dims(data, axis=0)
        return (data, label)
                

# 利用上面写好的那个函数，获取处理后的训练与测试数据集
dataset_generator_train_NC = create_dataset_train(dataset_dir, 'normal')
dataset_train_NC = ds.GeneratorDataset(dataset_generator_train_NC, num_parallel_workers=workers, column_names=["data", "label"], shuffle=True).batch(train_batch_size)
dataset_generator_train_AD = create_dataset_train(dataset_dir, 'disease')
dataset_train_AD = ds.GeneratorDataset(dataset_generator_train_AD, num_parallel_workers=workers, column_names=["data", "label"], shuffle=True).batch(train_batch_size)

dataset_generator_test = create_dataset_test(dataset_dir_test)
dataset_test = ds.GeneratorDataset(dataset_generator_test, num_parallel_workers=workers, column_names=["data", "label"], shuffle=False).batch(test_batch_size)



"""
构建网络
"""
from gan_model import *
from densenet import DenseNet21

"""
这个函数主要是用来处理预训练模型的 就是如果有预训练模型参数需要在训练之前输入 就把pretrained设为True 此处由于没有预训练模型提供 因此后面在训练的时候设置的是False

"""
from mindspore import load_checkpoint, load_param_into_net


def _generator(pretrained: bool = False):
    model = Generator()

    #存储路径
    model_ckpt = "./LoadPretrainedModel/0227.ckpt"

    if pretrained:
        # download(url=model_url, path=model_ckpt)
        param_dict = load_checkpoint(model_ckpt)
        load_param_into_net(model, param_dict)

    return model

def _discriminator(pretrained: bool = False):
    model = Discriminator()
    
    #存储路径
    model_ckpt = "./LoadPretrainedModel/0227.ckpt"

    if pretrained:
        # download(url=model_url, path=model_ckpt)
        param_dict = load_checkpoint(model_ckpt)
        load_param_into_net(model, param_dict)

    return model

def _status_discriminator(pretrained: bool = False):
    model = DenseNet21()
    
    #存储路径
    model_ckpt = "./LoadPretrainedModel/0227.ckpt"

    if pretrained:
        # download(url=model_url, path=model_ckpt)
        param_dict = load_checkpoint(model_ckpt)
        load_param_into_net(model, param_dict)

    return model


"""
训练
"""
###############################################################################################################
###############################################################################################################
###############################################################################################################
###############################################################################################################
###############################################################################################################
###############################################################################################################
###############################################################################################################
###############################################################################################################
###############################################################################################################
import mindspore as ms
# ms.set_context(device_target='GPU')
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# 定义网络，此处不采用预训练，即将pretrained设置为False
MindSpore_G = _generator(pretrained=False)
MindSpore_D = _discriminator(pretrained=False)
MindSpore_SD = DenseNet21(2)

best_ckpt_path = "./BestCheckpoint/DenseNet_best.ckpt"
param_dict = ms.load_checkpoint(best_ckpt_path)
ms.load_param_into_net(MindSpore_SD, param_dict)
nnStatus_Discriminator = ms.Model(MindSpore_SD)

#param.requires_grad = True表示所有参数都需要求梯度进行更新。
for param in MindSpore_G.get_parameters():
    param.requires_grad = True

for param in MindSpore_D.get_parameters():
    param.requires_grad = True

for param in MindSpore_SD.get_parameters():
    param.requires_grad = False

# 设置训练的轮数和学习率 #*****************************************************************************************
num_epochs = 150   

#学习率
lr_G = 0.0001
lr_D = 0.0004

# 定义优化器和损失函数
#Adam优化器，具体可参考论文https://arxiv.org/abs/1412.6980
opt_G = nn.Adam(params=MindSpore_G.trainable_params(), learning_rate=lr_G)

opt_D = nn.Adam(params=MindSpore_D.trainable_params(), learning_rate=lr_D)

# 损失函数
loss_ce = nn.CrossEntropyLoss()
loss_bce = nn.BCELoss()
loss_l1 = nn.L1Loss()
loss_mse = nn.MSELoss()
# loss_ssim = SSIM()

floor = ops.Floor()
"""
生成器
"""
#前向传播，计算loss
def forward_g(inputs, targets):
    generated = MindSpore_G(inputs)

    loss1 = loss_l1(generated, targets)
    loss2 = loss_mse(generated, targets)
    # loss3 = loss_ssim(generated, targets)

    generator_loss = loss1 + loss2
    return generator_loss

#计算梯度和loss
grad_g = ops.value_and_grad(forward_g, None, opt_G.parameters, has_aux=False)

def train_g(inputs, targets):
    loss, grads = grad_g(inputs, targets)
    opt_G(grads)
    return loss

# 实例化模型
netGenerator = ms.Model(MindSpore_G, opt_G)




"""
鉴别器
"""
#前向传播，计算loss
def forward_d(inputs, targets):
    pred = MindSpore_D(inputs)

    discriminator_loss = loss_bce(pred, targets)
    return discriminator_loss

#计算梯度和loss
grad_d = ops.value_and_grad(forward_d, None, opt_D.parameters, has_aux=True)

def train_d(inputs, targets):
    loss, grads = grad_d(inputs, targets)
    opt_G(grads)
    return loss

# 实例化模型
netDiscriminator = ms.Model(MindSpore_D,  opt_D)


# 模型存储路径
ckpt_path = "./Checkpoint"


# 开始循环训练
print("Start Training Loop ...")

# 前150 epoch 还未引入鉴别器（Phase 1）和状态鉴别器（Phase 2）
for epoch in range(num_epochs):
    start = time.time()
    losses = []
    MindSpore_G.set_train()

    # 为每轮训练读入数据
    for i, data in enumerate(dataset_train_NC):
        images = data[0]
        images = Tensor(images, ms.float32)
        gen = MindSpore_G(images)
        loss = train_g(gen, images)
        losses.append(loss)

    # 每轮训练结束后，可视化测试集部分样本结果并保存，同时保存模型权重
    TP = 0
    FP = 0
    FN = 0
    TN = 0
    LABELS = []
    SCORES = []
    for i, data in enumerate(dataset_test):
        images = data[0]
        label = data[1].asnumpy()

        # 生成图像
        gen = MindSpore_G(images)

        # 预测生成图像类别
        output = nnStatus_Discriminator.predict(gen)
        pred = np.argmax(output.asnumpy(), axis=1)

        softmax = nn.Softmax()
        score = softmax(output).asnumpy()[0][1]
        SCORES.append(score)
        LABELS.append(label)

        if pred == 1 and label == 1:
            TP += 1
        elif pred == 1 and label == 0:
            FP += 1
        elif pred == 0 and label == 1:
            FN += 1
        elif pred == 0 and label == 0:
            TN += 1 
        else:
            continue

    acc = (TP + TN)/(TP + TN + FP + FN)
    sen = TP/(TP + FN + 0.000001)
    spe = TN/(FP + TN + 0.000001)
    f1s = 2*(TP/(TP + FP + 0.000001))*(TP/(TP + FN + 0.000001))/((TP/(TP + FP + 0.000001)) + (TP/(TP + FN + 0.000001)) + 0.000001)
    fpr, tpr, thresholds = roc_curve(LABELS, SCORES)
    roc_auc = auc(fpr, tpr)


       

    print("-" * 50)
    print("Epoch: [%3d/%3d], Average Train Loss: [%5.3f]" % (
        epoch+1, num_epochs, sum(losses)/len(losses)
    ))
    print(
            'Test_ACC:{:.4f} {}/{}'.format(round(acc, 4), (TP + TN), (TP + TN + FP + FN)),
            'Test_SEN:{:.4f} {}/{}'.format(round(sen, 4), TP , (TP + FN)),
            'Test_SPE:{:.4f} {}/{}'.format(round(spe, 4), TN, (FP + TN)),
            'Test_AUC:{:.4f}'.format(round(roc_auc, 4) ),
            'Test_F1S:{:.4f}'.format(round(f1s, 4) ),
        ) 
    print("=" * 50) 

    ckpt_path_save = os.path.join(ckpt_path, '{}_Loss{}_TestSPE{}_TestSEN{}_G.ckpt'.format(
        epoch + 1,
        round(sum(losses)/len(losses), 4), 
        round(spe, 4),
        round(sen, 4),))
    
    ms.save_checkpoint(netGenerator, ckpt_path_save)

