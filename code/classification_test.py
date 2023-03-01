
import mindspore.context as context
# 设置  mindspore下的 GPU环境
context.set_context(device_id=1, device_target="GPU")

import mindspore as ms
import numpy as np
import mindspore.dataset as ds
from mindspore import nn, ops
import os
import nibabel as nib

from mindspore.common.tensor import Tensor
from sklearn.metrics import roc_curve, auc

dataset_dir= "./Dataset" # 数据集根目录
test_batch_size = 1
#image_size = [76, 94, 76] # 训练图像空间大小
workers = 2 # 并行线程个数
num_classes = 2 # 分类数量


# 自定义测试集
class create_dataset_test:
    def __init__(self, dataset_dir):  

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
                

# 利用上面写好的那个函数，获取处理后的测试数据集
dataset_generator_test = create_dataset_test(dataset_dir)
dataset_test = ds.GeneratorDataset(dataset_generator_test, num_parallel_workers=workers, column_names=["image", "label"], shuffle=False).batch(test_batch_size)

step_size_test = dataset_test.get_dataset_size()




"""
构建网络
"""
from densenet import DenseNet21


"""
这个函数主要是用来处理预训练模型的 就是如果有预训练模型参数需要在训练之前输入 就把pretrained设为True 此处由于没有预训练模型提供 因此后面在训练的时候设置的是False
"""
from mindspore import load_checkpoint, load_param_into_net
def _model(pretrained: bool = False):
    # num_classes = 2
    model = DenseNet21(2)
    #存储路径
    model_ckpt = "./LoadPretrainedModel/0227.ckpt"

    if pretrained:
        # download(url=model_url, path=model_ckpt)
        param_dict = load_checkpoint(model_ckpt)
        load_param_into_net(model, param_dict)

    return model


"""
模型测试
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


best_ckpt_path = "./BestCheckpoint/DenseNet_best.ckpt"
ROC_path = './Roc'

roc_labels = os.path.join(ROC_path, 'labels.txt')
fw = open(roc_labels, 'w') 
roc_scores = os.path.join(ROC_path, 'scores.txt')
fr = open(roc_scores, 'w') 

def val_model(best_ckpt_path, dataset_test):
    net = _model(pretrained=False)


    # 加载模型参数
    param_dict = ms.load_checkpoint(best_ckpt_path)
    ms.load_param_into_net(net, param_dict)
    model = ms.Model(net)

    TP = 0
    FP = 0
    FN = 0
    TN = 0

    LABELS = []
    SCORES = []

    # 加载测试集的数据进行测试
    for i, data in enumerate(dataset_test):
        images = data[0]
        label = data[1].asnumpy()


        # 预测图像类别
        output = model.predict(images)
        softmax = nn.Softmax()
        score = softmax(output).asnumpy()[0][1]


        SCORES.append(score)
        LABELS.append(label)

        fw.write(str(label) + '\n')
        fr.write(str(round(score, 4)) + '\n')


        pred = np.argmax(output.asnumpy(), axis=1)

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
    
    fw.close()
    fr.close()

    acc = (TP + TN)/(TP + TN + FP + FN)
    sen = TP/(TP + FN)
    spe = TN/(FP + TN)
    f1s = 2*(TP/(TP + FP + 0.000001))*(TP/(TP + FN + 0.000001))/((TP/(TP + FP + 0.000001)) + (TP/(TP + FN + 0.000001)) + 0.000001)
    fpr, tpr, thresholds = roc_curve(LABELS, SCORES)
    roc_auc = auc(fpr, tpr)


    # print log info
    print("=" * 80)
    print(
            'Test_ACC:{:.4f} {}/{}'.format(round(acc, 4), (TP + TN), (TP + TN + FP + FN)),
            'Test_SEN:{:.4f} {}/{}'.format(round(sen, 4), TP , (TP + FN)),
            'Test_SPE:{:.4f} {}/{}'.format(round(spe, 4), TN, (FP + TN)),
            'Test_AUC:{:.4f}'.format(round(roc_auc, 4) ),
            'Test_F1S:{:.4f}'.format(round(f1s, 4) ),
        ) 
    print("=" * 80)

# 使用测试数据集进行验证
val_model(best_ckpt_path=best_ckpt_path, dataset_test=dataset_test)




