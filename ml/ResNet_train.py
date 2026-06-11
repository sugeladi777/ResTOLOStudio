#科学计算部分
import numpy as np
import random
#import simpy as syp
#import matplotlib.pyplot as plt

#神经网络部分
import torch
import torch.nn as nn
from torch.nn import functional as F
#定义设备对象
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print(f"CUDA available: {torch.cuda.is_available()}")

#系统辅助部分
import os
import shutil
import time
import gc
import copy as cp
import threading

#视觉部分
import cv2
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from PIL import Image

#每次训练都给图像添加椒盐噪声，而且是不固定的那种
#改变path2image函数


def _recommended_dataloader_workers() -> int:
    # On Windows, and especially when training is launched from a background Python thread,
    # multiprocessing DataLoader workers are prone to hanging or exiting slowly.
    if os.name == "nt":
        return 0
    if threading.current_thread() is not threading.main_thread():
        return 0
    return min(4, os.cpu_count() or 1)

#------------------------图片的清洗和预处理-------------------------------
#数据备份器:输入文件夹路径,一定要加自身的名字
def preserver(original_path,preserve_path):
    try:
        shutil.copytree(original_path,preserve_path)
    except FileExistsError:
        pass
    print("文件备份完毕")

'''
训练数据集的清洗函数：此函数必须要求训练数据集是以下形式：
（一）./data/train/是母文件夹，放在项目下面，一定要带/号，这是函数的输入
（二）./data/train/4
./datatrain//5之类的是数据文件夹，在母文件夹下平行放置
(三)这个数据集输出的是处理后的文件名构成的Dataset
'''
def train_data_preparer(MotherPath,target_size=(256,256),target_format='.jpg'):
    #获取类名和数目
    contents=os.listdir(MotherPath)
    classes=[f for f in contents if os.path.isdir(os.path.join(MotherPath,f))]
    classnumber=len(classes)
    
    #清洗图片大小和类型
    pic_type={'.jpg','.png','.jpeg','.PNG','.rgb','.webp','.gif','.bmp'} #常见图片类型集合
    compteur=0                                               #计数器+文件名
    
    #开始进行清洗循环
    for genre in classes:                                     #每个文件名
        name_list=os.listdir(MotherPath+genre+'/')
        for name in name_list:
            #print(name)
            split_name=os.path.splitext(name)                         #文件名分割的字符串列表
            image_path=MotherPath+genre+'/'+name
            if len(split_name)!=2:                                    #排除文件夹
                try:
                    os.remove(image_path)
                except:
                    pass
            else: #排除非图片
                if split_name[1] not in pic_type:                     #排除非图片
                    try:
                        os.remove(image_path)
                    except:
                        pass
                else:
                    try:                                                        #防止抛出已有文件的错误

                        #清洗格式
                        image = Image.open(image_path) 
                        if image.mode=='RGBA':
                            image=image.convert('RGB')
                        else:
                            pass
                        # 读取图片
                        resized_image = image.resize(target_size)     # 调整图片尺寸
                        
                        #清洗名称
                        current_time=str(time.localtime().tm_year)+\
                            "_"+str(time.localtime().tm_mon)+\
                                "_"+str(time.localtime().tm_mday)+\
                                    "_"+str(time.localtime().tm_hour)+\
                                        "_"+str(time.localtime().tm_min)+\
                                            "_"+str(time.localtime().tm_sec)+\
                                                "_"+str(compteur)
                        output_filepath=MotherPath+genre+'/'+current_time+target_format
                        resized_image.save(output_filepath)
                        try:
                            os.remove(image_path)
                        except:
                            pass
                        compteur+=1
                    except:
                        compteur+=1
    
    #打印清洗完毕的图片个数和信息
    print("Total Training Picture Number:"+str(compteur))
    print('Traing Picture Name and Type Washed' )

'''
测试数据集的清洗函数
测试数据集的清洗函数：此函数必须要求测试数据集是以下形式：
（一）./data/test/是母文件夹，放在项目下面，一定要带/号，这是函数的输入
下面直接存放用于测试的图片

'''

def test_data_preparer(mother_trace,target_size=(256,256),target_format='.jpg'):
    name_list=os.listdir(mother_trace)                              #文件名列表
    pic_type={'.jpg','.png','.jpeg','.PNG','.rgb','.webp','.gif','.bmp'} #常见图片类型集合
    compteur=0                                               #计数器+文件名
    
    #开始进行清洗循环
    for name in name_list:
        #print(name)
        split_name=os.path.splitext(name)                         #文件名分割的字符串列表
        image_path=mother_trace+name
        if len(split_name)!=2:                                    #排除文件夹
            os.remove(image_path)
        else:
            if split_name[1] not in pic_type:                     #排除非图片
                os.remove(image_path)
            else:
                try:                                                        #防止抛出已有文件的错误

                    #清洗格式
                    image = cv2.imread(image_path)                     # 读取图片
                    resized_image = cv2.resize(image, target_size)     # 调整图片尺寸
                        
                    #清洗名称
                    current_time=str(time.localtime().tm_year)+\
                            "_"+str(time.localtime().tm_mon)+\
                                "_"+str(time.localtime().tm_mday)+\
                                    "_"+str(time.localtime().tm_hour)+\
                                        "_"+str(time.localtime().tm_min)+\
                                            "_"+str(time.localtime().tm_sec)+\
                                                "_"+str(compteur)
                    output_filepath=mother_trace+current_time+target_format
                    cv2.imwrite(output_filepath, resized_image)
                    os.remove(image_path)
                    compteur+=1
                except:
                    compteur+=1
    
    #打印清洗完毕的图片个数和信息
    print("Total Testing Picture Number:"+str(compteur))
    print('Testing Picture Name and Type Washed' )


#--------------------------类别不平衡处理--------------------------------
def oversample_minority_classes(data_dir, ir_threshold=10, target_ratio=0.5):
    """
    对数据集进行过采样以处理类别不平衡问题
    
    参数:
    data_dir: 数据集目录，包含各个类别的子目录
    ir_threshold: 不平衡比阈值，超过此值才进行过采样
    target_ratio: 目标比例，将少数类扩充至最大类数量的此比例
    
    返回:
    bool: 是否进行了过采样
    """
    # 统计各类别样本数
    class_counts = {}
    for class_name in os.listdir(data_dir):
        class_dir = os.path.join(data_dir, class_name)
        if os.path.isdir(class_dir):
            class_counts[class_name] = len(os.listdir(class_dir))
    
    if not class_counts:
        print("警告: 数据目录为空")
        return False
    
    print(f"原始类别分布: {class_counts}")
    
    # 计算不平衡比
    max_count = max(class_counts.values())
    min_count = min(class_counts.values())
    ir = max_count / min_count
    print(f"不平衡比 (IR): {ir:.2f}")
    
    # 如果不平衡比小于阈值，不进行过采样
    if ir <= ir_threshold:
        print(f"IR ({ir:.2f}) <= 阈值 ({ir_threshold})，不需要过采样")
        return False
    
    print(f"IR ({ir:.2f}) > 阈值 ({ir_threshold})，进行过采样")
    
    # 计算目标样本数
    target_count = int(max_count * target_ratio)
    print(f"目标样本数: {target_count} (最大类的 {target_ratio*100}%)")
    
    # 对少数类进行过采样
    for class_name, count in class_counts.items():
        class_dir = os.path.join(data_dir, class_name)
        images = os.listdir(class_dir)
        
        if count >= target_count:
            print(f"类别 {class_name}: 样本数 {count} >= {target_count}，不需要过采样")
            continue
        
        # 需要过采样
        needed = target_count - count
        print(f"类别 {class_name}: 样本数 {count} → 需要增加 {needed} 个样本")
        
        # 创建备份目录
        backup_dir = class_dir + '_backup'
        os.makedirs(backup_dir, exist_ok=True)
        for img in images:
            shutil.copy(os.path.join(class_dir, img), os.path.join(backup_dir, img))
        
        # 过采样：重复复制样本直到达到目标
        idx = 0
        while len(os.listdir(class_dir)) < target_count:
            for img in images:
                if len(os.listdir(class_dir)) >= target_count:
                    break
                src = os.path.join(backup_dir, img)
                name, ext = os.path.splitext(img)
                dst = os.path.join(class_dir, f"oversample_{idx}_{name}{ext}")
                shutil.copy(src, dst)
                idx += 1
        
        # 清理备份目录
        shutil.rmtree(backup_dir)
    
    # 打印过采样后的分布
    new_class_counts = {}
    for class_name in os.listdir(data_dir):
        class_dir = os.path.join(data_dir, class_name)
        if os.path.isdir(class_dir):
            new_class_counts[class_name] = len(os.listdir(class_dir))
    print(f"过采样后类别分布: {new_class_counts}")
    
    return True

#--------------------------图片的增强--------------------------------
#数据增强器：对每个图像进行变换
def fortifier(mother_trace):
    #增强器
    '''
    transform1 = transforms.RandomVerticalFlip(p=1)           #水平镜像反转
    transform2 = transforms.RandomHorizontalFlip(p=1)         #垂直镜像反转
    transform3 = transforms.Grayscale(num_output_channels=3)  #遗照（灰度化）
    transform4 = transforms.RandomAffine((0,90))                  #乱转(随机旋转)
    transform5 = transforms.RandomAffine(0, None, None, (15, 45)) #仿射扭曲
    transform6 = transforms.GaussianBlur(9, 3)              #近视眼（高斯模糊）
    transform7 = transforms.ColorJitter(saturation=(1, 2))    #饱和滤镜
    transform8 = transforms.ColorJitter(contrast=(1, 2))      #有色眼镜（对比滤镜）
    transform9 = transforms.ColorJitter(brightness=(1, 2))    #钛合金狗眼（高亮度）
    '''
    #L1级别
    transform1 = transforms.RandomVerticalFlip(p=1)           #水平镜像反转
    transform2 = transforms.RandomHorizontalFlip(p=1)         #垂直镜像反转
    transform3 = transforms.RandomAffine((60-0.5,60+0.5))                  #乱转(随机旋转)
    transform4 = transforms.RandomAffine((120-0.5,120+0.5))
    transform5 = transforms.RandomAffine((180-0.5,180+0.5))
    transform6 = transforms.RandomAffine((240-0.5,240+0.5))
    transform7 = transforms.RandomAffine((300-0.5,300+0.5))
    
    #L2级别
    transform8 = transforms.RandomAffine((0-0.5,0.5))
    transform9 = transforms.RandomAffine(0, (0.1,0))             #水平和垂直随机平移，然后都平移两次
    transform10 = transforms.RandomAffine(0, (0,0.1))
    transform11 = transforms.RandomAffine(0, (0.1,0.1))
    transform12 = transforms.RandomAffine(0, (0.1,0.1))
    
    #L3级别
    transform13 = transforms.ColorJitter(brightness=0.15, contrast=0.05, saturation=0.05, hue=0)
    transform14 = transforms.ColorJitter(brightness=0.05, contrast=0.35, saturation=0.05, hue=0)
    transform15 = transforms.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.10, hue=0)
    transform16 = transforms.ColorJitter(brightness=0.0, contrast=0.0, saturation=0.0, hue=(0.049,0.051))
    transform17 = transforms.ColorJitter(brightness=0.0, contrast=0.0, saturation=0.0, hue=(-0.051,0.049))

    #开始进行增强循环
    type_list=os.listdir(mother_trace)
    for genre in type_list:                                     #每个文件名
        name_list=os.listdir(mother_trace+genre+'/')
        
        # 如果该类别没有图片，跳过
        if not name_list:
            print(f"警告: 类别 {genre} 没有图片，跳过增强")
            continue
            
        for name in name_list:
            split_name=os.path.splitext(name)                         #文件名分割的字符串列表
            image_path=mother_trace+genre+'/'+name
            
            #开始增强并保存
            for i in range(1,18):
                image= Image.open(image_path)
                transform=locals()['transform'+str(i)]
                image_new=transform(image)
                image_new.save(mother_trace+genre+'/'+split_name[0]+"_t"+str(i)+split_name[1])
            '''
            for i in range(0,360): #每张图片均匀旋转
                image= Image.open(image_path)
                image_new=transform3(image)
                image_new.save(mother_trace+genre+'/'+split_name[0]+"_t"+str(i)+split_name[1])
            '''
            print(str(name)+"增强完毕！")       
    print("Pictures Fortified")

#类读取器：读取所有的类名称,母文件夹路径要带/
def class_name_getter(mother_trace):
    class_name_list=os.listdir(mother_trace)
    
    #打印每个数对应的类别名，方便查看和debug
    print("类别名获取完毕,按顺序分别为：")
    for i in range(0,len(class_name_list)):
        print(str(np.eye(len(class_name_list))[i])+"对应"+str(class_name_list[i]))
    
    return class_name_list,len(class_name_list)

#------------------------------数据集的制作---------------------------------
#数据集类
class MyDataset(Dataset):
    def __init__(self, x_data,y_data):
        #传入测试集和验证集，都是张量！
        self.xdata=x_data
        self.ydata=y_data
    
    def __getitem__(self, index):
        # 根据索引返回数据
        return self.xdata[index],self.ydata[index]
    
    def __len__(self):
        # 返回数据的长度
        return self.xdata.shape[0]

#-------------------------训练集均值和方差的计算-----------------------
#一定要清洗完毕的数据集才可以使用！
def normalizer(mother_trace):
    type_list=os.listdir(mother_trace)            #文件名列表
    
    # 统计总图片数
    total_images = 0
    for genre in type_list:
        name_list = os.listdir(mother_trace + genre + '/')
        total_images += len(name_list)
    
    if total_images == 0:
        raise ValueError("训练数据集为空！")
    
    #平均值的获取
    for k,genre in enumerate(type_list):                                     #每个文件名
        name_list=os.listdir(mother_trace+genre+'/')
        
        # 跳过空类别
        if not name_list:
            continue
        
        for i,name in enumerate(name_list):
            print("均值计算次数：（{}，{}）".format(str(k),str(i)))
            image_path=mother_trace+genre+'/'+name
            image= Image.open(image_path)
            image=np.array(image)
            image=image.transpose(2,0,1) #h,w,c 2 c,h,w
        
            #首次循环创建空数组
            if i==0 and k==0:
                shape_tuple=image.shape
                somme1=np.zeros(shape_tuple)
                somme1+=image/1000 #数据放缩1000倍，避免爆内存
            else:
                somme1+=image/1000
    aver=somme1/total_images*1000  # 使用总图片数
    
    #方差的获取
    for k,genre in enumerate(type_list):                                     #每个文件名
        name_list=os.listdir(mother_trace+genre+'/')
        
        # 跳过空类别
        if not name_list:
            continue
        
        for i,name in enumerate(name_list):
            print("方差计算次数：（{}，{}）".format(str(k),str(i)))
            image_path=mother_trace+genre+'/'+name
            image= Image.open(image_path)
            image=np.array(image)
            image=image.transpose(2,0,1) #h,w,c 2 c,h,w
        
            #首次循环创建空数组
            if k==0 and i==0:
                somme2=np.zeros(shape_tuple)
                somme2+=(image/1000-aver/1000)**2 #数据放缩1000倍，避免爆内存
            else:
                somme2+=(image/1000-aver/1000)**2
    std_err=np.sqrt(somme2/total_images)*1000  # 使用总图片数
    
    #变为batch直接相减可以接受的形状和类型：
    aver=aver.reshape(1,shape_tuple[0],shape_tuple[1],shape_tuple[2])
    std_err=std_err.reshape(1,shape_tuple[0],shape_tuple[1],shape_tuple[2])
    aver=torch.tensor(aver)
    std_err=torch.tensor(std_err)
    
    #输出环节，输出平均的“图片”和标准差的图片
    print("Statistical Calculation Complete!")
    return aver,std_err

#------------------------------------------------------------------
#训练数据集制作器,将保存所有数据的路径（而非本身）及其对应的标签，都在cpu上
def train_data_devider(mother_trace,batch_size=64):
    data_path=[]
    data_target=[]
    
    #开始存储路径
    type_list=os.listdir(mother_trace)            #文件名列表
    type_number=len(type_list)                    #这个函数是按首字母顺序排的，理论上不会出错
    
    # 创建类别名称到索引的映射
    class_to_idx = {genre: j for j, genre in enumerate(type_list)}
    
    # 记录有数据的类别
    classes_with_data = []
    
    for j,genre in enumerate(type_list):                                     #每个文件名
        name_list=os.listdir(mother_trace+genre+'/')
        
        # 如果该类别没有图片，跳过但记录
        if not name_list:
            print(f"警告: 类别 {genre} (索引 {j}) 没有训练数据")
            continue
        
        # 记录有数据的类别
        classes_with_data.append(genre)
        
        for i,name in enumerate(name_list):
            image_path=mother_trace+genre+'/'+name 
            
            #添加路径和标签
            data_path.append(image_path) #这是一个字符串列表，不用转换为array
            data_target.append(np.eye(type_number)[j])
    
    #检查是否有数据
    if not data_path:
        raise ValueError("训练数据集为空！")
    
    print(f"训练数据统计: 总类别数={type_number}, 有数据的类别数={len(classes_with_data)}, 总样本数={len(data_path)}")
    
    #转换为numpy数组
    data_target=np.array(data_target)
    data_path=np.array(data_path)
    
    #制作数据集对象
    train_dataset = MyDataset(data_path,data_target)
    num_workers = _recommended_dataloader_workers()
    pin_memory = torch.cuda.is_available()
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    
    #输出
    return train_dataset,train_dataloader

#------------------------------------------------------------------
#测试数据集制作器,将保存所有数据的路径（而非本身）及其对应的标签，都在cpu上
#这里的mother_trace是单独的./test/文件夹,和前面不一样,一定要加/
def test_data_devider(mother_trace):
    data_path=[]
    data_target=[]
    
    #开始存储路径
    type_list=os.listdir(mother_trace)            #文件名列表
    type_number=len(type_list)                    #这个函数是按首字母顺序排的，理论上不会出错
    testing_sum=0
    
    # 记录有数据的类别
    classes_with_data = []
    
    for j,genre in enumerate(type_list):                                     #每个文件名
        name_list=os.listdir(mother_trace+genre+'/')
        
        # 如果该类别没有图片，跳过但记录
        if not name_list:
            print(f"警告: 类别 {genre} (索引 {j}) 没有测试数据")
            continue
        
        # 记录有数据的类别
        classes_with_data.append(genre)
        
        for i,name in enumerate(name_list):
            image_path=mother_trace+genre+'/'+name 
            
            #添加路径和标签
            data_path.append(image_path) #这是一个字符串列表，不用转换为array
            data_target.append(np.eye(type_number)[j])
            testing_sum+=1
    
    #检查是否有数据
    if not data_path:
        raise ValueError("测试数据集为空！")
    
    print(f"测试数据统计: 总类别数={type_number}, 有数据的类别数={len(classes_with_data)}, 总样本数={testing_sum}")
    
    #转换为numpy数组
    data_target=np.array(data_target)
    data_path=np.array(data_path)
    
    #制作数据集对象,批的大小是整个数据集的大小
    test_dataset = MyDataset(data_path,data_target)
    test_dataloader = DataLoader(dataset=test_dataset,batch_size=testing_sum+1, shuffle=True,drop_last=False)
    
    #输出
    return test_dataset,test_dataloader

#路径批-图像数组批转化，全都在cpu上--------------------------------------
#每一批batch的训练集为1D字符串列表，标签为批大小*标签向量长度，因此是2D的
def path2image(batch,pepper=0.1,avatar=False):
    #添加图片
    batch_photo=[]
    for lieu in batch:
        image= Image.open(lieu)
        image=np.array(image).astype(np.float32)
        
        #添加椒盐噪声
        array_pepper=np.random.choice([-1,0,1],size=image.shape,p=[pepper,1-2*pepper,pepper])
        image+=array_pepper
        
        #交换颜色信道
        if avatar:
            shuffle_index=[0,1,2]
            shuffled_index=cp.deepcopy(shuffle_index)
            random.shuffle(shuffled_index)
            image[:,:,shuffle_index]=image[:,:,shuffled_index]
        else:
            pass
        
        batch_photo.append(np.array(image))
    batch_photo=np.array(batch_photo)
    
    #图片转化为正确的pytorch张量格式
    #batch_photo=batch_photo.astype(np.float32)
    batch_photo=torch.tensor(batch_photo) #转化为torch张量
    batch_photo=batch_photo.permute(0,3,1,2)# batch,h,w,c -> batch,c,h,w
    batch_photo=batch_photo.float()
    
    #图片的归一化操作(不写在这个函数里)
    
    return batch_photo

#-------------------------自定义多分类交叉熵------------------
def multi_cross_entropy(logits, y):
    max_values,max_indices=torch.max(logits,dim=1,keepdim=True)
    logits=logits-max_values
    s = torch.exp(logits)+1.e-21 #防止爆炸
    logits = s / torch.sum(s, dim=1, keepdim=True)
    entropie = -(y * torch.log(logits)).sum(dim=-1)
    return torch.mean(entropie)


#------------------------Fonction d'entraînement----------------
'''
training_path：训练集母文件夹路径,带/
testing_path：兼容旧接口保留，当前分类训练不再使用独立验证集
saving_path:保存集文件夹路径,带/
net:网络...
'''

def training(training_path,testing_path,saving_path,net,pepper,avatar,Epochs=20,lr=0.001,batch_size=64):
    ''' 
    #求均值和方差
    aver,std_err=normalizer(training_path)
    np.save(saving_path+"aver.npy",aver)
    np.save(saving_path+"std_err.npy",std_err)
    
    #读取均值和方差
    aver=np.load(saving_path+"aver.npy")
    std_err=np.load(saving_path+"std_err.npy")
    '''
    #读取时间
    current_time=str(time.localtime().tm_year) + \
        "_" + str(time.localtime().tm_mon) + \
            "_" + str(time.localtime().tm_mday) + \
                "_" + str(time.localtime().tm_hour) + \
                    "_" + str(time.localtime().tm_min) + \
                        "_" + str(time.localtime().tm_sec)
    
    # 划分训练数据集，当前分类训练不再使用独立验证集
    train_dataset,train_dataloader=train_data_devider(training_path,batch_size=batch_size)
    
    # 打印设备信息
    print(f"\n=== 训练配置 ===")
    print(f"使用设备: {device}")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU名称: {torch.cuda.get_device_name(0)}")
        print(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    print(f"训练轮数: {Epochs}")
    print(f"学习率: {lr}")
    print(f"批次大小: {batch_size}")
    print(f"训练数据集大小: {len(train_dataset)}")
    print("验证集: 不使用")
    print(f"================\n")
    
    #预装载
    net=net.to(device)
    net.train()
    print(f"模型已加载到 {device} 设备")
    
    # 开始训练，预定义辅助量
    step = 0
    Steps = []
    Loss_Steps = []
    Count_Epochs = []
    Loss_Epochs = []
    Err_Epochs = []
    Nan = False
    old_err = float("inf")
    loss_func = multi_cross_entropy
    optimizer = torch.optim.Adam(
        net.parameters(),
        lr,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=lr/10,
        amsgrad=True,
    )

    print("开始进入训练主循环")

    for epoch in range(Epochs):
        if Nan:
            print("检测到 NaN，提前停止训练")
            break

        epoch_losses = []

        for i, (batch, target) in enumerate(train_dataloader):
            net.train()

            epice=np.random.uniform(0,0.1,1)[0]*int(pepper)
            data=path2image(batch,pepper=epice,avatar=avatar)
            data=(data-127.5)/127.5
            data=data.float()

            data=data.to(device)
            target=target.to(device)

            prediction = net(data)
            loss = loss_func(prediction,target).float()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_value = float(loss.detach().cpu().numpy())
            if loss_value != loss_value:
                Nan = True
                break

            epoch_losses.append(loss_value)
            Loss_Steps.append(loss_value)
            Steps.append(step)
            step += 1

        if not epoch_losses:
            print(f"Epoch {epoch+1} 没有有效训练步，提前结束")
            break

        avg_train_loss = float(np.mean(epoch_losses))
        # 保持旧日志格式兼容 UI 解析，但这里不再代表独立验证误差
        err = avg_train_loss
        log_message = (
            f"Training Loss: {np.log10(avg_train_loss):.8f} \t "
            f"Training Steps: {len(epoch_losses)} \t "
            f"Prediction Error: {np.log10(err):.8f} \t Epoch {epoch+1}"
        )
        print(log_message)
        import sys
        sys.stdout.flush()

        with open(saving_path+current_time+".txt",'a') as f:
            if epoch == 0:
                f.write("Batch Size:"+str(batch_size)+", lr:"+str(lr)+"\n")
            f.write(
                "Training Loss:"+"%.8f"%np.log10(avg_train_loss)+
                "\t Training Steps:"+str(len(epoch_losses))+
                "\t Prediction Error:"+str('%.8f'%np.log10(err))+
                "\t Epoch"+str(epoch+1)+"\n"
            )
            f.write("Prediction Error:"+"%.8f"%np.log10(err)+"\t Epoch"+str(epoch+1)+"\n")

        if epoch % 20 == 0 or epoch == Epochs - 1:
            ckpt={
                'epoch': epoch,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_train_loss,
            }
            torch.save(ckpt,saving_path+"Model_"+current_time+".saving")
            if old_err >= avg_train_loss:
                torch.save(ckpt,saving_path+"Model_best_"+current_time+".saving")
                old_err=avg_train_loss

        Count_Epochs.append(epoch)
        Loss_Epochs.append(avg_train_loss)
        Err_Epochs.append(err)
    
    #返回集合
    return Steps,Loss_Steps,Count_Epochs,Loss_Epochs,Err_Epochs


#---------------------------主函数------------------------------------
def _parse_bool_arg(value):
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Invalid boolean value: {value}")


def main():
    import argparse
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='ResNet Training')
    parser.add_argument('--training_path', type=str, default='data/classification/train/', help='Training dataset path')
    parser.add_argument('--testing_path', type=str, default='data/classification/val/', help='Testing dataset path')
    parser.add_argument('--saving_path', type=str, default='data/classification/log/', help='Saving path for models and logs')
    parser.add_argument('--target_size', type=tuple, default=(224, 224), help='Target image size')
    parser.add_argument('--target_format', type=str, default='.jpg', help='Target image format')
    parser.add_argument('--pepper', type=_parse_bool_arg, default=True, help='Whether to add pepper noise')
    parser.add_argument('--avatar', type=_parse_bool_arg, default=True, help='Whether to shuffle color channels')
    parser.add_argument('--epochs', type=int, default=2001, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-6, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--pretrained_model', type=str, default='', help='Path to pretrained ResNet model')
    parser.add_argument('--imbalance', type=_parse_bool_arg, default=True, help='Whether to handle class imbalance via oversampling')
    
    args = parser.parse_args()
    
    #超参数
    NAME="resnet_train"
    training_path=args.training_path
    testing_path=args.testing_path
    saving_path=args.saving_path  #断点保存
    target_size=args.target_size
    target_format=args.target_format
    pepper=args.pepper
    avatar=args.avatar
    epochs=args.epochs
    lr=args.lr
    batch_size=args.batch_size
    enable_imbalance=args.imbalance
    
    # 确保保存路径存在
    import os
    os.makedirs(saving_path, exist_ok=True)
    
    normalized_training_path = os.path.abspath(training_path)
    normalized_testing_path = os.path.abspath(testing_path)

    # 清洗数据集；当训练集和测试集复用同一目录时只处理一次，避免重复改写同一批文件
    train_data_preparer(training_path,target_size=target_size,target_format=target_format)
    if normalized_testing_path != normalized_training_path:
        train_data_preparer(testing_path,target_size=target_size,target_format=target_format)
    else:
        print("训练集与测试集目录相同，跳过重复数据清洗")
        
    # 处理类别不平衡：过采样少数类（如果启用）
    if enable_imbalance:
        oversample_minority_classes(training_path, ir_threshold=10, target_ratio=0.5)
        
    #增强训练数据集
    fortifier(training_path)
    
    # 加载预训练的resnet模型
    resnet_model = None
    
    # 获取类别信息
    class_type,class_num=class_name_getter(training_path)
    
    # 尝试从预训练模型文件中确定模型类型
    if args.pretrained_model and os.path.exists(args.pretrained_model):
        try:
            checkpoint = torch.load(args.pretrained_model, weights_only=False)
            
            # 检查权重格式
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            else:
                # 尝试作为完整模型加载
                if hasattr(checkpoint, 'state_dict'):
                    state_dict = checkpoint.state_dict()
                elif isinstance(checkpoint, dict) and 'model' in checkpoint:
                    state_dict = checkpoint['model'].state_dict()
                else:
                    state_dict = checkpoint
            
            # 根据权重键和形状判断模型类型（创建时不用预训练权重）
            if 'layer3.6.conv1.weight' in state_dict:
                # ResNet-101
                print("检测到ResNet-101权重，使用ResNet-101模型")
                resnet_model = models.resnet101(weights=None)
            elif 'layer3.5.conv1.weight' in state_dict:
                # 检查conv1的形状来判断是ResNet-50还是ResNet-34
                conv1_weight = state_dict.get('layer3.5.conv1.weight')
                if conv1_weight is not None:
                    if conv1_weight.shape[2] == 1 and conv1_weight.shape[3] == 1:
                        # Bottleneck (1x1 conv) - ResNet-50
                        print("检测到ResNet-50权重，使用ResNet-50模型")
                        resnet_model = models.resnet50(weights=None)
                    else:
                        # BasicBlock (3x3 conv) - ResNet-34
                        print("检测到ResNet-34权重，使用ResNet-34模型")
                        resnet_model = models.resnet34(weights=None)
                else:
                    # 默认使用ResNet-50
                    print("无法检测模型类型，默认使用ResNet-50模型")
                    resnet_model = models.resnet50(weights=None)
            elif 'layer3.1.conv1.weight' in state_dict:
                # ResNet-18
                print("检测到ResNet-18权重，使用ResNet-18模型")
                resnet_model = models.resnet18(weights=None)
            else:
                # 尝试通过其他层的权重形状来判断
                if 'layer1.0.conv1.weight' in state_dict:
                    conv1_weight = state_dict['layer1.0.conv1.weight']
                    if conv1_weight.shape[2] == 3 and conv1_weight.shape[3] == 3:
                        # BasicBlock (3x3 conv) - ResNet-18 or ResNet-34
                        if 'layer3.5.conv1.weight' in state_dict:
                            print("检测到ResNet-34权重，使用ResNet-34模型")
                            resnet_model = models.resnet34(weights=None)
                        else:
                            print("检测到ResNet-18权重，使用ResNet-18模型")
                            resnet_model = models.resnet18(weights=None)
                    else:
                        # Bottleneck (1x1 conv) - ResNet-50 or ResNet-101
                        if 'layer3.6.conv1.weight' in state_dict:
                            print("检测到ResNet-101权重，使用ResNet-101模型")
                            resnet_model = models.resnet101(weights=None)
                        else:
                            print("检测到ResNet-50权重，使用ResNet-50模型")
                            resnet_model = models.resnet50(weights=None)
                else:
                    # 默认使用ResNet-50
                    print("无法检测模型类型，默认使用ResNet-50模型")
                    resnet_model = models.resnet50(weights=None)
            
            # 先替换 fc 层为当前数据集的类别数
            num_ftrs = resnet_model.fc.in_features
            resnet_model.fc = torch.nn.Linear(num_ftrs, class_num)

            # 迁移学习时跳过旧任务的分类头，只加载共享骨干权重
            filtered_state_dict = {
                key: value
                for key, value in state_dict.items()
                if key not in {"fc.weight", "fc.bias"}
            }
            missing_keys, unexpected_keys = resnet_model.load_state_dict(filtered_state_dict, strict=False)
            print(f"成功加载预训练模型: {args.pretrained_model}")
            if missing_keys:
                print(f"跳过并重建的层: {missing_keys}")
            if unexpected_keys:
                print(f"未使用的检查点参数: {unexpected_keys}")
        except Exception as e:
            print(f"加载预训练模型失败: {e}")
            # 加载失败后仍需替换fc层
            num_ftrs = resnet_model.fc.in_features
            resnet_model.fc = torch.nn.Linear(num_ftrs, class_num)
            print("将使用随机初始化的模型继续训练...")
    else:
        # 默认使用ResNet-50
        print("未提供预训练模型，默认使用ResNet-50模型")
        resnet_model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        # 默认模型需要替换fc层
        num_ftrs = resnet_model.fc.in_features
        resnet_model.fc = torch.nn.Linear(num_ftrs, class_num)
    
    Steps,Loss_Steps,Count_Epochs,Loss_Epochs,Err_Epochs=training(
        training_path,
        testing_path,
        saving_path,
        resnet_model,
        pepper,
        avatar,
        Epochs=epochs,
        lr=lr,
        batch_size=batch_size,
    )
    torch.save(
        {
            'epoch': epochs - 1,
            'model_state_dict': resnet_model.state_dict(),
            'steps': Steps,
            'loss_steps': Loss_Steps,
            'count_epochs': Count_Epochs,
            'loss_epochs': Loss_Epochs,
            'err_epochs': Err_Epochs,
        },
        os.path.join(saving_path, "Model_last.saving"),
    )
    '''
    #作图观察
    #误差随训练步数的关系
    plt.plot(Steps,Loss_Steps,color='b',label="Training Loss")
    plt.title("Loss with Steps")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(saving_path+"Loss with Steps.png")
    plt.show()

    #误差随Epoch的关系
    plt.plot(Count_Epochs,Loss_Epochs,color='b',label="Training Loss")
    plt.plot(Count_Epochs,Err_Epochs,color='r',label="Verification Error")
    plt.title("Loss and error with Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss or Error")
    plt.legend()
    plt.savefig(saving_path+"Loss with Steps.png")
    plt.show()
    '''
#---------------------------执行部分----------------------------------
if __name__=="__main__":
    main()
