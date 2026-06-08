import os
import numpy as np
import shutil
import random
import time

#For creating training and testing dataset of ResNet-101
'''
original_dataset:non-augmented unrepaired dataset
containing 2,3,4...
WITHOUT '/'!!!!!!!!!

'''
def devide(original_dataset,ratio=0.05,fois="_"):
    #Split the path of original dataset
    directory, filename = os.path.split(original_dataset)
    classes=os.listdir(original_dataset)
    
    #Create training and testing datasets
    jour=str(time.localtime().tm_mon)+str(time.localtime().tm_mday)
    training_dataset=original_dataset+"_train_"+jour+fois
    testing_dataset=original_dataset+"_test_"+jour+fois
    try:
        os.mkdir(training_dataset)
        os.mkdir(testing_dataset)
    except:
        print("Training and testing folders already existed, begin directly!")
        #shutil.rmtree(training_dataset)
        #shutil.rmtree(testing_dataset)
        #os.mkdir(training_dataset)
        #os.mkdir(testing_dataset) 
    for Sorte in classes:
        for dataset in [training_dataset,testing_dataset]:
            os.mkdir(dataset+"/"+Sorte)
    
    #print("Arrived1")
    #counting testing number
    Nummer=np.zeros(len(classes))
    for i in range(len(classes)):#每一类的测试集个数
        Nummer[i]+=max(1,int(0.9+len(os.listdir(original_dataset+"/"+classes[i]))*ratio))
    
    #randomly choosing files
    for i in range(len(classes)):
        #dividing testing dataset
        pics=os.listdir(original_dataset+"/"+classes[i])
        sampled_elements = random.sample(pics, int(Nummer[i]))
        #replicating pictures
        for pic in pics:
            if pic in sampled_elements:
                shutil.copy(original_dataset+"/"+classes[i]+"/"+pic,
                            testing_dataset+"/"+classes[i]+"/"+pic)
            else:
                shutil.copy(original_dataset+"/"+classes[i]+"/"+pic,
                            training_dataset+"/"+classes[i]+"/"+pic)
    print("Training and testing dataset prepared!")

def kfold(original_dataset,k=10):
    #Split the path of original dataset
    directory, filename = os.path.split(original_dataset)
    classes=os.listdir(original_dataset)

    #对于每一类，不放回地抽取--------------------------------------------
    shazishazi=[]# 种类i，K折，选取的图片
    for i in range(len(classes)):
        pics=os.listdir(original_dataset+"/"+classes[i])
        PIC_Lange=len(pics)
        
        # 如果该类别没有图片，记录空列表并跳过
        if PIC_Lange == 0:
            print(f"警告: 类别 {classes[i]} 没有图片，跳过")
            I_of_K = [[] for _ in range(k)]
            shazishazi.append(I_of_K)
            continue
        
        #开始进行每一类的筛选
        I_of_K=[]
        for klein in range(0,k):
            if klein<PIC_Lange%k:
                sampled_elements = random.sample(pics,PIC_Lange//k+1)
                for Teilchen in sampled_elements:
                    pics.remove(Teilchen)
            else:
                sampled_elements = random.sample(pics,PIC_Lange//k)
                for Teilchen in sampled_elements:
                    pics.remove(Teilchen)
            I_of_K.append(sampled_elements)
        shazishazi.append(I_of_K)
    
    #制作k个集合-------------------------------------------------------
    jour=str(time.localtime().tm_mon)+str(time.localtime().tm_mday)
    for fois in range(0,k):
        #Create training and testing datasets
        training_dataset=original_dataset+"_train_"+jour+"_"+str(fois)
        testing_dataset=original_dataset+"_test_"+jour+"_"+str(fois)
        try:
            os.mkdir(training_dataset)
            os.mkdir(testing_dataset)
        except:
            print("Training and testing folders already existed, begin directly!")
        for Sorte in classes:
            for dataset in [training_dataset,testing_dataset]:
                os.mkdir(dataset+"/"+Sorte)
        
        #开始整体复制操作
        for Suerte in range(len(classes)):
            pics=os.listdir(original_dataset+"/"+classes[Suerte])
            for pic in pics:
                if pic in shazishazi[Suerte][fois]:
                    shutil.copy(original_dataset+"/"+classes[Suerte]+"/"+pic,
                                testing_dataset+"/"+classes[Suerte]+"/"+pic)
                else:
                    shutil.copy(original_dataset+"/"+classes[Suerte]+"/"+pic,
                                training_dataset+"/"+classes[Suerte]+"/"+pic)
        

#--------------------------------主函数------------------------------
def main():
    original_dataset="./data/classification/DATASET1"
    ratio=1/10
    #devide(original_dataset,ratio=ratio) #divide the dataset only once
    kfold(original_dataset,k=10) #k-fold division
    
if __name__=="__main__":
    main()
    
    
    
    
    
    