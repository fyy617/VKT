#  🎇 VKT 🎇
VKT: A Hybrid Vision KAN-Transformer Model for Medical Image Classification

The VKT architecture synergistically integrates KAN and Transformer within a dual-branch structure: the ConvKan branch and the LG Attention branch. The ConvKan branch leverages a convolutional BCBR module and TCKan module to strengthen cross-channel interactions via parallel Token Kan and Channel Kan operations. Meanwhile, the LG Attention branch introduces an innovative combination of Local Attention and Global Attention mechanisms, effectively capturing fine-grained structures and global semantic information in images.

![image](https://github.com/user-attachments/assets/a01eb641-f08e-4645-b15c-afcc3167fed4)

![image](https://github.com/user-attachments/assets/4d4fb17f-cf30-4669-be7e-ffb29f270c61)

# 📌Installation📌
* `pip install packaging`
* `pip install timm==0.4.12`
* `pip install pytest chardet yacs termcolor`
* `pip install submitit tensorboardX`
* `pip install triton==2.0.0`
* `pip install addict==2.4.0`
* `pip install dataclasses`
* `pip install pyyaml`
* `pip install albumentations`
* `pip install tensorboardX`
* `pip install scikit-learn matplotlib thop h5py SimpleITK scikit-image medpy yacs`
* `pip install torch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu117`


## 📜Other requirements📜
* Linux System
* NVIDIA GPU
* CUDA 12.0+


# 📊Datasets📊
The dataset format is as follows：
```
├── datasets
│   ├── Brain
│     ├── class1
│           ├── 1.png
|           ├── ...
|     ├── class2
│           ├── 1.png
|           ├── ...
│   ├── Brain MRI
│     ├── class1
│           ├── 1.png
|           ├── ...
|     ├── class2
│           ├── 1.png
|           ├── ...
│     ├── class3
│           ├── 1.png
|           ├── ...
|     ├── class4
│           ├── 1.png
|           ├── ...
│   ├──...
```



##  Brain Tumor Dataset 
The Brain Tumor Dataset is a publicly available dataset published on Kaggle as the "Brain Dataset." This dataset consists of brain scan images from patients diagnosed with brain tumors and healthy individuals. The dataset is divided into training and testing sets, with separate files for features and labels. It contains a total of 4,600 images, categorized into two classes:
Brain Tumor: 2,513 images of brain scans from patients diagnosed with brain tumors.
Healthy: 2,087 images of brain scans from healthy individuals.

## Brain Tumor Classification (MRI)
This dataset is a publicly available dataset from Kaggle. It is a four-classification dataset of brain tumor MRI released by Sartaj Bhuvaji et al. of the National Institute of Technology Durgapur, India in July 2020. The dataset is divided into the following four categories: 926 images of glioma tumor, 937 images of meningioma tumor, 396 images of no tumor, and 901 images of pituitary tumor, a total of 3160 images.

![image](https://github.com/user-attachments/assets/3f885038-2601-457a-aad2-e8c46b45eb87)
(https://www.kaggle.com/datasets/preetviradiya/brian-tumor-dataset)

## CPN X-ray  (Covid19-Pneumonia-Normal Chest X-Ray Images)
Shastri et al collected a large number of publicly available and domain recognized X-ray images from the Internet, resulting in CPN-CX. The CPN-CX dataset is divided into 3 categories, namely COVID, NORMAL and PNEUMONIA. All images are preprocessed and resized to 256x256 in PNG format. It helps the researcher and medical community to detect and classify COVID19 and Pneumonia from Chest X-Ray Images using Deep Learning [Dataset URL](https://data.mendeley.com/datasets/dvntn9yhd2/1).![imgs_02](https://github.com/YubiaoYue/MedMamba/assets/141175829/996035b3-2dd5-4c01-b3d4-656f2bf52307)
(https://data.mendeley.com/datasets/dvntn9yhd2/1)

## Kvasir
The data is collected using endoscopic equipment at Vestre Viken Health Trust (VV) in Norway. The VV consists of 4 hospitals and provides health care to 470.000 people. One of these hospitals (the Bærum Hospital) has a large gastroenterology department from where training data have been collected and will be provided, making the dataset larger in the future. Furthermore, the images are carefully annotated by one or more medical experts from VV and the Cancer Registry of Norway (CRN). The CRN provides new knowledge about cancer through research on cancer. It is part of South-Eastern Norway Regional Health Authority and is organized as an independent institution under Oslo University Hospital Trust. CRN is responsible for the national cancer screening programmes with the goal to prevent cancer death by discovering cancers or pre-cancerous lesions as early as possible.[Kavsir Dataset](https://datasets.simula.no/kvasir/ "Download it") ![imgs_03](https://github.com/YubiaoYue/MedMamba/assets/141175829/b25b3795-7b30-4736-8fb4-f01787158763)
(https://www.kaggle.com/datasets/yasserhessein/the-kvasir-dataset/data)


## Train
To train the model, we used the PyTorch deep learning framework and selected the Adam optimizer to optimize the model parameters. Specifically, we used normalization for data preprocessing, set the initial learning rate to 0.0001, β_1 to the default value of 0.9, β_2 to the default value of 0.999, and used CrossEntropyLoss to calculate the loss function. During training, the batch size was set to 8, and iterative training was performed in a training cycle of 100 epochs.
```
python train.py
```
