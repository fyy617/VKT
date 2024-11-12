#  🎇 VKA 🎇
VKA: Vision KAN-Attention Model for Disease Diagnosis and Preoperative Prediction




# 📌Installation📌
* `pip install torch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu117`
* `pip install packaging`
* `pip install timm==0.4.12`
* `pip install pytest chardet yacs termcolor`
* `pip install submitit tensorboardX`
* `pip install triton==2.0.0`
* `pip install causal_conv1d==1.0.0  # causal_conv1d-1.0.0+cu118torch1.13cxx11abiFALSE-cp38-cp38-linux_x86_64.whl`
* `pip install mamba_ssm==1.0.1  # mmamba_ssm-1.0.1+cu118torch1.13cxx11abiFALSE-cp38-cp38-linux_x86_64.whl`
* `pip install scikit-learn matplotlib thop h5py SimpleITK scikit-image medpy yacs`
## 📜Other requirements📜:
* Linux System
* NVIDIA GPU
* CUDA 12.0+


# 📊Datasets📊
##  Brain
The Brain Tumor Dataset is a publicly available dataset published on Kaggle as the "Brain Dataset." This dataset consists of brain scan images from patients diagnosed with brain tumors and healthy individuals. The dataset is divided into training and testing sets, with separate files for features and labels. It contains a total of 4,600 images, categorized into two classes:
Brain Tumor: 2,513 images of brain scans from patients diagnosed with brain tumors.
Healthy: 2,087 images of brain scans from healthy individuals.

## Brain MRI
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
We use batch size of 4096 by default and we show how to train models with 8 GPUs. For multi-node training, adjust `--grad-accum-steps` according to your situations.

## Test
