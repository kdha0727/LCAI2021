# LARYNX CANCER AI DATATHON

> Information for baseline code

### 1. Preprocessing

* xml to png mask

```bash
$ python preprocessing.py \
--root_folder=[input_folder] \
--destination_folder=[output_folder] \
--img_height=[target_height] \ 
--img_width=[target_width]
```

![image](https://user-images.githubusercontent.com/92664643/145568215-2f51579f-7aac-41cf-9aa7-5f12e601769e.png)




### 2. Dataset

* `ImageDataset` :   Data Loader 



### 3. Model

* `ResNetUNet`  : Model Network



### 3. Pytorch_SSIM

* loss function 

### 4. Train

* Training code

```bash
$ python train.py \
--data_path=[data_path]\
--batch_size=[batch_size]\
--epoch=[num_epochs]\
--LR=[learning_rate]\
--n_channel=[num_input_channel]\
--img_height=[input_height]\
--img_width=[input_width]
```



