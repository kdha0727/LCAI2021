# LARYNX CANCER AI DATATHON

> Information for baseline code

<!--

### 1. Preprocessing

* xml to png mask

```bash
$ python pre_processing.py \                                 
--root_folder "path/to/your/tar/archive" \                 
--destination_folder "./data_processed"
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

-->

* How to Run

1. install requirements

```bash
$ pip install -r requirements.txt
```

2. preprocess your files

```bash
$ python pre_processing.py \                                 
--root_folder "path/to/your/unzipped/archive" \
--destination_folder "./data_processed"
```

3. run test scripts

**Preprocessed**

```bash
$ python test.py --data_path "./data_processed/test_set_for_LCAI"
```

**Original**

```bash
$ python test.py --data_path "path/to/your/unzipped/archive/test_set_for_LCAI"
```