# Computer vision
#Homework #2 : part2

## 1. Student
* R10522606 曾柏翔

## 2. Environment
* python==3.6+
* numpy==1.19.2+
* cyvlfeat==0.5.1+
* matplotlib==3.3.4+
* pillow==8.1.2+
* scipy==1.5.2+
* opencv-python==4.5.1+
* sklearn==0.24.1+
* pytorch==1.5.1+
* torchvision==0.6.1+
* tqdm==4.64.0

## 3. Usage

### 3.1 Get utilize.pt
```
python3 main.py
```

### 3.2 Evaluation
```
python3 eval.py $1 $2
    •$1 the path of model’s parameter
    •$2 the path of the annotation for test data
E.g.
python3 eval.py --path ./save_dir/Resnet/best_model.pt --test_anno ./p2_data/annotations/public_test_annos.json
```

