# Computer Vision Final Project


## Prerequisites
***
- Linux 
- Python 3.7

## Environment Setup
***
- Create and activate environment :
```
conda create --name ENVIRONMENT python=3.7
conda activate ENVIRONMENT
```
- Install PyTorch and other dependencies :
```
pip install torch torchvision
pip install -r requirements.txt
```
## Training
***
- Train a model :

```
bash train.sh 
```
- Fine-tune :
```
bash train_FT.sh
```

## Evaluation
***
```
bash eval.sh $1 $2
```
- $1 path to the folder containing test files (ex: ./CV22S_Ganzin/dataset/public/S5/)
- $2 path of the output image (ex: ./S5_solution/)


## Citation
***
> S. Back, J. Kim, R. Kang, S. Choi and K. Lee. **Segmenting unseen industrial components in a heavy clutter using rgb-d fusion and synthetic data.** 2020 IEEE International Conference on Image Processing (ICIP). IEEE, 2020. [[Paper]](https://ieeexplore.ieee.org/abstract/document/9190804)
 

```
[1] @inproceedings{back2020segmenting,
  title={Segmenting unseen industrial components in a heavy clutter using rgb-d fusion and synthetic data},
  author={Back, Seunghyeok and Kim, Jongwon and Kang, Raeyoung and Choi, Seungjun and Lee, Kyoobin},
  booktitle={2020 IEEE International Conference on Image Processing (ICIP)},
  pages={828--832},
  year={2020},
  organization={IEEE}
}
```
