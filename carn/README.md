### Requirements
- Python 3
- [PyTorch](https://github.com/pytorch/pytorch) (0.4.0), [torchvision](https://github.com/pytorch/vision)
- Numpy, Scipy
- Pillow, Scikit-image
- h5py
- importlib

### Dataset
1. Download [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K) and unzip on `dataset` directory as below:
  ```
  dataset
  └── DIV2K
      ├── DIV2K_train_HR
      ├── DIV2K_train_LR_bicubic
      ├── 
      	├── DIV2K_valid_HR
      	└── DIV2K_valid_LR_bicubic
  ```
2. To expand the training  dataset

   ```python
   python ./DataAugmentation/dataA.py  
   #修改源代码中的输入路径与输出路径
   #但最终构造 1 中形式的数据集
   ```

3. To accelerate training, we first convert training images to h5 format as follow (h5py module has to be installed).
```shell
$ cd datasets && python div2h5.py
```
### Test Pretrained Models
We provide the pretrained models in `checkpoint` directory. To test CARN on benchmark dataset:
```shell
$ python carn/sample.py 
#修改对应的默认参数即可
```
### Training Models
Here are our settings to train CARN and CARN-M. Note: We use two GPU to utilize large batch size, but if OOM error arise, please reduce batch size.
```shell
$ python carn/train.py --patch_size 64 \
                       --batch_size 64 \
                       --max_steps 600000 \
                       --decay 400000 \
                       --model carn \
                       --ckpt_name carn \
                       --ckpt_dir checkpoint/carn \
                       --scale 4 \
                       --num_gpu 2
```

### 框架来源

本文算法是基于CARN来进行调整的

CARN链接如下：[nmhkahn/CARN-pytorch: Fast, Accurate, and Lightweight Super-Resolution with Cascading Residual Network (ECCV 2018) (github.com)](https://github.com/nmhkahn/CARN-pytorch)

