# Toxic Comment Classification

## 主要功能
采用LSTM/C-LSTM/CNN等多种方法，对评论进行多标签分类

## 代码结构

### main

- main.py  
无交叉验证，取训练误差最小的参数，用于测试  

- main_fold.py  
k折交叉验证，测试时取k次的平均值  

### dataset

toxic comment

### net

lstm/c-lstm/cnn...

## 运行

无交叉验证的运行指令如：
```
python main.py \
--server=6099 \
--phase=Train \
--sen_len=250 \
--net_name=lstm \
--dir_date=20180317 \
--lr_base=1e-3 \
--batch_size=256 \
--gpu=0 \
```

有交叉验证的指令如：
```
python main_fold.py \
--server=6099 \
--phase=Train \
--sen_len=250 \
--net_name=lstm \
--dir_date=20180318_fold_20 \
--batch_size=256 \
--lr_base=1e-3 \
--gpu=1 \
--fold_count=10 \
```