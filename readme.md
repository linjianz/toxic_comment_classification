# Toxic Comment Classification

[toc]

## 主要功能
采用多种方法，对评论进行多标签分类  

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

