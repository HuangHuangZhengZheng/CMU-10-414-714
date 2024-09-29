# Normalization and Regularization

## Normalization and Initialization

![alt text](image.png)
注意看weight variance的曲线，几乎不变

norm的思想来源
![alt text](image-1.png)
- layer normalization
- batch normalization
![alt text](image-2.png)
这么看来batch_norm确实很奇怪, odd! :cry:
![alt text](image-3.png)

## Regularization
### L2 Regularization
针对的是过拟合?但是只要是减少function class的操作都是regularization的一种

![alt text](image-4.png)
然后发现weight decay和regularization有联系！

### dropout
![alt text](image-5.png)
![alt text](image-6.png)