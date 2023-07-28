# lecture 7：训练神经网络

## 1.前面的知识点回顾：

### normalization：



### batch normalization：

![image-20230725154548497](C:\Users\dell\AppData\Roaming\Typora\typora-user-images\image-20230725154548497.png)

#### 参数的范围选定：

## 2.进入正题：对于求解下山路线的算法优化

### SGD的缺点：

#### zigzag

#### 卡在局部最小值或者鞍点（梯度为零的地方）

### answer：

#### SGD+momentum 

![image-20230725162646790](C:\Users\dell\AppData\Roaming\Typora\typora-user-images\image-20230725162646790.png)

```python
# SGD 
while True:
dx = compute_gradient(x)
x+= learning_rate * dx
```

```python
# SGD + momentum
VX=0
while True:
dx = compute_gradient(x)
vx =rho*vx+dx
x += learning_rate * vx
```

#### Nesterov Momentum

```python
# Nesterov Momentum
dx = compute_gradient(x)
old_v=v
v= rho*v- learning_rate *dx
x+=-rho*old_v +(1 +rho)*v
```

![image-20230725162930897](C:\Users\dell\AppData\Roaming\Typora\typora-user-images\image-20230725162930897.png)

#### AdaGrad

#### RMSProp

![image-20230725171411131](C:\Users\dell\AppData\Roaming\Typora\typora-user-images\image-20230725171411131.png)

**效果：**

![image-20230725171554136](C:\Users\dell\AppData\Roaming\Typora\typora-user-images\image-20230725171554136.png)

#### 集大成者：AdaM

![image-20230725172759078](C:\Users\dell\AppData\Roaming\Typora\typora-user-images\image-20230725172759078.png)

### learning rate optimization

![image-20230725173433231](C:\Users\dell\AppData\Roaming\Typora\typora-user-images\image-20230725173433231.png)

### 二阶优化 L-BFGS

#### 牛顿法

#### 拟牛顿法

### model  ensemble（模型集成）

模型集成是融合多个训练好的模型，基于某种方式实现测试数据的多模型融合，这样来使最终的结果能够“取长补短”，融合各个模型的学习能力，提高最终模型的泛化能力。

指数衰减，polyak平均等。

## 3.正则化的优化：

#### 传统方法

#### dropout

随机的将激活函数的输出/卷积层映射置为0.

![image-20230726091032737](C:\Users\dell\AppData\Roaming\Typora\typora-user-images\image-20230726091032737.png)

通过数学期望来增加稳定性：

![image-20230726091740947](C:\Users\dell\AppData\Roaming\Typora\typora-user-images\image-20230726091740947.png)

```python
def predict(X):
# ensembled forward pass
Hl = np.maximum(0,np.dot(wl, X) + bl) * p # NOTE: scale the activations
H2 = np.maximum(0,np.dot(W2，H1) + b2) * p # NOTE: scale the activations
out = np.dot(W3，H2) + b3
```

#### Data argumentation

将原始图片在tag保持不变的情况下进行裁剪、翻转、色彩变化等一系列的操作。

#### DropConnect

将weight矩阵置0而不是激活函数

![image-20230726094724192](C:\Users\dell\AppData\Roaming\Typora\typora-user-images\image-20230726094724192.png)

#### Fractional Max Pooling

通过平均或者一些方法来最大化的消除消除

#### Stochastic Depth

随机去掉一些层

![image-20230726094616244](C:\Users\dell\AppData\Roaming\Typora\typora-user-images\image-20230726094616244.png)

## Transfer Learning

当你在训练一个模型的时候没有足够的数据量支撑，很可能会出现过拟合的情况。transfer learning就是在这个背景下诞生的。

**总结下来很简单，就是“抄作业”**

当你的数据量不够去训练出一个比较好的模型的时候，把别人用大数据集训练出的参数直接拿过来，然后根据自己的需要微调最后的几层而让前面的绝大多数层保持不变，因为别人的训练数据已经足够大，所以模型中的参数是有很强的泛型适应能力的，只需要抄作业然后微调即可。

但是注意，我们抄作业的时候要抄大致相同的类型，比如你在做一个动物种类分类的模型，而你抄了一个医学CT片子训练出的东西。想一想也知道这是一种很傻逼的行为。

我们熟知的Caffe，tensorflew与pyTorch都是这些模型库(model zoo)。