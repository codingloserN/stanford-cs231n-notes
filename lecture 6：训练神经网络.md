# lecture 6：训练神经网络

## mini-batch SGD:

### Activation function:

#### Sigmoid:

$$
1/(1+e^-x)
$$
![v2-707f1aa66391f2a838fd3b81c93d45d5_720w](E:\学习资料\课外小芝士\科研训练\cs231n笔记\图片\v2-707f1aa66391f2a838fd3b81c93d45d5_720w.webp)

![image-20230719201732011](C:\Users\dell\AppData\Roaming\Typora\typora-user-images\image-20230719201732011.png)

优点：

1. 不会饱和，他将所有的值都收敛到0-1之间了

sigmoid存在的三个问题：

1. 梯度消失。从函数图像中可以看到，当输入的绝对值比较大时，梯度过于平缓几乎为0。这样对我们反向传播时没有办法下山了，对参数无法进行调优。
2. 不是以0为中心的且函数值恒正，导致在反向传播时梯度要根据上游的值来确定，而上游的值全是正的，对寻找正确梯度时走的弯路比较多，看上图。其实这个地方也没怎么理解。
3. 指数运算相较于线性运算还是稍微有些复杂。

#### Tanh

#### ReLU

leaky-RELU

parameteric-RELU

maxout

### conclusion:

- Use ReLU.
- Be careful with your learning rates
- Try out Leaky ReLU / Maxout / ELUTry out tanh but don't expect much
- Don't use sigmoid

## Date preprocessing

### parameter initilization(weights):

过小：在逐层的相乘操作过后会导致值趋于0，这对我们梯度的反向传播是很不利的。

过大：数值很容易过大，再结合上梯度函数很容易让梯度饱和，即在绝对值大的地方梯度变化很小。

常用方法：`Xavier initialiaztion`.大致就是一个算法通过对权重的处理确保输入的方差等于输出的方差。

### 数据处理(x):

#### 高斯化

$$
X_i-E(X_i) \over {\sqrt{D(X_i)}}
$$

高斯化数据的目的时把数据放到以原点为中心的范围内，这一块是我们设计激活函数时最有效的定义域区域。

![image-20230721170533577](C:\Users\dell\AppData\Roaming\Typora\typora-user-images\image-20230721170533577.png)

#### 设置学习率：

学习率是我们手动设置时最重要的hyper参数：

一般来说学习率在$1*e^{-3}$~$1*e^{-5}$之间。

当学习率过低时，可以发现loss函数的输出值的变化很小。

而当学习率过高时，一般直接会报NaN的错。



## 超参数设定(hyperparameters setting)

(名词解释）**Epoch（时期）：**

当一个完整的数据集通过了神经网络一次并且返回了一次，这个过程称为一次>epoch。（也就是说，所有训练样本在神经网络中都 进行了一次正向传播 和一次反向传播 ）

再通俗一点，一个Epoch就是将所有训练样本训练一次的过程。

然而，当一个Epoch的样本（也就是所有的训练样本）数量可能太过庞大（对于计算机而言），就需要把它分成多个小块，也就是就是分成多个Batch 来进行训练。

#### 随机搜寻（random search）

#### 网格搜寻（grid search）

![image-20230721164930410](C:\Users\dell\AppData\Roaming\Typora\typora-user-images\image-20230721164930410.png)

Bad initialization：

![image-20230721165029690](C:\Users\dell\AppData\Roaming\Typora\typora-user-images\image-20230721165029690.png)

overfitting:

![image-20230721165120352](C:\Users\dell\AppData\Roaming\Typora\typora-user-images\image-20230721165120352.png)
