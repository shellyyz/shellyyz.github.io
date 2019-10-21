---
layout:     post
title:      RNN Captioning (I)
subtitle:   cs231n assignment 3
date:       2018-12-01
author:     Brianne Yao
# description: some common commands in Git.
# header-img: img/post-bg-ios9-web.jpg -->
catalog: true
comments: true
tags:
    - Git
    - Commands
---

{% include head.html %}

本文主要是翻译了cs231n第三次assignment的RNN captioning部分,加入了个人的理解.

### Setup
上来不管三七二十一先进行各种import需要的库以及用到的文件,函数,和类.
设置matplotlib为inline模式.设置plot的大小,位置,和colormap的模式.设定自动重载(并没有懂).

最后定义了一个计算相对误差的函数.

```python
# As usual, a bit of setup
import time, os, json
import numpy as np
import matplotlib.pyplot as plt

from cs231n.gradient_check import eval_numerical_gradient, eval_numerical_gradient_array
from cs231n.rnn_layers import *
from cs231n.captioning_solver import CaptioningSolver
from cs231n.classifiers.rnn import CaptioningRNN
from cs231n.coco_utils import load_coco_data, sample_coco_minibatch, decode_captions
from cs231n.image_utils import image_from_url

%matplotlib inline
plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

# for auto-reloading external modules
# see http://stackoverflow.com/questions/1907993/autoreload-of-modules-in-ipython
%load_ext autoreload
%autoreload 2

def rel_error(x, y):
    """ returns relative error """
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))
```

### Install h5py
用到的COCO数据集是以HDF5形式存储的.要加载HDF5类型的数据,我们需要下载h5py的package.

```
sudo pip install h5py
```

可以通过在命令前加上“！”来直接从Jupyter笔记本中运行命令

```python
!pip install h5py
```

### Microsoft COCO
该数据集是image captioning的标准试验台.数据集含有80000张训练图像和40000张校准图像,每一张图像都有5个标题.COCO数据集下载完后约有1GB.

作业中数据集经过了预处理,提取出了features.所有数据经过用ImageNet预训练过的VGG-16网络的fc7层得到的特征提取. 提取出的数据分别放在了文件 train2014_vgg16_fc7.h5 和 val2014_vgg16_fc7.h5 中.

为了减少运行时间和内存占用,我们将特征维度由4096减少到了512,减少后的特征分别在文件train2014_vgg16_fc7_pca.h5 和 val2014_vgg16_fc7_pca.h5中.

未经处理的数据大概有20GB,我们并没有下载.但所有数据的下载链接分别在train2014_urls.txt 和 val2014_urls.txt 中. 我们可以即时下载图像以进行可视化。由于图像是即时下载的，因此必须连接到互联网才能查看图像.

处理字符串strings比较低效,因此我们用encoded的标题版本.每个词对应一个integer ID,这样我们就可以用一串数字来表示标题. 单词和数字的对应关系在文件coco2014_vocab.json中,你可以用cs231n/coco_utils.py中的decode_captions函数来把整数ID的numpy array转化成strings.

我们还在每个标题的开头和结尾添加了特殊的tokens,在标题开头加了<START> token,在结尾加了<END> token. 不常用单词(vocabulary)中没有的单词用<UNK>来代替. 此外,由于我们想用minibatches训练数据,其中的标题长度不同,我们在<END>后用<NULL>来填充,并不计算<NULL>的loss及gradient.
由于比较繁琐,作业中已经有现成的对特殊tokens的处理细节.我们不必操心.

下面用cs231n/coco_utils.py中的load_coco_data 载入MS-COCO数据(包括标题,特征,链接,以及字典).

```python
# Load COCO data from disk; this returns a dictionary
# We'll work with dimensionality-reduced features for this notebook, but feel
# free to experiment with the original features by changing the flag below.
data = load_coco_data(pca_features=True)

# Print out all the keys and values from the data dictionary
for k, v in data.items():
    if type(v) == np.ndarray:
        print(k, type(v), v.shape, v.dtype)
    else:
        print(k, type(v), len(v))
```

### Look at the data
在用数据集之前,我们最好先看一眼数据集中的一些例子.

我们可以用cs231n/coco_utils.py中的sample_coco_minibatch函数在load_coco_data所返回的数据结构中抽取一撮儿数据.
下面的代码抽取了一小撮儿训练数据,并展示了数据的图像和标题. 多运行几次下面的代码看看数据集大概是神马样子的.

注意,我们用decode_captions函数解码标题,用数据集的Flickr URL即时下载数据,因此必须连网才能看到图像.

```python
# Sample a minibatch and show the images and captions
batch_size = 3

captions, features, urls = sample_coco_minibatch(data, batch_size=batch_size)
for i, (caption, url) in enumerate(zip(captions, urls)):
    plt.imshow(image_from_url(url))
    plt.axis('off')
    caption_str = decode_captions(caption, data['idx_to_word'])
    plt.title(caption_str)
    plt.show()
```

### Recurrent Neural Networks
我们用RNN语言模型来做image captioning.
文件cs231n/rnn_layers.py包含了RNN中用到的不同类型的层的implementations. 文件cs231n/classifiers/rnn.py用这些层来实现Image captioning模型.

我们先来运行RNN中不同类型层的代码cs231n/rnn_layers.py.

### Vanilla RNN: step forward
打开文件cs231n/rnn_layers.py.该文件实现了RNN中常用的不同类型层的forward 和backward 过程.

首先, 编写函数 rnn_step_forward实现了Vanilla RNN中单个timestep的forward pass.

激活函数是tanh.
输入数据维度D,
hidden state维度H,
minibatch size大小为N.

 + 输入:
   - x:  该timestep的输入数据, N*D.
   - prev_h:  前一个timestep的Hidden state, N*H.
   - Wx:  input-to-hidden的权值矩阵, D*H.
   - Wh:  hidden-to-hidden的权值矩阵, H*H.
   - b:  biases, H.
 + 返回值:
   - next_h: 下一个hidden state, N*H.
   - cache: 所有在backward pass中需要用到的量.

```python
def rnn_step_forward(x, prev_h, Wx, Wh, b):
    next_h = np.tanh( x @ Wx + prev_h @ Wh + b )
    cache = (next_h, x, prev_h, Wx, Wh, b)
    return next_h, cache
```

完成后运行下面的代码检查实现效果.误差应该小于e-8.
下面的代码是随机生成的数据. 作业中给出了ground truth的近似值.直接计算相对误差即可.

```python
N, D, H = 3, 10, 4

x = np.linspace(-0.4, 0.7, num=N*D).reshape(N, D)
prev_h = np.linspace(-0.2, 0.5, num=N*H).reshape(N, H)
Wx = np.linspace(-0.1, 0.9, num=D*H).reshape(D, H)
Wh = np.linspace(-0.3, 0.7, num=H*H).reshape(H, H)
b = np.linspace(-0.2, 0.4, num=H)

next_h, _ = rnn_step_forward(x, prev_h, Wx, Wh, b)
expected_next_h = np.asarray([
  [-0.58172089, -0.50182032, -0.41232771, -0.31410098],
  [ 0.66854692,  0.79562378,  0.87755553,  0.92795967],
  [ 0.97934501,  0.99144213,  0.99646691,  0.99854353]])

print('next_h error: ', rel_error(expected_next_h, next_h))
```

### Vanilla RNN: step backward
在文件cs231n/rnn_layers.py中编写rnn_step_backward函数.

vanilla RNN中单个timestep的backward pass.

 + 输入:
  - dnext_h: loss关于下一个Hidden state的梯度, N*H.
  - cache: forward pass返回的量.
 + 返回值:
  - dx: 输入数据的梯度. N*D.
  - dprev_h: 前一个Hidden state的梯度. N*H.
  - dWx: input-to-hidden的权值矩阵的梯度. D*H.
  - dWh: hidden-to-hidden的权值矩阵的梯度. H*H.
  - db: bias向量的梯度. H.

在计算tanh函数的梯度时可以用函数值来表示local derivative, 因为\\( (tanh(x))^' = 1 - tanh^2(x) \\).

**在写代码时候要注意维度匹配,有时矩阵需要转置.**

```python
def rnn_step_backward(dnext_h, cache):
    next_h, x, prev_h, Wx, Wh, b = cache
    # linear means the input of the tanh function
    dlinear = dnext_h * (1 - next_h ** 2)
    dWx = x.T @ dlinear
    dx = dlinear @ Wx.T
    dWh = prev_h.T @ dlinear
    dprev_h = dlinear @ Wh.T
    db = np.sum(dlinear, axis = 0)
    return dx, dprev_h, dWx, dWh, db
```

实现后运行下面的代码进行数值梯度检查.误差应该小于e-8或更小.
下面随机生成数据进行forward pass和backward pass.

```python
from cs231n.rnn_layers import rnn_step_forward, rnn_step_backward
np.random.seed(231)
N, D, H = 4, 5, 6
x = np.random.randn(N, D)
h = np.random.randn(N, H)
Wx = np.random.randn(D, H)
Wh = np.random.randn(H, H)
b = np.random.randn(H)

out, cache = rnn_step_forward(x, h, Wx, Wh, b)

dnext_h = np.random.randn(*out.shape)

fx = lambda x: rnn_step_forward(x, h, Wx, Wh, b)[0]
fh = lambda prev_h: rnn_step_forward(x, h, Wx, Wh, b)[0]
fWx = lambda Wx: rnn_step_forward(x, h, Wx, Wh, b)[0]
fWh = lambda Wh: rnn_step_forward(x, h, Wx, Wh, b)[0]
fb = lambda b: rnn_step_forward(x, h, Wx, Wh, b)[0]

dx_num = eval_numerical_gradient_array(fx, x, dnext_h)
dprev_h_num = eval_numerical_gradient_array(fh, h, dnext_h)
dWx_num = eval_numerical_gradient_array(fWx, Wx, dnext_h)
dWh_num = eval_numerical_gradient_array(fWh, Wh, dnext_h)
db_num = eval_numerical_gradient_array(fb, b, dnext_h)

dx, dprev_h, dWx, dWh, db = rnn_step_backward(dnext_h, cache)

print('dx error: ', rel_error(dx_num, dx))
print('dprev_h error: ', rel_error(dprev_h_num, dprev_h))
print('dWx error: ', rel_error(dWx_num, dWx))
print('dWh error: ', rel_error(dWh_num, dWh))
print('db error: ', rel_error(db_num, db))
```

### Vanilla RNN: forward
上面我们已经实现了vanilla RNN单个timestep的forward pass和backward pass.下面我们将这些片段连接起来,实现处理整个数据序列的RNN.

在文件cs231n/rnn_layers.py中,编写函数rnn_forward.其中要用到之前写的函数rnn_step_forward.

假定输入序列含有T个向量,每个向量维度为D. RNN中Hidden size大小为H, 每个minibatch含有N个序列.函数输出所有timesteps的Hidden states.

**要注意的一点,这里的输入数据x并不是图片,而是N张图片经过预处理之后得到的在D维字典空间中的向量,timestep一共有T个,因此有T个D维向量对应一张图.**

举个例子, 假如字典里有4个词(注意并不是说D=4, 而是V=4, V在后面的函数word_embedding中出现), dic = [cats: 0, dogs: 1, elephants: 2, eat: 3].
输入的x的N=2,也就是有两个句子,分别是 "cats eat dogs" 和 "dogs eat cats".
那么它们对应的向量就分别是 [0 3 1] 和 [1 3 0].

如果D=2, word embedding把所有的integer映射到D维空间中,比方说对应规则如下:
 - cats: 0 -> (2, 3)
 - dogs: 1 -> (3, 2)
 - elephants: 2 -> (1, 4)
 - eat: 3 -> (-2, -3)

经过Word embedding后, x中的每个句子(本来是T维的整数向量)变成了D\*T的矩阵. 考虑到minibatch中共有N张图像,也就是每个输入数据x的维度是 N\*T\*D.

 + 输入:
  - x: 整个时间序列的输入数据, N\*T\*D.
  - h0: 初始的Hidden state, N*H.
  - Wx: input-to-hidden的权值矩阵, D*H.
  - Wh: hidden-to-hidden的权值矩阵, H*H.
  - b: biases, H.
 + 返回值:
  - h: 整个时间序列的Hidden states, N\*T\*H.
  - cache: 所有backward pass中需要用到的量.

```python
def rnn_forward(x, h0, Wx, Wh, b):
    h, cache = [], []
    T = x.shape[1]
    prev_h = h0
    for i in range(T):
        next_h, cache_i = rnn_step_forward(x[:, i, :], prev_h, Wx, Wh, b)
        cache.append(cache_i)
        h.append(next_h)
        prev_h = next_h
    h = np.array(h) # h.shape = T * N * H
    h = h.transpose(1, 0, 2) # h.shape = N * T * H
    return h, cache
```

运行下面的代码来检查实现效果.误差应该小于e-7或者更小.

```python
N, T, D, H = 2, 3, 4, 5

x = np.linspace(-0.1, 0.3, num=N*T*D).reshape(N, T, D)
h0 = np.linspace(-0.3, 0.1, num=N*H).reshape(N, H)
Wx = np.linspace(-0.2, 0.4, num=D*H).reshape(D, H)
Wh = np.linspace(-0.4, 0.1, num=H*H).reshape(H, H)
b = np.linspace(-0.7, 0.1, num=H)

h, _ = rnn_forward(x, h0, Wx, Wh, b)
expected_h = np.asarray([
  [
    [-0.42070749, -0.27279261, -0.11074945,  0.05740409,  0.22236251],
    [-0.39525808, -0.22554661, -0.0409454,   0.14649412,  0.32397316],
    [-0.42305111, -0.24223728, -0.04287027,  0.15997045,  0.35014525],
  ],
  [
    [-0.55857474, -0.39065825, -0.19198182,  0.02378408,  0.23735671],
    [-0.27150199, -0.07088804,  0.13562939,  0.33099728,  0.50158768],
    [-0.51014825, -0.30524429, -0.06755202,  0.17806392,  0.40333043]]])
print('h error: ', rel_error(expected_h, h))
```

### Vanilla RNN: backward
文件cs231n/rnn_layers.py中的函数rnn_backward实现了vanilla RNN的backward pass. 这需要在整个序列中进行后向传播,需要用到之前定义过的函数rnn_step_backward.

 + 输入:
  - dh: 所有Hidden states的上游梯度, N\*T\*H. 注意dh包含的是每个timestep的loss function的上游梯度,而不是timestep过程中的梯度,这个是需要我们用rnn_step_forward函数来计算的.
 + 返回值:
  - dx: 输入数据的梯度, N\*T\*D.
  - dh0: 初始Hidden state的梯度, N*H.
  - dWx: input-to-hidden的权值矩阵的梯度. D*H.
  - dWh: hidden-to-hidden的权值矩阵的梯度. H*H.
  - db: bias向量的梯度. H.

注意是从最后一个timestep向前算的.

```python
def rnn_backward(dh, cache):
    __, __, __, Wx0, __, __ = cache[0] # obtain Wx0 to get D
    D = Wx0.shape[0]
    N, T, H = dh.shape
    dx = np.zeros((N, T, D)) # initialize dx to index dx[:, i, :]
    # initialize dWx, dWh, db as zero matrices to calculate the sum of gradients
    dWx = np.zeros((D, H))
    dWh = np.zeros((H, H))
    db = np.zeros((H,))
    # initialize dprev_h_i as zero matrix to fit the first loop which dh is unchanged
    dprev_h_i = np.zeros((N, H))
    for i in range(T-1, -1, -1):
        dx[:, i, :], dprev_h_i, dWx_i, dWh_i, db_i = rnn_step_backward(dh[:, i, :] + dprev_h_i, cache[i])
        if i==0:
            dh0 = dprev_h_i
        dWx += dWx_i
        dWh += dWh_i
        db += db_i
    return dx, dh0, dWx, dWh, db
```

编写好后运行下面的代码,误差应该小于e-6或更少.

```python
np.random.seed(231)

N, D, T, H = 2, 3, 10, 5

x = np.random.randn(N, T, D)
h0 = np.random.randn(N, H)
Wx = np.random.randn(D, H)
Wh = np.random.randn(H, H)
b = np.random.randn(H)

out, cache = rnn_forward(x, h0, Wx, Wh, b)

dout = np.random.randn(*out.shape)

dx, dh0, dWx, dWh, db = rnn_backward(dout, cache)

fx = lambda x: rnn_forward(x, h0, Wx, Wh, b)[0]
fh0 = lambda h0: rnn_forward(x, h0, Wx, Wh, b)[0]
fWx = lambda Wx: rnn_forward(x, h0, Wx, Wh, b)[0]
fWh = lambda Wh: rnn_forward(x, h0, Wx, Wh, b)[0]
fb = lambda b: rnn_forward(x, h0, Wx, Wh, b)[0]

dx_num = eval_numerical_gradient_array(fx, x, dout)
dh0_num = eval_numerical_gradient_array(fh0, h0, dout)
dWx_num = eval_numerical_gradient_array(fWx, Wx, dout)
dWh_num = eval_numerical_gradient_array(fWh, Wh, dout)
db_num = eval_numerical_gradient_array(fb, b, dout)

print('dx error: ', rel_error(dx_num, dx))
print('dh0 error: ', rel_error(dh0_num, dh0))
print('dWx error: ', rel_error(dWx_num, dWx))
print('dWh error: ', rel_error(dWh_num, dWh))
print('db error: ', rel_error(db_num, db))
```

### Word embedding: forward
在深度学习系统中通常将单词表示成向量.字典中的每个词都对应一个向量,这些向量与系统的其余部分一起学习.

编写文件cs231n/rnn_layers.py中的函数word_embedding_forward,实现将单词(用整数表示)转化为向量.

函数把minibatch中的N个序列(每个序列长度为T.假如字典中含有V个单词)中的每个单词对应到一个D维向量.(前面提到过)

 + 输入:
  - x: 整数数组(N, T),给出单词的indices. x中每个元素idx的范围必须为 0 <= idx < V.这个在表示cats eat dogs的时候可以看出来.
  - W: 权值矩阵(V, D)给出所有单词的单词向量.
 + 返回值:
  - out: 所有输入单词的单词向量数组, N\*T\*D.
  - cache: 所有在backward pass中用到的量.


```python
def word_embedding_forward(x, W):
    out = W[x, :]
    cache = (W, x)
    return out, cache
```

运行下面的代码检查实现效果.误差应小于e-8或更少.

```python
N, T, V, D = 2, 4, 5, 3

x = np.asarray([[0, 3, 1, 2], [2, 1, 0, 3]])
W = np.linspace(0, 1, num=V*D).reshape(V, D)

out, _ = word_embedding_forward(x, W)
expected_out = np.asarray([
 [[ 0.,          0.07142857,  0.14285714],
  [ 0.64285714,  0.71428571,  0.78571429],
  [ 0.21428571,  0.28571429,  0.35714286],
  [ 0.42857143,  0.5,         0.57142857]],
 [[ 0.42857143,  0.5,         0.57142857],
  [ 0.21428571,  0.28571429,  0.35714286],
  [ 0.,          0.07142857,  0.14285714],
  [ 0.64285714,  0.71428571,  0.78571429]]])

print('out error: ', rel_error(expected_out, out))
```


### Word embedding: backward
编写实现Word embedding的backward pass的函数.
由于单词是用整数表示的,我们不能后向传播到单词,只返回到Word embedding矩阵的梯度.

HINT: 看一下函数 np.add.at. 这里贴一下官方文档的examples来体会一下:

```python
>>> a = np.array([1, 2, 3, 4])
>>> np.negative.at(a, [0, 1])
>>> print(a)
array([-1, -2, 3, 4])

>>> a = np.array([1, 2, 3, 4])
>>> np.add.at(a, [0, 1, 2, 2], 1)
>>> print(a)
array([2, 3, 5, 4])

>>> a = np.array([1, 2, 3, 4])
>>> b = np.array([1, 2])
>>> np.add.at(a, [0, 1], b)
>>> print(a)
array([2, 4, 3, 4])
```

 + 输入:
  - dout: 上游的梯度, N\*T\*D.
  - cache: forward pass中返回的需要在backward pass中用到的量.
 + 返回值:
  - dW: word embedding矩阵的梯度, V*D.

```python
def word_embedding_backward(dout, cache):
    W, x = cache
    dW = np.zeros_like(W)
    np.add.at(dW, x, dout)
    return dW
```

运行下面的代码检查实现效果.误差应该小于e-11或更小.

```python
np.random.seed(231)

N, T, V, D = 50, 3, 5, 6
x = np.random.randint(V, size=(N, T))
W = np.random.randn(V, D)

out, cache = word_embedding_forward(x, W)
dout = np.random.randn(*out.shape)
dW = word_embedding_backward(dout, cache)

f = lambda W: word_embedding_forward(x, W)[0]
dW_num = eval_numerical_gradient_array(f, W, dout)

print('dW error: ', rel_error(dW, dW_num))
```

### Temporal Affine layer
每个timestep我们都用affine function把该timestep的RNN Hidden vector转化成字典中每个词的scores. 由于这个过程和作业2中的affine layer非常相似,本次作业直接在文件cs231n/rnn_layers.py给出了函数temporal_affine_forward 和 temporal_affine_backward 的代码.

temporal_affine_forward输入是一个minibatch中N个长度为T的D维向量.我们用affine function把它转化成一个新的M维的向量.

 + 输入:
  - x: 输入数据, N\*T\*D.
  - w: 权值矩阵, D*M.
  - b: biases, M.
 + 返回值:
  - out: 输出数据, N\*T\*M.
  - cache: backward pass中需要用的量.

```python
def temporal_affine_forward(x, w, b):
    N, T, D = x.shape
    M = b.shape[0]
    out = x.reshape(N * T, D).dot(w).reshape(N, T, M) + b
    cache = x, w, b, out
    return out, cache

def temporal_affine_backward(dout, cache):
    """
    Backward pass for temporal affine layer.

    Input:
    - dout: Upstream gradients of shape (N, T, M)
    - cache: Values from forward pass

    Returns a tuple of:
    - dx: Gradient of input, of shape (N, T, D)
    - dw: Gradient of weights, of shape (D, M)
    - db: Gradient of biases, of shape (M,)
    """
    x, w, b, out = cache
    N, T, D = x.shape
    M = b.shape[0]

    dx = dout.reshape(N * T, M).dot(w.T).reshape(N, T, D)
    dw = dout.reshape(N * T, M).T.dot(x.reshape(N * T, D)).T
    db = dout.sum(axis=(0, 1))

    return dx, dw, db
```

运行下面的代码来进行数值梯度检验,误差应小于e-9.

```python
np.random.seed(231)

# Gradient check for temporal affine layer
N, T, D, M = 2, 3, 4, 5
x = np.random.randn(N, T, D)
w = np.random.randn(D, M)
b = np.random.randn(M)

out, cache = temporal_affine_forward(x, w, b)

dout = np.random.randn(*out.shape)

fx = lambda x: temporal_affine_forward(x, w, b)[0]
fw = lambda w: temporal_affine_forward(x, w, b)[0]
fb = lambda b: temporal_affine_forward(x, w, b)[0]

dx_num = eval_numerical_gradient_array(fx, x, dout)
dw_num = eval_numerical_gradient_array(fw, w, dout)
db_num = eval_numerical_gradient_array(fb, b, dout)

dx, dw, db = temporal_affine_backward(dout, cache)

print('dx error: ', rel_error(dx_num, dx))
print('dw error: ', rel_error(dw_num, dw))
print('db error: ', rel_error(db_num, db))
```

### Temporal Softmax loss
在RNN语言模型中,在每个timestep产生了字典中每个单词的score. 我们知道每个timestep的groundtruth单词,因此用softmax loss function来计算每个timestep的loss和gradient.
把每个timestep的loss加起来,然后在minibatch中做平均.

但注意一个wrinkle. 由于我们处理的是minibatch,因此不同的captions的长度可能不同.因此我们在每个标题后面加了<NULL> tokens使他们长度相同. 我们不希望在计算loss和gradient时考虑这些tokens,因此除了scores和groundtruth labels我们的loss function同样接收一个叫做mask的数组,该数组含有那些scores中的元素需要在计算loss时被考虑到.

由于该过程和作业1中的softmax loss function很像,作业在文件cs231n/rnn_layers.py中直接给出了该函数temporal_softmax_loss.

时间序列的长度为T,在每个timestep对用大小为V的字典做预测,minibatch大小为N. 输入x给出了所有字典元素在所有timestep的scores. y给出了每个timestep中groundtruth元素的indeces.
我们在每个timestep用cross entropy loss,求出所有timestep的loss的和,然后在minibatch中做平均.

有时想ignore模型在某些timesteps的output,因为有些序列可能有NULL tokens. mask变量告诉我们那些元素在计算loss时需要考虑.

```python
def temporal_softmax_loss(x, y, mask, verbose=False):
    """
    Inputs:
    - x: Input scores, of shape (N, T, V)
    - y: Ground-truth indices, of shape (N, T) where each element is in the range
     0 <= y[i, t] < V
    - mask: Boolean array of shape (N, T) where mask[i, t] tells whether or not
    the scores at x[i, t] should contribute to the loss.

    Returns a tuple of:
    - loss: Scalar giving loss
    - dx: Gradient of loss with respect to scores x.
    """
    N, T, V = x.shape

    x_flat = x.reshape(N * T, V)
    y_flat = y.reshape(N * T)
    mask_flat = mask.reshape(N * T)

    probs = np.exp(x_flat - np.max(x_flat, axis=1, keepdims=True))
    probs /= np.sum(probs, axis=1, keepdims=True)
    loss = -np.sum(mask_flat * np.log(probs[np.arange(N * T), y_flat])) / N
    dx_flat = probs.copy()
    dx_flat[np.arange(N * T), y_flat] -= 1
    dx_flat /= N
    dx_flat *= mask_flat[:, None]

    if verbose: print('dx_flat: ', dx_flat.shape)

    dx = dx_flat.reshape(N, T, V)

    return loss, dx
```

运行下面的代码检查函数的loss和梯度的计算情况.dx的误差应小于e-7.

```python
# Sanity check for temporal softmax loss
from cs231n.rnn_layers import temporal_softmax_loss

N, T, V = 100, 1, 10

def check_loss(N, T, V, p):
    x = 0.001 * np.random.randn(N, T, V)
    y = np.random.randint(V, size=(N, T))
    mask = np.random.rand(N, T) <= p
    print(temporal_softmax_loss(x, y, mask)[0])

check_loss(100, 1, 10, 1.0)   # Should be about 2.3
check_loss(100, 10, 10, 1.0)  # Should be about 23
check_loss(5000, 10, 10, 0.1) # Should be about 2.3

# Gradient check for temporal softmax loss
N, T, V = 7, 8, 9

x = np.random.randn(N, T, V)
y = np.random.randint(V, size=(N, T))
mask = (np.random.rand(N, T) > 0.5)

loss, dx = temporal_softmax_loss(x, y, mask, verbose=False)

dx_num = eval_numerical_gradient(lambda x: temporal_softmax_loss(x, y, mask)[0], x, verbose=False)

print('dx error: ', rel_error(dx, dx_num))
```
