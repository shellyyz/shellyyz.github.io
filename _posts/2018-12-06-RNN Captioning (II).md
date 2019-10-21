---
layout:     post
title:      RNN Captioning (II)
subtitle:   cs231n assignment 3
date:       2018-12-06
author:     shellyyz
# description: some common commands in Git.
# header-img: img/post-bg-ios9-web.jpg -->
catalog: true
comments: true
tags:
    - Git
    - Commands
---


# RNN Captioning (II)
接上文RNN Captioning (I).

### RNN for image captioning
既然我们已经实现了RNN中的必要层,现在来把他们结合在一起,构造一个image captioning模型.
打开文件cs231n/classifiers/rnn.py看一眼类CaptioningRNN.

实现模型中函数loss的forward和backward pass. 现在只需要实现cell_type='rnn'的情况(for Vanialla RNNs). 随后将实现LSTM的情况.

先说一下CaptioningRNN这个类. 每个CationingRNN用RNN根据图像特征产生图像标题.
输入向量是D维,字典大小是V,序列长度是T, hidden dimension维度为H,单词向量的维度为W,每个minibatch大小为N.

注意在CaptionRNN中我们不使用任何正则手段.

在类CaptionRNN中先进行初始化.
构建一新的CaptionRNN的instance.
 + 初始化输入:
  - word_to_idx: 含有V个元素,将string映射到整数[0, V).
  - input_dim: 图像的特征向量维度D.
  - wordvec_dim: 单词向量的维度W.
  - hidden_dim: Dimension H for the hidden state of the RNN.  
  - cell_type: What type of RNN to use; either 'rnn' or 'lstm'.
  - dtype: numpy datatype to use; use float32 for training and float64 for
  numeric gradient checking.

初始化都干了什么:
 - 判断cell_type是不是满足.
 - 输入的量传给self(本来这个class里面是空的,现在有了属性), 诸如数据类型,字典词汇矩阵,一些量的维度vocab_size.
 - 初始化字典self.param放东西.
 - 提取出tokens. **字典(Dictionary) get() 函数返回指定键的值，如果值不在字典中返回默认值。**
 - 初始化单词向量矩阵W_embed, CNN->hidden state projection的W_proj(用来把输入的图像特征向量传给hidden state.).
 - 初始化RNN的参数,Wx(把单词向量传给hidden state), Wh(Hidden到Hidden), b.
 - 初始化输出到词汇的权值W_vocab(hidden到vocab), b_vocab.
 - 把参数们的数据类型转换成指定类型.

```python
def __init__(self, word_to_idx, input_dim=512, wordvec_dim=128,
                 hidden_dim=128, cell_type='rnn', dtype=np.float32):
    """
    Construct a new CaptioningRNN instance.

    Inputs:
    - word_to_idx: A dictionary giving the vocabulary. It contains V entries, and maps each string to a unique integer in the range [0, V).
    - input_dim: Dimension D of input image feature vectors.
    - wordvec_dim: Dimension W of word vectors.
    - hidden_dim: Dimension H for the hidden state of the RNN.
    - cell_type: What type of RNN to use; either 'rnn' or 'lstm'.
    - dtype: numpy datatype to use; use float32 for training and float64 for
    numeric gradient checking.
    """
    if cell_type not in {'rnn', 'lstm'}:
            raise ValueError('Invalid cell_type "%s"' % cell_type)

    self.cell_type = cell_type
    self.dtype = dtype
    self.word_to_idx = word_to_idx
    self.idx_to_word = {i: w for w, i in word_to_idx.items()}
    self.params = {}

    vocab_size = len(word_to_idx)

    self._null = word_to_idx['<NULL>']
    self._start = word_to_idx.get('<START>', None)
    self._end = word_to_idx.get('<END>', None)

    # Initialize word vectors
    self.params['W_embed'] = np.random.randn(vocab_size, wordvec_dim)
    self.params['W_embed'] /= 100

    # Initialize CNN -> hidden state projection parameters
    self.params['W_proj'] = np.random.randn(input_dim, hidden_dim)
    self.params['W_proj'] /= np.sqrt(input_dim)
    self.params['b_proj'] = np.zeros(hidden_dim)

    # Initialize parameters for the RNN
    dim_mul = {'lstm': 4, 'rnn': 1}[cell_type]
    self.params['Wx'] = np.random.randn(wordvec_dim, dim_mul * hidden_dim)
    self.params['Wx'] /= np.sqrt(wordvec_dim)
    self.params['Wh'] = np.random.randn(hidden_dim, dim_mul * hidden_dim)
    self.params['Wh'] /= np.sqrt(hidden_dim)
    self.params['b'] = np.zeros(dim_mul * hidden_dim)

    # Initialize output to vocab weights
    self.params['W_vocab'] = np.random.randn(hidden_dim, vocab_size)
    self.params['W_vocab'] /= np.sqrt(hidden_dim)
    self.params['b_vocab'] = np.zeros(vocab_size)

    # Cast parameters to correct dtype
    for k, v in self.params.items():
            self.params[k] = v.astype(self.dtype)
```

初始化后,在类中定义loss函数. 计算RNN的训练loss. 输入图像特征和ground-truth标题,用RNN或者LSTM来计算所有参数的loss和gradients.
 + 输入:
  - features:输入图像特征,N*D.
  - captions: ground-truth标题, 一个整数数组(N*T),每个元素的范围是0 <= y[i, t] < V

 + 返回值:
  - loss: 标量loss.
  - grads: 梯度字典平行于self.params

loss函数要干嘛:
+ 准备工作
 - 把ground-truth标题分成两部分captions_in和captions_out,captions_in没有最后一个词,captions_out没有第一个词,是我们期望RNN生成的. captions_in的第一个元是<START>. token, captions_out的第一个词是第一个单词.
 - 把captions_out中的NULL位置找出来,不为NULL的index给mask.
 - 接收W_proj和b_proj参数,以便计算初始的Hidden state.
 - 接收W_embed参数.
 - 接收输入到隐藏Wx,隐藏到隐藏Wh,以及RNN的bias参数.
 - 接收隐藏到词汇的参数w_vocab和b_vocab.
 - 初始化loss=0, grads={}.
+ 实现CaptionRNN的forward和backward过程.
 - forward pass:
   * 用image features计算初始的隐藏层. 生成大小为N*H的数组.
   * 用Word embedding层来讲captions_in里面的单词indices转化成向量,产生大小为N\*T\*W的数组.
   * 用vanilla RNN或者LSTM(取决于self.cell_type)来处理输入单词向量,并生成所有timestep的Hidden state向量, 生成数组大小为N\*T\*H.
   * 用temporal的affine变换把每个timestep的Hidden states变成vocabulary的scores, 产生大小为N\*T\*V的数组.
   * 用temporal的softmax和captions_out计算loss.用mask忽略掉输出是NULL的单词.
  - 在backward过程,需要计算loss对应所有参数的梯度.用loss和grads变量存储. grads中的变量名字和self.params的应该对应.

现在我们可以自由使用之前在layers.py里定义过的函数了.

```python
def loss(self, features, captions):
    """
    Compute training-time loss for the RNN. We input image features and
    ground-truth captions for those images, and use an RNN (or LSTM) to compute
    loss and gradients on all parameters.

    Inputs:
    - features: Input image features, of shape (N, D)
    - captions: Ground-truth captions; an integer array of shape (N, T) where
      each element is in the range 0 <= y[i, t] < V

    Returns a tuple of:
    - loss: Scalar loss
    - grads: Dictionary of gradients parallel to self.params
    """
    # Cut captions into two pieces: captions_in has everything but the last word
    # and will be input to the RNN; captions_out has everything but the first
    # word and this is what we will expect the RNN to generate. These are offset
    # by one relative to each other because the RNN should produce word (t+1)
    # after receiving word t. The first element of captions_in will be the START
    # token, and the first element of captions_out will be the first word.
    captions_in = captions[:, :-1]
    captions_out = captions[:, 1:]

    # You'll need this
    mask = (captions_out != self._null)

    # Weight and bias for the affine transform from image features to initial
    # hidden state
    W_proj, b_proj = self.params['W_proj'], self.params['b_proj']

    # Word embedding matrix
    W_embed = self.params['W_embed']

    # Input-to-hidden, hidden-to-hidden, and biases for the RNN
    Wx, Wh, b = self.params['Wx'], self.params['Wh'], self.params['b']

    # Weight and bias for the hidden-to-vocab transformation.
    W_vocab, b_vocab = self.params['W_vocab'], self.params['b_vocab']

    loss, grads = 0.0, {}


    h0, features_cache = affine_forward(features, W_proj, b_proj)

    captions_in_emb, emb_in_cache = word_embedding_forward(captions_in, W_embed)

    if self.cell_type == 'rnn':
        h, rnn_cache = rnn_forward(captions_in_emb, h0, Wx, Wh, b)
    elif self.cell_type == 'lstm':
        h, lstm_cache = lstm_forward(captions_in_emb, h0, Wx, Wh, b)

    temporal_out, temporal_cache = temporal_affine_forward(h, W_vocab, b_vocab)

    loss, dout = temporal_softmax_loss(temporal_out, captions_out, mask)

    # backward and grads
    dtemp, grads['W_vocab'], grads['b_vocab'] = temporal_affine_backward(dout, temporal_cache)
    if self.cell_type == 'rnn':
        drnn, dh0, grads['Wx'], grads['Wh'], grads['b'] = rnn_backward(dtemp, rnn_cache)
    elif self.cell_type == 'lstm':
        drnn, dh0, grads['Wx'], grads['Wh'], grads['b'] = lstm_backward(dtemp, lstm_cache)
    grads['W_embed'] = word_embedding_backward(drnn, emb_in_cache)
    dfeatures, grads['W_proj'], grads['b_proj'] = affine_backward(dh0, features_cache)

    return loss, grads
```

下面来定义一下模型在测试时的forward pass. 在输入的特征向量feature vectors中抽取一些.

在每个timestep中,把当前单词embed后和前一个hidden state传到RNN中得到下一个hidden state, 用hidden state得到所有词汇的scores, 然后选择得分最高的单词作为下一个单词. 初始的Hidden state是由输入图像的特征经过affine transform得到的, 初始单词是<START> token.

对于LSTMs来说也要跟进cell state, 初始cell state为0.

 + 输入:
  - features: 输入图像特征数组, N*D.
  - max_length: 生成标题的最大长度T.
 + 返回值:
  - captions: 给出抽样出的标题的数组, N* max_length. 每个元素为[0, v)中的整数.标题的第一个元素是第一个抽到的单词,而不是<START>.

sample函数要做的事情:
+ 初始化:
 - 接收维度N.
 - 初始化captions.
 - 接收参数, W_proj, b_proj, W_embed, Wx, Wh, b, W_vocab, b_vocab.
+ 实现模型测试时抽样. 需要用学到的输入图像特征的affine transform初始化RNN的hidden state. 未给RNN的第一个单词应该是<START>. 在每个timestep需要:
 - 用学到的word_embeddings嵌入上一个词.
 - 用上一个Hidden state和当前被嵌入单词来得到下一个Hidden state.
 - 用学到的affine变换计算下一个Hidden state词汇表中所有词汇的scores.
 - 选择得分最高的单词作为下一个单词,并把该单词(单词的index)存入变量captions中.

简便起见,在<END> token被抽到后不需要停止生成过程, 但想停的话也可.

HINT: 这里不能用rnn_forward或者lstm_forward函数,而是要在循环中用rnn_step_forward和lstm_step_forward.

注意: 这里仍然是对minibatch进行操作的. 如果用的是LSTM, 同样初始化第一个cell state为0.

```python
def sample(self, features, max_length=30):
    N = features.shape[0]
    captions = self._null * np.ones((N, max_length), dtype=np.int32)

    # Unpack parameters
    W_proj, b_proj = self.params['W_proj'], self.params['b_proj']
    W_embed = self.params['W_embed']
    Wx, Wh, b = self.params['Wx'], self.params['Wh'], self.params['b']
    W_vocab, b_vocab = self.params['W_vocab'], self.params['b_vocab']

    affine_out, affine_cache = affine_forward(features ,W_proj, b_proj)
    prev_word_idx = [self._start]*N
    prev_h = affine_out
    prev_c = np.zeros(prev_h.shape)

    for i in range(0,max_length):
        prev_word_embed  = W_embed[prev_word_idx]
        if self.cell_type == 'rnn':
            next_h, rnn_step_cache = rnn_step_forward(prev_word_embed, prev_h, Wx, Wh, b)
        elif self.cell_type == 'lstm':
            next_h, next_c,lstm_step_cache = lstm_step_forward(prev_word_embed, prev_h, prev_c, Wx, Wh, b)
            prev_c = next_c
        else:
            raise ValueError('Invalid cell_type "%s"' % self.cell_type)
        vocab_affine_out, vocab_affine_out_cache = affine_forward(next_h, W_vocab, b_vocab)
        captions[:,i] = list(np.argmax(vocab_affine_out, axis = 1))
        prev_word_idx = captions[:,i]
        prev_h = next_h

    return captions
```

完成后运行下面的代码检查实现效果,误差应小于e-10.

```python
N, D, W, H = 10, 20, 30, 40
word_to_idx = {'<NULL>': 0, 'cat': 2, 'dog': 3}
V = len(word_to_idx)
T = 13

model = CaptioningRNN(word_to_idx,
          input_dim=D,
          wordvec_dim=W,
          hidden_dim=H,
          cell_type='rnn',
          dtype=np.float64)

# Set all model parameters to fixed values
for k, v in model.params.items():
    model.params[k] = np.linspace(-1.4, 1.3, num=v.size).reshape(*v.shape)

features = np.linspace(-1.5, 0.3, num=(N * D)).reshape(N, D)
captions = (np.arange(N * T) % V).reshape(N, T)

loss, grads = model.loss(features, captions)
expected_loss = 9.83235591003

print('loss: ', loss)
print('expected loss: ', expected_loss)
print('difference: ', abs(loss - expected_loss))
```
