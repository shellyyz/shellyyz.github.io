---
layout:     post
title:      PCA Review
subtitle:   A Summary of Principle Component Analysis
date:       2019-09-26
author:     shellyyz
# description: some common commands in Git.
# header-img: img/post-bg-ios9-web.jpg -->
catalog: 	 true
comments: true
tags:
    - PCA
    - Subspace Learning
---
{% include head.html %}

- [主成分分析](#%e4%b8%bb%e6%88%90%e5%88%86%e5%88%86%e6%9e%90)
  - [Statistical PCA](#statistical-pca)
    - [第一个主成分](#%e7%ac%ac%e4%b8%80%e4%b8%aa%e4%b8%bb%e6%88%90%e5%88%86)
    - [d个主成分](#d%e4%b8%aa%e4%b8%bb%e6%88%90%e5%88%86)
    - [Sample PCA](#sample-pca)
  - [Geometric PCA](#geometric-pca)

# 主成分分析

最近刚好在做TPAMI相关领域文章的review， 这学期也选了 subspace learning 这门课，前段时间看了 Prof. Vidal 的视频，刚好可以做个整理——PCA这个基本又重要的问题角度多、影响广。

在机器学习中，给定一组高维数据点，每个数据点的维度可能高达几千上万，我们想要用几个方向来概括或者捕捉数据点的关键信息，使得数据点在这些低维度空间中的表示最不相同。这要求我们找到一组新的基，这组基所 span 的子空间维度 d 远小于原来数据的维度 D，然后将数据用这些新的基进行表示。

## Statistical PCA

设 $\boldsymbol{x}\in \mathbb{R}^{D}$, 且 $\boldsymbol{x}$ 是一个零均值的随机变量，即 $E(x)=\boldsymbol{0}$。

### 第一个主成分

定义变量 $\boldsymbol{x}$ 的第一个主成分 \(\boldsymbol{u}_1\) 为使得 \(Var(y_{1})\) 最大的方向，其中

$$y_{1}=<\boldsymbol{u}_{1},\boldsymbol{x}> ={\boldsymbol{u}_{1}}^{T}\boldsymbol{x}$$


求第一个主成分的问题就可以写为

$$\max_{\boldsymbol{u}_1} {Var(y_{1})}\;\;\;\; \rm{s.t.} ||\boldsymbol{u}_{1}||_{2}=1$$

对这个问题的目标函数进行简单分析，根据概率论中的基本定理，

$$\begin{aligned}
E(y_{1}) & =E({\boldsymbol{u}_{1}}^T \boldsymbol{x})={\boldsymbol{u}_{1}}^T E(\boldsymbol{x})=0\\
Var(y_{1})& =E(y^{2})-E(y_{1})^{2}\\
&=E(({\boldsymbol{u}_{1}}^T \boldsymbol{x})^2)\\
&=E({\boldsymbol{u}_{1}}^T \boldsymbol{x}\boldsymbol{x}^T \boldsymbol{u}_{1})\\
&={\boldsymbol{u}_{1}}^T E(\boldsymbol{x}\boldsymbol{x}^T) \boldsymbol{u}_{1}\\
&={\boldsymbol{u}_{1}}^T {\Sigma}_{\boldsymbol{x}} \boldsymbol{u}_{1}
\end{aligned} $$

这里提一下变量的协方差矩阵为

$$Cov(\boldsymbol{x})=E((\boldsymbol{x}-\boldsymbol{\mu})(\boldsymbol{x}-\boldsymbol{\mu})^T)$$

由于假设 $\boldsymbol{x}$ 是零均值的，我们得到上面的式子。

现在回到我们的问题：找到一个方差最大的方向。将之转化为一个优化问题：

$$\max_{\boldsymbol{u}_{1}} {\boldsymbol{u}_{1}}^T {\Sigma}_{\boldsymbol{x}} \boldsymbol{u}_{1}\;\;\;\; \rm{s.t.} ||\boldsymbol{u}_{1}||_{2}=1$$

大家都知道该问题的解就是半正定矩阵 ${\Sigma}_{\boldsymbol{x}}$ 的最大特征值对应的特征向量（简便起见，可以称其为“最大特征向量”）。
但如果我们直接去解这个优化问题，就有

$$\begin{aligned}
\mathcal{L}(\boldsymbol{u},\lambda) &= {\boldsymbol{u}_{1}}^T {\Sigma}_{\boldsymbol{x}} \boldsymbol{u}_{1} + \lambda(1-{\boldsymbol{u}_{1}}^T \boldsymbol{u}_{1})\\
\boldsymbol{0}&=\frac{\partial \boldsymbol{\mathcal{L}}}{\partial \boldsymbol{u}_{1}}=2{\Sigma}_{\boldsymbol{x}} \boldsymbol{u}_{1} - 2\lambda\boldsymbol{u}_{1}\\
&\Rightarrow {\Sigma}_{\boldsymbol{x}}\boldsymbol{u}_{1}=\lambda \boldsymbol{u}_{1}\\
\boldsymbol{0}&=\frac{\partial \boldsymbol{\mathcal{L}}}{\partial \lambda}=1-{\boldsymbol{u}_{1}}^T \boldsymbol{u}_{1}
\end{aligned}$$

*注意，拉格朗日函数对等式约束条件的系数 $\lambda$ 求导，得到的总是等式约束条件本身。*

现在可以看出，$\boldsymbol{u}_{1}$ 一定是矩阵 ${\Sigma}_{\boldsymbol{x}}$ 的一个特征向量，$\lambda$ 为对应特征值，但是是哪一个呢?
带入目标函数：

$$\begin{aligned}
\max & \; {\boldsymbol{u}_{1}}^T {\Sigma}_{\boldsymbol{x}} \boldsymbol{u}_{1}\\
\max & \; {\boldsymbol{u}_{1}}^T \lambda \boldsymbol{u}_{1}\\
\max & \; \lambda\\
\end{aligned}$$

就是最大的特征值对应的特征向量，没得跑了。

### d个主成分

定义变量 $\boldsymbol{x}$ 的 $d$ 个主成分为使得互不相关的 $y_{i}$ 方差最大的 $d$ 个方向，其中

$$y_{i}={\boldsymbol{u}_{i}}^T \boldsymbol{x}$$

根据前面对一个主成分的分析，

$$Var(y_{i})={\boldsymbol{u}_{i}}^{T} {\Sigma}_{\boldsymbol{x}} \boldsymbol{u}_{i}\\
Var(\boldsymbol{x})=\Sigma_{\boldsymbol{x}}=E(\boldsymbol{x}\boldsymbol{x}^T)\\
E(y_{i})=E({\boldsymbol{u}_{i}}^{T} \boldsymbol{x})={\boldsymbol{u}_{i}}^TE(\boldsymbol{x})=0$$

现在要求 $y_{i}$ 们互不相关，即

$$\begin{aligned}
\forall i\ne j, 0 &= E(y_{i} {y_{j}})\\
&=E({\boldsymbol{u}_{1}}^T \boldsymbol{x}\boldsymbol{x}^T \boldsymbol{u}_{1})\\
&={\boldsymbol{u}_{i}}^{T} E(\boldsymbol{x}\boldsymbol{x}^T) \boldsymbol{u}_{j}\\
&={\boldsymbol{u}_{i}}^{T} {\Sigma}_{\boldsymbol{x}} \boldsymbol{u}_{j}
\end{aligned}$$

现在我们已经知道第一个主成分，依次去求第二个。

$$\max Var(y_{2}) \;\; \rm{s.t.} \; ||\boldsymbol{u}_{1}||_{2}=1, {\boldsymbol{u}_{1}}^{T} {\Sigma}_{\boldsymbol{x}} \boldsymbol{u}_{2}=0$$

将 

$${\Sigma}_{\boldsymbol{x}}\boldsymbol{u}_{1}=\lambda \boldsymbol{u}_{1}, \lambda \ne 0$$

代入 ${\boldsymbol{u}_{i}}^T {\Sigma}_{\boldsymbol{x}} \boldsymbol{u}_{j}=0$ 我们有

$$\boldsymbol{u}_{1}^{T} \boldsymbol{u}_{2}=0$$

这样我们就把第二个约束条件从 uncorrelated 推到了 orthogonal：

$$\max Var(y_{2}) \;\; \rm{s.t.} \; ||\boldsymbol{u}_{1}||_{2}=1, {\boldsymbol{u}_{1}}^{T} \boldsymbol{u}_{2}=0$$

依照惯例，解该优化问题先写出其 Lagrangian （ $\boldsymbol{u}_{1}$ 已知）：

$$\begin{aligned}
\mathcal{L}(\boldsymbol{u},\lambda) &= {\boldsymbol{u}_{2}}^T {\Sigma}_{\boldsymbol{x}} \boldsymbol{u}_{2} + \lambda(1-{\boldsymbol{u}_{2}}^T \boldsymbol{u}_{2}) + \gamma {\boldsymbol{u}_{1}}^T \boldsymbol{u}_{2}\\
\boldsymbol{0}&=\frac{\partial \boldsymbol{\mathcal{L}}}{\partial \boldsymbol{u}_{2}} = 2{\Sigma}_{\boldsymbol{x}} \boldsymbol{u}_{2} - 2\lambda\boldsymbol{u}_{2} + \gamma \boldsymbol{u}_{1}\\
\Rightarrow \; & 2 {\boldsymbol{u}_{1}}^{T} {\Sigma}_{\boldsymbol{x}} \boldsymbol{u}_{2} - 2\lambda{\boldsymbol{u}_{1}}^{T}\boldsymbol{u}_{2} + \gamma {\boldsymbol{u}_{1}}^{T}\boldsymbol{u}_{1} = 0 \\
& 2 {\boldsymbol{u}_{2}}^{T} {\Sigma}_{\boldsymbol{x}} \boldsymbol{u}_{2} - 2\lambda{\boldsymbol{u}_{2}}^{T}\boldsymbol{u}_{2} + \gamma {\boldsymbol{u}_{2}}^{T}\boldsymbol{u}_{1} = 0 \\
\Rightarrow \; & \gamma = 0\\
& 2 {\boldsymbol{u}_{2}}^{T} {\Sigma}_{\boldsymbol{x}} \boldsymbol{u}_{2} - 2\lambda{\boldsymbol{u}_{2}}^{T}\boldsymbol{u}_{2} = 0\\
\Rightarrow \; & {\Sigma}_{\boldsymbol{x}} \boldsymbol{u}_{2} = \lambda\boldsymbol{u}_{2} \; (\boldsymbol{0}=\frac{\partial \boldsymbol{\mathcal{L}}}{\partial \boldsymbol{u}_{2}}, \gamma = 0)\\
& {\boldsymbol{u}_{2}}^{T} {\Sigma}_{\boldsymbol{x}} \boldsymbol{u}_{2} = \lambda
\end{aligned}$$

现在已经清楚，和第一个主成分分析一样，第二个主成分仍然是变量 x 协方差矩阵的特征向量。同样的还有，我们仍然要最大化该特征向量对应的特征值。这样的话，我们完全可以用最大特征值对应的特征向量来当作最优解。哪里不对呢？
注意，和第一个主成分求解不同的是，第二个主成分的优化问题多了一个限制条件：需要和第一个主成分垂直，因此第二个主成分不可能就是第一个特征向量，那是哪个？
注意，协方差矩阵是半正定的，存在由特征向量生成的 orthogonal basis。因此，假设 ${\Sigma}_{\boldsymbol{x}}$ 的特征值为：

$$\lambda_{1} \ge \lambda_{2} \ge ... \ge \lambda_{n}$$

那么第二个特征向量 \lambda_{2} 对应的特征向量就是第二个主成分（可能有重复的特征值，因此特征向量不唯一；这说明变量在不同的分量上的方差相同）。

我们 generalize 一下就得到了下面的定理：

**Theorem** Top $d$ principal components are 

$$y_{1}={\boldsymbol{u}_{1}}^{T} \boldsymbol{x}$$

where $\boldsymbol{u}_{i}$ is the $i$-th larges eigenvector of $\Sigma_{\boldsymbol{x}}$. 

上面的定理中特征向量不是唯一的：
其一，sign ambiguity， 可以取相反向量；其二，特征值的特征空间维度可能不唯一。


### Sample PCA

至此对于 Statistical PCA 的原理和方法都已经清晰。但是， Statistical PCA 要求已知变量 x 的协方差矩阵，否则无法计算主成分。
但如果我们已知变量 x 的一组独立同分布的样本数据 $\boldsymbol{x}_{1},...,\boldsymbol{x}_{N}$，可以对协方差矩阵进行估计：

$$\hat{\Sigma}_{N} = \frac{1}{N} \Sigma_{i=1}^{N} \boldsymbol{x}_{i} \boldsymbol{x}_{i}^{T}$$

然后我们计算样本方差的主成分。这叫做**Sample PCA**。

注意，对于协方差矩阵的估计，在实际中是困难的。尤其当数据的 ambient dimension 很大（比如图像数据）而样本点又很少（不足以估计协方差矩阵）的时候。只有当样本点数目 N 趋近于无穷大时，对于协方差矩阵的估计才是准确的。


## Geometric PCA

从几何的角度考虑PCA，不去管协方差什么的，而直接看矩阵本身。

给定一组存在于低维子空间的数据

$$\{\boldsymbol{x}_{j}\}_{j=1}^{N}, \; \boldsymbol{x}_{j}\in \mathcal{S}, \; \dim(\mathcal{S}) = d \ll D$$

对于仿射子空间 $\mathcal{S}$，我们可以表示为

$$\mathcal{S}=\{\boldsymbol{x}: \boldsymbol{x}=\mu+\boldsymbol{U}\boldsymbol{y}\}$$

其中 $\mu\in \mathcal{S}, \boldsymbol{u}\in \mathbb{R}^{D\times d}, \boldsymbol{y}\in \mathbb{R}^{d}$.

实际中的数据可能含有噪声或 outliers，但本质上还是存在于一个低维空间中。因此问题变成：找到能最好拟合数据的低维子空间——拟合误差越小越好。

数据模型: 

$$\boldsymbol{x}_{j}=\boldsymbol{\mu} + \boldsymbol{U}\boldsymbol{y}+\boldsymbol{\varepsilon}_{j}$$

我们要找到一个低维子空间 $\mathcal{S}$ 使得所有数据误差的二范数之和最小:

$$\min  {\Sigma_{j=1}^{N} ||\boldsymbol{\varepsilon}_{j}||_{2}^{2}}$$

即优化问题：

$$\min_{\boldsymbol{\mu}, \boldsymbol{U}} {\Sigma_{j=1}^{N} ||\boldsymbol{x}_{j}-\boldsymbol{\mu}-\boldsymbol{U} \boldsymbol{y}_{j}||_{2}^{2}}$$

这个问题存在 Translational ambiguity：

* $\boldsymbol{\mu}$ 和 $\boldsymbol{U} \boldsymbol{y}_{j}$ 可以互相转化。为消除这种ambiguity，我们要求可以互相转化。为消除这种ambiguity，我们要求
  
  $$\Sigma_{j} \boldsymbol{y}_{j}=0$$
  
  即要求低维空间的坐标位于原点周围：中心化。
* 基变换引起的 change of basis ambiguity:
  
  $$\boldsymbol{U}\rightarrow\boldsymbol{UA}, \; \boldsymbol{y}\rightarrow \boldsymbol{A}^{-1} \boldsymbol{y}\\
  \boldsymbol{\mu}+\boldsymbol{Uy}\rightarrow \boldsymbol{UA}\boldsymbol{A}^{-1}\boldsymbol{y}$$
  
  这样的话一点都不 elegant。为了消除这种ambiguity，我们要求 $\boldsymbol{U}$ 是 orthonormal 的。即

  $$\boldsymbol{U}^{T}\boldsymbol{U} =\boldsymbol{I}_{d}$$

  注意，这样子求解出来的 $\boldsymbol{U}$ 仍然不是唯一的，可以在子空间中进行任意旋转，但这不影响我们解决这个问题。

因此，我们要求 $\Sigma_{j} \boldsymbol{y}_{j}=0$ 来中心化系数，
要求 $\boldsymbol{U}^{T}\boldsymbol{U} =\boldsymbol{I}_{d}$ 使得所求基向量为正交基。这样，问题就变为了

$$\min_{\boldsymbol{\mu}, \boldsymbol{U}, \boldsymbol{y}_{j}} {\Sigma_{j=1}^{N} ||\boldsymbol{x}_{j}-\boldsymbol{\mu}-\boldsymbol{U} \boldsymbol{y}_{j}||_{2}^{2}} \;\; \rm{s.t.} \Sigma_{j} \boldsymbol{y}_{j}=0, \; \boldsymbol{U}^{T}\boldsymbol{U} =\boldsymbol{I}_{d}$$

可以看出，这个优化问题是非凸的：第一，目标函数中有一项是两个变量的乘积；第二，约束条件中要求 U 在一个球面上，而球面不是一个凸集。按照惯例用拉格朗日函数解：

$$\mathcal{L}={\Sigma_{j=1}^{N} ||\boldsymbol{x}_{j}-\boldsymbol{\mu}-\boldsymbol{U} \boldsymbol{y}_{j}||_{2}^{2}}+ \boldsymbol{\lambda}^{T}\Sigma_{j} \boldsymbol{y}_{j}+<\boldsymbol{\Lambda},\boldsymbol{I}_{d}-\boldsymbol{U}^{T}\boldsymbol{U}>
$$

注意，由于矩阵 $\boldsymbol{I}_{d}-\boldsymbol{U}^{T}\boldsymbol{U}$ 是对称阵，我们实际上不需要 $\Lambda$ 的size这么多限制条件，也就是 $d^2$ 个。我们只要 $d(d+1)/2$ 个限制条件。因此我们要求 $\Lambda$ 必须也是对称的。

对拉格朗日函数求对 $\boldsymbol{\mu}$ 的偏导数并令其为零：

$$
\frac{\partial \boldsymbol{\mathcal{L}}}{\partial \boldsymbol{\mu}} = -2{\Sigma}_{j=1}^{N} {(\boldsymbol{x}_{j}-\boldsymbol{\mu}-\boldsymbol{U}\boldsymbol{y}_{i})}=\boldsymbol{0} \\
\Rightarrow \hat{\boldsymbol{\mu}}=\frac{1}{N} \Sigma_{j=1}^{N} (\boldsymbol{x}_{j}+\boldsymbol{U}\Sigma_{j=1}^{N}\boldsymbol{y}_{j})=\frac{1}{N} \Sigma_{j=1}^{N} \boldsymbol{x}_{j}
$$

倘若我们令

$$\tilde{\boldsymbol{x}}_{j}=\boldsymbol{x}_{j}-\hat{\boldsymbol{\mu}}\\
\boldsymbol{X} = [\tilde{\boldsymbol{x}}_{1}, ..., \tilde{\boldsymbol{x}}_{N}]\\
\boldsymbol{Y} = [\boldsymbol{y}_{1}, ..., \boldsymbol{y}_{N}]
$$

那么拉格朗日函数可以写作

$$\mathcal{L}=||\boldsymbol{X}-\boldsymbol{UY}||_{F}^{2} + \boldsymbol{\lambda}^{T}\Sigma_{j} \boldsymbol{y}_{j}+<\boldsymbol{\Lambda},\boldsymbol{I}_{d}-\boldsymbol{U}^{T}\boldsymbol{U}>
$$

对 $\boldsymbol{Y}$ 的偏导数为

$$\frac{\partial \boldsymbol{\mathcal{L}}}{\partial \boldsymbol{Y}} = -2 \boldsymbol{U}^{\top}(\boldsymbol{X}-\boldsymbol{UY}) + \boldsymbol{\lambda} \boldsymbol{1}^{\top}_{N} =\boldsymbol{0}
$$

两边同时乘以 $\boldsymbol{1}_{N}$

$$
-2 \boldsymbol{U}^{\top}(\boldsymbol{X}-\boldsymbol{UY}) \boldsymbol{1}_{N} + \boldsymbol{\lambda} \boldsymbol{1}^{\top}_{N} \boldsymbol{1}_{N} =\boldsymbol{0} \\
-2 \boldsymbol{U}^{\top}\boldsymbol{X}\boldsymbol{1}_{N} + N\boldsymbol{\lambda}=\boldsymbol{0}
$$

而根据前面的定义，

$$\begin{aligned}
&\boldsymbol{X}\boldsymbol{1}_{N}= \Sigma (\boldsymbol{x}_{j}-\hat{\boldsymbol{\mu}}) = \boldsymbol{0}\\
&\Rightarrow \boldsymbol{\lambda} = \boldsymbol{0}\\
&\Rightarrow \frac{\partial \boldsymbol{\mathcal{L}}}{\partial \boldsymbol{Y}} = -2 \boldsymbol{U}^{\top} \boldsymbol{X} + 2 \boldsymbol{U}^{\top} \boldsymbol{UY} = \boldsymbol{0}\\
&\Rightarrow \boldsymbol{Y} = \boldsymbol{U}^{\top} \boldsymbol{X}\\
&\Rightarrow \boldsymbol{y}_{j} = \boldsymbol{U}^{\top}(\boldsymbol{x}_{j}-\hat{\boldsymbol{\mu}})
\end{aligned}
$$

几何意义就是，将$\boldsymbol{x}_{j}$ 在子空间中的点 $\boldsymbol{\mu}$ 的差向量 $\boldsymbol{x}_{j}-\hat{\boldsymbol{\mu}}$ 投影到该子空间上。

进一步地，拉格朗日函数：

$$\mathcal{L}=||\boldsymbol{X}-\boldsymbol{U}\boldsymbol{U}^{\top} \boldsymbol{X}||_{F}^{2} + <\boldsymbol{\Lambda},\boldsymbol{I}_{d}-\boldsymbol{U}^{T}\boldsymbol{U}>$$

将第一项展开：

$$\begin{aligned}
&||\boldsymbol{X}-\boldsymbol{U}\boldsymbol{U}^{\top} \boldsymbol{X}||_{F}^{2} \\
&= ||\boldsymbol{X}||_{F}^{2} - 2<\boldsymbol{X}, \boldsymbol{U}\boldsymbol{U}^{\top} \boldsymbol{X}> + ||\boldsymbol{U}\boldsymbol{U}^{\top} \boldsymbol{X}||_{F}^{2}\\
& =  ||\boldsymbol{X}||_{F}^{2} - 2 \rm{Tr}(\boldsymbol{X}^{\top} \boldsymbol{U}\boldsymbol{U}^{\top} \boldsymbol{X}) + ||\boldsymbol{U}\boldsymbol{U}^{\top} \boldsymbol{X}||_{F}^{2}\\
& = ||\boldsymbol{X}||_{F}^{2} - 2 \rm{Tr}(\boldsymbol{U}^{\top} \boldsymbol{X}\boldsymbol{X}^{\top}\boldsymbol{U}) + ||\boldsymbol{U}\boldsymbol{U}^{\top} \boldsymbol{X}||_{F}^{2}\\
\Rightarrow  \frac{\partial \boldsymbol{\mathcal{L}}}{\partial \boldsymbol{U}} &= -2 \boldsymbol{U}^{\top}(\boldsymbol{X}-\boldsymbol{UY}) + \boldsymbol{\lambda} \boldsymbol{1}^{\top}_{N} \\
&= -2\boldsymbol{X}\boldsymbol{X}^{\top}\boldsymbol{U} - 2\boldsymbol{U \Lambda} \\
&=\boldsymbol{0}\\
\Rightarrow  \boldsymbol{X}\boldsymbol{X}^{\top}\boldsymbol{U} &= - \boldsymbol{U \Lambda}
\end{aligned}$$

看到这个等式差不多要猜测这个很像是 $\boldsymbol{X}\boldsymbol{X}^{\top}$ 的特征值和特征向量。但是我们并没有说 $\boldsymbol{\Lambda}$ 是对角阵，但是已知 $\boldsymbol{\Lambda}$ 是对称阵，因此可以正交对角化成对称阵 $\tilde{\Lambda}$：

$$\tilde{\Lambda} = \boldsymbol{R\Lambda R}^{\top}$$

其中 $\boldsymbol{R}$ 是 orthonormal 的。将上式代入上上式就可以得到：

$$\boldsymbol{X}\boldsymbol{X}^{\top}\tilde{\boldsymbol{U}} = - \tilde{\boldsymbol{U}} \Lambda$$

其中 $\tilde{\boldsymbol{U}} = \boldsymbol{U}\boldsymbol{R}$ .

现在化简化简，只留变量相关的项，回到我们要优化的最终问题：

$$\begin{aligned}
\min -\rm{Tr} (\boldsymbol{X}^{\top} \boldsymbol{U}\boldsymbol{U}^{\top} \boldsymbol{X}) &\Leftrightarrow  
\min -<\boldsymbol{X}\boldsymbol{X}^{\top} \boldsymbol{U}, \boldsymbol{U}>\\
&\Leftrightarrow \min -\rm{Tr} (\boldsymbol{\Lambda} \boldsymbol{U}^{\top} \boldsymbol{U})\\
&\Leftrightarrow  \min -\rm{Tr} (\boldsymbol{\Lambda})\\
&\Leftrightarrow  \max \; \rm{sum\; of\; eigenvalues}\\
&\Rightarrow \rm{choose\; the\; largest\; d \; eigenvalues}
\end{aligned}$$

OK，最后我们得到如下 Geometric PCA 的定理，

**Theorem** 

$$\min_{\boldsymbol{\mu}, \boldsymbol{U}, \boldsymbol{y}_{j}} {\Sigma_{j=1}^{N} ||\boldsymbol{x}_{j}-\boldsymbol{\mu}-\boldsymbol{U} \boldsymbol{y}_{j}||_{2}^{2}} \;\; \rm{s.t.} \Sigma_{j} \boldsymbol{y}_{j}=0, \; \boldsymbol{U}^{T}\boldsymbol{U} =\boldsymbol{I}_{d}$$

的解就是

$$ \hat{\boldsymbol{\mu}}=\frac{1}{N} \Sigma_{j=1}^{N} \boldsymbol{x}_{j} \\
\hat{\boldsymbol{y}}_{j} = \boldsymbol{U}^{\top}(\boldsymbol{x}_{j}-\hat{\boldsymbol{\mu}}) \\
\boldsymbol{U} = [\boldsymbol{u}_{1}, ..., \boldsymbol{u}_{d}] \;\; \rm{are\; top\; d \; eigenvectors \; of}\; \boldsymbol{XX}^{\top}
$$

其中 

$$ \boldsymbol{X} = [\boldsymbol{x}_{1}-\hat{\boldsymbol{\mu}}, ..., \boldsymbol{x}_{N}-\hat{\boldsymbol{\mu}}]\\$$


仔细看一下这个定理，

$$\boldsymbol{XX}^{\top} = \Sigma_{j=1}^{N} (\boldsymbol{x}_{j}-\hat{\boldsymbol{\mu}})(\boldsymbol{x}_{j}-\hat{\boldsymbol{\mu}})^{\top} = N\Sigma_{N}$$

就是中心化后的数据样本的协方差矩阵啦。这样，我们就可以将 Geometric PCA 和 Statistical PCA 统一起来。也就是说，**Geometric PCA 的对象就是 Statistical PCA 中样本的协方差矩阵。**

最后补充一点，矩阵 $\boldsymbol{XX}^{\top}$ 是 $D\times D$ 的，我们做谱分解的计算复杂度较大，同时注意其 top 特征向量就是矩阵 $\boldsymbol{X}$ 的 top 奇异值对应的向量，而第二个矩阵的 size （D-by-N）更友好。