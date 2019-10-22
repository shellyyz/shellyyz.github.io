---
layout:     post
title:      R(AA^T)和SVD的补充
author:     Brianne Yao
# description: some common commands in Git.
# header-img: img/post-bg-ios9-web.jpg -->
catalog: 	 true
comments: true
tags:
    - Linear Algebra
    - Subspace Learning
---
{% include head.html %}

最近在上 Subspace Learning 这门课，每周 Quiz, 听课的时候感觉都挺明白的，quiz 的时候两行泪。

设 

$$A\in \mathbb{R}^{m\times n},$$

这个文章就简单说两件事情，一个是 

$$\rm{\mathcal{R}} {(AA^{\top})}=\rm{\mathcal{R}} {(A)}$$

（$\rm{\mathcal{R}}$ 为列空间 range space or column space）的证明；另一个是如何由 $AA^{\top}$ 的 Eigen Decomposition 

$$AA^{\top} = U \Lambda U^{\top}$$

得到 $A$ 的 SVD

$$A=U \Sigma V^{\top}$$

## $\rm{\mathcal{R}} {(AA^{\top})}=\rm{\mathcal{R}} {(A)}$

证明这个式子的方法有很多，现在用 nullity 的方法来证明。

首先，明显可以看出，

$$\rm{\mathcal{R}} (AA^{\top}) \subseteq \rm{\mathcal{R}} (A).$$

我们着重看另一个方向，其实只要证明

$$\rm{rank} (AA^{\top}) = \rm{rank} (A)$$

即可。由于 ( $\odot$ 表示两两 orthogonal 的 direct sum ) 

$$\rm{\mathcal{R}} (AA^{\top}) \odot \rm{\mathcal{N}} (AA^{\top}) = \rm{\mathcal{R}} (A)  \odot \rm{\mathcal{N}} (A^{\top}) = \mathbb{R}^{m},$$

我们只要证明

$$\rm{\mathcal{N}} (AA^{\top}) \subseteq \rm{\mathcal{N}} (A^{\top}).$$

$$
\begin{aligned}
& \forall \xi \in \rm{\mathcal{N}} (AA^{\top}), \; \xi \ne 0, \; AA^{\top} \xi = 0. \Rightarrow A^{\top} \xi \in \rm{\mathcal{N}} (A).\\
& \rm{Since} \; A^{\top} \xi \in \rm{\mathcal{R}} (A^{\top}), \; {\mathcal{N}} (A) \cap \rm{\mathcal{R}} (A^{\top}) = 0, \Rightarrow  A^{\top} \xi = 0.\\
& \Rightarrow \xi \in \rm{\mathcal{N}} (A^{\top}). \;\; \blacksquare
\end{aligned}
$$


## 从 Eigen Decomposition 到 SVD

给定 $AA^{\top}$ 的 Eigen Decomposition 

$$AA^{\top} = U \Lambda U^{\top}$$

如何得到 $A$ 的 SVD

$$A=U \Sigma V^{\top}$$

这里要注意几点

* $A$ 可能有重复的 singular value，也就是说 $AA^{\top}$ 有 geometry multiplicity 大于 1 的特征值对应的特征空间 eigen space。
* 即使在同一个奇异值对应的 $U$ 和 $V$，它们也不能任意对应。这是因为要满足

    $$U^{\top} A = \Sigma V^{\top}$$


设 $AA^{\top}$ 的 thin SVD 为

$$AA^{\top} = \bar{U} \bar{\Lambda} \bar{U}^{\top}$$

在第一部分我们已经证明，

$$\rm{\mathcal{R}} {(AA^{\top})}=\rm{\mathcal{R}} {(A)}$$

因此 $\bar{U}$ 也是 ${\mathcal{R}} {(A)}$ 的 orthogonal basis.

现在我们要做的是将 row space ${\mathcal{R}} {(A^{\top})}$ 的 orthogonal basis 和 $\bar{U}$ 联系起来。
可以证明，
$A^{\top} \bar{U}$
是互相垂直的，而又因为

$$\rm{dim} {{\mathcal{R}} {(A^{\top})}} = \rm{dim} {\mathcal{R}} {(A)}$$

因此 $A^{\top} \bar{U}$ 是 ${\mathcal{R}} {(A^{\top})}$ 的一组互相正交的基，但是模长一般不为1，而是
$\sqrt{\lambda_{i}}$.

* 证明：
  
  $$ \forall i\ne j, \; (A^{\top} u_{i})^{\top} (A^{\top} u_{j}) = u_{i} AA^{\top} u_{j} = u_{i}^{\top} U \Lambda U^{\top}  u_{j} = 0;\\
  \rm{otherwise} \; \; (A^{\top} u_{i})^{\top} (A^{\top} u_{i}) = \lambda_{i}. \; \blacksquare
  $$


将之二范数进行 normalize, 我们得到了 ${\mathcal{R}} {(A^{\top})}$ 的一组orthonormal basis:

$$ \bar{V}=\left[\frac{A^{\top} u_{1}}{||A^{\top} u_{1}||_2}, ..., \frac{A^{\top} u_{r}}{||A^{\top} u_{r}||_2}\right] = A^{\top} \bar{U} \; |\bar{\Lambda}|^{-1/2} = A^{\top} \bar{U} \bar{\Sigma}^{-1}$$

即

$$\bar{V} \bar{\Sigma}= A^{\top} \bar{U}$$

亦即

$$\bar{U}^{\top} A = \bar{\Sigma} \bar{V}^{\top}$$

而对于特征值为0的部分，只需要分别求出 $\bar{U}$ 和
$\bar{V}$ 的 orthocomplementary basis 就可以了，这部分它们之间的对应关系由于奇异值为0而非常自由。


