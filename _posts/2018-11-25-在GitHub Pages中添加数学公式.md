---
layout:     post
title:      在GitHub Pages中添加数学公式
subtitle:   GitHub博客中添加于我而言必不可少的公式的方法
date:       2018-11-25
author:     shellyyz
description: some common commands in Git.
# header-img: img/post-bg-ios9-web.jpg -->
catalog: 	 true
tags:
    - Git
---

# 在GitHub Pages中添加数学公式
1. 在根目录下的_config.yml文件中添加
```
markdown: kramdown
```
2. 在根目录下的_include文件夹下新建lib文件夹，然后在_include\lib中新建文件mathjax.html，编辑如下：
```
<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>
```
3. 在需要添加数学公式的文章的front下添加
```
{% i nclude lib/mathjax.html %}
```
文章中的行内公式用$$公式$$，行间公式用//(公式//)。

下面尝试一下：
$$min_{x} f(x)$$
或者行间插入公式//(min_{x} f(x)//).
