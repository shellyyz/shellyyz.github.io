---
layout:     post
title:      Add MathJax into GitHub Pages
subtitle:   在GitHub Pages中添加数学公式
date:       2018-11-28
author:     shellyyz
# description: some common commands in Git.
# header-img: img/post-bg-ios9-web.jpg -->
catalog: 	 true
tags:
    - Git
    - Commands
---
{% include head.html %}
# 在GitHub Pages中添加数学公式
$$E = m c^2$$
1. 首先在根目录中的_config.yml文件添加

```
markdown: kramdow
```

2. 然后在根目录中新建_includes文件夹（注意有s），在_includes中新建head.html文件。内写MathJax官方的文档：

```
<head>
    <script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>
    <script type="text/x-mathjax-config">
        MathJax.Hub.Config({
            tex2jax: {
            skipTags: ['script', 'noscript', 'style', 'textarea', 'pre'],
            inlineMath: [['$','$']]
            }
        });
    </script>
</head>
```

3. 在需要添加公式的.md文章中在头文件下添加一句

```
{% include head.html %}
```

----
下面来test一下效果。行间公式\\(min_{x} f(x)\\),行间公式$$min_{x} f(x)$$.
