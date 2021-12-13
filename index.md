## Graph Convolutional Networks原文阅读和一些推导

原文链接：https://arxiv.org/abs/1609.02907 发表于ICLR2016.

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

### 图的作用

图 (Graph) 是一种非常简单且有效的数据结构，其一般用于构建对象之间的拓扑关系。一般我们用集合$V$ (Vertex) 来保存单个对象信息，即一个节点；用$E$ (Edge) 来保存对象（节点）之间的关系信息，即节点之间的拓扑关系。我们用邻接矩阵 (adjacency matrix) $A$来表达节点之间的拓扑关系，这里我们只考虑无向简单图，即边的权重为1且边上无方向。除了$A$，我们用对角阵$D$来表达节点的度，即与当前节点$v_i$相连接的的节点的个数，$D_{ii} = \sum_{j} A_{ij}$。

由于图结构本身并没有图像那样标准而简单空间结构，为了实现卷积，即节点$v_i$与其相连的节点交互，我们引入了拉普拉斯算子（矩阵）$L = D - A$。由上述对$D$和$A$的描述，我们可以得到拉普拉斯矩阵的对角线上为节点$v_i$的度，非角线位置$L_{ij} = -1$ if $(v_i, v_j) \in E$ else 0, $i \neq j$。

下面我们对拉普拉斯矩阵$L$及其归一化形式$L_{sym}$的性质进行一些讨论。

```markdown
Syntax highlighted code block

# Header 1
## Header 2
### Header 3

- Bulleted
- List

1. Numbered
2. List

**Bold** and _Italic_ and `Code` text

[Link](url) and ![Image](src)
```

For more details see [Basic writing and formatting syntax](https://docs.github.com/en/github/writing-on-github/getting-started-with-writing-and-formatting-on-github/basic-writing-and-formatting-syntax).

### Jekyll Themes

Your Pages site will use the layout and styles from the Jekyll theme you have selected in your [repository settings](https://github.com/whzyf951620/GCN-/settings/pages). The name of this theme is saved in the Jekyll `_config.yml` configuration file.

### Support or Contact

Having trouble with Pages? Check out our [documentation](https://docs.github.com/categories/github-pages-basics/) or [contact support](https://support.github.com/contact) and we’ll help you sort it out.
