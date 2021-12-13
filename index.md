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

### GNN (Graph Neural Network)

GNN的目标是在$l_{th}$层使用神经网络学习一个映射$f: A, H^{(l)} \rightarrow H^{(l+1)}$，其中$H^{(l)}$为$l_{th}$层网络的输入，$H^{(l+1)}$为$l_{th}$层网络的输出。

### GCN (Graph Convolutional Network)
与GNN不同
由于图结构本身并没有图像那样标准而简单空间结构，为了实现卷积，即节点$v_i$与其相连的节点交互，我们引入了拉普拉斯算子（矩阵）$L = D - A$。由上述对$D$和$A$的描述，我们可以得到拉普拉斯矩阵的对角线上为节点$v_i$的度，非角线位置$L_{ij} = -1$ if $(v_i, v_j) \in E$ else 0, $i \neq j$。

下面我们对拉普拉斯矩阵$L$及其归一化形式$L_{sym}$的性质进行一些讨论。

首先，我们给出$L_{sym}$的定义：
$L_{sym} = D^{-\frac{1}{2}}LD^{-\frac{1}{2}} = I - D^{-\frac{1}{2}}AD^{-\frac{1}{2}}$。
其次，我们需要证明$L$的半正定性，即证明对于任意非零向量$\mathbf{x}$，其二次型$\mathbf{x}^T L \mathbf{x} \geq 0$。
证明：
根据矩阵分解，$L = U^T \Lambda U$，其中$\Lambda$为L所有特征值组成的对角阵。
设$\lambda_i$为矩阵$L$的特征值，其所对应的特征向量为$\mathbf{x_i}$，则$L\lambda_i = L\mathbf{x_i}$。
要证明半正定，即证明

For more details see [Basic writing and formatting syntax](https://docs.github.com/en/github/writing-on-github/getting-started-with-writing-and-formatting-on-github/basic-writing-and-formatting-syntax).

### Jekyll Themes

Your Pages site will use the layout and styles from the Jekyll theme you have selected in your [repository settings](https://github.com/whzyf951620/GCN-/settings/pages). The name of this theme is saved in the Jekyll `_config.yml` configuration file.

### Support or Contact

Having trouble with Pages? Check out our [documentation](https://docs.github.com/categories/github-pages-basics/) or [contact support](https://support.github.com/contact) and we’ll help you sort it out.
