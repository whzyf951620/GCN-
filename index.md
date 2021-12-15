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

图 (Graph) 是一种非常简单且有效的数据结构，其一般用于构建对象之间的拓扑关系。一般我们用集合$V \in \mathcal{R}^{N}$（Vertex）来保存单个对象信息，即一个节点；用$E \in \mathcal{R}^{N \times N}$ (Edge) 来保存对象（节点）之间的关系信息，即节点之间的拓扑关系。我们用邻接矩阵 (adjacency matrix) $A$来表达节点之间的拓扑关系，这里我们只考虑无向简单图，即边的权重为1且边上无方向。除了$A$，我们用对角阵$D$来表达节点的度，即与当前节点$v_i$相连接的的节点的个数，$D_{ii} = \sum_{j} A_{ij}$。

使用图神经网络的意义：现实生活中，像图像这样具有严格、致密拓扑结构的数据并不多，绝大多数的数据之间只存在少量、无规律的联系。图结构的意义在于，其可以在神经网络的训练过程中保持数据之间的拓扑结构，且其保持的无规律拓扑结构可以是单个样本中的信息，也可以是样本之间的拓扑结构，该特性也是卷积神经网络（CNNs）所不具备的。

### GNN (Graph Neural Network)

GNN的目标是在$l_{th}$层使用神经网络学习一个映射$f: (A, H^{(l)}) \rightarrow H^{(l+1)}$，其中$H^{(l)}$为$l_{th}$层网络的输入，$H^{(l+1)}$为$l_{th}$层网络的输出。

### GCN (Graph Convolutional Network)
与GNN不同
由于图结构本身并没有图像那样标准而简单空间结构，为了实现卷积，即节点$v_i$与其相连的节点交互，我们引入了拉普拉斯算子（矩阵）$L = D - A$。由上述对$D$和$A$的描述，我们可以得到拉普拉斯矩阵的对角线上为节点$v_i$的度，非角线位置$L_{ij} = -1$ if $(v_i, v_j) \in E$ else 0, $i \neq j$。

下面我们对拉普拉斯矩阵$L$及其归一化形式$L_{sym}$的性质进行一些讨论。

1、我们需要证明$L$的半正定性，即证明对于任意非零向量$\mathbf{x}$，其二次型$\mathbf{x}^T L \mathbf{x} \geq 0$，
其可等价于$L$的所有特征值都大于等于0。
证明1：
首先构造矩阵$G^{ij} \in \mathcal{R}^{N \times N}$，其构造方法为：$G_{ii}^{ij} = 1$，$G_{jj}^{ij} = 1$，$G_{ij}^{ij} = G_{ji}^{ij} = -1$，其余位置为0。
其中上标为矩阵的名称，下标为矩阵中元素的序号。
则易得$L = \sum_i \sum_j G^{ij}$。
又因为$\mathbf{x}^TG^{ij}\mathbf{x}^T = \mathbf{x}^T \cdot \[\cdots, \mathbf{x_i} - \mathbf{x_j}, \cdots, \mathbf{x_j} - \mathbf{x_i}, \cdots\]^T$。
上式又可以化简为$\mathbf{x_i}(\mathbf{x_i} - \mathbf{x_j}) + \mathbf{x_j}(\mathbf{x_j} - \mathbf{x_i}) = (\mathbf{x_i} - \mathbf{x_j})^2 \geq 0$。
则易得我们证明的目标$\mathbf{x}^T L \mathbf{x} = \mathbf{x}^T (\sum_i \sum_j G^{ij}) \mathbf{x} = \sum_i \sum_j (\mathbf{x_i} - \mathbf{x_j})^2 \geq 0$。
证毕。

2、我们给出$L_{sym}$的定义：
$L_{sym} = D^{-\frac{1}{2}}LD^{-\frac{1}{2}} = I - D^{-\frac{1}{2}}AD^{-\frac{1}{2}}$。
对于$L_{sym}$，我们给出两条性质的证明：
（1）$L_{sym}$的半正定性；
（2）$L_{sym}$的特征值范围为$\[0, 2\]$，该范围保证了在使用$L_{sym}$时，GCN不会出现梯度爆炸的情况。

证明2、(1):
与$L$证明相似，我们首先构造矩阵集合$G = \\{G^{i, j}\\}^{N, N}_{i, j}$。

由于$\mathbf{x}^TL_{sym}\mathbf{x}^T = (\mathbf{x}D^{-\frac{1}{2}})^T L (\mathbf{x}D^{-\frac{1}{2}})$。
根据证明1，易得上式为$\sum_i \sum_j (\frac{\mathbf{x_i}}{\sqrt{d_i}} - \frac{\mathbf{x_i}}{\sqrt{d_j}})^2 \geq 0$，
其中，$d_i$ 为矩阵$D$中对角线上的第$i$个元素。

证明2、(2):
我们首先构造矩阵集合$S = \\{S^{i, j}\\}^{N, N}_{i, j}$。

其中$S^{i, j}$与$G^{i, j}$类似，唯一不同的地方在于，$S_{ij}^{ij} = S_{ji}^{ij} = 1$。
与上述证明类似，$\mathbf{x_i}(\mathbf{x_i} + \mathbf{x_j}) + \mathbf{x_j}(\mathbf{x_j} + \mathbf{x_i}) = (\mathbf{x_i} + \mathbf{x_j})^2 \geq 0$。
由于$L = D - A = \sum_i \sum_j G^{ij}$，而$\sum_i \sum_j S^{ij} = D + A$。
我们假设$S_{sym} = D^{-\frac{1}{2}}SD^{-\frac{1}{2}} = I + D^{-\frac{1}{2}}AD^{-\frac{1}{2}}$。
根据证明2、(1)，易得$\mathbf{x}^TS_{sym}\mathbf{x}^T = \sum_i \sum_j (\frac{\mathbf{x_i}}{\sqrt{d_i}} + \frac{\mathbf{x_i}}{\sqrt{d_j}})^2 \geq 0$。
又因为$\mathbf{x}^TS_{sym}\mathbf{x}^T = \mathbf{x}^T\mathbf{x} + (\mathbf{x}D^{-\frac{1}{2}})^T A (\mathbf{x}D^{-\frac{1}{2}})$，显然其大于等于0，
即$\mathbf{x}^T\mathbf{x} \geq -(D^{-\frac{1}{2}}\mathbf{x})^T A (D^{-\frac{1}{2}}\mathbf{x})$，

$2 \mathbf{x}^T \mathbf{x} \geq \mathbf{x}^T \mathbf{x} - (D^{-\frac{1}{2}} \mathbf{x})^T A (D^{-\frac{1}{2}}\mathbf{x})$，
$2 \mathbf{x}^T \mathbf{x} \geq \mathbf{x}^T \mathbf{x} - (D^{-\frac{1}{2}} \mathbf{x})^T A (D^{-\frac{1}{2}}\mathbf{x})$，

$2 \mathbf{x}^T \mathbf{x} \geq \mathbf{x}^T (I - D^{-\frac{1}{2}}^T A D^{-\frac{1}{2}}) \mathbf{x}$，

$2 \geq I - D^{-\frac{1}{2}}^T A D^{-\frac{1}{2}}$，

$L_{sym} = I - D^{-\frac{1}{2}}^T A D^{-\frac{1}{2}} \leq 2$。
证毕。

根据矩阵分解，$L = U^T \Lambda U$，其中$\Lambda$为L所有特征值组成的对角阵。
设$\lambda_i$为矩阵$L$的特征值，其所对应的特征向量为$\mathbf{x_i}$，则$L\lambda_i = L\mathbf{x_i}$。
要证明半正定，即证明对于任意非零向量$\mathbf{x}$，其瑞利熵（Rayleigh quotient）都大于等于0，即$\frac{\mathbf{x}^TL\mathbf{x}}{\mathbf{x}^T\mathbf{x}} \geq 0$。
根据瑞利定理（Rayleigh theorem），$\lambda_{min} \leq \frac{\mathbf{x}^TL\mathbf{x}}{\mathbf{x}^T\mathbf{x}} \leq \lambda_{max}$，
且其最大值最小值都在非零向量$\mathbf{x}$为$L$的特征向量时取得。则正定性证明可以转化为：当$\mathbf{x}$为任意$L$的特征向量时，其瑞利熵大于等于0恒成立。

$\frac{\mathbf{x}^TL\mathbf{x}}{\mathbf{x}^T\mathbf{x}} = \frac{\mathbf{x}^T (\lambda\mathbf{x})}{\mathbf{x}^T\mathbf{x}} = \frac{\lambda(\mathbf{x}^T\mathbf{x})}{\mathbf{x}^T\mathbf{x}}$
