## Graph Convolutional Networks原文阅读和一些推导

原文链接：https://arxiv.org/abs/1609.02907 发表于ICLR2016.

<head>
    <script type="text/javascript" async
        src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.1/MathJax.js?config=TeX-AMS-MML_HTMLorMML">
    </script>
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

实际上，GCN的过程就是利用当前节点及其邻域节点的信息融合来生成对当前节点的表达，该表达可用于Node Classification Tasks；
再利用pooling技术对整张Graph的node representation information进行融合并提取其不变性，形成整张Graph的Representation，可用于解决Graph Classification Tasks。

#### Spatial-based GCN
基于空间的GCN一般是假设邻接矩阵中相邻节点之间的相似性，利用Clustering + Mean技术来完成graph中拓扑信息的融合和感受野的放大（类似于Pooling技术），
再利用全连接层对于节点特征维度进行放缩以达到最终的目的。

与CNN不同
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
即
$\mathbf{x}^T\mathbf{x} \geq -(D^{-\frac{1}{2}}\mathbf{x})^T A (D^{-\frac{1}{2}}\mathbf{x})$

$2 \mathbf{x}^T \mathbf{x} \geq \mathbf{x}^T \mathbf{x} - (D^{-\frac{1}{2}} \mathbf{x})^T A (D^{-\frac{1}{2}}\mathbf{x})$

$2 \mathbf{x}^T \mathbf{x} \geq \mathbf{x}^T (I - (D^{-\frac{1}{2}})^T A (D^{-\frac{1}{2}})) \mathbf{x}$

$2 \geq I - (D^{-\frac{1}{2}})^T A (D^{-\frac{1}{2}})$

易得$L_{sym} = I - (D^{-\frac{1}{2}})^T A (D^{-\frac{1}{2}}) \leq 2$。
证毕。

3、图卷积中的傅里叶变换
根据半正定矩阵分解，$L_{sym} = U \Lambda U^T$，其中$\Lambda$为L所有特征值组成的对角阵。
设$\lambda_i$为矩阵$L$的特征值，其所对应的特征向量为$\mathbf{u_i}$，则$\lambda_iL = L\mathbf{u_i}$。
根据简单的线性代数的知识可知，$U \in \mathcal{R}^{N \times N}$为归一化正交矩阵，又因为其线性无关，
所以$\\{\cdots, \mathbf{u_i}, \cdots \\}$为一组$\mathcal{R}^{N \times N}$空间中的正交基。

则我们给出图傅里叶变换（Graph Fourier Transform, GFT）：
对于任意一个图$G$上的信号$\mathbf{x}$，其傅里叶变换为：
$\mathbf{\tilde{x_k}} = \sum_i^N u_{ki}^T x_i = \<\mathbf{u_k}, \mathbf{x}\>$。
我们称特征向量$\mathbf{u}$为傅里叶基，$\mathbf{\tilde{x_k}}$为$\mathbf{x}$在傅里叶基上的傅里叶系数。
其实，我们可以看出傅里叶系数的本质就是$\mathbf{x}$在傅里叶基上的投影，衡量了$\mathbf{x}$与$\mathbf{u}$之间的相似度。
又$\mathbf{u}$为正交阵，有$\mathbf{u}\mathbf{u}^T = I$，则我们可以得到逆傅里叶变换$\mathbf{x} = \mathbf{u}\mathbf{\tilde{x_k}}$。
与常见时序信号傅里叶变换相似的，我们可以在图G的拉普拉斯矩阵的特征向量张成的特征空间中使用其傅里叶基来对G上的任意图信号$\mathbf{x}$进行分解。

下面，我们用一个恒等式来阐述图傅里叶变换与图频谱之间的关系：
$\mathbf{x}^T L \mathbf{x} = \mathbf{x}^T D \mathbf{x} - \mathbf{x}^T A \mathbf{x} = \sum_{i = 1}^n d_i x_i^2 - \sum_{i = 1}^n \sum_{j = 1}^n a_{ij} x_i x_j = \frac{1}{2} \left(\sum_{i = 1}^n d_i x_i^2 - 2 \cdot \sum_{i = 1}^n \sum_{j = 1}^n a_{ij} x_i x_j + \sum_{j = 1}^n d_j x_j^2 \right) = \frac{1}{2} \sum_{i = 1}^n \sum_{j = 1}^n a_{ij} (x_i - x_j)^2$，
上式推导中利用了$d_i = \sum_{j = 1}^n a_{ij}$。
又$\mathbf{x}^T L \mathbf{x} = \mathbf{x}^T (\mathbf{u} \Lambda \mathbf{u}^T) \mathbf{x} = (\mathbf{u}\mathbf{\tilde{x_k}})^T \mathbf{u} \Lambda \mathbf{u}^T
(\mathbf{u}\mathbf{\tilde{x_k}}) = \mathbf{\tilde{x_k}}^T \mathbf{u}^T \mathbf{u} \Lambda \mathbf{u}^T \mathbf{u} \mathbf{\tilde{x_k}} = \mathbf{\tilde{x_k}}^T \Lambda
\mathbf{\tilde{x_k}} = \sum_{k = 1}^N \lambda_k \mathbf{\tilde{x_k}}^2$，
假设$\Lambda$中的特征值从小打到大排序，$\mathbf{x}^T L \mathbf{x}$代表了信号$\mathbf{x}$的总差变，其刻画了信号整体的平滑度。
显然，其为$L$的特征值的线性组合。换句话说，与$\mathbf{x}$对应的$L$的特征值可以看作为当前信号$\mathbf{x}$的频率。
特征值越低，频率越低，对应的傅里叶基变化的越缓慢，相近节点上的信号值趋向于一致，反之亦然。

### Graph Filter
我们将图滤波器定义为：对给定图信号的频谱中各个频率分量的强度进行增强或者衰减的操作。
根据该定义，我们可以简单的将其定义为$H \in \mathcal{R}^{N \times N}$且$H: \mathcal{R}^N \rightarrow \mathcal{R}^N$:
$\mathbf{y} = H\mathbf{x} = \sum_{k = 1}^{N} (h(\lambda_k)\tilde{x_k})\mathbf{v_k}$。


