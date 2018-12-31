## 3. Theory

Deep subspace clustering is composed by three parts: spectral clustering, self-expressiveness and autoencoder.

### 3.1 Spectral Clustering

Spectral clustering algorithm is based on Laplacian Eigenmaps. Given a set of data points $x_1$, . . . $x_n$ and some notion of similarity s >= 0 between all pairs of data points $x_i$ and $x_j$ , the goal of clustering is to divide the data points into several groups such that points in the same group are similar and points in different groups are dissimilar to each other. In spectral clustering, data points are represented by a graph $G(V,E)$, where $V$ is the entire set of vertexes(a vertex is a data point) and $E$ is the set of edges between vertexes. Weights of edges can be presented by similarities between data points. The algorithm is to cut the graph into several sub-graphs such that the edges between different sub-graphs have low weights (points in different clusters are dissimilar from each other) and the edges within a group have high weights (points in the same cluster are similar to each other).

The algorithm cuts the graph using Laplacian Matrix, which is defined as :

$L=D-W$

where $D$ is is diagonal weight matrix and $W$ is weight matrix. The diagonal elements of $D$ is the sum of weights in their rows(or column) in $W$, that is:

$D_{ii} = \Sigma_{j} W_{ij}$

Given laplacian matrix $L$, the spectral clustering can be processed by following steps:

1. Compute the first k-smallest eigenvectors $u_1$, $u_2$, ..., $u_k$ of $L$
2. Let $U\in R^{n\times k}$ be the matrix containing the vectors $u_1$, $u_2$, ..., $u_k$ as columns.
3. For $i=1,..,n$, let $y_i\in R^{k}$ be the vector corresponding to the $i-th$ row of $U$.
4. Clusters the points $(y_i)_{i=1,...,n}$ in $R^{k}$ with the $k-means$ algorithm or using discrete method into $C_{1},...,C_{k}$

The intuition of spectral clustering is to map the data points to low-dimension subspaces. The relative similarities between data points in subspaces are similar to which in original spaces. Actually, given a laplacian matrix, $y^{T}Ly$ can be infered to $\Sigma_{i,j}w_{i,j}\parallel y_i - y_j \parallel^2$. Here $\parallel y_i - y_j \parallel^2$ is the distance between $i_{th}\ point$ and $j_{th}\ point$ in subspaces, and the $w_{i,j}$ is a penalty item. Given a $L$, we can minimize $y^{T}Ly$ to get the mapping which keeps the relative similarities between data points in original space. We can use the theory of rayleigh quotient to solove this problem because $L$ is a symmetric matrix. And that is why we compute the k-smallest eigenvectors of $L$.

### 3.2 Self-Expressiveness

The intuition of self-expressiveness is that a data point can be represented as the linear combination of all the other data points. It can be represented by $X=XC$ where $C$ is the self-representation coefficient matrix and $diag(C)=0$. To make this representation tight, the current data point should only use the points in the same subspace, that is to say in the same cluster. It can be achieved by minimize the $\parallel C \parallel_{p}\ s.t X=XC,\ diag{C}=0$, where $\parallel.\parallel_{p}$ is a arbitrary matrix norm. When noise is not considered, it has been proven that if we permute orders of elements in $C$ based on their clusters, the solution of minimization is block-diagonal with non-zero blocks corresponding to points in the same subspaces. 

It can be observed that the $C$ can be the $W$ in spectral clustering. Coefficients between points in the same cluster are bigger than that between point in different clusters. However, the self-expressiveness property only holds for linear subspaces. The traditional way of non-linear mapping is kernal based method. But the pre-defined kernel functions usually not suit the distribution well. So Pan Ji et al. proposed $Deep\ Subspace\ Clustering\ Networks$, which uses AutoEncoder to learn a more efficient mapping.

### 3.3 Architecture of Auto-Encoders

The traditional auto-encoders has the architecture as follows:

![](traditional_ae.png)

The input $x_i$ is mapped to $z_i$ through an encoder, and then reconstructed as $\hat{x_{i}}$ through a decoder. By this way, we can achieve the target of feature retrievement or dimension reduction. In this paper, author introduced a new layer that encodes the notion of self-expressiveness. The output of encoder is not treated as the input of the decoder directly but goes through a new self-expressiveness layer. The proposed architecture is:

![](proposed_net.png)

The output of encoder $Z_{\Theta_{e}}$ multiplies the coefficient matrix $\Theta_{S}$ by goes through the self-expressiveness layer. And decoder takes $Z_{\Theta_{e}}\Theta_{S}$ as input. To encode self-expressiveness, the author introduced a new loss function defined as:

$L(\Theta C)=\frac{1}{2}\parallel X-\hat{X}_{\Theta}\parallel^2_{F}+\lambda_{1}\parallel C\parallel_{p} + \frac{\lambda_2}{2}\parallel Z_{\Sigma_e}-Z_{\Sigma_e}C\parallel^2_F \ s.t.\ diag(C)=0$

where $\hat{X}_{\Theta}$ represents the data reconstructed by the auto-encoder. The second term of loss function is to find the tightest mapping and the third is to guarantee the self-expressiveness property of the new layer. And $C$ is treated as the parameters of an additional network layer, which lets us solve for $C$ using backpropagation.

## 4. Recurrence， Extension and Comparation

## 5. Discussion

