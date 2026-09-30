# 高效三阶格点型有限体积算法：数学推导、数据结构与移植实现指南

> 本文用于在另一套 CFD 程序中独立实现 DNDSR `NCFV` 模块的
> `EfficientDifferential` 算法。内容以马润之硕士学位论文第 3、4 章为理论
> 基础，并以当前 DNDSR 实现为可执行定义。文中会明确区分“论文核心算法”
> 与“DNDSR 工程实现选择”。

本文核对口径为 2026-09-23 的工作区版本。理论主线对应论文第 3.1--3.3 节
和第 4.1--4.3 节；紧致模板选择、exact node halo、宏面 Roe 冻结方式以及
变分重构属于当前 DNDSR 的工程实现或扩展，不应误认为都是论文唯一规定。

## 1. 算法目标与实现边界

高效算法仍然是三阶格点型有限体积方法。它与传统三阶方法的主要差别不是
降低重构多项式次数，而是改变高阶信息的保存方式和积分方式：

1. 使用完整二次多项式最小二乘问题求格点一阶导数，保证一阶导数具有二阶
   精度；求解后只保存逆算子中与一阶导数相关的行。
2. 不显式计算和存储守恒量 Hessian 或二阶导数。
3. 利用“二次多项式的梯度是一次函数”这一性质，以构造点处的一阶梯度差
   代替 Hessian 与位移的乘积。
4. 初始化时把体积分、界面状态平均、物理通量积分所需的几何关系压缩成
   “节点编号 + 一阶导数权重”。
5. 时间推进时不建立 Gauss 点，不保存 Gauss 点坐标，也不在积分点重复求解
   黎曼问题；每条原始网格边对应的对偶宏面只求解一次黎曼问题。

高效算法的最小运行期未知量为：

- 节点对偶控制体均值 \(\bar{\boldsymbol U}_i\)；
- 恢复后的原始网格节点值 \(\boldsymbol U_i\)；
- 节点一阶导数 \(\nabla\boldsymbol U_i\)；
- 必要时的限制器系数 \(\phi_i\)。

标准高效最小二乘路径不需要保存二次系数。DNDSR 还提供了变分重构扩展；
该扩展会保留完整二次系数用于变分迭代和节点值恢复，但高效通量积分仍只
读取节点值和一阶导数。若目标是首先复现论文中的低存储方案，应优先实现
`SVDLeastSquares + EfficientDifferential`，再考虑变分扩展。

## 2. 符号约定

设：

- \(\mathcal T\)：原始 O1 非结构网格；
- \(i,j\)：原始网格节点编号；
- \(V_i\)：以原始节点 \(i\) 为中心的对偶控制体；
- \(|V_i|\)：对偶控制体体积，二维时表示面积；
- \(e=(i,j)\)：原始网格边；
- \(\Gamma_e\)：原始边 \(e\) 对应的对偶宏面；
- \(K\)：组成对偶体或对偶面的微单纯形；
- \(|K|\)：微单纯形的体积、面积或长度；
- \(\boldsymbol A_K=|K|\boldsymbol n_K\)：微面的有向面积向量；
- \(S_e=\sum_{K\subset\Gamma_e}|K|\)：宏面的标量面积；
- \(\boldsymbol A_e=\sum_{K\subset\Gamma_e}\boldsymbol A_K\)：宏面的合面积向量；
- \(\boldsymbol n_e=\boldsymbol A_e/\|\boldsymbol A_e\|\)：宏面法向；
- \(\bar{\boldsymbol U}_i\)：\(V_i\) 上守恒变量平均值；
- \(\boldsymbol U_i\)：原始节点位置 \(\boldsymbol x_i\) 的恢复点值；
- \(\boldsymbol G_i=\nabla\boldsymbol U_i\)：节点守恒变量梯度；
- \(d\)：空间维数，取 2 或 3。

节点未知量是对偶控制体平均值：

\[
\bar{\boldsymbol U}_i=
\frac{1}{|V_i|}\int_{V_i}\boldsymbol U\,dV.
\]

半离散守恒方程为

\[
\boxed{
\frac{d\bar{\boldsymbol U}_i}{dt}=
-\frac1{|V_i|}
\left[
\sum_{e\in E(i)}B_{ie}\widehat{\boldsymbol I}_e+
\boldsymbol I_{\partial V_i}
\right].
}
\]

内部宏面通量 \(\widehat{\boldsymbol I}_e\) 只计算一次，两个端点通过相反的
\(B_{ie}\) 使用同一份结果，这是离散守恒和并行实现的基础。

必须同时保存 \(S_e\) 和 \(\boldsymbol A_e\)。对于不共面的三维宏面，通常有

\[
\|\boldsymbol A_e\|\ne S_e.
\]

不能用其中一个替代另一个。DNDSR 用 \(S_e\) 计算宏面平均和耗散项，用
\(\boldsymbol A_e\) 计算零阶物理通量和宏面法向。

## 3. 对偶控制体的唯一几何定义

### 3.1 构造点必须是原始顶点的算术平均

设原始边、面、体的顶点集合分别为 \(V(e),V(f),V(c)\)，构造点定义为

\[
\boldsymbol x_e=\frac{\boldsymbol x_i+\boldsymbol x_j}{2},
\qquad
\boldsymbol x_f=\frac{1}{N_f}\sum_{a\in V(f)}\boldsymbol x_a,
\qquad
\boldsymbol x_c=\frac{1}{N_c}\sum_{a\in V(c)}\boldsymbol x_a.
\]

这里的面点和体点是“原始角点坐标的等权平均”，不是面积质心或体积质心。
在四边形、六面体、棱柱和金字塔上，这两类点一般并不相同。移植时如果改用
几何质心，后续所有仿射系数和预计算权重都会改变，不再是本文所述算法。

建议把每个构造点同时表示成坐标和对原始节点的仿射展开：

\[
\boldsymbol y_a=\sum_m C_{am}\boldsymbol x_m,
\qquad \sum_m C_{am}=1.
\]

例如边中点的两个非零系数都是 \(1/2\)，面平均点的非零系数都是
\(1/N_f\)，体平均点的非零系数都是 \(1/N_c\)。这些 \(C_{am}\) 既用于
坐标生成，也用于构造点梯度插值，二者必须完全一致。

### 3.2 二维微几何

对于节点 \(i\) 的每一条拓扑链 `(cell, edge, node)`：

- 微控制体为三角形

\[
K_V=\operatorname{conv}
(\boldsymbol x_i,\boldsymbol x_e,\boldsymbol x_c).
\]

- 原始边 \(e=(i,j)\) 在每个相邻单元内贡献一条内部对偶微面

\[
K_S=\operatorname{conv}(\boldsymbol x_e,\boldsymbol x_c).
\]

- 若原始边位于物理边界，节点控制体还需要边界闭合线段

\[
K_B=\operatorname{conv}(\boldsymbol x_i,\boldsymbol x_e).
\]

### 3.3 三维微几何

对于节点 \(i\) 的每一条拓扑链 `(cell, face, edge, node)`：

- 微控制体为四面体

\[
K_V=\operatorname{conv}
(\boldsymbol x_i,\boldsymbol x_e,\boldsymbol x_f,\boldsymbol x_c).
\]

- 原始边 \(e=(i,j)\) 在每一个包含该边的 `(cell, face)` 链上贡献一个内部
  对偶微三角面

\[
K_S=\operatorname{conv}
(\boldsymbol x_e,\boldsymbol x_f,\boldsymbol x_c).
\]

- 位于物理边界的原始面为节点控制体贡献边界微三角面

\[
K_B=\operatorname{conv}
(\boldsymbol x_i,\boldsymbol x_e,\boldsymbol x_f).
\]

一个原始边周围的全部 \(K_S\) 合并成一个宏面 \(\Gamma_e\)。该宏面是两个
端点节点控制体的公共界面，也是守恒通量的唯一计算位置。

### 3.4 边界与周期控制体

物理边界上的节点控制体是由内部微体和边界微面闭合形成的“半控制体”，不与
域外虚拟半体拼接。边界微面保留其边界分区编号，用于选择远场、入口、出口、
滑移壁面或无滑移壁面条件。

平移周期边界的推荐处理方式不同：在分区前合并一一周期节点拓扑，使周期两侧
在商空间中成为普通相邻关系。这样周期面不再作为物理边界通量处理，跨周期的
原始边、宏面和对偶控制体都按内部拓扑构造。

若三个方向都周期，坐标运算必须采用最小镜像位移：

\[
\Delta x_\alpha\leftarrow
\Delta x_\alpha-L_\alpha
\operatorname{round}(\Delta x_\alpha/L_\alpha).
\]

这使跨周期面、周期棱和周期体对角线的邻接都能在同一局部坐标框架中生成。
不能直接对周期盒两端的绝对坐标求平均或求包围盒。

### 3.5 微单纯形度量和方向

二维三角形和三维四面体的度量分别为

\[
|K_2|=\frac12\| (\boldsymbol y_1-\boldsymbol y_0)
\times(\boldsymbol y_2-\boldsymbol y_0)\|,
\]

\[
|K_3|=\frac16\left|
\det[\boldsymbol y_1-\boldsymbol y_0,
     \boldsymbol y_2-\boldsymbol y_0,
     \boldsymbol y_3-\boldsymbol y_0]
\right|.
\]

内部微面的面积向量统一朝原始边第一个端点到第二个端点的方向。节点—边关联
保存 \(B_{ie}=\pm1\)，节点残差通过同一份宏面通量乘不同符号累加。这样跨
MPI 分区边也只需要计算一份通量，即可保证机器精度意义下的离散守恒。

初始化结束时应检查每个节点控制体的面积闭合：

\[
\sum_{e\in E(i)}B_{ie}\boldsymbol A_e
+\sum_{b\subset\partial V_i}\boldsymbol A_b=\boldsymbol 0.
\]

建议使用

\[
\varepsilon_i=
\frac{\left\|\sum B_{ie}\boldsymbol A_e+\sum\boldsymbol A_b\right\|}
{\sum_e S_e+\sum_b S_b}
\]

作为无量纲检查量。

## 4. 微单纯形的解析几何矩

设一个 \(q\) 维微单纯形有 \(m=q+1\) 个顶点 \(\boldsymbol y_a\)，令

\[
\boldsymbol s=\sum_{a=0}^{m-1}\boldsymbol y_a.
\]

零、一、二阶矩可以直接解析计算：

\[
M_0=\int_K1\,dK=|K|,
\]

\[
\boldsymbol M_1=\int_K\boldsymbol x\,dK
=\frac{|K|}{m}\boldsymbol s,
\]

\[
\boldsymbol M_2=\int_K\boldsymbol x\boldsymbol x^T\,dK
=\frac{|K|}{m(m+1)}
\left(\boldsymbol s\boldsymbol s^T+
\sum_a\boldsymbol y_a\boldsymbol y_a^T\right).
\]

对节点对偶体，将全部微体矩累加即可。实际程序最好直接使用相对于节点
\(\boldsymbol x_i\) 的局部坐标累积

\[
\int_{V_i}(\boldsymbol x-\boldsymbol x_i)\,dV,
\qquad
\int_{V_i}(\boldsymbol x-\boldsymbol x_i)
(\boldsymbol x-\boldsymbol x_i)^T\,dV,
\]

以避免细网格上由绝对坐标大数相减造成的消减误差。高效模式计算这些矩时
不需要任何 Gauss 点。

## 5. 只使用一阶导数的二次精确积分

这一节是整个高效算法的核心，可同时用于体积分、界面状态积分和物理通量
积分。

### 5.1 Taylor 展开与精确单纯形矩

在任意锚点 \(\boldsymbol x_i\) 展开一个二次函数 \(u\)：

\[
u(\boldsymbol x)=u_i+
\boldsymbol g_i\cdot\boldsymbol r+
\frac12\boldsymbol r^T\boldsymbol H\boldsymbol r,
\qquad
\boldsymbol r=\boldsymbol x-\boldsymbol x_i.
\]

记微单纯形顶点相对锚点的位移和位移和为

\[
\boldsymbol d_a=\boldsymbol y_a-\boldsymbol x_i,
\qquad
\boldsymbol D=\sum_a\boldsymbol d_a.
\]

代入第 4 节的矩公式，有

\[
\int_Ku\,dK=
|K|u_i+\frac{|K|}{m}\boldsymbol D\cdot\boldsymbol g_i
+\frac{|K|}{2m(m+1)}
\left[
\boldsymbol D^T\boldsymbol H\boldsymbol D+
\sum_a\boldsymbol d_a^T\boldsymbol H\boldsymbol d_a
\right].
\]

### 5.2 用梯度全微分消除 Hessian

对二次函数，梯度是一次函数，因此

\[
\boldsymbol H\boldsymbol d_a
=\boldsymbol g(\boldsymbol y_a)-\boldsymbol g_i
\]

是严格恒等式。对一般光滑函数，该式保留到所需截断精度。将其代入上式并
整理，得到

\[
\int_Ku\,dK=
|K|u_i+
\frac{|K|}{2m}\boldsymbol D\cdot\boldsymbol g_i
+\frac{|K|}{2m(m+1)}
\sum_a(\boldsymbol D+\boldsymbol d_a)
\cdot\boldsymbol g(\boldsymbol y_a).
\]

构造点梯度按与坐标相同的仿射系数插值：

\[
\boldsymbol g(\boldsymbol y_a)=
\sum_jC_{aj}\boldsymbol g_j.
\]

因为二次函数的梯度为一次函数，只要构造点坐标本身满足同一仿射关系，这个
插值对二次函数就是精确的。

### 5.3 压缩成“节点编号 + 梯度权重”

代入仿射插值后，积分写成

\[
\boxed{
\int_Ku\,dK=|K|u_i+
\sum_j\boldsymbol\ell_{ij}^{K}\cdot\boldsymbol g_j
}
\]

其中一个便于理解的权重形式是

\[
\boxed{
\boldsymbol\ell_{ij}^{K}=
\frac{|K|}{2m}\boldsymbol D\,\delta_{ij}
+\frac{|K|}{2m(m+1)}
\sum_a C_{aj}(\boldsymbol D+\boldsymbol d_a)
}.
\]

DNDSR 中使用的等价实现形式为

\[
\boldsymbol\ell_{ij}^{K}=
\frac{|K|}{m}\boldsymbol D\,\delta_{ij}
+\frac{|K|}{2m(m+1)}
\left[
\boldsymbol D\sum_a(C_{aj}-\delta_{ij})+
\sum_a\boldsymbol d_a(C_{aj}-\delta_{ij})
\right].
\]

两个形式代数完全相同。这里锚点不必是微单纯形顶点，因此同一套公式既能
处理包含节点 \(i\) 的微体，也能处理不包含端点节点的内部对偶微面。

### 5.4 建议的权重生成伪代码

```text
input:
    micro-simplex vertices y[a]
    affine support C[a][j]
    anchor node i and coordinate x[i]
    measure |K|

m = number_of_vertices(K)
d[a] = y[a] - x[i]
D = sum_a d[a]

for each unique support node j plus anchor i:
    sumC  = sum_a C[a][j]
    sumDC = sum_a C[a][j] * d[a]
    delta = 1 if j == i else 0

    ell = |K|/m * D * delta
    ell += |K|/(2*m*(m+1)) *
           (D * (sumC - m*delta) +
            (sumDC - delta*D))

    if norm(ell) is not negligible:
        accumulated_weight[j] += ell
```

构造阶段可以用哈希表累加，完成后应压缩成按本地节点编号排序的连续数组或
CSR，而不建议在每个时间步中反复使用 `unordered_map`。连续数组能减少哈希
开销、提高缓存命中率，也便于一次性收集 MPI ghost 依赖。

### 5.5 向量和张量函数

上述公式逐分量成立，因此可直接用于守恒变量向量。对于物理通量张量
\(\boldsymbol F=(\boldsymbol F_1,\ldots,\boldsymbol F_d)\)，微面法向通量为

\[
\boldsymbol F_n=\sum_{\beta=1}^d n_\beta\boldsymbol F_\beta.
\]

其高效积分为

\[
\int_K\boldsymbol F\cdot\boldsymbol n_K\,dS
=\boldsymbol F(\boldsymbol U_i)\cdot\boldsymbol A_K
+\sum_j\sum_{\alpha,\beta}
\ell_{ij,\alpha}^K n_{K,\beta}
\partial_\alpha\boldsymbol F_\beta(\boldsymbol U_j).
\]

因此物理通量梯度的几何权重是一个 \(d\times d\) 矩阵：

\[
\boldsymbol W_{ij}^{K}=\boldsymbol\ell_{ij}^{K}
\boldsymbol n_K^T,
\qquad
W_{ij}^{K,\alpha\beta}=\ell_{ij,\alpha}^{K}n_{K,\beta}.
\]

## 6. 对偶控制体零均值二次重构

### 6.1 必须求解完整二次问题

二维二次基有 5 个非恒定项，三维有 9 个：

\[
N_b=d+\frac{d(d+1)}2.
\]

要使节点一阶导数达到二阶精度，必须用完整二次基构造重构矩阵并求其伪逆，
然后只保留伪逆中对应一次项的前 \(d\) 行。不能在建矩阵时直接删除二次列；
后者实际变成线性最小二乘，一般只能得到一阶精度梯度。

### 6.2 按对偶体分方向半跨度归一化

对节点 \(i\)，从组成 \(V_i\) 的全部微体顶点相对坐标中统计

\[
x_{\min,\alpha},\quad x_{\max,\alpha},
\]

并定义论文式（3-34）对应的参考长度

\[
h_{i,\alpha}=\frac{x_{\max,\alpha}-x_{\min,\alpha}}2,
\qquad
\boldsymbol H_i=\operatorname{diag}(h_{i,1},\ldots,h_{i,d}).
\]

归一化坐标为

\[
\boldsymbol\xi_i=\boldsymbol H_i^{-1}
(\boldsymbol x-\boldsymbol x_i).
\]

二维基函数取

\[
\boldsymbol\psi=
[\xi,\eta,\tfrac12\xi^2,\xi\eta,\tfrac12\eta^2]^T,
\]

三维基函数取

\[
\boldsymbol\psi=
[\xi,\eta,\zeta,
\tfrac12\xi^2,\xi\eta,\xi\zeta,
\tfrac12\eta^2,\eta\zeta,\tfrac12\zeta^2]^T.
\]

`1/2` 只是二次系数约定，使纯二次项系数可直接对应二阶导数，不改变多项式
空间。

体积尺度 \(|V_i|^{1/d}\) 可以用于距离权重和 CFL，但不应代替上述分方向
参考长度；在拉伸网格上用单一体积尺度会显著增加重构矩阵刚性。

### 6.3 零均值基

用第 4 节的矩计算目标控制体上的基函数平均

\[
\langle\boldsymbol\psi_i\rangle_{V_i}
=\frac1{|V_i|}\int_{V_i}\boldsymbol\psi_i(\boldsymbol x)\,dV.
\]

若保存的是相对于 \(\boldsymbol x_i\) 的中心矩

\[
\boldsymbol c_i=\int_{V_i}(\boldsymbol x-\boldsymbol x_i)\,dV,
\qquad
\boldsymbol C_i=\int_{V_i}(\boldsymbol x-\boldsymbol x_i)
(\boldsymbol x-\boldsymbol x_i)^T\,dV,
\]

则

\[
\langle\boldsymbol\xi_i\rangle=
\boldsymbol H_i^{-1}\frac{\boldsymbol c_i}{|V_i|},
\qquad
\langle\boldsymbol\xi_i\boldsymbol\xi_i^T\rangle=
\boldsymbol H_i^{-1}\frac{\boldsymbol C_i}{|V_i|}\boldsymbol H_i^{-1}.
\]

把这些一、二阶分量按第 6.2 节基函数顺序排列，即可得到
\(\langle\boldsymbol\psi_i\rangle\)，无需数值积分。

定义零均值基

\[
\boldsymbol\varphi_i(\boldsymbol x)=
\boldsymbol\psi_i(\boldsymbol x)-
\langle\boldsymbol\psi_i\rangle_{V_i}.
\]

节点 \(i\) 的重构多项式写成

\[
\boldsymbol U_i^R(\boldsymbol x)=
\bar{\boldsymbol U}_i+
\boldsymbol a_i^T\boldsymbol\varphi_i(\boldsymbol x).
\]

由定义可知该多项式在目标控制体上的平均严格等于
\(\bar{\boldsymbol U}_i\)。

### 6.4 邻居控制体方程

对模板中的邻居节点 \(j\)，所有基函数仍使用目标节点 \(i\) 的锚点和尺度
\(\boldsymbol H_i\)，构造

\[
\boldsymbol A_{ij}=
\langle\boldsymbol\psi_i\rangle_{V_j}
-\langle\boldsymbol\psi_i\rangle_{V_i},
\]

\[
\boldsymbol b_{ij}=
\bar{\boldsymbol U}_j-\bar{\boldsymbol U}_i.
\]

邻居矩必须先平移到目标节点的周期最小镜像坐标框架。若已保存邻居自身的
中心矩，可用平移公式转换，而不需要重新遍历其微体。令
\(\boldsymbol r_{ij}=\boldsymbol x_j-\boldsymbol x_i\) 为最小镜像位移，邻居
相对于自身节点的矩为 \((V_j,\boldsymbol c_j,\boldsymbol C_j)\)，则相对于
目标节点 \(i\) 的矩为

\[
\boldsymbol M^{(1)}_{j\to i}=
\boldsymbol c_j+V_j\boldsymbol r_{ij},
\]

\[
\boldsymbol M^{(2)}_{j\to i}=
\boldsymbol C_j+
\boldsymbol r_{ij}\boldsymbol c_j^T+
\boldsymbol c_j\boldsymbol r_{ij}^T+
V_j\boldsymbol r_{ij}\boldsymbol r_{ij}^T.
\]

随后仍使用目标节点的 \(\boldsymbol H_i^{-1}\) 对这些矩归一化。

堆叠后得到

\[
\boldsymbol A_i\boldsymbol a_i\approx\boldsymbol b_i.
\]

距离权重可取

\[
w_{ij}=\left[
\max\left(
\frac{\|\boldsymbol x_j-\boldsymbol x_i\|}{|V_i|^{1/d}},
r_{\min}
\right)
\right]^{-p}.
\]

加权最小二乘解为

\[
\boldsymbol a_i=(\boldsymbol W_i\boldsymbol A_i)^+
\boldsymbol W_i\boldsymbol b_i.
\]

若使用正规方程，则

\[
\boldsymbol a_i=
(\boldsymbol A_i^T\boldsymbol W_i^2\boldsymbol A_i)^{-1}
\boldsymbol A_i^T\boldsymbol W_i^2\boldsymbol b_i.
\]

SVD 形式对畸变网格更稳健，并可以直接由奇异值检查秩和条件数。

### 6.5 只保存梯度相关的伪逆块

初始化时完整构造 \(\boldsymbol A_i\) 并计算完整伪逆，但标准高效模式只保存

\[
\boldsymbol R_i=
\left[(\boldsymbol W_i\boldsymbol A_i)^+
\boldsymbol W_i\right]_{1:d,:}.
\]

运行期先计算归一化一次系数

\[
\boldsymbol a_{i,1:d}=
\boldsymbol R_i\boldsymbol b_i,
\]

再还原物理梯度

\[
\boxed{
\nabla\boldsymbol U_i=
\boldsymbol H_i^{-1}\boldsymbol a_{i,1:d}.
}
\]

因此“只保存一阶导数块”发生在完整二次问题求逆之后，而不是之前。

### 6.6 模板选取

最低要求是：

1. 从原始网格节点—边图按 BFS 环扩展候选节点；
2. 模板点数不少于 \(N_b\)；
3. 加权矩阵满列秩；
4. 条件数不超过给定上限；
5. MPI 分区变化不能改变全局节点模板。

DNDSR 当前实现还做了以下紧致化：

- 完整保留一环边邻居；
- 目标点数为 `ceil(stencilSizeFactor * N_b)`，默认三维通常为 16；
- 后续候选优先成对选择近似反向点，并补强当前最弱谱方向；
- 同时减小归一化三次单项式不平衡和一次位移不平衡；
- 如果秩或条件数不满足，再确定性地追加候选点；
- 排序使用几何量和全局节点号，不依赖 rank 本地遍历顺序。

移植初版可以先保留完整 BFS 环，验证正确后再做紧致模板优化。

当前常用默认参数为：`stencilSizeFactor=1.7`、`maximumStencilRings=4`、
`distanceWeightPower=1`、`distanceWeightFloor=0.15`、
`svdTolerance=1e-11`、`maximumConditionNumber=1e12`。这些是工程默认值，
不是理论公式的一部分；移植后应根据网格族重新检查秩、条件数和误差。

## 7. 用一阶梯度恢复三阶节点值

对节点控制体的所有微体应用第 5 节公式并累加：

\[
\int_{V_i}\boldsymbol U\,dV
=|V_i|\boldsymbol U_i+
\sum_j\boldsymbol L^V_{ij}\cdot\nabla\boldsymbol U_j,
\]

其中

\[
\boldsymbol L^V_{ij}=
\sum_{K\subset V_i}\boldsymbol\ell_{ij}^{K}.
\]

除以体积并解出点值：

\[
\boxed{
\boldsymbol U_i=
\bar{\boldsymbol U}_i-
\sum_j\boldsymbol\omega^V_{ij}
\cdot\nabla\boldsymbol U_j,
\qquad
\boldsymbol\omega^V_{ij}=rac{\boldsymbol L^V_{ij}}{|V_i|}.
}
\]

`pointRecoveryStencil` 存储的就是非零
\((j,\boldsymbol\omega^V_{ij})\)。对二次函数，如果输入梯度精确，这个公式
应达到机器精度；结合二阶精度梯度后，恢复点值达到三阶精度。

解析初始场若给定的是节点点值而求解器保存控制体均值，可以反向使用同一式：

\[
\bar{\boldsymbol U}_i=
\boldsymbol U(\boldsymbol x_i)+
\sum_j\boldsymbol\omega^V_{ij}
\cdot\nabla\boldsymbol U(\boldsymbol x_j).
\]

## 8. 内部宏面的预计算权重

### 8.1 左右状态积分权重

对原始边 \(e=(i,j)\) 的每个微面 \(K\subset\Gamma_e\)，分别以端点
\(i\) 和 \(j\) 为锚点生成第 5 节的权重。累加得到

\[
\boldsymbol L^{U,L}_{e m}
=\sum_{K\subset\Gamma_e}\boldsymbol\ell_{im}^{K},
\qquad
\boldsymbol L^{U,R}_{e m}
=\sum_{K\subset\Gamma_e}\boldsymbol\ell_{jm}^{K}.
\]

左右面积平均为

\[
\boxed{
\bar{\boldsymbol U}_{e,L}=
\boldsymbol U_i+
\frac{\phi_i}{S_e}
\sum_m\boldsymbol L^{U,L}_{e m}\cdot\nabla\boldsymbol U_m,
}
\]

\[
\boxed{
\bar{\boldsymbol U}_{e,R}=
\boldsymbol U_j+
\frac{\phi_j}{S_e}
\sum_m\boldsymbol L^{U,R}_{e m}\cdot\nabla\boldsymbol U_m.
}
\]

无限制时 \(\phi_i=\phi_j=1\)。

### 8.2 物理通量梯度权重

每个微面生成

\[
\boldsymbol W^{L,K}_{em}=
\boldsymbol\ell_{im}^{K}\boldsymbol n_K^T,
\qquad
\boldsymbol W^{R,K}_{em}=
\boldsymbol\ell_{jm}^{K}\boldsymbol n_K^T.
\]

宏面权重为

\[
\boldsymbol W^{L}_{em}=\sum_K\boldsymbol W^{L,K}_{em},
\qquad
\boldsymbol W^{R}_{em}=\sum_K\boldsymbol W^{R,K}_{em}.
\]

左右物理通量积分写成

\[
\boxed{
\boldsymbol I^F_{e,L}=
\boldsymbol F(\boldsymbol U_i)\cdot\boldsymbol A_e
+\phi_i\sum_m\sum_{\alpha,\beta}
W^{L,\alpha\beta}_{em}
\partial_\alpha\boldsymbol F_\beta(\boldsymbol U_m),
}
\]

\[
\boxed{
\boldsymbol I^F_{e,R}=
\boldsymbol F(\boldsymbol U_j)\cdot\boldsymbol A_e
+\phi_j\sum_m\sum_{\alpha,\beta}
W^{R,\alpha\beta}_{em}
\partial_\alpha\boldsymbol F_\beta(\boldsymbol U_m).
}
\]

零阶项只需要宏面合面积向量，不需要为每个 support 节点重复保存零值权重。

### 8.3 面平均梯度权重

粘性通量需要界面平均梯度。构造点梯度采用仿射插值，因此微面上的一次
梯度场可由顶点平均精确积分：

\[
\int_K\nabla\boldsymbol U\,dS
=\frac{|K|}{m}\sum_a\nabla\boldsymbol U(\boldsymbol y_a)
=\sum_jg^K_j\nabla\boldsymbol U_j,
\]

\[
g_j^K=\frac{|K|}{m}\sum_aC_{aj}.
\]

累加后

\[
g_{em}=\sum_{K\subset\Gamma_e}g_m^K,
\qquad
\overline{\nabla\boldsymbol U}_{e,L/R}
=\frac{\phi_{i/j}}{S_e}
\sum_mg_{em}\nabla\boldsymbol U_m.
\]

几何梯度权重左右相同，限制器系数由左右锚点分别施加。

## 9. Euler 物理通量梯度

高效通量积分需要节点处的
\(\partial_\alpha\boldsymbol F_\beta\)。可以用自动微分、解析 Jacobian，或
像 DNDSR 一样显式使用链式法则。

令

\[
\boldsymbol U=(\rho,\boldsymbol m,\mathcal E)^T,
\qquad
\boldsymbol u=\boldsymbol m/\rho,
\]

\[
p=(\gamma-1)
\left(\mathcal E-\frac{|\boldsymbol m|^2}{2\rho}\right).
\]

对任意空间方向 \(x_\alpha\)，已知
\((\rho_\alpha,\boldsymbol m_\alpha,\mathcal E_\alpha)\)，有

\[
\boldsymbol u_\alpha=
\frac{\boldsymbol m_\alpha-\rho_\alpha\boldsymbol u}{\rho},
\]

\[
p_\alpha=(\gamma-1)
\left[
\mathcal E_\alpha-oldsymbol m_\alpha\cdot\boldsymbol u
+\frac12\rho_\alpha|\boldsymbol u|^2
\right].
\]

第 \(\beta\) 个方向的 Euler 通量为

\[
\boldsymbol F_\beta=
\begin{bmatrix}
m_\beta\\
\boldsymbol m u_\beta+p\boldsymbol e_\beta\\
(\mathcal E+p)u_\beta
\end{bmatrix}.
\]

其导数为

\[
\partial_\alpha F_{\rho,\beta}=m_{\beta,\alpha},
\]

\[
\partial_\alpha F_{m_k,\beta}=
m_{k,\alpha}u_\beta+m_k u_{\beta,\alpha}
+p_\alpha\delta_{k\beta},
\]

\[
\partial_\alpha F_{E,\beta}=
(\mathcal E_\alpha+p_\alpha)u_\beta
+(\mathcal E+p)u_{\beta,\alpha}.
\]

节点值和梯度完成 owner/ghost 同步后，每个 rank 可以为 owned 与 ghost 节点
本地计算物理通量梯度，不需要再通信一个独立的通量梯度场。

## 10. 一条原始边一次黎曼求解的无粘通量

当前 DNDSR 实现把高效物理通量积分和宏面 Roe 耗散组合为

\[
\boxed{
\widehat{\boldsymbol I}_e=
\frac12(\boldsymbol I^F_{e,L}+\boldsymbol I^F_{e,R})
+S_e\left[
\widehat{\boldsymbol F}
(\bar{\boldsymbol U}_{e,L},\bar{\boldsymbol U}_{e,R},\boldsymbol n_e)
-\frac12\left(
\boldsymbol F_n(\bar{\boldsymbol U}_{e,L})+
\boldsymbol F_n(\bar{\boldsymbol U}_{e,R})
\right)
\right].
}
\]

若 \(\widehat{\boldsymbol F}\) 为 Roe 通量，方括号就是宏面平均状态上的 Roe
耗散修正。每条原始边只调用一次黎曼求解器，而不是每个微三角面调用三次
或更多次。

这一公式的含义必须正确理解：

- 左右中心物理通量的面内变化由预计算微分权重积分；
- Roe 算子在宏面平均状态和宏面法向上冻结；
- 冻结后的线性耗散算子乘状态跳跃可由左右状态平均精确表示；
- 它不等价于对非线性 Roe 耗散在每个微面逐点积分。

因此该处理是高效算法最重要的成本—精度折中。若另一套程序要求尽可能复现
传统逐点 Roe 通量，需要另外对整个耗散函数做微分展开，或恢复逐微面/逐
Gauss 点黎曼求解；后两者会削弱当前高效方案的性能优势。

## 11. 限制器

高效路径不能直接获得任意界面点处的完整二次多项式值，因此限制对象是宏面
平均值，而不是边中点值。

对节点 \(i\) 的每个相邻宏面 \(e=(i,j)\)，先计算未限制平均

\[
\boldsymbol U^{*,e}_i=
\boldsymbol U_i+
\frac1{S_e}\sum_m
\boldsymbol L^{U,i}_{em}\cdot\nabla\boldsymbol U_m.
\]

逐守恒分量定义

\[
U_{\min}=\min(U_i,U_j),
\qquad
U_{\max}=\max(U_i,U_j),
\qquad
\Delta U=U_i^{*,e}-U_i.
\]

Barth--Jespersen 型候选系数为

\[
\phi_{i,e,k}=
\begin{cases}
\min\left(1,\dfrac{U_{\max,k}-U_{i,k}}{\Delta U_k}\right),
&\Delta U_k>0,\\[1.2ex]
\min\left(1,\dfrac{U_{\min,k}-U_{i,k}}{\Delta U_k}\right),
&\Delta U_k<0,\\
1,&\Delta U_k=0.
\end{cases}
\]

节点最终使用一个标量系数

\[
\phi_i=\min_{e\in E(i),k}\phi_{i,e,k}.
\]

同一个 \(\phi_i\) 必须乘在节点 \(i\) 侧的完整梯度修正和式外面。不能把
和式中每个共享节点梯度分别乘以其自身的 \(\phi_m\)，否则会改变论文中的
重构多项式和界面平均定义。

点值恢复使用未限制梯度；限制器只作用于：

- 左右宏面状态平均；
- 左右物理通量梯度积分修正；
- 左右粘性面平均梯度。

在光滑精度测试中应关闭限制器，以免局部极值附近的降阶掩盖基础格式精度。

## 12. 粘性通量

### 12.1 内部宏面

当前实现先计算左右宏面平均守恒状态和平均梯度，再使用一次粘性通量评价。
界面原始变量取左右原始变量的算术平均，而不是先平均守恒变量再转换：

\[
\boldsymbol Q_f=rac12
[\boldsymbol Q(\bar{\boldsymbol U}_{e,L})+
 \boldsymbol Q(\bar{\boldsymbol U}_{e,R})].
\]

守恒变量梯度采用 dGRP 型修正：

\[
\boldsymbol G_f=
\frac12(\boldsymbol G_L+\boldsymbol G_R)
+\boldsymbol n_e
\frac{(\bar{\boldsymbol U}_{e,R}-
       \bar{\boldsymbol U}_{e,L})^T}{2d_e},
\]

其中当前实现使用

\[
d_e=\frac{\min(|V_i|,|V_j|)}{S_e}.
\]

然后把守恒梯度转换为原始变量梯度并评价应力和热通量：

\[
\boldsymbol I^v_e=S_e
\boldsymbol F^v(\boldsymbol Q_f,\nabla\boldsymbol Q_f,\boldsymbol n_e).
\]

总通量积分为无粘积分减去粘性积分。

### 12.2 物理边界

当前 DNDSR 的一个 `BoundaryPiece` 对应一个边界微线段或微三角面，不把同一
节点上的全部边界微面强行拼成一个平面。这样每片都有明确的常单位法向和边界
分区。边界微面使用标量仿射积分权重

\[
\lambda^K_j=\frac{|K|}{m}\sum_aC_{aj},
\]

若移植程序要合并多个边界微面，只能合并边界分区和法向处理一致的片，并令
\(\lambda_{bj}=\sum_{K\subset b}\lambda_j^K\)。下式用 \(b\) 统一表示一个
当前微面或一个合法合并组。

边界内侧平均和梯度为

\[
\bar{\boldsymbol U}_b=rac1{S_b}sum_j\lambda_{bj}\boldsymbol U_j,
\qquad
\overline{\nabla\boldsymbol U}_b=
\frac{\phi_i}{S_b}sum_j\lambda_{bj}\nabla\boldsymbol U_j.
\]

当前 DNDSR 按论文边界处理路线，先在支撑节点上评价边界数值通量，再按相同
权重插值积分：

\[
\boldsymbol I_b^{\mathrm{inv}}\approx
\sum_j\lambda_{bj}
\widehat{\boldsymbol F}_b(\boldsymbol U_j,\boldsymbol n_b).
\]

边界粘性通量使用上述内侧平均和平均梯度；无滑移壁面会按壁面速度、绝热或
等温条件修正状态和法向梯度。

## 13. 推荐数据结构

### 13.1 构造点

```cpp
struct AffineCoefficient
{
    LocalNode node;
    double coefficient;
};

struct AffinePoint
{
    Vec3 coordinate;
    SmallVector<AffineCoefficient> support;
};
```

### 13.2 节点控制体

```cpp
struct PointRecoveryEntry
{
    LocalNode node;       // owned 或 exact-halo ghost
    Vec3 gradientWeight;  // 已除以 |V_i|
};

struct NodeControlVolume
{
    double volume;
    Vec3 firstMoment;
    Mat3 secondMoment;
    Vec3 referenceLengths;
    double lengthScale;
    Vector<PointRecoveryEntry> pointRecovery;
    Vector<BoundaryPiece> boundaryPieces;
};
```

### 13.3 内部宏面

```cpp
struct EfficientSurfaceEntry
{
    LocalNode node;
    Vec3 stateGradientWeight[2]; // 左/右状态积分
    Mat3 fluxGradientWeight[2];  // 左/右物理通量积分
    double gradientWeight;       // 面平均梯度
};

struct EdgeControlSurface
{
    LocalEdge edge;
    LocalNode endpoint[2];
    Vec3 areaVector;              // A_e
    double scalarArea;            // S_e
    Vector<EfficientSurfaceEntry> stencil;
};
```

### 13.4 边界片

```cpp
struct EfficientBoundaryEntry
{
    LocalNode node;
    double integrationWeight;
};

struct BoundaryPiece
{
    BoundaryZone zone;
    Vec3 areaVector;
    double scalarArea;
    Vector<EfficientBoundaryEntry> stencil;
};
```

每个 stencil 最终必须按节点编号排序、去重。权重结构中不要保存指向运行期
场数组的裸指针，因为场数组在 MPI halo 最终裁剪或重新分配后可能移动。

对于大规模实现，推荐“构造时 hash map，完成后 CSR/扁平数组”：

```text
surfaceOffsets[e] ... surfaceOffsets[e+1]
surfaceNode[k]
surfaceStateWeightLeft[k]
surfaceStateWeightRight[k]
surfaceFluxWeightLeft[k]
surfaceFluxWeightRight[k]
surfaceGradientWeight[k]
```

这比每个面永久持有 `unordered_map` 更适合运行期连续访问和 GPU/MPI 移植。

## 14. 初始化流程

推荐严格按以下顺序实现：

```text
1. 读取 O1 原始网格和边界分区。
2. 合并一一平移周期节点拓扑，再进行 MPI 分区。
3. 构建一层 point-complete cell halo，补全 owned 节点的 cell star。
4. 构建分布式原始 edge/face 拓扑和 node-edge incidence。
5. 生成 edge/face/cell 顶点平均构造点。
6. 枚举所有节点微体：
     - 累积体积和一、二阶矩；
     - 累积方向包围盒；
     - 生成点值恢复梯度权重。
7. 枚举所有 owned 原始边对应的内部微面：
     - 累积 S_e 和 A_e；
     - 分别生成左右状态梯度权重；
     - 生成左右物理通量梯度矩阵权重；
     - 生成面平均梯度权重。
8. 枚举物理边界微面，为每个边界片合并其仿射 support 权重并保存分区号。
9. 检查每个节点的面积向量闭合。
10. 发布 owned 节点边邻接，按最大模板环数稀疏拉取远程邻接行。
11. 交换候选节点坐标、体积矩和参考长度，构造完整二次重构矩阵。
12. 用 SVD/LS 选定模板，保存完整二次问题伪逆的前 d 行。
13. 收集重构、点值恢复、内部面、边界片和限制器的全部节点依赖。
14. 建立最终 exact owner/ghost node halo，并释放候选环临时数据。
15. 把所有几何 stencil 节点号重映射到最终 local/ghost 编号。
16. 建立节点场和 edge 通量场的持久稀疏 MPI 通信表。
```

高效模式完成后，体、内部面和边界面 Gauss 点容器都应为空。

## 15. 每次残差计算的运行流程

```text
1. pull 控制体均值 Ubar 的 exact node ghost。

2. 对每个 owned 节点：
     b[j] = Ubar[stencil[j]] - Ubar[i]
     normalizedLinear = inverseLinearRows[i] * b
     gradU[i] = inverse(H_i) * normalizedLinear

3. pull gradU ghost。

4. 对每个 owned 节点恢复点值：
     Upoint[i] = Ubar[i]
     for entry in pointRecovery[i]:
         Upoint[i] -= transpose(gradU[entry.node]) * entry.weight

5. pull Upoint ghost。当前 DNDSR 不改写已恢复并保存的 Upoint；在使用宏面
   平均、传统积分点状态或谱半径平均状态时，若候选非物理，再沿对应锚点
   状态方向缩回物理解区域。移植程序若选择直接限制 Upoint，必须单独验证
   其守恒性和精度影响。

6. 若启用限制器：
     - 用宏面状态积分权重计算每个 incident edge 的候选面平均；
     - 求节点单一系数 phi[i]；
     - pull phi ghost。

7. 在本地由 Upoint 和 gradU 计算 owned/ghost 节点物理通量梯度。

8. 对每条 owned 原始边：
     - 计算左右物理通量积分 IFL、IFR；
     - 计算左右宏面平均 UL、UR；
     - 只调用一次 Riemann solver；
     - 加宏面耗散修正；
     - 如有粘性，计算一次宏面粘性通量；
     - 保存唯一 edgeFlux[e] 和 spectralRadius[e]。

9. pull ghost edgeFlux 和 ghost edge spectral radius。

10. 对每个 owned 节点：
      residual = boundaryFlux[i]
      residual += sum_e B[i,e] * edgeFlux[e]
      RHS[i] = -residual / volume[i]
```

MPI 中 edge owner 只计算一次 edge 通量，其他 rank 拉取该 edge 的结果；不要
让两个端点 owner 各自重算再做残差 push-sum，否则会增加通信并可能破坏位级
守恒。

## 16. MPI 与周期并行策略

### 16.1 两类 halo 不应混淆

推荐把几何构建 halo 和运行期物理场 halo 分开：

- **几何 cell halo**：只在初始化阶段补全 owned 节点周围的 cell star；
- **exact node halo**：由最终重构和积分 stencil 的实际节点依赖决定，供运行期
  均值、梯度、点值和限制器通信使用。

把格心型方法的固定 cell halo 直接当成格点型运行 halo，通常不能覆盖二环
重构、周期角点或跨分区宏面的全部节点依赖；反过来盲目构建多层完整 cell
halo 又会产生大量无用 ghost。

### 16.2 初始化时扩环，确定后裁剪

一种长期高性能方案是：

1. 每个节点 owner 发布其一环边邻接的全局节点编号；
2. 初始化时按 `maximumStencilRings` 逐环稀疏 pull 所需远程邻接行；
3. 临时拉取这些候选节点的坐标、矩和参考长度；
4. 各 owned 节点独立选定最终模板；
5. 把最终重构和积分涉及的全局节点号取并集；
6. 重新建立精确 owner/ghost 映射和 MPI datatype；
7. 释放候选环、临时邻接和临时几何。

运行期不需要全局节点场复制，也不需要 `MPI_Allgatherv`。

### 16.3 周期角、周期棱和体对角线

三方向周期时，一个节点可能经 x/y/z 多个平移同时对应另一周期像。要正确
覆盖长方体周期棱和体对角线邻接，需要：

- 周期节点在网格读取阶段先合并成同一物理全局节点；
- edge/face 分布式键除规范化全局节点号外，还保留相对 periodic bits；
- 邻接扩环基于合并后的节点图，而不是基于边界面名称临时复制场；
- 几何位移使用最小镜像。

这样对角邻居会自然进入一环或后续 BFS 环，不需要为“角周期”单独写全局通信。

### 16.4 每次 RHS 的通信顺序

高效最小二乘路径通常需要：

1. node mean pull；
2. node gradient pull；
3. node point-value pull；
4. 若有限制器，node limiter pull；
5. edge flux 与 edge spectral-radius pull；
6. CFL 全局步长使用一次标量 `MPI_MIN`；
7. 残差报告使用一次标量 `MPI_SUM`。

物理通量梯度可以由已同步的点值和状态梯度在本地重算，从而避免新增大矩阵
MPI 场。可进一步通过通信—计算重叠优化，但不应改变上述依赖关系。

## 17. SSPRK3 时间推进

当前实现使用三阶段三阶 SSPRK：

\[
\boldsymbol U^{(1)}=\boldsymbol U^n+
\Delta t\,\mathcal L(\boldsymbol U^n),
\]

\[
\boldsymbol U^{(2)}=\frac34\boldsymbol U^n+
\frac14\left[
\boldsymbol U^{(1)}+
\Delta t\,\mathcal L(\boldsymbol U^{(1)})
\right],
\]

\[
\boldsymbol U^{n+1}=\frac13\boldsymbol U^n+
\frac23\left[
\boldsymbol U^{(2)}+
\Delta t\,\mathcal L(\boldsymbol U^{(2)})
\right].
\]

物理非定常精度验证必须在各网格上使用一致 CFL 策略，并确保时间误差不主导
空间误差。局部时间步只适合稳态伪时间推进，不适合固定物理终止时间的精度
比较。

## 18. 必须通过的验证

### 18.1 几何验证

- 每个微体和微面度量为正；
- 节点控制体体积为正；
- 内部边两端引用同一个宏面；
- 面积向量闭合误差接近机器精度；
- 周期平移前后体积、面积和权重一致；
- 1/2/4/8 MPI 进程下全局几何计数相同。

### 18.2 二次多项式验证

设计一般二次函数

\[
u(\boldsymbol x)=c+\boldsymbol a^T\boldsymbol x+
\frac12\boldsymbol x^T\boldsymbol H\boldsymbol x,
\]

其中 \(\boldsymbol H\) 应包含非零交叉项。逐项检查：

1. 解析微单纯形矩是否精确；
2. 完整二次 LS 是否恢复精确梯度；
3. 只保存伪逆前 \(d\) 行后，梯度结果是否不变；
4. 点值恢复误差是否接近机器零；
5. 宏面左右状态积分是否精确；
6. 物理通量权重对人为指定的二次向量通量是否精确；
7. MPI 分区变化后每个全局节点和边的结果是否一致。

如果二次函数梯度误差不是机器量级，优先检查：

- 是否误删了重构矩阵二次列；
- 是否对邻居使用了邻居自己的归一化尺度；
- 是否遗漏二阶矩交叉项；
- 是否把体积尺度误作分方向参考长度；
- 周期位移是否未做最小镜像；
- 模板矩阵是否秩亏或病态。

### 18.3 守恒和通量验证

- 无物理边界的周期域中，所有节点残差按体积积分后的全局和应接近零；
- 独立分解中心物理通量、Roe 耗散和粘性通量，确认其和等于生产 RHS；
- 每个物理步的黎曼调用次数应约为 `3 × owned edge count`，对应 SSPRK3 三个
  RHS，而不是微面数或 Gauss 点数；
- 高效模式保存的 Gauss 点数必须为零。

### 18.4 收敛验证

建议至少使用四级网格，并分别报告：

- 第一次重构后的节点值误差；
- 第一次重构后的梯度误差；
- 左右状态跳跃 \(U_R-U_L\) 的阶数；
- 中心物理通量积分误差；
- Roe 耗散积分误差；
- 固定物理时间下的体积加权 \(L_1,L_2,L_\infty\)；
- 质量、动量、总能量漂移；
- 不同 MPI 进程数下的逐全局节点差异。

重构达到三阶点值和二阶梯度，只能证明重构构件正确；完整格式还会受到宏面
Roe 冻结近似、边界通量、粘性通量、限制器、周期光滑性和时间误差影响。

## 19. 常见实现错误

1. **把面平均点、体平均点替换为几何质心。** 这会改变算法定义和全部权重。
2. **只用线性列构造重构矩阵。** 即使最后只保存一阶导数，也必须先求完整
   二次问题。
3. **零均值基未归一化。** 会导致拉伸网格上条件数迅速恶化。
4. **邻居控制体使用自己的尺度评价目标多项式。** 每个目标节点的整套方程
   必须统一使用目标节点尺度。
5. **绝对坐标二阶矩先累加后相减。** 细网格上容易产生消减误差。
6. **用 \(\|\boldsymbol A_e\|\) 代替 \(S_e\)。** 非平面宏面上二者不同。
7. **对每个 support 节点分别施加其自身限制器。** 正确做法是锚点侧单一系数
   乘完整修正和。
8. **运行期仍遍历微体/微面或构造 Gauss 点。** 高效模式应只遍历压缩 stencil。
9. **永久保存 `unordered_map` 或物理量裸指针。** 前者影响缓存，后者会在场
   重分配后失效。
10. **沿用格心求解器固定 ghost 层而不检查节点依赖。** 格点重构和宏面权重
    可能引用远程二环节点和周期对角节点。
11. **周期边界每步全局复制状态。** 应在初始化时合并周期拓扑并建立稀疏
    owner/ghost 通信。
12. **认为一次宏面 Roe 等价于逐点 Roe 积分。** 当前高效实现冻结了宏面耗散
    算子，这是需要单独评估的近似。
13. **用原始边投影长度代替当前粘性特征距离。** DNDSR 使用
    \(\min(V_L,V_R)/S_e\)。
14. **精度测试开启限制器或使用局部物理时间步。** 这会混入额外降阶因素。

## 20. 建议的移植阶段

### 阶段 A：串行几何与解析积分

实现构造点、微体/微面枚举、矩、面积闭合和第 5 节微分权重。用一般二次
多项式做到机器精度。

### 阶段 B：完整二次 LS 与点值恢复

先保存完整伪逆用于调试，确认梯度和点值精度；随后只保留前 \(d\) 行并比较
结果是否一致。

### 阶段 C：无黏中心物理通量积分

暂不加 Roe 耗散和限制器，验证左右物理通量高效积分、守恒性和空间阶数。

### 阶段 D：宏面 Roe 耗散与限制器

实现一边一次黎曼求解，分别输出中心项和耗散项误差。再加入宏面平均限制器。

### 阶段 E：边界和粘性项

依次实现远场/出入口、滑移壁面、无滑移绝热壁面、等温壁面及粘性梯度修正。

### 阶段 F：MPI 与周期

先保证全局编号下串行结果正确，再实现周期拓扑合并、初始化候选环稀疏 pull、
最终 exact node halo 和 edge owner 通量通信。每增加一个 MPI 功能都做逐全局
节点对比。

## 21. DNDSR 当前实现位置

| 内容 | 文件 |
|---|---|
| 对偶几何、解析矩、微分权重 | `src/NCFV/NCFVDualGeometry.cpp` |
| 几何和压缩权重数据结构 | `src/NCFV/NCFVDualGeometry.hpp` |
| 完整二次重构、归一化、SVD/LS | `src/NCFV/NCFVReconstruction.cpp` |
| 点值恢复、限制器 | `src/NCFV/NCFVReconstruction.cpp` |
| 物理通量梯度、高效宏面通量、粘性项 | `src/NCFV/NCFVSpatial.cpp` |
| 原始 edge/face 分布式拓扑 | `src/NCFV/NCFVTopology.cpp` |
| exact node owner/ghost halo | `src/NCFV/NCFVNodeHalo.cpp` |
| 初始化顺序、SSPRK3、重启和输出 | `src/NCFV/NCFVSolver.cpp` |
| 几何和多项式精确性测试 | `test/cpp/NCFV/test_NCFVGeometry.cpp` |
| MPI、周期和高效权重测试 | `test/cpp/NCFV/test_NCFVParallel.cpp` |

## 22. 最小复现清单

在另一套程序中，如果以下项目都已完成，就具备了高效算法的最小闭环：

- [ ] 原始节点、边、面、体顶点平均构造点；
- [ ] 二维微三角形/线段或三维微四面体/三角面枚举；
- [ ] 对偶体积、一二阶矩、宏面标量面积和面积向量；
- [ ] 分方向半跨度归一化的零均值完整二次矩阵；
- [ ] 完整二次伪逆求解后仅保存前 \(d\) 行；
- [ ] 体积分微分权重和节点值恢复；
- [ ] 左右宏面状态梯度权重；
- [ ] 左右物理通量梯度矩阵权重；
- [ ] 面平均梯度权重；
- [ ] 节点 Euler 物理通量梯度；
- [ ] 每条原始边一次黎曼求解；
- [ ] 边界片权重和边界分区；
- [ ] 节点—边守恒残差装配；
- [ ] SSPRK3 与一致 CFL；
- [ ] 周期最小镜像和稀疏 owner/ghost 通信；
- [ ] 二次多项式、几何闭合、守恒、MPI 不变性和四级网格收敛测试。

完成这些步骤后，另一套程序不需要依赖 DNDSR 的具体数组类，也能复现本文
高效算法的数学过程和主要性能特征。
