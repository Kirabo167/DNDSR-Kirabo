# NCFV 高效变分重构、稀疏求解与 BDF2 双时间推进

> 历史设计资料：文中的界面面均值变分重构已移除；当前高效变分采用边中点泛函与一次导数通量积分。文中的隐式时间推进不是当前求解器的可用功能。

## 1. 目的、符号与借鉴范围

本文把王乾博士论文第 4 章的界面跳跃积分（interfacial jump integration，IJI）思想，改写到三阶节点中心有限体积（NCFV）的对偶控制体上，并进一步给出稀疏系统的构造、可解性条件，以及 BDF2、LU-SGS 和 GMRES 的隐式推进过程。

论文中的基本做法是：每个原始单元上用零均值多项式表示重构，在所有单元界面上惩罚两侧多项式及其法向导数的跳跃，然后对全域泛函取极小值。下文保留“界面信息的加权跳跃平方和”这一变分主线，但针对 NCFV 的节点图和高效微分积分作离散调整：在每条原始节点边的中点比较两端对偶控制体多项式的值、一阶导数和二阶导数。该中点 jet 泛函与论文的面上 IJI 是相近的变分构造，二者的采样位置和导数范数不同，不能说成同一个离散格式。

以下推导采用三阶 NCFV 的二次重构。对偶控制体区域记为 \(\Omega_i\)，体积为 \(V_i=|\Omega_i|\)，其守恒变量平均值为
\[
\bar{\boldsymbol U}_i=\frac{1}{V_i}\int_{\Omega_i}\boldsymbol U(\boldsymbol x)\,dV.
\]
节点边 \(e=(i,j)\) 的几何长度为 \(\ell_e=\|\boldsymbol x_j-\boldsymbol x_i\|\)，中点为 \(\boldsymbol m_e=(\boldsymbol x_i+\boldsymbol x_j)/2\)。\(d\) 为空间维数，守恒变量个数记为 \(n_v\)。

## 2. 零均值二次多项式

### 2.1 基函数与平均约束

二次多项式去掉常数项后的系数数目为
\[
M=d+\frac{d(d+1)}2=
\begin{cases}
5,&d=2,\\
9,&d=3.
\end{cases}
\]
为改善拉伸网格上的条件数，对每个节点用分方向长度矩阵
\[
H_i=\operatorname{diag}(h_{i,1},\ldots,h_{i,d}),\qquad
\boldsymbol\xi_i=H_i^{-1}(\boldsymbol x-\boldsymbol x_i)
\]
做无量纲化。可取 \(h_{i,\alpha}\) 为节点对偶控制体在第 \(\alpha\) 个方向上的半跨度。三维基函数例如
\[
\boldsymbol\psi_i=
\left[\xi,\eta,\zeta,
\tfrac12\xi^2,\xi\eta,\xi\zeta,
\tfrac12\eta^2,\eta\zeta,\tfrac12\zeta^2\right]^T,
\]
二维时删去含 \(\zeta\) 的项。定义零均值基
\[
\boldsymbol\phi_i(\boldsymbol x)=
\boldsymbol\psi_i(\boldsymbol x)-
\frac{1}{V_i}\int_{\Omega_i}\boldsymbol\psi_i(\boldsymbol x)\,dV.
\]
于是 \(\int_{\Omega_i}\boldsymbol\phi_i\,dV=\boldsymbol 0\)。二维基函数的明确形式为
\[
\boldsymbol\psi_i=[\xi,\eta,\tfrac12\xi^2,\xi\eta,\tfrac12\eta^2]^T.
\]
构造零均值基时只需对这些单项式积分。令
\[
\boldsymbol M_i^{(1)}=\int_{\Omega_i}(\boldsymbol x-\boldsymbol x_i)\,dV,\qquad
\boldsymbol M_i^{(2)}=\int_{\Omega_i}(\boldsymbol x-\boldsymbol x_i)
(\boldsymbol x-\boldsymbol x_i)^T\,dV,
\]
则
\[
\langle\boldsymbol\xi_i\rangle=H_i^{-1}\boldsymbol M_i^{(1)}/V_i,\qquad
\langle\boldsymbol\xi_i\boldsymbol\xi_i^T\rangle=
H_i^{-1}\boldsymbol M_i^{(2)}H_i^{-1}/V_i.
\]
因此一次项和二次项的均值可由对偶微单纯形的解析几何矩得到，无需在重构初始化时布置 Gauss 点。节点 \(i\) 的重构为
\[
\boxed{\boldsymbol P_i(\boldsymbol x)=\bar{\boldsymbol U}_i+
\boldsymbol C_i^T\boldsymbol\phi_i(\boldsymbol x)},
\]
其中 \(\boldsymbol C_i\in\mathbb R^{M\times n_v}\) 是待定系数。零均值性质立即给出
\[
\frac{1}{V_i}\int_{\Omega_i}\boldsymbol P_i\,dV=\bar{\boldsymbol U}_i,
\]
因此重构不会改变有限体积未知量的定义，也不会破坏控制体平均守恒。

### 2.2 节点值、梯度与 Hessian

在原始节点位置恢复点值与梯度：
\[
\boldsymbol U_i^{\mathrm{node}}=\boldsymbol P_i(\boldsymbol x_i),\qquad
\boldsymbol G_i=\nabla\boldsymbol P_i(\boldsymbol x_i).
\]
二阶导数 \(\nabla^2\boldsymbol P_i\) 在二次多项式内为常数。它参与下文的界面变分泛函，使系数系统能辨别二次模态；高效微分通量阶段仍可只传递 \(\boldsymbol U_i^{\mathrm{node}}\) 与 \(\boldsymbol G_i\)，不必在每个微面或 Gauss 点存 Hessian。

## 3. 从界面跳跃到节点边 jet 泛函

### 3.1 论文中的 IJI 形式

对论文中的单元界面 \(f\)，两侧重构分别为 \(P_L,P_R\)，中心距离为 \(d_{LR}\)。其 IJI 可概括为
\[
I_f=\frac1{d_{LR}}\int_f\sum_{p=0}^{k}
\frac{d_{LR}^{2p}}{(p!)^2}
\left(\partial_n^p P_L-\partial_n^p P_R\right)^2dS,
\qquad I=\sum_f I_f.
\]
这里 \(p=0\) 是函数值跳跃，\(p=1,2,\ldots,k\) 是法向导数跳跃；距离幂使不同导数阶的项具有一致量纲。对重构系数求驻值便得到全场稀疏线性系统，这对应论文式（4-5）—（4-11）[1](#ref_001)。论文还证明了：在全域网格连通、重构空间满足相应非奇异性条件时，该系统对称正定[1](#ref_001)（第4.1.2节）。

### 3.2 适用于 NCFV 高效重构的离散界面信息

NCFV 的节点对偶界面由原始边及其周围微面组成。为避免在变分构造中再次引入面上高斯点，可在每条节点边的中点比较两侧重构的完整二次 jet。对边 \(e=(i,j)\) 定义
\[
\mathcal J_e=
\begin{bmatrix}
P_i(\boldsymbol m_e)-P_j(\boldsymbol m_e)\\[1mm]
\dfrac{\ell_e}{2}\left(\nabla P_i(\boldsymbol m_e)-\nabla P_j(\boldsymbol m_e)\right)\\[1mm]
\dfrac{\ell_e^2}{4}\left(\nabla^2P_i-\nabla^2P_j\right)
\end{bmatrix}.
\]
向量梯度按 Cartesian 分量排列；Hessian 按全部 \(d^2\) 个有序分量排列，因此混合导数在 Frobenius 范数中出现两次。可定义
\[
\boxed{\mathcal I(\boldsymbol C)=\frac12\sum_{e=(i,j)}\omega_e
\|\mathcal J_e\|_2^2,\qquad \omega_e>0.}
\]
当前工作区 NCFV 变分行的尺度对应 \(\omega_e=1\)、值行权重为 1、梯度行乘 \(\ell_e/2\)、Hessian 行乘 \(\ell_e^2/4\)。若改用面面积权、不同的各阶权函数或论文的纯法向导数内积，仍可得到变分格式，但矩阵条件数和具体重构结果会改变，需重新验证。

这个泛函的含义是：在每条节点边的代表位置，使两侧的函数值、斜率和曲率尽量一致。它以节点邻接图为耦合图，符合 NCFV 的紧致数据结构。函数值项由已知的两个控制体平均值提供偏置，梯度和 Hessian 项则约束高阶系数。

### 3.3 矩阵行表示

令 \(q=1+d+d^2\)，定义 \(\boldsymbol e_0=[1,0,\ldots,0]^T\in\mathbb R^q\)。对边 \(e=(i,j)\)，定义矩阵 \(D_i^e\in\mathbb R^{q\times M}\)：
\[
D_i^e=
\begin{bmatrix}
\boldsymbol\phi_i(\boldsymbol m_e)^T\\
\dfrac{\ell_e}{2}\nabla\boldsymbol\phi_i(\boldsymbol m_e)^T\\
\dfrac{\ell_e^2}{4}\nabla^2\boldsymbol\phi_i^T
\end{bmatrix},
\]
其中梯度与 Hessian 的每个 Cartesian 分量各占一行。对单个标量变量，界面 jet 残差写成
\[
\boldsymbol r_e=\boldsymbol e_0(\bar u_i-\bar u_j)
+D_i^e\boldsymbol c_i-D_j^e\boldsymbol c_j.
\]
向量守恒状态对每个分量使用同一套 \(D_i^e\)，只需把标量系数列向量替换为矩阵 \(\boldsymbol C_i\)。

## 4. Euler–Lagrange 方程与稀疏块矩阵

### 4.1 对系数取一阶变分

先考虑一个守恒分量。泛函为
\[
\mathcal I(\boldsymbol c)=\frac12\sum_{e=(i,j)}\omega_e
\boldsymbol r_e^T\boldsymbol r_e.
\]
对节点 \(i\) 的系数 \(\boldsymbol c_i\) 求导。每条相邻边向梯度贡献
\[
(D_i^e)^T\omega_e\boldsymbol r_e,
\]
全部相邻边贡献之和置零后，将含 \(\boldsymbol c_i\)、邻点系数和已知均值分别整理，得到
\[
\boxed{
K_i\boldsymbol c_i-
\sum_{j\in N(i)}C_{ij}\boldsymbol c_j=
\sum_{j\in N(i)}\boldsymbol g_{ij}(\bar u_j-\bar u_i),}
\]
其中
\[
K_i=\sum_{e\ni i}\omega_e(D_i^e)^TD_i^e,\quad
C_{ij}=\omega_{ij}(D_i^{ij})^TD_j^{ij},\quad
\boldsymbol g_{ij}=\omega_{ij}(D_i^{ij})^T\boldsymbol e_0.
\]
若边权在不同边上变化，以上求和中的 \(\omega_e\) 按边分别取值。对所有节点组装得到
\[
\boxed{\mathbb A\boldsymbol c=\boldsymbol b,\qquad
\mathbb A_{ii}=K_i,\quad
\mathbb A_{ij}=-C_{ij}\ (j\in N(i)),\quad
\mathbb A_{ij}=0\ (j\notin N(i)).}
\]
右端为
\[
\boldsymbol b_i=\sum_{j\in N(i)}\boldsymbol g_{ij}(\bar u_j-\bar u_i).
\]
因为反向边给出 \(C_{ji}=C_{ij}^T\)，全局矩阵 \(\mathbb A\) 对称。它的块稀疏图就是原始节点边图：每个节点只有一个 \(M\times M\) 对角块以及相邻节点的非对角块。因此储存和矩阵向量乘法的复杂度随节点数和边数线性增长，块内的小矩阵尺寸固定为 5 或 9。

将全部守恒变量并列，方程成为
\[
\mathbb A\boldsymbol C=\boldsymbol B,
\]
其中 \(\boldsymbol C_i\in\mathbb R^{M\times n_v}\)，同一几何矩阵对 \(n_v\) 个分量重复求解。这允许复用分解、预条件器和稀疏矩阵结构。

### 4.2 局部固定点 / Block Gauss–Seidel

若逐节点解局部对角块，可写为
\[
\boldsymbol c_i^{(s+1)}=K_i^{-1}\left[
\sum_{j<i}C_{ij}\boldsymbol c_j^{(s+1)}+
\sum_{j>i}C_{ij}\boldsymbol c_j^{(s)}+
\boldsymbol b_i\right].
\]
这是按节点顺序的 block Gauss–Seidel。带松弛的 block SOR 为
\[
\boldsymbol c_i^{(s+1)}\leftarrow
(1-\omega)\boldsymbol c_i^{(s)}+
\omega K_i^{-1}\left[
\sum_{j<i}C_{ij}\boldsymbol c_j^{(s+1)}+
\sum_{j>i}C_{ij}\boldsymbol c_j^{(s)}+\boldsymbol b_i\right].
\]
当前代码所做的“对每个节点直接解 \(K_i\)，再重复邻点系数更新”，外层轮数是全局耦合迭代次数，并不是反复近似局部 \(K_i\) 的逆。使用最小二乘系数作初值可减少外层误差；每轮 MPI ghost 同步后，各 rank 的并行更新不再等同于全局串行排序的精确 Gauss–Seidel，实际收敛率需用重构残差或二次场固定点误差监控。

## 5. 有解性、唯一性与精确性

### 5.1 半正定性

由泛函 Hessian 的构造，任意系数扰动 \(\boldsymbol z\) 满足
\[
\boldsymbol z^T\mathbb A\boldsymbol z=
\sum_{e=(i,j)}\omega_e
\|D_i^e\boldsymbol z_i-D_j^e\boldsymbol z_j\|_2^2\ge 0.
\]
因此 \(\mathbb A\) 总是对称半正定。这里已知平均值只进入右端，不改变系统矩阵。

### 5.2 何时严格正定

半正定要成为正定，还需排除零能量的非零系数场。一个直接的充分条件是：

1. 每个连通分量上的节点图连通，所有边权严格为正，边的坐标均在两侧多项式的同一个几何坐标框架中比较（周期边需采用一致的周期展开位移）；
2. 每个节点的局部行矩阵 \(\{D_i^e:e\ni i\}\) 满列秩，使 \(K_i\) 正定、局部更新有定义；
3. 每个连通分量上，二次多项式空间的控制体平均泛函具有满列秩。二维需至少有 6 个几何独立的控制体平均约束，三维需至少 10 个；数量足够仍不够，还必须几何非退化。

证明如下。若 \(\boldsymbol z^T\mathbb A\boldsymbol z=0\)，每条边的值、梯度和 Hessian jet 跳跃都为零。由于局部多项式次数不超过 2，Hessian 为常数；沿连通节点图传播，所有单元扰动多项式具有相同 Hessian。中点梯度相等继而使线性项相同，中点值相等再使常数项相同。因此所有局部扰动多项式是同一个全局二次多项式 \(q(\boldsymbol x)\)。每个扰动多项式在自己的对偶控制体上均值为零，故
\[
\int_{\Omega_i}q(\boldsymbol x)\,dV=0\quad\text{对该连通分量的所有 }i.
\]
若控制体平均对 \(\mathbb P_2\) 满列秩，则只能有 \(q\equiv0\)，从而所有 \(\boldsymbol z_i=0\)。因此 \(\mathbb A\) 对称正定，方程有唯一解。

这也指出了可解性论证的边界：论文的 SPD 结论不能脱离网格非退化和 k-exact / 均值泛函满秩条件直接照搬到任意节点图。局部 \(K_i\) 可逆只说明每个 block 更新可算，不能单独证明全局矩阵无零空间。若条件不满足，应扩充节点模板、加入有物理依据的边界/周期约束，或用秩揭示 QR/SVD 检查并处理零空间；简单加 \(\epsilon I\) 虽可强制可逆，却会改变重构并破坏严格多项式精确性。

### 5.3 二次多项式精确性

设精确场 \(u(\boldsymbol x)\in\mathbb P_2\)，且输入 \(\bar u_i\) 是其在 \(\Omega_i\) 上的真实平均。零均值基可精确表示 \(u-\bar u_i\)，所以存在一组系数使每个 \(P_i=u\)。这时每条边 jet 残差为零，\(\mathcal I=0\)。若上述唯一性条件成立，该零泛函解就是唯一极小值解，故重构对任意二次多项式精确。相应的一阶导数在光滑场上具备构造三阶 NCFV 所需的二阶精度基础。

三阶收敛仍需形状规则网格、稳定重构、足够收敛的变分迭代、适当的边界处理和未触发降阶的限制器；Roe 耗散、非线性物理通量的高效积分近似及时间误差也会影响观测阶数。不能仅凭二次精确性宣称所有流动情形均达到三阶。

## 6. 变分系统的数值解法与效率

### 6.1 可选求解器

在严格正定条件下，建议按问题规模选择：

- **小网格直接法**：稀疏 Cholesky，用于基准计算和验证唯一解；
- **PCG**：\(\mathbb A\) 对称正定时的首选迭代法，适合用 block Jacobi \(\operatorname{diag}(K_i)\)、不完全 Cholesky 或 AMG 预条件；
- **Block Gauss–Seidel / SOR**：重用每个节点的小型 \(K_i\) 分解，内存少，适合与时间推进耦合；
- **GMRES**：适用于因近似线性化、非对称边界项或非对称预条件而形成的非对称变分方程，不应仅因“GMRES 通用”就替代 SPD 系统上的 PCG。

可用边项直接做 matrix-free 乘法。给定向量 \(\boldsymbol z\)，先逐边计算
\(\boldsymbol q_e=D_i^e\boldsymbol z_i-D_j^e\boldsymbol z_j\)，再把
\((D_i^e)^T\omega_e\boldsymbol q_e\) 和
\(-(D_j^e)^T\omega_e\boldsymbol q_e\) 加回两端。这样不用存储全局 CSR 矩阵；但每轮仍需邻点交换与一次全局内积归约（PCG）。

### 6.2 与高效 differential 通量的衔接

每次残差计算先由当前控制体平均 \(\bar{\boldsymbol U}\) 解变分系数，随后只抽取
\[
\boldsymbol U_i^{\mathrm{node}}=\boldsymbol P_i(\boldsymbol x_i),\qquad
\boldsymbol G_i=\nabla\boldsymbol P_i(\boldsymbol x_i).
\]
把这两个场送入预计算的对偶体积/宏面微分权重。对光滑状态，物理通量梯度按链式法则计算：
\[
\partial_\alpha\boldsymbol F_\beta(\boldsymbol U_i)
=\frac{\partial\boldsymbol F_\beta}{\partial\boldsymbol U}
\bigg|_{\boldsymbol U_i}\partial_\alpha\boldsymbol U_i.
\]
于是体积积分、宏面状态平均和中心物理通量积分使用点值与一阶导数的压缩 stencil，不需要在时间步内重建 Gauss 点；每条原始边对应的对偶宏面仍只做一次 Roe/Riemann 求解，端点残差用相反符号累加以保持守恒。变分重构阶段使用 Hessian，是为了确定二次系数；它不意味着高效 flux kernel 必须读取 Hessian。

效率上的代价也应明确：与只保存梯度的 LS 路径相比，变分重构必须在系数迭代中保留每节点 \(M\times n_v\) 个二次系数，并进行邻点同步。可以在残差阶段只保留恢复出的节点值和梯度，但不能在变分系数尚未收敛时丢弃系数。几何矩阵 \(K_i\) 和变分算子只依赖网格，可预计算；系数解可用前一残差/时间步作初值。

如果希望每个时间残差都把变分方程解到很紧的容差，成本可能超过 flux 积分节省的成本。可使用 warm start 和固定的少量 block sweeps，并把“重构迭代误差”纳入总非线性缺陷监控。若 GMRES 需要严格的 Newton–Krylov 线性化，则矩阵向量乘法应包括重构映射的响应；冻结变分系数得到的是近似 Jacobian / Picard 预条件，需在外层更新重构并验证缺陷收敛。

## 7. 守恒半离散方程

对静止网格，用节点对偶体平均 \(\bar{\boldsymbol U}_i\) 作未知量，记广延守恒量
\[
\boldsymbol Q_i=V_i\bar{\boldsymbol U}_i.
\]
内部宏面 \(e\) 的有向积分通量记为 \(\widehat{\boldsymbol I}_e\)，节点关联符号为 \(B_{ie}=\pm1\)。有限体积残差为
\[
\boldsymbol{\mathcal R}_i(\bar{\boldsymbol U})=
-\sum_{e\in E(i)}B_{ie}\widehat{\boldsymbol I}_e-
\boldsymbol I_{\partial V_i},\qquad
\boxed{\frac{d\boldsymbol Q_i}{dt}=\boldsymbol{\mathcal R}_i(\bar{\boldsymbol U}).}
\]
同一内部面通量在两端符号相反，因此全域内部通量严格抵消。定义 \(J_{ij}=\partial\boldsymbol{\mathcal R}_i/\partial\bar{\boldsymbol U}_j\)。精确变分重构映射会让 (J) 的数学依赖通过全局重构系统扩展到整个连通网格；工程上常用冻结重构的局部面 Jacobian 作为近似，或用矩阵自由差分/自动微分把重构影响计入 Jacobian-vector product。

## 8. BDF2 物理时间离散

### 8.1 定步长 BDF2

对于均匀物理步长 \(\Delta t\)，在新时刻隐式离散为
\[
\boxed{
\frac{3\boldsymbol Q_i^{n+1}-4\boldsymbol Q_i^n+
\boldsymbol Q_i^{n-1}}{2\Delta t}
=\boldsymbol{\mathcal R}_i(\bar{\boldsymbol U}^{n+1}).}
\]
因 \(V_i\) 固定，也可逐控制体写成平均状态形式
\[
V_i\frac{3\bar{\boldsymbol U}_i^{n+1}-4\bar{\boldsymbol U}_i^n+
\bar{\boldsymbol U}_i^{n-1}}{2\Delta t}
=\boldsymbol{\mathcal R}_i(\bar{\boldsymbol U}^{n+1}).
\]
其时间导数近似为二阶精度。第一物理步没有 \(n-1\) 层，使用后向 Euler 启动：
\[
\frac{\boldsymbol Q_i^1-\boldsymbol Q_i^0}{\Delta t}
=\boldsymbol{\mathcal R}_i(\bar{\boldsymbol U}^{1}).
\]
启动步只用于建立两层历史；之后每步保持 \(\bar{\boldsymbol U}^n\) 和 \(\bar{\boldsymbol U}^{n-1}\) 不变，直至新物理层内迭代收敛，再整体移动历史。不能在伪时间子迭代中提前覆盖历史场。

变步长时令 \(r=\Delta t_n/\Delta t_{n-1}\)，导数公式为
\[
\frac1{\Delta t_n}\left[
\frac{1+2r}{1+r}\boldsymbol Q^{n+1}
-(1+r)\boldsymbol Q^n+\frac{r^2}{1+r}\boldsymbol Q^{n-1}\right].
\]
定步长格式对应 \(r=1\)。时间步变化时必须同时更新残差和隐式对角项中的首系数，不能仍使用 \(3/2\)。

### 8.2 双时间伪时间内迭代

为把每个非线性 BDF2 方程变成可迭代的伪稳态问题，在新物理步内引入伪时间 \(\tau\)：
\[
V_i\frac{\partial\bar{\boldsymbol U}_i}{\partial\tau}
+V_i\frac{3\bar{\boldsymbol U}_i-4\bar{\boldsymbol U}_i^n+
\bar{\boldsymbol U}_i^{n-1}}{2\Delta t}
=\boldsymbol{\mathcal R}_i(\bar{\boldsymbol U}).
\]
伪时间导数收敛为零后，所得解正是原 BDF2 离散解。用伪时间后向 Euler 修正，从当前内迭代状态 \(\bar{\boldsymbol U}^{(m)}\) 求增量 \(\delta\bar{\boldsymbol U}\)。定义物理缺陷
\[
\boldsymbol F_i^{(m)}=
V_i\frac{3\bar{\boldsymbol U}_i^{(m)}-4\bar{\boldsymbol U}_i^n+
\bar{\boldsymbol U}_i^{n-1}}{2\Delta t}
-\boldsymbol{\mathcal R}_i(\bar{\boldsymbol U}^{(m)}).
\]
一阶线性化后求
\[
\boxed{
\sum_j\left[
\left(\frac{V_i}{\Delta\tau_i}+\frac{3V_i}{2\Delta t}\right)
\boldsymbol I\,\delta_{ij}-J_{ij}
\right]\delta\bar{\boldsymbol U}_j=-\boldsymbol F_i^{(m)},\qquad
\bar{\boldsymbol U}^{(m+1)}=
\bar{\boldsymbol U}^{(m)}+\lambda\delta\bar{\boldsymbol U}.}
\]
\(\Delta\tau_i\) 可按局部 CFL 选取，影响伪时间收敛速度；物理步长 \(\Delta t\) 决定真实时间离散精度。每个物理步应检查物理缺陷范数，而不只检查增量范数。对于守恒量方程，BDF2 的物理质量项作用于全部守恒变量；若用于压力不带物理时间导数的人工压缩系统，则需另设物理质量矩阵，不能直接套用此处全变量单位矩阵。

## 9. LU-SGS 线性求解与 BDF2 校正

### 9.1 矩阵分裂

将双时间线性系统记为
\[
\mathbb A\,\delta\boldsymbol U=\boldsymbol b,
\qquad \mathbb A=\mathbb D+\mathbb L+\mathbb U,
\]
其中 \(\mathbb D\) 是节点对角块，\(\mathbb L\)、\(\mathbb U\) 分别为按节点编号排序后的严格下、上三角耦合块。精确 Jacobian 可能昂贵，可用局部 Roe/Rusanov 波速构造对角占优近似，例如
\[
D_i\approx
\left[\frac{V_i}{\Delta\tau_i}+\frac{3V_i}{2\Delta t}
+\frac12\sum_{e\ni i}S_e\lambda_e\right]\boldsymbol I,
\]
其中 \(S_e\) 是宏面面积，\(\lambda_e\) 为正的对流谱半径；粘性问题可再加入粘性谱半径估计。面间块由冻结重构后数值通量的局部导数给出。该式是预条件近似，不是完整高阶离散的精确 Jacobian。

### 9.2 对称前后扫描

LU-SGS 用如下因子近似
\[
\mathbb P=(\mathbb D+\mathbb L)\mathbb D^{-1}(\mathbb D+\mathbb U),
\]
对每个增量方程先正向再反向扫描。

**正向扫描**，按节点从小到大：
\[
(D_i+L_{ii})\boldsymbol y_i=
\boldsymbol b_i-\sum_{j<i}L_{ij}\boldsymbol y_j.
\]
实际 \(L_{ii}=0\)，故每步只需解一个 \(n_v\times n_v\) 的局部对角块。

**反向扫描**，按节点从大到小：
\[
D_i\delta\boldsymbol U_i=
D_i\boldsymbol y_i-\sum_{j>i}U_{ij}\delta\boldsymbol U_j,
\]
即
\[
\delta\boldsymbol U_i=\boldsymbol y_i-
D_i^{-1}\sum_{j>i}U_{ij}\delta\boldsymbol U_j.
\]
所得 \(\delta\boldsymbol U\) 是一次 LU-SGS 近似解；若只做一组前后扫，它是低成本近似线性求解，若再以线性残差做 correction sweeps，则继续逼近原线性系统。伪代码为：

```text
输入：当前状态 U、物理历史 U^n/U^(n-1)、缺陷 b、近似块 L/D/U
y = 0
for i = 1,...,N:
    y_i = D_i^{-1} (b_i - sum_{j<i} L_ij y_j)
deltaU = 0
for i = N,...,1:
    deltaU_i = y_i - D_i^{-1} sum_{j>i} U_ij deltaU_j
U <- U + relaxation * deltaU
若物理缺陷未收敛，则重新计算重构、通量和下一次修正
```

节点顺序和跨 MPI 分区通信会影响 LU-SGS 预条件质量。分布式实现通常在 forward/backward sweep 之间同步 ghost；跨 rank 的节点只能使用当轮收到的边界值，因而是近似 SGS。通过图着色、多色 SGS 或在 rank 间采用 block Jacobi 可改善并行实现，但都应以物理缺陷收敛为判断依据。

## 10. GMRES 与 LU-SGS 预条件

### 10.1 为什么用于流动时间方程

BDF2 线性系统含有对流通量 Jacobian，通常非对称；中心/耗散项、边界条件、粘性项也会影响谱性质。GMRES 不要求矩阵对称正定，能最小化 Krylov 子空间中的线性残差。LU-SGS 可作为左预条件器，形成
\[
\mathbb P^{-1}\mathbb A\,\delta\boldsymbol U=
\mathbb P^{-1}\boldsymbol b.
\]
一次 \(\mathbb P^{-1}\) 应用就是上一节的一组 LU-SGS 前后扫描。若使用会随 Krylov 步变化的非线性预条件操作，则应换用 flexible GMRES；若每次都是固定的线性 LU-SGS 近似，标准左预条件 GMRES 即可。

### 10.2 Arnoldi–GMRES 步骤

令 \(\mathbb M=\mathbb P^{-1}\)，从 \(\delta\boldsymbol U_0\) 开始：

1. 计算左预条件初始残差 \(\boldsymbol r_0=\mathbb M(\boldsymbol b-\mathbb A\delta\boldsymbol U_0)\)，令 \(\beta=\|\boldsymbol r_0\|_2\)、\(\boldsymbol v_1=\boldsymbol r_0/\beta\)。分布式范数含一次 MPI 全局求和。
2. 对 \(j=1,\ldots,m\)：计算 \(\boldsymbol w=\mathbb M\mathbb A\boldsymbol v_j\)，用 modified Gram–Schmidt 对 \(\boldsymbol v_1,\ldots,\boldsymbol v_j\) 正交化，得到 Hessenberg 矩阵第 \(j\) 列及新向量 \(\boldsymbol v_{j+1}\)。
3. 对 Hessenberg 小矩阵逐列应用 Givens 旋转，更新最小二乘残差估计；若 \(\|r_j\|/\|r_0\|<\varepsilon_{\mathrm{lin}}\)，结束当前 Krylov 循环。
4. 求解小型上三角系统并更新 \(\delta\boldsymbol U=\delta\boldsymbol U_0+V_m\boldsymbol y_m\)。未收敛时以新残差重启，直到达到容差或最大重启数。

核心伪代码：

```text
deltaU = 0
repeat for each restart:
    r = M * (b - A * deltaU)
    beta = norm(r)
    v[1] = r / beta
    for j = 1,...,restart_length:
        w = M * (A * v[j])       # A 的作用可矩阵自由计算
        for k = 1,...,j:
            H[k,j] = dot(w, v[k])
            w -= H[k,j] * v[k]
        H[j+1,j] = norm(w)
        v[j+1] = w / H[j+1,j]
        更新 Givens 旋转和残差估计
        若达到线性容差则 break
    解 min || beta*e1 - H*y ||，并令 deltaU += V*y
直到线性容差或重启上限
```

### 10.3 Jacobian–vector product 与重构耦合

GMRES 只需要线性算子 \(\mathbb A\boldsymbol v\)，不必显式存储完整 Jacobian。可用解析面通量导数、自动微分，或残差差分
\[
\mathbb A\boldsymbol v\approx
\frac{\boldsymbol F(\bar{\boldsymbol U}+\epsilon\boldsymbol v)-
\boldsymbol F(\bar{\boldsymbol U})}{\epsilon}.
\]
最后一种做法要求每次 \(\boldsymbol F\) 评估都按一致规则完成变分重构。冻结系数得到的 \(\mathbb A\boldsymbol v\) 是近似线性化，可用于预条件或 inexact Newton，但不能称为对完整变分离散残差的精确 Jacobian。

建议的双时间内层顺序是：

```text
固定 U^n、U^(n-1)，令 U* = U^n
for pseudo iteration m = 0,...:
    从 U* warm-start 并更新变分系数 C
    用 EfficientDifferential 计算积分残差 R(U*)
    形成 BDF2 物理缺陷 F(U*)
    若 ||F|| 已达标，结束当前物理步
    构造/作用 A = V/dtau + 3V/(2dt) - dR/dU
    用 LU-SGS 直接近似求解，或用 GMRES(LU-SGS) 求解 A*dU = -F
    U* <- U* + relaxation*dU
提交 U^(n+1)=U*，之后才移动两层时间历史
```

## 11. 实施时应报告的残差与限制

建议分别监控：

1. **变分重构残差**：\(\|\mathbb A C-B\|\) 或界面 jet 泛函 \(\mathcal I\) 的相对下降；
2. **物理 BDF2 缺陷**：\(\|3Q^{n+1}-4Q^n+Q^{n-1}-2\Delta t\mathcal R(Q^{n+1})\|\) 的体积/变量归一化范数；
3. **线性求解残差**：\(\|b-A\delta U\|/\|b\|\)，不要只报告 GMRES 内部的预条件残差；
4. **守恒误差**：全域质量、动量和能量的边界通量收支差。

激波附近应在变分系数得到后实施适当限制，限制器可能使局部多项式不再满足未限制变分方程，也会降低局部阶数。物理步长收敛判据应基于限制后的实际残差。有限次重构 sweeps、冻结 Jacobian、LU-SGS 扫描和松弛都是求解策略；它们不会改变目标方程的形式，但若内迭代未充分收敛，会引入额外迭代误差。

## 12. 与源论文及移植说明的对应

- 王乾博士论文《非结构网格紧致高精度有限体积方法》第 4 章第 4.1 节（印刷页 74–79）给出零均值重构、面上 IJI、全局稀疏方程、SPD 证明与 Block Jacobi / Gauss–Seidel / SOR 解法；第 2 章第 2.3.2 节（印刷页 25–27）介绍隐式残差线性化、LU-SGS 与 GMRES+LU-SGS[1](#ref_001)。
- 本文将其“全域界面泛函的极小化”转化为节点边中点的值—梯度—Hessian jet 泛函，以适配二次 NCFV 重构和压缩 differential 权重。该改写的矩阵与当前 NCFV 变分重构行算子相符，但它是 NCFV 离散选择，不是论文逐式原样复制。
- 用户提供的《ncfv_efficient_differential_porting_zh.md》描述了高效算法的对偶几何、解析微分积分权重、点值恢复和宏面一次 Riemann 求解；本文沿用这些定义，并把 BDF2 作为隐式时间推进扩展单独推导。王乾论文中非定常示例使用的是 SDIRK4，因此本文的 BDF2 方程不是对论文时间格式的复述。

## 参考文献

[]{#ref_001}[1] 王乾. 非结构网格紧致高精度有限体积方法[D]. 北京: 清华大学, 2017.
