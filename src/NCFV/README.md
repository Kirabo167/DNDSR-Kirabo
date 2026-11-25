# NCFV — Node Center Finite Volume Method

`NCFV` 全称为 **Node Center Finite Volume Method**，是 DNDSR 内并列的
独立三阶格点型有限体积求解器模块。它复用 DNDSR 的 CGNS 读取、网格分区、
MPI 数组、JSON 配置、Euler 黎曼通量和共享 RANS 闭合。数学推导和
路线分析见
[`docs/reports/DNDSR_高效三阶格点型有限体积法_数学推导_MPI设计_可行性报告.pdf`](../../docs/reports/DNDSR_高效三阶格点型有限体积法_数学推导_MPI设计_可行性报告.pdf)。

## 1. 唯一的对偶控制体几何定义

第一版只接受 O1 原始网格。所有构造点都由原始角点的坐标算术平均定义：

\[
\boldsymbol x_e=\frac{\boldsymbol x_i+\boldsymbol x_j}{2},\qquad
\boldsymbol x_f=\frac1{N_f}\sum_{a\in V(f)}\boldsymbol x_a,\qquad
\boldsymbol x_c=\frac1{N_c}\sum_{a\in V(c)}\boldsymbol x_a.
\]

这里的面点和体点不是面积质心或体积质心。对于四边形、六面体、棱柱和
金字塔，两者通常不同；上述顶点等权平均就是算法定义。

- 二维节点份额：对每个拓扑链 `(cell, edge, node)` 构造微三角形
  `conv(node, edge-average, cell-average)`。
- 二维内部对偶面：原始边对应相邻单元中的线段
  `conv(edge-average, cell-average)`；物理边界用
  `conv(node, edge-average)` 闭合。
- 三维节点份额：对每个拓扑链 `(cell, face, edge, node)` 构造微四面体
  `conv(node, edge-average, face-average, cell-average)`。
- 三维内部对偶面：每条原始边对应所有
  `conv(edge-average, face-average, cell-average)` 微三角形的并；物理边界
  使用 `conv(node, edge-average, face-average)`。

初始化阶段保存 cell/face/edge 构造点坐标、节点对偶体体积以及关于节点
\(x_j\) 的局部矩 \(c_{1,j}=\int(x-x_j)dV\)、
\(C_{2,j}=\int(x-x_j)(x-x_j)^T dV\)、边宏面有向面积、边界片及压缩积分
权重。局部矩在每个微单纯形上直接用相对坐标累积，不先构造绝对二阶矩
\(\int xx^T dV\)，从而避免细网格上的消减误差。对每个节点还检查
\(\sum_e B_{ie}\boldsymbol A_e+\sum_b\boldsymbol A_b=0\) 的几何闭合。

## 2. 两种 JSON 可切换算法

两种积分模式共用同一套对偶拓扑、可选重构方法、限制器、Euler 通量和
SSPRK3 时间推进，只替换体/面的积分实现。

### `EfficientDifferential`

设一个线段、三角形或四面体微单纯形有 \(n\) 个构造顶点
\(\boldsymbol r_r\)，以原始节点 \(i\) 为展开点，令
\(\boldsymbol d_r=\boldsymbol r_r-\boldsymbol x_i\)、
\(\boldsymbol D=\sum_r\boldsymbol d_r\)。构造点梯度按与坐标完全相同的
顶点系数插值，
\(\boldsymbol g(\boldsymbol r_r)=\sum_m C_{rm}\boldsymbol g_m\)。
将二阶项用全微分

\[
\boldsymbol H_i\boldsymbol d_r
=\boldsymbol g(\boldsymbol r_r)-\boldsymbol g_i+O(h^2)
\]

替换后，单纯形积分可压缩为

\[
\int_K u\,dK=|K|u_i+\sum_m\boldsymbol\ell_{im}^{K}\!\cdot\boldsymbol g_m,
\]

\[
\boldsymbol\ell_{im}^{K}
=\frac{|K|}{n}\boldsymbol D\,\delta_{im}
+\frac{|K|}{2n(n+1)}\left[
\boldsymbol D\sum_r(C_{rm}-\delta_{im})
+\sum_r\boldsymbol d_r(C_{rm}-\delta_{im})\right].
\]

这些权重在初始化时逐微单纯形累加。节点恢复、左右宏面平均和中心物理通量
由压缩权重计算。每条原始边调用一次黎曼求解器形成宏面耗散。启用 RANS 时，
额外保留单纯形体积分点与二次重构系数，用于非线性湍流源项。

每个内部宏面使用一个按 DNDS 进程局部节点编号严格排序、去重的
`efficientStencil`。每个条目同时保存该节点对左右重构的守恒量梯度权重、
物理通量梯度矩阵权重和黏性梯度权重；零阶守恒量及通量权重只在左右锚点
非零，分别直接复用宏面的 `measure` 和 `vectorMeasure`，不在每个 support
条目中复制零值。边界片把相同的仿射状态/梯度
积分系数合并为单一 `integrationWeight`，控制体格点恢复使用
`pointRecoveryStencil`。计算阶段不再在五组独立稀疏数组中重复查找同一
节点，而是用条目内的节点编号直接访问已同步的点值、状态梯度和物理通量
梯度。编号范围 `[0, NodeHalo::NumNode())` 表示 owned
节点，`[NodeHalo::NumNode(), NodeHalo::NumNodeProc())` 表示本 rank 的精确
依赖 ghost；它与网格对象为几何初始化保留的 cell/node halo 相互独立。
权重中不保存跨阶段裸指针，以免 MPI 数组重新分配后产生悬空地址。

### `TraditionalQuadrature`

保存完整二次重构系数，并把每个对偶微体/微面映射到 DNDSR 已有规则：

- 二维体：`Tri3` Hammer 型规则；二维面：`Line2` Gauss-Legendre 规则；
- 三维体：`Tet4` 单纯形规则；三维面：`Tri3` Hammer 型规则。

体积分点用于初始化对偶体矩和零均值二次基；内部面、边界面积分点在每次
残差计算中评价左右二次多项式并调用数值通量。体积分由
`quadratureOrder` 控制（默认 4）；界面积分独立采用
`surfaceQuadratureOrder=3`。初始化阶段逐微面计算并保存物理积分点坐标
\(\boldsymbol{x}_g\)、正标量权重 \(w_g\) 和有向量权重
\(\boldsymbol{n}_g w_g\)，运行期不再重复进行参考坐标到物理坐标的映射。
这里“三阶”表示公式能精确积分二次多项式；DNDSR 底层求积接口接收的是
“需要精确的多项式次数”，因此实际向接口传入 2。所得二维微线段规则为
2 点 Gauss--Legendre（精确到三次），三维微三角面规则为 3 点 Hammer
（精确到二次），两者都能精确积分界面上的任意二次多项式，同时不保存
多余的高阶积分点。

## 3. 重构、通量与 MPI

节点对偶体均值采用完整二次基重构。JSON 配置项
`reconstruction.method` 可以选择 `LeastSquares`（正规方程和 LDLT）、
`SVDLeastSquares`（Jacobi SVD 伪逆）或 `Variational`（变分重构）；
中文别名分别为 `最小二乘重构`、`SVD最小二乘重构` 和 `变分重构`。
默认仍为 `SVDLeastSquares`。无黏和层流高效模式中的 LS 与 SVDLS 只保存逆算子的前
`dimension` 行，因此一阶梯度仍来自完整二次问题。传统模式和 RANS 保存全部
5（二维）或 9（三维）行。候选点按原始节点边图逐环扩展，完整一环
无条件保留；二环候选优先按近似反向成对加入，直到达到
`ceil(stencilSizeFactor * basisSize)`。配对先补强当前加权矩阵的最弱谱方向，
再最小化归一化三次奇单项式之和、一次位移偏置和反向缺陷。
默认三维配置因此在棱柱精度网格上使用 16 点，而不是旧实现的 26--40 点。
模板选择不依赖邻接遍历顺序。`LeastSquares` 从
\(A^TW^2A\) 的对称特征值估计秩和 \(\kappa(WA)\)，然后求解正规方程；
`SVDLeastSquares` 直接使用加权矩阵的奇异值与伪逆。只有目标点数不能满足
秩或条件数上限时才按对应指标增加候选点。每个算子还记录线性系数对
归一化三次点值基的响应范数
`cubicErrorIndicator`。初始化日志输出模板点数、一环保留数、最大条件数和
最大三次指标，便于定位局部精度损失。

模板选点策略独立于重构方法和限制器，通过
`reconstruction.stencilSelectionStrategy` 选择：

- `Original`（默认）：保留原来的几何排序、逐圈反射分组和接受逻辑，
  用于复现已有算例；旧 JSON 缺少该字段时也采用这一策略。
- `RankAware`：保留第一圈和点数预算，直接用加权设计矩阵的 SVD 检查
  反射模板的秩与条件数；拒绝秩亏或比同点数紧凑模板条件数差十倍以上的
  选择。先搜索允许圈层内的可靠反射模板，必要时再用谱方向紧凑选择回退，
  并统一几何排序键的坐标舍入，减少 MPI 分区带来的选点差异。

例如，在任一 NCFV JSON 的 `reconstruction` 对象中设置
`"stencilSelectionStrategy": "RankAware"` 即可启用修复策略；改为
`"Original"` 即可恢复原策略。两种策略都支持 `completeStencilRings`，
所有后续共享梯度、冻结预算、压力与通量限制器开关保持独立。
三棱柱 LS13 在 `Original` 下仍可能出现原有的 `8/9` 秩亏，
需要使用该配置时应选择 `RankAware`。初始化日志会打印实际选用的策略。

`Variational` 在两种积分模式下采用相同的边中点变分重构，保留完整二次系数，
对相邻原始节点的边中点值、一阶导数和 Hessian 跳跃构造局部泛函，
以最小二乘二次多项式为初值进行邻居迭代。`variationalWeight` 控制
值与一阶导数跳跃相对 Hessian 跳跃的权重；`variationalIterations`
和 `variationalRelaxation` 分别控制迭代轮数与松弛系数。
传统积分使用完整二次系数在求积点计算通量；高效积分从二次系数的
一次项提取格点梯度，并使用高效模式的梯度权重恢复点值及计算通量。
高效积分不使用变分 Hessian 做通量积分，也不启用此前移除的界面面均值
变分泛函和 PCG 路径。

无黏通量默认使用标准特征 Roe 路径 `Roe`；配置也支持 `HLLC`、
`HLLEP`、`HLLEP_V1` 及已列入 schema 的 Roe 变体，可用于同条件精度对照。
`Roe` 路径调用
`RoeFlux_IdealGas_HartenYee<...,0>`，对三组 Roe 特征值施加 Harten--Yee
H2 熵修正。`Roe_M2` 在共享 Euler 通量库中的实际含义是标量
LLF/Rusanov；NCFV 配置校验会拒绝该选项，避免再把它误当作 Roe 通量。

两种模式均按马润之硕士论文式（3-34）使用对偶控制体的分方向半跨度：

\[
\Delta x_i=(x_{\max,i}-x_{\min,i})/2,\quad
\Delta y_i=(y_{\max,i}-y_{\min,i})/2,\quad
\Delta z_i=(z_{\max,i}-z_{\min,i})/2.
\]

令 \(H_i=\operatorname{diag}(\Delta x_i,\Delta y_i,\Delta z_i)\)，
基函数使用 \(\boldsymbol\xi=H_i^{-1}(\boldsymbol x-\boldsymbol x_i)\)，
再减去目标控制体上的体均值；邻居积分也必须使用目标节点的 \(H_i\)。
二次矩按 \(H_i^{-1}M_{2,i}H_i^{-1}\) 缩放，含所有交叉项。
LS 与 SVDLS 的高效模式从完整伪逆截取一次系数后，按
\(\nabla u_i=H_i^{-1}\boldsymbol a_{1,i}\) 还原物理梯度。
传统模式在积分点及限制器采样点使用同一套尺度，并对基函数求物理导数。
代码中纯二次项仍采用 \(\xi_d^2/2\) 的系数约定，不影响归一化及多项式空间。

初始化期间直接从每个微单纯形的构造顶点累积节点相对坐标极值，
缓存 `lowerOffset`、`upperOffset`、`referenceLengths`，不需要积分点。
即使 `retainMicroGeometry=false` 也保留这些缓存；二维只检查 x/y 方向，
未使用的 z 尺度设为 1。退化的有效方向直接报错，不静默退回体积尺度。
参考长度随专用节点 owner/ghost 映射交换。全平移周期拓扑在 CGNS 读取后
先合并同一物理节点；所有微几何在相应周期像的局部坐标框架内累积，因此
不能对周期盒两端的绝对坐标直接取包围盒。

`lengthScale=V_i^(1/d)` 仅保留给距离权重及原有 CFL/壁面尺度用途，
不再用于重构基函数。此修改不改变距离权重公式、重构模板目标大小、
积分规则或 JSON 模式名称；秩和条件数检查仍基于归一化后的加权矩阵。
重启只读取物理状态，参考长度随网格重新初始化，无需更改重启文件格式。

NCFV 只保留长期稀疏 MPI 路径：

1. 读取时把三个有序主面/供体面映射为 DNDSR 周期 ID，调用
   `Deduplicate1to1Periodic` 合并周期节点拓扑，再分区。周期边和面使用
   “规范化全局节点键 + 相对 periodic bits”构建分布式哈希目录，不复制
   全局拓扑。
2. 常规路径构建一层**点相邻** cell halo。DNDSR 的 `cell2cell` 是
   `cell -> node -> cell`，这一层足以补全 owned 节点的 cell star、
   对偶体几何和 owner 侧一环边邻接。
3. `NodeHalo` 在初始化时发布 owned 节点的边邻接行，按
   `maximumStencilRings` 逐环稀疏 pull 所需远程邻接行及坐标/中心矩/参考
   长度。LS/SVDLS 选定模板后，把重构、点值恢复、内部面、边界面积分所引用的
   全局节点编号取并集，重新建立最终精确 owner/ghost 映射和持久 MPI 类型。
   最终释放候选环、临时几何和邻接表。
4. 节点均值、梯度或二次系数、点值及必要的限制器系数均借用这个最终映射。
   运行期只对依赖列表中的 owner/ghost 做点对点稀疏 pull；周期路径不再调用
   `MPI_Allgatherv`，也不让每个 rank 保存全局状态。
5. 原始边仍以相邻单元 owner 的最小 rank 唯一拥有。edge owner 计算唯一一份
   积分通量，其他 rank 只 pull 所需 ghost edge；节点 owner 按有向关联号
   \(B_{ie}=\pm1\) 累加后除以对偶体积。因此跨分区边两端严格使用同一份
   通量，不需要残差 push-sum。
6. SSPRK3 每个物理步只在第一子步更新 CFL，在第三子步计算一次全局残差
   范数；局部步长在精确 halo 路径中只保存 owned 值，不再执行无用途的
   ghost 交换。物理 CFL 仍需要一次 `MPI_MIN`，残差报告仍需要一次
   `MPI_SUM`，这些是标量规约而非全局场复制。

默认 `reconstruction.limitPointRecovery=false` 时，高效模式限制器约束
论文式（3-94）--（3-100）的宏界面平均值，而不是
原始边中点值。对于界面 \(S_{ij}\) 的 \(i\) 侧，先用预计算权重得到

\[
\bar U_i^{ij}=U_i+S_{ij}^{-1}\sum_m\boldsymbol\omega^i_{ij,m}
\cdot\nabla U_m,
\]

再用端点格点值 \(U_i,U_j\) 构成逐分量上下界。所得单个锚点系数
\(\phi_i\) 作用于该侧完整权重和；不能把和式内每个共享梯度永久乘以各自
节点的系数，否则不再等价于论文式（3-96）。格点值恢复仍使用未限制梯度；
限制系数只在该侧界面均值、物理通量梯度积分和黏性界面梯度中生效。
跨分区节点计算限制器时，会在初始化阶段额外构造所需 ghost 原始边的宏面
权重；重构限制与通量限制均关闭时不保存这部分权重。

间断问题可选用 `reconstruction.limitPointRecovery=true`（要求高效模式且
启用限制器）。此时以守恒体均值为锚，以自身及一环相邻节点的体均值构成
逐守恒分量上下界，对原始恢复点和每个宏界面的完整增量共用一个系数。
再检查这些候选状态的密度、压力，必要时通过二分进一步收缩系数。
系数为零时点值及界面状态退化为本节点体均值；体均值不被裁剪。
共享梯度保留原值，避免对同一宏面叠加不同节点的限幅。
高效模式增量方向判断采用固定绝对容差 `1e-10`；绝对值不超过该阈值的
增量不参与限幅系数的收缩，避免舍入噪声触发完全限幅。

`reconstruction.barthZeroOrderFallback=true` 选择二元 Barth 退阶，默认关闭。
要求高效无黏模式、启用限制器和 `limitPointRecovery=true`，不与共享梯度
扫面或完整通量校正同时启用。恢复点、内部宏面均值或边界片均值超过自身及
一环守恒均值界限 `1e-10` 时，该锚点直接取系数零；合格候选取一。
非物理重构也退阶。系数零时点值和内部面该侧状态等于本控制体守恒均值，
物理通量积分只保留常值项；共享源梯度不被改写。外推边界退阶为同一体均值
的常值通量，并对最终模板点形成的实际边界均值再次检查，超界的边界片也
使用常值通量。此策略不承诺下一时刻均值严格满足最大值原理。

`reconstruction.sharedGradientFaceSweep=true` 选择另一种共享梯度策略，
默认关闭，要求高效模式且启用限制器。每个物理界面只遍历一次，使用当时
已经更新的共享梯度计算两侧完整均值，以各侧自身及一环体均值为固定上下界。
若任一侧越界，取两侧守恒分量要求及密度、压力保正要求的最小系数，
把两侧面积分模板和点值恢复模板并集中的每个节点梯度同乘该系数一次。
同节点参加后续界面时继续累乘；每个界面内部先去重，不重复处理同一节点。
这一路径代替原有侧面锚点限制，侧面系数保持 1，体均值不修改。

MPI 实现按 rank 0 到 rank P-1 顺序处理各自拥有的界面；每个 rank 内按原始
节点编号和周期位移排序，处理后立即把远程节点的新增累计系数交给 owner，
并同步标量累计系数后再进入下一 rank。独立工作 halo 包含远程锚点点值恢复
所需的完整节点集合。该顺序等价实时逐界面更新，不是先冻结全部界面系数
再统一相乘；限制器的这部分面计算跨 rank 串行，结果可能依赖分区和顺序。
一次遍历结束后不反复检查或追加扫面。最后写回、同步共享梯度，重新恢复
并保正节点值，再以更新后的状态和梯度计算物理通量链式导数。
该实验策略不与 `limitFluxCorrection` 或 `directionalPairBounds` 同时启用。

`reconstruction.sharedGradientFaceSweepParallel=true` 改用完整模板界面分组，
要求同时开启 `sharedGradientFaceSweep`；默认 false 保留原 rank 顺序。
初始化时将两侧点值恢复与界面积分的所有非零梯度依赖合并、去重，
由节点 owner 仲裁各 rank 提议的颜色。同一颜色中任意两界面的完整支持
集合不相交，包括已经合并的周期节点；每个接受的节点/颜色对均检查唯一性。
分组只在固定网格初始化时计算，不集中收集全网格冲突图。
运行时同色界面在各 rank 同时处理，随后用缓存的稀疏点对点表将新增系数
送回 owner，再同步累计系数，供下一颜色计算。运行阶段没有按 rank 串行的
面循环或 `Alltoall[v]` 系数交换；每个界面仍将同一系数乘到全部支持节点一次。
颜色顺序改变了原来的面顺序，结果不保证与旧 rank 顺序或不同分区完全一致；
一遍扫面也不额外保证最终所有界面仍在极值界内。诊断接口分别记录事件刚处理
后的状态和全遍历结束后的状态，避免混淆这两种约束。

`ncfv_shared_gradient_np{1,2,4,8}` 检查跨 rank 支持冲突、独立累乘积和
同颜色顺序的串行通信参考，并报告事件/最终正性、界面越界量及 z 梯度。
小网格 RP1 测速入口为 `cases/ncfv_euler/3D/shared_gradient_parallel/benchmark.py`，
使用 40×40×20 网格、单元尺度比 Δz/Δx=4、固定时间步 0.0002，共 12 步，
排除前两步并在计时循环外保存最终 H5。依次比较 64、128、256 个独立物理核，
同一可执行文件通过配置开关选择原串行或分组实现；不混用超线程。

2026-10-28 本机短程实测（第 3–12 步均值，单位秒/物理步）：

| MPI 进程/物理核 | 原 rank 串行 | 完整模板分组并行 | 同进程数加速比 | 颜色数 |
| --- | ---: | ---: | ---: | ---: |
| 64 | 0.424129 | 0.085455 | 4.96 | 210 |
| 128 | 0.387420 | 0.053195 | 7.28 | 211 |
| 256 | 1.554458 | 0.056427 | 27.55 | 206 |

分组初始化一次耗时分别为 1.12、0.71、0.81 秒；上表不含初始化和最终 H5 写盘。
这只支持上述小网格的强扩展结论，不能直接外推到 400×400×20。
1/2/4/8 进程测试的系数乘积误差不超过 2.78e-17，同颜色串行参考的梯度差为 0。
处理每个界面时的守恒分量越界量不超过 8.89e-16，但平滑三维场中，遍历结束
后的最大越界量仍可达 4.38e-3；原 rank 串行版本同样有此现象。
RP1 在 t=0.0024 时并行版的最大层间密度差依次为 6.49e-5、7.36e-5、1.29e-4，
最大 |w| 分别为 2.98e-6、1.66e-6、4.83e-6。正密度、正压力检查通过，但此改动
没有消除 z 向差异，部分差异比原串行顺序更大。
完整原始数据、每步日志、各层 NPZ 和最终 H5 位于
`data/out/NCFV/shared_gradient_parallel_validation_20261028/`。

`reconstruction.limitPressure=true` 在此基础上额外约束恢复点和全部宏面
状态的理想气体压力，要求 `limitPointRecovery=true`，默认关闭。上下界
为自身及一环相邻守恒均值计算出的压力极值，扩展容差为
`1e-10 * max(1, abs(p_min), abs(p_max))`。仍使用同一个系数收缩整个守恒
增量，不直接裁剪压力、能量或体均值，也不分别修改各守恒分量。

压力在正密度守恒线段上为凹函数，因此两端压力合格仍可能在中间超过
上界。实现用 `rho*(E-p_bound/(gamma-1))-|m|^2/2` 的二次多项式检查
内部峰值并二分求首次越界点；每个候选均限制从零开始的整个可行前缀，
保证处理后续候选时继续缩小公共系数不会破坏已处理的压力约束。
该选项约束重构状态，不单独保证下一时刻体均值无新极值。

空间重构保正不能单独保证更新后的体均值为正。非定常计算还可开启
`time.rejectNonPhysicalSteps=true`：SSPRK3 每个子步同步检查所有 owned
体均值，出现非物理状态时恢复整个物理步的起始状态，将全局步长减半重试。
仅接受的步更新物理时间和迭代数，不直接裁剪守恒量。
`time.maximumStepRetries` 默认 12，可设为 0–40；超过次数或步长下限时报错。
该选项要求 `time.useLocalTimeStep=false`。两个新增开关默认均关闭；
这套保护不构成对完整非线性离散的 TVD 或精度证明。

### 间断限制的可选对照策略

以下选项均在 `reconstruction` 中，默认关闭，供独立数值验证使用。

1. `directionalPairBounds`：要求 `limitPointRecovery=true`。恢复点仍用
   一环均值界限，每个界面改用其两端守恒均值界限；一个节点继续共用一个
   系数，因而某个横向等值界面也可能使其他方向的高阶增量降为零。
2. `limitFluxCorrection`：高效无黏模式下限制完整高阶通量，而不是只限制
   黎曼输入。要求全局 CFL 步长、CFL≤1 和几何闭合检查。令
   `n=vectorMeasure/A`、`alpha=max(|u_i·n|+c_i,|u_j·n|+c_j)`，以均值计算
   LLF 低阶通量 `F_L=A*(F_i+F_j-alpha*(U_j-U_i))/2`，构造
   `bar=(U_i+U_j)/2-(F_j-F_i)/(2*alpha)`。完整原高阶通量与低阶之差
   `C=F_H-F_L` 通过共用边系数 θ 缩放，使两端修正态
   `bar-theta*C/(alpha*A)`、`bar+theta*C/(alpha*A)` 满足各自一环均值及
   低阶 bar 的守恒分量界限和压力下界。最终唯一共享通量为
   `F_L+theta*C`，保持内部边通量守恒。界限 padding 使用 64 倍机器精度。
   压力上界集合非凸，故该方法不承诺更新后的压力上界或完整 Euler TVD。
3. `fluxCorrectionPairBounds`：要求启用通量限制。进一步把每条边的修正
   bar 界限收紧为该边两端均值与原低阶 bar 的极值，防止较宽一环区间
   容许横向反扩散。低阶 bar 始终可行，不把 Euler 的合法压缩误判成非法
   原始均值极值。
4. `completeStencilRings`：保留首个满足模板规模、秩和条件数要求的完整
   拓扑邻接环，不从该环中再裁出紧致子集。均匀三维 Hex 内部通常由
   16 点增为完整二环 24 点；边界按实际几何需要扩展。此选项修改几何
   采样集合，不依赖解值、指定坐标轴或二维假设；用于检查模板裁剪对
   三维对称性和间断重构的影响，也保留真实三维二次场的精确重构。

这些策略不清零或改写任何坐标方向的导数。按坐标检测后强行清零导数的
早期试验仅保留在冻结诊断中，未作为三维求解策略保留在主程序。

通量限制模式的非周期边界使用节点均值 LLF，边界局部为一阶；全周期
光滑波可用于分离这个边界误差。每个 RK 子步显式检查实际
`dt*sum(alpha*A)/V <= 1`，包括第一子步之后固定步长、最小步长向上钳制
的情形；超限时报告降低 CFL/最小步长，不静默沿用失效的凸组合条件。
关闭 Barth 但启用通量限制时，仍先对恢复点保正，以保证高阶通量本身
可计算；不施加 Barth 分量范围界限。`maximumFluxCorrection` 在 `[0,1]`
内，默认 1，设为 0 可获得同一低阶 LLF 参照。

## 4. 构建与运行

模块目录和命名空间仍为 `src/NCFV/`、`DNDS::NCFV`。独立程序目标为
`ncfv_euler2D` 和 `ncfv_euler3D`，不再由 `euler` 分派。公共代码放在 `ncfv_core`，
二维、三维实例化分别放在 `ncfv_2D`、`ncfv_3D`；模块测试用 `ncfv` 聚合两者。
JSON 中 `solver.type="ncfv_euler"`、`solver.discretization="NCFV"`、
`solver.model="IdealGas"`；`dimension` 和 `fieldNVariables=dimension+2` 必须匹配程序。

```bash
cmake --preset release-test
cmake --build build --target all_ncfv_euler --parallel 4

(cd build && ./app/ncfv_euler2D.exe ../cases/ncfv_euler/2D/ncfv_euler2D.json)
(cd build && ./app/ncfv_euler2D.exe ../cases/ncfv_euler/2D/NCFV_traditional.json)
(cd build && mpirun -np 4 ./app/ncfv_euler3D.exe \
    ../cases/ncfv_euler/3D/NCFV_periodic_hex_iv10.json)
```

模式只由下列 JSON 字段切换：

```json
"algorithm": {
    "mode": "EfficientDifferential"
}
```

改为 `"TraditionalQuadrature"` 即进入普通三阶数值积分路径。二维示例为
`cases/ncfv_euler/2D/ncfv_euler2D.json` 和 `cases/ncfv_euler/2D/NCFV_traditional.json`，三维示例为
`cases/ncfv_euler/3D/ncfv_euler3D.json`。命令行也支持 JSON pointer 覆盖及配置 schema 输出：

```bash
./app/ncfv_euler2D.exe ../cases/ncfv_euler/2D/ncfv_euler2D.json --emit-schema
./app/ncfv_euler2D.exe ../cases/ncfv_euler/2D/ncfv_euler2D.json \
  -k /algorithm/mode -v TraditionalQuadrature
```

传统变分重构的三棱柱等熵涡示例为
`cases/ncfv_euler/3D/NCFV_traditional_variational_vortex.json`，默认 IV10、
固定步长 0.2，推进至 `t=1.6`：

```bash
cd build
mpirun -np 1 ./app/ncfv_euler3D.exe ../cases/ncfv_euler/3D/NCFV_traditional_variational_vortex.json
```

在同一网格、变分参数和时间步长下，仅切换通量积分实现可运行
“变分重构一次导数 + 高效通量积分”：

```bash
mpirun -np 1 ./app/ncfv_euler3D.exe ../cases/ncfv_euler/3D/NCFV_traditional_variational_vortex.json \
  -k /algorithm/mode -v EfficientDifferential \
  -k /io/outputPrefix -v ../data/out/NCFV/efficient_variational/iv10/solution
```

等熵涡的体积加权格点密度误差写入
`io.outputPrefix + ".diagnostics.csv"`。

真实三维平移周期网格建议显式给出周期长度及有序面配对：

```json
"mesh": {
    "periodicLengths": [10, 10, 4],
    "periodicBoundaryPairs": [
        "bc-2", "bc-2-1",
        "bc-3", "bc-3-1",
        "bc-4", "bc-4-1"
    ]
}
```

每对顺序都是 `[main, donor]`。省略 `periodicBoundaryPairs` 时会按
`Periodic boundaryZones` 的配置顺序推断；部分周期只列出启用轴对应的
Periodic 面对，并按 x/y/z 启用轴顺序排列。例如仅 z 周期使用
`periodicLengths=[0,0,Lz]`、`periodicBoundaryPairs=["ZMIN","ZMAX"]`，
这两个面的物理边界模式均为 `Periodic`，x/y 的边界模式独立配置。
显式填写更不易因 JSON 重排而改变方向。
程序固定构建一层点相邻几何 cell halo，再按实际重构、恢复和
通量积分依赖裁剪精确的节点 owner/ghost halo；这两层通信结构均不再由
JSON 开关选择。

`--check-config` 可检查配置与程序是否匹配，不读取网格，也不运行时间推进。
schema 位于对应维度目录中；旧版被禁用的 `Roe_M2` 配置已迁移为 `Roe`，历史误差与
计时数据需重新计算后才能比较。迁移路径、校验命令及上游差异见
[求解器拆分说明](../../docs/guides/solver_split_zh.md)。

## 5. 黏性、边界、初场与 I/O

历史更名说明：VertexFV → NCFV 的更名本身不改变高效/传统模式或重启数据结构；
后续分方向归一化更新见第 3 节。
新配置及默认输出使用 `data/out/NCFV/`。更名前已完成的计算、重启和视频仍
保留在 `data/out/vertexFV/`，已发布报告的文件名也不变。
历史收敛分析脚本默认继续读取归档结果。重启时可在新配置中将
`io.restartInput` 指向原重启前缀，网格路径则使用 `cases/ncfv_euler/3D/` 下的新位置。
旧源码目录、头文件和程序目标不再作为接口保留。

层流 Navier–Stokes 通量复用 `Euler::Gas` 的守恒量梯度转换、牛顿应力与
傅里叶热传导内核，积分通量为 `F_inviscid - F_viscous`。黏度模型可选
`Constant`、`Sutherland`、`DensityProportional`（此时参数为运动黏度）。
`gasConstant`、`prandtlNumber`、参考温度和 Sutherland 常数必须与输入场
采用同一套量纲。传统模式在面积分点求二次重构的解析梯度；高效模式用
构造顶点插值系数压缩梯度积分，无 Gauss 点。左右重构迹的跳跃只作为黏性
惩罚项加入平均梯度，不代替真实法向导数。

内部黏性面按论文式（4-153）--（4-170）先计算原始变量算术平均
\(\widetilde q=(q_L+q_R)/2\)，再以守恒变量形式构造 dGRP 梯度

\[
\overline{\nabla U}=\frac12
(\overline{\nabla U_L}+\overline{\nabla U_R})
+\frac{\bar U_R-\bar U_L}{2\Delta\widetilde x}\,\boldsymbol n,
\qquad
\Delta\widetilde x=\frac{\min(\Omega_i,\Omega_j)}{S_{ij}},
\]

然后在 \(\widetilde q\) 处用链式法则转换为原始变量梯度。宏面黏性通量乘
标量面积 \(S_{ij}=\sum_k S_{ij,k}\)，而不是合成有向面积向量的模。

物理边界按 CGNS 名称在 `physics.boundaryZones` 中配置。在每个重构位置，令
\(n\) 为外法向，\(v=u_L\cdot n\)，\(a_L\) 为域内声速，\(K=p/\rho^\gamma\)。

| 边界模式 | 施加的对流边界条件 |
|---|---|
| `FarField`、`CharacteristicFarField`、`SupersonicInlet`、`SupersonicOutlet` | 按当地 \(v-a_L,v,v+a_L\) 的符号选取特征数据。出射量及零速量取域内，入射量取分区 `primitive` 或默认 `farFieldPrimitive`。同一分区可同时出现流入、流出及回流。 |
| `PressureOutlet` | 亚声速出流保留域内 \(J_+,K,u_\tau\)，给定 `staticPressure`。亚声速回流保留域内 \(J_+\)，从分区储库取 \(K,u_\tau\)。超声速出流取全部内态；超声速回流取储库的密度和速度及给定静压。 |
| `Extrapolation` | 对所有局部马赫数直接使用域内重构态和梯度。需要纯外推的旧算例将原 `SupersonicOutlet` 改为此模式。 |
| `SlipWall`、`Symmetry` | 用重构内态与零法向速度求壁压 \(p_b\)，通量为 \((0,p_b n,0)\)。 |
| `NoSlipAdiabaticWall`、`NoSlipIsothermalWall` | 对流壁压由域内重构与给定法向壁速确定。壁速、剪切功、绝热或等温约束同时进入逐片黏性通量。 |
| `Periodic` | 周期节点配对后，共享内部界面通量，两侧分别使用自己的重构。 |

静压亚声速出口用 \(p_b=p_{\rm set}\)、\(\rho_b=(p_b/K_b)^{1/\gamma}\)、
\(v_b=J_{+,L}-2a_b/(\gamma-1)\) 恢复边界迹。
壁面按相对法向马赫数 \(M=(v-w_n)/a_L\) 选择一侧波关系。压缩时，
\(p_b/p_L=1+\gamma M(\sqrt{1+z^2}+z)\)，\(z=(\gamma+1)M/4\)；
膨胀时 \(p_b/p_L=[1+(\gamma-1)M/2]^{2\gamma/(\gamma-1)}\)，达到真空时通量为零。
固定网格上的切向移动壁仍给出零对流质量通量。无滑移壁若配置非零法向壁速 \(w_n\)，
求解器取 \(u_b=u_L+(w_n-v)n\)。以 \(P=p_b/p_L\)、\(\beta=(\gamma-1)/(\gamma+1)\)
计，压缩侧 \(\rho_b/\rho_L=(P+\beta)/(1+\beta P)\)，膨胀侧 \(\rho_b/\rho_L=P^{1/\gamma}\)。
固定控制面上的通量为 \((\rho_bw_n,\rho_bw_nu_b+p_bn,(\rho E_b+p_b)w_n)\)，包括法向质量输运。

`strongState=true` 另外约束初始及每个 RK 子步的节点均值。对 `SupersonicInlet`，
仅当本节点所有开启强施加的入口面按当前均值均为严格超声速流入时，才给定整个节点状态。
其余流态由边界通量施加。符合条件的强入口优先于无滑移壁，同优先级取区 ID 最小者。
壁面之间的弱通量分别用各自法向求值。`limitFluxCorrection=true` 仍用对偶均值构造一阶 LLF
边界通量、许可状态界及 CFL 波速；该路径给固定壁使用反射外态以维持正性限制器的通量分解。

传统格式在每个求积点计算边界通量。高效格式默认按支撑节点的积分权重累加通量。
光滑流动算例可设置 `algorithm.macroFaceBoundary=true`，
并关闭 `reconstruction.enableLimiter` 和 `reconstruction.limitFluxCorrection`。
此时同一节点控制体内同分区、同向共面的微面合为宏面。设 \(H\) 为上表确定的向外通量，
则利用对二次多项式精确的几何权重计算
\[
I_B=A_B H_i+\sum_j D_{w_{Bj}} H_j .
\]
沿面微分包括域内重构梯度，随位置变化的入射边界数据还贡献自己的方向导数。
物理宏面的对流项直接由此给出。单独的 `MMSExactExterior` 制造解对照使用内外双态和一次配置的黎曼通量。
三阶面均值估计适用于光滑且特征类型稳定的面片。

当 `physics.viscous.enabled=true` 时，宏面公式给出对流项。每个原始边界片用本片的内侧平均态、
重构梯度和单位外法向计算黏性通量，再按面积从对流项中扣除。
`NoSlipAdiabaticWall` 令法向导热为零，移动壁面仍有剪切应力做功。
`NoSlipIsothermalWall` 使用分区的 `wallTemperature` 构造壁面状态并计算热通量。
具体取内侧原始变量 \(W_L=(\rho_L,u_L,p_L)\)，无滑移壁的黏性状态取
\(W_w=(\rho_L,w,p_L)\)（绝热）或 \(W_w=(\rho_L,w,\rho_L R T_w)\)（等温）。
程序给原始变量梯度加上法向修正 \(n(W_w-W_L)^T/d_w\)，\(d_w\) 取控制体长度尺度的一半。
绝热壁再去除温度梯度的法向分量；等温壁用壁面状态计算输运系数及热流。
滑移壁与对称面返回零黏性通量，开放边界用内侧状态及梯度计算黏性通量。

`time.useCFLTimeStep=true` 时，`dt_i=CFL*V_i/sum(lambda_conv+lambda_visc)`。
`useLocalTimeStep=true` 为稳态局部伪时间，输出时间记为各步最小步长的累加；
非定常物理时间必须设置 `false`，对全域取 MPI 最小步长。`endTime` 控制
终止时刻并截短最后一步；`iterations` 是本次最多推进的额外步数。

初场顺序是统一状态 → `initialField.boxes` → `planes` → `expressions`
→ `nodeFile`；重启输入优先于这些新初场。节点文件为 CSV 或空白分隔文本：

```text
original_node,rho,u,v,w,p
0,1,1,1,0,1
1,1,1,1,0,1
```

节点编号是 `node2nodeOrig` 的分区无关原始编号；二维去掉 w。
`nodeFileIndexBase` 可为 0 或 1。`nodeFileVariables=Conservative` 时字段为
`rho,rhou,rhov,[rhow],rhoE`。输入值解释为节点对偶体均值；一般光滑解析场
不能将点值直接当作高阶均值。ExprTk 程序可使用 `x[3]`、`globalNode`、
`UPrim[dimension+2]`，设置 `inRegion=1` 并返回 0。

VTK 使用原始网格输出分布式 `.pvtu`/`.vtu`，密度、压力、温度、Mach、
能量、速度和局部步长作为节点数据；这些量由对偶均值转换，非逐点精确值。
`outputInterval`/`restartInterval` 控制频率，`writeInitial`/`writeFinal`/
`writeFinalRestart` 控制首末输出。
H5 重启使用原始节点编号重分配，支持更换 MPI 进程数；JSON 重启逐 rank
存储并校验原始节点次序，要求相同分区。`restartInput` 不含 `.dnds.h5` 或
`.dir` 后缀。完整运行配置会写入 `outputPrefix.resolved.json`。

可运行的黏性示例为 `cases/ncfv_euler/2D/NCFV_viscous.json`。

### 湍流闭合

二维、三维均通过 `physics.turbulence.model` 选择模型，同时启用
`physics.viscous.enabled`。在 `[rho,u,v,(w),p]` 后追加对应的原始变量，
同步设置 `solver.fieldNVariables`。

| `model` | 追加原始变量 | 二维 / 三维字段数 |
| --- | --- | --- |
| `none`（默认） | 无 | 4 / 5 |
| `SA` | `nuTilde` | 5 / 6 |
| `KOmegaWilcox` | `k, omega` | 6 / 7 |
| `KOmegaSST` | `k, omega` | 6 / 7 |
| `KOmegaSSTSAS` | `k, omega` | 6 / 7 |
| `RealizableKE` | `k, epsilon` | 6 / 7 |

初始值、远场储库、分区原始变量和节点文件使用同一字段顺序。分区的 `primitive`
为空时继承 `farFieldPrimitive`。两方程的初始值和储库值均须为正。
守恒节点文件的附加变量是 `rhoNuTilde`、`rhoK,rhoOmega`
或 `rhoK,rhoEpsilon`。SA 的 `rhoNuTilde` 在存储时为
\(\rho\tilde\nu/\texttt{viscosityReference}\)，原始变量和 VTK 使用物理 \(\tilde\nu\)。
开启 `useQCR` 可使用二次应力修正。SA 支持 `sa.useFt2`、`saRotationCorrection`
和 `saVersion`。SA 的 `desScale` 大于 100 时为 RANS，启用混合尺度时
`desMode` 的 `0/1/2/3` 分别对应 DDES、IDDES、WMLES、DES97。SST 使用
`desScale` 乘局部单元尺度构造混合长度，默认的大值关闭此修正。

例如，二维 SST 算例采用以下片段，其余参数可沿用二维黏性算例。

```json
{
    "solver": {"type": "ncfv_euler", "discretization": "NCFV", "model": "IdealGas", "fieldNVariables": 6},
    "physics": {
        "viscous": {"enabled": true},
        "turbulence": {"model": "KOmegaSST"},
        "initialPrimitive": [1.0, 0.1, 0.0, 1.0, 0.0001, 1.0],
        "farFieldPrimitive": [1.0, 0.1, 0.0, 1.0, 0.0001, 1.0]
    }
}
```

内部通量对湍流标量使用质量通量迎风，两方程闭合还向动量和能量通量加上
\(2\rho k/3\) 的法向应力与做功。黏性通量使用共享模型的涡黏度、湍流导热和
各方程扩散系数。体源项在两种模式下均从对偶均值重构二次场，在微单纯形求积点
评价后做体积平均。RANS 的显式 CFL 包含湍流扩散及源项耗散率。

选择 `physics.turbulence.model="KOmegaSSTSAS"` 后，SST-1994 的守恒 \(\omega\) 方程增加
Menter–Egorov 的 \(\rho q_{\mathrm{SAS}}\)，\(k\) 耗散采用 RANS 长度。两种积分模式都从
限幅后的守恒量二次重构求速度拉普拉斯。以 \(\boldsymbol m=\rho\boldsymbol u\) 表示动量时，

\[
\nabla^2 u_i = \frac{\nabla^2 m_i-u_i\nabla^2\rho-2\nabla\rho\cdot\nabla u_i}{\rho}.
\]

SAS 的网格下限使用对偶体局部长度 \(\Delta=V_{\mathrm{dual}}^{1/d}\)。体积分点上计算的
SAS \(\omega\) 负导数参与显式 CFL 估计。解析三维尾流时使用三维网格和全局物理时间步
`time.useLocalTimeStep=false`。源项公式见 [SST-SAS 模型说明](../../docs/solver-guide/acm_rans_zh.md#6-sst-sas-尺度自适应模型)。

开放边界按域内法向速度的符号选取湍流对流信息。流入及回流采用储库，流出采用
域内重构并给零法向标量扩散。`Extrapolation` 始终沿用域内重构状态及梯度，周期面走内部
通量。滑移和对称面给零标量法向通量。无滑移壁面令 \(\tilde\nu_w=0\) 或
\(k_w=0\)，并按共享闭合给定 \(\omega_w\) 或 \(\varepsilon_w\)。壁面目标通过
通量施加到对偶体，参考距离取对偶质心至该壁面。壁面节点的湍流均值由输运方程推进。
壁距离搜索仅使用无滑移边界，IDDES 和 WMLES 额外使用局部单元的最大跨度与
壁法向跨度。重启校验湍流模型与 SA 的缩放系数，VTK 按模型输出 `NuTilde`、
`TurbulentKineticEnergy`、`Omega` 或 `Epsilon`。

## 6. 周期等熵涡

`cases/ncfv_euler/3D/NCFV_iv40.json` 使用用户提供的三棱柱网格及论文
式 (3-71)、(3-72)。`prepare_iv40.py` 将旧 CGNS ElementRange 边界转换为
读取器支持的 PointRange/FaceCenter，所有坐标及连接保持原值。

三个周期长度为 `[10,10,4]`，六个边界均配置 `Periodic`。默认先在网格
初始化中合并同一周期节点，得到唯一 owner 的完整对偶控制体；重构模板、
构造点坐标均使用最近周期像。运行期按初始化得到的精确节点依赖做稀疏
owner/ghost 通信，边通量仍按 edge owner 唯一计算。每个 rank 不再保存或
交换全局周期场。支持三维全平移或部分平移周期，要求启用方向有足够
网格层数且对面节点匹配。高效积分可启用限制器；传统积分的周期限幅
尚不支持。周期配对不等价于重构导数严格为零：紧致三维最小二乘在间断处
仍需单独检查方向不变性，详见 RP1 的 `audit_z_invariance.cpp`。

解析涡的初场在高效模式使用精确一阶导数及预计算全微分权重转换成均值；
传统模式对解析守恒量做体数值积分。两者均非直接把节点点值赋给均值。
`β=5`、中心 `(5,5)`、背景 `(rho,u,v,w,p)=(1,1,1,0,1)`，完整周期为 10；
配置推进至论文三维算例的 `t=2`。黏性及限制器关闭，使用全局物理步长。
`solution.diagnostics.csv` 给出恢复密度点值相对解析解的体积加权误差、
总守恒量和熵偏差；逐 rank `.nodes.rankNNNN.csv` 保存原始编号、坐标、
对偶体份额体积、均值、恢复点值和解析点值，便于独立复核。

```bash
cd build
mpirun --oversubscribe -np 4 ./app/ncfv_euler3D.exe ../cases/ncfv_euler/3D/NCFV_iv40.json
```

### 非周期三棱柱制造解

`physics.exponentialMMS=true` 启用三维非周期理想气体 Euler 制造解，取 \(\gamma=1.4\)。
令 \(e=\exp(0.75x+0.35y+0.65z+0.65t)\)，原始变量
\((\rho,u,v,w,p)=(1+0.1e,\ 0.2+0.1e,\ 0.1+0.2e,\ 0.3+0.3e,\ 1.3+1.2e)\)。
制造解给定初始守恒均值和 \(S=\partial_tU+\nabla\cdot F(U)\)。
`FarField` 和 `CharacteristicFarField` 使用域内重构的出射量，制造解只给定入射特征。
在此三维制造解上，可将全局 `physics.boundaryMode` 设为 `MMSExactExterior`，
通过完整解析外态和双态黎曼通量复算历史对照组。
重启读取文件中的守恒均值和物理时刻。
制造解使用全域物理步长。三次 SSPRK3 残差使用 \(t_n,t_n+\Delta t,t_n+\Delta t/2\)，
其中 \(\Delta t\) 为当前尝试实际采用的步长。[原文 §6.1.3](https://arxiv.org/abs/2305.01466)

两种积分都组装 \(d\bar U_i/dt=-\sum F_{\rm out}/V_i+\bar S_i\)。
高效格式用节点源值和源项一阶导数，
\(\bar S_i=S(x_i,t)+\sum_j\nabla S(x_j,t)^T w^V_{ij}\)。
`pointRecoveryStencil` 中的 \(w^V\) 已除以控制体体积。传统格式在体积求积点累加 \(S\)，再除以体积。
`NodeControlVolume::VolumeMean` 同时支持这两种规则，可用于其他给定值和空间梯度的源项。
制造解把空间指数基的控制体均值预先算好，每个子步只更新对应的时间系数。

随附的 \(n=12\) 网格按 gs 验证使用的规则重建，在单位立方体内将每个砖块分成两个 O1 三棱柱，
只对内部节点施加每分量最大 \(0.1/n\) 的扰动，NumPy 随机种子为 \(230501466+n\)。
共有 2,197 个节点、3,456 个棱柱和六个平面边界分区。示例采用 Roe、32 点 SVD 重构目标、
\(\Delta t=0.05/n\)、\(t_{\rm end}=0.5\)。[算例配置](../../cases/ncfv_euler/3D/NCFV_exponential_mms_prism12.json)

在计算节点的 `build` 目录可以使用同一网格比较边界和积分方法。随附算例默认使用特征宏面边界。

```bash
casefile=../cases/ncfv_euler/3D/NCFV_exponential_mms_prism12.json
mpiexec -n 8 ./app/ncfv_euler3D.exe "$casefile"
mpiexec -n 8 ./app/ncfv_euler3D.exe "$casefile" -k /physics/boundaryMode -v MMSExactExterior
mpiexec -n 8 ./app/ncfv_euler3D.exe "$casefile" -k /algorithm/macroFaceBoundary -v false -k /algorithm/mode -v TraditionalQuadrature
```

`retainMicroGeometry=true` 时，程序结束后用每个微四面体上的 24 点六次求积独立计算参考均值，
输出 `rho_mean_L2_Q6` 和 `rho_source_mean_L2_Q6`，分别度量密度解和连续方程源项的体积加权均方根误差。
效率比较可关闭几何保留，随之跳过这次诊断。用 gs 的 \(n=16,20,24\) 网格复验时，
对应设置 `mesh.meshFile`、`time.timeStep=0.05/n`、`time.iterations=10*n`。
相邻两级的观测阶为 \(\log(E_n/E_m)/\log(m/n)\)。

随附的历史 CSV 保留当时的配置名，其中 `FarField` 行来自旧版完整解析外态对照组。
用原始日志重新汇总该表时，给收敛脚本传入
`--summarize-only --mode CharacteristicFarField FarField`。
新一轮计算默认保存 `CharacteristicFarField` 和 `MMSExactExterior` 两组日志。

目标机器也可以单独运行三棱柱 MPI 回归。该用例检查内部均匀场上的源项注入、末步截短时
三个子步的时刻及黎曼调用次数。每次残差的特征边界方案调用次数等于全局内部棱数，
解析外态对照组另有 \(6(n+1)^2\) 次边界调用。

```bash
mpiexec -n 2 ./test/cpp/ncfv_test_parallel --test-case='NCFV prism MMS*'
```

### 三棱柱制造解的特征边界与 P2 面积分

在上述算例中取外法向 \(n\)，记 \(v=u\cdot n\)、\(a=\sqrt{\gamma p/\rho}\)、
\(P_\tau=I-nn^T\)、\(K=p/\rho^\gamma\)。法向 Euler 通量的声学左特征向量给出
\(dv\pm dp/(\rho a)\)。沿等熵线 \(dp/(\rho a)=2\,da/(\gamma-1)\)，对应的声学变量为
\(J_\pm=v\pm2a/(\gamma-1)\)。三维强迫制造解中熵随位置变化，这里用
\((J_+,J_-,K,P_\tau u)\) 构造边界迹。时间演化仍使用完整三维方程及体源项。

在每个面上的重构位置，按内态特征速度 \(v+a,v-a,v\) 的符号决定数据来源。
整个测试期间 \(|v|/a\le 0.925\)，三个低坐标面流入，三个高坐标面流出。

| 特征量 | 亚声速入口 | 亚声速出口 |
|---|---|---|
| \(J_+\) | 域内重构 | 域内重构 |
| \(J_-\) | 给定 \(g_-\) | 给定 \(g_-\) |
| \(K,P_\tau u\) | 给定 \(g_K,g_\tau\) | 域内重构 |

令 \(b=(0.2,0.1,0.3)\)、\(c=(0.1,0.2,0.3)\)、\(r=1+0.1e\)、\(p_g=1.3+1.2e\)，
其中 \(e\) 是前述时空指数。入射函数为

\[
g_- = n\cdot b+(n\cdot c)e - \frac{2}{\gamma-1}\sqrt{\gamma p_g/r},\qquad
g_K=p_g/r^\gamma,\qquad g_\tau=P_\tau b+(P_\tau c)e .
\]

程序按需计算这些函数及其沿面导数。将选出的特征量还原为单一的边界迹 \(U_b\)：

\[
v_b=(J_++J_-)/2,\qquad a_b=(\gamma-1)(J_+-J_-)/4,\qquad
\rho_b=\left(\frac{a_b^2}{\gamma K_b}\right)^{1/(\gamma-1)},\quad
p_b=\rho_b a_b^2/\gamma,\quad u_b=P_\tau u_b+v_bn .
\]

若域内迹取精确值，上式逐点恢复制造解的物理通量。实际计算中，入口和出口的
出射数据均取自数值重构，并在每个 SSPRK3 子步更新。角点按各自面法向分别处理。
令 \(B(U_L,g)\) 表示这一特征组合，\(H=F_n(B(U_L,g))\)。特征宏面向外的通量积分为

\[
I_B=A_B H_i+\sum_j (D_w H)_j,\qquad
D_w H=F'_n(U_b)\left[B_{U_L}(\nabla U_L\,w)+B_g\,D_wg\right],
\quad w=w_{Bj}.
\]

例如 \(D_we=e(0.75,0.35,0.65)\cdot w\)。显式的 \(D_wg\) 项使时空变化的入口数据
也参与 P2 修正。积分只需数值内态梯度和入射函数的方向导数。每个特征边界宏面调用
零次黎曼求解器，内部棱继续用 Roe。

在光滑且远离声速转换的面上，\(B\) 与 \(F_n\) 的导数有界。若重构点值误差为
\(O(h^3)\)、梯度误差为 \(O(h^2)\)，上述链式法则保持这两个阶。形状规则的网格上
\(\sum_j|w_{Bj}|/A_B=O(h)\)，因此 P2 积分后的**面平均通量误差为 \(O(h^3)\)**。
推进后的对偶均值还取决于完整空间残差与时间积分，须另做网格加密验证。

#### 误差度量与复算

在 \(n=12,16,20,24,32,48\) 上分别运行特征边界和完整解析外态对照组。每一级两组共用网格、
体源、内部 Roe 通量和时间步 \(\Delta t=0.05/n\)，推进至 \(t=0.5\)。两组均按六次精确的
微四面体求积取得参考对偶均值。全域和贴边两组误差分别取
\(E_D=(\sum_{i\in D}V_i|\bar U_i-\bar U_i^{\rm ref}|^2/\sum_{i\in D}V_i)^{1/2}\)。
贴边集合只计有物理边界片的控制体，角点只计一次。

独立的边界通量测试在 \(t=0.5\) 将高精度参考对偶均值送入求解器，再由数值重构得到内态及梯度。
用实际组装的残差恢复每个贴边控制体的总边界通量
\(\Phi_i=V_i(\bar S_i-\mathrm{RHS}_i)-\sum_e s_{ie}\Phi_e\)。
对照值 \(\Phi_i^*\) 用该控制体各外侧微三角上的六次求积直接积分解析物理通量。
设 \(A_i\) 为这些三角形的总面积，报告每个分量的
\((\sum_i|\Phi_i-\Phi_i^*|^2/A_i\,/\,\sum_i A_i)^{1/2}\)。
两处参考求积仅用于初始化测试输入和独立计误差。

下表的阶数取六级网格上 \(\log E\) 对 \(\log(1/n)\) 的最小二乘斜率。通量列用上述独立测试；
均值列用实际 SSPRK3 推进结果。最后一列给出原解析外态方案在相同条件下的全域均值阶数。

| 分量 | 特征边界通量 | 特征方案全域均值 | 特征方案贴边均值 | 解析外态全域均值 |
|---|---:|---:|---:|---:|
| 密度 / 质量通量 | 2.937 | 2.844 | 2.738 | 2.742 |
| \(x\) 动量 | 3.032 | 2.991 | 2.954 | 2.878 |
| \(y\) 动量 | 2.948 | 2.890 | 2.835 | 2.873 |
| \(z\) 动量 | 2.991 | 2.899 | 2.828 | 2.922 |
| 总能量 | 2.982 | 2.818 | 2.768 | 2.885 |

从 \(n=12\) 到 \(48\)，质量通量误差由 \(9.93\times10^{-5}\) 降至 \(1.71\times10^{-6}\)。
最细网格上，特征方案与解析外态对照组的全域密度误差分别为 \(5.68\times10^{-7}\) 和
\(5.93\times10^{-7}\)，贴边密度误差分别为 \(1.13\times10^{-6}\) 和 \(6.92\times10^{-7}\)。
特征方案在最后两级上的全域、贴边密度阶分别为 2.893 和 2.828。\(n=24\) 时将时间步减半，
五个守恒量的全域和贴边误差变化均小于 0.30%。边界通量测试支持三阶；
有限网格上推进解的各分量观测阶仍有差异。
[边界通量误差](../../cases/ncfv_euler/3D/diagnostics/prism_mms_boundary_flux.csv) ·
[两种边界的逐级均值误差和观测阶](../../cases/ncfv_euler/3D/diagnostics/prism_mms_characteristic_convergence.csv)

从仓库根目录，用项目的 Python 环境执行以下命令。脚本生成网格并保存每一级的原始日志、
全域和贴边误差以及相邻加密阶。最后一条命令单独验证边界通量。

```bash
d=cases/ncfv_euler/3D/diagnostics
out=/tmp/ncfv-prism-convergence
python "$d/run_prism_mms_convergence.py" --solver build/app/ncfv_euler3D.exe --output "$out" --ranks 8 --sizes 12 16 20 24 32
python "$d/run_prism_mms_convergence.py" --solver build/app/ncfv_euler3D.exe --output "$out" --ranks 16 --sizes 48
python "$d/run_prism_mms_convergence.py" --summarize-only --output "$out" --sizes 12 16 20 24 32 48
python "$d/compile_reconstruction_probe.py" --source "$d/prism_mms_boundary_flux.cpp" --output "$out/flux_probe"
OMP_NUM_THREADS=1 mpiexec -n 4 "$out/flux_probe" cases/ncfv_euler/3D/NCFV_exponential_mms_prism12.json \
    "$out/boundary_flux.csv" "$out"/prism_n{12,16,20,24,32,48}_j0.1.cgns
```

同一程序的 `--wall-outlet` 模式使用独立的光滑场快照。四个侧壁的法向速度为零，
\(x=1\) 出口的静压为 1.35。从一次实际残差中提取各面内部控制体的通量，
与六次面求积比较，棱边、角点不参与此项误差统计。\(n=12,16,24,32\) 四级拟合得到
壁压 3.033 阶，出口质量、三个动量、能量通量分别为 2.893、2.871、2.857、2.866、2.891 阶。
各级壁面对流质量和能量通量的均方根误差都在 \(10^{-16}\) 以内。
[滑移壁和静压出口通量误差](../../cases/ncfv_euler/3D/diagnostics/prism_wall_pressure_outlet_flux.csv)

```bash
OMP_NUM_THREADS=1 mpiexec -n 8 "$out/flux_probe" --wall-outlet cases/ncfv_euler/3D/NCFV_exponential_mms_prism12.json \
    "$out/wall_outlet.csv" "$out"/prism_n{12,16,24,32}_j0.1.cgns
```

## 7. 测试与当前边界

```bash
cmake --build build -t ncfv_unit_tests -j8
ctest --test-dir build -R '^ncfv_' --output-on-failure
```

`ncfv_test_geometry` 检查线、三角形和四面体全微分公式对任意二次多项式
达到机器精度。`ncfv_test_parallel` 在三维 Hex 网格上同时检查坐标平均、
几何闭合、两种积分存储不变量、1/2/4/8-rank 拓扑以及自由流残差；真实
六面成对周期 Hex 网格还验证拓扑合并、精确 halo 裁剪、两种积分路径的一步
SSPRK3 自由流保持。测试也在
完整 MPI 对偶宏面上用含交叉项的任意二次多项式分别验证格点恢复权重、
左右界面平均权重和通量矩阵权重，并验证跨分区限制器确实约束宏面平均值。

I/O 测试还验证非均匀 CSV 初场、一步推进、VTK/H5 输出、状态回读及强
无滑移壁条件。`ncfv_turbulence_np{1,2}` 检查两种积分模式的各湍流闭合、
静压出口回流、等温壁面推进、VTK 和 H5 重启。整套黏性及 RANS 格式的收敛阶
仍需用网格加密与制造解测定。
