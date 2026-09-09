# 第1章　绪论

## 1.1　研究背景

不可压缩流动广泛存在于水动力学、低速空气动力学、能源动力装备、生物流动以及复杂工业装置中。其控制方程看似只是将密度视为常数并施加速度散度为零的约束，但这一约束改变了方程组的数学结构：压力不再由独立的状态方程给出，也没有通常意义上的演化方程，而是作为拉格朗日乘子维持速度场的无散性。以常密度牛顿流体为例，控制方程可写为

| $$\nabla\cdot\boldsymbol{u}=0,$$ | (1-1) |
|:---:|---:|

| $$\frac{\partial\boldsymbol{u}}{\partial t}+\nabla\cdot(\boldsymbol{u}\otimes\boldsymbol{u})+\frac{1}{\rho_0}\nabla p-\nabla\cdot\!\left[\nu\left(\nabla\boldsymbol{u}+\nabla\boldsymbol{u}^{\mathsf T}\right)\right]=\boldsymbol{f}.$$ | (1-2) |
|:---:|---:|

式中，$\boldsymbol{u}$ 为速度，$p$ 为压力，$\rho_0$ 为常密度，$\nu$ 为运动黏度，$\boldsymbol{f}$ 为体积力。早期的标记与单元法已经通过压力迭代满足质量守恒^[\[1\]](#ref-1)^；Chorin 随后提出人工可压缩方法，把椭圆型压力约束嵌入可推进的伪时间双曲系统^[\[2\]](#ref-2)^。此后，围绕“如何以稳定、准确且可扩展的方式施加不可压缩约束”，形成了投影/分步法、压力修正法、整体耦合法、人工可压缩法以及介观或粒子法等多条技术路线。

不可压缩计算的困难并不止于求解一个压力泊松方程。对于非结构网格、高雷诺数流动、复杂边界和大规模并行计算，数值方法还需同时处理以下矛盾：第一，质量守恒与压力—速度耦合必须足够严格；第二，空间离散既要在光滑区保持高阶精度，又要在强剪切、回流或网格畸变处保持稳定；第三，稳态收敛效率、非定常时间精度与内存开销之间存在竞争；第四，算法应适应分区网格和局部通信，避免全局椭圆求解成为扩展性瓶颈；第五，在变密度不可压缩流动中，密度输运、动量守恒和压力约束必须保持一致，而不能简单套用常密度公式。

因此，评价某一不可压缩算法不能只比较单步成本或残差下降速度，还应同时考察离散质量守恒、压力唯一性、时空收敛阶、网格适应性、线性/非线性求解代价、并行通信模式以及对复杂物理模型的可扩展性。本章以这些维度为主线，梳理主要不可压缩流动计算方法，重点讨论人工可压缩（artificial compressibility method，ACM）及其预处理技术，并结合 DNDSR 中常密度与变密度 ACM 的研究内容界定可形成的博士论文问题。

## 1.2　不可压缩流动计算方法的主要谱系

### 1.2.1　投影法与分步法

投影法利用 Helmholtz–Hodge 分解，把一个未必无散的预测速度投影到无散空间。Chorin 的分步思想^[\[3\]](#ref-3)^和 Temam 的数学分析^[\[4\]](#ref-4)^奠定了该类方法的基础。典型离散过程先忽略新时刻压力得到中间速度 $\boldsymbol{u}^{\star}$，再由压力修正恢复不可压缩约束：

| $$\frac{\boldsymbol{u}^{\star}-\boldsymbol{u}^{n}}{\Delta t}=\mathcal{N}(\boldsymbol{u}^{n}),$$ | (1-3) |
|:---:|---:|

| $$\nabla^2\phi=\frac{\rho_0}{\Delta t}\nabla\cdot\boldsymbol{u}^{\star},$$ | (1-4) |
|:---:|---:|

| $$\boldsymbol{u}^{n+1}=\boldsymbol{u}^{\star}-\frac{\Delta t}{\rho_0}\nabla\phi,\qquad p^{n+1}=p^{n}+\phi.$$ | (1-5) |
|:---:|---:|

Bell、Colella 和 Glaz 构造了二阶精度的不可压缩 Navier–Stokes 投影格式^[\[5\]](#ref-5)^。后续研究表明，压力增量形式、旋转形式、边界条件以及离散梯度和散度是否相容，都会影响分裂误差、压力边界层和长期质量守恒；Guermond、Minev 和 Shen 对这些变体作了系统总结^[\[6\]](#ref-6)^。

投影法的优点是结构清晰，速度推进与压力约束可分别使用成熟算法，且时间精度理论较完善。其主要代价是每个物理时间步通常都需求解全局椭圆问题。在几何多尺度、强各向异性网格或大规模分布式环境中，压力泊松方程的预条件、粗网格构造和全局同步可能成为主要瓶颈。此外，若速度与压力空间、边界条件或非正交修正不相容，形式上的投影并不自动保证离散层面的严格无散。

### 1.2.2　压力修正与分离式有限体积法

压力修正方法在工程有限体积软件中应用广泛。Patankar 和 Spalding 提出的 SIMPLE 方法通过猜测压力、求解动量方程、建立压力修正方程并反复校正速度与压力来实现稳态耦合^[\[7\]](#ref-7)^。SIMPLEC 通过调整速度修正近似提高收敛性^[\[8\]](#ref-8)^；PISO 则在一个时间步内执行多次校正，更适合瞬态计算^[\[9\]](#ref-9)^。在同位网格上，Rhie–Chow 型动量插值通过在面通量中引入压力—速度耦合，抑制奇偶解耦和棋盘格压力^[\[10\]](#ref-10)^。

这类方法的突出优势是模块化强：动量、压力、能量、组分和湍流方程可以分离求解，便于处理工业多物理模型，并可复用标量输运线性求解器。局限在于收敛速度依赖松弛参数和压力修正近似；分离误差在强耦合、强旋转、强非正交或大密度比问题中可能加剧。对非定常问题，还需区分外层物理时间误差与单个时间步内压力—速度迭代未收敛造成的代数误差。

### 1.2.3　整体耦合与混合有限元方法

整体耦合方法把速度和压力放入同一鞍点系统中同时求解。典型混合有限元以高阶速度和低阶压力构造离散鞍点系统，Taylor 与 Hood 的经典单元是代表性方案^[\[11\]](#ref-11)^。该路线具有清晰的变分基础；具体稳定性取决于单元配对、维数和网格条件，采用稳定化技术后也可使用等阶插值。其核心计算困难是块鞍点系统规模大、条件数高，通常需要基于压力 Schur 补的块预条件器。整体耦合可避免速度—压力外层分离，但单次迭代的内存、通信和预条件代价取决于离散与求解器实现。

### 1.2.4　涡量—流函数、格子玻尔兹曼与粒子方法

在二维单连通区域中，涡量—流函数形式可自动满足连续性方程并消去压力，但三维推广、复杂边界和压力恢复较困难。格子玻尔兹曼方法从离散速度分布函数出发，碰撞—迁移过程局部、并行度高，适合复杂孔隙或多相界面；其不可压缩极限、边界处理、低马赫数误差和网格灵活性仍需专门控制^[\[12\]](#ref-12)^。光滑粒子流体动力学适合自由液面与大变形问题；不可压缩 SPH 可通过投影求解压力泊松方程，但粒子一致性、边界精度和压力噪声仍是关键问题^[\[13\]](#ref-13)^。近期，Lallemand 等从动理学来源、数值分析、边界条件和网格细化角度系统总结了近不可压流动 LBM 的理论与进展^[\[14\]](#ref-14)^；Zhang 等的综述则表明，SPH 的研究已扩展到 Riemann 型粒子通量、高阶重构、多分辨率离散、流固耦合和高效邻域搜索^[\[15\]](#ref-15)^。

上述方法并非简单替代关系。其压力约束方式、全局通信模式和适用场景的差异可概括为表1-1。

表1-1　主要不可压缩流动数值方法比较

| 方法类别 | 压力与连续性处理 | 主要优势 | 主要局限与典型适用场景 |
|:---|:---|:---|:---|
| 投影/分步法 | 每个物理步求压力泊松方程并投影速度 | 时间离散理论较完整；无散约束直观 | 全局椭圆求解；边界与分裂误差；常用于非定常流动 |
| SIMPLE/SIMPLEC/PISO | 由离散动量方程构造压力修正并迭代 | 工业多物理扩展成熟；模块化 | 松弛与校正策略敏感；强耦合时外迭代较多 |
| 整体耦合有限元/有限体积 | 同时求解速度—压力鞍点系统 | 避免速度—压力外层分离；便于变分分析 | 块系统规模大；需高质量预条件器 |
| 人工可压缩法 | 引入伪时间压力导数，将约束双曲化 | 可采用迎风通量、局部时间步和显式/隐式推进；并行局部性好 | 经典 ACM 需收敛伪时间；参数与边界敏感；时间精确非定常通常需双时间 |
| 格子玻尔兹曼法 | 由分布函数低马赫极限恢复宏观方程 | 碰撞—迁移局部；并行效率高 | 低马赫限制、边界和网格灵活性问题 |
| 粒子/涡量方法 | 粒子投影、弱可压缩关系或涡量约束 | 自由界面、大变形或特定二维问题有优势 | 一致性、压力噪声、三维边界和计算量问题 |

注：表中比较是依据投影法综述^[\[6\]](#ref-6)^、压力修正方法^[\[7\]](#ref-7)^、整体耦合有限元^[\[11\]](#ref-11)^、格子玻尔兹曼与粒子方法代表性工作^[\[12\]](#ref-12)^^[\[13\]](#ref-13)^及近期综述^[\[14\]](#ref-14)^^[\[15\]](#ref-15)^所作的典型算法层面定性归纳；具体精度、内存、通信和运行时间仍取决于离散、求解器、网格、收敛准则与硬件平台。

## 1.3　人工可压缩与预处理方法的发展

### 1.3.1　经典人工可压缩思想

Chorin 的核心思想是在连续性方程中加入仅用于迭代的压力伪时间导数：

| $$\frac{1}{\beta^2}\frac{\partial p}{\partial\tau}+\nabla\cdot\boldsymbol{u}=0,$$ | (1-6) |
|:---:|---:|

其中，$\tau$ 为伪时间，$\beta^2$ 为人工压力尺度；以 $\rho_0$ 为密度基准时，相应的基本人工速度尺度为 $c_a=\sqrt{\beta^2/\rho_0}$。当伪时间迭代收敛时，$\partial p/\partial\tau\rightarrow0$，原不可压缩约束得以恢复^[\[2\]](#ref-2)^。这一变换使压力扰动能够以有限的人工波速传播，方程组具有类似可压缩流 Euler 方程的双曲特征结构，从而可以借用迎风差分、近似 Riemann 求解器、局部时间步和多级时间推进等成熟技术。

人工可压缩并不是把物理流体改成真正的可压缩流体。$c_a$ 不是真实声速，伪时间过程也不代表可观测的物理瞬态；只有伪时间稳态满足原不可压缩方程。$\beta^2$ 太小会使压力信息传播缓慢，太大又会拉大特征速度尺度、收紧显式稳定条件或恶化迭代刚性。因此，人工可压缩方法从一开始就隐含了一个预处理问题：如何调整伪时间质量矩阵，使对流模态与人工声学模态的尺度更匹配。

### 1.3.2　方程级预处理与特征速度整形

Turkel 将预处理系统化为对时间导数左乘状态相关矩阵，通过改变迭代方程的特征结构而不改变稳态解^[\[16\]](#ref-16)^。Choi 和 Merkle 将预处理用于黏性流动^[\[17\]](#ref-17)^，Weiss 和 Smith 又给出了适用于常密度与变密度流动的统一构造^[\[18\]](#ref-18)^。Turkel 的综述进一步指出，预处理的本质是缩小不同波族之间的尺度差，而参数选择、边界条件和离散耗散必须与预处理后的特征系统一致^[\[19\]](#ref-19)^。

以本研究采用的常密度状态 $\boldsymbol{Q}=(u,v,w,p)^{\mathsf T}$ 为例，Scheme-A 型伪时间预处理可写为^[\[20\]](#ref-20)^

| $$\boldsymbol{\Gamma}(\boldsymbol{Q})\frac{\partial\boldsymbol{Q}}{\partial\tau}+\boldsymbol{M}\frac{\partial\boldsymbol{Q}}{\partial t}+\boldsymbol{\mathcal{R}}(\boldsymbol{Q})=\boldsymbol{0},\qquad \boldsymbol{M}=\operatorname{diag}(1,1,1,0),$$ | (1-7) |
|:---:|---:|

其中 $\boldsymbol{\mathcal{R}}$ 为空间残差。在笛卡尔坐标下，预处理矩阵为

| $$\boldsymbol{\Gamma}=\begin{bmatrix}1&0&0&(1+\alpha)u/\beta^2\\0&1&0&(1+\alpha)v/\beta^2\\0&0&1&(1+\alpha)w/\beta^2\\0&0&0&1/\beta^2\end{bmatrix}.$$ | (1-8) |
|:---:|---:|

沿单位法向 $\boldsymbol{n}$，令 $q_n=\boldsymbol{u}\cdot\boldsymbol{n}$，预处理通量算子 $\boldsymbol{\Gamma}^{-1}\boldsymbol{A}_n$ 的两支人工声学特征值为

| $$\lambda_{\pm}=\frac{(1-\alpha)q_n\pm\sqrt{(1-\alpha)^2q_n^2+4\beta^2/\rho_0}}{2},\qquad \lambda_{t,1}=\lambda_{t,2}=q_n.$$ | (1-9) |
|:---:|---:|

式（1-9）直接揭示了参数作用：$c_a$ 决定人工压力波的基本速度尺度，$\alpha$ 调整对流速度在声学根中心位置中的权重。参数不能脱离网格尺度、局部速度、黏性谱半径和边界条件单独优化；对于显式推进，应兼顾允许的伪时间步，对于隐式推进，则应兼顾非线性残差、Jacobian 近似和线性系统条件数。

需要特别区分两种常被混称为“预处理”的操作。式（1-7）中的 $\boldsymbol{\Gamma}$ 属于方程级或伪时间预处理，它改变迭代系统的特征传播；块 Jacobi、LU-SGS 或 Krylov 方法中的预条件器则属于代数线性求解器预处理，用于近似逆转离散 Jacobian。二者可以同时存在，但优化目标和正确性条件不同。若论文不作此区分，容易把物理/伪时间建模与线性代数加速混为一谈。

### 1.3.3　非定常计算与双时间推进

经典 ACM 直接适用于稳态问题。为获得时间精确的非定常解，需要保留物理时间导数，并在每个物理时间层内把伪时间缺陷迭代到足够小。Rogers 和 Kwak 将迎风离散用于时间精确不可压缩 Navier–Stokes 计算^[\[21\]](#ref-21)^，随后给出了稳态与非稳态统一的三维算法^[\[22\]](#ref-22)^。若物理时间采用二阶后向差分，则第 $n+1$ 层的伪时间方程可表示为

| $$\boldsymbol{\Gamma}\frac{\partial\boldsymbol{Q}^{\,n+1}}{\partial\tau}+\boldsymbol{M}\frac{3\boldsymbol{Q}^{\,n+1}-4\boldsymbol{Q}^{\,n}+\boldsymbol{Q}^{\,n-1}}{2\Delta t}+\boldsymbol{\mathcal{R}}\!\left(\boldsymbol{Q}^{\,n+1}\right)=\boldsymbol{0}.$$ | (1-10) |
|:---:|---:|

由于 $\boldsymbol{M}$ 的压力对角元为零，压力没有物理时间历史；它只在每一物理步内作为约束变量随伪时间调整。双时间方法的物理时间精度不仅取决于外层 BDF 阶次，还取决于每步内层缺陷是否收敛到与截断误差相匹配的水平。若内层过早停止，迭代误差会污染外层时间精度，故应报告内层容限、最大迭代数和未收敛时间步比例，而不能只报告名义上的 BDF2。

### 1.3.4　非结构网格、高阶离散与现代扩展

ACM 具有可按面构造双曲数值通量的特点，因而容易与非结构有限体积法结合。Malan、Lewis 和 Nithiarasu 给出了面向非结构网格、稳态与瞬态黏性不可压缩流动的改进人工可压缩有限体积框架^[\[23\]](#ref-23)^。其后研究逐步从结构网格低阶差分扩展到非结构网格、紧致高阶离散、通量重构和复杂边界，并持续探索伪时间收敛加速与稳健性改进。

另一条研究线不再要求伪时间完全收敛，而是构造具有耗散或物理解释的压力演化方程。Clausen 提出的熵阻尼人工可压缩形式通过压力扩散改善显式计算的稳定性^[\[24\]](#ref-24)^；Toutant 的广义压力方程从连续性和动量关系出发讨论非定常黏性不可压缩流动中的压力演化^[\[25\]](#ref-25)^。这类方法与经典稳态 ACM 在误差解释上并不相同：前者把有限参数下的压力方程视为实际时间模型的一部分，后者则要求伪时间残差趋零后恢复原方程。

近年的研究重点包括高阶通量重构框架中不同 ACM 形式的稳定性与效率比较^[\[26\]](#ref-26)^、变密度流动的高阶人工可压缩时间推进^[\[27\]](#ref-27)^、封闭运动域低马赫数可压缩方程中人工声速与动态压力的显式处理^[\[28\]](#ref-28)^，以及覆盖不可压、理想气体和真实气体全速度范围的守恒压力基有限体积框架^[\[29\]](#ref-29)^。Kim 和 Lee 从预处理矩阵层面统一了 ACM 与局部低马赫数预处理，使同一求解器能够在不可压流动和含激波可压缩流动之间切换^[\[30\]](#ref-30)^。Chen 等把 EDAC 与 Runge–Kutta 间断 Galerkin 方法及自适应网格结合，面向显式、紧致和高阶非结构离散^[\[31\]](#ref-31)^；Cappanera 和 Giordano 则从变密度不可压缩 Navier–Stokes 方程出发研究人工可压缩方法^[\[32\]](#ref-32)^。Jung 和 Giometto 进一步重新审视广义压力演化方程，指出人为降低声速时熵产生闭合不宜被解释为严格的物理模型，并提出含可调压力阻尼项的广义伪可压缩形式^[\[33\]](#ref-33)^。

截至 2026 年 9 月 9 日的上述文献显示，ACM 的研究重心已从“能否消去压力泊松方程”转向有限参数下的误差含义、跨马赫数一致性、压力扰动控制、变密度推广以及高阶离散的效率。对于 DNDSR，还应进一步回答有限伪时间容差是否破坏时间精度、预处理矩阵在极限状态能否保持可对角化、边界特征是否与内部通量一致，以及高阶离散与隐式线性化能否共同产生可验证的效率收益。这些是由当前实现边界引出的待验证研究问题，而不是上述文献已经证明的结论。

## 1.4　面向 DNDSR 的 ACM 研究内容与方法

### 1.4.1　常密度非结构有限体积框架

DNDSR 常密度 ACM 以 $\boldsymbol{Q}=(u,v,w,p)^{\mathsf T}$ 为单元状态，在面局部正交基下构造对流—压力通量，并在受支持的混合非结构单元上形成半离散残差^[\[20\]](#ref-20)^：

| $$\boldsymbol{\mathcal{R}}_i(\boldsymbol{Q})=\frac{1}{|\Omega_i|}\left[\sum_{f\subset\partial\Omega_i}\sum_g w_g\!\left(\widehat{\boldsymbol{F}}_{n,f,g}-\widehat{\boldsymbol{F}}^{\,v}_{n,f,g}\right)|\Gamma_f|-\int_{\Omega_i}\boldsymbol{S}\,\mathrm{d}\Omega\right].$$ | (1-11) |
|:---:|---:|

空间重构包含一阶常值、Green–Gauss 二阶梯度以及紧致有限体积（compact finite volume，CFV）变分重构。Wang 等建立的非结构网格变分重构方法通过局部耦合获得高阶多项式^[\[34\]](#ref-34)^；多维 WBAP 类限制思想可在候选多项式之间进行非线性加权，以提高复杂流动中的稳健性^[\[35\]](#ref-35)^。在本研究框架中，高阶基函数采用零均值构造，限制高阶系数时保持单元平均值不变；面通量使用高斯积分，黏性项采用带中心连线修正的面梯度，以兼顾非正交网格上的一致性。

当前实现可选 Rusanov 或 Roe 型通量^[\[20\]](#ref-20)^。Roe 方法以满足守恒跳跃关系的平均 Jacobian 分解波族^[\[36\]](#ref-36)^。ACM 中的波并非真实声波，因此耗散矩阵必须基于预处理算子 $\boldsymbol{\Gamma}^{-1}\boldsymbol{A}_n$，而不能直接照搬可压缩 Euler 特征系统。代码还提供 Harten 型熵修正^[\[20\]](#ref-20)^；显式推进采用 SSPRK3，其多级 TVD 构造可追溯至 Shu 和 Osher^[\[37\]](#ref-37)^。

值得重点研究的是特征值碰撞。当

| $$\alpha\rho_0q_n^2=\beta^2$$ | (1-12) |
|:---:|---:|

时，一支人工声学根与重复对流根合并，预处理算子可能出现 Jordan 块。若仍直接求逆病态特征向量矩阵，数值耗散会随接近碰撞条件而失去连续性；若简单退化为统一标量耗散，又会过度抹平不同波族。DNDSR 采用合流 Hermite 多项式计算矩阵函数 $f(\boldsymbol{\Gamma}^{-1}\boldsymbol{A}_n)$，在重复根处保留导数信息，从而使 Roe 耗散连续通过碰撞极限^[\[20\]](#ref-20)^。这一处理已有代数单元测试覆盖，但其谱稳定性、误差常数和对实际高阶流场收敛的影响仍需系统数值实验。

### 1.4.2　伪时间推进、隐式线性化与代数预处理

DNDSR 的稳态求解当前可选显式 SSPRK3 局部伪时间推进或后向 Euler 线性化^[\[20\]](#ref-20)^。隐式残差系统的一般形式为

| $$\left[\frac{1}{\Delta\tau_i}\frac{\partial\!\left(\boldsymbol{\Gamma}\Delta\boldsymbol{Q}\right)}{\partial\Delta\boldsymbol{Q}}+\frac{\partial\boldsymbol{\mathcal{R}}}{\partial\boldsymbol{Q}}\right]\delta\boldsymbol{Q}=-\boldsymbol{\mathcal{D}},$$ | (1-13) |
|:---:|---:|

其中 $\boldsymbol{\mathcal{D}}$ 为当前非线性缺陷。线性系统可采用块 Jacobi、LU-SGS 或左预处理 GMRES^[\[20\]](#ref-20)^。这里的块 Jacobi/LU-SGS 是代数预条件器，与式（1-8）的方程级预处理矩阵具有不同层次。分布式 LU-SGS 对本进程单元按局部顺序扫掠，而跨进程邻居使用滞后的 ghost 状态，因此更准确的表述是“分布式块 SGS 预条件器/平滑器”，而不是全局串行精确 SGS。

当前面通量 Jacobian 以一阶面状态构造，并冻结高阶重构、限制器和涡黏性，而非对完整高阶残差进行一致 Newton 线性化^[\[20\]](#ref-20)^。因此，该策略本质上是“高阶非线性残差 + 低阶近似 Jacobian”的 defect-correction。它降低了装配和内存成本，但其收敛速度会受高阶修正与冻结 Jacobian 不一致的影响。博士论文可围绕 Jacobian 保真度、线性迭代次数、总残差评估次数和并行通信量建立统一成本模型，避免仅用单次迭代残差下降判断隐式方法优劣。

### 1.4.3　守恒型变密度人工可压缩推广

对无密度扩散的低速变密度流动，守恒变量选为

| $$\boldsymbol{U}=(\rho,\rho u,\rho v,\rho w,p)^{\mathsf T},\qquad \boldsymbol{M}_{v}=\operatorname{diag}(1,1,1,1,0).$$ | (1-14) |
|:---:|---:|

控制方程包括密度输运、动量守恒和速度无散约束：

| $$\frac{\partial\rho}{\partial t}+\nabla\cdot(\rho\boldsymbol{u})=0,$$ | (1-15) |
|:---:|---:|

| $$\frac{\partial(\rho\boldsymbol{u})}{\partial t}+\nabla\cdot(\rho\boldsymbol{u}\otimes\boldsymbol{u}+p\boldsymbol{I}-\boldsymbol{\tau})=\rho\boldsymbol{f},$$ | (1-16) |
|:---:|---:|

| $$\nabla\cdot\boldsymbol{u}=0.$$ | (1-17) |
|:---:|---:|

变密度版本采用平方根密度 Roe 平均，以保持包括质量与散度通量在内的割线性质；Rusanov 谱界同时检查左态、Roe 平均态和右态，避免平均态低估低密度端的人工声速。其预处理声学根中速度尺度变为 $\sqrt{\beta^2/\rho}$，对流根 $q_n$ 为三重根，分别对应密度接触模态和两支剪切模态。与常密度情形相同，特征碰撞处使用合流矩阵函数而不是病态特征矩阵求逆^[\[20\]](#ref-20)^。

这一模型应严格限定为“无密度扩散的低速不可压缩变密度输运”，不能外推为低马赫可压缩多组分模型：当前方程中没有能量方程、状态方程、温度/组分扩散、表面张力和完整浮力闭合。已有变密度人工可压缩文献也表明，不同守恒变量、压力项位置和预处理矩阵会导致不同的常密度极限与特征结构，因此推广时必须逐项验证守恒跳跃、正性和极限一致性，而不能只增加一条密度方程。

### 1.4.4　湍流、并行实现与现有证据边界

常密度和变密度模块均提供层流、Spalart–Allmaras、Wilcox $k$–$\omega$、SST 及 realizable $k$–$\varepsilon$ 选项。湍流变量未并入 ACM 的 $4\times4$ 或 $5\times5$ Roe 系统，而是分离推进；流场残差评估时冻结涡黏性，随后更新湍流输运变量。该设计保持了流场 Riemann 求解器的紧致结构，但不是流动—湍流全耦合 Newton 法。对于变密度模型，湍流历史使用 $\rho\phi$ 的守恒形式并复用流场质量通量，从而减少密度输运与湍流输运之间的不一致^[\[20\]](#ref-20)^。

网格与数据链支持 CGNS 读入、METIS 分区、主 ghost 层构造、平移周期边界、几何升阶与二分，残差通过 MPI 分布式装配^[\[20\]](#ref-20)^。近年来，非结构紧致有限体积方法仍在向三阶及更高阶、复杂流动和计算效率方向发展^[\[38\]](#ref-38)^；面心有限体积等新型混合/有限体积框架则通过避免梯度重构并以面变量构造全局系统来求解不可压缩流动^[\[39\]](#ref-39)^。DNDSR 的 CFV–ACM 组合与这些方向具有直接对话关系，但必须用相同网格、相同收敛准则和总工作量开展对照，才能判断其实际优势。

当前回归测试已经覆盖通量 Jacobian、预处理矩阵互逆、一般 $\alpha$ 特征系统、Roe/Rusanov 一致性、Jordan 碰撞极限、非正交线性黏性梯度、变密度 Roe 割线性质、BDF 压力历史屏蔽及 MPI 装配^[\[20\]](#ref-20)^。这些结果只在已覆盖样本和给定容差内支持所列代数关系，不能替代制造解收敛阶、标准实验对比、湍流模型认证或并行强缩放。若某一圆柱算例未达到内迭代容限，则不应把该结果作为定量验证。

## 1.5　现状评述与拟解决的关键问题

综合上述方法谱系，可以得到三点判断。第一，压力泊松方程并非不可压缩计算的唯一正确入口；投影、压力修正、整体耦合和 ACM 分别把约束代价放在全局椭圆求解、外层校正、鞍点预条件或伪时间收敛上。方法优劣取决于问题规模、时间精度要求、网格拓扑和并行平台。第二，ACM 的主要价值不是“免去一次泊松求解”这一孤立事实，而是把约束传播改写成可与迎风通量、高阶重构和局部隐式推进统一设计的系统；代价则是引入参数、伪时间容差和特征边界处理。第三，现代 ACM 研究已经进入“方程、离散和求解器协同设计”阶段，单独优化 $\beta^2$（或 $c_a$）、Roe 耗散或某个线性迭代器很难保证端到端效率。

基于现有实现，后续博士研究宜围绕以下相互关联的问题展开。

1. **预处理参数的无量纲化与自适应。** 建立 $\alpha$、$\beta^2$（或 $c_a$）与局部对流、黏性尺度、网格尺度和伪时间 CFL 的关系，考察稳态谱聚集、显式稳定域和隐式条件数，形成跨网格和跨雷诺数可迁移的参数策略。

2. **特征碰撞下的连续矩阵函数理论。** 从 Jordan/Hermite 极限出发，证明耗散矩阵在碰撞前后的一致性与有界性，并通过可控的一维波包、二维涡与非结构网格算例量化相对于标量回退和直接特征分解的误差与鲁棒性。

3. **高阶空间残差与隐式 Jacobian 的一致性。** 比较冻结重构 Jacobian、颜色有限差分、矩阵自由 Jacobian–向量积及局部高阶块近似，统一统计非线性迭代、Krylov 迭代、通信和残差评估成本。

4. **双时间误差与停止准则。** 推导内层缺陷对 BDF1/BDF2 物理时间误差的传播关系，设计与目标截断误差匹配的自适应内层容限，并对未收敛物理步实施拒绝或时间步回退。

5. **压力零空间与边界一致性。** 对封闭域显式施加压力规范条件，构造与预处理特征一致的入口、出口、壁面和周期边界，并检查压力平移是否影响基于特征量的光滑指标。

6. **常密度—变密度统一验证。** 验证 $\rho\rightarrow\rho_0$ 时通量、特征系统和时间离散的连续退化；建立静止密度接触、变密度涡、Rayleigh–Taylor 类问题以及复杂非结构网格上的质量、动量和无散误差预算。

7. **与主流不可压缩算法的公平比较。** 选择投影法或压力修正法作为外部基准，在相同网格、离散阶次、物理容差和硬件条件下比较总时间、内存、全局归约次数、强/弱缩放和解质量。该比较应避免把成熟软件的完整算法与尚未调优的单一模块作不对等结论。

8. **湍流与工程可信度。** 完成平板边界层、零压梯度/不利压梯度边界层、方柱/圆柱绕流等分级验证，并披露网格、壁面分辨率、统计窗口、守恒误差和不确定度。只有在标准基准通过后，才可把研究实现表述为经过认证的工程求解能力。

此外，当前实现尚存在需要在论文中透明披露的边界：高阶隐式 Jacobian 冻结重构和限制器；跨 MPI 分区的 LU-SGS 使用滞后耦合；常密度 BDF2 仅覆盖层流路径；缺少 restart 与 BDF 历史恢复；旋转周期和系统化压力规范尚未形成完整用户路径。这些限制不是应被隐藏的“缺点列表”，而是界定研究问题、设计验证矩阵和解释结果适用范围的必要条件。

## 1.6　本章小结

不可压缩流动的数值本质是速度演化与压力约束的耦合。投影法和压力修正法通过椭圆压力方程施加约束，整体耦合法直接求解鞍点系统，格子玻尔兹曼与粒子方法则从不同离散层次重建宏观不可压缩行为。人工可压缩法通过伪时间压力演化把约束传播双曲化，并可借助方程级预处理调节人工声学与对流特征尺度；其稳态正确性依赖伪时间收敛，非定常正确性依赖双时间内层缺陷控制。

DNDSR 的研究基础由 Scheme-A 预处理、Roe/Rusanov 通量、特征碰撞的 Hermite/Jordan 连续处理、非结构 CFV 高阶重构、显式与隐式伪时间推进、BDF 双时间以及守恒型变密度推广构成。该基础已经具备代数和 MPI 回归测试，但尚需以收敛阶、基准实验、公平算法比较和并行可扩展性建立完整证据链。因而，后续工作的核心不应是继续罗列算法部件，而应围绕“预处理参数—高阶离散—隐式求解—约束误差”之间的耦合关系形成可验证的理论与数值结论。

\newpage

# 参考文献

::: {#ref-1}
[1] HARLOW F H, WELCH J E. Numerical calculation of time-dependent viscous incompressible flow of fluid with free surface[J]. The Physics of Fluids, 1965, 8(12): 2182-2189. DOI: [10.1063/1.1761178](https://doi.org/10.1063/1.1761178).
:::

::: {#ref-2}
[2] CHORIN A J. A numerical method for solving incompressible viscous flow problems[J]. Journal of Computational Physics, 1967, 2(1): 12-26. DOI: [10.1016/0021-9991(67)90037-X](https://doi.org/10.1016/0021-9991(67)90037-X).
:::

::: {#ref-3}
[3] CHORIN A J. Numerical solution of the Navier-Stokes equations[J]. Mathematics of Computation, 1968, 22(104): 745-762. DOI: [10.1090/S0025-5718-1968-0242392-2](https://doi.org/10.1090/S0025-5718-1968-0242392-2).
:::

::: {#ref-4}
[4] TEMAM R. Sur l’approximation de la solution des équations de Navier-Stokes par la méthode des pas fractionnaires (II)[J]. Archive for Rational Mechanics and Analysis, 1969, 33(5): 377-385. DOI: [10.1007/BF00247696](https://doi.org/10.1007/BF00247696).
:::

::: {#ref-5}
[5] BELL J B, COLELLA P, GLAZ H M. A second-order projection method for the incompressible Navier-Stokes equations[J]. Journal of Computational Physics, 1989, 85(2): 257-283. DOI: [10.1016/0021-9991(89)90151-4](https://doi.org/10.1016/0021-9991(89)90151-4).
:::

::: {#ref-6}
[6] GUERMOND J L, MINEV P, SHEN J. An overview of projection methods for incompressible flows[J]. Computer Methods in Applied Mechanics and Engineering, 2006, 195(44-47): 6011-6045. DOI: [10.1016/j.cma.2005.10.010](https://doi.org/10.1016/j.cma.2005.10.010).
:::

::: {#ref-7}
[7] PATANKAR S V, SPALDING D B. A calculation procedure for heat, mass and momentum transfer in three-dimensional parabolic flows[J]. International Journal of Heat and Mass Transfer, 1972, 15(10): 1787-1806. DOI: [10.1016/0017-9310(72)90054-3](https://doi.org/10.1016/0017-9310(72)90054-3).
:::

::: {#ref-8}
[8] VAN DOORMAAL J P, RAITHBY G D. Enhancements of the SIMPLE method for predicting incompressible fluid flows[J]. Numerical Heat Transfer, 1984, 7(2): 147-163. DOI: [10.1080/01495728408961817](https://doi.org/10.1080/01495728408961817).
:::

::: {#ref-9}
[9] ISSA R I. Solution of the implicitly discretised fluid flow equations by operator-splitting[J]. Journal of Computational Physics, 1986, 62(1): 40-65. DOI: [10.1016/0021-9991(86)90099-9](https://doi.org/10.1016/0021-9991(86)90099-9).
:::

::: {#ref-10}
[10] RHIE C M, CHOW W L. Numerical study of the turbulent flow past an airfoil with trailing edge separation[J]. AIAA Journal, 1983, 21(11): 1525-1532. DOI: [10.2514/3.8284](https://doi.org/10.2514/3.8284).
:::

::: {#ref-11}
[11] TAYLOR C, HOOD P. A numerical solution of the Navier-Stokes equations using the finite element technique[J]. Computers & Fluids, 1973, 1(1): 73-100. DOI: [10.1016/0045-7930(73)90027-3](https://doi.org/10.1016/0045-7930(73)90027-3).
:::

::: {#ref-12}
[12] CHEN S, DOOLEN G D. Lattice Boltzmann method for fluid flows[J]. Annual Review of Fluid Mechanics, 1998, 30: 329-364. DOI: [10.1146/annurev.fluid.30.1.329](https://doi.org/10.1146/annurev.fluid.30.1.329).
:::

::: {#ref-13}
[13] CUMMINS S J, RUDMAN M. An SPH projection method[J]. Journal of Computational Physics, 1999, 152(2): 584-607. DOI: [10.1006/jcph.1999.6246](https://doi.org/10.1006/jcph.1999.6246).
:::

::: {#ref-14}
[14] LALLEMAND P, LUO L S, KRAFCZYK M, et al. The lattice Boltzmann method for nearly incompressible flows[J]. Journal of Computational Physics, 2021, 431: 109713. DOI: [10.1016/j.jcp.2020.109713](https://doi.org/10.1016/j.jcp.2020.109713).
:::

::: {#ref-15}
[15] ZHANG C, ZHU Y J, WU D, et al. Smoothed particle hydrodynamics: Methodology development and recent achievement[J]. Journal of Hydrodynamics, 2022, 34(5): 767-805. DOI: [10.1007/s42241-022-0052-1](https://doi.org/10.1007/s42241-022-0052-1).
:::

::: {#ref-16}
[16] TURKEL E. Preconditioned methods for solving the incompressible and low speed compressible equations[J]. Journal of Computational Physics, 1987, 72(2): 277-298. DOI: [10.1016/0021-9991(87)90084-2](https://doi.org/10.1016/0021-9991(87)90084-2).
:::

::: {#ref-17}
[17] CHOI Y H, MERKLE C L. The application of preconditioning in viscous flows[J]. Journal of Computational Physics, 1993, 105(2): 207-223. DOI: [10.1006/jcph.1993.1069](https://doi.org/10.1006/jcph.1993.1069).
:::

::: {#ref-18}
[18] WEISS J M, SMITH W A. Preconditioning applied to variable and constant density flows[J]. AIAA Journal, 1995, 33(11): 2050-2057. DOI: [10.2514/3.12946](https://doi.org/10.2514/3.12946).
:::

::: {#ref-19}
[19] TURKEL E. Preconditioning techniques in computational fluid dynamics[J]. Annual Review of Fluid Mechanics, 1999, 31: 385-416. DOI: [10.1146/annurev.fluid.31.1.385](https://doi.org/10.1146/annurev.fluid.31.1.385).
:::

::: {#ref-20}
[20] MA R. DNDSR ACM and ACMVariable modules: high-order artificial-compressibility solvers[CP/OL]. Version eec8d03. (2026-09-08)[2026-09-09]. [https://github.com/Kirabo167/DNDSR/tree/eec8d0311e9df5f7924b2e60f70589200366deec](https://github.com/Kirabo167/DNDSR/tree/eec8d0311e9df5f7924b2e60f70589200366deec).
:::

::: {#ref-21}
[21] ROGERS S E, KWAK D. Upwind differencing scheme for the time-accurate incompressible Navier-Stokes equations[J]. AIAA Journal, 1990, 28(2): 253-262. DOI: [10.2514/3.10382](https://doi.org/10.2514/3.10382).
:::

::: {#ref-22}
[22] ROGERS S E, KWAK D, KIRIS C. Steady and unsteady solutions of the incompressible Navier-Stokes equations[J]. AIAA Journal, 1991, 29(4): 603-610. DOI: [10.2514/3.10627](https://doi.org/10.2514/3.10627).
:::

::: {#ref-23}
[23] MALAN A G, LEWIS R W, NITHIARASU P. An improved unsteady, unstructured, artificial compressibility, finite volume scheme for viscous incompressible flows: Part I. Theory and implementation[J]. International Journal for Numerical Methods in Engineering, 2002, 54(5): 695-714. DOI: [10.1002/nme.447](https://doi.org/10.1002/nme.447).
:::

::: {#ref-24}
[24] CLAUSEN J R. Entropically damped form of artificial compressibility for explicit simulation of incompressible flow[J]. Physical Review E, 2013, 87(1): 013309. DOI: [10.1103/PhysRevE.87.013309](https://doi.org/10.1103/PhysRevE.87.013309).
:::

::: {#ref-25}
[25] TOUTANT A. Numerical simulations of unsteady viscous incompressible flows using general pressure equation[J]. Journal of Computational Physics, 2018, 374: 822-842. DOI: [10.1016/j.jcp.2018.07.058](https://doi.org/10.1016/j.jcp.2018.07.058).
:::

::: {#ref-26}
[26] TROJAK W, VADLAMANI N R, TYACKE J, et al. Artificial compressibility approaches in flux reconstruction for incompressible viscous flow simulations[J]. Computers & Fluids, 2022, 247: 105634. DOI: [10.1016/j.compfluid.2022.105634](https://doi.org/10.1016/j.compfluid.2022.105634).
:::

::: {#ref-27}
[27] LUNDGREN L, NAZAROV M. A high-order artificial compressibility method based on Taylor series time-stepping for variable density flow[J]. Journal of Computational and Applied Mathematics, 2023, 421: 114846. DOI: [10.1016/j.cam.2022.114846](https://doi.org/10.1016/j.cam.2022.114846).
:::

::: {#ref-28}
[28] BECCANTINI A, CORRE C, GOUNAND S, et al. An artificial compressibility approach to solve low Mach number flows in closed domains[J]. Computers & Fluids, 2024, 280: 106364. DOI: [10.1016/j.compfluid.2024.106364](https://doi.org/10.1016/j.compfluid.2024.106364).
:::

::: {#ref-29}
[29] DENNER F, EVRARD F, VAN WACHEM B G M. Conservative finite-volume framework and pressure-based algorithm for flows of incompressible, ideal-gas and real-gas fluids at all speeds[J]. Journal of Computational Physics, 2020, 409: 109348. DOI: [10.1016/j.jcp.2020.109348](https://doi.org/10.1016/j.jcp.2020.109348).
:::

::: {#ref-30}
[30] KIM M, LEE S. Unified approach to artificial compressibility and local low Mach number preconditioning[J]. Journal of Computational Physics, 2024, 508: 112998. DOI: [10.1016/j.jcp.2024.112998](https://doi.org/10.1016/j.jcp.2024.112998).
:::

::: {#ref-31}
[31] CHEN L W, LIU Y L, TIAN Z L, et al. Discontinuous Galerkin method for incompressible viscous flow based on the entropically damped artificial compressibility[J]. Computers & Fluids, 2025, 299: 106684. DOI: [10.1016/j.compfluid.2025.106684](https://doi.org/10.1016/j.compfluid.2025.106684).
:::

::: {#ref-32}
[32] CAPPANERA L, GIORDANO S. Artificial compressibility method for the incompressible Navier-Stokes equations with variable density[J]. Numerical Methods for Partial Differential Equations, 2025, 41(5): e70033. DOI: [10.1002/num.70033](https://doi.org/10.1002/num.70033).
:::

::: {#ref-33}
[33] JUNG J, GIOMETTO M. A general formulation for pseudo-compressibility methods for the simulation of fluid flow[J]. Computers & Fluids, 2026, 317: 107174. DOI: [10.1016/j.compfluid.2026.107174](https://doi.org/10.1016/j.compfluid.2026.107174).
:::

::: {#ref-34}
[34] WANG Q, REN Y X, PAN J, et al. Compact high order finite volume method on unstructured grids III: Variational reconstruction[J]. Journal of Computational Physics, 2017, 337: 1-26. DOI: [10.1016/j.jcp.2017.02.031](https://doi.org/10.1016/j.jcp.2017.02.031).
:::

::: {#ref-35}
[35] LI W, REN Y X. The multi-dimensional limiters for solving hyperbolic conservation laws on unstructured grids II: Extension to high order finite volume schemes[J]. Journal of Computational Physics, 2012, 231(11): 4053-4077. DOI: [10.1016/j.jcp.2012.01.029](https://doi.org/10.1016/j.jcp.2012.01.029).
:::

::: {#ref-36}
[36] ROE P L. Approximate Riemann solvers, parameter vectors, and difference schemes[J]. Journal of Computational Physics, 1981, 43(2): 357-372. DOI: [10.1016/0021-9991(81)90128-5](https://doi.org/10.1016/0021-9991(81)90128-5).
:::

::: {#ref-37}
[37] SHU C W, OSHER S. Efficient implementation of essentially non-oscillatory shock-capturing schemes[J]. Journal of Computational Physics, 1988, 77(2): 439-471. DOI: [10.1016/0021-9991(88)90177-5](https://doi.org/10.1016/0021-9991(88)90177-5).
:::

::: {#ref-38}
[38] ZHANG J, LI Z, XIAO J, et al. A third-order compact finite volume scheme on unstructured grid for fluid flows[J]. Computers & Fluids, 2024, 277: 106284. DOI: [10.1016/j.compfluid.2024.106284](https://doi.org/10.1016/j.compfluid.2024.106284).
:::

::: {#ref-39}
[39] VIEIRA L M, GIACOMINI M, SEVILLA R, et al. A face-centred finite volume method for laminar and turbulent incompressible flows[J]. Computers & Fluids, 2024, 279: 106339. DOI: [10.1016/j.compfluid.2024.106339](https://doi.org/10.1016/j.compfluid.2024.106339).
:::
