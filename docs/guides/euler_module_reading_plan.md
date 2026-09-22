# Euler 模块自底向上阅读计划

本文给出一条从 Euler 模块内部依赖最少的数值内核出发，逐层读到空间离散、隐式求解和应用入口的路线。目标不是按文件名依次浏览，而是建立四条能够相互对应的主线：

1. 状态量和模型在代码中如何表示；
2. 一个面的数值通量如何产生；
3. 一个单元的空间残差如何累积；
4. 时间推进器如何反复调用空间算子并更新解。

本文依据当前仓库源码编写。当前初始化实现文件名是
[EulerSolver_Init.hxx](../../src/Euler/EulerSolver_Init.hxx)，不是
<code>EulerSolver_Init.hpp</code>。

## 1. 阅读范围与主线选择

“从不依赖其余模块的地方开始”在这里应理解为“先读 Euler 内部的叶子模块”。这些文件仍会依赖 DNDS、Eigen、Geom 或 CFV 的基础类型，不可能完全脱离仓库其他部分。

Euler 内部最适合先读的文件是：

- [Euler.hpp](../../src/Euler/Euler.hpp)：模型枚举、编译期 traits 和并行状态数组；
- [Gas.hpp](../../src/Euler/Gas.hpp)：热力学、无黏/黏性通量和 Riemann 求解器，是第一遍最值得精读的数值内核；
- [CLDriver.hpp](../../src/Euler/CLDriver.hpp)：依赖少，但属于升力目标控制旁支；
- [ChemicalSource.hpp](../../src/Euler/Chemistry/ChemicalSource.hpp)：接口本身隔离良好，但只在反应流主线中重要。

第一遍建议采用以下范围：

- 用 <code>NS_2D</code> 建立四变量二维 Euler/Navier–Stokes 的心智模型；
- 暂时关闭 RANS、化学反应、旋转坐标系、轴对称、CL driver 和多重网格；
- 先跟踪一条 Roe 通量路径，不比较所有 Riemann 求解器变体；
- 用现有的 evaluator pipeline 测试验证调用顺序。该测试实际使用普通 <code>NS</code>，它是“二维网格、三维速度/状态”，与 <code>NS_2D</code> 不完全相同。

核心主线预计需要约 30–40 小时。RANS、反应流和完整 I/O 可在主线完成后分别追加。

## 2. Euler 内部依赖图

~~~mermaid
flowchart TB
    Base["DNDS / CFV / Geom 基础接口"]
    E["Euler.hpp<br/>模型、Traits、并行数组"]
    G["Gas.hpp<br/>热力学与通量核"]
    CL["CLDriver.hpp<br/>升力控制旁支"]
    Chem["ChemicalSource.hpp/.cpp<br/>化学反应旁支"]
    RANS["RANS_ke.hpp<br/>RANS 数学核"]
    Set["EulerEvaluatorSettings.hpp<br/>统一配置"]
    Phys["PhysicsProperties.hpp<br/>物性统一接口"]
    BC["EulerBC.hpp<br/>边界配置与记录器"]
    Jac["EulerJacobian.hpp<br/>隐式矩阵容器"]
    Src["SourceTermContributor.hpp<br/>源项组合"]
    Eval["EulerEvaluator.hpp + *.hxx<br/>空间离散"]
    Solver["EulerSolver.hpp + *.hxx<br/>时间推进与数据所有权"]
    App["SingleBlockApp.hpp + app/Euler<br/>程序入口"]

    Base --> E
    Base --> G
    E --> RANS
    E --> Jac
    E --> Set
    G --> Set
    CL -.可选.-> Set
    Set --> Phys
    Chem -.反应流.-> Phys
    Set --> BC
    RANS --> Src
    Phys --> Src
    G --> Eval
    BC --> Eval
    Jac --> Eval
    Phys --> Eval
    Src --> Eval
    Base --> Eval
    Eval --> Solver
    Solver --> App
~~~

从下往上阅读时，可以把代码分成五层：

1. **类型和局部数学核**：<code>Euler.hpp</code>、<code>Gas.hpp</code>；
2. **模型适配**：settings、physics、BC、Jacobian、source contributor；
3. **空间离散**：<code>EulerEvaluator</code>；
4. **时间推进和所有权**：<code>EulerSolver</code>；
5. **应用与编译连接**：<code>SingleBlockApp</code>、app 入口、显式实例化。

## 3. 开始前只需掌握的外部基础

不要先完整阅读 DNDS、Geom 和 CFV。遇到下列概念时，定点补课即可。

| 概念 | 最小理解 | 建议入口 |
|---|---|---|
| <code>real</code>、<code>index</code>、<code>rowsize</code>、<code>ssp&lt;T&gt;</code> | 浮点、全局索引、行宽和共享指针别名 | [DNDS/Defines.hpp](../../src/DNDS/Defines.hpp#L110) |
| father / son / transformer | 本地拥有数据、ghost 数据及二者的 MPI 交换 | [DNDS/ArrayPair.hpp](../../src/DNDS/ArrayPair.hpp#L162) |
| <code>tUDof</code>、<code>tURec</code> | 单元自由度和重构系数使用的 Eigen 矩阵类型 | [CFV/VRDefines.hpp](../../src/CFV/VRDefines.hpp#L79) |
| mesh connectivity | cell、face、node 之间的拓扑和边界标识 | [Geom/Mesh/Mesh.hpp](../../src/Geom/Mesh/Mesh.hpp#L37) |
| variational reconstruction | 从单元均值建立高阶多项式重构 | [CFV/VariationalReconstruction.hpp](../../src/CFV/VariationalReconstruction.hpp#L115) |

阅读并行数组时先记住：

- father 部分由当前 MPI rank 拥有；
- son 部分保存邻居 rank 的 ghost 值；
- transformer 负责 pull/push；
- 算法在访问邻居单元前，必须确保相应 ghost 已经更新。

第一遍不需要深入数组序列化、HDF5、CUDA 或网格分区实现。

## 4. 模型、状态量和核心数据结构

### 4.1 模型 traits

先读 [Euler.hpp 的模型与 traits](../../src/Euler/Euler.hpp#L874-L1166)。

| 模型 | 几何维数 <code>gDim</code> | 物理速度维数 <code>dim</code> | 状态量概念 |
|---|---:|---:|---|
| <code>NS_2D</code> | 2 | 2 | <code>[rho, rho*u, rho*v, rho*E]</code> |
| <code>NS</code> | 2 | 3 | <code>[rho, rho*u, rho*v, rho*w, rho*E]</code> |
| <code>NS_3D</code> | 3 | 3 | 三维五变量 |
| <code>NS_SA</code> | 2 | 3 | 普通 NS 状态再加 SA 变量 |
| <code>NS_2EQ</code> | 2 | 3 | 普通 NS 状态再加两个 RANS 变量 |
| <code>NS_EX</code> | 2 | 动态 | 可扩展组分/能量状态 |
| <code>NS_EX_3D</code> | 3 | 动态 | 三维可扩展状态 |

容易误解的一点是：普通 <code>NS</code> 不是严格的四变量二维模型，而是二维网格上的三速度分量模型。代码中的 <code>I4 = dim + 1</code> 是能量分量索引的历史命名，在二维四变量路径中并不一定等于字面上的整数 4。

读完本节后，应能独立回答：

1. 当前模板参数在编译期决定了什么？
2. <code>nVars</code>、<code>gDim</code>、<code>dim</code> 分别控制哪一类数组或循环？
3. <code>if constexpr</code> 的模型分支为什么不会产生运行时开销？

### 4.2 Euler 数组包装

接着读 [Euler.hpp 的数组类型](../../src/Euler/Euler.hpp#L56-L858)，重点是
[ArrayDOFV](../../src/Euler/Euler.hpp#L92)、
[ArrayRECV](../../src/Euler/Euler.hpp#L400) 和
[ArrayGRADV](../../src/Euler/Euler.hpp#L642)。

| 名称 | 典型形状 | 数学含义 |
|---|---|---|
| <code>u</code> / <code>ArrayDOFV</code> | <code>nVars × 1</code> | 单元守恒量均值 |
| <code>uRec</code> / <code>ArrayRECV</code> | <code>nRecBasis × nVars</code> | 单元内高阶重构系数 |
| <code>uGrad</code> / <code>ArrayGRADV</code> | <code>gDim × nVars</code> | 状态量空间梯度 |
| <code>rhs</code> | 与 <code>u</code> 相同 | 半离散方程右端 |
| <code>dTau</code> | 每单元一个标量 | 局部伪时间步 |
| <code>betaPP</code> | 每单元/变量的系数 | 重构限制或正性保持压缩系数 |
| <code>alphaPP</code> | 每单元一个系数 | 解增量正性保持系数 |
| <code>JD</code> | 标量或块对角 | 隐式算子的单元对角块 |
| <code>JSource</code> | 单元块矩阵 | 源项 Jacobian |

第一遍只阅读构造、索引、算术、范数、dot 和 MPI reduction。不要把
[Euler.hpp 中的 JacobianValue](../../src/Euler/Euler.hpp#L780)
当成真正的隐式矩阵类型；它目前是带 TODO 的占位结构。

本阶段笔记应产出一张“变量—数组形状—father/son 状态”的表。

## 5. 分阶段阅读计划

### 阶段 0：建立最小基础词汇（1–2 小时）

**阅读**

- [DNDS/Defines.hpp](../../src/DNDS/Defines.hpp#L110)
- [DNDS/ArrayPair.hpp](../../src/DNDS/ArrayPair.hpp#L162)
- [CFV/VRDefines.hpp](../../src/CFV/VRDefines.hpp#L79)

**只解决**

- 核心标量和索引类型是什么；
- Eigen 矩阵在代码中按“行/列”表示什么；
- father、son 和 ghost exchange 是什么；
- 一个 cell array 如何通过 <code>(*array)[iCell]</code> 取得本地条目。

**验收**

看到一个数组参数时，能判断它是“当前 rank 拥有的数据”还是还包含 ghost，并知道在何处寻找 transformer。

### 阶段 1：模型和数据容器（2–3 小时）

**阅读顺序**

1. [EulerModel 与 EulerModelTraits](../../src/Euler/Euler.hpp#L874-L1166)
2. [ArrayDOFV](../../src/Euler/Euler.hpp#L92)
3. [ArrayRECV](../../src/Euler/Euler.hpp#L400)
4. [ArrayGRADV](../../src/Euler/Euler.hpp#L642)

**重点问题**

- 为什么同一个 evaluator 模板能够覆盖二维、三维、SA、两方程和反应流？
- 哪些尺寸编译期固定，哪些在 EX 模型中运行时决定？
- 解、重构和梯度为什么使用不同矩阵形状？

**阶段产出**

画出 <code>NS_2D</code> 和 <code>NS</code> 两条状态布局，标明密度、动量、能量及附加变量索引。

### 阶段 2：单点热力学和一个面通量（4–5 小时）

这是第一遍的数值核心。按下面顺序读 [Gas.hpp](../../src/Euler/Gas.hpp)：

1. [IdealGasThermal](../../src/Euler/Gas.hpp#L196)
2. [守恒量/原始量转换](../../src/Euler/Gas.hpp#L310)
3. [物理无黏通量](../../src/Euler/Gas.hpp#L369-L430)
4. [RoePreamble](../../src/Euler/Gas.hpp#L240)
5. [Roe 求解器](../../src/Euler/Gas.hpp#L1142)
6. [运行时 Riemann 分派](../../src/Euler/Gas.hpp#L1711)
7. [黏性通量](../../src/Euler/Gas.hpp#L1897)
8. [守恒量梯度转原始量梯度](../../src/Euler/Gas.hpp#L1992)
9. [压力正性压缩系数](../../src/Euler/Gas.hpp#L2086)

第一遍先跳过 batch 版本、所有 Roe M1–M9 修正和不同求解器的逐项比较。

**验收**

给定左/右守恒状态和面法向，能够手写：

~~~text
UL, UR
  -> 原始量、压力、声速
  -> Roe 平均量
  -> 对流数值通量
  -> 可选黏性通量
  -> 面谱半径
~~~

然后用以下测试对应代码中的公式：

- [test_GasThermo.cpp](../../test/cpp/Euler/test_GasThermo.cpp#L56)
- [test_RiemannSolvers.cpp](../../test/cpp/Euler/test_RiemannSolvers.cpp#L156)

### 阶段 3：配置如何变成统一物理接口（3–4 小时）

**阅读顺序**

1. [StateValue](../../src/Euler/EulerEvaluatorSettings.hpp#L154)
2. [EulerEvaluatorSettings](../../src/Euler/EulerEvaluatorSettings.hpp#L344)
3. [PhysicsProperties](../../src/Euler/Physics/PhysicsProperties.hpp#L50)

<code>EulerEvaluatorSettings</code> 第一遍按分组浏览即可：

- 数值通量、Jacobian、重构和 RANS：约 352 行起；
- 化学 reactor step：约 438 行起；
- 旋转坐标系：约 465 行起；
- 区域初始化器：约 533 行起；
- 理想气体物性：约 620 行起；
- 反应流设置：约 690 行起。

不要逐个背诵 <code>DNDS_FIELD</code>。先回答“哪个配置组影响哪个算法分支”。

<code>PhysicsProperties</code> 是定比热理想气体和 Cantera 多组分气体之间的统一门面。优先读：

- 尺度换算：约 140–204 行；
- EOS 和热力状态：约 220 行起；
- <code>conservativeThermal</code>：约 282 行；
- 原始量/守恒量/总静态状态转换：约 392–1023 行；
- 输运和扩散：约 1050–1169 行；
- 化学推进接口：约 1170–1267 行。

**验收**

从一个 JSON 中的气体参数或边界状态出发，能指出它经过 settings 后如何被 evaluator 以统一物性 API 使用。

### 阶段 4：边界、Jacobian 和源项适配（2–3 小时）

**阅读**

- [EulerBCType 与 BoundaryHandler](../../src/Euler/EulerBC.hpp#L36)
- [JacobianDiagBlock](../../src/Euler/EulerJacobian.hpp#L81)
- [SourceTermContributor](../../src/Euler/SourceTermContributor.hpp#L43)

在 <code>EulerBC.hpp</code> 中先读：

- 边界类型枚举；
- [BoundaryHandler](../../src/Euler/EulerBC.hpp#L236) 的 JSON 映射；
- 边界积分和输出记录器只浏览类名，稍后再回读。

在 <code>EulerJacobian.hpp</code> 中理解：

- 稳健块求逆；
- scalar diagonal 与 block diagonal 两种模式；
- local LU/LDLT 是单元局部操作，不等同于全局稀疏直接解。

在 <code>SourceTermContributor.hpp</code> 中只需先认识：

- [SourceCellAux](../../src/Euler/SourceTermContributor.hpp#L43)；
- 体力、旋转、轴对称等基础源项核；
- variant/visitor 如何把不同源项包装成统一调用。

固定变量模型仍有不少 <code>if constexpr</code> 直接路径；不要误以为所有源项都只经过 variant。

### 阶段 5：先从接口和测试建立 evaluator 地图（2 小时）

**阅读**

1. [EulerEvaluator 类](../../src/Euler/EulerEvaluator.hpp#L81)
2. [EvaluateDt 声明](../../src/Euler/EulerEvaluator.hpp#L418)
3. [EvaluateRHS 声明](../../src/Euler/EulerEvaluator.hpp#L461)
4. [evaluator pipeline 测试](../../test/cpp/Euler/test_EulerEvaluator.cpp#L146)

先确认 evaluator 的角色：它是“计算器”，持有 mesh、VFV、边界、物性和工作缓冲的引用或运行态，但顶层解数组和时间推进生命周期主要由 solver 管理。

重点成员：

- mesh / VFV / boundary handler；
- 面谱半径与梯度缓冲；
- wall distance；
- settings / physics；
- source contributors；
- 边界积分与输出记录器。

pipeline 测试是一份很好的“可执行说明书”，其主序列是：

~~~text
solver 配置和网格初始化
  -> InitializeUDOF
  -> EvaluateDt
  -> EvaluateRHS
  -> RHS norm
  -> Jacobian 初始化
  -> forward/backward sweep
  -> increment norm
~~~

**验收**

不看实现，仅根据声明和测试，写出 evaluator 的主要输入、输出和可修改缓冲。

### 阶段 6：跟踪一个面的 RHS 累积（4–5 小时）

先读 [EvaluateRHS 实现](../../src/Euler/EulerEvaluator_EvaluateRHS.hxx#L56)，按以下顺序分块：

1. 56–155：清零、flags 和可选直接梯度；
2. 186–225：面循环骨架；
3. 522–595：调用 <code>fluxFace</code>、积分及写回两侧单元；
4. 703–749：单元源项；
5. 225–517：最后补读左右状态重构、周期变换、边界 ghost 和黏性梯度。

随后跳到 [fluxFace](../../src/Euler/EulerEvaluator_EvaluateDt.hxx#L1097) 和
[generateBoundaryValue](../../src/Euler/EulerEvaluator_EvaluateDt.hxx#L2069)。

建议选一个内部面，逐变量做一张跟踪表：

| 步骤 | 左侧 | 右侧 | 输出 |
|---|---|---|---|
| 单元均值 | <code>u[iCellL]</code> | <code>u[iCellR]</code> | 两个中心状态 |
| 高阶重构 | <code>uRec[L]</code> | <code>uRec[R]</code> | 积分点 <code>UL/UR</code> |
| 边界处理 | 内部面不改 | 内部面不改 | ghost 状态仅边界面产生 |
| 通量 | <code>UL</code> | <code>UR</code> | <code>Flux</code>、谱半径 |
| 面积分 | 同一通量 | 同一通量 | 面通量积分 |
| 单元累积 | 减或加 | 反号 | 守恒的两个 cell RHS |

空间残差主线应能压缩为：

~~~text
u
  -> ghost pull
  -> reconstruction / limiter
  -> 面积分点 UL、UR 和梯度
  -> generateBoundaryValue（仅边界面）
  -> fluxFace
  -> 面积分
  -> 以相反符号除以 cell volume 累积到左右 cell
  -> cell source / JSource
  -> rhs
~~~

**验收**

1. 能解释内部面为什么对两侧单元严格反号；
2. 能指出边界虚状态实际在哪里生成；
3. 能区别数值通量、物理通量、黏性通量和谱半径；
4. 能指出 source 为什么在面循环之后按 cell 添加。

### 阶段 7：CFL 和局部时间步（1–2 小时）

阅读 [EvaluateDt 实现](../../src/Euler/EulerEvaluator_EvaluateDt.hxx#L847-L1060)：

- 面对流谱半径；
- 黏性谱半径；
- 面贡献汇总到单元；
- CFL、体积和最大时间步的组合。

概念上可写成：

\[
\Delta \tau_i =
\min\left(
\frac{\mathrm{CFL}\,V_i\,s_i}
{\sum_{f\in\partial i}\lambda_f A_f},
\Delta t_{\max}
\right),
\]

其中实现会根据配置加入黏性、网格尺度、局部修正和时间步平滑。

注意：<code>EulerEvaluator_EvaluateDt.hxx</code> 的文件名具有误导性。它还包含壁距、面通量、源项、边界状态和输出字段实现，不只包含时间步计算。

### 阶段 8：重构、限制器与正性保持（3–4 小时）

在主线通量已经理解后，再读 [EulerEvaluator.hxx](../../src/Euler/EulerEvaluator.hxx) 的：

- [梯度和重构限制器](../../src/Euler/EulerEvaluator.hxx#L2161)；
- [beta 压缩](../../src/Euler/EulerEvaluator.hxx#L2303)；
- [正性保持 alpha](../../src/Euler/EulerEvaluator.hxx#L2756)；
- 时间步平滑：约 3004 行。

同时回看 solver 中调用这些函数的顺序。需要区分两类系数：

- <code>betaPP</code>：作用于空间重构，避免积分点状态产生负密度/负压力或强振荡；
- <code>alphaPP</code>：作用于时间更新增量，确保 <code>u + alpha*uInc</code> 可接受。

**验收**

写出“单元均值 → 未限制重构 → beta 压缩 → 面状态 → RHS → 解增量 → alpha 压缩 → 新解”的时序。

### 阶段 9：隐式算子和线性求解（3–4 小时）

先读 [EulerEvaluator.hxx 的隐式部分](../../src/Euler/EulerEvaluator.hxx#L39-L949)：

- <code>LUSGSMatrixInit</code>；
- 隐式矩阵向量乘；
- forward/backward sweep；
- SGS、带重构耦合 SGS 和局部 LU。

再读 [EulerSolver::solveLinear](../../src/Euler/EulerSolver.hxx#L1532) 和其预条件器分派。

数据流是：

~~~text
rhs + JSource + dTau
  -> 隐式矩阵初始化
  -> SGS / LU-SGS / FGMRES
  -> uInc
  -> alphaPP
  -> u <- u + alphaPP * uInc
~~~

**验收**

能区分：

- evaluator 提供的矩阵/扫掠操作；
- solver 选择的线性算法；
- ODE 方法要求的非线性/线性回调；
- 局部块 Jacobian 与全局耦合之间的关系。

### 阶段 10：顶层数据所有权和初始化（3–4 小时）

先读 [EulerSolver 类](../../src/Euler/EulerSolver.hpp#L76)，关注：

- MPI、mesh、VFV 和 evaluator；
- <code>u</code>、ODE 增量、时间平均、重构、Jacobian、PP 数组；
- boundary handler 和输出数组；
- [Configuration](../../src/Euler/EulerSolver.hpp#L165) 的分组标题；
- [RunningEnvironment](../../src/Euler/EulerSolver.hpp#L1469)。

然后读 [ReadMeshAndInitialize](../../src/Euler/EulerSolver_Init.hxx#L38)：

~~~text
读取/准备 mesh
  -> ghost、升阶、二分和几何修正
  -> wall distance / boundary mesh
  -> 建立 VariationalReconstruction
  -> 分配解、重构和临时数组
  -> 构造 EulerEvaluator
  -> 配置 Jacobian 和输出 transformer
~~~

重要：该函数完成对象和数组的构造，但并不真正初始化 <code>u</code> 或读取 restart。实际的
<code>InitializeUDOF</code> / restart 分支位于
[RunImplicitEuler 开头](../../src/Euler/EulerSolver.hxx#L104-L127)。

**验收**

画出 Solver 拥有的对象树，并标注哪些对象由 shared pointer 持有、哪些只在 evaluator 中引用。

### 阶段 11：完整时间推进编排（4–5 小时）

阅读 [RunImplicitEuler](../../src/Euler/EulerSolver.hxx#L59-L1506)。不要从第一行逐句读到底，按回调分块：

1. RHS 外层回调：ghost、重构、限制器、正性保持、<code>EvaluateRHS</code>；
2. ODE 所见的 RHS 包装；
3. <code>fdtau</code>：调用 <code>EvaluateDt</code>；
4. <code>fincrement</code>：安全更新；
5. <code>fsolve</code>：构造隐式矩阵并求解；
6. 内迭代停止回调；
7. 物理时间步完成回调；
8. 外层时间循环和 <code>ode-&gt;Step</code> / <code>StepPP</code>。

再读：

- [functor_fstop](../../src/Euler/EulerSolver_Init.hxx#L553)：内迭代收敛、日志、内部输出、restart 和 CFL ramp；
- [functor_fmainloop](../../src/Euler/EulerSolver_Init.hxx#L835)：一个物理步完成后的统计、时间平均、输出和结束判断；
- [InitializeRunningEnvironment](../../src/Euler/EulerSolver.hxx#L1801)：如何按配置选择具体 ODE 与缓冲。

完整调用链应能够复述为：

~~~text
app/Euler/euler*.cpp
  -> RunSingleBlockConsoleApp
  -> EulerSolver::ConfigureFromJson
  -> EulerSolver::ReadMeshAndInitialize
     -> EulerEvaluator::InitializeFV
     -> 构造 EulerEvaluator
  -> EulerSolver::RunImplicitEuler
     -> InitializeUDOF / ReadRestart
     -> InitializeRunningEnvironment
     -> EvaluateDt
     -> ODE::Step
        -> reconstruction / limiter
        -> EvaluateRHS
           -> generateBoundaryValue
           -> fluxFace
              -> Gas::RiemannSolver
              -> Gas::ViscousFlux
           -> source
        -> solveLinear
     -> functor_fstop
     -> functor_fmainloop
  -> PrintData / PrintRestart
~~~

### 阶段 12：应用入口和显式实例化（1–2 小时）

阅读：

- [SingleBlockApp.hpp](../../src/Euler/SingleBlockApp.hpp#L79)
- [app/Euler](../../app/Euler)
- [Euler/CMakeLists.txt](../../src/Euler/CMakeLists.txt#L20)
- [_explicit_instantiation](../../src/Euler/_explicit_instantiation)

<code>RunSingleBlockConsoleApp</code> 负责：

- MPI/命令行；
- schema 或 JSON 配置；
- 输出目录和日志；
- 构造 solver；
- 顺序调用初始化与求解。

<code>app/Euler/euler*.cpp</code> 仅选择模板模型并包装 MPI 生命周期。

<code>_explicit_instantiation</code> 中约 54 个很短的 cpp 文件不是 54 份算法。它们只是把 6 组大型模板实现为 9 个模型生成独立编译单元，以降低重复模板实例化成本。理解一个生成文件和
[生成脚本](../../src/Euler/_explicit_instantiation/__generate_explicit_inst__.py)
即可。

### 阶段 13：按研究方向添加旁支

主线完成后只选择与你当前工作相关的一支。

#### RANS

读 [RANS_ke.hpp](../../src/Euler/RANS_ke.hpp)，每个模型按同一三联组阅读：

1. 湍黏度；
2. 黏性通量；
3. 源项和 Jacobian。

建议顺序为 RKE、SST、Wilcox、SA。随后回到 evaluator 搜索
<code>hasSA</code> 和 <code>has2EQ</code>，观察模型核如何接入主路径。

#### 反应流

读：

1. [ChemicalSource.hpp](../../src/Euler/Chemistry/ChemicalSource.hpp)
2. [ChemicalSource.cpp](../../src/Euler/Chemistry/ChemicalSource.cpp)
3. [PhysicsProperties.hpp](../../src/Euler/Physics/PhysicsProperties.hpp)
4. [ConstVolTrajectory.hpp](../../src/Euler/Physics/ConstVolTrajectory.hpp)
5. [test_EulerEvaluatorReactive.cpp](../../test/cpp/Euler/test_EulerEvaluatorReactive.cpp#L97)

<code>ChemicalSource.hpp</code> 使用 PIMPL 隔离 Cantera 类型；具体 Cantera reactor、热物性、生成率和输运实现在 cpp 中。重点区分：

- Euler 对流/扩散状态更新；
- 单元内化学源；
- fully coupled 与 Strang splitting；
- 定容轨迹测试与完整流动求解。

#### 输出与 restart

最后读 [EulerSolver_PrintData.hxx](../../src/Euler/EulerSolver_PrintData.hxx)：

- <code>PrintData</code>；
- <code>PrintRestart</code>；
- restart 映射；
- <code>ReadRestart</code> 和跨 solver restart。

这部分不改变空间离散公式，可以从核心数值阅读中完全后移。

## 6. 关键数值代码说明

### 6.1 守恒量与热力状态

非反应理想气体的守恒状态写成

\[
\mathbf U =
\begin{bmatrix}
\rho & \rho\mathbf v & \rho E
\end{bmatrix}^{T}.
\]

压力的核心关系可概括为

\[
p=(\gamma_{\mathrm{eq}}-1)
\left(
\rho E-\frac{|\rho\mathbf v|^2}{2\rho}-\rho E_{\mathrm{base}}
\right),
\qquad
a^2=\gamma\frac{p}{\rho}.
\]

普通定比热气体中 <code>gammaEq</code>、<code>gamma</code> 和能量基准通常退化为熟悉的关系；反应流中则必须区分热力学导数、等效比热比和组分能量基准。阅读时不要把所有 <code>gamma</code> 名称视作完全相同的物理量。

### 6.2 ALE 面法向无黏通量

在移动网格/旋转框架的 ALE 形式下，法向通量可概括为

\[
\mathbf F_n =
\begin{bmatrix}
\rho(\mathbf v\cdot\mathbf n-v_{g,n})\\
\rho\mathbf v(\mathbf v\cdot\mathbf n-v_{g,n})+p\mathbf n\\
\rho E(\mathbf v\cdot\mathbf n-v_{g,n})
+p(\mathbf v\cdot\mathbf n)
\end{bmatrix}.
\]

读 <code>Gas.hpp</code> 时要分清：

- 物理通量：单侧状态代入控制方程；
- 数值通量：由左右状态和特征信息构造；
- 黏性通量：依赖状态梯度和输运系数；
- 谱半径：用于耗散、CFL 和隐式近似，不等同于通量本身。

### 6.3 一个单元 RHS 的组成

半离散形式可以用

\[
\frac{d\mathbf U_i}{dt}
=-\frac{1}{V_i}
\sum_{f\in\partial i}
\int_f \widehat{\mathbf F}\cdot\mathbf n_i\,dS
+\mathbf S_i
\]

理解。实现中的关键点是：

- 面只计算一次；
- 同一内部面通量以相反符号写入左右单元；
- 边界面通过虚状态或专用壁面公式封闭；
- cell source 在面贡献完成后添加；
- MPI 分区面要求 ghost 状态和归属规则保持一致。

### 6.4 Evaluator 与 Solver 的分工

| 层 | 主要职责 | 不负责 |
|---|---|---|
| <code>EulerEvaluator</code> | dt、面通量、边界状态、源项、RHS、隐式算子局部操作 | 顶层 ODE 生命周期、输出日程和主循环 |
| <code>EulerSolver</code> | 拥有解与工作数组、初始化、ODE 回调、线性算法选择、CFL/收敛、I/O | 重新实现局部气体动力学公式 |
| <code>SingleBlockApp</code> | 命令行、配置合并、日志和 solver 启动 | 数值离散 |

这个分工是阅读大型模板函数时最重要的导航原则。

## 7. 用测试作为可执行说明书

### 7.1 推荐测试顺序

1. [test_GasThermo.cpp](../../test/cpp/Euler/test_GasThermo.cpp#L56)
2. [test_RiemannSolvers.cpp](../../test/cpp/Euler/test_RiemannSolvers.cpp#L156)
3. [test_EulerEvaluator.cpp](../../test/cpp/Euler/test_EulerEvaluator.cpp#L146)
4. 需要反应流时再读 [test_EulerEvaluatorReactive.cpp](../../test/cpp/Euler/test_EulerEvaluatorReactive.cpp#L97)

前两个测试对应“单点热力学”和“单面通量”；第三个对应“一个 evaluator 步骤”；它们正好与自底向上的三层一致。

### 7.2 建议验证命令

根据项目要求，若 C++ 源发生变化，应先重建相应目标。本阅读文档本身不要求运行这些测试；实际单步调试时可使用：

~~~bash
cmake --build build \
  -t euler_test_gas_thermo euler_test_riemann_solvers -j8

ctest --test-dir build \
  -R '^euler_(gas_thermo|riemann_solvers)$' \
  --output-on-failure

cmake --build build -t euler_test_evaluator_pipeline -j8

ctest --test-dir build \
  -R '^euler_evaluator_pipeline_np1$' \
  --output-on-failure
~~~

### 7.3 建议算例

[euler_config_IV.json](../../cases/euler/euler_config_IV.json) 是等熵涡配置，适合理解初始化、误差评估和高阶重构。它的现有执行路径使用普通 <code>NS</code>，因此状态包含三维速度分量；不要在调试时错误地按四变量 <code>NS_2D</code> 解释数组。

## 8. 第一遍主动跳过的内容

以下代码不是不重要，而是会打断主线：

- <code>_explicit_instantiation</code> 下除一个示例外的其余短 cpp；
- <code>EulerSolver_PrintData.hxx</code> 的完整输出/restart 实现；
- <code>EulerEvaluator_EvaluateDt.hxx</code> 约 1–817 行的壁距算法；
- 同文件约 2831 行后的输出字段注册；
- 全部多重网格、两阶段嵌套求解和化学分裂；
- <code>CLDriver.hpp</code> 和 <code>SpecialFields.hpp</code>；
- 不研究湍流时的 <code>RANS_ke.hpp</code>；
- 不研究反应流时的 <code>Chemistry</code> 和组分输运；
- 每一个 <code>DNDS_FIELD</code> 配置项；
- Gas 中所有 batch 重复实现和 Roe M1–M9 变体；
- 每种边界条件的详细公式，只先读 dispatcher，再按算例点读。

## 9. 容易误读和需要核对的地方

1. [Euler.hpp::JacobianValue](../../src/Euler/Euler.hpp#L780) 是占位结构；真正使用的对角块容器是 [EulerJacobian.hpp::JacobianDiagBlock](../../src/Euler/EulerJacobian.hpp#L81)。

2. [EulerBC.hpp](../../src/Euler/EulerBC.hpp) 主要负责类型、配置映射和记录器。边界 ghost 的实际公式集中在 [generateBoundaryValue](../../src/Euler/EulerEvaluator_EvaluateDt.hxx#L2069) 及其后各分支。

3. <code>EulerEvaluator_EvaluateDt.hxx</code> 不只是 dt 文件。它同时包含壁距、<code>fluxFace</code>、source、BC 和输出字段实现。

4. <code>ReadMeshAndInitialize</code> 的名称容易让人以为解已初始化。它主要完成 mesh、VFV、evaluator 和数组构造；真正的初值/restart 选择在 <code>RunImplicitEuler</code> 开头。

5. [FixUMaxFilter](../../src/Euler/EulerEvaluator.hxx#L1368) 当前是空操作。读调用链时不要假设它已经修改状态。

6. 阅读 RANS 分支时，可把下列位置列为“待核对点”，但在完成公式与索引推导前不要直接判定为缺陷：
   - RKE epsilon 相关路径在 [RANS_ke.hpp](../../src/Euler/RANS_ke.hpp#L268) 附近读取附加变量索引；
   - SST <code>gammaC</code> 混合在 [RANS_ke.hpp](../../src/Euler/RANS_ke.hpp#L721) 附近。

## 10. 每阶段的阅读记录模板

建议不要只做逐行翻译。对每个关键函数固定记录以下项目：

~~~text
函数：
所在层：局部数学核 / 物理适配 / 空间离散 / 时间推进

输入：
- 每个参数的数学含义
- 数组形状
- father/son 与 ghost 更新要求
- 无量纲量还是物理量

输出或副作用：
- 返回值
- 被修改的数组
- MPI 通信
- 缓存或统计量

分支：
- 编译期 model 分支
- 运行时配置分支
- 第一遍采用的固定路径

不变量：
- 密度/压力正性
- 内部面守恒
- 维数和变量索引
- 单位或无量纲尺度

验证：
- 对应单元测试
- 可打印的中间量
- 极简手算状态
~~~

主线完成时，至少应产出四份笔记：

1. 一张模型 traits 与状态布局表；
2. 一张“一个面”的 <code>UL/UR → Flux → 两侧 RHS</code> 跟踪表；
3. 一张“一个单元”的面贡献、源项和体积归一化表；
4. 一张“一次隐式时间步”的 ODE callback 调用图。

## 11. 完成标准

完成核心阅读不以“看完所有文件”为标准，而以能回答以下问题为标准：

1. <code>NS_2D</code> 和 <code>NS</code> 的状态布局为何不同？
2. 守恒状态如何得到压力、声速和黏性所需原始变量？
3. 左右状态如何经过 Riemann 求解器产生一个面通量？
4. 内部面通量如何守恒地写入两个单元？
5. 边界虚状态在何处、依据什么配置生成？
6. <code>EvaluateDt</code> 使用哪些谱半径构造局部时间步？
7. 重构压缩 <code>betaPP</code> 与更新压缩 <code>alphaPP</code> 有何不同？
8. evaluator 和 solver 分别拥有、计算和调度什么？
9. 隐式矩阵、SGS/GMRES、ODE 方法之间如何连接？
10. 从 <code>app/Euler/euler*.cpp</code> 到 <code>Gas::RiemannSolver</code> 的完整调用链是什么？

能够不依赖全局搜索回答这十个问题，就已经掌握 Euler 主模块的结构。之后再进入某个 RANS 模型、反应流、特殊边界或输出格式，阅读成本会显著降低。
