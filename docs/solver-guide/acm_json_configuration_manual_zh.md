# DNDSR ACM JSON 配置说明书 {#acm_json_configuration_manual_zh}

版本日期：2026年10月13日。适用范围：当前本机 DNDSR 的常密度 ACM 程序，以及三维 Re=3000 圆柱 SST BDF2 LU-SGS 配置。

本说明书对应 `cases/acm/3D/acm3D_cylinder_Re3000_coarse_SST_BDF2_LUSGS.json`。主状态顺序为`[u,v,w,p,rho*k,rho*omega]`，当前rho0=1。当前内迭代收敛方式为 `Relative`，分别要求流场和SST残差相对各自本物理步初值下降到 `1e-3`；保留的 `1e-8` 仅在切换为 `Absolute` 时使用。

阅读顺序：先看第1—2节了解算例与读取规则，再看第5节选择收敛方式，第6—9节配置SST、网格和重构，第10—13节配置边界、输出并运行，第14节解释日志与常见错误。

本说明书已随统一状态修改更新；实现与边界/重启约定见[模型入口说明](acm_model_state_zh.md)。

## 1 算例概览

| 项目 | 本例设置 |
| --- | --- |
| 程序 | `acm2EQ3D.exe`，常密度三维两方程ACM |
| 空间离散 | CFV变分重构，二次多项式，CWBAP限制器 |
| 湍流模型 | `KOmegaSST`，输运 `k` 与 `omega` |
| 物性 | `rho0=1`，`dynamicViscosity=1/3000` |
| 来流及圆柱 | `U_inf=1`，`D=1`，因此 `Re_D=3000` |
| 网格 | `CylinderRe3900_coarse.cgns`；文件名不决定实际Re |
| 周期方向 | z方向，长度 `1.5*pi≈4.71238898038469` |
| 物理时间 | `dt=0.01`，`nSteps=10000`，完成后 `t=100` |
| 内迭代 | 局部CFL伪时间推进，统一六变量修正；流场LU-SGS，SST局部耦合隐式块 |
| 收敛方式 | `Relative`，流场与SST分别使用 `1e-3` |
| 场输出 | 初始场及每50个物理步输出；相邻输出物理时间差0.5 |

网格配置中的说明记录了210656个单元，包含26112个六面体和184544个三棱柱。当前模型是粗网格上的全湍流URANS设定，入口湍流量采用假定的1%湍流强度和0.07D长度尺度。时间步、内迭代阈值及网格分辨率应分别做敏感性检查后再用于定量结果比较。

## 2 JSON读取与默认值

程序读取一个完整配置文件，支持 `//` 和块注释，再应用命令行JSON Pointer覆盖项。相对路径按启动程序时的工作目录解释；按本说明书从 `build/` 启动时，`../data/...` 指向仓库的 `data/`。

表格中的“默认参考”表示配置对象或schema的默认参考值，不意味着可以删除所有字段。ACM一般按完整JSON读取；仅兼容加载器列出的新增字段会自动补入。应以当前完整算例为模板修改并执行 `--check-config`。`cacheDiffBaseSize` 的三维对象构造值为4；通用VR schema默认可能显示255，本说明书按三维ACM对象列4。

当前圆柱JSON没有显式写入 `dualTimeResidualScales`、`dualTimeRelativeTolerances`、`dualTimeRequireConvergence`，加载器分别补入 `[1,1,1,1,1,1]`、`[0,0,0,0,0,0]`、`true`。缺少 `dualTimeConvergenceCriterion` 的旧配置补入 `Absolute`；要用相对判据，应明确写入 `Relative`。

兼容加载器还补入缺少的 `physicalTimeStep`、稳态收敛/CFL字段、两组BDF相对阈值，以及受控变分重构的容差、扫描上限、检查间隔、松弛和GMRES字段；缺少整个 `restartSettings` 时补入关闭重启的默认对象。

## 3 程序与方程选择 solver

| 字段 | 本例值 | 默认参考 | 含义及使用条件 |
| --- | --- | --- | --- |
| `type` | `"ACM"` | `"ACM"` | 方程族。当前必须为 ACM，与所运行的可执行程序匹配。 |
| `discretization` | `"CFV"` | `"CFV"` | 空间离散模块。当前为 CFV。 |
| `model` | `"ConstantDensity2EQ3D"` | `"ConstantDensity2EQ3D"` | 三维常密度模型；二维程序需配套 ConstantDensity2D。 |
| `fieldNVariables` | `6` | `6` | 主状态为[u,v,w,p,rho*k,rho*omega]，SST占两个守恒输运分量。 |

二维SST算例使用acm2EQ2D.exe、ConstantDensity2EQ2D和二维网格，状态长度仍为6。SA入口为5，层流入口为4。

## 4 物性与人工可压缩性 acmSettings

| 字段 | 本例值 | 默认参考 | 含义及使用条件 |
| --- | --- | --- | --- |
| `rho0` | `1.0` | `1.0` | 常密度，必须为正；本例为1。 |
| `beta2` | `4.0` | `1.0` | 人工可压缩性参数 beta²，必须为正；控制伪时间预条件和特征速度，不是物理时间步。 |
| `alpha` | `0.5` | `0` | Turkel耦合参数，要求有限值；参与预条件矩阵、特征分解、通量和远场边界。 |
| `dynamicViscosity` | `0.000333333333333` | `0` | 分子动力黏度 mu；SST要求正值。运动黏度 nu=mu/rho0。 |
| `riemannSolverType` | `"Roe"` | `"Roe"` | 对流数值通量，可选 Roe 或 Rusanov；本例使用 Roe。 |
| `pressureStorage` | `"PhysicalP"` | `"PhysicalP"` | 当前只支持 PhysicalP，即第4分量直接存压力变量；不支持 ScaledPOverBeta2。 |
| `entropyFixRatio` | `0.05` | `0.05` | Roe通量的熵修正宽度系数，要求非负。 |
| `pressureReference` | `0` | `0` | 保留的压力参考信息。当前网格求解/输出直接使用状态中的p，不自动减去此值。 |
| `farFieldValue` | `[1.0, 0, 0, 0]` | `[0, 0, 0, 0]` | 保留的远场状态字段；网格边界实际读取 boundaryConditions 的value或顶层 boundaryValue。 |
| `enableViscousFlux` | `true` | `false` | 启用黏性通量；SST必须为true，并在分子黏度上耦合涡黏度。 |

本例的雷诺数计算为 `Re_D=rho0*U_inf*D/dynamicViscosity=3000`。更改雷诺数时应按所用无量纲化同步检查黏度、密度、速度和长度尺度，而不是改网格文件名。SST/RANS的压力变量采用平均流模型的压力约定；`PhysicalP` 只说明未按 `beta2` 缩放存储。

## 5 时间推进与收敛 timeMarchSettings

### 5.1 物理步与伪时间步

BDF2的外层一步推进物理时间 `physicalTimeStep`。每个物理步内部用伪时间迭代消除物理缺陷；`cfl`、`maximumPseudoTimeStep` 和 `implicitRelaxation` 控制这一内部求解过程。修改伪时间参数不会把物理时间步从0.01自动改成其他值。

| 字段 | 本例值 | 默认参考 | 含义及使用条件 |
| --- | --- | --- | --- |
| `integrator` | `"BDF2DualTimeLUSGS"` | `"ExplicitSSPRK3"` | 时间推进类型；本例是物理BDF2双时间推进，流场内迭代使用LU-SGS。 |
| `nSteps` | `10000` | `0` | 外层步数。BDF2中为物理步数；稳态算法中为伪时间外层步数。 |
| `pseudoTimeStep` | `0.0005` | `0.01` | 关闭CFL步长时使用的固定伪时间步。当前开启CFL，不作为实际物理时间步。 |
| `physicalTimeStep` | `0.01` | `0.01` | 均匀物理时间步 dt。当前0.01，完整10000步对应物理时间100。 |
| `useCFLTimeStep` | `true` | `false` | true时按当前流场和黏性谱半径计算伪时间步。 |
| `useLocalTimeStep` | `true` | `true` | true保留各单元局部伪时间步；false采用MPI全局最小步长。 |
| `cfl` | `5.0` | `0.5` | 伪时间CFL系数，必须为正；不直接等于物理时间步。 |
| `maximumPseudoTimeStep` | `0.005` | `1e+100` | CFL计算得到的伪时间步上限，必须为正。 |
| `maxImplicitIterations` | `100` | `20` | 每个物理步最多内迭代次数，必须为正；达到上限不等于收敛。 |

首个物理步采用BE（日志为BDF1），之后采用均匀步长BDF2。SST某个单元变量的BDF2历史若与正值下限不相容，会仅对该单元变量回退到BE，选择在该物理步内固定。回退依据本步固定的已完成历史判断。

### 5.2 明确选择绝对或相对判据

| 字段 | 本例值 | 默认参考 | 含义及使用条件 |
| --- | --- | --- | --- |
| `implicitTolerance` | `1e-08` | `1e-10` | Absolute模式的绝对残差阈值，要求非负。Relative模式保留此值，但不拿它提前结束内迭代。 |
| `dualTimeConvergenceCriterion` | `"Relative"` | `"Absolute"` | 只允许 Absolute 或 Relative。缺少字段时补入 Absolute；不是流场/SST组选择。 |
| `dualTimeFlowRelativeTolerance` | `0.001` | `0` | 正值启用流场 [u,v,w,p] 的组RMS；Relative目标为该值乘本物理步流场初值。范围[0,1)。 |
| `dualTimeTurbulenceRelativeTolerance` | `0.001` | `0` | 正值启用活跃湍流变量组RMS；SST为 [k,omega]，使用其独立的本步初值。范围[0,1)。 |
| `dualTimeResidualScales` | 未写入，加载器补默认 | `[1.0, 1.0, 1.0, 1.0, 1.0, 1.0]` | 6个正的逐方程绝对残差尺度，顺序 [u,v,w,p,q1,q2]；对应组启用组RMS时必须保持1。 |
| `dualTimeRelativeTolerances` | 未写入，加载器补默认 | `[0, 0, 0, 0, 0, 0]` | 逐方程相对系数，顺序同左；组RMS关闭时使用。Relative模式要求每个活跃方程有正系数。 |
| `dualTimeRequireConvergence` | 未写入，加载器补默认 | `true` | true时内迭代未满足所选条件就报错，不提交该步历史或物理时间；false仅用于预算受限的诊断推进。 |

当前两组RMS都已启用。每个物理步在任何修正之前记录流场初值 `R_flow,0` 和SST初值 `R_SST,0`，本步内保持不变；下个物理步重新记录。流场RMS包含四个流场方程，SST RMS包含两个活跃输运方程。

```text
Relative:
  R_flow <= 1e-3 * R_flow,0
  R_SST  <= 1e-3 * R_SST,0

Absolute:
  R_flow <= 1e-8
  R_SST  <= 1e-8
```

同一模式下流场与活跃SST组都要通过。`Relative` 不使用 `1e-8` 兜底或提前停止，`Absolute` 不使用 `1e-3` 放宽目标；选择的是收敛方式。此前的 `Both`、`Either`、`Flow`、`Turbulence` 值不再合法，已有这些值的配置应改为 `Absolute` 或 `Relative`。

例如本物理步流场初值为40.2，则其相对目标为0.0402；SST初值为6114，则其独立目标为6.114。两个目标不能互相替代。相对初值为零时目标也为零，仅接受当前残差为零。

某组相对字段设为0时改用该组逐方程控制。`Relative` 模式必须为每个活跃方程或其组设置正的相对系数，否则配置检查报错。启用组RMS时，同组逐方程尺度必须为1，逐方程相对系数必须为0，避免两套配置互相覆盖。层流不检查湍流组；SA的第二存储槽不参与判断。

### 5.3 隐式线性修正

| 字段 | 本例值 | 默认参考 | 含义及使用条件 |
| --- | --- | --- | --- |
| `implicitRelaxation` | `0.7` | `1.0` | 流场隐式修正松弛系数，位于(0,1]；当前每次加入0.7倍修正。 |
| `lusgsSweeps` | `2` | `2` | 每次LU-SGS线性修正的固定残差校正次数，至少1；不是每步内迭代上限。 |
| `gmresSubspace` | `20` | `10` | GMRES的Arnoldi子空间维数，至少2；当前LU-SGS模式不启用。 |
| `gmresRestarts` | `3` | `3` | GMRES重启次数上限，非负；当前LU-SGS模式不启用。 |
| `gmresRelativeTolerance` | `0.0001` | `1e-06` | GMRES线性预条件残差的相对阈值；与物理步收敛阈值分别设置。 |
| `gmresPreconditioner` | `"LUSGS"` | `"LUSGS"` | GMRES左预条件器，可选 LUSGS 或 BlockJacobi；当前仅保留配置。 |

如将 `integrator` 改为 `BDF2DualTimeGMRES`，流场修正改用GMRES；SST守恒变量参与统一GMRES向量，局部源项块在预条件器中求解。流场GMRES与重构GMRES各有一套参数，开启其中一个不会同时开启另一个。

### 5.4 稳态专用字段

| 字段 | 本例值 | 默认参考 | 含义及使用条件 |
| --- | --- | --- | --- |
| `steadyRelativeTolerance` | `0` | `0` | 仅用于稳态隐式算法，相对当前外层空间残差；BDF2中必须为0。 |
| `steadyAdaptiveCFL` | `false` | `false` | 稳态隐式算法的自适应CFL开关；BDF2中必须为false。 |
| `steadyCFLMin` | `0.1` | `0.1` | 稳态自适应CFL下限，必须为正；当前模式不启用。 |
| `steadyCFLMax` | `20.0` | `20.0` | 稳态自适应CFL上限，不小于下限；当前模式不启用。 |
| `steadyCFLGrowth` | `1.5` | `1.5` | 稳态CFL最大增长倍数，至少为1；当前模式不启用。 |
| `steadyCFLReduction` | `0.5` | `0.5` | 稳态失败/残差增长后的CFL缩小系数，位于(0,1)；当前模式不启用。 |

支持的时间推进枚举如下。切换到稳态模式时，BDF专用相对设置应清零，`dualTimeConvergenceCriterion` 保持 `Absolute`；稳态相对控制另用 `steadyRelativeTolerance`。

| integrator | 用途 |
| --- | --- |
| `ExplicitSSPRK3` | 显式伪时间推进 |
| `ImplicitEulerBlockJacobi` | 稳态伪时间隐式Block-Jacobi |
| `ImplicitEulerLUSGS` | 稳态伪时间隐式LU-SGS |
| `ImplicitEulerGMRES` | 稳态伪时间隐式GMRES |
| `BDF2DualTimeLUSGS` | 物理时间BDF2，流场内迭代LU-SGS |
| `BDF2DualTimeGMRES` | 物理时间BDF2，流场内迭代GMRES |

## 6 SST模型与输运 turbulenceSettings

| 字段 | 本例值 | 默认参考 | 含义及使用条件 |
| --- | --- | --- | --- |
| `model` | `"KOmegaSST"` | `"KOmegaWilcox"` | 两方程入口的闭合模型；当前KOmegaSST输运rho*k、rho*omega。 |
| `initialValue` | `[0.00015, 0.31943828249997]` | `[1e-06, 1.0]` | 湍流初值 [q1,q2]；本例为 [k,omega]，新入口以initialState尾部的rho*q为初值，此字段保留兼容。 |
| `farFieldValue` | `[0.00015, 0.31943828249997]` | `[1e-06, 1.0]` | 湍流入口/远场入流值 [k,omega]；出流采用外推/零法向梯度处理。 |
| `minimumValue` | `[1e-12, 1e-10]` | `[1e-12, 1e-10]` | 每个活跃变量的正值下限。各变量独立限制，k触底不冻结omega。 |
| `maximumValue` | `[1000000, 1e+12]` | `[1000000, 1e+12]` | 各湍流变量的有限上限，不小于对应下限。 |
| `maximumEddyViscosityRatio` | `100000` | `100000` | 涡黏度与分子黏度比 mu_t/mu 的上限，非负。 |
| `wallOmegaCoefficient` | `800.0` | `800.0` | 壁面omega系数C，omega_w=C*nu/d²，d采用截断后的壁距；当前C=800。 |
| `enableSourceTerms` | `true` | `true` | 是否计算模型的生产/破坏等源项；正常SST计算设true。 |
| `secondOrderReconstruction` | `true` | `true` | 湍流变量采用带限制器的二阶Green-Gauss重构；不随流场多项式阶数自动改变。 |
| `transportSubsteps` | `1` | `1` | 统一入口固定为1，所有变量共享阶段/内迭代。 |
| `transportTimeScale` | `0.1` | `0.25` | 湍流与流场伪时间步的比例，位于(0,1]；当前0.1，不改变physicalTimeStep。 |
| `wallDistanceMethod` | `1` | `1` | 壁距算法：0暴力搜索，1 AABB树搜索。 |
| `wallDistanceExecution` | `0` | `0` | 壁距MPI执行：0各rank并行，1串行，>1按rank模值分批执行。 |
| `wallDistanceSubdivide` | `0` | `0` | 壁面四边形/曲面细分级别，非负；当前0。 |
| `minimumWallDistance` | `1e-10` | `1e-10` | 计算模型/壁面值时使用的正壁距下限，必须为正。 |
| `wallDistanceVerbose` | `0` | `0` | 壁距计算诊断输出级别，非负；0关闭额外信息。 |

统一入口使用dTau_turb=dTau_flow*transportTimeScale作为湍流伪时间尺度，当前比例0.1；物理BDF导数始终使用dt=0.01。变量上下限独立约束各分量修正。

当前初始与远场值为 `k=0.00015`、`omega≈0.3194382825`，来自配置所采用的 `I=1%`、`ell=0.07D`：`k=1.5*(I*U_inf)^2`，`omega=sqrt(k)/(0.09^0.25*ell)`。两方程入口还可选 `KOmegaWilcox`、`RealizableKEpsilon`，守恒尾部顺序分别为 `[rho*k,rho*omega]`、`[rho*k,rho*epsilon]`；SA使用独立入口，尾部为 `rho*nuTilde`。Laminar使用原四变量入口。

## 7 网格与周期配置 meshSettings

| 字段 | 本例值 | 默认参考 | 含义及使用条件 |
| --- | --- | --- | --- |
| `meshFile` | `"../data/mesh/ACM3D_Cylinder_Re3900/CylinderRe3900_coarse.cgns"` | `""` | CGNS网格路径，相对程序启动工作目录解析。当前网格文件名含Re3900，实际Re由黏度等输入决定。 |
| `meshElevation` | `0` | `0` | 0不提升几何阶次，1将O1几何提升到O2。 |
| `meshDirectBisect` | `0` | `0` | 直接网格二分次数，合法范围0至4；0不细化。 |
| `meshReorderCells` | `0` | `0` | 0不重排局部单元，1启用局部单元重排。 |
| `periodicTolerance` | `1e-09` | `1e-09` | 周期节点匹配容差，非负；需要与网格坐标尺度相容。 |
| `periodicTranslation1` | `[0, 0, 4.71238898038469]` | `[0, 0, 0]` | 第一组周期平移向量；当前沿z方向平移1.5*pi。 |
| `periodicTranslation2` | `[0, 0, 0]` | `[0, 0, 0]` | 第二组周期平移向量；当前全0，未设置非零平移。 |
| `periodicTranslation3` | `[0, 0, 0]` | `[0, 0, 0]` | 第三组周期平移向量；当前全0，未设置非零平移。 |

本例网格路径为 `../data/mesh/ACM3D_Cylinder_Re3900/CylinderRe3900_coarse.cgns`。网格已有跨展向的周期边界配对；第一周期平移为 `[0,0,4.71238898038469]`。平移方向和长度必须与所用CGNS周期面一致。

### 7.1 MPI划分 partitionOptions

| 字段 | 本例值 | 默认参考 | 含义及使用条件 |
| --- | --- | --- | --- |
| `metisType` | `"KWAY"` | `"KWAY"` | 网格划分方法：KWAY多路划分，RB递归二分。 |
| `metisUfactor` | `20` | `20` | METIS负载不平衡控制参数，至少1；沿用METIS ufactor含义。 |
| `metisSeed` | `0` | `0` | METIS随机种子；用于固定划分的随机序列。 |
| `edgeWeightMethod` | `0` | `0` | 划分图边权：0不加权，1按面尺寸加权。 |
| `metisNcuts` | `3` | `3` | METIS尝试的划分次数，至少1。 |

MPI进程数通过 `mpirun -np N` 指定，不由 `metisNcuts` 或 `nFacesPerRank` 决定。修改划分参数不会自动改变物理边界条件。

## 8 流场重构 reconstructionSettings

| 字段 | 本例值 | 默认参考 | 含义及使用条件 |
| --- | --- | --- | --- |
| `type` | `"Variational"` | `"GreenGauss"` | 流场重构类型：FirstOrder、GreenGauss、Variational。 |
| `variationalIterations` | `3` | `3` | 固定次数VR扫描数；启用受控重构时也用于最小扫描/检查条件。 |
| `variationalTolerance` | `0` | `0` | 缩放后的重构方程最大缺陷阈值；0使用固定扫描，正值启用收敛控制。 |
| `variationalMaxIterations` | `10000` | `10000` | 受控重构的最大扫描次数；必须覆盖variationalIterations。当前固定扫描模式不启用。 |
| `variationalCheckInterval` | `10` | `10` | 受控重构检查方程缺陷的间隔，至少1；首轮、末轮等也会检查。 |
| `variationalRelaxation` | `0.7` | `0.7` | 受控重构缺陷更新的松弛系数，位于(0,1]；当前固定扫描走vfvSettings的SOR设置。 |
| `variationalUseGMRES` | `false` | `false` | 受控重构中使用GMRES；当前false。 |
| `variationalGMRESSubspace` | `20` | `20` | 重构GMRES子空间维数，至少2；与流场GMRES参数分别设置。 |
| `variationalGMRESRestarts` | `2` | `2` | 重构GMRES重启次数，非负。 |
| `variationalGMRESRelativeTolerance` | `0.1` | `0.1` | 每次重构GMRES线性求解的相对阈值，位于(0,1]；不是BDF2物理步阈值。 |
| `resetVariationalCoefficients` | `false` | `false` | 每次残差求值前是否清零VR系数。当前BDF2固定扫描模式由程序强制设true，即使JSON为false。 |
| `enableLimiter` | `true` | `true` | 是否启用所选重构限制器；当前true。 |
| `limiterType` | `"CWBAP"` | `"LocalExtrema"` | LocalExtrema、WBAP或CWBAP；WBAP/CWBAP要求Variational。 |

当前 `Variational + variationalTolerance=0 + variationalIterations=3` 表示每次残差求值进行固定3次VR扫描；并未要求重构方程自身收敛。固定扫描采用 `vfvSettings.SORInstead=true` 和 `jacobiRelax=1`，当前0.7的 `variationalRelaxation` 属于受控重构设置。

BDF2固定扫描模式中，程序强制每次从零VR系数起步，保证同一当前状态对应同一物理缺陷；因此JSON中的 `resetVariationalCoefficients=false` 不会让当前模式使用暖启动。若把 `variationalTolerance` 改为正值，会启用重构方程缺陷控制并检查扫描上限；这与物理步 `Relative`/`Absolute` 判据分别控制。

## 9 CFV多项式与限制器 vfvSettings

| 字段 | 本例值 | 默认参考 | 含义及使用条件 |
| --- | --- | --- | --- |
| `maxOrder` | `2` | `1` | 重构多项式次数；ACM支持0至3。本例2为二次多项式，不是BDF时间阶数。 |
| `intOrder` | `4` | `1` | 全局积分阶次；本例4。 |
| `ignoreMeshGeometryDeficiency` | `false` | `false` | 是否忽略网格几何缺陷提示；当前false。 |
| `nIterCellSmoothScale` | `15` | `15` | 单元平滑尺度迭代次数，非负。 |
| `intOrderVR` | `-1` | `-1` | VR积分阶次；负值继承intOrder，当前-1实际使用4。 |
| `intOrderVRBC` | `-2` | `-2` | VR边界积分阶次；-1继承intOrder，<-1继承VR积分阶次；当前-2实际使用4。 |
| `cacheDiffBase` | `false` | `false` | 缓存积分点基函数/导数值，改变存储与计算方式。当前关闭。 |
| `cacheDiffBaseSize` | `4` | `4` | 缓存的导数基项范围；三维对象构造值为4。cacheDiffBase=false时不启用缓存。 |
| `jacobiRelax` | `1.0` | `1.0` | 固定VR SOR/Jacobi扫描的松弛系数；当前1。 |
| `SORInstead` | `true` | `true` | true用SOR扫描，false用松弛Jacobi；当前true。 |
| `smoothThreshold` | `0.01` | `0.01` | 重构平滑性指标阈值，参与限制器判别。 |
| `WBAP_nStd` | `10.0` | `10.0` | WBAP类限制器的n参数，当前10。 |
| `normWBAP` | `true` | `false` | 使用多项式范数形式的WBAP处理；当前true。 |
| `limiterBiwayAlter` | `0` | `0` | 双向限制处理：0 wbap-L2，1 minmod。 |
| `subs2ndOrder` | `0` | `0` | 二阶替代方法：0原VR，1 Gauss，2最小二乘，11 GGMP；主要用于一次多项式场景。 |
| `subs2ndOrderGGScheme` | `1` | `1` | GG二阶替代插值方式：0不插值，1距离插值；当前不启用替代方法。 |
| `svdTolerance` | `0` | `0` | 重构线性代数的SVD容差设置；当前0沿用算子相应处理。 |
| `bcWeight` | `1.0` | `1.0` | 边界重构条件的权重，当前1。 |

当前 `maxOrder=2` 是二次多项式的重构设置，`intOrder=4` 是积分设置。流场多项式阶数、SST的二阶重构、BDF2时间阶数分别配置。不能把三者都理解为同一个“二阶开关”。

### 9.1 基函数 baseSettings

| 字段 | 本例值 | 默认参考 | 含义及使用条件 |
| --- | --- | --- | --- |
| `localOrientation` | `false` | `false` | 是否使用局部方向构造基函数；当前false。 |
| `anisotropicLengths` | `false` | `false` | 是否使用各向异性长度尺度；当前false。 |

### 9.2 泛函权重 functionalSettings

| 字段 | 本例值 | 默认参考 | 含义及使用条件 |
| --- | --- | --- | --- |
| `scaleType` | `"BaryDiff"` | `"BaryDiff"` | 重构泛函的尺度类型；当前BaryDiff，使用重心差相关尺度。 |
| `scaleMultiplier` | `1.0` | `1.0` | 泛函尺度倍率，非负；当前1。 |
| `dirWeightScheme` | `"Factorial"` | `"Factorial"` | 方向权重方案；当前Factorial。 |
| `dirWeightCombPowV` | `1` | `1` | 方向权重组合幂参数；当前1。 |
| `manualDirWeights` | `[1.0, 1.0, 0.5, 0.16666666666666666, 0.041666666666666664]` | `[1.0, 1.0, 0.5, 0.16666666666666666, 0.041666666666666664]` | 手动方向权重向量；当前Factorial方案下保留，不按Manual方案启用。 |
| `geomWeightScheme` | `"GWNone"` | `"GWNone"` | 几何权重方案；GWNone表示不启用额外几何权重方案。 |
| `geomWeightPower` | `0.5` | `0.5` | 几何权重幂参数；当前GWNone下不用于启用额外几何权重。 |
| `geomWeightPower1` | `0` | `0` | 几何权重第一附加幂参数，当前0。 |
| `geomWeightPower2` | `0` | `0` | 几何权重第二附加幂参数，当前0。 |
| `useAnisotropicFunctional` | `false` | `false` | 是否启用各向异性泛函；当前false。 |
| `tanWeightScale` | `1.0` | `1.0` | 各向异性泛函中的切向权重尺度；当前不启用该路径。 |
| `anisotropicType` | `"InertiaCoord"` | `"InertiaCoord"` | 各向异性泛函类型；当前InertiaCoord，仅在对应功能开启时使用。 |
| `inertiaWeightPower` | `1.0` | `1.0` | 惯性权重幂参数；当前1，随对应泛函路径使用。 |
| `geomWeightBias` | `0` | `0` | 几何权重偏置，当前0。 |
| `greenGauss1Weight` | `0` | `0` | Green-Gauss类型1泛函项权重；当前0。 |
| `greenGauss1Bias` | `0.5` | `0.5` | Green-Gauss类型1项的偏置，当前0.5。 |
| `greenGauss1Penalty` | `0` | `0` | Green-Gauss类型1惩罚项系数，当前0。 |
| `greenGaussSpacial` | `0` | `0` | 源码字段拼写为Spacial；0默认空间模式，1均匀权重模式。 |

这些字段控制重构泛函的尺度及权重，不是流场或SST的时间收敛阈值。当前未启用各向异性泛函和额外几何权重方案，相关幂次/偏置仍作为完整JSON的一部分保留。

## 10 边界与初值

### 10.1 顶层状态字段

| 字段 | 本例值 | 默认参考 | 含义及使用条件 |
| --- | --- | --- | --- |
| `defaultBoundaryType` | `"BCFar"` | `"BCFar"` | 未单独映射的外部边界所用类型；当前BCFar。周期面由网格周期配对处理。 |
| `initialState` | `[1.0, 0.001, 0, 0, 0.00015, 0.31943828249997]` | `[1.0, 0, 0, 0, 1e-06, 1.0]` | 整个模型的均匀初值 [u,v,w,p,rho*k,rho*omega]；v=0.001用于打破严格横向对称性。 |
| `boundaryValue` | `[1.0, 0, 0, 0]` | `[1.0, 0, 0, 0]` | 未单独映射边界的默认流场数据 [u,v,w,p]。 |
| `leftState` | `[1.0, 0, 0, 0]` | `[1.0, 0, 0, 0]` | 预览内核的左状态；不替代网格运行的initialState。 |
| `rightState` | `[0, 0, 0, 0]` | `[0, 0, 0, 0]` | 预览内核的右状态；不是网格中的分区初值。 |
| `unitNormal` | `[1.0, 0, 0]` | `[1.0, 0, 0]` | 预览内核的面法线；物理网格运行使用网格实际面法线。 |
| `nFacesPerRank` | `16` | `1` | 预览内核的每rank面数；不决定MPI进程数或网格单元数。 |

`initialState=[1,0.001,0,0,0.00015,0.31943828249997]` 给流场和SST守恒量设置均匀初值，此例rho0=1；这个小横向扰动本身不随z变化。三维网格并不会自动给初始化添加展向扰动。

### 10.2 boundaryConditions数组

| 字段 | 本例 WALL / FAR | 默认参考 | 含义及使用条件 |
| --- | --- | --- | --- |
| `type` | `"BCWall"` / `"BCFar"` | `"BCFar"` | 边界类型；WALL为BCWall，FAR为BCFar。 |
| `name` | `"WALL"` / `"FAR"` | `""` | CGNS中的边界区名称，必须非空并与网格匹配。 |
| `value` | `[0, 0, 0, 0]` / `[1.0, 0, 0, 0]` | `[0, 0, 0, 0]` | 流场边界数据 [u,v,w,p]；各分量是否使用由边界类型决定，不含k、omega。 |
| `frameOption` | `0` / `0` | `0` | 保留的参考系选项；本例0，不作为Euler移动参考系配置使用。 |
| `anchorOption` | `0` / `0` | `0` | 保留的锚定选项；本例0。 |
| `integrationOption` | `0` / `0` | `0` | 保留的边界积分选项；本例0。 |
| `specialOption` | `0` / `0` | `0` | 特殊边界子类型；当前BCSpecial只支持0。 |
| `rectifyOption` | `0` / `0` | `0` | 保留的对称修正选项；本例0。 |
| `valueExtra` | `[]` / `[]` | `[]` | 附加边界数据；本例空数组，不填写Euler温度/能量数据。 |

`WALL` 使用无滑移壁面，壁面速度为0，流场压力按该边界外推；SST壁面处理另外施加k面值0及 `omega_w=800*nu/d²`。`FAR` 使用特征远场处理，导入向域内传播的特征信息，流场目标为 `[1,0,0,0]`，SST远场入流值从 `turbulenceSettings.farFieldValue` 取得。

其他流场边界可用 `BCWallInvis`（滑移壁）、`BCSym`（对称）、`BCOut`（外推出口）、`BCOutP`（给定压力出口）、`BCIn`（给定状态入口）、`BCInPsTs`（在ACM中给定速度并外推压力）和 `BCSpecial`。`BCWallIsothermal` 沿用无滑移壁面名称，常密度ACM各入口均不求温度或能量。

## 11 输出 outputSettings

| 字段 | 本例值 | 默认参考 | 含义及使用条件 |
| --- | --- | --- | --- |
| `interval` | `50` | `0` | 外层步输出间隔；BDF2中为物理步数。0关闭场输出，writeInitial也随之关闭。 |
| `directory` | `"../data/outACM/acm3D_CylinderRe3000_coarse_SST_BDF2_LUSGS"` | `"../data/outACM"` | 输出目录，相对启动工作目录解析。 |
| `prefix` | `"CylinderRe3000_coarse_SST_BDF2_LUSGS"` | `"acm"` | 文件及时间序列前缀。 |
| `writeInitial` | `true` | `true` | interval>0时是否写初始场；当前true，写第0步。 |

当前输出目录为 `../data/outACM/acm3D_CylinderRe3000_coarse_SST_BDF2_LUSGS`。文件名使用8位步号，例如 `CylinderRe3000_coarse_SST_BDF2_LUSGS_00000050.vtkhdf`；对应物理时间0.5。还会生成同前缀的 `.vtkhdf.series` 时间序列。

当前场文件包含 `Velocity`、`Pressure`、`TurbulenceK`、`TurbulenceOmega`。文件步号与VTK时间值分别代表外层步号和真实物理时间。建议在ParaView中打开 `.vtkhdf.series` 查看时间序列。

## 12 重启 restartSettings

| 字段 | 本例值 | 默认参考 | 含义及使用条件 |
| --- | --- | --- | --- |
| `flowFile` | `""` | `""` | 同分区VTK-HDF流场初始化文件；空字符串关闭。当前BDF2不接受这种无历史重启。 |
| `completedSteps` | `0` | `0` | 稳态初始场所代表的已完成步数，非负；当前新算例为0。 |
| `writerRanks` | `0` | `0` | 写重启文件时的MPI进程数；启用重启时为正，并须与读入进程数相同。 |
| `requireTurbulence` | `false` | `false` | 稳态RANS重启时是否强制要求全部模型湍流字段；当前未启用重启。 |

当前BDF2配置从新初值开始。现有VTK-HDF重启文件不保存完整BDF2历史，不能通过设置非空 `flowFile` 延续当前物理时间计算；配置检查会拒绝。该初始化路径适用于支持的稳态运行，且要求相同分区和MPI进程数。

## 13 常用操作

### 13.1 检查当前JSON

```bash
cd /mnt/ssd-SATARAID5/home/mrz/projects/DNDSR/build
HWLOC_COMPONENTS=-gl OMP_NUM_THREADS=1 ./app/acm2EQ3D.exe \
  ../cases/acm/3D/acm3D_cylinder_Re3000_coarse_SST_BDF2_LUSGS.json \
  --check-config
```

这一步检查类型、字段和参数组合，不读入网格或推进时间。因此配置检查成功后，仍需实际运行确认网格路径、边界名称及数值行为。

### 13.2 用32个MPI进程运行

```bash
HWLOC_COMPONENTS=-gl OMP_NUM_THREADS=1 mpirun -np 32 ./app/acm2EQ3D.exe \
  ../cases/acm/3D/acm3D_cylinder_Re3000_coarse_SST_BDF2_LUSGS.json
```

其中 `OMP_NUM_THREADS=1` 为每个MPI进程的OpenMP线程设置，`HWLOC_COMPONENTS=-gl` 用于无图形计算环境。实际进程/线程数应与分配到的计算资源相匹配。

### 13.3 切换为绝对收敛

修改JSON中一个字段为 `"dualTimeConvergenceCriterion": "Absolute"` 即可使用保留的 `implicitTolerance=1e-8`；两组相对系数可保留，用于之后切换回 `Relative`。也可用命令行覆盖进行检查：

```bash
HWLOC_COMPONENTS=-gl OMP_NUM_THREADS=1 ./app/acm2EQ3D.exe \
  ../cases/acm/3D/acm3D_cylinder_Re3000_coarse_SST_BDF2_LUSGS.json \
  --check-config \
  -k /timeMarchSettings/dualTimeConvergenceCriterion -v '"Absolute"'
```

### 13.4 加严相对阈值或更改物理步长

把两组阈值从 `1e-3` 改为 `1e-4` 会要求下降四个数量级；需要足够的内迭代预算。把 `physicalTimeStep` 从0.01改为0.005，并将 `nSteps` 从10000改为20000，仍对应目标物理时间100；若保留每0.5时间输出一次，应同时把 `outputSettings.interval` 改为100。

## 14 日志及常见配置错误

| 日志字段 | 解释 |
| --- | --- |
| `ACM physical step`、`time` | 物理步号及真实物理时间 |
| `BDF1` / `BDF2` | 流场当前采用的时间历史系数 |
| `residual A -> B` | 本物理步流场初始RMS与最终RMS |
| `dtauMin` | 本步伪时间步的最小值 |
| `inner` | 已完成的内迭代次数 |
| `converged` | 所选方式是否满足；1满足，0未满足 |
| `convergenceCriterion` | 实际选择的Absolute或Relative |
| `flowRatio` | 流场最终残差/本步流场初值 |
| `turbResidual`、`turbInitial`、`turbRatio` | 湍流当前RMS、独立初值和对应比例 |
| `flowTarget`、`turbTarget` | 组RMS模式下的实际目标 |
| `eqResidual` | `[u,v,w,p,q1,q2]` 各方程原始RMS |
| `eqTarget` / `flowEqTarget` / `turbEqTarget` | 未启用相应组RMS时的逐方程目标 |
| `turbBE` | 各湍流变量采用局部BE回退的单元数 |
| `recSweeps`、`recDefect`、`recConverged` | 启用受控重构时的扫描数、缺陷和收敛状态 |

当前配置的相对日志应满足 `flowRatio<=1e-3` 且 `turbRatio<=1e-3`，同时 `converged=1`。初值与最终值均为零时比例记0；初值为零、最终非零时比例为无穷，当前相对模式不收敛。原始残差会完整报告，正值裁剪不把未收敛变量伪装成收敛。

| 情况或报错 | 原因与处理 |
| --- | --- |
| `must be Absolute or Relative` | 使用了旧组选择名或拼写不符，改成两种合法值之一 |
| `requires a positive relative tolerance` | Relative下活跃方程/组缺少正阈值，补入组或逐方程系数 |
| `steady convergence controls apply only...` | BDF2启用了稳态专用控制，将steadyRelativeTolerance置0、steadyAdaptiveCFL置false |
| `histories were not committed` | 内迭代预算内未收敛；检查残差、伪时间参数和重构，再调整迭代预算 |
| `grouped RMS and per-equation...` | 同组同时设置组RMS和非默认逐方程尺度/相对阈值，选择一套配置 |
| `WBAP/CWBAP requires Variational` | 重构类型与限制器组合不合法，使用Variational或更换限制器 |
| `VTK-HDF restart contains no BDF2 history` | 当前物理推进无法使用这种场文件重启，flowFile保持空 |
| 参数检查通过但找不到网格 | 相对路径按启动目录解析，检查是否在build目录运行 |
| `inner=100` | 仅表示达到迭代上限；还需查看converged，不能当成已收敛 |

## 15 文件与代码索引

- 当前算例：[acm3D_cylinder_Re3000_coarse_SST_BDF2_LUSGS.json](../../cases/acm/3D/acm3D_cylinder_Re3000_coarse_SST_BDF2_LUSGS.json)。
- 三维字段schema：[acm3D_schema.json](../../cases/acm/3D/acm3D_schema.json)。
- 配置分组和加载：[ACMConfig.hpp](../../src/ACM/ACMConfig.hpp)、[ACM.cpp](../../src/ACM/ACM.cpp)。
- 绝对/相对判据：[ACMTime.hpp](../../src/ACM/ACMTime.hpp)、[ACMTime.cpp](../../src/ACM/ACMTime.cpp)。
- 物理推进、输出和重启：[ACMSolver.hxx](../../src/ACM/ACMSolver.hxx)。
- 湍流设置及输运：[ACMTurbulence.hpp](../../src/ACM/ACMTurbulence.hpp)、[ACMTurbulenceTransport.hxx](../../src/ACM/ACMTurbulenceTransport.hxx)。
- CFV重构参数：[VRSettings.hpp](../../src/CFV/VRSettings.hpp)。
- 方程与实现详解：[ACM RANS说明](acm_rans_zh.md)、[BDF2双时间说明](acm_bdf2_dual_time_zh.md)。
