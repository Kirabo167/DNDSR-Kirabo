# ACM BDF2 双时间推进说明

## 1. 实现范围

ACM 求解器现在支持定常伪时间推进和非定常 BDF2 双时间推进。BDF2 模块参考了 Euler
求解器的时间历史管理和隐式线性求解组织方式，并采用所给 FORTRAN 程序中的 ACM
物理时间方程。它没有直接复用 Euler 的通用 BDF 状态方程，因为 ACM 压力只参与伪时间
人工压缩过程，不应具有物理时间导数。

SA/两方程入口已改用统一的五/六分量守恒输运状态，分别为acmSA2D/acmSA3D、
acm2EQ2D/acm2EQ3D。层流使用acm2D/acm3D。入口、JSON和隐式近似的完整约定见
[ACM统一状态说明](acm_model_state_zh.md)。

可选格式为：

- `BDF2DualTimeLUSGS`：内迭代直接使用分布式 ACM LU-SGS；
- `BDF2DualTimeGMRES`：内迭代使用左预条件 GMRES，预条件器仍由
  `gmresPreconditioner` 选择。

## 2. 控制方程

ACM 的四分量流场块和物理状态分别定义为

```text
U = [u, v, w, p]^T
Q = M U,   M = diag(1, 1, 1, 0)
```

双时间方程写成

```text
Gamma(U) * dU/dtau + dQ/dt = R(U, t)
```

其中 `tau` 是用于收敛每个物理步的伪时间，`t` 是真实物理时间，`R` 是现有空间离散
残差。矩阵 `M` 的压力对角元为零，因此速度采用 BDF 物理时间离散，压力没有物理
时间导数。SA/两方程在U中追加rho*q，物理质量矩阵相应为diag(1,1,1,0,1[,1])。

对新物理时刻的当前内迭代值 `U*`，非线性缺陷为

```text
D(U*) = R(U*, t[n+1])
        - M * (a0*U* + a1*U[n] + a2*U[n-1]) / dt
```

第一物理步使用后向欧拉启动：

```text
(a0, a1, a2) = (1, -1, 0)
```

从第二物理步开始使用等时间步 BDF2：

```text
(a0, a1, a2) = (3/2, -2, 1/2)
```

## 3. 隐式内迭代

每次伪时间修正近似求解

```text
[Gamma(U*)/dtau + a0*M/dt - dR/dU] * deltaU = D(U*)
U* <- U* + implicitRelaxation * deltaU
```

现有 `ACMEvaluator::AssembleImplicitLinearization` 提供
`Gamma/dtau - dR/dU` 的冻结重构、一阶面通量线性化；BDF2 模块只向三个速度分量的单元
对角块加入 `a0/dt`。同一矩阵既可交给 LU-SGS，也可交给 GMRES。边界条件、残差和
CFL 估计均接收目标物理时刻 `t[n+1]`。

`U[n]` 和 `U[n-1]` 在整个内迭代期间保持不变。只有一个物理步结束后才移动历史，避免
伪时间修正污染已经完成的物理时间层。

### 各 RANS 模型的物理时间项

四种 RANS 模型均在统一主状态中保存两个已完成物理层：SA 尾部为 `rho*nuTilde`，
Wilcox/SST 为 `[rho*k, rho*omega]`，realizable k-epsilon 为 `[rho*k, rho*epsilon]`。
每个活跃守恒输运变量的非线性缺陷为：

```text
D_rhoq(U*) = rho0 * R_q(U*, q*, t[n+1])
             - (a0*(rho*q)* + a1*(rho*q)[n] + a2*(rho*q)[n-1]) / dt
```

所有主状态量在同一次非线性内迭代中修正。流场使用原4×4面耦合算子，湍流使用局部
源项Jacobian、对流/扩散谱半径近似及时间对角项，GMRES向量包含完整五/六分量。
流场与闭合的交叉导数冻结，湍流空间非对角Jacobian尚未建立；这是分块近似。
各守恒输运分量独立按rho0乘原始上下限约束增量，k触底不会把omega的修正一起置零。
随后用完整新状态重新求流场和湍流缺陷。按 `dualTimeConvergenceCriterion`
选择Absolute、Relative或AbsoluteOrRelative。当前算例采用OR方式：每个活跃组各自满足绝对或相对条件即可；流场与湍流组仍需分别达标。
历史 `q[n]`、`q[n-1]` 在内迭代中保持不变，
物理步结束后才移动。

BDF2 不保证正性。当某单元某个活跃变量的历史组合
`-a1*q[n]-a2*q[n-1] < a0*minimumValue` 时，该变量在整个物理步中改用 BE 系数
`(1,-1,0)`。选择仅依赖已完成历史，因此不会在内迭代中跳变；其他变量和流场继续使用
原来的系数。这些单元的该变量在
本步采用一阶时间离散；回退较多时需要检查时间步和启动方式，不能宣称全场二阶精度。
收敛统计仍使用完整、未投影的空间减时间缺陷，正值裁剪不会把无法满足的方程伪装成收敛。

统一求解器的历史与主状态同宽，SA不存储第六分量。未收敛物理步不移动历史和物理时间。
公共 `BDF2TurbulenceHistory` 保留用于原始变量时间核测试及旧内部实现；它的SA第二
存储分量固定为零，不参与正值检查、BDF导数和RMS缺陷。

## 4. 配置语义

BDF2 通过 `timeMarchSettings` 选择，可设置：

```json
{
  "integrator": "BDF2DualTimeLUSGS",
  "nSteps": 4000,
  "physicalTimeStep": 0.01,
  "maxImplicitIterations": 20,
  "implicitTolerance": 1e-8,
  "dualTimeConvergenceCriterion": "AbsoluteOrRelative",
  "dualTimeFlowRelativeTolerance": 1e-4,
  "linearAbsoluteTolerance": 1e-10,
  "linearRelativeTolerance": 1e-6
}
```

相关字段含义如下：

- `nSteps`：BDF2 模式下为物理时间步数；
- `physicalTimeStep`：固定物理时间步 `dt`；
- `pseudoTimeStep`：固定内迭代伪时间步 `dtau`；
- `useCFLTimeStep`、`cfl`、`maximumPseudoTimeStep`：只控制 `dtau`；
- `maxImplicitIterations`：每个物理步允许的最大内迭代次数；
- `implicitTolerance`：全局 RMS 物理缺陷收敛阈值；
- `implicitRelaxation`：每次隐式修正的松弛因子。

物理双时间推进默认逐方程计算 MPI 全局 RMS，顺序为 `[u,v,w,p,rho*q1,rho*q2]`，
未活跃分量保持零并排除于收敛判定。新增设置为：

- `dualTimeResidualScales`：六个正的方程特征残差尺度，默认均为 1；
- `dualTimeRelativeTolerances`：各方程相对本物理步初始缺陷的目标，默认均为 0，
  即关闭相对判据；每个值必须位于 `[0,1)`；
- `dualTimeRequireConvergence`：默认 `true`。所选收敛条件未满足就报错退出，且不提交
  物理历史、不更新物理时间、不写该步的已接受场。短跑诊断可显式设为 `false`，
  此时日志仍显示 `converged=0`。

通过 `dualTimeConvergenceCriterion` 明确选择收敛方式：

| JSON 值 | 原始 RMS 目标 |
| --- | --- |
| `Absolute` | 绝对容差；逐方程模式为 `implicitTolerance * dualTimeResidualScales[i]` |
| `Relative` | 本物理步初始残差乘相对阈值；逐方程模式为 `initialRMS[i] * dualTimeRelativeTolerances[i]` |
| `AbsoluteOrRelative` | 以上两个目标的较大值，即绝对或相对条件满足其一即可 |

缺少该字段的旧配置默认使用`Absolute`。`Relative`保留为不使用绝对兜底的严格相对对照；
`AbsoluteOrRelative`采用`max(atol, rtol*本步初值)`，初值为零时仍保留绝对阈值。
在逐方程模式下，每个活跃方程分别通过自己的OR条件；某个方程成功不能绕过其他方程。
这些判据只控制内迭代；应另做加严目标和减半物理步长的精度验证。

当前为流场和SST分别设置四个数量级的相对目标和`1e-8`绝对阈值：

```json
{
  "implicitTolerance": 1e-8,
  "dualTimeConvergenceCriterion": "AbsoluteOrRelative",
  "dualTimeFlowRelativeTolerance": 1e-4,
  "dualTimeTurbulenceRelativeTolerance": 1e-4,
  "steadyRelativeTolerance": 0.0
}
```

`dualTimeFlowRelativeTolerance` 与 `dualTimeTurbulenceRelativeTolerance` 仅用于 BDF2
双时间推进，默认均为 0，表示继续使用对应组的逐方程目标。
正值启用该组的原始 RMS 判据：流场 RMS 包含 `[u,v,w,p]`，湍流 RMS 只包含模型的活跃
守恒输运变量（SST 为 `[rho*k,rho*omega]`，SA 仅为 `rho*nuTilde`）。每个物理步在任何流场或湍流修正之前
分别记录 `R_flow,0` 和 `R_turb,0`；两者在本步内固定，下一物理步重新记录。

```text
AbsoluteOrRelative:
  R_flow <= max(implicitTolerance, dualTimeFlowRelativeTolerance * R_flow,0)
  R_turb <= max(implicitTolerance, dualTimeTurbulenceRelativeTolerance * R_turb,0)
```

上述示例及当前圆柱配置选择`AbsoluteOrRelative`。流场和SST分别达到任一阈值即可。
`1e-4`是相对本物理步初值的四数量级下降，不可用`implicitTolerance=1e-4`替代。
线性修正使用`linearAbsoluteTolerance=1e-10`和`linearRelativeTolerance=1e-6`，
检查乘非线性松弛系数之前的真实`b-Aδ` RMS；不以预条件范数代替真实绝对残差。
`lusgsSweeps`是独立LU-SGS的最大扫掠数，在GMRES预条件中仍是固定扫掠数。
选择 `Relative` 时，每个活跃方程或其 RMS 组都必须配置正的相对阈值；缺少阈值会报错。
层流不检查湍流组，SA 不检查第二个湍流存储分量。任一活跃残差非有限或负值都拒绝收敛。
日志用 `convergenceCriterion=Absolute/Relative` 显示所选方式。
此前的 `Both`、`Either`、`Flow`、`Turbulence` 组选择名称已移除，使用这些值的 JSON
需明确改为 `Absolute` 或 `Relative`。
为避免同一组出现互相覆盖的设置，启用组相对判据时，
该组的 `dualTimeResidualScales` 必须为 1、`dualTimeRelativeTolerances` 必须为 0。
`steadyRelativeTolerance` 仍然只适用于稳态隐式推进，不应在 BDF2 中启用。

这里借鉴 Euler 的相对阈值收敛方式：Euler 的 `rhsThresholdInternal` 在
`EulerSolver_Init.hxx::functor_fstop` 中作用于密度分量的相对残差，并受
`nTimeStepInternalMin` 约束。其参考值在步内取历次残差的最大值；ACM 保留本需求指定的
固定初始残差与现有 RMS 范数。CFV 重构由 Euler 驱动，SOR/GMRES/PCG 使用
`implicitReconstructionControl.recThreshold` 判断重构增量或残差的相对下降，
与物理步内迭代的停止条件分别配置。

BDF2 的固定次数变分重构（`variationalTolerance=0`）现在每次 RHS 从零系数开始，
使缺陷由当前状态唯一决定，不受先前 RHS 或线性化调用的暖启动影响。固定次数仍不保证
重构方程充分收敛；需要严格精度时设置正的 `variationalTolerance`，启用现有方程缺陷
控制。该受控路径可以保留暖启动，达到重构迭代上限时会报错。

为了保持旧算例可读，加载器在内存中为缺少 `physicalTimeStep` 的旧 JSON 补入默认值
`0.01`。这不会回写或改动磁盘上的 JSON 文件。动态 `--emit-schema` 输出会列出新字段和
两种 BDF2 枚举值。

## 5. 输出和日志

层流 BDF2 可选择 `dualTimeConvergenceCriterion="RelativeUpdate"`，以实际施加的松弛后更新量
`||ΔU||₂/||U_new||₂ < dualTimeUpdateRelativeTolerance` 提前退出。范数对所有 MPI 分区的自有单元
及 `[u,v,w,p]` 求和，不含幽灵单元；默认阈值 `1e-5`，最少修正次数由
`dualTimeUpdateMinimumIterations` 设置，默认2。零状态且零更新的比值定义为0，非零更新除以
零状态定义为无穷。阈值为0禁用更新量提前退出，此时需显式允许预算耗尽后推进。
每轮新增 `ACM dual-time inner=...`，同时报告 `updateRatio`、真实 BDF 缺陷 `residual` 和
相对缺陷 `residualRatio`。`updateConverged` 与 `residualConverged` 分别标记两种检查；物理步的
`converged` 对应所选停止判据。更新量模式下的残差诊断仍使用原绝对或相对阈值，更新量达标
不表示方程残差达标。现有 JSON 缺省字段在内存补齐，旧判据不变；该新模式目前限于层流。

BDF2 模式的日志使用 `ACM physical step`，并显示物理时间、当前 BDF 阶数、初末缺陷、
最小伪时间步和内迭代次数。第一个物理步显示 `BDF1`，第二步起显示 `BDF2`。流场输出
文件仍按外层步编号命名，但 VTK 时间序列值使用真实物理时间。
日志增加 `flowRatio`、`turbInitial` 和 `turbRatio`，展示本物理步的流场下降比例、初始湍流
RMS 和湍流下降比例。统一RANS入口的 `eqResidual` 输出五/六个活跃守恒方程RMS，
并显示 `model` 和 `nVars`。层流入口保留其原有目标日志。
`BDF2` 标记描述流场历史系数，不能覆盖有回退单元的湍流变量时间阶数。

## 6. 当前限制

- 只支持固定 `physicalTimeStep`，尚未实现变步长 BDF2 系数；
- 当前允许层流及全部四种 RANS 模型；活跃守恒输运变量保存两个已完成物理时间层，
  隐式系统采用分块近似，尚未建立流场/闭合交叉导数和湍流空间非对角Jacobian；
- 旧配置缺少新增字段时补入严格默认值；曾依靠迭代预算强制推进的短跑，需要显式设置
  `dualTimeRequireConvergence=false`。生产计算不应通过这个选项绕过收敛检查；
- 重启文件和 BDF2 两层历史尚未序列化，重启接续非定常计算前需先补齐该功能。

### SA 与 realizable k-epsilon 的物理时间验证

以下2026-10-08结果属于旧时间核及分离式实现的历史验证，不作为新统一求解器的精度结论。
2026-10-13统一实现已通过九项相关CTest（含np=1/2/4/8和GMRES测试）、Wilcox物理
离散根测试、六组入口短步运行及稳态重启检查。短步运行不代替湍流物理精度验证。

2026-10-08 的时间核测试为 20 个用例、130 个断言，覆盖活跃变量掩码、两层历史、BE 启动、
非法提交不改变历史及全部模型的 BDF2 配置。七项 ACM CTest（含 np=1/2/4 的 MPI 回归）均通过。

`test_ACMSelfPeriodic.cpp` 中的新增非周期测试使用封闭/入口出口单元：SA 关闭模型源项以隔离
对流—扩散输运，验证其半离散指数衰减；realizable k-epsilon 使用零应变、均匀高 Reynolds 数耗散，
验证 `dk/dt=-epsilon` 和 `d(epsilon)/dt=-1.92*epsilon^2/k`。两个模型、两种 BDF2 线性求解器
均与独立推导的 BE/BDF2 离散根一致，且达到内迭代容差。

物理终点固定为 0.32，时间步依次为 0.08、0.04、0.02：

| 模型 | 三个时间步的误差 | 相邻误差比（两种求解器一致） |
| --- | --- | --- |
| SA 半离散输运 | 9.547e-4、2.213e-4、5.296e-5 | 4.315、4.178 |
| realizable k-epsilon 均匀耗散 | 1.317e-4、3.287e-5、8.152e-6 | 4.005、4.032 |

这些结果证明对应测试问题上的二阶物理时间精度。SA 的壁面生产/破坏源项另由启用源项的圆柱
运行检查覆盖；时间阶数测试不代表圆柱升阻力或湍流统计量的物理验证。新增模型的 84 组重构配置
矩阵全部通过，报告位于 `/tmp/acm-bdf2-all-rans-matrix-20261008-165906/report.json`。

原有圆柱网格另用于启用模型源项的运行检查：二维 23,791 单元网格检查全部五种流动模型
与两种 BDF2 的组合，三维 210,656 单元网格检查 SA/k-epsilon 与两种 BDF2 的组合。
每组运行三个物理步，`physicalTimeStep=0.01`、`cfl=0.2`，每步最多五次内迭代；
二维使用两个 MPI 进程，三维使用四个。检查脚本记录内迭代收敛标志，短跑结果仅用于验证
程序执行、历史提交和输出，不能代替充分收敛后的圆柱物理结果。

14/14 组圆柱检查全部通过，退出码均为零，BDF 阶数序列均为 `[1,2,2]`，流场输出有限且
活跃湍流变量保持正值。三维 SA 的最小 `nuTilde` 约为 `9.32e-7`；三维 k-epsilon 的最小
`k` 约为 `9.23e-4`，最小 `epsilon` 约为 `8.89e-4`。这些短跑均未达到内迭代收敛阈值。
完整命令、日志路径和输出范围见 `/tmp/acm-bdf2-all-rans-cylinder-20261008-165905/report.json`。

可从项目根目录用检查脚本复现新增模型的二维圆柱短跑；换成
`BDF2DualTimeLUSGS` 即检查另一条路径：

```bash
HWLOC_COMPONENTS=-gl OMP_NUM_THREADS=1 venv/bin/python scripts/check_acm_rans_cylinder.py \
  --integrator BDF2DualTimeGMRES --models SpalartAllmaras RealizableKEpsilon \
  --steps 3 --physical-time-step 0.01 --max-implicit-iterations 5 --allow-unconverged \
  --viscosity 0.0002564102564102564 --np 2
```

三维检查在上述命令中再加入
`--case cases/acm/3D/acm3D_cylinder_Re3900_coarse_SST.json --limiter LocalExtrema`，
加入 `--dimension 3`，并将 `--np 2` 改为 `--np 4`。脚本从 `build/` 调用对应维度的
层流的 `app/acm2D.exe` / `app/acm3D.exe`、SA的 `app/acmSA2D.exe` / `app/acmSA3D.exe`，
或两方程的 `app/acm2EQ2D.exe` / `app/acm2EQ3D.exe`，
通过命令行覆盖模型和推进参数，并逐步核查 BE/BDF2 阶数及最终输出字段。

## 7. 代码位置

- `src/ACM/ACMBDF2.hpp/.cpp`：BDF 系数、物理质量矩阵、缺陷、对角项和历史管理；
- `src/ACM/ACMRANSSolver.hpp/.hxx`：统一守恒主状态及历史、目标时间传递、完整残差和隐式内迭代；
- `src/ACM/ACMTurbulenceTransport.hpp/.hxx`：原始闭合工作缓冲、守恒面通量、源项线性化及旧内部时间核；
- `src/ACM/ACMSolver.hxx`：四变量流场求解器、LU-SGS/GMRES算子复用；
- `src/ACM/ACMTime.hpp`：JSON 可选的积分格式和 `physicalTimeStep`；
- `src/ACM/ACM.cpp`：旧 JSON 默认值兼容和 BDF2 模型范围检查；
- `test/cpp/ACM/test_ACMTime.cpp`：启动阶数、压力屏蔽、活跃变量历史、缺陷/矩阵一致性和枚举测试；
- `test/cpp/ACM/test_ACMSelfPeriodic.cpp`：RANS 物理时间离散根与 SA/k-epsilon 时间步收敛测试。
