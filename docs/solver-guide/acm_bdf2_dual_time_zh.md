# ACM BDF2 双时间推进说明

## 1. 实现范围

ACM 求解器现在支持定常伪时间推进和非定常 BDF2 双时间推进。BDF2 模块参考了 Euler
求解器的时间历史管理和隐式线性求解组织方式，并采用所给 FORTRAN 程序中的 ACM
物理时间方程。它没有直接复用 Euler 的通用 BDF 状态方程，因为 ACM 压力只参与伪时间
人工压缩过程，不应具有物理时间导数。

可选格式为：

- `BDF2DualTimeLUSGS`：内迭代直接使用分布式 ACM LU-SGS；
- `BDF2DualTimeGMRES`：内迭代使用左预条件 GMRES，预条件器仍由
  `gmresPreconditioner` 选择。

## 2. 控制方程

ACM 状态和物理状态分别定义为

```text
U = [u, v, w, p]^T
Q = M U,   M = diag(1, 1, 1, 0)
```

双时间方程写成

```text
Gamma(U) * dU/dtau + dQ/dt = R(U, t)
```

其中 `tau` 是用于收敛每个物理步的伪时间，`t` 是真实物理时间，`R` 是现有空间离散
残差。矩阵 `M` 的最后一个对角元为零，因此速度采用 BDF 物理时间离散，压力没有物理
时间导数。

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

四种 RANS 模型均保存原始活跃变量的两个已完成物理层：SA 为 `nuTilde`，Wilcox/SST 为
`[k, omega]`，realizable k-epsilon 为 `[k, epsilon]`。常密度下 `rho0` 在时间项中约去，
每个活跃变量的非线性缺陷为：

```text
D_q(q*) = R_q(U*, q*, t[n+1])
          - (a0*q* + a1*q[n] + a2*q[n-1]) / dt
```

每次流场隐式修正后，使用更新后的速度冻结流场，采用有正值界限的 SSPRK3 子步沿伪时间
松弛 `D_q`。每个活跃变量独立按正值上下限约束其阶段增量，k 触底不会把 omega 的阶段
步长一起置零。随后以新的湍流黏度重新求流场缺陷。按 `dualTimeConvergenceCriterion`
选择流场、湍流或两组的组合条件，满足所选条件才结束内迭代。
历史 `q[n]`、`q[n-1]` 在内迭代中保持不变，
物理步结束后才移动。

BDF2 不保证正性。当某单元某个活跃变量的历史组合
`-a1*q[n]-a2*q[n-1] < a0*minimumValue` 时，该变量在整个物理步中改用 BE 系数
`(1,-1,0)`。选择仅依赖已完成历史，因此不会在内迭代中跳变；其他变量和流场继续使用
原来的系数。日志 `turbBE=[q1单元数,q2单元数]` 报告这种局部回退。这些单元的该变量在
本步采用一阶时间离散；回退较多时需要检查时间步和启动方式，不能宣称全场二阶精度。
收敛统计仍使用完整、未投影的空间减时间缺陷，正值裁剪不会把无法满足的方程伪装成收敛。

公共 `BDF2TurbulenceHistory` 按活跃变量数验证历史；SA 的第二存储分量固定为零，不参与
正值检查、BDF 导数和 RMS 缺陷。提交时先验证整个待提交字段，失败不移动任何历史层。
首步使用 BE，随后使用 BDF2。固定空间离散的 SA 输运衰减和均匀 k-epsilon 非线性耗散
测试分别验证离散解及时间步减半时约四倍的误差缩小。

## 4. 配置语义

BDF2 通过 `timeMarchSettings` 选择，可设置：

```json
{
  "integrator": "BDF2DualTimeLUSGS",
  "nSteps": 4000,
  "physicalTimeStep": 0.01,
  "maxImplicitIterations": 20,
  "implicitTolerance": 1e-10
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

物理双时间推进默认逐方程计算 MPI 全局 RMS，顺序为 `[u,v,w,p,q1,q2]`，
未活跃分量保持零并排除于收敛判定。新增设置为：

- `dualTimeResidualScales`：六个正的方程特征残差尺度，默认均为 1；
- `dualTimeRelativeTolerances`：各方程相对本物理步初始缺陷的目标，默认均为 0，
  即关闭相对判据；每个值必须位于 `[0,1)`；
- `dualTimeRequireConvergence`：默认 `true`。任一方程未收敛就报错退出，且不提交
  物理历史、不更新物理时间、不写该步的已接受场。短跑诊断可显式设为 `false`，
  此时日志仍显示 `converged=0`。

每个活跃方程的原始 RMS 目标是
`max(implicitTolerance*dualTimeResidualScales[i], initialRMS[i]*dualTimeRelativeTolerances[i])`。
因此可以用具有物理依据的尺度区分速度、压力、k 和 omega，或指定各方程的残差下降比例。
默认尺度为 1、组合方式为 `Both` 时仍采用原容差，所有活跃方程都必须通过，
不再允许某一分量被混合 RMS 稀释。其他组合方式仍要求所选组内的每个活跃方程通过。
这些判据只控制内迭代；应另做加严目标和减半物理步长的精度验证。

如果希望为流场和 SST 分别设置三个数量级的下降目标，并在任一组达标时结束内迭代，
在 `timeMarchSettings` 中设置：

```json
{
  "implicitTolerance": 1e-8,
  "dualTimeConvergenceCriterion": "Either",
  "dualTimeFlowRelativeTolerance": 1e-3,
  "dualTimeTurbulenceRelativeTolerance": 1e-3,
  "steadyRelativeTolerance": 0.0
}
```

两个新字段仅用于 BDF2 双时间推进，默认均为 0，表示继续使用对应组的逐方程目标。
正值启用该组的原始 RMS 判据：流场 RMS 包含 `[u,v,w,p]`，湍流 RMS 只包含模型的活跃
输运变量（SST 为 `[k,omega]`，SA 仅为 `nuTilde`）。每个物理步在任何流场或湍流修正之前
分别记录 `R_flow,0` 和 `R_turb,0`；两者在本步内固定，下一物理步重新记录。

```text
R_flow <= max(implicitTolerance, dualTimeFlowRelativeTolerance * R_flow,0)
R_turb <= max(implicitTolerance, dualTimeTurbulenceRelativeTolerance * R_turb,0)
```

通过 `dualTimeConvergenceCriterion` 选择使用哪组收敛判据：

| JSON 值 | 结束内迭代的条件 |
| --- | --- |
| `Both` | 流场和湍流两组都通过；旧配置缺省值 |
| `Either` | 流场或湍流任一活跃组通过 |
| `Flow` | 只判断流场 |
| `Turbulence` | 只判断湍流（SST 或其他活跃 RANS 模型） |

上述示例及圆柱配置选择 `Either`。各组仍采用绝对条件或自身相对条件，选择组的方式不会
改变组内目标；相对字段为 0 时也按同样选择组合对应组的逐方程判据。
`1e-3` 表示相对本物理步初值
下降三个数量级，不能用 `implicitTolerance=1e-3` 代替。某组初始 RMS 为零或很小时，
绝对容差仍然有效；层流的 `Both`、`Either`、`Flow` 均只判断流场，不能由不存在的
湍流组满足 `Either`，选择 `Turbulence` 会报配置错误。任一活跃组存在非有限或负残差时
仍拒绝收敛，即使未选择该组。日志用 `convergenceCriterion` 显示所选方式。
为避免同一组出现互相覆盖的设置，启用组相对判据时，
该组的 `dualTimeResidualScales` 必须为 1、`dualTimeRelativeTolerances` 必须为 0。
`steadyRelativeTolerance` 仍然只适用于稳态隐式推进，不应在 BDF2 中启用。

BDF2 的固定次数变分重构（`variationalTolerance=0`）现在每次 RHS 从零系数开始，
使缺陷由当前状态唯一决定，不受先前 RHS 或线性化调用的暖启动影响。固定次数仍不保证
重构方程充分收敛；需要严格精度时设置正的 `variationalTolerance`，启用现有方程缺陷
控制。该受控路径可以保留暖启动，达到重构迭代上限时会报错。

为了保持旧算例可读，加载器在内存中为缺少 `physicalTimeStep` 的旧 JSON 补入默认值
`0.01`。这不会回写或改动磁盘上的 JSON 文件。动态 `--emit-schema` 输出会列出新字段和
两种 BDF2 枚举值。

## 5. 输出和日志

BDF2 模式的日志使用 `ACM physical step`，并显示物理时间、当前 BDF 阶数、初末缺陷、
最小伪时间步和内迭代次数。第一个物理步显示 `BDF1`，第二步起显示 `BDF2`。流场输出
文件仍按外层步编号命名，但 VTK 时间序列值使用真实物理时间。
日志增加 `flowRatio`、`turbInitial` 和 `turbRatio`，展示本物理步的流场下降比例、初始湍流
RMS 和湍流下降比例。开启组判据时，`flowTarget` / `turbTarget` 输出本步实际组目标；
未开启组判据时，`eqTarget` 或混合模式下的 `flowEqTarget` / `turbEqTarget` 输出对应的
逐方程目标。`eqResidual` 始终输出六个原始方程 RMS；`turbBE` 报告湍流的
局部 BE 回退数。`BDF2` 标记描述流场历史系数，不能覆盖有回退单元的湍流变量时间阶数。

## 6. 当前限制

- 只支持固定 `physicalTimeStep`，尚未实现变步长 BDF2 系数；
- 当前允许层流及全部四种 RANS 模型；活跃湍流变量保存两个已完成物理时间层，
  内迭代采用分离式显式伪时间推进，尚无湍流源项/扩散项的隐式 Jacobian；
- 旧配置缺少新增字段时补入严格默认值；曾依靠迭代预算强制推进的短跑，需要显式设置
  `dualTimeRequireConvergence=false`。生产计算不应通过这个选项绕过收敛检查；
- 重启文件和 BDF2 两层历史尚未序列化，重启接续非定常计算前需先补齐该功能。

### SA 与 realizable k-epsilon 的物理时间验证

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
`app/acm2D.exe` 或 `app/acm3D.exe`，
通过命令行覆盖模型和推进参数，并逐步核查 BE/BDF2 阶数及最终输出字段。

## 7. 代码位置

- `src/ACM/ACMBDF2.hpp/.cpp`：BDF 系数、物理质量矩阵、缺陷、对角项和历史管理；
- `src/ACM/ACMTurbulenceTransport.hpp/.hxx`：活跃 RANS 历史及物理缺陷、分离式伪时间子步；
- `src/ACM/ACMSolver.hxx`：物理步循环、目标时间传递、LU-SGS/GMRES 内迭代；
- `src/ACM/ACMTime.hpp`：JSON 可选的积分格式和 `physicalTimeStep`；
- `src/ACM/ACM.cpp`：旧 JSON 默认值兼容和 BDF2 模型范围检查；
- `test/cpp/ACM/test_ACMTime.cpp`：启动阶数、压力屏蔽、活跃变量历史、缺陷/矩阵一致性和枚举测试；
- `test/cpp/ACM/test_ACMSelfPeriodic.cpp`：RANS 物理时间离散根与 SA/k-epsilon 时间步收敛测试。
