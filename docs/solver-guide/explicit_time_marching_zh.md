# Euler 求解器显式时间推进实现说明

本文说明 DNDSR `EulerSolver` 系列（`euler`、`euler3D`、`eulerSA`、
`euler2EQ` 等）中显式物理时间推进的实际代码路径。这里的“显式时间推进”特指
`timeMarchControl.odeCode = 2` 选择的三阶段 SSP-RK3；ACM 模块拥有另一套独立的
显式伪时间实现，不在本文范围内。

## 1. 总览

空间离散后，求解器把守恒方程写成

\[
\frac{\mathrm d U}{\mathrm d t}=R(U,t),
\]

其中 `U` 是各单元的守恒量数组，`R` 由重构、限制器、边界状态、数值通量、
黏性项和源项共同构成。显式推进使用 Shu--Osher 形式的 SSP-RK3：

\[
\begin{aligned}
U^{(1)} &= U^n + \Delta t R(U^n),\\
U^{(2)} &= \frac34 U^n + \frac14 U^{(1)}
           + \frac14\Delta t R(U^{(1)}),\\
U^{n+1} &= \frac13 U^n + \frac23 U^{(2)}
           + \frac23\Delta t R(U^{(2)}).
\end{aligned}
\]

主要调用链如下：

```text
app/Euler/euler*.cpp
  -> RunSingleBlockConsoleApp()
     -> EulerSolver::ReadMeshAndInitialize()
     -> EulerSolver::RunImplicitEuler()       # 名称是历史遗留，显式模式也走这里
        -> InitializeRunningEnvironment()
           -> odeCode == 2: 构造 ExplicitSSPRK3TimeStepAsImplicitDualTimeStep
        -> 外层物理时间循环
           -> EvaluateDt()                    # 估计谱半径与 CFL 步长
           -> ode->Step()
              -> 3 次 frhs()                  # 重构、限制、通量和源项
              -> 3 次 fincrement()            # 正性约束后更新 U
           -> functor_fmainloop()              # 更新时间、残差、输出和终止判断
```

核心文件：

- `src/Solver/ODE.hpp`：通用 ODE 接口和 SSP-RK3 的三个 stage；
- `src/Euler/EulerSolver.hxx`：积分器选择、时间步外循环，以及传给 ODE 的回调；
- `src/Euler/EulerEvaluator_EvaluateDt.hxx`：CFL/谱半径时间步估计；
- `src/Euler/EulerEvaluator_EvaluateRHS.hxx`：有限体积右端项；
- `src/Euler/EulerSolver_Init.hxx`：步后记账、输出和终止判断；
- `test/cpp/Solver/test_ODE.cpp`：SSP-RK3 三阶精度测试。

## 2. 如何启用

配置入口是 `Configuration::TimeMarchControl`。最小的相关配置为：

```json
{
  "timeMarchControl": {
    "odeCode": 2,
    "dtImplicit": 5.0e-4,
    "dtImplicitMin": 0.0,
    "dtCFLLimitScale": 0.5,
    "nTimeStep": 2000,
    "tEnd": 1.0
  },
  "implicitCFLControl": {
    "CFL": 1.0,
    "useLocalDt": false
  }
}
```

尽管字段和入口函数仍带有 `Implicit`，它们也被显式推进复用。已有示例可见
`cases/euler/3D/euler3D_config_Box.json`。

几个容易混淆的配置项：

| 配置项 | 显式模式中的作用 |
|---|---|
| `odeCode = 2` | 选择 SSP-RK3 |
| `dtImplicit` | 全局物理时间步的上限，通常也是固定时间步 |
| `dtCFLLimitScale` | 将 CFL=1 的全局最小稳定步长缩放为物理步长上限；可把它理解为显式推进的目标 CFL |
| `dtImplicitMin` | 对最终步长施加下限；设置不当可能覆盖 CFL 安全限制 |
| `nTimeStep` | 最大外层时间步数 |
| `tEnd` | 物理时间终点；达到后提前退出 |
| `useDtPPLimit` | 在进入 RK 三阶段之前，进一步用正性条件缩小本步 `dt` |
| `implicitReconstructionControl.useExplicit` | 选择显式二阶重构路径，与是否使用显式时间积分无关 |
| `convergenceControl.nTimeStepInternal` | 传入统一 ODE 接口，但 SSP-RK3 不使用内迭代次数 |

## 3. 一个外层时间步如何确定 `dt`

每个外层 step 开头执行以下逻辑：

1. 令候选值 `curDtImplicit = timeMarchControl.dtImplicit`；
2. 以 `CFLNow = implicitCFLControl.CFL` 调用 `EvaluateDt()`；
3. 每个单元计算

   \[
   \Delta t_i = \min\left(
   \frac{\mathrm{CFLNow}\,V_i\,s_i}
        {\sum_{f\in i}\lambda_f A_f + 10^{-100}},
   \mathrm{MaxDt}\right),
   \]

   其中 `V_i` 是单元体积，`s_i` 是网格平滑尺度比，`lambda_f` 包含对流和
   黏性谱半径；
4. MPI `Allreduce(MIN)` 得到所有进程上的 `curDtMin`；
5. 外层物理步长被限制为

   \[
   \Delta t = \min\left(
   \texttt{dtImplicit},
   \frac{\texttt{curDtMin}}{\texttt{CFLNow}}
   \texttt{dtCFLLimitScale}
   \right);
   \]
6. 可选的正性时间步限制器继续减小 `dt`，随后应用 `dtImplicitMin`；
7. 如果本步会越过下一个按物理时间输出的节点，则把 `dt` 截断到该输出时刻。

这里有一个重要结论：`curDtMin` 本身已经乘过 `CFLNow`，外层又除以
`CFLNow`，所以最终 CFL 上限实际上由 `dtCFLLimitScale` 给出。
`implicitCFLControl.CFL` 在这段物理步长公式中抵消。若
`dtCFLLimitScale` 保持默认的极大值，则显式物理步长主要由 `dtImplicit` 控制，
并不会自动取配置中的 `implicitCFLControl.CFL`。因此显式计算若希望由 CFL 控制，
应显式设置 `dtCFLLimitScale`。

当前 Euler 显式积分器构造时固定传入 `localDtStepping = false`，所以三个 RK stage
都乘同一个全局 `dt`。`EvaluateDt()` 仍会生成单元数组 `dTau`，但它不会参与 stage
增量；`useLocalDt` 也不会使当前 SSP-RK3 变成局部时间推进。代码中的相应注释是
“TODO: add local stepping options”。

## 4. SSP-RK3 三阶段的实际代码

`ExplicitSSPRK3TimeStepAsImplicitDualTimeStep::Step()` 先保存
`xLast = x`，然后顺序执行：

```text
stage 1:
    rhs = R(x)
    x   = x + dt * rhs

stage 2:
    rhs = R(x)
    x   = 1/4 * x + 3/4 * xLast + 1/4 * dt * rhs

stage 3:
    rhs = R(x)
    x   = 2/3 * x + 1/3 * xLast + 2/3 * dt * rhs
```

数组操作与上式一一对应：`operator*=` 完成凸组合系数，`addTo()` 加回
`xLast`，`fincrement()` 加入经过保护的 RHS 增量。每次 `frhs()` 的原始结果还会
分别保存到 `rhsbuf[0..2]`。

该类继承 `ImplicitDualTimeStep` 只是为了让 Euler 主循环对所有积分器使用同一套
接口。显式 `Step()` 中以下参数不会被使用：

- `xinc`；
- `fsolve`（线性系统求解）；
- `maxIter`；
- `fstop`（内迭代收敛判断）。

所以选择 `odeCode = 2` 后，不会执行 LU-SGS/GMRES 时间隐式求解，也没有伪时间
内迭代；成本主要是每个物理步的三次完整 RHS 计算，以及各 stage 中所需的重构。

## 5. 每次 `frhs()` 做什么

ODE 层只认识回调，不依赖 CFD。Euler 层把 `frhs` 绑定到 `frhsOuter`，一次调用的
主要过程是：

1. 拉取/同步当前守恒量的 ghost 数据；
2. 根据当前 stage 的物理时间生成边界值；
3. 修复/过滤单元均值，并按配置更新边界 anchor/profile；
4. 计算变分重构系数，或走 `useExplicit` 指定的直接二阶重构；
5. 执行梯度限制器和重构正性限制器 `beta`；
6. `EvaluateRHS()` 遍历内部面和边界面，计算 Riemann 通量、黏性通量和源项，
   再除以单元体积形成 `dU/dt`；
7. 同步 RHS ghost 数据，并按设置冻结某些被动变量。

`frhs` 的参数 `ct` 被用于边界和源项时间：`tSimu + ct * curDtImplicit`。
当前 SSP-RK3 实现实际传入的三个 `ct` 依次为 `0.5`、`1.0`、`0.25`。
这与经典非自治 SSP-RK3 常写的 stage 时间 `0`、`1`、`1/2` 不同；对于定常边界或
自治方程没有影响，但若边界条件/源项显式依赖时间，应以这组实际取值为准，并在
修改算法时重点核对。

## 6. 每次 `fincrement()` 如何保护解

SSP-RK3 没有直接执行 `x += alpha * increment`。每个 stage 都通过 Euler 层的
`fincrement` 回调：

1. `EvaluateCellRHSAlpha()` 根据当前单元均值和候选增量计算正性缩放系数；
2. 候选增量逐单元乘以该系数；
3. 对两方程湍流模型，可再乘 `RANSRelax`；
4. `AddFixedIncrement()` 调用 `CompressInc()` 做最后的增量压缩并加到解上；
5. `AssertMeanValuePP()` 检查更新后的单元均值满足正性要求。

因此实际执行的是“带逐 stage 正性修正的 SSP-RK3”。一旦限制器被触发，更新不再
严格等同于未修改的线性 SSP-RK3 公式，但换取了密度、压力及相关变量的鲁棒性。

## 7. 步后处理与终止

三个 stage 完成后，`functor_fmainloop()`：

- 执行 `tSimu += curDtImplicit`；若本步为时间输出截断步，则精确设置为输出时刻；
- 用 `ode->getLatestRHS()` 计算和记录残差；
- 累积时间平均量；
- 按步数或物理时间输出流场、边界 profile 和 restart；
- 在 `tSimu >= tEnd` 时终止。外层循环还受 `nTimeStep` 限制。

需要注意，显式类的 `getLatestRHS()` 当前返回 `rhsbuf[0]`，即本步第一个 stage、
在旧解附近计算的 RHS，而不是第三个 stage 的 RHS。“latest”这一名称在显式路径中
容易产生误解；当前残差输出和部分可视化标量使用的就是该值。

## 8. MPI 与数据交换

显式算法没有全局线性求解，但并非完全无全局通信：

- 每个 stage 的解、重构量、限制器量和 RHS 需要 ghost pull；
- `EvaluateDt()` 用 MPI 全局最小归约得到稳定物理步长；
- 正性限制器和监控量可能执行额外归约；
- 所有 rank 必须以相同顺序进入三个 stage 和输出逻辑。

因此其并行同步点比隐式 Krylov 求解少，但每个物理步仍至少包含三次空间算子的
通信开销。

## 9. 验证与调试建议

通用 ODE 单元测试 `test/cpp/Solver/test_ODE.cpp` 用谐振子验证 SSP-RK3 的三阶收敛，
同时包含 golden-value 测试。修改 `src/Solver/ODE.hpp` 后，建议至少构建并运行
`solver_test_ode`。

调试 Euler 显式推进时，建议依次检查：

1. 日志中的 `curDtImplicit` 是否符合预期，而不是只看 `CFLNow`；
2. 是否设置了合理的 `dtCFLLimitScale`；
3. `dtImplicitMin` 是否意外抬高了稳定步长；
4. `nLimInc`、`alphaMinInc`、`nLimBeta` 是否频繁触发；
5. 时间相关边界条件是否接受当前三个 `ct`；
6. 每步三次 RHS 带来的重构和 MPI 通信成本是否符合性能预期。

## 10. 已知命名与实现注意事项

- 主入口名仍是 `RunImplicitEuler()`，但它同时驱动显式和隐式积分器；
- `dtImplicit`、`curDtImplicit` 在显式路径中实际代表物理 `dt`；
- `odeCode = 2` 的日志打印为 `SSPRK4`，而类名、算法系数和单元测试均表明它是
  三阶段三阶 `SSPRK3`；
- 显式类名中的 `AsImplicitDualTimeStep` 表示接口适配，不表示算法是隐式的；
- 当前 Euler SSP-RK3 不支持局部时间步更新；
- 对时间相关问题，stage 时间参数与标准 SSP-RK3 写法不一致，修改前应补充针对
  非自治 ODE 或时间相关边界的回归测试。
