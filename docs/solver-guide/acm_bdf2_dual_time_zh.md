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
松弛 `D_q`。随后以新的湍流黏度重新求流场缺陷。只有流场和湍流两个缺陷都达到
`implicitTolerance` 才提前结束内迭代。历史 `q[n]`、`q[n-1]` 在内迭代中保持不变，
物理步结束后才移动。

公共 `BDF2TurbulenceHistory` 按活跃变量数验证历史；SA 的第二存储分量固定为零，不参与
正值检查、BDF 导数和 RMS 缺陷。提交时先验证整个待提交字段，失败不移动任何历史层。
首步使用 BE，随后使用 BDF2。固定空间离散的 SA 输运衰减和均匀 k-epsilon 非线性耗散
测试分别验证离散解及时间步减半时约四倍的误差缩小。

## 4. 配置语义

BDF2 通过 `timeMarchSettings` 选择，当前算例 JSON 文件未被本次实现修改。以后需要启用
时可设置：

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

为了保持旧算例可读，加载器在内存中为缺少 `physicalTimeStep` 的旧 JSON 补入默认值
`0.01`。这不会回写或改动磁盘上的 JSON 文件。动态 `--emit-schema` 输出会列出新字段和
两种 BDF2 枚举值。

## 5. 输出和日志

BDF2 模式的日志使用 `ACM physical step`，并显示物理时间、当前 BDF 阶数、初末缺陷、
最小伪时间步和内迭代次数。第一个物理步显示 `BDF1`，第二步起显示 `BDF2`。流场输出
文件仍按外层步编号命名，但 VTK 时间序列值使用真实物理时间。

## 6. 当前限制

- 只支持固定 `physicalTimeStep`，尚未实现变步长 BDF2 系数；
- 当前允许层流及全部四种 RANS 模型；活跃湍流变量保存两个已完成物理时间层，
  内迭代采用分离式显式伪时间推进，尚无湍流源项/扩散项的隐式 Jacobian；
- 物理步在达到最大内迭代次数后仍继续推进，使用者应根据日志中的 `converged` 和缺陷
  判断内迭代是否充分收敛；
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
  --steps 3 --physical-time-step 0.01 --max-implicit-iterations 5 \
  --viscosity 0.0002564102564102564 --np 2
```

三维检查在上述命令中再加入
`--case cases/acm3D/acm3D_cylinder_Re3900_coarse_SST.json --limiter LocalExtrema`，
并将 `--np 2` 改为 `--np 4`。脚本从 `build/` 调用已编译的 `app/euler.exe`，
通过命令行覆盖模型和推进参数，并逐步核查 BE/BDF2 阶数及最终输出字段。

## 7. 代码位置

- `src/ACM/ACMBDF2.hpp/.cpp`：BDF 系数、物理质量矩阵、缺陷、对角项和历史管理；
- `src/ACM/ACMTurbulenceTransport.hpp/.hxx`：活跃 RANS 历史及物理缺陷、分离式伪时间子步；
- `src/ACM/ACMSolver.hxx`：物理步循环、目标时间传递、LU-SGS/GMRES 内迭代；
- `src/ACM/ACMTime.hpp`：JSON 可选的积分格式和 `physicalTimeStep`；
- `src/ACM/ACM.cpp`：旧 JSON 默认值兼容和 BDF2 模型范围检查；
- `test/cpp/ACM/test_ACMTime.cpp`：启动阶数、压力屏蔽、活跃变量历史、缺陷/矩阵一致性和枚举测试；
- `test/cpp/ACM/test_ACMSelfPeriodic.cpp`：RANS 物理时间离散根与 SA/k-epsilon 时间步收敛测试。
