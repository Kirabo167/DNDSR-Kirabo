# ACM Wilcox k-omega：时间推进与重构支持

更新日期：2026-10-08。

后续支持范围更新：SA 和 realizable k-epsilon 也已接入按活跃变量管理的 BDF2 物理历史，
当前层流及四种 RANS 均支持两种 BDF2 双时间求解器。下文的运行数值保留为 Wilcox 接入时的验证记录；
当前通用历史与模型范围见 `acm_bdf2_dual_time_zh.md`。

`KOmegaWilcox` 的涡黏度、扩散、生产、耗散、交叉扩散及壁面状态原已实现。本次将其接入原有 BE 启动的 BDF2 双时间路径，并验证与当前合法时间推进及流场重构配置的组合。

## 配置范围

| 配置项 | 支持范围 |
| --- | --- |
| 时间推进 | `ExplicitSSPRK3`、`ImplicitEulerBlockJacobi`、`ImplicitEulerLUSGS`、`ImplicitEulerGMRES`、`BDF2DualTimeLUSGS`、`BDF2DualTimeGMRES` |
| 流场重构 | `FirstOrder`、`GreenGauss`、`Variational` |
| Variational 多项式次数 | `vfvSettings.maxOrder=0,1,2,3`；零次为分片常数，1–3 次使用现有 CFV 多项式 |
| 限制器 | `LocalExtrema`；Variational 可用 `WBAP`、`CWBAP`；可关闭限制器 |
| 重构求解 | 固定迭代次数；按方程缺陷收敛的迭代或 GMRES；允许重置系数及 simultaneous 更新 |
| 时间步 | 固定伪时间步、局部 CFL 步、全局最小 CFL 步；隐式稳态可用相对容差与自适应 CFL |
| GMRES 预条件器 | BlockJacobi、LUSGS |
| 独立湍流场重构 | `secondOrderReconstruction=false` 为一阶面状态；`true` 为受限二阶 Green–Gauss；独立于流场多项式次数 |

RANS 要求 `acmSettings.enableViscousFlux=true` 和正的 `dynamicViscosity`。WBAP/CWBAP 对流场仍要求 Variational。大于三次的多项式不在现有 CFV 基函数范围内，现在会在配置校验中报告错误。

## 代码接入

1. `ACMBDF2.cpp` 的 `SupportsBDF2TurbulenceModel` 统一声明 Laminar、Wilcox、SST 的支持范围，配置校验、运行驱动、历史初始化和物理推进使用同一判断。
2. Wilcox 与 SST 共用原始 `[k,omega]` 的两层已完成物理历史。每次内迭代构造 `R_q-(a0*q+a1*q_n+a2*q_nm1)/dt`，沿用受正值限制的 SSPRK3 伪时间子步；仅在物理步结束时移动历史。
3. 流场仍由原有四变量隐式系统求修正，湍流场分离推进；每个物理步同时判断流场和湍流缺陷。
4. 零次 Variational 原先在 CFV 建矩阵时触发空矩阵断言。现在为度量分配非空线性基，并令流场重构按分片常数计算；保留用户请求的次数标识。用非均匀二维、三维 MPI 流场核对其残差与 `FirstOrder` 一致。

## 验证记录

构建命令为 `cmake --build build -t euler acm_unit_tests -j 4`。ACM 的 core、time、turbulence、self_periodic 及 parallel np=1/2/4 共七项 CTest 通过。时间核为 18 个用例、91 个断言；闭合核为 7 个用例、66 个断言。

封闭对称边界单元中的均匀 Wilcox 衰减满足

$$
\dot k=-0.09k\omega,\qquad \dot\omega=-0.0708\omega^2.
$$

回归测试独立求解各物理步的后退欧拉/BDF2 非线性代数根，再与两种线性求解器的最终结果对照。三物理步、两种湍流重构设置全部通过，误差小于 `1e-10`；物理步的湍流缺陷降到约 `8e-12`，日志为 `converged=1`。测试因此检查了启动阶数、两层历史及物理耗散项，而不仅检查有限值。

配置矩阵共 156 组，最终全部通过。其中 122 组在 192 单元小圆柱网格上运行，34 组在三维验证网格上运行；包含 154 组 Wilcox 与 2 组 SST 回归。各组完成两步、输出字段有限且湍流变量为正，BDF2 的阶数序列为 `[1,2]`。矩阵另覆盖零次重构、湍流一阶、关闭限制器、收敛控制重构、两种预条件器、固定/全局时间步及稳态自适应配置。命令、日志和字段统计见 `/tmp/acm-wilcox-config-matrix-20261008/report.final.json`，重跑前的配置错误日志保留在同目录。

| 原有圆柱网格 | MPI | 路径 | 两物理步结果 |
| --- | --- | --- | --- |
| 二维 23,791 单元 | 2 | BDF2 GMRES、BDF2 LU-SGS | 两者均完成；末步 k 约 `[9.224e-4,1.060e-3]`，omega 约 `[9.863,3145.729]` |
| 三维 210,656 单元 | 4 | BDF2 GMRES | 完成；末步 k 约 `[9.814e-4,1.019e-3]`，omega 约 `[9.986,838.441]` |

原有圆柱短跑只允许每物理步 2–3 次内迭代，均为 `converged=0`。它们验证了网格、MPI、重构和输出链路，不能证明圆柱统计量或物理时间精度。用于研究计算时仍需让两个缺陷充分收敛并做时间步、网格收敛检查。BDF2 仍使用固定物理时间步，尚不支持物理历史重启。

## 复现圆柱短跑

从仓库根目录执行：

```bash
venv/bin/python scripts/check_acm_rans_cylinder.py \
  --models KOmegaWilcox --integrator BDF2DualTimeGMRES \
  --steps 2 --max-implicit-iterations 3 --np 2 \
  --reconstruction Variational --limiter CWBAP --order 2 \
  --viscosity 0.0002564102564102564 \
  --output /tmp/acm-wilcox-new-run
```

脚本使用构建目录中的 `app/euler.exe`，按模型设置湍流初值并打开黏性通量；设置 `HWLOC_COMPONENTS=-gl` 避免无显示节点上的 GL/X11 探测。`--no-turbulence-second-order`、`--disable-limiter`、`--reconstruction-tolerance` 和 `--reconstruction-use-gmres` 可用于核对相应选项。
