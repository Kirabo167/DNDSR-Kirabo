---
title: "ACM 常密度 RANS 实现与 SST 的 BDF2 双时间推进"
author: "DNDSR 开发记录"
date: "2026-09-30"
lang: zh-CN
---

# 目标与适用范围

本文件记录 DNDSR 常密度 ACM 模块中 RANS 闭合的状态组织、空间离散、流场耦合，以及本次为 `KOmegaSST` 增加物理时间 BDF2 双时间推进的实现。`Laminar` 和 `KOmegaSST` 可以选择 `BDF2DualTimeGMRES` 或 `BDF2DualTimeLUSGS`；SA、Wilcox $k$–$\omega$ 与 realizable $k$–$\epsilon$ 仍只用于稳态伪时间推进。

这里的“支持 BDF2”表示代码构造了 SST 两个输运变量的 BE/BDF2 物理时间缺陷，并在每个物理步的内迭代中推进流场和湍流场。物理时间的二阶精度还要求双时间内迭代充分收敛；达到预设迭代上限时继续推进只会得到不完全收敛的物理步。

# 一、常密度平均流与 RANS 变量

流场保持四分量状态 $U=(u,v,w,p)^T$。令 $\rho_0$ 为常密度，$\mu$ 为分子动力黏度，$\mu_t$ 为湍流动力黏度，则稳态目标方程写为

$$
\nabla\cdot\boldsymbol u=0,\qquad
\nabla\cdot(\boldsymbol u\otimes\boldsymbol u)
+\frac{\nabla p^*}{\rho_0}
-\frac{1}{\rho_0}\nabla\cdot\{2(\mu+\mu_t)\boldsymbol S^d\}=0,
$$

其中 $\boldsymbol S^d=(\nabla\boldsymbol u+\nabla\boldsymbol u^T)/2-(\nabla\cdot\boldsymbol u)I/3$。对输运 $k$ 的双方程模型，Reynolds 应力的各向同性部分进入修正压力 $p^*=\bar p+2\rho_0 k/3$。因此流场特征分解、预条件矩阵和隐式块矩阵仍为 $4\times4$，只需在黏性应力及黏性 CFL 中使用 $\mu+\mu_t$。

RANS 原始变量放在独立的分布式双分量场。其统一空间形式是

$$
\partial_t q_a=\mathcal R_a(U,q)
=-\nabla\cdot(\boldsymbol u q_a)
+\nabla\cdot(D_a\nabla q_a)+S_a.
$$

| 配置模型 | 活跃变量 | 闭合与实现要点 |
| --- | --- | --- |
| `Laminar` | 无 | $\mu_t=0$ |
| `SpalartAllmaras` | $\tilde\nu$ | $\mu_t=\rho_0\tilde\nu f_{v1}$；壁面破坏与梯度源项 |
| `KOmegaWilcox` | $k,\omega$ | 应力限制涡黏度、交叉扩散与生产项限制 |
| `KOmegaSST` | $k,\omega$ | $F_1/F_2$ 混合、交叉扩散、壁面距离及涡黏度限制 |
| `RealizableKEpsilon` | $k,\epsilon$ | realizable 涡黏度上限、低 Reynolds 数修正及额外 $\epsilon$ 源项 |

面上采用有限体积对流与扩散通量，单元内加入模型源项。流场与湍流场分别重构并交换 MPI ghost 值。壁面为 $k_w=0$，SST 的 $\omega_w=C_\omega\nu/d^2$（默认 $C_\omega=800$）；远场湍流值仅在局部入流时施加，出流为零法向梯度。每次流场残差求值前由当前 SST 状态计算并冻结面涡黏度。

# 二、SST 闭合的关键公式

SST 使用 $q=(k,\omega)$、$\nu=\mu/\rho_0$、$\beta^*=0.09$、$a_1=0.31$、壁面距离 $d$ 和涡量模 $|\Omega|$。本实现采用涡量型应力限制：

$$
\mu_t=\rho_0\frac{a_1 k}{\max(a_1\omega,|\Omega|F_2)}.
$$

由 $F_1$ 混合近壁 $k$–$\omega$ 系数与远壁 $k$–$\epsilon$ 转换系数。两个原始变量的扩散系数为

$$
D_k=\frac{\mu+\sigma_k\mu_t}{\rho_0},\qquad
D_\omega=\frac{\mu+\sigma_\omega\mu_t}{\rho_0}.
$$

令 $S^2=2S^d_{ij}S^d_{ij}$，$P_k=\max(\mu_t S^2/\rho_0-2k\nabla\cdot\boldsymbol u/3,0)$，则源项为

$$
S_k=\min(P_k,20\beta^*k\omega)-\beta^*k\omega,
$$

$$
S_\omega=\gamma S^2-\beta\omega^2
+2(1-F_1)\sigma_{\omega2}\frac{\nabla k\cdot\nabla\omega}{\omega}.
$$

$F_1,F_2$ 的具体参数、分母下限和正值约束见 `src/ACM/ACMTurbulence.cpp`；空间通量和 SSPRK3 实现在 `src/ACM/ACMTurbulenceTransport.hxx`。分子黏度、密度和湍流变量均按 SI 或自洽无量纲单位输入；常密度原始变量输运中的动力黏度通量须除以 $\rho_0$。

## 其他闭合在同一输运框架中的实现

SA 的单个未知量为修正运动黏度 $\tilde\nu$。代码用 $\chi=\tilde\nu/\nu$ 和阻尼函数 $f_{v1}$ 得到 $\mu_t=\rho_0\tilde\nu f_{v1}$；扩散系数为 $(\nu+\tilde\nu)/\sigma$，$\sigma=2/3$。源项包含修正涡量乘 $\tilde\nu$ 的生产、与 $(\tilde\nu/d)^2$ 成正比的壁面破坏，以及 $(c_{b2}/\sigma)|\nabla\tilde\nu|^2$。壁面规定 $\tilde\nu=0$。

Wilcox $k$–$\omega$ 使用应变率限制的 $\mu_t=\rho_0 k/\max(\omega,c_{\mathrm{lim}}|S|/\sqrt{\beta^*})$，其中 $c_{\mathrm{lim}}=7/8$。两变量的扩散系数均为 $(\mu+0.5\mu_t)/\rho_0$。$k$ 方程采用有限生产项与 $\beta^*k\omega$ 破坏项；$\omega$ 方程包含 $\omega P_k/k$ 型生产、$\beta\omega^2$ 破坏和仅在 $\nabla k\cdot\nabla\omega>0$ 时启用的交叉扩散。壁面同样设置 $k=0$ 与 $\omega=C_\omega\nu/d^2$。

Realizable $k$–$\epsilon$ 使用含低 Reynolds 数阻尼 $f_\mu$ 的 $\rho_0 C_\mu f_\mu k^2/\epsilon$，再施加可实现性与配置上限。扩散系数分别是 $(\mu+\mu_t/\sigma_k)/\rho_0$ 和 $(\mu+\mu_t/\sigma_\epsilon)/\rho_0$。源项采用 $S_k=P_k-\epsilon$，并将 $\epsilon$ 的生产、破坏和梯度修正 $E$ 按湍流时间尺度 $T_t$ 缩放。壁面 $k=0$，$\epsilon$ 按当前近壁 $k$ 和 $\nu/d^2$ 给定。以上三种模型目前保留稳态伪时间推进；本次 BDF2 物理历史只接入 SST。

# 三、BE 启动与 BDF2 物理时间缺陷

令 $\Delta t$ 为固定物理时间步，$n+1$ 为待求物理层。首步使用后向欧拉系数 $(a_0,a_1,a_2)=(1,-1,0)$，从第二步起使用 BDF2 系数 $(3/2,-2,1/2)$：

$$
\mathcal D_t x^{n+1}
=\frac{a_0x^{n+1}+a_1x^n+a_2x^{n-1}}{\Delta t}.
$$

流场的物理时间质量矩阵为 $M_U=\operatorname{diag}(1,1,1,0)$：速度有物理时间导数，人工可压缩性压力没有。流场缺陷与 SST 缺陷分别为

$$
\mathcal F_U=\mathcal R_U(U^{n+1},q^{n+1})
-M_U\mathcal D_t U^{n+1},
$$

$$
\mathcal F_q=\mathcal R_q(U^{n+1},q^{n+1})
-\mathcal D_t q^{n+1},\qquad q=(k,\omega)^T.
$$

$\rho_0$ 恒定，故输运 $\rho_0 k,\rho_0\omega$ 与输运原始 $k,\omega$ 的时间离散在除以 $\rho_0$ 后一致。首步的两个历史层都初始化为初始场，但 $a_2=0$；只有完成整个物理步后才将 $q^n$ 移到第二历史层并提交 $q^{n+1}$。内迭代不修改已完成的历史层。

# 四、双时间内迭代与代码接入

每个物理步执行如下过程：

1. 使用流场和 SST 历史取得当前 BE/BDF2 系数，设置目标物理时间，并依据当前涡黏度计算伪时间 CFL。
2. 求流场缺陷 $\mathcal F_U$，组装原有的四变量冻结重构隐式矩阵，并在速度对角上加入 $a_0M_U/\Delta t$。GMRES 或 LU-SGS 解出流场修正。
3. 用修正后的流场评估 SST 空间残差，减去 $\mathcal D_t q$；采用带正值界限的分离式 SSPRK3 伪时间子步更新 $k,\omega$。
4. 用新的 SST 涡黏度重算流场缺陷，同时计算 SST 物理时间缺陷。只有两个缺陷都达到 `implicitTolerance` 才提前结束内迭代。
5. 完成物理步后提交流场与 SST 历史。即使达到 `maxImplicitIterations` 而没有收敛，现有驱动仍会提交该物理步，日志中的 `converged=0` 与 `turbResidual` 必须一起检查。

主要代码位置如下：

| 文件 | 本次职责 |
| --- | --- |
| `src/ACM/ACMBDF2.hpp/.cpp` | SST 原始变量的 BE/BDF2 物理时间导数 |
| `src/ACM/ACMTurbulenceTransport.hpp/.hxx` | SST 两级物理历史、时间缺陷、伪时间子步与提交 |
| `src/ACM/ACMSolver.hxx` | 将 SST 更新插入每次流场双时间内迭代；同时判断两个缺陷 |
| `src/ACM/ACM.cpp` | 允许 `Laminar` 和 `KOmegaSST`，拒绝其他 BDF2 RANS 选择 |
| `test/cpp/ACM/test_ACMTime.cpp` | 系数、配置范围与启动阶数回归 |

SST 的物理时间项已加入，但空间扩散和化学生产/耗散项没有构造隐式 Jacobian；湍流伪时间更新仍为显式分离式，因此需要设置足够的内迭代次数和适当的湍流子步参数。

# 五、配置与使用

基于已有圆柱算例，将 `turbulenceSettings.model` 设为 `KOmegaSST`，打开黏性通量并保证 `dynamicViscosity>0`。`timeMarchSettings` 选择 `BDF2DualTimeGMRES` 或 `BDF2DualTimeLUSGS`，设置正的 `physicalTimeStep`、`maxImplicitIterations` 和收敛容差。示意：

```json
"timeMarchSettings": {
  "integrator": "BDF2DualTimeGMRES",
  "nSteps": 100,
  "physicalTimeStep": 0.01,
  "maxImplicitIterations": 50,
  "implicitTolerance": 1e-8
},
"turbulenceSettings": {
  "model": "KOmegaSST",
  "initialValue": [0.001, 10.0],
  "farFieldValue": [0.001, 10.0]
}
```

实际 JSON 还须包含网格、边界及完整的 ACM 配置；示意片段不能独立加载。当前 BDF2 模式不接受 VTK-HDF 重启，因为流场与 SST 的两个物理时间历史都尚未序列化。

```{=openxml}
<w:p><w:r><w:br w:type="page"/></w:r></w:p>
```

# 六、验证与限制

针对 SST BE/BDF2 物理时间导数与两种积分器配置范围，`acm_test_time` 的 17 个用例、89 个断言全部通过。圆柱短跑使用 2 个 MPI rank、两个物理步；各自的日志均显示 `BDF1` 后接 `BDF2`，最终 VTK-HDF 中的压力、速度、$k$ 和 $\omega$ 均有限，$k,\omega$ 保持正值。层流 BDF2 GMRES 两物理步回归也通过。

| 二维圆柱、Re=3900 物性 | 第一步流场缺陷 | 第二步流场缺陷 | 第一步/第二步 SST 缺陷 |
| --- | --- | --- | --- |
| GMRES，最多 10 次内迭代 | 26.26 → 11.56 | 13.65 → 6.842 | $6.396\times10^5$ / $2.228\times10^5$ |
| LU-SGS，相同设置 | 26.26 → 11.56 | 13.65 → 6.842 | $6.396\times10^5$ / $2.228\times10^5$ |

两种路径的第二步湍流缺陷均低于第一步，但都未达到 `implicitTolerance`，日志显示 `converged=0`。短跑只能说明 BE/BDF2 路径、MPI 通信和输出可执行，不能证明时间精度。机器可读报告保存在 `/tmp/acm2d-sst-bdf2-re3900-{gmres,lusgs}-20260930/`。

原有三维 Re=3900 SST 圆柱网格（210,656 单元）在 4 个 MPI rank 上也完成了 GMRES 的两个物理步。第二步流场缺陷为 35.45 → 29.27，SST 缺陷为 $1.557\times10^6$；最终 $k$ 约为 $[9.85\times10^{-4},1.02\times10^{-3}]$，$\omega$ 约为 $[9.99,836.89]$，均有限且为正。每步仅允许 2 次内迭代，`converged=0`；结果只证明三维运行链路。

模型当前仍有限制：固定物理时间步；SST 分离式显式伪时间子迭代；内迭代达到上限后允许以 `converged=0` 提交物理步；不支持 BDF2 重启；不支持 SA、Wilcox 和 realizable $k$–$\epsilon$ 的物理时间 BDF2。输出有限值并不等于物理预测准确，需另行验证壁面分辨率、物理时间步收敛与升阻力统计。
