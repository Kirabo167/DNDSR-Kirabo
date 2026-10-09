---
title: "DNDSR 可压缩 RANS 与 ACM 常密度 RANS 实现对比"
subtitle: "控制方程、状态组织、SST 闭合、离散与时间推进"
author: "DNDSR 代码对照记录"
date: "2026-10-08"
lang: zh-CN
---

# 范围与记号

本文对照当前仓库的 `src/Euler/` 可压缩 Navier–Stokes/RANS 求解器和 `src/ACM/` 常密度人工可压缩性求解器。重点是**实际代码路径**，而非把两套实现都假定为某篇文献中的标准模型。可压缩侧以固定变量数的 `NS_SA`、`NS_2EQ`（含三维版本）为主；`NS_EX` 的反应流和动态变量扩展只在相关处说明。常密度侧的配置名为 `SpalartAllmaras`、`KOmegaWilcox`、`KOmegaSST` 和 `RealizableKEpsilon`。

记 $\bar\rho$ 为平均密度，$\rho_0$ 为 ACM 常数密度，$\mu$ 为分子动力黏度，$\mu_t$ 为湍流动力黏度，$k$ 为比湍动能，$\omega$ 为比耗散率，$\epsilon$ 为湍动能耗散率。偏应变率为

$$
S^d_{ij}=\tfrac12(\partial_j u_i+\partial_i u_j)-\tfrac13(\nabla\cdot\boldsymbol u)\delta_{ij}.
$$

# 一、平均方式与控制方程

## 1.1 可压缩侧：密度加权平均与守恒变量

密度波动不可忽略时，速度通常采用 Favre 平均 $\tilde u_i=\overline{\rho u_i}/\bar\rho$。只写本文关注的平均方程，其物理结构为

$$
\partial_t\bar\rho+\nabla\cdot(\bar\rho\tilde{\boldsymbol u})=0,
$$

$$
\partial_t(\bar\rho\tilde u_i)
+\partial_j\!\left(\bar\rho\tilde u_i\tilde u_j+p\delta_{ij}
+\tfrac23\bar\rho k\delta_{ij}-2(\mu+\mu_t)S^d_{ij}\right)=0.
$$

代码还求解总能量 $E$，用状态与物性得到热力学压力、温度、声速及分子输运系数。理想气体的基本关系为 $p=(\gamma_{\mathrm{eq}}-1)[E-\rho|\boldsymbol u|^2/2-\rho E_{\mathrm{base}}]$；存在反应组分时，$E_{\mathrm{base}}$ 和物性闭合还依赖组分。能量黏性通量包含 $\boldsymbol\tau\cdot\boldsymbol u+\lambda_{\mathrm{eff}}\nabla T$，其中 $\lambda_{\mathrm{eff}}=\lambda_{\mathrm{mol}}+C_p\mu_t/Pr_t$。这里代码以“总通量减黏性通量”的符号约定装配。[E1][E2][E3]

`NS_2EQ` 的单元守恒状态按顺序为 $[\rho,\rho u,\rho v,\rho w,E,\rho k,\rho\omega]$；选择 realizable $k$–$\epsilon$ 时末项改为 $\rho\epsilon$。`NS_SA` 比基本 NS 多一个 SA 输运分量。`NS_2EQ` 的二维网格版本仍保留三个速度分量；`NS_2D` 才是仅两个速度分量的四变量基本模型。状态长度进入同一重构、面通量、残差和隐式线性求解流程。[E4]

## 1.2 ACM 侧：恒定密度、速度—压力状态及伪时间

令 $\rho=\rho_0$ 并施加 $\nabla\cdot\boldsymbol u=0$ 后，连续性不再是密度演化方程。ACM 的四变量场为 $U=[u,v,w,p]^T$，用人工可压缩性伪时间消除不可压压力约束：

$$
\Gamma(U)\,\partial_\tau U
=\mathcal R_U(U,q)-M\,\partial_t U,
\qquad M=\operatorname{diag}(1,1,1,0).
$$

$\mathcal R_U$ 包含速度对流、压力梯度及 $2(\mu+\mu_t)S^d$ 的散度。$\Gamma$ 是伪时间预条件矩阵；物理时间中只有速度有质量项，压力没有。四变量面通量的连续性分量为法向速度，动量分量含 $p/\rho_0$；没有密度、总能量、温度或声速方程。[A1][A2][A3]

ACM 的湍流原始变量存在**独立的两分量分布式场**：SA 用 $[\tilde\nu,\text{未用}]$，两方程模型用 $[k,\omega]$ 或 $[k,\epsilon]$。因此流场的特征分解和隐式块仍为 $4\times4$；湍流场通过 $\mu_t$ 与动量方程双向耦合。[A4][A5]

## 1.3 各向同性 $k$ 应力与压力的差别

Boussinesq 闭合的雷诺应力为

$$
-\bar\rho\,\widetilde{u_i''u_j''}
=2\mu_tS^d_{ij}-\tfrac23\bar\rho k\delta_{ij}.
$$

Euler 的双方程数值通量在动量法向分量显式加 $2\rho k/3$，并在能量通量加对应的速度做功。因此 Euler 状态中的 $p$ 仍由热力学关系确定。[E5] ACM 通量没有独立的 $k$ 各向同性项；若采用同一闭合推导，其存储的压力应解释为 $p^*=\bar p+2\rho_0k/3$。需要平均静压时，应后处理 $\bar p=p^*-2\rho_0k/3$。ACM 配置和输出仍使用 `PhysicalP`/`Pressure` 名称，不能仅凭该名称判断已做上述还原。[A1][A6]

# 二、湍流输运变量为何不同

## 2.1 从可压缩守恒形式推到常密度原始形式

对任一湍流比变量 $q\in\{\tilde\nu,k,\omega,\epsilon\}$，可压缩侧的结构可写成

$$
\partial_t(\rho q)+\nabla\cdot(\rho\boldsymbol u q)
=\nabla\cdot(\mu_q\nabla q)+S_q^{\mathrm{cons}}.
\tag{1}
$$

利用乘积法则，左端等于

$$
\rho(\partial_tq+\boldsymbol u\cdot\nabla q)
+q[\partial_t\rho+\nabla\cdot(\rho\boldsymbol u)].
$$

第二项由连续性方程消去。若进一步令 $\rho=\rho_0$，则 $\nabla\cdot\boldsymbol u=0$，式 (1) 除以 $\rho_0$ 后成为

$$
\partial_tq+\nabla\cdot(\boldsymbol u q)
=\nabla\cdot\!\left(\frac{\mu_q}{\rho_0}\nabla q\right)
+\frac{S_q^{\mathrm{cons}}}{\rho_0}.
\tag{2}
$$

这正是 ACM 湍流场的方程结构。**转换时必须同时处理三件事**：把 $\rho q$ 转为 $q$，把动力黏度型扩散系数除以 $\rho_0$，把单位体积源项除以 $\rho_0$。例如 SST 的 $k$ 扩散，Euler 为 $(\mu+\sigma_k\mu_t)\nabla k$，ACM 为 $D_k\nabla k$，其中 $D_k=(\mu+\sigma_k\mu_t)/\rho_0$。[E6][A7]

可压缩流在密度变化时，$\nabla\cdot(\boldsymbol u q)$ 与 $\boldsymbol u\cdot\nabla q$ 不等价；只有常密度不可压条件下才可直接互换。Euler 还要对守恒梯度使用商法则，得到 $\nabla k=[\nabla(\rho k)-k\nabla\rho]/\rho$；ACM 直接重构 $k$、$\omega$ 等原始变量。[E2][E6][A7]

# 三、同名 RANS 闭合的代码对照

| 项目 | Euler 可压缩实现 | ACM 常密度实现 |
|:---|:---|:---|
| 模型选择 | 编译期 `NS_SA`/`NS_2EQ` 等确定变量数，运行期 `ransModel` 在两方程模型间选 Wilcox/SST/RKE；`NS_2EQ` 默认 Wilcox。 | 运行期 `turbulenceSettings.model`；默认 `Laminar`。 |
| 分子黏度 | 从温度、压力、组成和物性模型求得；可变。 | `acmSettings.dynamicViscosity`，配置的常数。 |
| 涡黏度 | 从守恒状态恢复原始 $k,\omega,\epsilon$，结合速度梯度、壁距计算。 | 直接用独立原始湍流场和速度梯度、壁距计算。 |
| SA | SA 输运分量并入流场守恒数组；支持负 $\tilde\nu$ 扩展及 SA-DES/旋转修正等配置路径。 | 独立输运 $\tilde\nu$；实现生产、壁面破坏、梯度项及正值限制；无上述 DES 路径。 |
| Wilcox $k$–$\omega$ | $\mu_t=\rho k/\omega_{\mathrm{lim}}$，扩散、生产/耗散和交叉扩散在 `RANS_ke.hpp`。 | 相同类型的应变限制涡黏度与两条原始变量输运方程，系数按 ACM 代码定义。 |
| SST $k$–$\omega$ | $F_1/F_2$、$a_1=0.31$ 的涡黏度限制；另有可选 DES 长度尺度、源项近似 Jacobian。 | $F_1/F_2$ 和相同结构的涡黏度限制；原始变量源项和分离式输运；无 DES 长度尺度。 |
| Realizable $k$–$\epsilon$ | $\rho k,\rho\epsilon$ 守恒输运与涡黏度上限。 | $k,\epsilon$ 原始输运、低 Reynolds 数修正及可实现性上限。 |
| 黏性/热通量 | 动量用 $\mu+\mu_t$；总能量用 $\lambda_{\mathrm{mol}}+C_p\mu_t/Pr_t$；反应流另有组分扩散。 | 动量用 $\mu+\mu_t$；无总能量和湍流热通量方程。 |

两套 SST 的主涡黏度形式一致：

$$
\mu_t=\rho\,\frac{a_1k}{\max(a_1\omega,|\Omega|F_2)},
\qquad a_1=0.31,
$$

但 Euler 的 $\rho$ 与 $\mu$ 随状态改变，ACM 用 $\rho_0$ 和固定 $\mu$。两套实现还各自对 $k,\omega,d,\mu_t/\mu$ 施加数值上下限，所以同一名义模型不保证逐面得到完全相同的 $\mu_t$。[E6][A8]

SST 扩散与耗散的维度可作为核对点：

$$
\begin{aligned}
&\text{Euler:}\quad \partial_t(\rho k)+\nabla\cdot(\rho\boldsymbol u k)
=\nabla\cdot[(\mu+\sigma_k\mu_t)\nabla k]
+P_k^{\mathrm{cons}}-\beta^*\rho k\omega,\\
&\text{ACM:}\quad \partial_t k+\nabla\cdot(\boldsymbol u k)
=\nabla\cdot\!\left[\frac{\mu+\sigma_k\mu_t}{\rho_0}\nabla k\right]
+P_k^{\mathrm{cons}}/\rho_0-\beta^*k\omega.
\end{aligned}
$$

Euler 的 $\omega$ 源项有 $\rho$ 乘子，ACM 的相应源项无该乘子；交叉扩散同理。代码层面，Euler 对双方程源项提供破坏项的对角线性化近似，ACM 湍流输运使用分离的显式伪时间推进。[E6][E7][A8][A9]

# 四、面通量、边界与时间推进

## 4.1 对流与黏性面通量

Euler 的基本流场面通量处理密度、动量、能量的可压缩波系；RANS 变量作为随流对流的附加标量，数值通量使用法向流速及耗散特征值。声速进入基本流场的谱半径和时间步约束。ACM 使用人工可压缩性特征值处理四变量流场；其独立湍流场用局部 Rusanov 型标量通量：

$$
\widehat{u_nq}=\tfrac12\left[u_{nL}q_L+u_{nR}q_R
-\max(|u_{nL}|,|u_{nR}|)(q_R-q_L)\right].
$$

两侧均有有限体积面梯度及湍流扩散，Euler 的湍流扩散通量用动力黏度系数，ACM 用除以 $\rho_0$ 后的运动扩散系数。Euler 还计算温度梯度和热传导；ACM 的四变量黏性通量只有前三个速度分量，压力/连续性分量的黏性通量为零。[E2][E8][A2][A7]

## 4.2 壁面和远场

对于 SST/Wilcox，Euler 在壁面构造 $\rho k=0$，并以 $\rho\omega_w=C_\omega\mu/d^2$ 构造镜像状态；ACM 构造 $k_w=0$、$\omega_w=C_\omega\nu/d^2$，$\nu=\mu/\rho_0$。两式在恒定密度下相互对应，当前默认 $C_\omega=800$。Euler 的等温壁还改变壁面温度和密度，进而改变 $\mu$；ACM 的等温壁标签不引入能量方程。ACM 的湍流远场在入流给定配置值，在出流使用零法向梯度。[E9][A10]

## 4.3 BDF2 与求解器耦合

Euler 的 BDF2/变步长 BDF 双时间对象作用于**整个守恒状态向量**，因此选用 RANS 模型时，$\rho k,\rho\omega$ 等与密度、动量和总能量处在同一时间积分对象和隐式系统中。RANS 源项提供对角线性化近似，但不意味着所有交叉导数均构成完整 Jacobian。[E7][E10]

ACM 的 BDF2 对流场使用物理时间质量矩阵 $M=\operatorname{diag}(1,1,1,0)$；层流及全部四种 RANS 模型可以选择 `BDF2DualTimeGMRES` 或 `BDF2DualTimeLUSGS`。活跃原始变量各有两层物理历史：SA 为 $\tilde\nu$，Wilcox/SST 为 $k,\omega$，realizable k-epsilon 为 $k,\epsilon$。未使用的 SA 第二分量置零，不参与正值检查和时间残差。首次物理步用后退欧拉，随后用

$$
D_tx^{n+1}=
\frac{\tfrac32x^{n+1}-2x^n+\tfrac12x^{n-1}}{\Delta t}.
$$

流场的 GMRES/LU-SGS 修正和湍流的正值限制 SSPRK3 伪时间子步交替推进，直到两个缺陷都满足容差，或达到内迭代上限。BDF2 模式当前不能从缺少两层历史的 VTK-HDF 文件重启。[A3][A9][A11]

# 五、对照时发现的具体实现差异与复核点

1. **SST 的 $\gamma$ 混合存在源码差异。** Euler 的 SST 源项先算 `gamma1`、`gamma2`，随后写成 `gammaC = gamma1 * F1 + gamma1 * (1 - F1)`，因而实际恒为 `gamma1`；ACM 使用 `gamma1 * F1 + gamma2 * (1 - F1)`。这会使远壁 $\omega$ 生产项不同，值得单独核对是否为笔误。[E6][A8]

2. **SST 的负生产量限制存在源码差异。** Euler 先执行 `PkTilde = max(PkTilde, verySmallReal)`，下一句却以原始 `Pk` 做 `min(Pk, upperLimit)`，可能让前一句下限失效；ACM 的 `limitedProduction` 从非负 $P_k$ 出发再取上限。故不能以“同一生产项限制器”概括当前两套程序。[E6][A8]

3. **各向同性 $k$ 应力的存放位置不同。** Euler 的双方程面通量显式加入 $2\rho k/3$；ACM 必须把相应项吸收入修正压力才能与该推导等价。直接比较两程序输出的 `Pressure` 会混入 $2\rho k/3$ 差值。[E5][A1][A6]

4. **扩展物理不同。** Euler 有总能量、可变物性、湍流热导率、可选 QCR、SA-DES/SST-DES 及反应组分路径；ACM 常密度实现没有对应的能量和组分方程。比较两程序时应固定 $\rho,\mu$，关闭额外修正，并统一入口/壁面变量与无量纲尺度。[E2][E3][E6][A4]

5. **运行链路验证不等于模型一致性验证。** 当前 ACM 的短时 SST 圆柱 BDF2 运行记录显示两物理步可执行、湍流变量有限且为正，但内迭代报告 `converged=0`。这些记录不能用于判定两套 RANS 解在物理上相同或 BDF2 已达到二阶时间精度。[A12]

# 六、逐项复现对比的建议

先用两套程序的层流圆柱算例核对网格、速度、Reynolds 数、压力参考和边界定义；再选择同一闭合（建议 SST）、固定相同 $\rho_0$、$\mu$、入口 $k,\omega$、壁面距离和 $C_\omega$。比较原始 $k,\omega$、$\mu_t/\mu$、SST 的 $F_1/F_2$、局部源项和面扩散通量，再比较平均流。压力比较使用相同定义：Euler 热力学压力对 ACM 的 $p^*-2\rho_0k/3$。对非定常算例，应先使每个物理步的内迭代缺陷收敛，再做时间步与网格收敛比较。上述流程是后续验证方案，本文没有新运行跨求解器数值对比。

# 代码依据

下表中的路径均相对于仓库根目录。行号按 2026-10-08 当前工作树记录；源文件发生修改后应重新定位。

| 代号 | 文件和位置 | 支持的说明 |
|:---|:---|:---|
| E1 | `src/Euler/Gas.hpp:300-345` | 守恒/原始状态、压力和总能量闭合 |
| E2 | `src/Euler/Gas.hpp:1865-1955` | 动量应力、热传导和能量黏性通量 |
| E3 | `src/Euler/EulerEvaluator_EvaluateDt.hxx:1196-1240` | 可变物性、$\mu_t$、$C_p\mu_t/Pr_t$ |
| E4 | `src/Euler/Euler.hpp:870-975` | 模型枚举及 5/6/7 变量状态 |
| E5 | `src/Euler/EulerEvaluator_EvaluateDt.hxx:1570-1590` | 双方程通量中的各向同性 $k$ 应力 |
| E6 | `src/Euler/RANS_ke.hpp:498-743` | SST 涡黏度、扩散、源项及两处复核点 |
| E7 | `src/Euler/EulerEvaluator_EvaluateDt.hxx:1836-1875` | RANS 源项及隐式近似 |
| E8 | `src/Euler/EulerEvaluator_EvaluateDt.hxx:1557-1590` | RANS 标量对流通量 |
| E9 | `src/Euler/EulerEvaluator_EvaluateDt.hxx:2625-2685` | 壁面热状态和 RANS 镜像状态 |
| E10 | `src/Euler/EulerSolver.hxx:1868-1890` | 守恒状态的 BDF/双时间积分器 |
| A1 | `src/ACM/ACM.cpp:132-158` | 四变量物理通量 |
| A2 | `src/ACM/ACM.cpp:665-690` | 常密度速度黏性通量 |
| A3 | `src/ACM/ACMSolver.hxx` | BDF2 流场缺陷及各 RANS 模型交替推进 |
| A4 | `src/ACM/ACMTurbulence.hpp:18-105` | 独立湍流场、模型及参数 |
| A5 | `src/ACM/ACMSolver.hxx:142-160` | 面涡黏度回调耦合 |
| A6 | `src/ACM/ACMSettings.hpp:24-76` | `PhysicalP` 压力存储名称 |
| A7 | `src/ACM/ACMTurbulenceTransport.hxx:840-913` | 原始变量对流、扩散和源项残差 |
| A8 | `src/ACM/ACMTurbulence.cpp:334-675` | 四种涡黏度、扩散及源项闭合 |
| A9 | `src/ACM/ACMTurbulenceTransport.hxx` | 按活跃变量管理物理历史和 BDF2 缺陷 |
| A10 | `src/ACM/ACMTurbulence.cpp:686-739` | RANS 壁面、远场镜像状态 |
| A11 | `src/ACM/ACM.cpp:979-994,1055-1062` | BDF2 模型选择与重启限制 |
| A12 | `docs/solver-guide/acm_rans_bdf2_implementation_zh.md` | 既有 ACM 短时圆柱运行记录和限制 |
