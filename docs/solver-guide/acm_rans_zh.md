# ACM 常密度 RANS 方程与实现

## 1. 从常密度 ACM 到平均流方程

令 $U=(u,v,w,p)^T$，密度 $\rho_0$ 和分子动力黏度 $\mu$ 为常数。ACM 的稳态目标方程可写成

$$
\nabla\cdot\boldsymbol u=0,\qquad
\nabla\cdot(\boldsymbol u\otimes\boldsymbol u)
 +\frac{\nabla p^*}{\rho_0}
 -\frac{1}{\rho_0}\nabla\cdot[2(\mu+\mu_t)\boldsymbol S^d]=0,
\quad
\boldsymbol S^d=\tfrac12(\nabla\boldsymbol u+\nabla\boldsymbol u^T)
 -\tfrac13(\nabla\cdot\boldsymbol u)I.
$$

对于输运 $k$ 的双方程模型，面通量使用有效压力 $p^*=\bar p+2\rho_0 k/3$。默认 `includeIsotropicStress=true` 时，主状态保存静压 $\bar p$，各向同性应力在面通量中显式计入；兼容模式 `false` 才直接保存 $p^*$。SA 模型没有 $k$ 变量，其压力直接采用模型的平均压力约定。因此RANS流动子块将黏性应力中的 $\mu$ 换为 $\mu+\mu_t$；流动子块保留`[u,v,w,p]`和$4\times4$预条件矩阵；新主状态追加密度加权的SA或两个RANS变量。伪时间方程为 $\Gamma(U)\partial_\tau U+R_{\rm flow}(U,q)=0$。二维计算仍保存四分量，但 $w$ 不参与二维空间导数。

闭合计算使用原始变量$q$，新主状态存储$\rho_0q$。除以常密度后的输运形式为

$$
\partial_\tau q_a+\nabla\cdot(\boldsymbol u q_a)
=\nabla\cdot(D_a\nabla q_a)+S_a.
$$

用原始变量输运时，动力黏度形式的模型通量和源项须除以 $\rho_0$。因此 $D_a$ 的量纲为 $\mathrm{m^2/s}$，而 $S_a$ 是原始变量的变化率。每个湍流 SSPRK 阶段重算梯度、对流、扩散和源项；流场残差使用该次求值前冻结的面涡黏度。稳态收敛时，两套伪时间残差均应趋零。

## 2. SST $k$–$\omega$ 闭合

取 $q=(k,\omega)$，$\nu=\mu/\rho_0$，$\beta^*=0.09$，$a_1=0.31$，$d$ 为壁面距离。定义 $|\Omega|=\sqrt{2\Omega_{ij}\Omega_{ij}}$，$S^2=2S^d_{ij}S^d_{ij}$。本实现采用涡量型 SST 应力限制器：

$$
\mu_t=\rho_0\frac{a_1 k}{\max(a_1\omega,|\Omega|F_2)},
\qquad
P_k=\frac{\mu_t}{\rho_0}S^2
-\frac23k\nabla\cdot\boldsymbol u.
$$

代码再将 $\mu_t/\mu$ 限于配置的上限，并将 $k$ 方程生产项限于 $20\beta^*k\omega$。SST 混合函数是

$$
\begin{aligned}
CD_{k\omega}&=\max(2\rho_0\sigma_{\omega2}\nabla k\cdot\nabla\omega/\omega,10^{-20}),\\
A_1&=\min\!\left[\max\!\left(\frac{\sqrt{k}}{\beta^*\omega d},
\frac{500\nu}{d^2\omega}\right),\frac{4\rho_0\sigma_{\omega2}k}{CD_{k\omega}d^2}\right],\\
A_2&=\max\!\left(\frac{2\sqrt{k}}{\beta^*\omega d},
\frac{500\nu}{d^2\omega}\right),\\
F_1&=\tanh(A_1^4),\qquad F_2=\tanh(A_2^2).
\end{aligned}
$$

实际求值对正分母与函数参数做有限值保护。令 $\sigma_k=F_1\sigma_{k1}+(1-F_1)\sigma_{k2}$，$\sigma_\omega$、$\beta$、$\gamma$ 同理混合，其中
$\sigma_{k1}=0.85$、$\sigma_{k2}=1$、$\sigma_{\omega1}=0.5$、$\sigma_{\omega2}=0.856$、$\beta_1=0.075$、$\beta_2=0.0828$，
$\gamma_i=\beta_i/\beta^*-\sigma_{\omega i}\kappa^2/\sqrt{\beta^*}$，$\kappa=0.41$。于是

$$
\begin{aligned}
D_k&=(\mu+\sigma_k\mu_t)/\rho_0,
&S_k&=\min(P_k,20\beta^*k\omega)-\beta^*k\omega,\\
D_\omega&=(\mu+\sigma_\omega\mu_t)/\rho_0,
&S_\omega&=\gamma\frac{\rho_0 P_k}{\mu_t}-\beta\omega^2
+2(1-F_1)\sigma_{\omega2}\frac{\nabla k\cdot\nabla\omega}{\omega}.
\end{aligned}
$$

壁面施加 $k_w=0$ 和 $\omega_w=C_\omega\nu/d^2$，默认 $C_\omega=800$；入口使用配置的 $k,\omega$，远场只在局部入流时使用配置值，出流用零法向梯度。扩散使用镜像虚单元实现面上的 Dirichlet 值；对流入流必须直接使用规定面值，不能使用镜像值。壁面与对称面湍流对流通量为零。

## 3. 其他模型和代码位置

| 配置值 | 输运变量 | 涡黏度与特征 |
| --- | --- | --- |
| `Laminar` | 无 | $\mu_t=0$ |
| `SpalartAllmaras` | $\tilde\nu$ | $\mu_t=\rho_0\tilde\nu f_{v1}$；含壁面破坏与梯度平方源项 |
| `KOmegaWilcox` | $k,\omega$ | $\rho_0k/\omega$ 加应力限制；含交叉扩散和生产限制 |
| `KOmegaSST` | $k,\omega$ | 上述 $F_1/F_2$ 混合模型 |
| `KOmegaSSTSAS` | $k,\omega$ | SST-1994 加 Menter–Egorov 尺度自适应源项 |
| `RealizableKEpsilon` | $k,\epsilon$ | realizable 黏度限制；含低 Reynolds 数修正与额外 $\epsilon$ 源项 |

模型公式、正值界限和壁面状态在 [`ACMTurbulence.cpp`](../../src/ACM/ACMTurbulence.cpp)；面通量、梯度、MPI ghost 交换与 SSPRK3 在 [`ACMTurbulenceTransport.hxx`](../../src/ACM/ACMTurbulenceTransport.hxx)；与流场黏性通量和黏性 CFL 的耦合在 [`ACMEvaluator.hxx`](../../src/ACM/ACMEvaluator.hxx)，求解流程在 [`ACMSolver.hxx`](../../src/ACM/ACMSolver.hxx)。`turbulenceSettings.model` 选择模型，`initialValue` 和 `farFieldValue` 按表中的变量顺序填写。VTK-HDF 字段分别命名为 `TurbulenceNuTilde`、`TurbulenceK`、`TurbulenceOmega` 或 `TurbulenceEpsilon`，重启按所选模型读取。

新模型的统一主状态和完整残差在同一阶段/内迭代中更新。SA入口为acmSA2D/3D，两方程入口为acm2EQ2D/3D。物理BDF2保存完整守恒状态的两级历史，采用逐变量正性保护。隐式系统使用流场面耦合、局部湍流源项/谱半径近似和 rho*k 到流场的各向同性压力交叉导数，其他交叉导数冻结；细节见[统一状态与模型入口](acm_model_state_zh.md)。

流场的FirstOrder、GreenGauss、Variational及WBAP/CWBAP设置保持原有要求。湍流重构仍由secondOrderReconstruction控制；流场Variational不自动提高湍流多项式次数。初值以initialState的rho*q尾部为准，farFieldValue及上下界仍填原始量q。VTK-HDF同时保存原始和密度加权湍流数据。

底层四变量兼容类及旧分离输运核保留用于流动子块、闭合工作缓存和原有回归；原生四变量入口仅接受层流配置。新统一入口使用[ACMRANSSolver.hxx](../../src/ACM/ACMRANSSolver.hxx)推进全部活跃方程。


## 4. SST 文献核对与空间离散修复（2026-11-13）

本项目 `KOmegaSST` 保持 **SST-1994 原始版本**，不混用 SST-2003 的应变率限制器、
10 倍生产限制、交叉扩散下限或两方程同时限制的生产项。

核对来源：

1. Menter, F. R., “Two-Equation Eddy-Viscosity Turbulence Models for Engineering Applications,”
   AIAA Journal 32(8), 1598–1605 (1994), [DOI: 10.2514/3.12149](https://doi.org/10.2514/3.12149)。
2. [NASA/TMR 标准 SST 及版本区别](https://tmbwg.github.io/turbmodels/sst.html#sst-std)：
   本次直接查阅的公式依据，包括守恒形式、完整 Reynolds 应力生产项、20 倍 k 生产限制、
   `CD=1e-20` 和 `omega_wall=10*6*nu/(beta1*d1^2)=800*nu/d1^2`。
3. [Menter, NASA TM 103975 (1992)](https://ntrs.nasa.gov/citations/19930013620)：
   查阅其 SST 推导与 omega 生产项说明，作为历史背景；早期报告中的常数不覆盖 1994/TMR 定义。

模型与数值保护须区分：涡黏度比上限、正值下限和扩散极值保护是数值措施，不是新增的 SST 常数。
三个模块都使用完整 `P=tau:grad(u)`；只有 k 方程使用生产上限，omega 使用未截断的
`gamma*P/nu_t`，允许膨胀导致的负生产。Euler 删除了 omega 生产中固定 `nu_t>=1e-8` 的
有量纲下限，并将默认的应变率近似改为该原始形式。ACM 的生产项按密度换成原始变量变化率。
密度守恒梯度先换算成 k、omega 与速度梯度，不能把 `grad(rho*k)` 直接当作 `grad(k)`。

常密度与变密度 ACM 共用 [标量空间算子](../../src/CFV/ScalarTransport.hpp)：

- 在静态网格上缓存按方向缩放的加权最小二乘 SVD 系数，使用实际邻居/周期像坐标；
  模板秩不足时报告错误，避免静默使用不一致梯度。Dirichlet 虚单元坐标与穿过实际面点的反射值匹配。
- 对流按共享体积/质量通量方向取迎风状态。规定入流取原始面值；扩散仍使用镜像值。
  常密度的冻结迎风量和隐式边界导数同步采用这一约定。
- 对当前模板的局部极值，截断会加深极值的面法向扩散梯度；一个面只产生一个通量，
  两侧以相反符号累加。极值标志通过 MPI ghost 交换，内部平滑模板保留完整非正交修正。
  这不构成完整高阶对流、源项和时间推进的无条件正性证明；现有变量界限仍保留。
- Euler 使用自己的 CFV 重构和边界接口；其远场/入口没有复用 ACM 的镜像入流路径。
  Euler SST 的不可穿透面在减去黏性通量之前清零 k/omega 对流通量，保留真实壁面扩散，
  避免耗散特征值方案引入伪对流。

回归覆盖压缩/膨胀、强剪切触发生产上限、弱黏度、二维/三维公式、倾斜拉伸梯度模板、
扩散面守恒、规定入流及其隐式导数，以及真实 Euler 面通量。共享边界修复也纠正了 SA 入流，
其已有 BE/BDF2 独立解析测试随正确的边界离散更新，精度阈值保持不变。

完整 SUBOFF 的收敛变化需要从相同检查点、相同参数和残差阈值另做对照；
小网格或核函数回归通过不能替代工程网格上的验证。


## 5. 共享有界面系数与近下限平台（2026-11-16）

对 `Suboff潜艇计算 · gs` 对话中的 2026-11-16 算法进行了源码片段与验证记录核对。
此前“严格局部极值”的开关会遗漏仅比邻居高约机器舍入量的下限平台。
改为非负共享传递系数可以限制这一问题，但它有明确的精度代价，不能称为任意倾斜网格上的线性精确替换。

常密度和变密度 ACM 的 SST 默认使用 `turbulenceSettings.boundedSSTDiffusion=true`：

$$
\Delta q=q_R-q_L,\quad d_n=\boldsymbol d\cdot\boldsymbol n>0,\quad
C_0=\frac{\boldsymbol n\cdot\boldsymbol n}{d_n},\quad
C=\operatorname{clip}\left(\frac{g_n}{\Delta q},0,2C_0\right),\quad
F_D=D_f\,\Delta q\,C.
$$

这里的 q_L、q_R 是单元均值，边界 q_R 为相应扩散虚单元值；g_n 是完整非正交修正后的法向梯度。
相等状态取 C=C_0，而通量直接计算为零，避免对含巨大切向分量的修正梯度再次投影。
常密度 D_f 是运动扩散系数，变密度守恒残差使用 rho_f*D_f。
一个面的同一通量以相反符号累加到两侧，因而保持守恒；扩散传递方向与单元差值一致，
幅度不超过两点通量的两倍，不再依赖精确极值标志。

本地移植保留了两点修正：

1. SST 的 F1、扩散系数和源项继续用原始模型梯度计算；只限制最终传递通量，
   不把截断后的梯度再代入混合函数，以免同时改变另一方程的系数。
2. 使用有向正投影，不以 abs(d_n) 掩盖反向连接；零/负投影返回零传递。
   不使用固定的 `1e-14` 长度截断，以免不同长度单位下出现不同算法。

常密度隐式面块使用同一冻结 D_f*C，包括扩散边界的虚单元导数。
这属于 Picard 冻结系数线性化，不是非线性限幅函数的完整 Newton 导数。
变密度 ACM 仍采用既有分离式湍流更新，流场隐式步骤冻结湍流变量。

**精度限制：**例如 n=(1,0)、d=(1,1)、梯度=(1,-1) 的线性场，其单元差为零，
真实法向导数为 1，本算法却给零扩散。因此该模式不能保证任意倾斜网格上的线性精确或原高阶收敛率。
未触发截断的可接受梯度会保留；强非正交区域需要单独验证网格收敛性。
将 `boundedSSTDiffusion=false` 可恢复 2026-11-13 的严格极值限幅路径，用于受控对照。
旧配置省略新字段时采用 true；该开关不改变 SST-1994 的闭合常数，也不改变压力存储约定。

新增回归覆盖近下限平台、相等状态的精确零通量、面交换守恒、两体积显式步的 CFL 条件界限、
真实周期网格上的冻结矩阵/残差匹配、变密度守恒扩散量纲、新旧模式，以及上述精度反例。
这些检查不能证明完整对流/源项/时间算法无条件保正，也不能替代 SUBOFF 完整网格的续算和收敛验证。

## 6. SST-SAS 尺度自适应模型

常密度 `acm2EQ2D` 和 `acm2EQ3D` 通过 `turbulenceSettings.model="KOmegaSSTSAS"` 选择该模型。
在算例中设置 `turbulenceSettings.variationalReconstruction=true`、`vfvSettings.maxOrder` 至少为 2、
`vfvSettings.intOrder` 至少为 4。这样可以在体积求积点同时求得速度梯度、速度拉普拉斯和湍流变量梯度。
解析剪切层或尾流的非定常结构时，使用三维网格和物理时间推进，例如 `ESDIRK3DualTimeGMRES`。

SAS 用应变率大小 $S=\sqrt{2S_{ij}S_{ij}}$ 和速度拉普拉斯的模 $|\nabla^2\boldsymbol u|$
计算 von Kármán 尺度，用局部网格长度 $\Delta$ 对它设置下界。常密度 $\omega$ 方程的单位质量新增源项为

$$
\begin{aligned}
L &= \frac{\sqrt{k}}{(\beta^*)^{1/4}\omega},\\
L_{\mathrm{vK}} &= \max\left(\frac{\kappa S}{|\nabla^2\boldsymbol u|},\;
      C_s\sqrt{\frac{\kappa\zeta_2}{\beta/\beta^*-\gamma}}\,\Delta\right),\\
q_{\mathrm{SAS}} &= \max\left(\zeta_2\kappa S^2\left(\frac{L}{L_{\mathrm{vK}}}\right)^2
      - \frac{2C k}{\sigma_\Phi}\max\left(\frac{|\nabla\omega|^2}{\omega^2},
      \frac{|\nabla k|^2}{k^2}\right),\;0\right).
\end{aligned}
$$

这里 $\kappa=0.41$、$\zeta_2=3.51$、$C_s=0.11$、$C=2$、$\sigma_\Phi=2/3$。
三维 $\Delta=V^{1/3}$，二维取单元面积平方根。$\beta$ 和 $\gamma$ 使用局部 SST 混合系数。
实现以 $S/L_{\mathrm{vK}}$ 的等价形式求值，速度拉普拉斯为零时该源项为零。
依据为 [Menter 与 Egorov（2010）](https://doi.org/10.1007/s10494-010-9264-5) 和
[CFX Theory Guide §2.7，式 (2–210) 至 (2–215)](https://ansyshelp.ansys.com/public/Views/Secured/corp/v251/en/cfx_thry/thry_turb_sas.html)。

体积积分将 $\rho_0 q_{\mathrm{SAS}}$ 加入守恒 $\omega$ 残差。组装隐式矩阵在相同求积点使用
冻结梯度、$k$ 和混合系数后的 SAS $\omega$ 导数。矩阵自由 Newton–Krylov 直接对完整空间残差求差分。
