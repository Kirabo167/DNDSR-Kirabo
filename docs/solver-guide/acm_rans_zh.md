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

对于输运 $k$ 的双方程模型，$p^*=\bar p+2\rho_0 k/3$ 吸收了 Reynolds 应力的各向同性部分；SA 模型没有 $k$ 变量，其压力直接采用模型的平均压力约定。因此 RANS 流场只需将黏性应力中的 $\mu$ 换为 $\mu+\mu_t$；`[u,v,w,p]` 四分量状态、ACM 通量和 $4\times4$ 预条件矩阵无需扩展。伪时间方程为 $\Gamma(U)\partial_\tau U+R_{\rm flow}(U,q)=0$。二维计算仍保存四分量，但 $w$ 不参与二维空间导数。

湍流变量 $q$ 独立存放，稳态伪时间输运统一写成

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
P_k=\max\left(\frac{\mu_t}{\rho_0}S^2
-\frac23k\nabla\cdot\boldsymbol u,0\right).
$$

代码再将 $\mu_t/\mu$ 限于配置的上限，并将 $k$ 方程生产项限于 $20\beta^*k\omega$。SST 混合函数是

$$
\begin{aligned}
CD_{k\omega}&=\max(2\rho_0\sigma_{\omega2}\nabla k\cdot\nabla\omega/\omega,10^{-10}),\\
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
&S_\omega&=\gamma S^2-\beta\omega^2
+2(1-F_1)\sigma_{\omega2}\frac{\nabla k\cdot\nabla\omega}{\omega}.
\end{aligned}
$$

壁面施加 $k_w=0$ 和 $\omega_w=C_\omega\nu/d^2$，默认 $C_\omega=800$；入口使用配置的 $k,\omega$，远场只在局部入流时使用配置值，出流用零法向梯度。有限体积边界状态用镜像值实现面上的 Dirichlet 值。

## 3. 其他模型和代码位置

| 配置值 | 输运变量 | 涡黏度与特征 |
| --- | --- | --- |
| `Laminar` | 无 | $\mu_t=0$ |
| `SpalartAllmaras` | $\tilde\nu$ | $\mu_t=\rho_0\tilde\nu f_{v1}$；含壁面破坏与梯度平方源项 |
| `KOmegaWilcox` | $k,\omega$ | $\rho_0k/\omega$ 加应力限制；含交叉扩散和生产限制 |
| `KOmegaSST` | $k,\omega$ | 上述 $F_1/F_2$ 混合模型 |
| `RealizableKEpsilon` | $k,\epsilon$ | realizable 黏度限制；含低 Reynolds 数修正与额外 $\epsilon$ 源项 |

模型公式、正值界限和壁面状态在 [`ACMTurbulence.cpp`](../../src/ACM/ACMTurbulence.cpp)；面通量、梯度、MPI ghost 交换与 SSPRK3 在 [`ACMTurbulenceTransport.hxx`](../../src/ACM/ACMTurbulenceTransport.hxx)；与流场黏性通量和黏性 CFL 的耦合在 [`ACMEvaluator.hxx`](../../src/ACM/ACMEvaluator.hxx)，求解流程在 [`ACMSolver.hxx`](../../src/ACM/ACMSolver.hxx)。`turbulenceSettings.model` 选择模型，`initialValue` 和 `farFieldValue` 按表中的变量顺序填写。VTK-HDF 字段分别命名为 `TurbulenceNuTilde`、`TurbulenceK`、`TurbulenceOmega` 或 `TurbulenceEpsilon`，重启按所选模型读取。

当前四变量流场可用显式或隐式伪时间推进，湍流输运仍为显式分离推进；物理时间 BDF2 模式仅接受 `Laminar`。这套实现适用于常密度稳态 RANS，尚不提供湍流方程隐式 Jacobian 或非定常湍流历史。旧版本将 SA 和 realizable $k$–$\epsilon$ 的 VTK-HDF 字段误标为 `TurbulenceK/Omega`；新版本按变量真实名称读取，旧文件应重新导出或显式重建湍流初值。已有 [`test_ACMTurbulence.cpp`](../../test/cpp/ACM/test_ACMTurbulence.cpp) 覆盖模型配置、SST 解析剪切流、闭合核、边界状态和变量命名；模型的物理预测仍需使用网格收敛与基准算例验证。

## 4. 圆柱算例的多模型链路检查

从项目根目录运行：

```bash
cmake --build build -t euler acm_test_turbulence -j4
./build/test/cpp/acm_test_turbulence
venv/bin/python scripts/check_acm_rans_cylinder.py --steps 5 --np 1
```

检查脚本默认使用 `cases/acm2D/acm2D.json` 的 Re=20 网格和物性，用相同的显式 SSPRK3、CFL 与步数运行层流基线和四种 RANS 模式，分别检查完整步数、有限残差、VTK-HDF 字段名、有限场值及湍流变量正值。结果写到 `/tmp/acm-rans-cylinder-<时间>/report.json`，原始求解日志保存在各模型子目录。该短算例是求解链路测试；Re=20 不用于判断 RANS 的物理准确性。可通过 `--case`、`--np`、`--steps` 和 `--viscosity` 改用合适的高 Reynolds 数算例；物理验证还需足够长的稳态收敛、壁面量和网格敏感性检查。

在无图形会话的计算节点上，脚本默认设置 `HWLOC_COMPONENTS=-gl`，避免 OpenMPI 的 hwloc GL 插件连接 X11 显示而停住。直接运行求解器时也可设置同一环境变量。脚本的本机套接字预检查用于区分执行沙箱阻断与求解器失败；若系统禁止创建套接字，MPI 算例必须转到允许本机通信的执行环境。
