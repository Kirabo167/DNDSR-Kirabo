# ACM 任务书要求与高阶求解路径

任务：耦合人工压缩性与高阶变分重构有限体积法，建立 Newton–Krylov
隐式求解框架，提供至少三阶的时空离散，并支持 SST、Wilcox、SA、SA-IDDES。
本文描述常密度 `src/ACM` 的实现；`src/ACMVariable` 是独立模块，不能把这里
新增的时间积分和 IDDES 选项直接用于变密度入口。

## 1. 模型与执行入口

| 任务书模型 | `turbulenceSettings.model` | 二维入口 | 三维入口 |
| --- | --- | --- | --- |
| k–omega SST，1994 版 | `KOmegaSST` | `acm2EQ2D.exe` | `acm2EQ3D.exe` |
| k–omega Wilcox | `KOmegaWilcox` | `acm2EQ2D.exe` | `acm2EQ3D.exe` |
| Spalart–Allmaras | `SpalartAllmaras` | `acmSA2D.exe` | `acmSA3D.exe` |
| SA-IDDES | `SpalartAllmarasIDDES` | `acmSA2D.exe` | `acmSA3D.exe` |

二维 IDDES 入口可用于算法和边界验证。实际解析三维湍流结构的 IDDES 计算应使用
三维非定常网格，并检查解析尺度、时间步和入口脉动等条件。
层流仍使用 `acm2D.exe` 或 `acm3D.exe`。

统一 RANS 状态为 `[u,v,w,p,rho*nuTilde]` 或 `[u,v,w,p,rho*k,rho*omega]`。
`initialState` 的湍流尾部填写密度加权值；入口 `farFieldValue`、上下界和闭合
计算采用原始变量。统一入口要求 `transportSubsteps=1`。

## 2. 空间离散

流场采用已有 CFV 变分重构。湍流新增 `variationalReconstruction=true` 路径，
使用相同次数的多项式和网格求积，计算：

1. 求积点处的湍流状态和梯度，以及闭合所需的速度梯度；
2. 面上的对流、扩散和涡黏度；
3. 单元内部求积点的非线性源项，再积分为单元平均源项。

湍流标量使用独立 CFV 重构对象，避免把 k、omega 或 nuTilde 当作速度分量进行
周期旋转。重构每次从零开始，以逐变量相对系数方程缺陷控制收敛；不能用很大的
omega 数值掩盖 k 的重构误差。源项和面闭合的壁面距离在相应求积点由节点壁面
距离插值得到；几何表示及距离场本身的精度仍需网格验证。

高阶黏性梯度采用两侧多项式梯度平均，加重构面值跳跃稳定项。
这避免以单元均值差强制修正高阶点梯度造成的不一致。低阶路径保留原算法。
湍流高阶多项式仅按物理上下界缩放，检查面和体积求积点，保持单元均值。
靠近物理界限或启用流场限幅时，局部精度可能降低。

下面是用于光滑解精度验证的配置片段，需合并到完整算例：

```json
{
  "vfvSettings": {
    "maxOrder": 3,
    "intOrder": 6
  },
  "reconstructionSettings": {
    "type": "Variational",
    "variationalTolerance": 1e-11,
    "variationalMaxIterations": 2000,
    "enableLimiter": false
  },
  "turbulenceSettings": {
    "model": "KOmegaSST",
    "variationalReconstruction": true,
    "variationalTolerance": 1e-11,
    "variationalMaxIterations": 2000,
    "boundedSSTDiffusion": false,
    "transportSubsteps": 1,
    "transportTimeScale": 1.0
  }
}
```

`maxOrder` 表示多项式次数，不是最终解的实测阶次。这里使用三次多项式和六阶
求积以验证黏性空间算子至少三阶。二次多项式也可选择，但应分别测量各类方程的阶次。

`boundedSSTDiffusion=true` 保留稳健模式。它在倾斜网格上不能保证线性精确性，
因此上述精度验证使用 false；在高阶路径中 false 使用带面值跳跃的多项式扩散，
在原一阶/二阶路径中 false 仍使用既有严格极值限制。不要把稳健模式的结果直接
当作任意非正交网格上的三阶证明。

## 3. 三阶物理时间：ESDIRK3

新增 `timeMarchSettings.integrator="ESDIRK3DualTimeGMRES"`。
它采用 Kennedy–Carpenter 四阶段、刚性精确 ESDIRK3，首阶段显式，随后三个阶段
分别进行双时间非线性求解。已有 `ExplicitSSPRK3` 是稳态伪时间方法，不能用它
代替物理时间三阶验证；BE/BDF2 仍保留兼容。

第 s 阶段满足：

\[
M\frac{U_s-B_s}{a_{ss}\Delta t}=R(U_s,t_n+c_s\Delta t),\qquad
B_s=U_n+\Delta t\sum_{j<s}a_{sj}R(U_j,t_n+c_j\Delta t).
\]

M 的压力分量为零，连续性始终作为代数约束求解。所有活跃流场和湍流变量共享
阶段、物理时间和接受条件。边界使用真实阶段时间；内部阶段时间并非单调递增。
任何阶段不收敛均拒绝整个物理步，恢复接受前的状态，不推进物理时间和完成步数。
`dualTimeRequireConvergence=false` 不允许跳过 ESDIRK 阶段的收敛要求。
ESDIRK 是单步法，场重启不需要 BDF2 两层历史。

```json
{
  "timeMarchSettings": {
    "integrator": "ESDIRK3DualTimeGMRES",
    "newtonKrylov": true,
    "physicalTimeStep": 0.01,
    "pseudoTimeStep": 1.0,
    "maxImplicitIterations": 200,
    "implicitTolerance": 1e-11,
    "linearRelativeTolerance": 1e-8,
    "linearAbsoluteTolerance": 1e-13,
    "gmresPreconditioner": "LUSGS"
  }
}
```

这里的时间步和容差是示例，不能替代算例尺度分析。非线性误差和重构误差必须
小于待测时间截断误差。湍流物理界限、非光滑切换或不一致的初始约束可能降低
观测阶次，压力及时间相关边界也应独立验证。

## 4. Newton–Krylov

`newtonKrylov=true` 适用于 `ImplicitEulerGMRES`、`BDF2DualTimeGMRES` 和
`ESDIRK3DualTimeGMRES`。GMRES 的算子是对完整空间残差进行中心差分的 J*v，
扰动时重新求解变分重构、更新涡黏度、源项和面通量；物理时间质量项与伪时间
预条件项另行加入。组装的低阶流场/湍流耦合矩阵用于 Block-Jacobi 或 LU-SGS
预条件。原 `dualTimeRANSMatrixFreeJacobian` 冻结流场重构的近似选项保持兼容。

这属于带伪时间正则化及阻尼的非精确 Newton–Krylov。限幅和正值边界使算子
可能分段可微，不应宣称处处具有二次非线性收敛。完整重构的 J*v 比冻结重构
更昂贵；需要结合预条件器、CFL、物理步长和线性容差做效率对比。

Krylov 变量缩放对零初始速度/压力修正采用人工压缩性特征单位，避免用
1e-14 的修正量充当流场单位，引起 rho*k 到动量耦合的数值秩丢失。

## 5. SA-IDDES

SA-IDDES 保留 SA 的输运变量、涡黏度、扩散和边界条件，用混合长度替换源项中的
壁面距离。实现包括低 Reynolds 数修正 Psi、近壁屏蔽和提升函数：

\[
\widetilde d=\widetilde f_d(1+f_e)d+
(1-\widetilde f_d)C_{DES}\Psi\Delta,
\qquad C_{DES}=0.65,
\]

\[
\Delta=\min\{h_{max},\max(C_wd,C_wh_{max},h_{wn})\},\quad C_w=0.15.
\]

`hMax` 在本实现中取单元节点之间的最大距离，`hNormal` 为最近壁面方向上的
最大节点投影间距，均为长度；不能用体积直接替代。使用实际节点坐标及周期像。
等距壁面方向相互抵消时，选取有效的节点最近壁面方向，并使用标量距离插值，
避免在区域中轴上得到伪零距离。缺少壁面或节点模板完全不能解析壁面方向时
报告错误，避免悄悄退化成 SA 或用虚构网格尺度。
保留本项目基础 SA 的 ft2 项，在 Psi 中采用相应 ft2 形式；这与忽略 ft2 的
SA-noft2-IDDES 变体需要区分。

参考依据：

1. Shur, Spalart, Strelets, Travin (2008), *A hybrid RANS–LES approach with delayed-DES
   and wall-modelled LES capabilities*, IJHFF 29(6), 1638–1649，
   [DOI: 10.1016/j.ijheatfluidflow.2008.07.001](https://doi.org/10.1016/j.ijheatfluidflow.2008.07.001)。
2. [NASA TMR SA 模型及变体](https://tmbwg.github.io/turbmodels/spalart.html)。
3. [OpenFOAM SA-IDDES 参考实现](https://github.com/OpenFOAM/OpenFOAM-dev/blob/master/src/MomentumTransportModels/momentumTransportModels/LES/SpalartAllmarasIDDES/SpalartAllmarasIDDES.C)，
   交叉核对屏蔽和提升函数，注意其忽略 ft2 的约定。
4. Kennedy and Carpenter (2003), *Additive Runge–Kutta schemes for convection–diffusion–reaction
   equations*, Applied Numerical Mathematics 44, 139–181，
   [DOI: 10.1016/S0168-9274(02)00138-1](https://doi.org/10.1016/S0168-9274(02)00138-1)。

## 6. 验证与验收边界

新增回归覆盖非自治三阶时间收敛、拒绝失败阶段、真实 Wilcox 齐次衰减、
层流剪切衰减、非正交三角网格上的流场与湍流对流扩散空间收敛、完整 Newton
残差与原隐式解的一致性、IDDES 的屏蔽/LES 极限/尺度变换和真实壁面输运。
现有 ACM 回归继续覆盖 SST、Wilcox、SA、边界、守恒和 MPI。

光滑解验证使用三次多项式、六阶求积及严格阶段收敛。非正交三角网格的
流场测试包含压力梯度、对流和黏性项；湍流测试隔离对流扩散算子。

| 验证 | 粗网格/步长 | 中网格/步长 | 细网格/步长 | 最后一级观测阶次 |
| --- | --- | --- | --- | --- |
| 湍流空间残差 L2，n=6/12/24 | 1.012e-2 | 1.724e-4 | 1.070e-5 | 约 4.01 |
| 流场空间残差 L2，n=6/12/24 | 2.300e-2 | 3.054e-4 | 2.339e-5 | 约 3.71 |
| 层流剪切时间误差 L2，步数=4/8/16 | 3.454e-5 | 4.535e-6 | 5.819e-7 | 约 2.96 |

层流时间测试以相同空间算子的半离散 Fourier 衰减解为参考，分离时间与空间误差。
Wilcox 时间测试另与连续齐次衰减解析解比较，两次减半误差比均大于 7。

完整配置示例为
[`acm2EQ2D_SST_high_order.json`](../../cases/acm/2D/2EQ/acm2EQ2D_SST_high_order.json)
和 [`acmSA3D_IDDES_high_order.json`](../../cases/acm/3D/SA/acmSA3D_IDDES_high_order.json)。
它们分别用于无来流齐次衰减和三维闭壁输运接线验证，默认不写结果文件；
后者不是具有已解析湍流结构的 IDDES 工程基准。
示例网格的外边界名为 `FAR`，因此配置在 `boundaryConditions` 中显式把该区
映射为对称面或壁面。仅修改 `defaultBoundaryType` 不会覆盖内置 FAR 区域的语义。

本轮验证记录：15 项 ACM CTest 全部通过，包括 np=1/2/4/8 的二维/三维高阶
扩散守恒测试和既有变密度兼容回归；22 项运行时配置与错误配置拒绝检查通过；
六个常密度入口均成功构建并检查 schema。两份完整示例均完成两个 ESDIRK3
物理步、所有阶段收敛。SST 示例每步共 12 次非线性修正，SA-IDDES 示例每步
共 6 次修正。迭代次数取决于这些小规模验证算例，不能据此推断工程计算加速比。

这些验证分别检查模型核、空间算子和时间推进。它们不等于四种湍流模型在任意
工程网格上均达到三阶，也不替代三维湍流统计验证、压力独立收敛测试、网格/时间步
独立性及相同硬件上的求解效率比较。完整 SUBOFF 或 IDDES 工程验证需使用一致的
网格、初始场、入口条件和收敛标准进行。

运行相关检查：

```bash
cmake --build build -t acm_test_core acm_test_time acm_test_turbulence acm_test_self_periodic acm_test_parallel -j4
ctest --test-dir build -R '^acm_' --output-on-failure
```
