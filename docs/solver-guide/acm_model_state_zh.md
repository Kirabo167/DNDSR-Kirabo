# ACM 统一守恒状态与模型入口 {#acm_model_state_zh}

常密度 ACM 参照 Euler 按模型确定编译时状态长度，提供层流、SA、两方程的二维和三维入口。SA/两方程求解器的主状态、残差、修正量和已完成物理时间历史采用同一分布式数组，流场及湍流量在同一 RK 阶段或非线性内迭代中更新。

| 可执行程序 | solver.model | fieldNVariables | 主状态 |
| --- | --- | --- | --- |
| acm2D / acm3D | ConstantDensity2D / ConstantDensity3D | 4 | [u,v,w,p]，仅层流 |
| acmSA2D / acmSA3D | ConstantDensitySA2D / ConstantDensitySA3D | 5 | [u,v,w,p,rho*nuTilde] |
| acm2EQ2D / acm2EQ3D | ConstantDensity2EQ2D / ConstantDensity2EQ3D | 6 | [u,v,w,p,rho*k,rho*omega]，或[rho*k,rho*epsilon]作为尾部 |

二维入口仍保留三个速度分量。rho是配置中的常数rho0；本模块不增加密度方程或能量方程。ACMVariable维持自己的五分量变密度实现。

两方程入口通过turbulenceSettings.model选择KOmegaWilcox、KOmegaSST或RealizableKEpsilon。SA入口只接受SpalartAllmaras；层流入口只接受Laminar。元数据、状态长度和闭合模型不匹配时，--check-config即拒绝运行。

## 状态和配置约定

initialState是统一主状态，尾部必须写密度加权量。以rho0=2的SST为例：

```json
{
  "solver": {
    "type": "ACM",
    "discretization": "CFV",
    "model": "ConstantDensity2EQ3D",
    "fieldNVariables": 6
  },
  "initialState": [1, 0, 0, 0, 0.0003, 0.63887656499994]
}
```

这对应k=0.00015、omega=0.31943828249997。turbulenceSettings.farFieldValue、minimumValue、maximumValue仍按原始变量q填写；上下界在主状态更新时换成rho0乘q的界限。initialValue保留在共享配置中，新模型初始化以initialState的尾部为准。

boundaryValue、boundaryConditions[].value、leftState/rightState仍是四分量流场数据；湍流远场/壁面数据由闭合边界接口生成。它们不是统一主状态的别名，不应直接填写5/6分量数组。

新模型要求transportSubsteps=1。每个阶段都求完整残差，已不使用旧的流场一步之后再做若干湍流显式子步。transportTimeScale保留为湍流伪时间尺度/预条件参数；它不缩放物理BDF导数。流场重构由reconstructionSettings/vfvSettings控制，湍流空间重构仍由secondOrderReconstruction控制。

## 空间残差及耦合

主状态U的尾部是rho*q。闭合计算需要q时，从U尾部除以rho0投影到原始变量工作缓冲。每次残差求值都从U重新投影，因此这些工作缓冲不决定主状态。

湍流对流使用同一重构流场、同一ACM Riemann连续性面通量，再按该通量方向选取湍流迎风值。一个面的通量以相反符号累加到相邻控制体，守恒输运的残差为rho0乘原始输运残差。rho0为常数，所以原始闭合和守恒形式相互一致。

k/omega/epsilon或nuTilde决定涡黏度，涡黏度反馈到流场黏性残差。各量共享RK阶段及内迭代，避免阶段之间使用另一个时间层的湍流状态。

## 时间推进与隐式系统

六种现有推进名称全部保留：ExplicitSSPRK3、ImplicitEulerBlockJacobi、ImplicitEulerLUSGS、ImplicitEulerGMRES、BDF2DualTimeLUSGS、BDF2DualTimeGMRES。

共享GMRES接口保留Euler的总轮数预算：传入N最多执行N轮。ACM/ACMVariable的gmresRestarts和常密度ACM的variationalGMRESRestarts表示首轮之后允许的重启次数，调用端换算为1+该值；配置0仍执行首轮，配置3最多执行4轮，达到容差时可提前结束。Euler的nGmresIter仍按总轮数解释。

ExplicitSSPRK3同时推进所有主状态分量。隐式主状态修正采用5/6分量向量：流场部分复用4×4人工压缩算子和分布式面耦合；湍流部分采用局部源项Jacobian、正的对流/扩散谱半径近似，以及伪时间/物理时间对角项。两方程的局部源项块包括变量之间的耦合。

当前线性化冻结流场与闭合的交叉导数，也没有建立湍流空间非对角Jacobian。因此这是分块近似/Picard耦合，不能解释成完整全耦合Newton系统。GMRES向量、点积和修正量包含全部5/6个分量，LU-SGS用于流场子块，湍流子块在单元内求解。

物理BDF2历史保存完整主状态。物理质量矩阵是diag(1,1,1,0,1[,1])，压力不含物理时间导数。第一步用BE，后续用BDF2；各湍流守恒分量在历史预测低于自身密度加权下限时独立回退BE。未满足收敛条件的物理步不会提交历史或时间，除非明确开启诊断推进。

Absolute/Relative仍通过dualTimeConvergenceCriterion选择。流场和湍流分别保存本物理步的初始残差，所有活跃方程均需满足所选条件。湍流残差现在属于rho*q方程，绝对残差量级随rho0缩放；相对残差比不受这个常数缩放影响。逐方程设置顺序为[u,v,w,p,rho*q1,rho*q2]，SA只启用前5项。

## 输出与重启

新模型VTK-HDF输出包括Velocity、Pressure、Density、原始湍流字段及密度加权字段：

- SA：TurbulenceNuTilde、rho_TurbulenceNuTilde。
- Wilcox/SST：TurbulenceK、TurbulenceOmega及对应rho_字段。
- RKE：TurbulenceK、TurbulenceEpsilon及对应rho_字段。

同分区稳态重启读取守恒字段并检查其与原始字段、rho0的一致性。旧的原始湍流字段文件可以作为初始化输入。现有VTK-HDF重启没有完整BDF2两级历史，仍不用于初始化物理双时间推进。

## 构建与运行

```bash
cmake --build build --target acmSA2D acmSA3D acm2EQ2D acm2EQ3D -j2
cd build
./app/acmSA2D.exe ../cases/acm/2D/SA/acmSA2D.json --check-config
./app/acm2EQ3D.exe ../cases/acm/3D/2EQ/acm2EQ3D.json --check-config
mpirun -np 4 ./app/acm2EQ3D.exe \
  ../cases/acm/3D/acm3D_cylinder_Re3000_coarse_SST_BDF2_LUSGS.json
```

cases/acm/{2D,3D}/{SA,2EQ}中提供四份完整示例及配套schema。已有三维SST圆柱和SUBOFF配置已切换为ConstantDensity2EQ3D、fieldNVariables=6，保留原有文件路径。all_acm/all_solvers及cases/update_schemas.sh包含新入口。

实现来源：[模型布局与换算](../../src/ACM/ACMModelState.hpp)、[统一求解器](../../src/ACM/ACMRANSSolver.hpp)、[时间推进](../../src/ACM/ACMRANSSolver.hxx)、[输运及源项线性化](../../src/ACM/ACMTurbulenceTransport.hxx)、[入口校验](../../src/ACM/SingleBlockApp.hpp)。
