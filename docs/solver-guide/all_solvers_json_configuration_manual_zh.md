# DNDSR 全求解器 JSON 配置说明书 {#all_solvers_json_configuration_manual_zh}

版本：2026-10-13。本说明书依据本机 DNDSR 当前工作树的源码和实际程序导出的 JSON Schema 编写，包含当前尚未提交的 ACM 绝对/相对收敛方式修改。后续源码变更时，应重新导出 schema 并核对行为。

覆盖范围：Euler/CFV 的9个入口、常密度 ACM 的6个入口、变密度 ACMVariable 的2个入口、节点中心 NCFV 的2个入口，共19个原生求解器。另收录 EulerP 的 Python/Host/CUDA 配置接口。`ACM.exe` 是常密度三维兼容入口；`eulerState`、Cantera 零维工具属于辅助程序，不另算网格流场求解器。

本书前半部分用于选程序、修改算例和理解各模块差异；后半部分按求解器族列出全部参数的中文含义、数据类型、schema 默认参考、取值约束及适用条件。`[]` 表示数组中的每个条目，`<key>` 表示对象中的动态键。同一族的二维/三维/模型变体共享参数表，入口差异单独列出。

阅读顺序：先看第1—3章，再看对应求解器的第4—7章；遇到具体字段时查第10—13章。EulerP 使用者看第8章。JSON 片段标注为“修改片段”的，只表示应修改的字段；应合并到完整配套算例中运行。

阅读索引：

[1 求解器与状态变量对照](#dndsr_config_ch1) · [2 JSON 加载、默认值与运行路径](#dndsr_config_ch2) · [3 不同模块的配置位置](#dndsr_config_ch3) · [4 Euler/CFV 配置使用说明](#dndsr_config_ch4)

[5 常密度 ACM 配置使用说明](#dndsr_config_ch5) · [6 变密度 ACMVariable 配置使用说明](#dndsr_config_ch6) · [7 节点中心 NCFV 配置使用说明](#dndsr_config_ch7) · [8 EulerP 的 Python/Host/CUDA 配置](#dndsr_config_ch8)

[9 配置修改与验证流程](#dndsr_config_ch9) · [10 Euler/CFV 完整参数表](#dndsr_config_ch10) · [11 常密度 ACM 完整参数表](#dndsr_config_ch11) · [12 变密度 ACMVariable 完整参数表](#dndsr_config_ch12)

[13 节点中心 NCFV 完整参数表](#dndsr_config_ch13) · [14 较长默认值与结构示例](#dndsr_config_ch14) · [15 配套算例与schema索引](#dndsr_config_ch15) · [16 来源、范围与核对记录](#dndsr_config_ch16)



## 1 求解器与状态变量对照 {#dndsr_config_ch1}

| 可执行程序 | solver.model | 几何维度 | 状态分量与用途 |
| --- | --- | --- | --- |
| euler.exe | NS | 2D | 5：rho、rho×u、rho×v、rho×w、rho×E；二维几何保留三速度 |
| euler2D.exe | NS_2D | 2D | 4：rho、rho×u、rho×v、rho×E |
| euler3D.exe | NS_3D | 3D | 5：rho、rho×u、rho×v、rho×w、rho×E |
| eulerSA.exe | NS_SA | 2D | 6：五个流动守恒量加rho*nuTilde |
| eulerSA3D.exe | NS_SA_3D | 3D | 6：同上，三维几何 |
| euler2EQ.exe | NS_2EQ | 2D | 7：五个流动守恒量加两个RANS守恒输运量 |
| euler2EQ3D.exe | NS_2EQ_3D | 3D | 7：同上，三维几何 |
| eulerEX.exe | NS_EX | 2D | 动态：五个流动量及扩展组分/标量，长度由配置确定 |
| eulerEX3D.exe | NS_EX_3D | 3D | 动态：同上，三维几何 |
| acm2D.exe | ConstantDensity2D | 2D | 4：[u,v,w,p]，仅层流 |
| acm3D.exe | ConstantDensity3D | 3D | 4：[u,v,w,p]，仅层流 |
| acmSA2D.exe | ConstantDensitySA2D | 2D | 5：[u,v,w,p,rho*nuTilde] |
| acmSA3D.exe | ConstantDensitySA3D | 3D | 5：[u,v,w,p,rho*nuTilde] |
| acm2EQ2D.exe | ConstantDensity2EQ2D | 2D | 6：[u,v,w,p,rho*k,rho*omega/epsilon] |
| acm2EQ3D.exe | ConstantDensity2EQ3D | 3D | 6：[u,v,w,p,rho*k,rho*omega/epsilon] |
| acmVariable2D.exe | VariableDensity2D | 2D | 5：[rho,mx,my,mz,p]，m=rho*速度 |
| acmVariable3D.exe | VariableDensity3D | 3D | 5：[rho,mx,my,mz,p]，m=rho*速度 |
| ncfv_euler2D.exe | IdealGas | 2D | 4：节点/对偶体守恒量；输入原始数组为[rho,u,v,p] |
| ncfv_euler3D.exe | IdealGas | 3D | 5：节点/对偶体守恒量；输入原始数组为[rho,u,v,w,p] |

Euler入口设置 `solver.type="Euler"`、`solver.discretization="CFV"`；常密度 ACM 设置 `ACM/CFV`；变密度设置 `ACMVariable/CFV`；NCFV 设置 `ncfv_euler/NCFV`。`solver.fieldNVariables` 必须与表中模型一致，EX 为动态例外。求解器由编译后的入口决定；修改元数据不会改变方程实现。

Euler 两方程模型中，Wilcox/SST 的附加守恒量为 `[rho*k,rho*omega]`，Realizable k-epsilon 为 `[rho*k,rho*epsilon]`。常密度 ACM 的 `initialState` 不含rho或能量；变密度 ACM 的状态不含可压缩总能量；这些数组不能互换。

## 2 JSON 加载、默认值与运行路径 {#dndsr_config_ch2}

| 模块 | 配置加载顺序 | 省略字段时的处理 |
| --- | --- | --- |
| Euler | 配套默认配置→算例→CLI覆盖 | 优先读取算例同目录的`<target>_default_config.json`；不存在时用编译默认对象 |
| ACM | 完整算例→兼容补项→CLI覆盖 | 只补源码明确列出的兼容字段；不要按schema default任意删字段 |
| ACMVariable | 完整算例→兼容补项→CLI覆盖 | 同样使用完整单文件；不读取Euler式相邻默认配置 |
| NCFV | 类型化默认配置→算例→CLI覆盖 | 按配置对象补默认值，边界/初值条目也合并各自默认值 |
| EulerP | Python脚本/字典→对象接口 | 没有与上述19个入口等价的独立JSON文件驱动 |

后半部分的“默认参考”是实际导出 schema 中的值，供识别参数使用。它不保证字段可以省略，也不保证等于按模型构造后的值或最终合并值。例如：Euler SA 构造时选择 `RANS_SA`，两方程构造时选择 `RANS_KOWilcox`，但通用字段schema可以显示 `RANS_None`；Euler远场状态schema中的空数组是格式模板，实际模型构造时按状态长度生成值；CFV缓存大小schema可能为255，按维度构造时可设为dim+1。已有相邻默认JSON也会覆盖编译默认行为。

配置文件允许 JSONC 注释。标准 JSON 校验器需先去掉注释。`$schema` 用于编辑器提示，其相对路径按配置文件位置解释；网格、输出、重启等运行路径按**程序启动工作目录**解释。通常从 `build/` 或 `build-reactive/` 启动，用 `../cases/...`、`../data/...` 指向仓库目录。反应机理还可受Cantera搜索路径及 `DNDS_MECH_PATH` 影响，应记录该环境变量。

所有原生入口支持 `--emit-schema` 和 `--check-config`，以及成对 `-k /JSON/pointer -v JSON值` 覆盖。字符串值可用JSON引号形式；程序也提供普通字符串回退。`--check-config` 检查参数和选择，不读取网格、不推进，也不能验证网格边界名称、所有物理条件和实际运行稳定性。Schema校验可发现拼错的字段；不要依赖未知键必然被运行时拒绝。

```bash
# 以下命令从仓库根目录对应的build目录执行
cd build
./app/euler2D.exe --emit-schema > /tmp/euler2D_schema.json
./app/euler2D.exe ../cases/euler/2D/euler2D_config.json --check-config
./app/ncfv_euler2D.exe ../cases/ncfv_euler/2D/ncfv_euler2D.json \
  -k /time/iterations -v 10 --check-config
mpirun -np 4 ./app/acm2EQ3D.exe \
  ../cases/acm/3D/acm3D_cylinder_Re3000_coarse_SST_BDF2_LUSGS.json
```

部分Euler构建在schema JSON前打印以 `JSON:` 开头的注册日志；若重定向结果含这些行，先去掉注册日志再供标准JSON解析器使用。仓库的 `cases/update_schemas.sh` 已处理各入口schema更新流程。

## 3 不同模块的配置位置 {#dndsr_config_ch3}

| 目的 | Euler/CFV | ACM、ACMVariable | NCFV |
| --- | --- | --- | --- |
| 网格 | dataIOControl | meshSettings | mesh |
| 物性/湍流 | eulerSettings | acmSettings、turbulenceSettings | physics、physics.viscous |
| 初始场 | eulerSettings初值对象 | initialState及变密度扰动字段 | physics.initialPrimitive、initialField |
| 重构 | vfvSettings、implicitReconstructionControl | reconstructionSettings、vfvSettings | reconstruction、algorithm |
| 限制器 | limiterControl | reconstructionSettings | reconstruction.enableLimiter |
| 推进 | timeMarchControl、implicitCFLControl | timeMarchSettings | time |
| 收敛 | convergenceControl | timeMarchSettings | 显式步数/终止时间，当前无相同的隐式收敛选择 |
| 边界 | bcSettings、boundaryDefinition | boundaryConditions、defaultBoundaryType | physics.boundaryZones |
| 输出 | outputControl、dataIOControl | outputSettings | io |
| 重启 | restartState、dataIOControl | 仅常密度ACM的restartSettings | io重启字段 |

CFV是Euler/ACM共享的空间重构模块，没有独立网格流场入口和通用时间收敛JSON。NCFV的二次节点中心重构使用自己的配置结构，不能把 `vfvSettings` 粘贴给NCFV。

## 4 Euler/CFV 配置使用说明 {#dndsr_config_ch4}

### 4.1 状态、模型与初值

推荐状态对象为 `{"type":"primRhoP","state":[...]}`，适用于 `eulerSettings.farFieldStaticValue`、盒/平面初值和流动边界 `value`。旧的裸数组按 `consSensible` 解释，不会自动当作原始变量。

| type | 流动分量含义 | 备注 |
| --- | --- | --- |
| cons | rho、动量、总能量密度 | 反应热力学按相应总能量约定转换 |
| consSensible | rho、动量、显能密度 | 非反应理想气体时与相应守恒形式一致 |
| primRhoP | rho、速度、静压p | 扩展组分按质量分数输入 |
| primRhoT | rho、速度、温度T | EOS换算静压 |
| primTP | T、速度、静压p | 第一个分量是温度，由EOS换算rho |
| 上述名字加_phy | 相同分量顺序，使用物理单位 | 结合T0/rho0/U0/L0换算到代码单位 |

二维 `euler2D` 使用2速度；`euler`、SA、2EQ、EX 的二维入口仍保留3速度。被动/RANS附加分量跟随所选状态转换。修改对象时同时更新 `type` 和 `state`，避免merge-patch将旧数组按新类型重解释。`none/invalid` 属于内部哨兵，不作为有效算例状态。

初值先取远场基准，再按盒、平面和ExprTk规则设置局部区域；相交区域的效果应按求解器初始化顺序核对。EOS及参考尺度位于 `eulerSettings.idealGasProperty`。`muModel=0/1/2` 分别为常动力黏度、Sutherland、常运动黏度。

修改片段（二维四变量均匀来流）：

```json
{
  "solver": {"type":"Euler","discretization":"CFV","model":"NS_2D","fieldNVariables":4},
  "eulerSettings": {
    "farFieldStaticValue": {"type":"primRhoP","state":[1.0,1.0,0.0,1.0]}
  }
}
```

### 4.2 时间推进与收敛

| odeCode | 方法 | 推进性质 |
| --- | --- | --- |
| 0 | ESDIRK4 | 隐式，5阶段 |
| 101 | SSP-SDIRK4 | 隐式，3阶段 |
| 202 | ESIDRK3 | 隐式，3阶段 |
| 203 | Trapz / Crank–Nicolson | 隐式，单隐式阶段 |
| 204 | ESDIRK2 | 隐式，2阶段 |
| 1、102 | BDF2 | 隐式，两步历史 |
| 103 | Backward Euler | 隐式，一阶 |
| 2 | SSPRK3 | 显式，3阶段 |
| 401 | DITR | 隐式，两阶段耦合 |
| 411、412 | DITR-U2R2、DITR-U2R1 | 隐式，两阶段耦合 |
| 413 | DITR-U3R1 | 隐式，两阶段耦合、两步历史 |

Euler物理步长由 `dtImplicit` 等设置控制，内迭代伪时间步由 `implicitCFLControl` 控制。`steadyQuit=true` 是稳态模式控制。`convergenceControl.rhsThresholdInternal` 是内部相对RHS阈值；当前代码退出检查使用密度分量 `resRel(0)`。内部参考在首轮初始化，并逐分量吸收已遇到的残差最大值；非定常 `resBaseType=1` 还把物理RHS参考计入。它与ACM“本物理步固定的流场/湍流独立初值”规则不同。

`res_base` 主要控制外层显示残差的参考值；`recThreshold` 控制CFV重构；GMRES参数控制线性解预算。缩小其中一个阈值不等价于缩小另外两个。`normOrd=1/2` 分别为L1/L2，当前代码对3及以上取最大范数。

修改片段（非定常BDF2）：

```json
{
  "timeMarchControl": {"steadyQuit":false,"odeCode":1,"dtImplicit":0.01,"nTimeStep":1000,"tEnd":10.0},
  "convergenceControl": {"nTimeStepInternal":200,"nTimeStepInternalMin":5,"rhsThresholdInternal":0.001}
}
```

### 4.3 SA、两方程和EX反应流

SA选择 `eulerSA/eulerSA3D`，并配套 `ransModel="RANS_SA"`。两方程选择 `euler2EQ/euler2EQ3D`，再通过 `RANS_KOWilcox`、`RANS_KOSST`、`RANS_RKE` 选闭合模型。各模型的生产/涡黏度限幅配置只在相应模型下生效。SST不能只在普通四变量NS入口中改一个模型名启用。

EX选择 `eulerEX/eulerEX3D`，优先在 `solver.fieldNVariables` 中明确动态状态长度；也支持旧调用 `eulerEX.exe 14 config.json`。启用 `reactiveFlow.enabled` 需要Cantera构建和配套机理、组分顺序、状态长度。`nSpeciesOverride` 与 `thermoFile` 当前保留未使用；不要用它们绕过机理定义。当前输运实现只支持 `MixtureAveraged`。

`timeMarchControl.sourceStrangSplitting=1` 为物理时间化学半步—流动整步—化学半步；`linearSolverControl.sourceTauSplitting=1` 为伪时间源项分裂。化学 `chemAbsTol` 和 `reactorStepSettings` 容差分别作用于化学求解，不是全耦合流动收敛阈值。打开Strang分裂时，最新/输出RHS为流动部分，应按分裂后的物理量检查结果。

### 4.4 Euler 边界条件完整格式

`bcSettings` 为对象数组，`name` 必须匹配CGNS区域且不能重复。支持 `BCFar`、`BCWall`、`BCWallInvis`、`BCWallIsothermal`、`BCOut`、`BCOutP`、`BCIn`、`BCInPsTs`、`BCSym`、`BCSpecial`。

| 字段 | 含义和使用条件 |
| --- | --- |
| type、name | 必需，边界类型与区域名称 |
| value | BCFar/BCOut/BCOutP/BCIn使用完整带类型状态，必需；BCInPsTs使用原始总压/总温参数；等温壁必须提供壁温参数 |
| frameOption | 参考系处理选项，默认0；流动/壁面路径使用 |
| anchorOption | 流动边界锚点处理选项，默认0 |
| integrationOption | 边界积分选项，默认0 |
| specialOption | 壁面/特殊边界子类型编号，默认0 |
| rectifyOption | 对称边界校正选项，默认0 |
| valueExtra | 附加数值，如锚点位置；仅对应算法读取时生效 |
| __bcId | 输出的内部编号；输入时忽略 |

普通远场/入口value示例：`{"type":"BCFar","name":"farfield","value":{"type":"primRhoP","state":[1,1,0,1]}}`，适用于euler2D。黏性壁可用 `{"type":"BCWall","name":"wall"}`。BCOutP按完整状态中的静压处理。

BCInPsTs 的原始数组长度等于nVars，内容为物理单位 `[总压Pa,总温K,方向分量...,被动量/组分...]`；不是普通原始状态。BCWallIsothermal 原始数组同样长度为nVars，第0分量为壁温K。原始数组也可包装成 `{"type":"nonState","state":[...]}`。不能把这些特殊载荷声明成primRhoP。

平移/旋转周期关系由 `boundaryDefinition` 和Geom边界命名映射识别。`bcNameMapping` 不是物理BC数组的替代。复杂旋转周期、锚点和特殊编号应复制同模型的配套算例；选项不在所有边界类型上通用。

### 4.5 Euler 透明JSON对象的子字段

`dataIOControl.meshPartitionOptions`：`metisType="KWAY"` 或 `"RB"`；`metisUfactor=20`；`metisSeed=0`；`edgeWeightMethod=0`（1按面尺寸加权）；`metisNcuts=3`。这些是Geom的实际默认值，完整约束与ACM/NCFV划分表相同。

`dataIOControl.meshWallDistOptions`：`subdivide_quad=1`（壁面四边形细分），`method=0`（0暴力、1树搜索），`wallDistExecution=0`（0并行、1串行、>1分批），`minWallDist=1e-10`，`verbose=0`。这些属于Geom节点壁距构建；`eulerSettings.wallDist*` 为Euler自己的壁距相关路径，两组配置不要混写。

`timeMarchControl.odeSettingsExtra`：DITR/Hermite实现支持 `nMG`、`thetaMMG`、`coefIncMidMG`，分别为内部多重网格次数、相关耦合权重、中间增量系数；源码成员参考值为0、1、1，构造可另设次数。其他积分器对非空扩展键默认报错；不使用时保持空对象或配套默认值。

`linearSolverControl.coarseGridLinearSolverControlList` 为按层键建立的对象，每个条目包含后文列出的粗网格线性求解字段；不是数组。重启、划分网格、重构矩阵序列化对象均使用后文列出的7个序列化字段。

ACM配置修订：统一入口以initialState的rho*q尾部为初值，闭合和上下界字段仍用原始量q；新模型采用共同时间阶段和分块隐式修正。原四变量入口只接受层流。[完整实现说明](acm_model_state_zh.md)。

## 5 常密度 ACM 配置使用说明 {#dndsr_config_ch5}

### 5.1 物性、空间离散与湍流

层流主状态为[u,v,w,p]，SA和两方程入口分别追加rho*nuTilde和两个RANS守恒分量。详见[统一状态与入口](acm_model_state_zh.md)。`acmSettings.rho0` 是物理常密度；`dynamicViscosity` 是动力黏度，`nu=mu/rho0`。`beta2` 和 `alpha` 控制人工压缩预条件和特征速度，不能用来代替物理压缩性EOS。压力只支持 `PhysicalP`。

`reconstructionSettings.type` 可选 `FirstOrder`、`GreenGauss`、`Variational`。Variational的多项式次数由 `vfvSettings.maxOrder` 决定；2表示二次多项式。WBAP/CWBAP要求Variational。固定扫描由 `variationalIterations` 控制；正的 `variationalTolerance` 才开启受控重构。流场重构和湍流二阶重构分别设置。

`turbulenceSettings.model` 与入口配套：acm2D/acm3D只接受Laminar，acmSA2D/acmSA3D只接受SpalartAllmaras，acm2EQ2D/acm2EQ3D接受KOmegaWilcox、KOmegaSST、RealizableKEpsilon。SA守恒尾部为 `[rho*nuTilde]`；SST为 `[rho*k,rho*omega]`。启用RANS需正分子黏度和黏性通量，初值、正下限、上限、涡黏度比和壁距要相互一致。

### 5.2 时间推进和绝对/相对收敛选择

| integrator | 使用场景 |
| --- | --- |
| ExplicitSSPRK3 | 显式稳态伪时间推进 |
| ImplicitEulerBlockJacobi | 稳态隐式Block-Jacobi |
| ImplicitEulerLUSGS | 稳态隐式LU-SGS |
| ImplicitEulerGMRES | 稳态隐式GMRES |
| BDF2DualTimeLUSGS | 非定常物理BDF2，LU-SGS内迭代 |
| BDF2DualTimeGMRES | 非定常物理BDF2，GMRES内迭代 |

`steadyRelativeTolerance`、`steadyAdaptiveCFL` 只用于稳态隐式算法；BDF2中分别设置0、false。BDF2物理步使用 `physicalTimeStep`，内迭代步使用固定伪时间或CFL局部步。启动及变量正性受限时的历史回退由程序处理。

`dualTimeConvergenceCriterion` 只选 **Absolute** 或 **Relative**，选择的是收敛方式。所有活跃流场和湍流方程都要达到所选方式；没有Both/Either/Flow/Turbulence选项。

| 模式 | 判据 |
| --- | --- |
| Absolute | 使用implicitTolerance；逐方程路径结合dualTimeResidualScales形成绝对目标 |
| Relative | 使用本物理步固定初始缺陷乘相对系数；流场和湍流分别保存初值，逐方程路径也各自保存初值 |

两组相对系数为正时，分别使用流场组RMS和湍流组RMS。将系数设为 `1e-3` 表示各组相对本步各自初值下降1000倍。对应组标量为0时，使用逐方程 `dualTimeRelativeTolerances`；Relative要求每个活跃组或方程有有效正系数。组RMS路径要求对应 `dualTimeResidualScales` 为1。初始残差为0时相对目标也是0。

修改片段（SST非定常）：

```json
{
  "timeMarchSettings": {
    "integrator":"BDF2DualTimeLUSGS",
    "physicalTimeStep":0.01,
    "maxImplicitIterations":200,
    "implicitTolerance":1e-8,
    "dualTimeConvergenceCriterion":"Relative",
    "dualTimeFlowRelativeTolerance":1e-3,
    "dualTimeTurbulenceRelativeTolerance":1e-3,
    "dualTimeRequireConvergence":true,
    "steadyRelativeTolerance":0,
    "steadyAdaptiveCFL":false
  }
}
```

在此Relative模式中 `implicitTolerance=1e-8` 可保留，但不作为另一条自动退出途径。若要按绝对方式停止，改为 `"Absolute"`。缺失选择字段的旧配置兼容为Absolute。`dualTimeRequireConvergence=true` 时达到迭代上限仍未收敛会报错，不提交该物理步的历史；false仅用于有限预算诊断推进。

### 5.3 边界、输出与重启

`boundaryConditions[]` 使用 `type/name/value`，普通常密度值为 `[u,v,w,p]`。支持Euler式BC名字，也提供 FarField、VelocityInlet、PressureOutlet、NoSlipWall、SlipWall、Symmetry 别名。frame/anchor/integration/rectify等选项中有保留字段；不能假定其具备Euler同名字段的全部功能。

常密度ACM没有温度/能量方程，`BCWallIsothermal` 与 `BCWall` 采用相同速度及压力法向梯度处理；`BCInPsTs` 的前三个值是给定面速度，压力从内场外推，不解释为Euler总压总温。`BCOutP` 指定第4分量压力；`BCSpecial` 当前仅支持 `specialOption=0` 的给定状态。

`outputSettings.interval=0` 关闭场输出，包括初始化场。`restartSettings.flowFile` 从VTK-HDF场初始化稳态运行，空路径关闭重启；`completedSteps` 指已完成稳态步数，`writerRanks` 指原写出MPI规模，`requireTurbulence` 可要求湍流数据完整。该格式不含BDF2历史，程序拒绝用它初始化双时间推进；与Euler的重启格式不互通。常密度BDF2固定预算重构会在每次缺陷求值前强制重置系数，避免求值依赖上一次调用。更详细的当前圆柱示例见 [ACM专项配置说明](acm_json_configuration_manual_zh.md)。

## 6 变密度 ACMVariable 配置使用说明 {#dndsr_config_ch6}

流场固定 `[rho,mx,my,mz,p]`，不是 `[rho,u,v,w,p]`。`acmSettings.rho0` 是限制器归一化参考值，实际密度来自状态；`densityFloor` 为正密度下限。湍流历史采用rho*q保守形式，与流场质量通量相配套。

本模块使用完整单文件、CFV重构与相同的五类湍流名字，但配置表有独立边界：没有常密度ACM新增的 `dualTimeConvergenceCriterion`、组相对阈值、稳态自适应CFL、受控变分重构额外字段或 `restartSettings`。原生隐式内迭代仍由 `implicitTolerance` 的全局RMS绝对值控制。

`physicalODECode=-1`、`pseudoODECode=-1` 时使用原生integrator。选择共享ODE接口时只能启用其中一个。支持编号集合 `-1,0,1,2,101,102,103,202,203,204,401,411,412,413`，但物理路径拒绝2与101，因为当前缺少相应端点投影；伪时间路径可用。压力为代数约束，物理时间质量矩阵压力分量为0；其历史处理不同于可压缩Euler。

修改片段（变密度均匀流）：

```json
{
  "solver":{"type":"ACMVariable","discretization":"CFV","model":"VariableDensity2D","fieldNVariables":5},
  "initialState":[2.0,2.0,0.0,0.0,0.0],
  "boundaryValue":[2.0,2.0,0.0,0.0,0.0]
}
```

该例rho=2、mx=2，故u=1。`initialDensityAmplitude` 与 `initialDensityWaveNumber` 用于余弦密度测试；幅度须保证整个场仍高于densityFloor。`BCSpecial.specialOption=1` 是精确平移余弦密度边界测试，`valueExtra=[相对幅值,kx,ky,kz]`，边界基准value为保守动量状态。`BCInPsTs` 在本模块表示密度/速度入口，不含总温；`BCOutP` 使用第5分量压力。当前没有与常密度ACM等价的重启加载配置。

## 7 节点中心 NCFV 配置使用说明 {#dndsr_config_ch7}

### 7.1 网格、重构与物性

NCFV读取O1 CGNS网格，在节点构建对偶控制体，采用二次多项式重构。`algorithm.mode="EfficientDifferential"` 使用预计算权重；`"TraditionalQuadrature"` 使用传统体/面积分。`quadratureOrder` 和 `surfaceQuadratureOrder` 不改变多项式次数，运行时积分设置主要作用于传统路径。

`reconstruction.method` 支持 LeastSquares、SVDLeastSquares、Variational及表列中文别名。模板大小、环数、距离权重、秩阈值和条件数控制共同决定重构可用性。限制器是单系数Barth–Jespersen，不能直接指定CFV的CWBAP参数。

`physics.viscous.enabled=true` 开启层流黏性/导热通量；黏度模型为Constant/Sutherland/DensityProportional。当前没有SA、SST或反应组分配置入口。NCFV Roe使用Harten–Yee H2熵修正，Roe_M2当前拒绝；使用后文NCFV枚举，勿照搬Euler可接受列表。

### 7.2 初值、边界与周期

初值基准为 `physics.initialPrimitive`，二维 `[rho,u,v,p]`、三维 `[rho,u,v,w,p]`。`initialField` 可配置CSV节点场、解析等熵涡、盒、半空间和ExprTk规则。CSV首列为全局节点编号，编号基数由 `nodeFileIndexBase` 指定；其后按Primitive或Conservative顺序给值，不能在Conservative中写速度和压力。

边界逐区域配置在 `physics.boundaryZones`。支持 FarField、SlipWall、Symmetry、NoSlipAdiabaticWall、NoSlipIsothermalWall、SupersonicInlet、SupersonicOutlet、PressureOutlet、Periodic。`requireBoundaryZoneCoverage=true` 要求CGNS所有外区域均覆盖。`strongState` 在每个RK阶段后强制节点入口/壁面状态。

周期平移由 `mesh.periodicLengths` 与 `periodicBoundaryPairs` 给定；配对列表必须主/从成对。空列表按Periodic区域条目顺序推断。零长度不自动形成一个非零周期。

### 7.3 显式推进、输出与重启

当前驱动是显式SSPRK3，通过 `time.iterations` 和 `endTime` 控制停止，没有ACM式隐式Absolute/Relative选择。若指定非负 `endTime` 且开启CFL时间步，必须 `useLocalTimeStep=false`；程序拒绝以各节点局部伪时间步推进共同物理终止时间。

修改片段（固定步长非定常）：

```json
{
  "time":{"iterations":1000,"timeStep":0.001,"useCFLTimeStep":false,"useLocalTimeStep":false,"endTime":1.0},
  "io":{"writeVTK":true,"writeInitial":true,"writeFinal":true,"outputInterval":100}
}
```

`io.restartInput` 指基础路径，不带 `.dnds.h5` 或 `.dir` 后缀。`restartInterval=0` 只关闭周期重启，`writeFinalRestart` 单独控制最终重启。`writeResolvedConfiguration` 可保存实际合并后的参数，便于复现。

## 8 EulerP 的 Python/Host/CUDA 配置 {#dndsr_config_ch8}

EulerP由Python `DNDSR.EulerP.EulerP_Solver.Solver` 组织网格、FV、Evaluator和推进。没有独立 `eulerP.exe config.json` 原生入口；其字典可以用JSON保存，再由脚本读取并调用对应API。若修改C++绑定，应先构建并安装Python组件后使用。

| 接口/配置 | 字段及当前行为 |
| --- | --- |
| ReadMesh(mesh_file, dim, other_options) | dim为2/3，选项包括read_mode、partition_options、periodic_geometry、periodic_tolerance、elevation、bisect、serializer_factory、reorder_parts、reorder_inner_parts、build_serial_out |
| ReadMesh兼容名 | meshElevation→elevation、meshDirectBisect→bisect、serializerFactory→serializer_factory、inner_process_parts→reorder_parts、second_level_parts→reorder_inner_parts；readMeshMode的Serial/Parallel/Distributed映射cgns/h5/h5 |
| BuildFV(fv_settings) | FiniteVolumeSettings四字段：maxOrder=1、intOrder=1、ignoreMeshGeometryDeficiency=false、nIterCellSmoothScale=15；不是完整38字段VRSettings |
| Physics字典 | params含gamma=1.4、mu0=1e-100、cp=1、Rg=1、TRef=273.15、muModel=0；reference_values为向量；cv未列入JSON接口 |
| 高层BuildEval默认 | 覆盖mu0=1e-10，reference_values为5个1；创建空BC输入列表 |
| Evaluator.setConfig字典 | threadsPerBlock=128（CUDA块线程数）、pullAllInputArgs=true（核启动前等待输入MPI拉取）、serializeCUDAExecution=false（CUDA串行化诊断）；未知键报错 |
| BCInput | type、name、value；类型字符串为Far、Wall、WallInvis、WallIsothermal、Out、OutP、In、InPsTs、Sym、Special |
| runningDevice | Host或CUDA后端；CUDA需相应构建及运行环境 |
| CalculateOneRHS | zero_grad=false；true在重构后清零梯度，用于一阶空间求值 |
| IntegrateDt_ExplicitInterval | tStart、tEnd、CFL、step0、max_step=100000；显式RK3，按全局最小谱步长推进 |

EulerP当前流动状态为5个守恒量 `[rho,mx,my,mz,rhoE]`；其内部原始变量第4下标存 **rho*e内能密度**，不像Euler普通原始变量存压力。`muModel` 已暴露，但当前黏度核返回mu0；不要把名称的存在理解为所有黏度模型都已实现。

ReadMesh当前默认：`periodic_tolerance=1e-9`、`read_mode=null`（按.cgns/.dnds.h5扩展名识别）、`elevation=""`、`bisect=0`、`reorder_parts=1`、`reorder_inner_parts=1`、`build_serial_out=true`；未指定partition_options时用Geom默认划分选项。`periodic_geometry` 是传给SetPeriodicGeometry的字典，可含translation1/2/3、rotationCenter1/2/3、eulerAngles1/2/3的三分量向量；不要假定缺省字典适合自己的周期网格。`serializer_factory` 是Python序列化对象，不能直接用普通JSON值代替。

当前EulerP物理边界专门化仍有TODO/assert分支，不能把Euler的边界配置当作已完整实现。使用此实验接口时参考仓库的周期测试和已实现路径。高层BuildEval不接收任意physics/BC字典参数；需按公开对象接口构建Physics、BCHandler与Evaluator，保持引用和数据一致。

## 9 配置修改与验证流程 {#dndsr_config_ch9}

1. 选同族、同维度、同模型的完整算例；同时配套schema与可执行程序。
2. 明确输入是速度、动量、压力还是能量；校对所有状态数组长度和单位。
3. 修改网格、物性、边界、推进及输出位置；相对路径按启动目录核对。
4. 先执行schema校验和 `--check-config`；再用少量步数运行，检查网格区域匹配、残差与物理量。
5. 记录实际合并配置、构建选项、MPI规模和输出位置；做网格/时间步/迭代阈值敏感性检查时每次只改明确的配置项。

同名字段在不同模块可能具有不同含义，例如rho0、value和implicitTolerance。后续完整表中只在对应模块解释该字段，不把四个族合并成一张可互换的配置。

## 10 Euler/CFV 完整参数表 {#dndsr_config_ch10}

本表共372项。字段组标题是JSON路径前缀；拼接表内参数名得到完整路径。默认参考严格取导出的schema，未声明时标明“未声明”。schema下界为0不表示运行时允许所有0值；正文和中文说明中的正值/组合限制仍需满足。

### 10.1 `solver`

| 参数 | 默认参考 | 类型 | 中文含义与约束 |
| --- | --- | --- | --- |
| `type` | `"Euler"` | 字符串 | 方程族，必须与可执行程序一致；具体常量见入口对照表。 schema约束：固定="Euler"。 |
| `discretization` | `"CFV"` | 字符串 | 空间离散模块：Euler/ACM/ACMVariable为CFV，节点中心求解器为NCFV。 schema约束：固定="CFV"。 |
| `model` | 见入口对照表 | 字符串 | 编译模型标识；不能仅修改此字符串把程序转换为另一个模型。 schema约束：固定模型值按第1章各入口确定；EX状态长度动态。入口差异见第1章。 |
| `fieldNVariables` | 见入口对照表 | 整数 | 状态分量数；固定模型不可改变，动态EX模型需与组分和编译/运行设置一致。 schema约束：固定模型值按第1章各入口确定；EX状态长度动态。入口差异见第1章。 |

<!-- config-fields: euler|solver.type;solver.discretization;solver.model;solver.fieldNVariables -->

### 10.2 `timeMarchControl`

| 参数 | 默认参考 | 类型 | 中文含义与约束 |
| --- | --- | --- | --- |
| `dtImplicit` | `1e+100` | 实数 | 物理隐式步长上限；稳态常用极大值，非定常时按物理时间分辨率设置。 schema约束：≥0。 |
| `dtImplicitMin` | `0` | 实数 | 物理时间步下限。 schema约束：≥0。 |
| `nTimeStep` | `1000000` | 整数 | 外层时间步数上限。 schema约束：≥0。 |
| `steadyQuit` | `false` | 布尔 | 启用稳态收敛退出；程序将外层步数压到不超过1，并使用无限大物理步长。 |
| `useRestart` | `false` | 布尔 | 从restartState指定的重启文件初始化。 |
| `useImplicitPP` | `false` | 布尔 | 启用隐式修正的正性保持处理。 |
| `rhsFPPMode` | `0` | 整数 | 右端通量正性保持模式编号；沿用配套算例，勿与重构限制器混淆。 |
| `rhsFPPScale` | `1` | 实数 | 右端通量正性保持的缩放系数。 |
| `rhsFPPRelax` | `0.9` | 实数 | 右端通量正性保持的松弛系数。 schema约束：≥0；≤1。 |
| `incrementPPRelax` | `0.9` | 实数 | 解增量正性保持处理的松弛系数。 schema约束：≥0；≤1。 |
| `odeCode` | `0` | 整数 | 时间积分器编号，完整编号表见时间推进章节；1/102为BDF2，103为后退Euler，2为显式SSPRK3。 |
| `tEnd` | `3e+200` | 实数 | 非定常计算终止物理时间。 |
| `odeSetting1` | `0` | 实数 | 积分器参数1；DITR系列用作第二阶段位置c2。 |
| `odeSetting2` | `0` | 实数 | 积分器参数2；DITR系列控制后退Euler启动。 |
| `odeSetting3` | `0` | 实数 | 积分器参数3；DITR系列用作thetaM1。 |
| `odeSetting4` | `0` | 实数 | 积分器参数4；401为mask，413为thetaM2，其他编号按对应积分器处理。 |
| `odeSettingsExtra` | 未声明 | JSON | 积分器专用JSON对象；不是通用容差表，已实现键见扩展格式章节。 |
| `partitionMeshOnly` | `false` | 布尔 | 只构建并写出划分网格，然后在坐标变换和求解器设置前退出。 |
| `dtIncreaseLimit` | `2` | 实数 | 每步允许的物理时间步最大增长倍数。 schema约束：≥1。 |
| `dtIncreaseAfterCount` | `0` | 整数 | 经过指定步数后才允许增加时间步。 schema约束：≥0。 |
| `dtCFLLimitScale` | `1e+100` | 实数 | 用CFL谱半径限制物理时间步时的比例系数。 |
| `useDtPPLimit` | `false` | 布尔 | 启用正性保持时间步限制。 |
| `dtPPLimitRelax` | `0.8` | 实数 | 正性保持时间步限制的松弛系数。 schema约束：≥0；≤1。 |
| `dtPPLimitScale` | `1` | 实数 | 正性保持时间步限制的缩放系数。 schema约束：≥0。 |
| `sourceStrangSplitting` | `0` | 整数 | 反应流物理时间Strang分裂：0关，1开；开启时输出/最新RHS只表示流动部分。 |

<!-- config-fields: euler|timeMarchControl.dtImplicit;timeMarchControl.dtImplicitMin;timeMarchControl.nTimeStep;timeMarchControl.steadyQuit;timeMarchControl.useRestart;timeMarchControl.useImplicitPP;timeMarchControl.rhsFPPMode;timeMarchControl.rhsFPPScale;timeMarchControl.rhsFPPRelax;timeMarchControl.incrementPPRelax;timeMarchControl.odeCode;timeMarchControl.tEnd;timeMarchControl.odeSetting1;timeMarchControl.odeSetting2;timeMarchControl.odeSetting3;timeMarchControl.odeSetting4;timeMarchControl.odeSettingsExtra;timeMarchControl.partitionMeshOnly;timeMarchControl.dtIncreaseLimit;timeMarchControl.dtIncreaseAfterCount;timeMarchControl.dtCFLLimitScale;timeMarchControl.useDtPPLimit;timeMarchControl.dtPPLimitRelax;timeMarchControl.dtPPLimitScale;timeMarchControl.sourceStrangSplitting -->

### 10.3 `implicitReconstructionControl`

| 参数 | 默认参考 | 类型 | 中文含义与约束 |
| --- | --- | --- | --- |
| `useExplicit` | `false` | 布尔 | 使用显式重构路径。 |
| `nInternalRecStep` | `1` | 整数 | 每次内部重构的最大扫描数。 schema约束：≥1。 |
| `zeroGrads` | `false` | 布尔 | 重构前清零梯度/系数。 |
| `recLinearScheme` | `0` | 整数 | 重构线性求解方法编号；0为SOR，1为GMRES，2为FPCG。 |
| `nGmresSpace` | `5` | 整数 | 重构GMRES的Krylov子空间维数。 schema约束：≥1。 |
| `nGmresIter` | `10` | 整数 | 重构GMRES迭代预算。 schema约束：≥1。 |
| `gmresRecScale` | `1` | 整数 | 重构GMRES缩放模式。 |
| `fpcgResetScheme` | `0` | 整数 | 柔性预条件共轭梯度历史重置策略。 |
| `fpcgResetThres` | `0.6` | 实数 | FPCG历史重置阈值。 schema约束：≥0；≤1。 |
| `fpcgResetReport` | `0` | 整数 | FPCG重置信息报告频率。 |
| `fpcgMaxPHistory` | `20` | 整数 | FPCG保留的预条件搜索方向历史长度。 schema约束：≥1。 |
| `recThreshold` | `1e-05` | 实数 | 重构迭代相对阈值；SOR比较增量与首轮增量，GMRES/PCG按各自线性残差处理。 schema约束：≥0。 |
| `nRecConsolCheck` | `1` | 整数 | 重构迭代控制台报告间隔。 schema约束：≥1。 |
| `nRecMultiplyForZeroedGrad` | `1` | 整数 | 梯度清零时重构迭代预算的倍率。 schema约束：≥1。 |
| `storeRecInc` | `false` | 布尔 | 保留重构增量供后续使用。 |
| `dampRecIncDTau` | `false` | 布尔 | 按伪时间步阻尼重构增量。 |
| `zeroRecForSteps` | `0` | 整数 | 前若干外层步强制使用零重构。 schema约束：≥0。 |
| `zeroRecForStepsInternal` | `0` | 整数 | 前若干内迭代强制使用零重构。 schema约束：≥0。 |

<!-- config-fields: euler|implicitReconstructionControl.useExplicit;implicitReconstructionControl.nInternalRecStep;implicitReconstructionControl.zeroGrads;implicitReconstructionControl.recLinearScheme;implicitReconstructionControl.nGmresSpace;implicitReconstructionControl.nGmresIter;implicitReconstructionControl.gmresRecScale;implicitReconstructionControl.fpcgResetScheme;implicitReconstructionControl.fpcgResetThres;implicitReconstructionControl.fpcgResetReport;implicitReconstructionControl.fpcgMaxPHistory;implicitReconstructionControl.recThreshold;implicitReconstructionControl.nRecConsolCheck;implicitReconstructionControl.nRecMultiplyForZeroedGrad;implicitReconstructionControl.storeRecInc;implicitReconstructionControl.dampRecIncDTau;implicitReconstructionControl.zeroRecForSteps;implicitReconstructionControl.zeroRecForStepsInternal -->

### 10.4 `outputControl`

| 参数 | 默认参考 | 类型 | 中文含义与约束 |
| --- | --- | --- | --- |
| `nConsoleCheck` | `1` | 整数 | 外层步控制台报告间隔。 schema约束：≥1。 |
| `nConsoleCheckInternal` | `1` | 整数 | 内迭代控制台报告间隔。 schema约束：≥1。 |
| `consoleOutputMode` | `0` | 整数 | 控制台模式：0基本信息，1包含壁面力。 |
| `consoleOutputEveryFix` | `0` | 整数 | 修正迭代达到该间隔时强制报告。 |
| `consoleMainOutputFormat` | 见14章 D1 | 数组 | 外层控制台输出格式字符串列表；占位符须匹配程序提供的数据。 |
| `consoleMainOutputFormatInternal` | 见14章 D2 | 数组 | 内迭代控制台输出格式字符串列表。 |
| `logfileOutputTitles` | 见14章 D3 | 数组 | CSV日志列标题列表。 |
| `nPrecisionLog` | `10` | 整数 | 日志浮点输出精度。 schema约束：≥1。 |
| `dataOutAtInit` | `false` | 布尔 | 初始化完成后输出场数据。 |
| `restartOutAtInit` | `false` | 布尔 | 初始化完成后写重启文件。 |
| `meshOutAtInit` | `false` | 布尔 | 初始化阶段输出CGNS网格。 |
| `nDataOut` | `10` | 整数 | 按外层步计数的场输出间隔。 schema约束：≥1。 |
| `nDataOutC` | `1` | 整数 | 按收敛步计数的场输出间隔。 schema约束：≥1。 |
| `nDataOutInternal` | `10000` | 整数 | 内迭代场输出间隔。 schema约束：≥1。 |
| `nDataOutCInternal` | `10000` | 整数 | 内部收敛时的场输出间隔。 |
| `nRestartOut` | `2147483647` | 整数 | 按外层步计数的重启输出间隔。 |
| `nRestartOutC` | `2147483647` | 整数 | 按收敛步计数的重启输出间隔。 |
| `nRestartOutInternal` | `2147483647` | 整数 | 内迭代重启输出间隔。 |
| `nRestartOutCInternal` | `2147483647` | 整数 | 内部收敛时的重启输出间隔。 |
| `nTimeAverageOut` | `2147483647` | 整数 | 时间平均场的外层步输出间隔。 |
| `nTimeAverageOutC` | `2147483647` | 整数 | 时间平均场的收敛步输出间隔。 |
| `tDataOut` | `3e+200` | 实数 | 按物理时间间隔输出场数据。 |
| `lazyCoverDataOutput` | `false` | 布尔 | 覆盖先前场输出文件，减少多时刻文件存储。 |
| `useCollectiveTimer` | `false` | 布尔 | 使用MPI集体计时汇总性能信息。 |

<!-- config-fields: euler|outputControl.nConsoleCheck;outputControl.nConsoleCheckInternal;outputControl.consoleOutputMode;outputControl.consoleOutputEveryFix;outputControl.consoleMainOutputFormat;outputControl.consoleMainOutputFormatInternal;outputControl.logfileOutputTitles;outputControl.nPrecisionLog;outputControl.dataOutAtInit;outputControl.restartOutAtInit;outputControl.meshOutAtInit;outputControl.nDataOut;outputControl.nDataOutC;outputControl.nDataOutInternal;outputControl.nDataOutCInternal;outputControl.nRestartOut;outputControl.nRestartOutC;outputControl.nRestartOutInternal;outputControl.nRestartOutCInternal;outputControl.nTimeAverageOut;outputControl.nTimeAverageOutC;outputControl.tDataOut;outputControl.lazyCoverDataOutput;outputControl.useCollectiveTimer -->

### 10.5 `implicitCFLControl`

| 参数 | 默认参考 | 类型 | 中文含义与约束 |
| --- | --- | --- | --- |
| `CFL` | `10` | 实数 | 隐式内迭代伪时间CFL；不是物理时间步。 schema约束：≥0。 |
| `nForceLocalStartStep` | `2147483647` | 整数 | 达到指定步数时强制开始局部伪时间推进。 schema约束：≥0。 |
| `nCFLRampStart` | `2147483647` | 整数 | CFL渐增开始步数。 schema约束：≥0。 |
| `nCFLRampLength` | `2147483647` | 整数 | CFL渐增持续步数。 schema约束：≥1。 |
| `CFLRampEnd` | `0` | 实数 | 渐增结束时的CFL。 |
| `useLocalDt` | `true` | 布尔 | 按单元使用局部伪时间步；false使用统一步长。 |
| `nSmoothDTau` | `0` | 整数 | 局部伪时间步平滑次数。 schema约束：≥0。 |
| `RANSRelax` | `1` | 实数 | RANS方程内迭代松弛系数。 schema约束：≥0；≤1。 |

<!-- config-fields: euler|implicitCFLControl.CFL;implicitCFLControl.nForceLocalStartStep;implicitCFLControl.nCFLRampStart;implicitCFLControl.nCFLRampLength;implicitCFLControl.CFLRampEnd;implicitCFLControl.useLocalDt;implicitCFLControl.nSmoothDTau;implicitCFLControl.RANSRelax -->

### 10.6 `convergenceControl`

| 参数 | 默认参考 | 类型 | 中文含义与约束 |
| --- | --- | --- | --- |
| `nTimeStepInternal` | `20` | 整数 | 每个时间步内部迭代上限；0表示不设置次数上限。 schema约束：≥0。 |
| `nTimeStepInternalMin` | `5` | 整数 | 允许收敛退出前至少进行的内部迭代次数。 schema约束：≥0。 |
| `nAnchorUpdate` | `1` | 整数 | 锚点更新间隔。 schema约束：≥1。 |
| `nAnchorUpdateStart` | `0` | 整数 | 开始锚点更新的迭代位置。 schema约束：≥0。 |
| `rhsThresholdInternal` | `1e-10` | 实数 | 内部RHS相对阈值；当前退出分支比较密度分量resRel(0)，不是流场/SST独立组阈值。 schema约束：≥0。 |
| `res_base` | `0` | 实数 | 外层显示残差归一化基准；0自动设置，不等同于内迭代参考值。 |
| `resBaseType` | `0` | 整数 | 残差参考策略；1在非定常下把物理RHS范数计入内部参考。 |
| `mergeMultiResidual` | `0` | 整数 | 多阶段残差合并策略；0关闭合并。 |
| `normOrd` | `1` | 整数 | 残差范数：1为L1，2为L2，当前实现3及以上取最大范数。 schema约束：≥1。 |
| `useVolWiseResidual` | `false` | 布尔 | 使用单元体积加权残差。 |
| `useCLDriver` | `false` | 布尔 | 启用按目标升力系数调整迎角的驱动器。 |

<!-- config-fields: euler|convergenceControl.nTimeStepInternal;convergenceControl.nTimeStepInternalMin;convergenceControl.nAnchorUpdate;convergenceControl.nAnchorUpdateStart;convergenceControl.rhsThresholdInternal;convergenceControl.res_base;convergenceControl.resBaseType;convergenceControl.mergeMultiResidual;convergenceControl.normOrd;convergenceControl.useVolWiseResidual;convergenceControl.useCLDriver -->

### 10.7 `dataIOControl`

| 参数 | 默认参考 | 类型 | 中文含义与约束 |
| --- | --- | --- | --- |
| `uniqueStamps` | `true` | 布尔 | 为每次运行生成唯一输出标记。 |
| `meshRotZ` | `0` | 实数 | 绕z轴旋转网格的角度，单位度。 |
| `meshScale` | `1` | 实数 | 网格坐标缩放倍率。 schema约束：≥0。 |
| `meshElevation` | `0` | 整数 | 几何升阶：0关闭，1从O1提升到O2；不直接指定解重构阶次。 |
| `meshElevationInternalSmoother` | `0` | 整数 | 升阶内部平滑器：0局部，1耦合。 |
| `meshElevationIter` | `1000` | 整数 | 升阶几何平滑迭代次数。 |
| `meshElevationNSearch` | `30` | 整数 | 升阶RBF邻居搜索数量。 schema约束：≥1。 |
| `meshElevationRBFRadius` | `1` | 实数 | 升阶RBF支撑半径。 schema约束：≥0。 |
| `meshElevationRBFPower` | `1` | 实数 | 升阶RBF幂参数。 |
| `meshElevationRBFKernel` | `"InversedDistanceA1"` | 字符串 | 升阶RBF核函数名称。 |
| `meshElevationMaxIncludedAngle` | `15` | 实数 | 升阶最大夹角，单位度。 |
| `meshElevationRefDWall` | `0.001` | 实数 | 升阶使用的参考壁距。 schema约束：≥0。 |
| `meshElevationBoundaryMode` | `0` | 整数 | 升阶边界范围：0仅黏性壁，1包括无黏壁。 |
| `meshDirectBisect` | `0` | 整数 | 网格二分细化次数。 schema约束：≥0。 |
| `meshReorderCells` | `0` | 整数 | 局部单元重排：0自然顺序，1启用重排。 |
| `meshBuildWallDist` | `0` | 整数 | 构建壁距场：0关闭，1开启。 |
| `meshWallDistOptions` | `{"subdivide_quad": 1, "method": 0, "wallDistExecution": 0, "minWallDist": 1e-10, "verbose": 0}` | 对象 | Geom壁距构建选项对象；完整子字段见扩展格式章节。 |
| `meshFormat` | `0` | 整数 | 原始网格格式编号；使用相应算例的格式设置。 |
| `meshPartitionOptions` | `{"metisType": "KWAY", "metisUfactor": 20, "metisSeed": 0, "edgeWeightMethod": 0, "metisNcuts": 3}` | 对象 | Geom网格划分选项对象；包括METIS算法、种子、负载和边权。 |
| `meshFile` | `"data/mesh/NACA0012_WIDE_H3.cgns"` | 字符串 | 输入网格路径，按启动工作目录解析。 |
| `meshFilePartitionedInput` | `""` | 字符串 | readMeshMode=1/2的已划分网格路径；空值按当前MPI进程数推导，可带序列化后缀。 |
| `outPltName` | `"data/out/debugData_"` | 字符串 | 场输出文件路径前缀。 |
| `outLogName` | `""` | 字符串 | 日志文件路径前缀；空值使用outPltName。 |
| `outRestartName` | `""` | 字符串 | 重启输出路径前缀；空值使用outPltName。 |
| `outPltMode` | `0` | 整数 | 场输出模式：0串行汇集，1分布式输出。 |
| `readMeshMode` | `0` | 整数 | 0读取原始网格并划分，1读取同MPI规模预划分网格，2读取H5并重新划分。 |
| `outPltTecplotFormat` | `true` | 布尔 | 输出Tecplot格式。 |
| `outPltVTKFormat` | `false` | 布尔 | 输出VTK XML格式。 |
| `outPltVTKHDFFormat` | `false` | 布尔 | 输出VTK HDF格式。 |
| `outAtPointData` | `true` | 布尔 | 输出节点插值数据。 |
| `outAtCellData` | `true` | 布尔 | 输出单元中心数据。 |
| `nASCIIPrecision` | `5` | 整数 | ASCII场数据精度。 schema约束：≥1。 |
| `vtuFloatEncodeMode` | `"binary"` | 字符串 | VTU浮点编码；binary为二进制，ascii为文本。 schema约束：枚举："binary"、"ascii"。 |
| `hdfChunkSize` | `32768` | 整数 | 场输出HDF5分块尺寸。 schema约束：≥0。 |
| `hdfDeflateLevel` | `0` | 整数 | 场输出HDF5压缩级别。 schema约束：≥0；≤9。 |
| `hdfCollOnMeta` | `true` | 布尔 | HDF5元数据集体I/O。 |
| `hdfCollOnData` | `false` | 布尔 | HDF5数据数组集体I/O。 |
| `outVolumeData` | `true` | 布尔 | 输出体网格数据。 |
| `outBndData` | `false` | 布尔 | 输出边界网格数据。 |
| `outCellScalarNames` | `[]` | 数组 | 附加单元标量输出名称列表；须来自输出注册表。 |
| `outBndScalarNames` | `[]` | 数组 | 附加边界标量名称列表，按边界邻接单元选取。 |
| `serializerSaveURec` | `false` | 布尔 | 重启文件中保存重构系数。 |
| `allowAsyncPrintData` | `false` | 布尔 | 允许异步场输出。 |
| `rectifyNearPlane` | `0` | 整数 | 近坐标平面节点校正位掩码：1=x，2=y，4=z，可按位组合。 |
| `rectifyNearPlaneThres` | `1e-10` | 实数 | 近坐标平面节点校正阈值。 schema约束：≥0。 |
| `meshPartitionedReaderType` | `"H5"` | 字符串 | 预划分网格序列化读取后端JSON/H5。 schema约束：枚举："JSON"、"H5"。 |

<!-- config-fields: euler|dataIOControl.uniqueStamps;dataIOControl.meshRotZ;dataIOControl.meshScale;dataIOControl.meshElevation;dataIOControl.meshElevationInternalSmoother;dataIOControl.meshElevationIter;dataIOControl.meshElevationNSearch;dataIOControl.meshElevationRBFRadius;dataIOControl.meshElevationRBFPower;dataIOControl.meshElevationRBFKernel;dataIOControl.meshElevationMaxIncludedAngle;dataIOControl.meshElevationRefDWall;dataIOControl.meshElevationBoundaryMode;dataIOControl.meshDirectBisect;dataIOControl.meshReorderCells;dataIOControl.meshBuildWallDist;dataIOControl.meshWallDistOptions;dataIOControl.meshFormat;dataIOControl.meshPartitionOptions;dataIOControl.meshFile;dataIOControl.meshFilePartitionedInput;dataIOControl.outPltName;dataIOControl.outLogName;dataIOControl.outRestartName;dataIOControl.outPltMode;dataIOControl.readMeshMode;dataIOControl.outPltTecplotFormat;dataIOControl.outPltVTKFormat;dataIOControl.outPltVTKHDFFormat;dataIOControl.outAtPointData;dataIOControl.outAtCellData;dataIOControl.nASCIIPrecision;dataIOControl.vtuFloatEncodeMode;dataIOControl.hdfChunkSize;dataIOControl.hdfDeflateLevel;dataIOControl.hdfCollOnMeta;dataIOControl.hdfCollOnData;dataIOControl.outVolumeData;dataIOControl.outBndData;dataIOControl.outCellScalarNames;dataIOControl.outBndScalarNames;dataIOControl.serializerSaveURec;dataIOControl.allowAsyncPrintData;dataIOControl.rectifyNearPlane;dataIOControl.rectifyNearPlaneThres;dataIOControl.meshPartitionedReaderType -->

### 10.8 `dataIOControl.restartWriter`

| 参数 | 默认参考 | 类型 | 中文含义与约束 |
| --- | --- | --- | --- |
| `type` | `"H5"` | 字符串 | 序列化后端：JSON或H5；读取端必须与文件格式匹配。 schema约束：枚举："JSON"、"H5"。 |
| `hdfDeflateLevel` | `0` | 整数 | HDF5 deflate压缩级别，0关闭压缩。 schema约束：≥0；≤9。 |
| `hdfChunkSize` | `0` | 整数 | HDF5数组分块尺寸。 schema约束：≥0。 |
| `hdfCollOnData` | `false` | 布尔 | HDF5数据数组集体I/O开关。 |
| `hdfCollOnMeta` | `true` | 布尔 | HDF5元数据集体I/O开关。 |
| `jsonBinaryDeflateLevel` | `5` | 整数 | JSON所携带二进制数组的压缩级别。 schema约束：≥0；≤9。 |
| `jsonUseCodecOnUInt8` | `true` | 布尔 | JSON序列化时对uint8数组使用编码器。 |

<!-- config-fields: euler|dataIOControl.restartWriter.type;dataIOControl.restartWriter.hdfDeflateLevel;dataIOControl.restartWriter.hdfChunkSize;dataIOControl.restartWriter.hdfCollOnData;dataIOControl.restartWriter.hdfCollOnMeta;dataIOControl.restartWriter.jsonBinaryDeflateLevel;dataIOControl.restartWriter.jsonUseCodecOnUInt8 -->

### 10.9 `dataIOControl.meshPartitionedWriter`

| 参数 | 默认参考 | 类型 | 中文含义与约束 |
| --- | --- | --- | --- |
| `type` | `"H5"` | 字符串 | 序列化后端：JSON或H5；读取端必须与文件格式匹配。 schema约束：枚举："JSON"、"H5"。 |
| `hdfDeflateLevel` | `0` | 整数 | HDF5 deflate压缩级别，0关闭压缩。 schema约束：≥0；≤9。 |
| `hdfChunkSize` | `0` | 整数 | HDF5数组分块尺寸。 schema约束：≥0。 |
| `hdfCollOnData` | `false` | 布尔 | HDF5数据数组集体I/O开关。 |
| `hdfCollOnMeta` | `true` | 布尔 | HDF5元数据集体I/O开关。 |
| `jsonBinaryDeflateLevel` | `5` | 整数 | JSON所携带二进制数组的压缩级别。 schema约束：≥0；≤9。 |
| `jsonUseCodecOnUInt8` | `true` | 布尔 | JSON序列化时对uint8数组使用编码器。 |

<!-- config-fields: euler|dataIOControl.meshPartitionedWriter.type;dataIOControl.meshPartitionedWriter.hdfDeflateLevel;dataIOControl.meshPartitionedWriter.hdfChunkSize;dataIOControl.meshPartitionedWriter.hdfCollOnData;dataIOControl.meshPartitionedWriter.hdfCollOnMeta;dataIOControl.meshPartitionedWriter.jsonBinaryDeflateLevel;dataIOControl.meshPartitionedWriter.jsonUseCodecOnUInt8 -->

### 10.10 `boundaryDefinition`

| 参数 | 默认参考 | 类型 | 中文含义与约束 |
| --- | --- | --- | --- |
| `PeriodicTranslation1` | `[1, 0, 0]` | 数组 | 第1组周期平移向量，三分量。 |
| `PeriodicTranslation2` | `[0, 1, 0]` | 数组 | 第2组周期平移向量，三分量。 |
| `PeriodicTranslation3` | `[0, 0, 1]` | 数组 | 第3组周期平移向量，三分量。 |
| `PeriodicRotationCent1` | `[0, 0, 0]` | 数组 | 第1组旋转周期中心，三分量。 |
| `PeriodicRotationCent2` | `[0, 0, 0]` | 数组 | 第2组旋转周期中心，三分量。 |
| `PeriodicRotationCent3` | `[0, 0, 0]` | 数组 | 第3组旋转周期中心，三分量。 |
| `PeriodicRotationEulerAngles1` | `[0, 0, 0]` | 数组 | 第1组旋转周期Euler角，单位度。 |
| `PeriodicRotationEulerAngles2` | `[0, 0, 0]` | 数组 | 第2组旋转周期Euler角，单位度。 |
| `PeriodicRotationEulerAngles3` | `[0, 0, 0]` | 数组 | 第3组旋转周期Euler角，单位度。 |
| `periodicTolerance` | `1e-08` | 实数 | 周期节点匹配容差。 schema约束：≥0。 |

<!-- config-fields: euler|boundaryDefinition.PeriodicTranslation1;boundaryDefinition.PeriodicTranslation2;boundaryDefinition.PeriodicTranslation3;boundaryDefinition.PeriodicRotationCent1;boundaryDefinition.PeriodicRotationCent2;boundaryDefinition.PeriodicRotationCent3;boundaryDefinition.PeriodicRotationEulerAngles1;boundaryDefinition.PeriodicRotationEulerAngles2;boundaryDefinition.PeriodicRotationEulerAngles3;boundaryDefinition.periodicTolerance -->

### 10.11 `limiterControl`

| 参数 | 默认参考 | 类型 | 中文含义与约束 |
| --- | --- | --- | --- |
| `useLimiter` | `true` | 布尔 | 启用重构斜率限制器。 |
| `usePPRecLimiter` | `true` | 布尔 | 启用正性保持重构限制器。 |
| `useViscousLimited` | `true` | 布尔 | 黏性通量使用受限重构。 |
| `smoothIndicatorProcedure` | `0` | 整数 | 平滑指标处理编号。 |
| `limiterProcedure` | `0` | 整数 | 限制器变体：0为WBAP/V2，1为CWBAP/V3。 |
| `nPartialLimiterStart` | `2147483647` | 整数 | 开始部分限制的外层步数。 |
| `nPartialLimiterStartLocal` | `2147483647` | 整数 | 开始局部部分限制的步数。 |
| `preserveLimited` | `false` | 布尔 | 在步间保留受限重构系数。 |
| `ppRecLimiterCompressToMean` | `true` | 布尔 | 正性保持限制时向单元均值压缩。 |

<!-- config-fields: euler|limiterControl.useLimiter;limiterControl.usePPRecLimiter;limiterControl.useViscousLimited;limiterControl.smoothIndicatorProcedure;limiterControl.limiterProcedure;limiterControl.nPartialLimiterStart;limiterControl.nPartialLimiterStartLocal;limiterControl.preserveLimited;limiterControl.ppRecLimiterCompressToMean -->

### 10.12 `linearSolverControl`

| 参数 | 默认参考 | 类型 | 中文含义与约束 |
| --- | --- | --- | --- |
| `jacobiCode` | `1` | 整数 | 预条件器：0 Jacobi，1 Gauss-Seidel，2 ILU。 |
| `sgsIter` | `0` | 整数 | SGS扫描迭代预算。 schema约束：≥0。 |
| `sgsWithRec` | `0` | 整数 | SGS修正与重构耦合。 |
| `gmresCode` | `0` | 整数 | 线性解法：0 LU-SGS，1 GMRES，2 LU-SGS加GMRES。 |
| `gmresScale` | `0` | 整数 | GMRES缩放：0无，1参考状态，2平均状态。 |
| `nGmresSpace` | `10` | 整数 | 流场GMRES的Krylov子空间维数。 schema约束：≥1。 |
| `nGmresIter` | `2` | 整数 | 流场GMRES外层迭代预算。 schema约束：≥1。 |
| `nSgsConsoleCheck` | `100` | 整数 | SGS控制台报告间隔。 schema约束：≥1。 |
| `nGmresConsoleCheck` | `100` | 整数 | GMRES控制台报告间隔。 schema约束：≥1。 |
| `initWithLastURecInc` | `false` | 布尔 | 用上一轮重构增量初始化GMRES。 |
| `sourceTauSplitting` | `0` | 整数 | 反应源伪时间分裂：0关，1开；区别于物理Strang分裂。 |
| `multiGridLP` | `0` | 整数 | 多重网格层级参数；0关闭。 |
| `multiGridLPInnerNIter` | `4` | 整数 | 多重网格内部迭代预算。 schema约束：≥1。 |
| `multiGridLPStartIter` | `0` | 整数 | 开始启用多重网格的迭代位置。 schema约束：≥0。 |
| `multiGridLPInnerNSee` | `10` | 整数 | 多重网格内部控制台报告间隔。 schema约束：≥1。 |

<!-- config-fields: euler|linearSolverControl.jacobiCode;linearSolverControl.sgsIter;linearSolverControl.sgsWithRec;linearSolverControl.gmresCode;linearSolverControl.gmresScale;linearSolverControl.nGmresSpace;linearSolverControl.nGmresIter;linearSolverControl.nSgsConsoleCheck;linearSolverControl.nGmresConsoleCheck;linearSolverControl.initWithLastURecInc;linearSolverControl.sourceTauSplitting;linearSolverControl.multiGridLP;linearSolverControl.multiGridLPInnerNIter;linearSolverControl.multiGridLPStartIter;linearSolverControl.multiGridLPInnerNSee -->

### 10.13 `linearSolverControl.coarseGridLinearSolverControlList.<key>`

| 参数 | 默认参考 | 类型 | 中文含义与约束 |
| --- | --- | --- | --- |
| `jacobiCode` | `0` | 整数 | 预条件器：0 Jacobi，1 Gauss-Seidel，2 ILU。作用于对应粗网格层。 |
| `sgsIter` | `0` | 整数 | SGS扫描迭代预算。作用于对应粗网格层。 schema约束：≥0。 |
| `gmresCode` | `0` | 整数 | 线性解法：0 LU-SGS，1 GMRES，2 LU-SGS加GMRES。作用于对应粗网格层。 |
| `gmresScale` | `0` | 整数 | GMRES缩放：0无，1参考状态，2平均状态。作用于对应粗网格层。 |
| `nGmresIter` | `2` | 整数 | 流场GMRES外层迭代预算。作用于对应粗网格层。 schema约束：≥1。 |
| `nSgsConsoleCheck` | `100` | 整数 | SGS控制台报告间隔。作用于对应粗网格层。 schema约束：≥1。 |
| `nGmresConsoleCheck` | `100` | 整数 | GMRES控制台报告间隔。作用于对应粗网格层。 schema约束：≥1。 |
| `multiGridNIter` | `-1` | 整数 | 该粗网格层前平滑次数；-1自动选择。 |
| `multiGridNIterPost` | `0` | 整数 | 该粗网格层后平滑次数。 schema约束：≥0。 |
| `centralSmoothInputResidual` | `0` | 整数 | 在粗网格上平滑输入残差。 |

<!-- config-fields: euler|linearSolverControl.coarseGridLinearSolverControlList.<key>.jacobiCode;linearSolverControl.coarseGridLinearSolverControlList.<key>.sgsIter;linearSolverControl.coarseGridLinearSolverControlList.<key>.gmresCode;linearSolverControl.coarseGridLinearSolverControlList.<key>.gmresScale;linearSolverControl.coarseGridLinearSolverControlList.<key>.nGmresIter;linearSolverControl.coarseGridLinearSolverControlList.<key>.nSgsConsoleCheck;linearSolverControl.coarseGridLinearSolverControlList.<key>.nGmresConsoleCheck;linearSolverControl.coarseGridLinearSolverControlList.<key>.multiGridNIter;linearSolverControl.coarseGridLinearSolverControlList.<key>.multiGridNIterPost;linearSolverControl.coarseGridLinearSolverControlList.<key>.centralSmoothInputResidual -->

### 10.14 `linearSolverControl.directPrecControl`

| 参数 | 默认参考 | 类型 | 中文含义与约束 |
| --- | --- | --- | --- |
| `useDirectPrec` | `false` | 布尔 | 启用直接预条件路径；需相应构建依赖。 |
| `iluCode` | `0` | 整数 | ILU填充设置：0不增加填充，-1完全分解。 |
| `orderingCode` | `-2147483648` | 整数 | 矩阵重排：INT32_MIN自动，0自然顺序，1 METIS，2 MMD。 |

<!-- config-fields: euler|linearSolverControl.directPrecControl.useDirectPrec;linearSolverControl.directPrecControl.iluCode;linearSolverControl.directPrecControl.orderingCode -->

### 10.15 `timeAverageControl`

| 参数 | 默认参考 | 类型 | 中文含义与约束 |
| --- | --- | --- | --- |
| `enabled` | `false` | 布尔 | 启用解场的时间平均。 |

<!-- config-fields: euler|timeAverageControl.enabled -->

### 10.16 `others`

| 参数 | 默认参考 | 类型 | 中文含义与约束 |
| --- | --- | --- | --- |
| `nFreezePassiveInner` | `0` | 整数 | 前若干内迭代冻结被动标量。 schema约束：≥0。 |
| `axisSymmetric` | `0` | 整数 | 轴对称模式编号；0关闭。 |
| `printRecMatrix` | `false` | 布尔 | 将重构矩阵输出到文件，供诊断使用。 |

<!-- config-fields: euler|others.nFreezePassiveInner;others.axisSymmetric;others.printRecMatrix -->

### 10.17 `others.recMatrixWriter`

| 参数 | 默认参考 | 类型 | 中文含义与约束 |
| --- | --- | --- | --- |
| `type` | `"H5"` | 字符串 | 序列化后端：JSON或H5；读取端必须与文件格式匹配。 schema约束：枚举："JSON"、"H5"。 |
| `hdfDeflateLevel` | `0` | 整数 | HDF5 deflate压缩级别，0关闭压缩。 schema约束：≥0；≤9。 |
| `hdfChunkSize` | `0` | 整数 | HDF5数组分块尺寸。 schema约束：≥0。 |
| `hdfCollOnData` | `false` | 布尔 | HDF5数据数组集体I/O开关。 |
| `hdfCollOnMeta` | `true` | 布尔 | HDF5元数据集体I/O开关。 |
| `jsonBinaryDeflateLevel` | `5` | 整数 | JSON所携带二进制数组的压缩级别。 schema约束：≥0；≤9。 |
| `jsonUseCodecOnUInt8` | `true` | 布尔 | JSON序列化时对uint8数组使用编码器。 |

<!-- config-fields: euler|others.recMatrixWriter.type;others.recMatrixWriter.hdfDeflateLevel;others.recMatrixWriter.hdfChunkSize;others.recMatrixWriter.hdfCollOnData;others.recMatrixWriter.hdfCollOnMeta;others.recMatrixWriter.jsonBinaryDeflateLevel;others.recMatrixWriter.jsonUseCodecOnUInt8 -->

### 10.18 `restartState`

| 参数 | 默认参考 | 类型 | 中文含义与约束 |
| --- | --- | --- | --- |
| `iStep` | `-1` | 整数 | 重启时的外层步编号。 |
| `iStepInternal` | `-1` | 整数 | 重启时的内迭代编号。 |
| `odeCodePrev` | `-1` | 整数 | 重启状态记录的前次积分器编号。 |
| `lastRestartFile` | `""` | 字符串 | 主重启文件路径。 |
| `otherRestartFile` | `""` | 字符串 | 备用/异维度重启文件路径。 |
| `otherRestartStoreDim` | `[0]` | 数组 | 备用重启数据的维度映射设置。 |

<!-- config-fields: euler|restartState.iStep;restartState.iStepInternal;restartState.odeCodePrev;restartState.lastRestartFile;restartState.otherRestartFile;restartState.otherRestartStoreDim -->

### 10.19 `eulerSettings`

| 参数 | 默认参考 | 类型 | 中文含义与约束 |
| --- | --- | --- | --- |
| `useScalarJacobian` | `false` | 布尔 | 用标量近似隐式Jacobian。 |
| `useRoeJacobian` | `false` | 布尔 | 使用Roe型隐式Jacobian。 |
| `noRsOnWall` | `false` | 布尔 | 壁面不调用Riemann求解器。 |
| `noGRPOnWall` | `false` | 布尔 | 壁面关闭GRP处理。 |
| `ignoreSourceTerm` | `false` | 布尔 | 忽略源项；会改变物理模型，应按具体目的设置。 |
| `direct2ndRecMethod` | `1` | 整数 | 直接二阶重构方法编号。 |
| `specialBuiltinInitializer` | `0` | 整数 | 内置测试初值编号；含义取决于模型和测试分支。 |
| `uRecAlphaCompressPower` | `1` | 实数 | 重构alpha正性压缩的幂参数。 |
| `uRecBetaCompressPower` | `1` | 实数 | 重构beta压缩的幂参数。 |
| `forceVolURecBeta` | `true` | 布尔 | 强制采用体积相关重构beta设置。 |
| `ppEpsIsRelaxed` | `false` | 布尔 | 使用松弛的正性保持epsilon。 |
| `RANSTopLimit` | `100000` | 实数 | RANS变量上限；当前主要作用于SA的nuTilde。 schema约束：≥0。 |
| `RANSBottomLimit` | `0.01` | 实数 | RANS变量下限。 schema约束：≥0。 |
| `riemannSolverType` | `"Roe"` | 字符串 | 主Riemann通量类型；具体可接受名字由该模块枚举决定。 |
| `riemannSolverTypeAux` | `"UnknownRS"` | 字符串 | 辅助通量类型；UnknownRS继承主通量。 |
| `riemannSolverTypeWall` | `"UnknownRS"` | 字符串 | 壁面通量类型；UnknownRS继承主通量。 |
| `rsFixScale` | `1` | 实数 | Riemann通量熵修正比例。 |
| `rsIncFScale` | `1` | 实数 | Riemann增量通量缩放。 |
| `rsMeanValueEig` | `0` | 整数 | Riemann平均特征值处理模式。 |
| `rsRotateScheme` | `0` | 整数 | Riemann旋转处理方案编号。 |
| `minWallDist` | `1e-12` | 实数 | 壁距正下限。 schema约束：≥0。 |
| `wallDistExection` | `0` | 整数 | 壁距执行模式：0并行，1串行；保留源码拼写Exection。 |
| `wallDistRefineMax` | `1` | 实数 | 壁面细化搜索上限。 |
| `wallDistScheme` | `0` | 整数 | 壁距算法编号。 |
| `wallDistCellLoadSize` | `32768` | 整数 | 壁距计算的单元批次大小。 schema约束：≥1。 |
| `wallDistIter` | `1000` | 整数 | 壁距方程迭代预算。 schema约束：≥1。 |
| `wallDistLinSolver` | `0` | 整数 | 壁距线性解法：0 Jacobi，1 GMRES。 |
| `wallDistResTol` | `0.0001` | 实数 | 壁距方程残差容差。 schema约束：≥0。 |
| `wallDistIterStart` | `100` | 整数 | 开始壁距方程迭代的位置。 schema约束：≥0。 |
| `wallDistPoissonP` | `2` | 整数 | 壁距Poisson模型幂参数。 |
| `wallDistDTauScale` | `100` | 实数 | 壁距方程伪时间尺度。 schema约束：≥0。 |
| `wallDistNJacobiSweep` | `10` | 整数 | 壁距方程Jacobi扫描次数。 schema约束：≥1。 |
| `SADESScale` | `3e+200` | 实数 | SA-DES长度尺度系数。 |
| `SADESMode` | `1` | 整数 | SA-DES模式编号；仅用于对应SA路径。 |
| `SAVersion` | `0` | 整数 | SA版本：0当前版本，1旧版本；影响旋转系数、应变率与源项Jacobian。 |
| `ransModel` | `"RANS_None"` | 字符串 | RANS闭合枚举：RANS_None、RANS_SA、RANS_KOWilcox、RANS_KOSST、RANS_RKE；须匹配入口变量数。 |
| `ransUseQCR` | `0` | 整数 | 启用RANS二次本构关系修正。 |
| `ransSARotCorrection` | `1` | 整数 | 启用SA旋转修正。 |
| `ransEigScheme` | `0` | 整数 | RANS特征值处理方案。 |
| `ransForce2nd` | `0` | 整数 | 强制RANS输运使用二阶处理。 |
| `ransSource2nd` | `0` | 整数 | RANS源项二阶处理。 |
| `source2nd` | `0` | 整数 | 通用源项二阶处理。 |
| `usePrimGradInVisFlux` | `0` | 整数 | 黏性通量采用原始变量梯度。 |
| `useSourceGradFixGG` | `0` | 整数 | 源项梯度采用Green-Gauss修正。 |
| `nCentralSmoothStep` | `0` | 整数 | 中心平滑次数。 schema约束：≥0。 |
| `centralSmoothEps` | `0.5` | 实数 | 中心平滑epsilon系数。 |
| `pointImplicitSourceUpdateOut` | `0` | 整数 | 打印化学点隐式Newton残差比最小/最大值：0关，1开。 |
| `reactiveSourceScale` | `1` | 实数 | 直接缩放化学RHS和Jacobian；0用于关闭反应的诊断比较。 schema约束：≥0；≤1。 |
| `constMassForce` | `[0, 0, 0]` | 数组 | 三分量恒定单位质量体力。 |
| `cLDriverBCNames` | `[]` | 数组 | 用于升阻力积分/CL驱动的边界名称列表。 |
| `farFieldStaticValue` | `{"type": "consSensible", "state": []}` | 数组/对象 | 远场状态，推荐{type,state}形式；分量数与模型一致，同时更新类型和数组。 |

<!-- config-fields: euler|eulerSettings.useScalarJacobian;eulerSettings.useRoeJacobian;eulerSettings.noRsOnWall;eulerSettings.noGRPOnWall;eulerSettings.ignoreSourceTerm;eulerSettings.direct2ndRecMethod;eulerSettings.specialBuiltinInitializer;eulerSettings.uRecAlphaCompressPower;eulerSettings.uRecBetaCompressPower;eulerSettings.forceVolURecBeta;eulerSettings.ppEpsIsRelaxed;eulerSettings.RANSTopLimit;eulerSettings.RANSBottomLimit;eulerSettings.riemannSolverType;eulerSettings.riemannSolverTypeAux;eulerSettings.riemannSolverTypeWall;eulerSettings.rsFixScale;eulerSettings.rsIncFScale;eulerSettings.rsMeanValueEig;eulerSettings.rsRotateScheme;eulerSettings.minWallDist;eulerSettings.wallDistExection;eulerSettings.wallDistRefineMax;eulerSettings.wallDistScheme;eulerSettings.wallDistCellLoadSize;eulerSettings.wallDistIter;eulerSettings.wallDistLinSolver;eulerSettings.wallDistResTol;eulerSettings.wallDistIterStart;eulerSettings.wallDistPoissonP;eulerSettings.wallDistDTauScale;eulerSettings.wallDistNJacobiSweep;eulerSettings.SADESScale;eulerSettings.SADESMode;eulerSettings.SAVersion;eulerSettings.ransModel;eulerSettings.ransUseQCR;eulerSettings.ransSARotCorrection;eulerSettings.ransEigScheme;eulerSettings.ransForce2nd;eulerSettings.ransSource2nd;eulerSettings.source2nd;eulerSettings.usePrimGradInVisFlux;eulerSettings.useSourceGradFixGG;eulerSettings.nCentralSmoothStep;eulerSettings.centralSmoothEps;eulerSettings.pointImplicitSourceUpdateOut;eulerSettings.reactiveSourceScale;eulerSettings.constMassForce;eulerSettings.cLDriverBCNames;eulerSettings.farFieldStaticValue -->

### 10.20 `eulerSettings.SAConfig`

| 参数 | 默认参考 | 类型 | 中文含义与约束 |
| --- | --- | --- | --- |
| `productionLimit` | `100000` | 实数 | 该模型生产项与破坏项之间的限幅倍率，仅选中模型时生效。 schema约束：≥0。 |

<!-- config-fields: euler|eulerSettings.SAConfig.productionLimit -->

### 10.21 `eulerSettings.KOmegaConfig`

| 参数 | 默认参考 | 类型 | 中文含义与约束 |
| --- | --- | --- | --- |
| `turbulentViscosityLimit` | `100000` | 实数 | 该模型涡黏度限幅倍率，仅选中模型时生效。 schema约束：≥0。 |
| `productionLimit` | `20` | 实数 | 该模型生产项与破坏项之间的限幅倍率，仅选中模型时生效。 schema约束：≥0。 |

<!-- config-fields: euler|eulerSettings.KOmegaConfig.turbulentViscosityLimit;eulerSettings.KOmegaConfig.productionLimit -->

### 10.22 `eulerSettings.KOmegaSSTConfig`

| 参数 | 默认参考 | 类型 | 中文含义与约束 |
| --- | --- | --- | --- |
| `turbulentViscosityLimit` | `100000` | 实数 | 该模型涡黏度限幅倍率，仅选中模型时生效。 schema约束：≥0。 |
| `productionLimit` | `20` | 实数 | 该模型生产项与破坏项之间的限幅倍率，仅选中模型时生效。 schema约束：≥0。 |

<!-- config-fields: euler|eulerSettings.KOmegaSSTConfig.turbulentViscosityLimit;eulerSettings.KOmegaSSTConfig.productionLimit -->

### 10.23 `eulerSettings.RKEConfig`

| 参数 | 默认参考 | 类型 | 中文含义与约束 |
| --- | --- | --- | --- |
| `turbulentViscosityLimit` | `100000` | 实数 | 该模型涡黏度限幅倍率，仅选中模型时生效。 schema约束：≥0。 |

<!-- config-fields: euler|eulerSettings.RKEConfig.turbulentViscosityLimit -->

### 10.24 `eulerSettings.reactorStepSettings`

| 参数 | 默认参考 | 类型 | 中文含义与约束 |
| --- | --- | --- | --- |
| `rtol` | `1e-10` | 实数 | Cantera反应器积分相对容差。 schema约束：≥0。 |
| `atol` | `1e-18` | 实数 | Cantera反应器积分绝对容差。 schema约束：≥0。 |
| `maxOrder` | `0` | 整数 | Cantera反应器最大积分阶次；非正值保留Cantera默认。 |
| `maxSteps` | `10000000` | 整数 | Cantera反应器最大内部步数。 schema约束：≥1。 |

<!-- config-fields: euler|eulerSettings.reactorStepSettings.rtol;eulerSettings.reactorStepSettings.atol;eulerSettings.reactorStepSettings.maxOrder;eulerSettings.reactorStepSettings.maxSteps -->

### 10.25 `eulerSettings.frameConstRotation`

| 参数 | 默认参考 | 类型 | 中文含义与约束 |
| --- | --- | --- | --- |
| `enabled` | `false` | 布尔 | 启用恒定转速旋转参考系。 |
| `axis` | `[0, 0, 1]` | 数组 | 旋转轴三分量单位向量。 |
| `center` | `[0, 0, 0]` | 数组 | 旋转中心坐标三分量。 |
| `rpm` | `0` | 实数 | 转速，单位转/分钟。 |

<!-- config-fields: euler|eulerSettings.frameConstRotation.enabled;eulerSettings.frameConstRotation.axis;eulerSettings.frameConstRotation.center;eulerSettings.frameConstRotation.rpm -->

### 10.26 `eulerSettings.cLDriverSettings`

| 参数 | 默认参考 | 类型 | 中文含义与约束 |
| --- | --- | --- | --- |
| `AOAInit` | `0` | 实数 | 初始迎角，单位度。 |
| `AOAAxis` | `"z"` | 字符串 | 迎角旋转轴x/y/z。 schema约束：枚举："x"、"y"、"z"。 |
| `CL0Axis` | `"y"` | 字符串 | 升力参考方向x/y/z。 schema约束：枚举："x"、"y"、"z"。 |
| `CD0Axis` | `"x"` | 字符串 | 阻力参考方向x/y/z。 schema约束：枚举："x"、"y"、"z"。 |
| `refArea` | `1` | 实数 | 力系数参考面积。 schema约束：≥0。 |
| `refDynamicPressure` | `0.5` | 实数 | 力系数参考动压。 schema约束：≥0。 |
| `targetCL` | `0` | 实数 | 目标升力系数。 |
| `CLIncrementRelax` | `0.25` | 实数 | 迎角增量欠松弛系数。 schema约束：≥0；≤1。 |
| `thresholdTargetRatio` | `0.5` | 实数 | 接近目标CL时缩紧阈值的比例。 schema约束：≥0；≤1。 |
| `nIterStartDrive` | `2147483647` | 整数 | 开始CL驱动的迭代位置。 schema约束：≥0。 |
| `nIterConvergeMin` | `50` | 整数 | CL收敛检查前最少迭代数。 schema约束：≥1。 |
| `CLconvergeThreshold` | `0.001` | 实数 | 短窗口CL波动收敛阈值。 schema约束：≥0。 |
| `CLconvergeWindow` | `10` | 整数 | 短窗口采样长度。 schema约束：≥2。 |
| `CLconvergeLongWindow` | `100` | 整数 | 达到目标后退出所用长窗口长度。 schema约束：≥1。 |
| `CLconvergeLongThreshold` | `0.0001` | 实数 | 长窗口CL波动阈值。 schema约束：≥0。 |
| `CLconvergeLongStrictAoA` | `false` | 布尔 | 迎角发生更新时重置长窗口计数。 |

<!-- config-fields: euler|eulerSettings.cLDriverSettings.AOAInit;eulerSettings.cLDriverSettings.AOAAxis;eulerSettings.cLDriverSettings.CL0Axis;eulerSettings.cLDriverSettings.CD0Axis;eulerSettings.cLDriverSettings.refArea;eulerSettings.cLDriverSettings.refDynamicPressure;eulerSettings.cLDriverSettings.targetCL;eulerSettings.cLDriverSettings.CLIncrementRelax;eulerSettings.cLDriverSettings.thresholdTargetRatio;eulerSettings.cLDriverSettings.nIterStartDrive;eulerSettings.cLDriverSettings.nIterConvergeMin;eulerSettings.cLDriverSettings.CLconvergeThreshold;eulerSettings.cLDriverSettings.CLconvergeWindow;eulerSettings.cLDriverSettings.CLconvergeLongWindow;eulerSettings.cLDriverSettings.CLconvergeLongThreshold;eulerSettings.cLDriverSettings.CLconvergeLongStrictAoA -->

### 10.27 `eulerSettings.boxInitializers[]`

| 参数 | 默认参考 | 类型 | 中文含义与约束 |
| --- | --- | --- | --- |
| `x0` | `0` | 实数 | 初值盒区域x坐标最小值。 |
| `x1` | `0` | 实数 | 初值盒区域x坐标最大值。 |
| `y0` | `0` | 实数 | 初值盒区域y坐标最小值。 |
| `y1` | `0` | 实数 | 初值盒区域y坐标最大值。 |
| `z0` | `0` | 实数 | 初值盒区域z坐标最小值。 |
| `z1` | `0` | 实数 | 初值盒区域z坐标最大值。 |
| `v` | `{"type": "consSensible", "state": []}` | 数组/对象 | 区域内初值；Euler用带类型状态，NCFV用与维度匹配的原始变量数组。 |

<!-- config-fields: euler|eulerSettings.boxInitializers[].x0;eulerSettings.boxInitializers[].x1;eulerSettings.boxInitializers[].y0;eulerSettings.boxInitializers[].y1;eulerSettings.boxInitializers[].z0;eulerSettings.boxInitializers[].z1;eulerSettings.boxInitializers[].v -->

### 10.28 `eulerSettings.planeInitializers[]`

| 参数 | 默认参考 | 类型 | 中文含义与约束 |
| --- | --- | --- | --- |
| `a` | `0` | 实数 | 初值半空间ax+by+cz+h≥0的x系数。 |
| `b` | `0` | 实数 | 初值半空间ax+by+cz+h≥0的y系数。 |
| `c` | `0` | 实数 | 初值半空间ax+by+cz+h≥0的z系数。 |
| `h` | `0` | 实数 | 初值半空间ax+by+cz+h≥0的偏置。 |
| `v` | `{"type": "consSensible", "state": []}` | 数组/对象 | 选中半空间内的初值；状态格式与本模块一致。 |

<!-- config-fields: euler|eulerSettings.planeInitializers[].a;eulerSettings.planeInitializers[].b;eulerSettings.planeInitializers[].c;eulerSettings.planeInitializers[].h;eulerSettings.planeInitializers[].v -->

### 10.29 `eulerSettings.exprtkInitializers[]`

| 参数 | 默认参考 | 类型 | 中文含义与约束 |
| --- | --- | --- | --- |
| `exprs` | `[]` | 数组 | ExprTk表达式行列表，按换行拼接；用于位置相关初值。 |
| `stateType` | 未声明 | 字符串 | 表达式生成数组的状态类型；与远场支持的五种类型及其_phy形式一致。 schema约束：枚举："cons"、"consSensible"、"primRhoP"、"primRhoT"、"primTP"、"cons_phy"、"consSensible_phy"、"primRhoP_phy"、"primRhoT_phy"、"primTP_phy"。 |

<!-- config-fields: euler|eulerSettings.exprtkInitializers[].exprs;eulerSettings.exprtkInitializers[].stateType -->

### 10.30 `eulerSettings.idealGasProperty`

| 参数 | 默认参考 | 类型 | 中文含义与约束 |
| --- | --- | --- | --- |
| `gamma` | `1.4` | 实数 | 理想气体比热比，有限且大于1。 schema约束：≥1。 |
| `Rgas` | `287` | 实数 | 物理单位的比气体常数，有限且大于0。 schema约束：≥2.22507385851e-308。 |
| `muGas` | `1` | 实数 | 分子动力黏度参考值，单位Pa·s。 schema约束：≥0。 |
| `prGas` | `0.72` | 实数 | 分子Prandtl数。 schema约束：≥0。 |
| `prTurb` | `0.9` | 实数 | 湍流Prandtl数。 schema约束：≥2.22507385851e-308。 |
| `scTurb` | `0.9` | 实数 | 组分湍流扩散共用Schmidt数。 schema约束：≥2.22507385851e-308。 |
| `TRef` | `273.15` | 实数 | Sutherland参考温度，单位K。 |
| `CSutherland` | `110.4` | 实数 | Sutherland常数，单位K。 |
| `muModel` | `1` | 整数 | 分子黏度模型：0常黏度，1 Sutherland，2常运动黏度。 |
| `T0` | `1` | 实数 | 无量纲化参考温度K，有限且为正。 schema约束：≥2.22507385851e-308。 |
| `rho0` | `1` | 实数 | 无量纲化参考密度kg/m³，有限且为正。 schema约束：≥2.22507385851e-308。 |
| `U0` | `1` | 实数 | 无量纲化参考速度m/s，有限且为正。 schema约束：≥2.22507385851e-308。 |
| `L0` | `1` | 实数 | 无量纲化参考长度m，有限且为正。 schema约束：≥2.22507385851e-308。 |

<!-- config-fields: euler|eulerSettings.idealGasProperty.gamma;eulerSettings.idealGasProperty.Rgas;eulerSettings.idealGasProperty.muGas;eulerSettings.idealGasProperty.prGas;eulerSettings.idealGasProperty.prTurb;eulerSettings.idealGasProperty.scTurb;eulerSettings.idealGasProperty.TRef;eulerSettings.idealGasProperty.CSutherland;eulerSettings.idealGasProperty.muModel;eulerSettings.idealGasProperty.T0;eulerSettings.idealGasProperty.rho0;eulerSettings.idealGasProperty.U0;eulerSettings.idealGasProperty.L0 -->

### 10.31 `eulerSettings.reactiveFlow`

| 参数 | 默认参考 | 类型 | 中文含义与约束 |
| --- | --- | --- | --- |
| `enabled` | `false` | 布尔 | 启用多组分反应流；需要EX模型与Cantera构建支持。 schema约束：NS/SA/2EQ入口固定false；EX入口允许true/false。 |
| `mechanismFile` | `""` | 字符串 | Cantera YAML反应机理文件路径，按启动目录解析。 |
| `thermoFile` | `""` | 字符串 | 保留字段，当前未使用；热力学数据由机理YAML提供。 |
| `transportModel` | `"MixtureAveraged"` | 字符串 | 请求的Cantera输运模型；当前只实现MixtureAveraged。 |
| `CFLScale` | `1` | 实数 | 化学点隐式伪时间倍率。 schema约束：≥2.22507385851e-308。 |
| `chemRelaxEps` | `0` | 实数 | 化学点隐式最小伪时间步。 schema约束：≥0。 |
| `chemAbsTol` | `1e-10` | 实数 | 化学点隐式残差绝对容差；不是流场物理步收敛阈值。 schema约束：≥0。 |
| `TBase` | `0` | 实数 | 显焓/显能记账的基温K；非正值采用各组分Cantera温度下限的最小值。 |
| `nSpeciesOverride` | `0` | 整数 | 保留字段，当前未使用；组分数从机理读取。 |
| `useCellTWarmCache` | `true` | 布尔 | 温度反演使用逐单元热启动缓存。 |

<!-- config-fields: euler|eulerSettings.reactiveFlow.enabled;eulerSettings.reactiveFlow.mechanismFile;eulerSettings.reactiveFlow.thermoFile;eulerSettings.reactiveFlow.transportModel;eulerSettings.reactiveFlow.CFLScale;eulerSettings.reactiveFlow.chemRelaxEps;eulerSettings.reactiveFlow.chemAbsTol;eulerSettings.reactiveFlow.TBase;eulerSettings.reactiveFlow.nSpeciesOverride;eulerSettings.reactiveFlow.useCellTWarmCache -->

### 10.32 `vfvSettings`

| 参数 | 默认参考 | 类型 | 中文含义与约束 |
| --- | --- | --- | --- |
| `maxOrder` | `1` | 整数 | 重构多项式次数；2为二次多项式。 schema约束：≥0。 |
| `intOrder` | `1` | 整数 | 全局积分阶次；不等于时间积分阶次。 schema约束：≥0。 |
| `ignoreMeshGeometryDeficiency` | `false` | 布尔 | 是否忽略网格几何缺陷提示。 |
| `nIterCellSmoothScale` | `15` | 整数 | 单元平滑尺度迭代次数，非负。 schema约束：≥0。 |
| `intOrderVR` | `-1` | 整数 | VR积分阶次；负值继承intOrder。 |
| `intOrderVRBC` | `-2` | 整数 | VR边界积分阶次；-1继承intOrder，<-1继承VR积分阶次。 |
| `cacheDiffBase` | `false` | 布尔 | 缓存积分点基函数/导数值，改变存储与计算方式。 |
| `cacheDiffBaseSize` | `255` | 整数 | 缓存的导数基项范围；通用schema为255，按维度构造的求解器对象可能设dim+1。 |
| `jacobiRelax` | `1` | 实数 | 固定VR SOR/Jacobi扫描的松弛系数。 schema约束：≥0。 |
| `SORInstead` | `true` | 布尔 | true用SOR扫描，false用松弛Jacobi。 |
| `smoothThreshold` | `0.01` | 实数 | 重构平滑性指标阈值，参与限制器判别。 schema约束：≥0。 |
| `WBAP_nStd` | `10` | 实数 | WBAP类限制器的n参数。 schema约束：≥0。 |
| `normWBAP` | `false` | 布尔 | 使用多项式范数形式的WBAP处理。 |
| `limiterBiwayAlter` | `0` | 整数 | 双向限制处理：0 wbap-L2，1 minmod。 |
| `subs2ndOrder` | `0` | 整数 | 二阶替代方法：0原VR，1 Gauss，2最小二乘，11 GGMP；主要用于一次多项式场景。 |
| `subs2ndOrderGGScheme` | `1` | 整数 | GG二阶替代插值方式：0不插值，1距离插值。 |
| `svdTolerance` | `0` | 实数 | 重构线性代数SVD容差，0沿用算子相应处理。 schema约束：≥0。 |
| `bcWeight` | `1` | 实数 | 边界重构条件的权重。 schema约束：≥0。 |

<!-- config-fields: euler|vfvSettings.maxOrder;vfvSettings.intOrder;vfvSettings.ignoreMeshGeometryDeficiency;vfvSettings.nIterCellSmoothScale;vfvSettings.intOrderVR;vfvSettings.intOrderVRBC;vfvSettings.cacheDiffBase;vfvSettings.cacheDiffBaseSize;vfvSettings.jacobiRelax;vfvSettings.SORInstead;vfvSettings.smoothThreshold;vfvSettings.WBAP_nStd;vfvSettings.normWBAP;vfvSettings.limiterBiwayAlter;vfvSettings.subs2ndOrder;vfvSettings.subs2ndOrderGGScheme;vfvSettings.svdTolerance;vfvSettings.bcWeight -->

### 10.33 `vfvSettings.baseSettings`

| 参数 | 默认参考 | 类型 | 中文含义与约束 |
| --- | --- | --- | --- |
| `localOrientation` | `false` | 布尔 | 是否使用局部方向构造基函数。 |
| `anisotropicLengths` | `false` | 布尔 | 是否使用各向异性长度尺度。 |

<!-- config-fields: euler|vfvSettings.baseSettings.localOrientation;vfvSettings.baseSettings.anisotropicLengths -->

### 10.34 `vfvSettings.functionalSettings`

| 参数 | 默认参考 | 类型 | 中文含义与约束 |
| --- | --- | --- | --- |
| `scaleType` | `"BaryDiff"` | 字符串 | 泛函尺度类型：MeanAACBB、BaryDiff或CellMax。 |
| `scaleMultiplier` | `1` | 实数 | 泛函尺度倍率，非负。 schema约束：≥0。 |
| `dirWeightScheme` | `"Factorial"` | 字符串 | 方向权重方案：Factorial、HQM_OPT、ManualDirWeight或TEST_OPT。 |
| `dirWeightCombPowV` | `1` | 整数 | 方向权重组合幂参数。 |
| `manualDirWeights` | `[1, 1, 0.5, 0.166666666667, 0.0416666666667]` | 数组 | 手动方向权重向量；只在相应方向权重方案中使用。 |
| `geomWeightScheme` | `"GWNone"` | 字符串 | 几何权重：GWNone、HQM_SD或SD_Power。 |
| `geomWeightPower` | `0.5` | 实数 | 几何权重幂参数。 |
| `geomWeightPower1` | `0` | 实数 | 几何权重第一附加幂参数。 |
| `geomWeightPower2` | `0` | 实数 | 几何权重第二附加幂参数。 |
| `useAnisotropicFunctional` | `false` | 布尔 | 是否启用各向异性泛函。 |
| `tanWeightScale` | `1` | 实数 | 各向异性泛函中的切向权重尺度。 |
| `anisotropicType` | `"InertiaCoord"` | 字符串 | 各向异性类型：InertiaCoord、InertiaCoordBB、Norm、CentDiff、WallDist、InertiaCoordBBNorm或InertiaCoordBBSym；仅相应功能开启时生效。 |
| `inertiaWeightPower` | `1` | 实数 | 惯性权重幂参数。 |
| `geomWeightBias` | `0` | 实数 | 几何权重偏置。 |
| `greenGauss1Weight` | `0` | 实数 | Green-Gauss类型1泛函项权重。 |
| `greenGauss1Bias` | `0.5` | 实数 | Green-Gauss类型1项的偏置。 |
| `greenGauss1Penalty` | `0` | 实数 | Green-Gauss类型1惩罚项系数。 |
| `greenGaussSpacial` | `0` | 整数 | 源码字段拼写为Spacial；0默认空间模式，1均匀权重模式。 |

<!-- config-fields: euler|vfvSettings.functionalSettings.scaleType;vfvSettings.functionalSettings.scaleMultiplier;vfvSettings.functionalSettings.dirWeightScheme;vfvSettings.functionalSettings.dirWeightCombPowV;vfvSettings.functionalSettings.manualDirWeights;vfvSettings.functionalSettings.geomWeightScheme;vfvSettings.functionalSettings.geomWeightPower;vfvSettings.functionalSettings.geomWeightPower1;vfvSettings.functionalSettings.geomWeightPower2;vfvSettings.functionalSettings.useAnisotropicFunctional;vfvSettings.functionalSettings.tanWeightScale;vfvSettings.functionalSettings.anisotropicType;vfvSettings.functionalSettings.inertiaWeightPower;vfvSettings.functionalSettings.geomWeightBias;vfvSettings.functionalSettings.greenGauss1Weight;vfvSettings.functionalSettings.greenGauss1Bias;vfvSettings.functionalSettings.greenGauss1Penalty;vfvSettings.functionalSettings.greenGaussSpacial -->

### 10.35 `顶层`

| 参数 | 默认参考 | 类型 | 中文含义与约束 |
| --- | --- | --- | --- |
| `bcSettings` | 未声明 | 数组 | 逐边界条件数组；其联合类型子字段、状态和原始参数格式见Euler边界章节。 |
| `bcNameMapping` | `{}` | 对象 | 边界名称映射对象；用于Geom边界/周期配对识别，不代替bcSettings的物理条件。 |

<!-- config-fields: euler|bcSettings;bcNameMapping -->

### 10.36 同族入口的schema默认差异

| 入口 | model | fieldNVariables | dimension |
| --- | --- | --- | --- |
| `euler2EQ.exe` | `NS_2EQ` | 7 | 由入口确定 |
| `eulerEX.exe` | `NS_EX` | 5（EX动态参考值） | 由入口确定 |
| `eulerSA.exe` | `NS_SA` | 6 | 由入口确定 |
| `euler2D.exe` | `NS_2D` | 4 | 由入口确定 |
| `euler.exe` | `NS` | 5 | 由入口确定 |
| `euler2EQ3D.exe` | `NS_2EQ_3D` | 7 | 由入口确定 |
| `eulerEX3D.exe` | `NS_EX_3D` | 5（EX动态参考值） | 由入口确定 |
| `eulerSA3D.exe` | `NS_SA_3D` | 6 | 由入口确定 |
| `euler3D.exe` | `NS_3D` | 5 | 由入口确定 |

除以上元数据/维度外，同族已导出的字段schema默认值一致。Euler的reactiveFlow.enabled约束另有模型差异：NS/SA/2EQ固定false，EX允许true/false。按模型构造后的默认值、相邻默认配置和算例覆盖仍按第2章处理。

## 11 常密度 ACM 完整参数表 {#dndsr_config_ch11}

本表共146项。字段组标题是JSON路径前缀；拼接表内参数名得到完整路径。默认参考严格取导出的schema，未声明时标明“未声明”。schema下界为0不表示运行时允许所有0值；正文和中文说明中的正值/组合限制仍需满足。

### 11.1 `solver`

| 参数 | 默认参考 | 类型 | 中文含义与约束 |
| --- | --- | --- | --- |
| `type` | `"ACM"` | 字符串 | 方程族，必须与可执行程序一致；具体常量见入口对照表。 schema约束：固定="ACM"。 |
| `discretization` | `"CFV"` | 字符串 | 空间离散模块：Euler/ACM/ACMVariable为CFV，节点中心求解器为NCFV。 schema约束：固定="CFV"。 |
| `model` | 见入口对照表 | 字符串 | 编译模型标识；不能仅修改此字符串把程序转换为另一个模型。 schema约束：固定模型值按第1章各入口确定。入口差异见第1章。 |
| `fieldNVariables` | 见入口对照表 | 整数 | 层流4分量，SA5分量，两方程6分量；新增尾部为密度加权的RANS输运量。 schema约束：固定模型值按第1章各入口确定。入口差异见第1章。 |

<!-- config-fields: acm|solver.type;solver.discretization;solver.model;solver.fieldNVariables -->

### 11.2 `acmSettings`

| 参数 | 默认参考 | 类型 | 中文含义与约束 |
| --- | --- | --- | --- |
| `rho0` | `1` | 实数 | 常密度rho，必须为正；运动黏度nu=mu/rho。 schema约束：≥0。 |
| `beta2` | `1` | 实数 | 人工可压缩性参数beta²，必须为正；控制伪时间预条件和特征速度。 schema约束：≥0。 |
| `alpha` | `0` | 实数 | Turkel耦合参数，要求有限值；参与预条件矩阵、特征分解、通量和远场边界。 |
| `dynamicViscosity` | `0` | 实数 | 分子动力黏度 mu；SST要求正值。运动黏度 nu=mu/rho0。 schema约束：≥0。 |
| `riemannSolverType` | `"Roe"` | 字符串 | 对流数值通量Roe或Rusanov。 schema约束：枚举："Rusanov"、"Roe"。 |
| `pressureStorage` | `"PhysicalP"` | 字符串 | 只支持PhysicalP，即状态中直接存储压力p。 schema约束：枚举："PhysicalP"。 |
| `entropyFixRatio` | `0.05` | 实数 | Roe通量的熵修正宽度系数，要求非负。 schema约束：≥0。 |
| `pressureReference` | `0` | 实数 | 保留的压力参考信息；当前网格求解和输出直接使用状态p，不自动减去此值。 |
| `farFieldValue` | `[0, 0, 0, 0]` | 数组 | 保留的远场状态字段；网格边界实际读取 boundaryConditions 的value或顶层 boundaryValue。 |
| `enableViscousFlux` | `false` | 布尔 | 启用黏性通量；SST必须为true，并在分子黏度上耦合涡黏度。 |

<!-- config-fields: acm|acmSettings.rho0;acmSettings.beta2;acmSettings.alpha;acmSettings.dynamicViscosity;acmSettings.riemannSolverType;acmSettings.pressureStorage;acmSettings.entropyFixRatio;acmSettings.pressureReference;acmSettings.farFieldValue;acmSettings.enableViscousFlux -->

### 11.3 `timeMarchSettings`

| 参数 | 默认参考 | 类型 | 中文含义与约束 |
| --- | --- | --- | --- |
| `integrator` | `"ExplicitSSPRK3"` | 字符串 | 稳态伪时间或BDF2双时间算法，六种名称及启用条件见ACM时间推进章节。 schema约束：枚举："ExplicitSSPRK3"、"ImplicitEulerBlockJacobi"、"ImplicitEulerLUSGS"、"ImplicitEulerGMRES"、"BDF2DualTimeLUSGS"、"BDF2DualTimeGMRES"。 |
| `nSteps` | `0` | 整数 | 外层步数。BDF2中为物理步数；稳态算法中为伪时间外层步数。 schema约束：≥0。 |
| `pseudoTimeStep` | `0.01` | 实数 | 关闭CFL时间步时使用的固定伪时间步。 schema约束：≥0。 |
| `physicalTimeStep` | `0.01` | 实数 | BDF2均匀物理时间步dt；与局部伪时间步分开设置。 schema约束：≥0。 |
| `useCFLTimeStep` | `false` | 布尔 | true按流场及黏性谱半径计算伪时间步；false使用pseudoTimeStep。 |
| `useLocalTimeStep` | `true` | 布尔 | true保留各单元局部伪时间步；false采用MPI全局最小步长。 |
| `cfl` | `0.5` | 实数 | 伪时间CFL系数，必须为正；不直接等于物理时间步。 schema约束：≥0。 |
| `maximumPseudoTimeStep` | `1e+100` | 实数 | CFL计算得到的伪时间步上限，必须为正。 schema约束：≥0。 |
| `maxImplicitIterations` | `20` | 整数 | 每个物理步最多内迭代次数，必须为正；达到上限不等于收敛。 schema约束：≥1。 |
| `implicitTolerance` | `1e-8` | 实数 | 非线性绝对RMS阈值；AbsoluteOrRelative模式与相对阈值按OR组合。 schema约束：≥0。 |
| `dualTimeConvergenceCriterion` | `"Absolute"` | 字符串 | Absolute、Relative或AbsoluteOrRelative；OR模式使用max(atol, rtol*本步初值)，每个活跃组各自达标。旧文件缺少字段时补Absolute。 |
| `dualTimeFlowRelativeTolerance` | `0` | 实数 | 正值启用流场 [u,v,w,p] 的组RMS；Relative目标为该值乘本物理步流场初值。范围[0,1)。 schema约束：≥0；≤1。 |
| `dualTimeTurbulenceRelativeTolerance` | `0` | 实数 | 正值启用活跃湍流变量组RMS；SST为 [k,omega]，使用其独立的本步初值。范围[0,1)。 schema约束：≥0；≤1。 |
| `dualTimeResidualScales` | `[1, 1, 1, 1, 1, 1]` | 数组 | 6个正的逐方程绝对残差尺度，顺序 [u,v,w,p,q1,q2]；对应组启用组RMS时必须保持1。 |
| `dualTimeRelativeTolerances` | `[0, 0, 0, 0, 0, 0]` | 数组 | 逐方程相对系数，顺序同左；组RMS关闭时使用。Relative模式要求每个活跃方程有正系数。 |
| `dualTimeRequireConvergence` | `true` | 布尔 | true时内迭代未满足所选条件就报错，不提交该步历史或物理时间；false仅用于预算受限的诊断推进。 |
| `steadyRelativeTolerance` | `0` | 实数 | 稳态隐式算法的相对阈值，参考该外层步的空间残差；BDF2中必须为0。 schema约束：≥0；≤1。 |
| `steadyAdaptiveCFL` | `false` | 布尔 | 稳态隐式算法的自适应CFL开关；BDF2中必须为false。 |
| `steadyCFLMin` | `0.1` | 实数 | 稳态自适应CFL下限，必须为正。 schema约束：≥0。 |
| `steadyCFLMax` | `20` | 实数 | 稳态自适应CFL上限，不小于下限。 schema约束：≥0。 |
| `steadyCFLGrowth` | `1.5` | 实数 | 稳态CFL最大增长倍数，至少1。 schema约束：≥1。 |
| `steadyCFLReduction` | `0.5` | 实数 | 稳态失败/残差增长后的CFL缩小系数，范围(0,1)。 schema约束：≥0；≤1。 |
| `implicitRelaxation` | `1` | 实数 | 隐式流场修正的松弛系数，范围(0,1]。 schema约束：≥0；≤1。 |
| `lusgsSweeps` | `2` | 整数 | 独立LU-SGS的最大扫掠数；每遍检查真实残差。GMRES预条件中仍为固定扫掠数。 schema约束：≥1。 |
| `gmresSubspace` | `10` | 整数 | GMRES的Arnoldi子空间维数，至少2；仅用于相应GMRES路径。 schema约束：≥2。 |
| `gmresRestarts` | `3` | 整数 | 首轮之后的GMRES重启次数上限；0仍执行首轮，3最多执行4轮。 schema约束：≥0。 |
| `linearRelativeTolerance` | `1e-06` | 实数 | GMRES/LU-SGS真实线性RMS相对阈值，与线性绝对阈值按OR组合。 |
| `linearAbsoluteTolerance` | `1e-10` | 实数 | 真实线性RMS绝对阈值，在非线性松弛之前评估；不使用预条件范数。 |
| `gmresPreconditioner` | `"LUSGS"` | 字符串 | GMRES左预条件器LUSGS或BlockJacobi。 schema约束：枚举："BlockJacobi"、"LUSGS"。 |

<!-- config-fields: acm|timeMarchSettings.integrator;timeMarchSettings.nSteps;timeMarchSettings.pseudoTimeStep;timeMarchSettings.physicalTimeStep;timeMarchSettings.useCFLTimeStep;timeMarchSettings.useLocalTimeStep;timeMarchSettings.cfl;timeMarchSettings.maximumPseudoTimeStep;timeMarchSettings.maxImplicitIterations;timeMarchSettings.implicitTolerance;timeMarchSettings.dualTimeConvergenceCriterion;timeMarchSettings.dualTimeFlowRelativeTolerance;timeMarchSettings.dualTimeTurbulenceRelativeTolerance;timeMarchSettings.dualTimeResidualScales;timeMarchSettings.dualTimeRelativeTolerances;timeMarchSettings.dualTimeRequireConvergence;timeMarchSettings.steadyRelativeTolerance;timeMarchSettings.steadyAdaptiveCFL;timeMarchSettings.steadyCFLMin;timeMarchSettings.steadyCFLMax;timeMarchSettings.steadyCFLGrowth;timeMarchSettings.steadyCFLReduction;timeMarchSettings.implicitRelaxation;timeMarchSettings.lusgsSweeps;timeMarchSettings.gmresSubspace;timeMarchSettings.gmresRestarts;timeMarchSettings.linearRelativeTolerance;timeMarchSettings.linearAbsoluteTolerance;timeMarchSettings.gmresPreconditioner -->

### 11.4 `turbulenceSettings`

| 参数 | 默认参考 | 类型 | 中文含义与约束 |
| --- | --- | --- | --- |
| `model` | `"Laminar"` | 字符串 | 层流入口固定为Laminar；SA入口固定为SpalartAllmaras；两方程入口可选KOmegaWilcox、KOmegaSST、RealizableKEpsilon，默认KOmegaWilcox。 |
| `initialValue` | `[1e-06, 1]` | 数组 | 湍流初值[q1,q2]；SA仅首分量活跃，SST为[k,omega]，新入口在主状态尾部存储rho*q。 |
| `farFieldValue` | `[1e-06, 1]` | 数组 | 湍流入口/远场值[q1,q2]；按选定模型解释。 |
| `minimumValue` | `[1e-12, 1e-10]` | 数组 | 每个活跃变量的正值下限。各变量独立限制，k触底不冻结omega。 |
| `maximumValue` | `[1000000, 1e+12]` | 数组 | 各湍流变量的有限上限，不小于对应下限。 |
| `maximumEddyViscosityRatio` | `100000` | 实数 | 涡黏度与分子黏度比 mu_t/mu 的上限，非负。 schema约束：≥0。 |
| `wallOmegaCoefficient` | `800` | 实数 | 壁面omega系数C，omega_w=C*nu/d²，d使用正壁距下限。 schema约束：≥0。 |
| `enableSourceTerms` | `true` | 布尔 | 是否计算模型的生产/破坏等源项；正常SST计算设true。 |
| `secondOrderReconstruction` | `true` | 布尔 | 湍流变量采用带限制器的二阶Green-Gauss重构；不随流场多项式阶数自动改变。 |
| `transportSubsteps` | 统一入口为1 | 整数 | 新SA/两方程入口固定1；分离预算只用于旧兼容输运核。 |
| `transportTimeScale` | `0.25` | 实数 | 湍流与流场伪时间步比例，范围(0,1]；不改变物理时间步。 schema约束：≥0；≤1。 |
| `wallDistanceMethod` | `1` | 整数 | 壁距算法：0暴力搜索，1 AABB树搜索。 schema约束：≥0；≤1。 |
| `wallDistanceExecution` | `0` | 整数 | 壁距MPI执行：0各rank并行，1串行，>1按rank模值分批执行。 schema约束：≥0。 |
| `wallDistanceSubdivide` | `0` | 整数 | 壁面四边形/曲面细分级别，非负。 schema约束：≥0。 |
| `minimumWallDistance` | `1e-10` | 实数 | 计算模型/壁面值时使用的正壁距下限，必须为正。 schema约束：≥0。 |
| `wallDistanceVerbose` | `0` | 整数 | 壁距计算诊断输出级别，非负；0关闭额外信息。 schema约束：≥0。 |

<!-- config-fields: acm|turbulenceSettings.model;turbulenceSettings.initialValue;turbulenceSettings.farFieldValue;turbulenceSettings.minimumValue;turbulenceSettings.maximumValue;turbulenceSettings.maximumEddyViscosityRatio;turbulenceSettings.wallOmegaCoefficient;turbulenceSettings.enableSourceTerms;turbulenceSettings.secondOrderReconstruction;turbulenceSettings.transportSubsteps;turbulenceSettings.transportTimeScale;turbulenceSettings.wallDistanceMethod;turbulenceSettings.wallDistanceExecution;turbulenceSettings.wallDistanceSubdivide;turbulenceSettings.minimumWallDistance;turbulenceSettings.wallDistanceVerbose -->

### 11.5 `meshSettings`

| 参数 | 默认参考 | 类型 | 中文含义与约束 |
| --- | --- | --- | --- |
| `meshFile` | `""` | 字符串 | CGNS网格路径，按程序启动工作目录解析。 |
| `meshElevation` | `0` | 整数 | 0不提升几何阶次，1将O1几何提升到O2。 schema约束：≥0；≤1。 |
| `meshDirectBisect` | `0` | 整数 | 直接网格二分次数，合法范围0至4；0不细化。 schema约束：≥0；≤4。 |
| `meshReorderCells` | `0` | 整数 | 0不重排局部单元，1启用局部单元重排。 schema约束：≥0；≤1。 |
| `periodicTolerance` | `1e-09` | 实数 | 周期节点匹配容差，非负；需要与网格坐标尺度相容。 schema约束：≥0。 |
| `periodicTranslation1` | `[0, 0, 0]` | 数组 | 第一组周期平移向量，三分量。 |
| `periodicTranslation2` | `[0, 0, 0]` | 数组 | 第二组周期平移向量，三分量。 |
| `periodicTranslation3` | `[0, 0, 0]` | 数组 | 第三组周期平移向量，三分量。 |

<!-- config-fields: acm|meshSettings.meshFile;meshSettings.meshElevation;meshSettings.meshDirectBisect;meshSettings.meshReorderCells;meshSettings.periodicTolerance;meshSettings.periodicTranslation1;meshSettings.periodicTranslation2;meshSettings.periodicTranslation3 -->

### 11.6 `meshSettings.partitionOptions`

| 参数 | 默认参考 | 类型 | 中文含义与约束 |
| --- | --- | --- | --- |
| `metisType` | `"KWAY"` | 字符串 | 网格划分方法：KWAY多路划分，RB递归二分。 schema约束：枚举："KWAY"、"RB"。 |
| `metisUfactor` | `20` | 整数 | METIS负载不平衡控制参数，至少1；沿用METIS ufactor含义。 schema约束：≥1。 |
| `metisSeed` | `0` | 整数 | METIS随机种子；用于固定划分的随机序列。 |
| `edgeWeightMethod` | `0` | 整数 | 划分图边权：0不加权，1按面尺寸加权。 schema约束：≥0；≤1。 |
| `metisNcuts` | `3` | 整数 | METIS尝试的划分次数，至少1。 schema约束：≥1。 |

<!-- config-fields: acm|meshSettings.partitionOptions.metisType;meshSettings.partitionOptions.metisUfactor;meshSettings.partitionOptions.metisSeed;meshSettings.partitionOptions.edgeWeightMethod;meshSettings.partitionOptions.metisNcuts -->

### 11.7 `reconstructionSettings`

| 参数 | 默认参考 | 类型 | 中文含义与约束 |
| --- | --- | --- | --- |
| `type` | `"GreenGauss"` | 字符串 | 流场重构类型：FirstOrder、GreenGauss、Variational。 schema约束：枚举："FirstOrder"、"GreenGauss"、"Variational"。 |
| `variationalIterations` | `3` | 整数 | 固定次数VR扫描数；启用受控重构时也用于最小扫描/检查条件。 schema约束：≥1。 |
| `variationalTolerance` | `0` | 实数 | 缩放后的重构方程最大缺陷阈值；0使用固定扫描，正值启用收敛控制。 schema约束：≥0。 |
| `variationalMaxIterations` | `10000` | 整数 | 受控变分重构扫描上限；需覆盖最小扫描预算。 schema约束：≥1。 |
| `variationalCheckInterval` | `10` | 整数 | 受控重构检查方程缺陷的间隔，至少1；首轮、末轮等也会检查。 schema约束：≥1。 |
| `variationalRelaxation` | `0.7` | 实数 | 受控缺陷更新松弛，范围(0,1]；固定扫描采用vfvSettings松弛。 schema约束：≥0；≤1。 |
| `variationalUseGMRES` | `false` | 布尔 | 受控重构启用GMRES线性求解。 |
| `variationalGMRESSubspace` | `20` | 整数 | 重构GMRES子空间维数，至少2；与流场GMRES参数分别设置。 schema约束：≥2。 |
| `variationalGMRESRestarts` | `2` | 整数 | 首轮之后的重构GMRES重启次数；0仍执行首轮，2最多执行3轮。 schema约束：≥0。 |
| `variationalGMRESRelativeTolerance` | `0.1` | 实数 | 每次重构GMRES线性求解的相对阈值，位于(0,1]；不是BDF2物理步阈值。 schema约束：≥0；≤1。 |
| `resetVariationalCoefficients` | `false` | 布尔 | 每次残差求值前清零VR系数；常密度BDF2固定扫描模式会强制清零。 |
| `enableLimiter` | `true` | 布尔 | 启用选定的重构限制器。 |
| `limiterType` | `"LocalExtrema"` | 字符串 | LocalExtrema、WBAP或CWBAP；WBAP/CWBAP要求Variational。 schema约束：枚举："LocalExtrema"、"WBAP"、"CWBAP"。 |

<!-- config-fields: acm|reconstructionSettings.type;reconstructionSettings.variationalIterations;reconstructionSettings.variationalTolerance;reconstructionSettings.variationalMaxIterations;reconstructionSettings.variationalCheckInterval;reconstructionSettings.variationalRelaxation;reconstructionSettings.variationalUseGMRES;reconstructionSettings.variationalGMRESSubspace;reconstructionSettings.variationalGMRESRestarts;reconstructionSettings.variationalGMRESRelativeTolerance;reconstructionSettings.resetVariationalCoefficients;reconstructionSettings.enableLimiter;reconstructionSettings.limiterType -->

### 11.8 `outputSettings`

| 参数 | 默认参考 | 类型 | 中文含义与约束 |
| --- | --- | --- | --- |
| `interval` | `0` | 整数 | 外层步场输出间隔；0关闭场输出，初始化场也随之关闭。 schema约束：≥0。 |
| `directory` | `"../data/outACM"` | 字符串 | 输出目录，相对启动工作目录解析。 |
| `prefix` | `"acm"` | 字符串 | 文件及时间序列前缀。 |
| `writeInitial` | `true` | 布尔 | interval>0时是否写初始场。 |

<!-- config-fields: acm|outputSettings.interval;outputSettings.directory;outputSettings.prefix;outputSettings.writeInitial -->

### 11.9 `restartSettings`

| 参数 | 默认参考 | 类型 | 中文含义与约束 |
| --- | --- | --- | --- |
| `flowFile` | `""` | 字符串 | 同分区VTK-HDF流场初始化文件；空字符串关闭。 |
| `completedSteps` | `0` | 整数 | 稳态初始场所代表的已完成步数，非负。 schema约束：≥0。 |
| `writerRanks` | `0` | 整数 | 写重启文件时的MPI进程数；启用重启时为正，并须与读入进程数相同。 schema约束：≥0。 |
| `requireTurbulence` | `false` | 布尔 | 稳态RANS重启时是否强制要求全部模型湍流字段。 |

<!-- config-fields: acm|restartSettings.flowFile;restartSettings.completedSteps;restartSettings.writerRanks;restartSettings.requireTurbulence -->

### 11.10 `vfvSettings`

| 参数 | 默认参考 | 类型 | 中文含义与约束 |
| --- | --- | --- | --- |
| `maxOrder` | `1` | 整数 | 重构多项式次数；ACM/ACMVariable支持0至3；2为二次多项式。 schema约束：≥0。 |
| `intOrder` | `1` | 整数 | 全局积分阶次；不等于时间积分阶次。 schema约束：≥0。 |
| `ignoreMeshGeometryDeficiency` | `false` | 布尔 | 是否忽略网格几何缺陷提示。 |
| `nIterCellSmoothScale` | `15` | 整数 | 单元平滑尺度迭代次数，非负。 schema约束：≥0。 |
| `intOrderVR` | `-1` | 整数 | VR积分阶次；负值继承intOrder。 |
| `intOrderVRBC` | `-2` | 整数 | VR边界积分阶次；-1继承intOrder，<-1继承VR积分阶次。 |
| `cacheDiffBase` | `false` | 布尔 | 缓存积分点基函数/导数值，改变存储与计算方式。 |
| `cacheDiffBaseSize` | `255` | 整数 | 缓存的导数基项范围；通用schema为255，按维度构造的求解器对象可能设dim+1。 |
| `jacobiRelax` | `1` | 实数 | 固定VR SOR/Jacobi扫描的松弛系数。 schema约束：≥0。 |
| `SORInstead` | `true` | 布尔 | true用SOR扫描，false用松弛Jacobi。 |
| `smoothThreshold` | `0.01` | 实数 | 重构平滑性指标阈值，参与限制器判别。 schema约束：≥0。 |
| `WBAP_nStd` | `10` | 实数 | WBAP类限制器的n参数。 schema约束：≥0。 |
| `normWBAP` | `false` | 布尔 | 使用多项式范数形式的WBAP处理。 |
| `limiterBiwayAlter` | `0` | 整数 | 双向限制处理：0 wbap-L2，1 minmod。 |
| `subs2ndOrder` | `0` | 整数 | 二阶替代方法：0原VR，1 Gauss，2最小二乘，11 GGMP；主要用于一次多项式场景。 |
| `subs2ndOrderGGScheme` | `1` | 整数 | GG二阶替代插值方式：0不插值，1距离插值。 |
| `svdTolerance` | `0` | 实数 | 重构线性代数SVD容差，0沿用算子相应处理。 schema约束：≥0。 |
| `bcWeight` | `1` | 实数 | 边界重构条件的权重。 schema约束：≥0。 |

<!-- config-fields: acm|vfvSettings.maxOrder;vfvSettings.intOrder;vfvSettings.ignoreMeshGeometryDeficiency;vfvSettings.nIterCellSmoothScale;vfvSettings.intOrderVR;vfvSettings.intOrderVRBC;vfvSettings.cacheDiffBase;vfvSettings.cacheDiffBaseSize;vfvSettings.jacobiRelax;vfvSettings.SORInstead;vfvSettings.smoothThreshold;vfvSettings.WBAP_nStd;vfvSettings.normWBAP;vfvSettings.limiterBiwayAlter;vfvSettings.subs2ndOrder;vfvSettings.subs2ndOrderGGScheme;vfvSettings.svdTolerance;vfvSettings.bcWeight -->

### 11.11 `vfvSettings.baseSettings`

| 参数 | 默认参考 | 类型 | 中文含义与约束 |
| --- | --- | --- | --- |
| `localOrientation` | `false` | 布尔 | 是否使用局部方向构造基函数。 |
| `anisotropicLengths` | `false` | 布尔 | 是否使用各向异性长度尺度。 |

<!-- config-fields: acm|vfvSettings.baseSettings.localOrientation;vfvSettings.baseSettings.anisotropicLengths -->

### 11.12 `vfvSettings.functionalSettings`

| 参数 | 默认参考 | 类型 | 中文含义与约束 |
| --- | --- | --- | --- |
| `scaleType` | `"BaryDiff"` | 字符串 | 泛函尺度类型：MeanAACBB、BaryDiff或CellMax。 |
| `scaleMultiplier` | `1` | 实数 | 泛函尺度倍率，非负。 schema约束：≥0。 |
| `dirWeightScheme` | `"Factorial"` | 字符串 | 方向权重方案：Factorial、HQM_OPT、ManualDirWeight或TEST_OPT。 |
| `dirWeightCombPowV` | `1` | 整数 | 方向权重组合幂参数。 |
| `manualDirWeights` | `[1, 1, 0.5, 0.166666666667, 0.0416666666667]` | 数组 | 手动方向权重向量；只在相应方向权重方案中使用。 |
| `geomWeightScheme` | `"GWNone"` | 字符串 | 几何权重：GWNone、HQM_SD或SD_Power。 |
| `geomWeightPower` | `0.5` | 实数 | 几何权重幂参数。 |
| `geomWeightPower1` | `0` | 实数 | 几何权重第一附加幂参数。 |
| `geomWeightPower2` | `0` | 实数 | 几何权重第二附加幂参数。 |
| `useAnisotropicFunctional` | `false` | 布尔 | 是否启用各向异性泛函。 |
| `tanWeightScale` | `1` | 实数 | 各向异性泛函中的切向权重尺度。 |
| `anisotropicType` | `"InertiaCoord"` | 字符串 | 各向异性类型：InertiaCoord、InertiaCoordBB、Norm、CentDiff、WallDist、InertiaCoordBBNorm或InertiaCoordBBSym；仅相应功能开启时生效。 |
| `inertiaWeightPower` | `1` | 实数 | 惯性权重幂参数。 |
| `geomWeightBias` | `0` | 实数 | 几何权重偏置。 |
| `greenGauss1Weight` | `0` | 实数 | Green-Gauss类型1泛函项权重。 |
| `greenGauss1Bias` | `0.5` | 实数 | Green-Gauss类型1项的偏置。 |
| `greenGauss1Penalty` | `0` | 实数 | Green-Gauss类型1惩罚项系数。 |
| `greenGaussSpacial` | `0` | 整数 | 源码字段拼写为Spacial；0默认空间模式，1均匀权重模式。 |

<!-- config-fields: acm|vfvSettings.functionalSettings.scaleType;vfvSettings.functionalSettings.scaleMultiplier;vfvSettings.functionalSettings.dirWeightScheme;vfvSettings.functionalSettings.dirWeightCombPowV;vfvSettings.functionalSettings.manualDirWeights;vfvSettings.functionalSettings.geomWeightScheme;vfvSettings.functionalSettings.geomWeightPower;vfvSettings.functionalSettings.geomWeightPower1;vfvSettings.functionalSettings.geomWeightPower2;vfvSettings.functionalSettings.useAnisotropicFunctional;vfvSettings.functionalSettings.tanWeightScale;vfvSettings.functionalSettings.anisotropicType;vfvSettings.functionalSettings.inertiaWeightPower;vfvSettings.functionalSettings.geomWeightBias;vfvSettings.functionalSettings.greenGauss1Weight;vfvSettings.functionalSettings.greenGauss1Bias;vfvSettings.functionalSettings.greenGauss1Penalty;vfvSettings.functionalSettings.greenGaussSpacial -->

### 11.13 `顶层`

| 参数 | 默认参考 | 类型 | 中文含义与约束 |
| --- | --- | --- | --- |
| `defaultBoundaryType` | `"BCFar"` | 字符串 | 未按名称显式配置的外边界类型；名称和别名见ACM边界表。 schema约束：枚举："BCFar"、"BCWall"、"BCWallInvis"、"BCWallIsothermal"、"BCOut"、"BCOutP"、"BCIn"、"BCInPsTs"、"BCSym"、"BCSpecial"、"FarField"、"VelocityInlet"、"PressureOutlet"、"NoSlipWall"、"SlipWall"、"Symmetry"。 |
| `initialState` | `[1, 0, 0, 0]` | 数组 | 层流均匀初值[u,v,w,p]，二维也保留w分量。SA入口追加rho*nuTilde，默认1e-6；两方程入口追加rho*k及rho*omega/epsilon，默认[1e-6,1]，总长度分别为5或6。 |
| `boundaryValue` | `[1, 0, 0, 0]` | 数组 | 未显式设置区域值时的默认常密度状态[u,v,w,p]。 |
| `leftState` | `[1, 0, 0, 0]` | 数组 | 无网格通量预览的左状态[u,v,w,p]。 |
| `rightState` | `[0, 0, 0, 0]` | 数组 | 无网格通量预览的右状态[u,v,w,p]。 |
| `unitNormal` | `[1, 0, 0]` | 数组 | 预览内核的面法线；物理网格运行使用网格实际面法线。 |
| `nFacesPerRank` | `1` | 整数 | 预览内核的每rank面数；不决定MPI进程数或网格单元数。 schema约束：≥1。 |

<!-- config-fields: acm|defaultBoundaryType;initialState;boundaryValue;leftState;rightState;unitNormal;nFacesPerRank -->

### 11.14 `boundaryConditions[]`

| 参数 | 默认参考 | 类型 | 中文含义与约束 |
| --- | --- | --- | --- |
| `type` | `"BCFar"` | 字符串 | 该区域的ACM边界条件类型。 schema约束：枚举："BCFar"、"BCWall"、"BCWallInvis"、"BCWallIsothermal"、"BCOut"、"BCOutP"、"BCIn"、"BCInPsTs"、"BCSym"、"BCSpecial"、"FarField"、"VelocityInlet"、"PressureOutlet"、"NoSlipWall"、"SlipWall"、"Symmetry"。 |
| `name` | `""` | 字符串 | CGNS边界区域名称，区分大小写。 |
| `value` | `[0, 0, 0, 0]` | 数组 | 常密度边界状态[u,v,w,p]，四分量；特殊边界参数按类型解释。 |
| `frameOption` | `0` | 整数 | 保留的参考系选项。 schema约束：≥0。 |
| `anchorOption` | `0` | 整数 | 保留的锚定选项。 schema约束：≥0。 |
| `integrationOption` | `0` | 整数 | 保留的边界积分选项。 schema约束：≥0。 |
| `specialOption` | `0` | 整数 | 特殊边界子类型。 schema约束：≥0。 |
| `rectifyOption` | `0` | 整数 | 保留的对称修正选项。 schema约束：≥0。 |
| `valueExtra` | `[]` | 数组 | 附加边界数据。 |

<!-- config-fields: acm|boundaryConditions[].type;boundaryConditions[].name;boundaryConditions[].value;boundaryConditions[].frameOption;boundaryConditions[].anchorOption;boundaryConditions[].integrationOption;boundaryConditions[].specialOption;boundaryConditions[].rectifyOption;boundaryConditions[].valueExtra -->

### 11.15 同族入口的schema默认差异

| 入口 | model | fieldNVariables | dimension |
| --- | --- | --- | --- |
| `acm2D.exe` | `ConstantDensity2D` | 4 | 由入口确定 |
| `acm3D.exe` | `ConstantDensity3D` | 4 | 3 |
| `acmSA2D.exe` | `ConstantDensitySA2D` | 5 | 2 |
| `acmSA3D.exe` | `ConstantDensitySA3D` | 5 | 3 |
| `acm2EQ2D.exe` | `ConstantDensity2EQ2D` | 6 | 2 |
| `acm2EQ3D.exe` | `ConstantDensity2EQ3D` | 6 | 3 |

ACM新入口的initialState长度、闭合模型、transportSubsteps及黏性开关/黏度默认另受模型约束；其他族的默认差异按第2章处理。

## 12 变密度 ACMVariable 完整参数表 {#dndsr_config_ch12}

本表共127项。字段组标题是JSON路径前缀；拼接表内参数名得到完整路径。默认参考严格取导出的schema，未声明时标明“未声明”。schema下界为0不表示运行时允许所有0值；正文和中文说明中的正值/组合限制仍需满足。

### 12.1 `solver`

| 参数 | 默认参考 | 类型 | 中文含义与约束 |
| --- | --- | --- | --- |
| `type` | `"ACMVariable"` | 字符串 | 方程族，必须与可执行程序一致；具体常量见入口对照表。 schema约束：固定="ACMVariable"。 |
| `discretization` | `"CFV"` | 字符串 | 空间离散模块：Euler/ACM/ACMVariable为CFV，节点中心求解器为NCFV。 schema约束：固定="CFV"。 |
| `model` | 见入口对照表 | 字符串 | 编译模型标识；不能仅修改此字符串把程序转换为另一个模型。 schema约束：固定模型值按第1章各入口确定。入口差异见第1章。 |
| `fieldNVariables` | 见入口对照表 | 整数 | 固定五分量[rho,mx,my,mz,p]，mx=rho*u；湍流变量独立存储。 schema约束：固定模型值按第1章各入口确定。入口差异见第1章。 |

<!-- config-fields: acmVariable|solver.type;solver.discretization;solver.model;solver.fieldNVariables -->

### 12.2 `acmSettings`

| 参数 | 默认参考 | 类型 | 中文含义与约束 |
| --- | --- | --- | --- |
| `rho0` | `1` | 实数 | 限制器归一化参考密度，必须为正；实际密度由状态rho给定。 schema约束：≥0。 |
| `densityFloor` | `1e-10` | 实数 | 实际密度正下限，必须为正。 schema约束：≥0。 |
| `beta2` | `1` | 实数 | 人工可压缩性参数beta²，必须为正；控制伪时间预条件和特征速度。 schema约束：≥0。 |
| `alpha` | `0` | 实数 | Turkel耦合参数，要求有限值；参与预条件矩阵、特征分解、通量和远场边界。 |
| `dynamicViscosity` | `0` | 实数 | 分子动力黏度 mu；SST要求正值。运动黏度 nu=mu/rho0。 schema约束：≥0。 |
| `riemannSolverType` | `"Roe"` | 字符串 | 对流数值通量Roe或Rusanov。 schema约束：枚举："Rusanov"、"Roe"。 |
| `pressureStorage` | `"PhysicalP"` | 字符串 | 只支持PhysicalP，即状态中直接存储压力p。 schema约束：枚举："PhysicalP"。 |
| `entropyFixRatio` | `0.05` | 实数 | Roe通量的熵修正宽度系数，要求非负。 schema约束：≥0。 |
| `pressureReference` | `0` | 实数 | 保留的压力参考信息；当前网格求解和输出直接使用状态p，不自动减去此值。 |
| `farFieldValue` | `[1, 0, 0, 0, 0]` | 数组 | 远场参考状态[rho,mx,my,mz,p]；边界区域值仍按边界配置读取。 |
| `enableViscousFlux` | `false` | 布尔 | 启用黏性通量；SST必须为true，并在分子黏度上耦合涡黏度。 |

<!-- config-fields: acmVariable|acmSettings.rho0;acmSettings.densityFloor;acmSettings.beta2;acmSettings.alpha;acmSettings.dynamicViscosity;acmSettings.riemannSolverType;acmSettings.pressureStorage;acmSettings.entropyFixRatio;acmSettings.pressureReference;acmSettings.farFieldValue;acmSettings.enableViscousFlux -->

### 12.3 `timeMarchSettings`

| 参数 | 默认参考 | 类型 | 中文含义与约束 |
| --- | --- | --- | --- |
| `physicalODECode` | `-1` | 整数 | 共享物理ODE编号；-1使用原生integrator；2和101在物理路径中拒绝。 |
| `pseudoODECode` | `-1` | 整数 | 共享伪时间ODE编号；-1关闭；非负时与physicalODECode互斥。 |
| `integrator` | `"ExplicitSSPRK3"` | 字符串 | 稳态伪时间或BDF2双时间算法，六种名称及启用条件见ACM时间推进章节。 schema约束：枚举："ExplicitSSPRK3"、"ImplicitEulerBlockJacobi"、"ImplicitEulerLUSGS"、"ImplicitEulerGMRES"、"BDF2DualTimeLUSGS"、"BDF2DualTimeGMRES"。 |
| `nSteps` | `0` | 整数 | 外层步数。BDF2中为物理步数；稳态算法中为伪时间外层步数。 schema约束：≥0。 |
| `pseudoTimeStep` | `0.01` | 实数 | 关闭CFL时间步时使用的固定伪时间步。 schema约束：≥0。 |
| `physicalTimeStep` | `0.01` | 实数 | BDF2均匀物理时间步dt；与局部伪时间步分开设置。 schema约束：≥0。 |
| `useCFLTimeStep` | `false` | 布尔 | true按流场及黏性谱半径计算伪时间步；false使用pseudoTimeStep。 |
| `useLocalTimeStep` | `true` | 布尔 | true保留各单元局部伪时间步；false采用MPI全局最小步长。 |
| `cfl` | `0.5` | 实数 | 伪时间CFL系数，必须为正；不直接等于物理时间步。 schema约束：≥0。 |
| `maximumPseudoTimeStep` | `1e+100` | 实数 | CFL计算得到的伪时间步上限，必须为正。 schema约束：≥0。 |
| `maxImplicitIterations` | `20` | 整数 | 每个物理步最多内迭代次数，必须为正；达到上限不等于收敛。 schema约束：≥1。 |
| `implicitTolerance` | `1e-8` | 实数 | 变密度隐式/物理阶段的绝对RMS阈值，与implicitRelativeTolerance按OR组合。 |
| `implicitRelativeTolerance` | `0` | 实数 | 相对本次非线性求解或ODE阶段初值的RMS阈值；算例显式采用1e-4。 |
| `implicitRelaxation` | `1` | 实数 | 隐式流场修正的松弛系数，范围(0,1]。 schema约束：≥0；≤1。 |
| `lusgsSweeps` | `2` | 整数 | 独立LU-SGS的最大扫掠数；每遍检查真实残差。GMRES预条件中仍为固定扫掠数。 schema约束：≥1。 |
| `gmresSubspace` | `10` | 整数 | GMRES的Arnoldi子空间维数，至少2；仅用于相应GMRES路径。 schema约束：≥2。 |
| `gmresRestarts` | `3` | 整数 | 首轮之后的GMRES重启次数上限；0仍执行首轮，3最多执行4轮。 schema约束：≥0。 |
| `linearRelativeTolerance` | `1e-06` | 实数 | GMRES/LU-SGS真实线性RMS相对阈值，与线性绝对阈值按OR组合。 |
| `linearAbsoluteTolerance` | `1e-10` | 实数 | 真实线性RMS绝对阈值，在非线性松弛之前评估；不使用预条件范数。 |
| `gmresPreconditioner` | `"LUSGS"` | 字符串 | GMRES左预条件器LUSGS或BlockJacobi。 schema约束：枚举："BlockJacobi"、"LUSGS"。 |

<!-- config-fields: acmVariable|timeMarchSettings.physicalODECode;timeMarchSettings.pseudoODECode;timeMarchSettings.integrator;timeMarchSettings.nSteps;timeMarchSettings.pseudoTimeStep;timeMarchSettings.physicalTimeStep;timeMarchSettings.useCFLTimeStep;timeMarchSettings.useLocalTimeStep;timeMarchSettings.cfl;timeMarchSettings.maximumPseudoTimeStep;timeMarchSettings.maxImplicitIterations;timeMarchSettings.implicitTolerance;timeMarchSettings.implicitRelativeTolerance;timeMarchSettings.implicitRelaxation;timeMarchSettings.lusgsSweeps;timeMarchSettings.gmresSubspace;timeMarchSettings.gmresRestarts;timeMarchSettings.linearRelativeTolerance;timeMarchSettings.linearAbsoluteTolerance;timeMarchSettings.gmresPreconditioner -->

### 12.4 `turbulenceSettings`

| 参数 | 默认参考 | 类型 | 中文含义与约束 |
| --- | --- | --- | --- |
| `model` | `"Laminar"` | 字符串 | 闭合模型选择：Laminar、SA、Wilcox k-omega、SST或Realizable k-epsilon。 schema约束：枚举："Laminar"、"SpalartAllmaras"、"KOmegaWilcox"、"KOmegaSST"、"RealizableKEpsilon"。 |
| `initialValue` | `[1e-06, 1]` | 数组 | 湍流初值[q1,q2]；SA仅首分量活跃，SST为[k,omega]，新入口在主状态尾部存储rho*q。 |
| `farFieldValue` | `[1e-06, 1]` | 数组 | 湍流入口/远场值[q1,q2]；按选定模型解释。 |
| `minimumValue` | `[1e-12, 1e-10]` | 数组 | 每个活跃变量的正值下限。各变量独立限制，k触底不冻结omega。 |
| `maximumValue` | `[1000000, 1e+12]` | 数组 | 各湍流变量的有限上限，不小于对应下限。 |
| `maximumEddyViscosityRatio` | `100000` | 实数 | 涡黏度与分子黏度比 mu_t/mu 的上限，非负。 schema约束：≥0。 |
| `wallOmegaCoefficient` | `800` | 实数 | 壁面omega系数C，omega_w=C*nu/d²，d使用正壁距下限。 schema约束：≥0。 |
| `enableSourceTerms` | `true` | 布尔 | 是否计算模型的生产/破坏等源项；正常SST计算设true。 |
| `secondOrderReconstruction` | `true` | 布尔 | 湍流变量采用带限制器的二阶Green-Gauss重构；不随流场多项式阶数自动改变。 |
| `transportSubsteps` | `4` | 整数 | 每次流场修正对应的湍流SSPRK3子步预算，至少1。 schema约束：≥1。 |
| `transportTimeScale` | `0.25` | 实数 | 湍流与流场伪时间步比例，范围(0,1]；不改变物理时间步。 schema约束：≥0；≤1。 |
| `wallDistanceMethod` | `1` | 整数 | 壁距算法：0暴力搜索，1 AABB树搜索。 schema约束：≥0；≤1。 |
| `wallDistanceExecution` | `0` | 整数 | 壁距MPI执行：0各rank并行，1串行，>1按rank模值分批执行。 schema约束：≥0。 |
| `wallDistanceSubdivide` | `0` | 整数 | 壁面四边形/曲面细分级别，非负。 schema约束：≥0。 |
| `minimumWallDistance` | `1e-10` | 实数 | 计算模型/壁面值时使用的正壁距下限，必须为正。 schema约束：≥0。 |
| `wallDistanceVerbose` | `0` | 整数 | 壁距计算诊断输出级别，非负；0关闭额外信息。 schema约束：≥0。 |

<!-- config-fields: acmVariable|turbulenceSettings.model;turbulenceSettings.initialValue;turbulenceSettings.farFieldValue;turbulenceSettings.minimumValue;turbulenceSettings.maximumValue;turbulenceSettings.maximumEddyViscosityRatio;turbulenceSettings.wallOmegaCoefficient;turbulenceSettings.enableSourceTerms;turbulenceSettings.secondOrderReconstruction;turbulenceSettings.transportSubsteps;turbulenceSettings.transportTimeScale;turbulenceSettings.wallDistanceMethod;turbulenceSettings.wallDistanceExecution;turbulenceSettings.wallDistanceSubdivide;turbulenceSettings.minimumWallDistance;turbulenceSettings.wallDistanceVerbose -->

### 12.5 `meshSettings`

| 参数 | 默认参考 | 类型 | 中文含义与约束 |
| --- | --- | --- | --- |
| `meshFile` | `""` | 字符串 | CGNS网格路径，按程序启动工作目录解析。 |
| `meshElevation` | `0` | 整数 | 0不提升几何阶次，1将O1几何提升到O2。 schema约束：≥0；≤1。 |
| `meshDirectBisect` | `0` | 整数 | 直接网格二分次数，合法范围0至4；0不细化。 schema约束：≥0；≤4。 |
| `meshReorderCells` | `0` | 整数 | 0不重排局部单元，1启用局部单元重排。 schema约束：≥0；≤1。 |
| `periodicTolerance` | `1e-09` | 实数 | 周期节点匹配容差，非负；需要与网格坐标尺度相容。 schema约束：≥0。 |
| `periodicTranslation1` | `[0, 0, 0]` | 数组 | 第一组周期平移向量，三分量。 |
| `periodicTranslation2` | `[0, 0, 0]` | 数组 | 第二组周期平移向量，三分量。 |
| `periodicTranslation3` | `[0, 0, 0]` | 数组 | 第三组周期平移向量，三分量。 |

<!-- config-fields: acmVariable|meshSettings.meshFile;meshSettings.meshElevation;meshSettings.meshDirectBisect;meshSettings.meshReorderCells;meshSettings.periodicTolerance;meshSettings.periodicTranslation1;meshSettings.periodicTranslation2;meshSettings.periodicTranslation3 -->

### 12.6 `meshSettings.partitionOptions`

| 参数 | 默认参考 | 类型 | 中文含义与约束 |
| --- | --- | --- | --- |
| `metisType` | `"KWAY"` | 字符串 | 网格划分方法：KWAY多路划分，RB递归二分。 schema约束：枚举："KWAY"、"RB"。 |
| `metisUfactor` | `20` | 整数 | METIS负载不平衡控制参数，至少1；沿用METIS ufactor含义。 schema约束：≥1。 |
| `metisSeed` | `0` | 整数 | METIS随机种子；用于固定划分的随机序列。 |
| `edgeWeightMethod` | `0` | 整数 | 划分图边权：0不加权，1按面尺寸加权。 schema约束：≥0；≤1。 |
| `metisNcuts` | `3` | 整数 | METIS尝试的划分次数，至少1。 schema约束：≥1。 |

<!-- config-fields: acmVariable|meshSettings.partitionOptions.metisType;meshSettings.partitionOptions.metisUfactor;meshSettings.partitionOptions.metisSeed;meshSettings.partitionOptions.edgeWeightMethod;meshSettings.partitionOptions.metisNcuts -->

### 12.7 `reconstructionSettings`

| 参数 | 默认参考 | 类型 | 中文含义与约束 |
| --- | --- | --- | --- |
| `type` | `"GreenGauss"` | 字符串 | 流场重构类型：FirstOrder、GreenGauss、Variational。 schema约束：枚举："FirstOrder"、"GreenGauss"、"Variational"。 |
| `variationalIterations` | `3` | 整数 | 固定次数变分重构扫描预算；本模块没有常密度ACM的受控重构额外字段。 schema约束：≥1。 |
| `resetVariationalCoefficients` | `false` | 布尔 | 每次残差求值前是否重置变分重构系数。 |
| `enableLimiter` | `true` | 布尔 | 启用选定的重构限制器。 |
| `limiterType` | `"LocalExtrema"` | 字符串 | LocalExtrema、WBAP或CWBAP；WBAP/CWBAP要求Variational。 schema约束：枚举："LocalExtrema"、"WBAP"、"CWBAP"。 |

<!-- config-fields: acmVariable|reconstructionSettings.type;reconstructionSettings.variationalIterations;reconstructionSettings.resetVariationalCoefficients;reconstructionSettings.enableLimiter;reconstructionSettings.limiterType -->

### 12.8 `outputSettings`

| 参数 | 默认参考 | 类型 | 中文含义与约束 |
| --- | --- | --- | --- |
| `interval` | `0` | 整数 | 外层步场输出间隔；0关闭场输出，初始化场也随之关闭。 schema约束：≥0。 |
| `directory` | `"../data/outACM"` | 字符串 | 输出目录，相对启动工作目录解析。 |
| `prefix` | `"acm"` | 字符串 | 文件及时间序列前缀。 |
| `writeInitial` | `true` | 布尔 | interval>0时是否写初始场。 |

<!-- config-fields: acmVariable|outputSettings.interval;outputSettings.directory;outputSettings.prefix;outputSettings.writeInitial -->

### 12.9 `vfvSettings`

| 参数 | 默认参考 | 类型 | 中文含义与约束 |
| --- | --- | --- | --- |
| `maxOrder` | `1` | 整数 | 重构多项式次数；ACM/ACMVariable支持0至3；2为二次多项式。 schema约束：≥0。 |
| `intOrder` | `1` | 整数 | 全局积分阶次；不等于时间积分阶次。 schema约束：≥0。 |
| `ignoreMeshGeometryDeficiency` | `false` | 布尔 | 是否忽略网格几何缺陷提示。 |
| `nIterCellSmoothScale` | `15` | 整数 | 单元平滑尺度迭代次数，非负。 schema约束：≥0。 |
| `intOrderVR` | `-1` | 整数 | VR积分阶次；负值继承intOrder。 |
| `intOrderVRBC` | `-2` | 整数 | VR边界积分阶次；-1继承intOrder，<-1继承VR积分阶次。 |
| `cacheDiffBase` | `false` | 布尔 | 缓存积分点基函数/导数值，改变存储与计算方式。 |
| `cacheDiffBaseSize` | `255` | 整数 | 缓存的导数基项范围；通用schema为255，按维度构造的求解器对象可能设dim+1。 |
| `jacobiRelax` | `1` | 实数 | 固定VR SOR/Jacobi扫描的松弛系数。 schema约束：≥0。 |
| `SORInstead` | `true` | 布尔 | true用SOR扫描，false用松弛Jacobi。 |
| `smoothThreshold` | `0.01` | 实数 | 重构平滑性指标阈值，参与限制器判别。 schema约束：≥0。 |
| `WBAP_nStd` | `10` | 实数 | WBAP类限制器的n参数。 schema约束：≥0。 |
| `normWBAP` | `false` | 布尔 | 使用多项式范数形式的WBAP处理。 |
| `limiterBiwayAlter` | `0` | 整数 | 双向限制处理：0 wbap-L2，1 minmod。 |
| `subs2ndOrder` | `0` | 整数 | 二阶替代方法：0原VR，1 Gauss，2最小二乘，11 GGMP；主要用于一次多项式场景。 |
| `subs2ndOrderGGScheme` | `1` | 整数 | GG二阶替代插值方式：0不插值，1距离插值。 |
| `svdTolerance` | `0` | 实数 | 重构线性代数SVD容差，0沿用算子相应处理。 schema约束：≥0。 |
| `bcWeight` | `1` | 实数 | 边界重构条件的权重。 schema约束：≥0。 |

<!-- config-fields: acmVariable|vfvSettings.maxOrder;vfvSettings.intOrder;vfvSettings.ignoreMeshGeometryDeficiency;vfvSettings.nIterCellSmoothScale;vfvSettings.intOrderVR;vfvSettings.intOrderVRBC;vfvSettings.cacheDiffBase;vfvSettings.cacheDiffBaseSize;vfvSettings.jacobiRelax;vfvSettings.SORInstead;vfvSettings.smoothThreshold;vfvSettings.WBAP_nStd;vfvSettings.normWBAP;vfvSettings.limiterBiwayAlter;vfvSettings.subs2ndOrder;vfvSettings.subs2ndOrderGGScheme;vfvSettings.svdTolerance;vfvSettings.bcWeight -->

### 12.10 `vfvSettings.baseSettings`

| 参数 | 默认参考 | 类型 | 中文含义与约束 |
| --- | --- | --- | --- |
| `localOrientation` | `false` | 布尔 | 是否使用局部方向构造基函数。 |
| `anisotropicLengths` | `false` | 布尔 | 是否使用各向异性长度尺度。 |

<!-- config-fields: acmVariable|vfvSettings.baseSettings.localOrientation;vfvSettings.baseSettings.anisotropicLengths -->

### 12.11 `vfvSettings.functionalSettings`

| 参数 | 默认参考 | 类型 | 中文含义与约束 |
| --- | --- | --- | --- |
| `scaleType` | `"BaryDiff"` | 字符串 | 泛函尺度类型：MeanAACBB、BaryDiff或CellMax。 |
| `scaleMultiplier` | `1` | 实数 | 泛函尺度倍率，非负。 schema约束：≥0。 |
| `dirWeightScheme` | `"Factorial"` | 字符串 | 方向权重方案：Factorial、HQM_OPT、ManualDirWeight或TEST_OPT。 |
| `dirWeightCombPowV` | `1` | 整数 | 方向权重组合幂参数。 |
| `manualDirWeights` | `[1, 1, 0.5, 0.166666666667, 0.0416666666667]` | 数组 | 手动方向权重向量；只在相应方向权重方案中使用。 |
| `geomWeightScheme` | `"GWNone"` | 字符串 | 几何权重：GWNone、HQM_SD或SD_Power。 |
| `geomWeightPower` | `0.5` | 实数 | 几何权重幂参数。 |
| `geomWeightPower1` | `0` | 实数 | 几何权重第一附加幂参数。 |
| `geomWeightPower2` | `0` | 实数 | 几何权重第二附加幂参数。 |
| `useAnisotropicFunctional` | `false` | 布尔 | 是否启用各向异性泛函。 |
| `tanWeightScale` | `1` | 实数 | 各向异性泛函中的切向权重尺度。 |
| `anisotropicType` | `"InertiaCoord"` | 字符串 | 各向异性类型：InertiaCoord、InertiaCoordBB、Norm、CentDiff、WallDist、InertiaCoordBBNorm或InertiaCoordBBSym；仅相应功能开启时生效。 |
| `inertiaWeightPower` | `1` | 实数 | 惯性权重幂参数。 |
| `geomWeightBias` | `0` | 实数 | 几何权重偏置。 |
| `greenGauss1Weight` | `0` | 实数 | Green-Gauss类型1泛函项权重。 |
| `greenGauss1Bias` | `0.5` | 实数 | Green-Gauss类型1项的偏置。 |
| `greenGauss1Penalty` | `0` | 实数 | Green-Gauss类型1惩罚项系数。 |
| `greenGaussSpacial` | `0` | 整数 | 源码字段拼写为Spacial；0默认空间模式，1均匀权重模式。 |

<!-- config-fields: acmVariable|vfvSettings.functionalSettings.scaleType;vfvSettings.functionalSettings.scaleMultiplier;vfvSettings.functionalSettings.dirWeightScheme;vfvSettings.functionalSettings.dirWeightCombPowV;vfvSettings.functionalSettings.manualDirWeights;vfvSettings.functionalSettings.geomWeightScheme;vfvSettings.functionalSettings.geomWeightPower;vfvSettings.functionalSettings.geomWeightPower1;vfvSettings.functionalSettings.geomWeightPower2;vfvSettings.functionalSettings.useAnisotropicFunctional;vfvSettings.functionalSettings.tanWeightScale;vfvSettings.functionalSettings.anisotropicType;vfvSettings.functionalSettings.inertiaWeightPower;vfvSettings.functionalSettings.geomWeightBias;vfvSettings.functionalSettings.greenGauss1Weight;vfvSettings.functionalSettings.greenGauss1Bias;vfvSettings.functionalSettings.greenGauss1Penalty;vfvSettings.functionalSettings.greenGaussSpacial -->

### 12.12 `顶层`

| 参数 | 默认参考 | 类型 | 中文含义与约束 |
| --- | --- | --- | --- |
| `defaultBoundaryType` | `"BCFar"` | 字符串 | 未按名称显式配置的外边界类型；名称和别名见ACM边界表。 schema约束：枚举："BCFar"、"BCWall"、"BCWallInvis"、"BCWallIsothermal"、"BCOut"、"BCOutP"、"BCIn"、"BCInPsTs"、"BCSym"、"BCSpecial"、"FarField"、"VelocityInlet"、"PressureOutlet"、"NoSlipWall"、"SlipWall"、"Symmetry"。 |
| `initialState` | `[1, 1, 0, 0, 0]` | 数组 | 变密度均匀初值[rho,mx,my,mz,p]，mx=rho*u，二维也保留mz。 |
| `initialDensityAmplitude` | `0` | 实数 | 余弦密度扰动相对幅值；绝对值小于1，schema进一步限制到[-0.999,0.999]。 schema约束：≥-0.999；≤0.999。 |
| `initialDensityWaveNumber` | `[0, 0, 0]` | 数组 | 余弦密度扰动波数[kx,ky,kz]，单位为物理坐标对应的弧度波数。 |
| `boundaryValue` | `[1, 1, 0, 0, 0]` | 数组 | 默认变密度边界状态[rho,mx,my,mz,p]。 |
| `leftState` | `[1, 1, 0, 0, 0]` | 数组 | 无网格通量预览左状态[rho,mx,my,mz,p]。 |
| `rightState` | `[2, 2, 0, 0, 0]` | 数组 | 无网格通量预览右状态[rho,mx,my,mz,p]。 |
| `unitNormal` | `[1, 0, 0]` | 数组 | 预览内核的面法线；物理网格运行使用网格实际面法线。 |
| `nFacesPerRank` | `1` | 整数 | 预览内核的每rank面数；不决定MPI进程数或网格单元数。 schema约束：≥1。 |

<!-- config-fields: acmVariable|defaultBoundaryType;initialState;initialDensityAmplitude;initialDensityWaveNumber;boundaryValue;leftState;rightState;unitNormal;nFacesPerRank -->

### 12.13 `boundaryConditions[]`

| 参数 | 默认参考 | 类型 | 中文含义与约束 |
| --- | --- | --- | --- |
| `type` | `"BCFar"` | 字符串 | 该区域的ACM边界条件类型。 schema约束：枚举："BCFar"、"BCWall"、"BCWallInvis"、"BCWallIsothermal"、"BCOut"、"BCOutP"、"BCIn"、"BCInPsTs"、"BCSym"、"BCSpecial"、"FarField"、"VelocityInlet"、"PressureOutlet"、"NoSlipWall"、"SlipWall"、"Symmetry"。 |
| `name` | `""` | 字符串 | CGNS边界区域名称，区分大小写。 |
| `value` | `[1, 0, 0, 0, 0]` | 数组 | 变密度边界状态[rho,mx,my,mz,p]，五分量；动量不是速度。 |
| `frameOption` | `0` | 整数 | 保留的参考系选项。 schema约束：≥0。 |
| `anchorOption` | `0` | 整数 | 保留的锚定选项。 schema约束：≥0。 |
| `integrationOption` | `0` | 整数 | 保留的边界积分选项。 schema约束：≥0。 |
| `specialOption` | `0` | 整数 | 特殊边界子类型：0恒定状态，1精确平移余弦密度测试。 schema约束：≥0。 |
| `rectifyOption` | `0` | 整数 | 保留的对称修正选项。 schema约束：≥0。 |
| `valueExtra` | `[]` | 数组 | 附加边界数据。 |

<!-- config-fields: acmVariable|boundaryConditions[].type;boundaryConditions[].name;boundaryConditions[].value;boundaryConditions[].frameOption;boundaryConditions[].anchorOption;boundaryConditions[].integrationOption;boundaryConditions[].specialOption;boundaryConditions[].rectifyOption;boundaryConditions[].valueExtra -->

### 12.14 同族入口的schema默认差异

| 入口 | model | fieldNVariables | dimension |
| --- | --- | --- | --- |
| `acmVariable2D.exe` | `VariableDensity2D` | 5 | 由入口确定 |
| `acmVariable3D.exe` | `VariableDensity3D` | 5 | 由入口确定 |

除以上元数据/维度外，同族已导出的字段schema默认值一致。按模型构造后的默认值、相邻默认配置和算例覆盖仍按第2章处理。

## 13 节点中心 NCFV 完整参数表 {#dndsr_config_ch13}

本表共102项。字段组标题是JSON路径前缀；拼接表内参数名得到完整路径。默认参考严格取导出的schema，未声明时标明“未声明”。schema下界为0不表示运行时允许所有0值；正文和中文说明中的正值/组合限制仍需满足。

### 13.1 `solver`

| 参数 | 默认参考 | 类型 | 中文含义与约束 |
| --- | --- | --- | --- |
| `type` | `"ncfv_euler"` | 字符串 | 方程族，必须与可执行程序一致；具体常量见入口对照表。 schema约束：固定="ncfv_euler"。 |
| `discretization` | `"NCFV"` | 字符串 | 空间离散模块：Euler/ACM/ACMVariable为CFV，节点中心求解器为NCFV。 schema约束：固定="NCFV"。 |
| `model` | 见入口对照表 | 字符串 | 编译模型标识；不能仅修改此字符串把程序转换为另一个模型。 schema约束：固定模型值按第1章各入口确定。入口差异见第1章。 |
| `fieldNVariables` | 见入口对照表 | 整数 | 固定为dimension+2：二维4、三维5，按节点中心守恒量布局。 schema约束：固定模型值按第1章各入口确定。入口差异见第1章。 |

<!-- config-fields: ncfv|solver.type;solver.discretization;solver.model;solver.fieldNVariables -->

### 13.2 `顶层`

| 参数 | 默认参考 | 类型 | 中文含义与约束 |
| --- | --- | --- | --- |
| `dimension` | 见入口对照表 | 整数 | 空间维度，必须与2D/3D可执行程序匹配。 schema约束：固定模型值按第1章各入口确定。入口差异见第1章。 |

<!-- config-fields: ncfv|dimension -->

### 13.3 `mesh`

| 参数 | 默认参考 | 类型 | 中文含义与约束 |
| --- | --- | --- | --- |
| `meshFile` | `""` | 字符串 | 输入一阶O1 CGNS网格；当前NCFV不接受O2曲边网格。 |
| `reorderCells` | `0` | 整数 | 重排局部原始单元。 schema约束：≥0；≤1。 |
| `periodicTolerance` | `1e-09` | 实数 | 周期节点匹配容差。 schema约束：≥0。 |
| `periodicLengths` | `[0, 0, 0]` | 数组 | 平移周期长度[Lx,Ly,Lz]；对应分量为0时关闭该方向周期。 |
| `periodicBoundaryPairs` | `[]` | 数组 | 平铺边界名称列表[主1,从1,…]；空值按Periodic区域顺序推断配对。 |

<!-- config-fields: ncfv|mesh.meshFile;mesh.reorderCells;mesh.periodicTolerance;mesh.periodicLengths;mesh.periodicBoundaryPairs -->

### 13.4 `mesh.partitionOptions`

| 参数 | 默认参考 | 类型 | 中文含义与约束 |
| --- | --- | --- | --- |
| `metisType` | `"KWAY"` | 字符串 | 网格划分方法：KWAY多路划分，RB递归二分。 schema约束：枚举："KWAY"、"RB"。 |
| `metisUfactor` | `20` | 整数 | METIS负载不平衡控制参数，至少1；沿用METIS ufactor含义。 schema约束：≥1。 |
| `metisSeed` | `0` | 整数 | METIS随机种子；用于固定划分的随机序列。 |
| `edgeWeightMethod` | `0` | 整数 | 划分图边权：0不加权，1按面尺寸加权。 schema约束：≥0；≤1。 |
| `metisNcuts` | `3` | 整数 | METIS尝试的划分次数，至少1。 schema约束：≥1。 |

<!-- config-fields: ncfv|mesh.partitionOptions.metisType;mesh.partitionOptions.metisUfactor;mesh.partitionOptions.metisSeed;mesh.partitionOptions.edgeWeightMethod;mesh.partitionOptions.metisNcuts -->

### 13.5 `algorithm`

| 参数 | 默认参考 | 类型 | 中文含义与约束 |
| --- | --- | --- | --- |
| `mode` | `"EfficientDifferential"` | 字符串 | EfficientDifferential预计算几何权重；TraditionalQuadrature使用运行时传统积分。 schema约束：枚举："EfficientDifferential"、"TraditionalQuadrature"。 |
| `quadratureOrder` | `4` | 整数 | 传统体积分Gauss/Hammer阶次；不改变二次重构次数。 schema约束：≥2；≤6。 |
| `surfaceQuadratureOrder` | `3` | 整数 | 传统内部界面/边界积分阶次；3可精确积分二次迹多项式。 schema约束：≥3；≤7。 |
| `retainMicroGeometry` | `true` | 布尔 | 保留构建单纯形微几何，供诊断使用。 |
| `checkGeometryClosure` | `true` | 布尔 | 拒绝矢量面积闭合误差不合格的对偶网格。 |
| `closureTolerance` | `2e-10` | 实数 | 对偶面积闭合相对容差。 schema约束：≥0。 |

<!-- config-fields: ncfv|algorithm.mode;algorithm.quadratureOrder;algorithm.surfaceQuadratureOrder;algorithm.retainMicroGeometry;algorithm.checkGeometryClosure;algorithm.closureTolerance -->

### 13.6 `reconstruction`

| 参数 | 默认参考 | 类型 | 中文含义与约束 |
| --- | --- | --- | --- |
| `method` | `"SVDLeastSquares"` | 字符串 | 最小二乘、SVD最小二乘或变分重构；可用表列中文别名。 schema约束：枚举："LeastSquares"、"SVDLeastSquares"、"Variational"、"最小二乘重构"、"SVD最小二乘重构"、"变分重构"。 |
| `stencilSizeFactor` | `1.7` | 实数 | 模板节点数相对于二次多项式基项数的倍率。 schema约束：≥1；≤8。 |
| `maximumStencilRings` | `4` | 整数 | 邻接广度搜索的最大节点环数。 schema约束：≥1；≤8。 |
| `distanceWeightPower` | `1` | 实数 | 最小二乘逆距离权重幂。 schema约束：≥0；≤8。 |
| `distanceWeightFloor` | `0.15` | 实数 | 归一化模板距离的正下限。 schema约束：≥1e-08；≤1。 |
| `svdTolerance` | `1e-11` | 实数 | 奇异值相对秩截断阈值，也通过平方作用于法矩阵特征值。 schema约束：≥0；≤1。 |
| `maximumConditionNumber` | `1e+12` | 实数 | 允许的最大重构条件数估计。 schema约束：≥1。 |
| `variationalIterations` | `30` | 整数 | 传统变分重构的邻居外扫描预算。 schema约束：≥1；≤10000。 |
| `variationalRelaxation` | `1` | 实数 | 传统中点变分扫描松弛系数。 schema约束：≥0；≤1。 |
| `variationalWeight` | `1` | 实数 | 变分w：值跳跃和一阶导数跳跃平方相对于Hessian跳跃的共用权重。 schema约束：≥0。 |
| `enableLimiter` | `true` | 布尔 | 启用单系数Barth–Jespersen限制器。 |

<!-- config-fields: ncfv|reconstruction.method;reconstruction.stencilSizeFactor;reconstruction.maximumStencilRings;reconstruction.distanceWeightPower;reconstruction.distanceWeightFloor;reconstruction.svdTolerance;reconstruction.maximumConditionNumber;reconstruction.variationalIterations;reconstruction.variationalRelaxation;reconstruction.variationalWeight;reconstruction.enableLimiter -->

### 13.7 `physics`

| 参数 | 默认参考 | 类型 | 中文含义与约束 |
| --- | --- | --- | --- |
| `gamma` | `1.4` | 实数 | 理想气体比热比。 schema约束：≥1。 |
| `riemannSolver` | `"Roe"` | 字符串 | 无黏Riemann通量；Roe采用Harten–Yee H2熵修正，Roe_M2不支持。 schema约束：枚举："Roe"、"HLLC"、"HLLEP"、"HLLEP_V1"、"Roe_M1"、"Roe_M3"、"Roe_M4"、"Roe_M5"、"Roe_M6"、"Roe_M7"、"Roe_M8"。 |
| `boundaryMode` | `"FarField"` | 字符串 | 未显式配置边界区域时采用的物理条件。 schema约束：枚举："FarField"、"SlipWall"、"Symmetry"、"NoSlipAdiabaticWall"、"NoSlipIsothermalWall"、"SupersonicInlet"、"SupersonicOutlet"、"PressureOutlet"、"Periodic"。 |
| `initialPrimitive` | `[1, 0.1, 0, 1]` | 数组 | 均匀原始变量初值：二维[rho,u,v,p]，三维[rho,u,v,w,p]。 |
| `farFieldPrimitive` | `[1, 0.1, 0, 1]` | 数组 | 远场原始变量数组，顺序同initialPrimitive。 |
| `requireBoundaryZoneCoverage` | `false` | 布尔 | 要求每个CGNS外边界区域都有显式boundaryZones条目。 |

<!-- config-fields: ncfv|physics.gamma;physics.riemannSolver;physics.boundaryMode;physics.initialPrimitive;physics.farFieldPrimitive;physics.requireBoundaryZoneCoverage -->

### 13.8 `physics.boundaryZones[]`

| 参数 | 默认参考 | 类型 | 中文含义与约束 |
| --- | --- | --- | --- |
| `name` | `""` | 字符串 | CGNS边界区域名称，区分大小写。 |
| `mode` | `"FarField"` | 字符串 | 该区域的边界条件类型。 schema约束：枚举："FarField"、"SlipWall"、"Symmetry"、"NoSlipAdiabaticWall"、"NoSlipIsothermalWall"、"SupersonicInlet"、"SupersonicOutlet"、"PressureOutlet"、"Periodic"。 |
| `primitive` | `[]` | 数组 | 给定原始变量数组；空值使用远场值。 |
| `wallVelocity` | `[]` | 数组 | 壁面速度，二维2分量、三维3分量；空值静止壁。 |
| `wallTemperature` | `1` | 实数 | 等温壁温，与气体常数和代码单位一致。 schema约束：≥0。 |
| `staticPressure` | `1` | 实数 | 压力出口静压。 schema约束：≥0。 |
| `strongState` | `true` | 布尔 | 每个RK阶段后强制施加入口/壁面节点状态。 |

<!-- config-fields: ncfv|physics.boundaryZones[].name;physics.boundaryZones[].mode;physics.boundaryZones[].primitive;physics.boundaryZones[].wallVelocity;physics.boundaryZones[].wallTemperature;physics.boundaryZones[].staticPressure;physics.boundaryZones[].strongState -->

### 13.9 `physics.viscous`

| 参数 | 默认参考 | 类型 | 中文含义与约束 |
| --- | --- | --- | --- |
| `enabled` | `false` | 布尔 | 启用层流Navier–Stokes黏性和热通量。 |
| `model` | `"Sutherland"` | 字符串 | 分子黏度模型：Constant常黏度，Sutherland，DensityProportional密度比例。 schema约束：枚举："Constant"、"Sutherland"、"DensityProportional"。 |
| `dynamicViscosity` | `1e-05` | 实数 | 常数/参考动力黏度。 schema约束：≥0。 |
| `gasConstant` | `1` | 实数 | 理想气体比气体常数。 schema约束：≥0。 |
| `prandtlNumber` | `0.72` | 实数 | 分子Prandtl数。 schema约束：≥0。 |
| `referenceTemperature` | `273.15` | 实数 | Sutherland参考温度。 schema约束：≥0。 |
| `sutherlandConstant` | `110.4` | 实数 | Sutherland常数。 |
| `spectralRadiusFactor` | `4` | 实数 | 显式黏性谱半径倍率，参与稳定时间步估计。 schema约束：≥0。 |

<!-- config-fields: ncfv|physics.viscous.enabled;physics.viscous.model;physics.viscous.dynamicViscosity;physics.viscous.gasConstant;physics.viscous.prandtlNumber;physics.viscous.referenceTemperature;physics.viscous.sutherlandConstant;physics.viscous.spectralRadiusFactor -->

### 13.10 `initialField`

| 参数 | 默认参考 | 类型 | 中文含义与约束 |
| --- | --- | --- | --- |
| `nodeFile` | `""` | 字符串 | 可选节点初值CSV/文本文件：全局节点编号后跟原始或守恒状态。 |
| `isentropicVortex` | `false` | 布尔 | 用解析等熵涡初始化对偶体平均值。 |
| `vortexStrength` | `5` | 实数 | 等熵涡强度beta。 |
| `vortexCenter` | `[5, 5, 0]` | 数组 | 涡初始中心[x,y,z]。 |
| `nodeFileVariables` | `"Primitive"` | 字符串 | 节点文件状态类型：Primitive原始变量，Conservative守恒变量。 schema约束：枚举："Primitive"、"Conservative"。 |
| `nodeFileIndexBase` | `0` | 整数 | 节点文件编号基数，0或1。 |
| `requireCompleteNodeFile` | `true` | 布尔 | 要求文件包含所有本进程拥有的全局节点。 |

<!-- config-fields: ncfv|initialField.nodeFile;initialField.isentropicVortex;initialField.vortexStrength;initialField.vortexCenter;initialField.nodeFileVariables;initialField.nodeFileIndexBase;initialField.requireCompleteNodeFile -->

### 13.11 `initialField.boxes[]`

| 参数 | 默认参考 | 类型 | 中文含义与约束 |
| --- | --- | --- | --- |
| `xMin` | `0` | 实数 | 初值盒区域x坐标最小值。 |
| `xMax` | `0` | 实数 | 初值盒区域x坐标最大值。 |
| `yMin` | `0` | 实数 | 初值盒区域y坐标最小值。 |
| `yMax` | `0` | 实数 | 初值盒区域y坐标最大值。 |
| `zMin` | `0` | 实数 | 初值盒区域z坐标最小值。 |
| `zMax` | `0` | 实数 | 初值盒区域z坐标最大值。 |
| `primitive` | `[]` | 数组 | 区域内初值；Euler用带类型状态，NCFV用与维度匹配的原始变量数组。 |

<!-- config-fields: ncfv|initialField.boxes[].xMin;initialField.boxes[].xMax;initialField.boxes[].yMin;initialField.boxes[].yMax;initialField.boxes[].zMin;initialField.boxes[].zMax;initialField.boxes[].primitive -->

### 13.12 `initialField.planes[]`

| 参数 | 默认参考 | 类型 | 中文含义与约束 |
| --- | --- | --- | --- |
| `a` | `0` | 实数 | 初值半空间ax+by+cz+h≥0的x系数。 |
| `b` | `0` | 实数 | 初值半空间ax+by+cz+h≥0的y系数。 |
| `c` | `0` | 实数 | 初值半空间ax+by+cz+h≥0的z系数。 |
| `h` | `0` | 实数 | 初值半空间ax+by+cz+h≥0的偏置。 |
| `primitive` | `[]` | 数组 | 选中半空间内的初值；状态格式与本模块一致。 |

<!-- config-fields: ncfv|initialField.planes[].a;initialField.planes[].b;initialField.planes[].c;initialField.planes[].h;initialField.planes[].primitive -->

### 13.13 `initialField.expressions[]`

| 参数 | 默认参考 | 类型 | 中文含义与约束 |
| --- | --- | --- | --- |
| `program` | `[]` | 数组 | ExprTk程序，可用inRegion、globalNode、x[3]和UPrim[dimension+2]。 |

<!-- config-fields: ncfv|initialField.expressions[].program -->

### 13.14 `time`

| 参数 | 默认参考 | 类型 | 中文含义与约束 |
| --- | --- | --- | --- |
| `iterations` | `0` | 整数 | 显式SSPRK3迭代步数。 schema约束：≥0。 |
| `timeStep` | `0.0001` | 实数 | 关闭CFL时间步时使用的固定步长。 schema约束：≥0。 |
| `useCFLTimeStep` | `false` | 布尔 | 按对流/黏性谱半径估算显式步长。 |
| `useLocalTimeStep` | `true` | 布尔 | 每个对偶控制体采用局部CFL步长；非定常计算应使用统一物理步长。 |
| `cfl` | `0.5` | 实数 | 显式CFL系数。 schema约束：≥0。 |
| `minimumTimeStep` | `1e-12` | 实数 | CFL步长的下限截断。 schema约束：≥0。 |
| `maximumTimeStep` | `10000000000` | 实数 | CFL步长的上限截断。 schema约束：≥0。 |
| `endTime` | `-1` | 实数 | 终止物理时间；负值关闭基于时间的终止。 |
| `reportInterval` | `10` | 整数 | 残差报告间隔。 schema约束：≥1。 |

<!-- config-fields: ncfv|time.iterations;time.timeStep;time.useCFLTimeStep;time.useLocalTimeStep;time.cfl;time.minimumTimeStep;time.maximumTimeStep;time.endTime;time.reportInterval -->

### 13.15 `io`

| 参数 | 默认参考 | 类型 | 中文含义与约束 |
| --- | --- | --- | --- |
| `writeVTK` | `false` | 布尔 | 启用分布式VTU/PVTU场输出。 |
| `writeInitial` | `false` | 布尔 | 输出初始化/重启后的场。 |
| `writeFinal` | `false` | 布尔 | 输出请求迭代结束后的场。 |
| `outputInterval` | `0` | 整数 | 周期场输出间隔；0关闭周期输出。 schema约束：≥0。 |
| `outputPrefix` | `"../data/out/NCFV/solution"` | 字符串 | 场文件输出路径前缀。 |
| `vtkSeriesName` | `"../data/out/NCFV/solution_series"` | 字符串 | PVTU时间序列路径前缀；空值关闭序列。 |
| `vtkFloatEncoding` | `"binary"` | 字符串 | VTU浮点编码binary/ascii。 schema约束：枚举："binary"、"ascii"。 |
| `asciiPrecision` | `12` | 整数 | ASCII场输出精度，范围1至17。 schema约束：≥1；≤17。 |
| `restartInput` | `""` | 字符串 | 输入重启基础路径，不带.dnds.h5/.dir后缀；空值从头开始。 |
| `restartInterval` | `0` | 整数 | 检查点写入间隔；0关闭周期检查点。 schema约束：≥0。 |
| `writeFinalRestart` | `false` | 布尔 | 结束后写重启文件。 |
| `restartPrefix` | `"../data/out/NCFV/restart"` | 字符串 | 重启输出基础路径。 |
| `writeResolvedConfiguration` | `true` | 布尔 | 在输出旁写出实际解析后的NCFV配置。 |

<!-- config-fields: ncfv|io.writeVTK;io.writeInitial;io.writeFinal;io.outputInterval;io.outputPrefix;io.vtkSeriesName;io.vtkFloatEncoding;io.asciiPrecision;io.restartInput;io.restartInterval;io.writeFinalRestart;io.restartPrefix;io.writeResolvedConfiguration -->

### 13.16 `io.restartSerializer`

| 参数 | 默认参考 | 类型 | 中文含义与约束 |
| --- | --- | --- | --- |
| `type` | `"H5"` | 字符串 | 序列化后端：JSON或H5；读取端必须与文件格式匹配。 schema约束：枚举："JSON"、"H5"。 |
| `hdfDeflateLevel` | `0` | 整数 | HDF5 deflate压缩级别，0关闭压缩。 schema约束：≥0；≤9。 |
| `hdfChunkSize` | `0` | 整数 | HDF5数组分块尺寸。 schema约束：≥0。 |
| `hdfCollOnData` | `false` | 布尔 | HDF5数据数组集体I/O开关。 |
| `hdfCollOnMeta` | `true` | 布尔 | HDF5元数据集体I/O开关。 |
| `jsonBinaryDeflateLevel` | `5` | 整数 | JSON所携带二进制数组的压缩级别。 schema约束：≥0；≤9。 |
| `jsonUseCodecOnUInt8` | `true` | 布尔 | JSON序列化时对uint8数组使用编码器。 |

<!-- config-fields: ncfv|io.restartSerializer.type;io.restartSerializer.hdfDeflateLevel;io.restartSerializer.hdfChunkSize;io.restartSerializer.hdfCollOnData;io.restartSerializer.hdfCollOnMeta;io.restartSerializer.jsonBinaryDeflateLevel;io.restartSerializer.jsonUseCodecOnUInt8 -->

### 13.17 同族入口的schema默认差异

| 入口 | model | fieldNVariables | dimension |
| --- | --- | --- | --- |
| `ncfv_euler2D.exe` | `IdealGas` | 4 | 2 |
| `ncfv_euler3D.exe` | `IdealGas` | 5 | 3 |

除以上元数据/维度外，同族已导出的字段schema默认值一致。按模型构造后的默认值、相邻默认配置和算例覆盖仍按第2章处理。

## 14 较长默认值与结构示例 {#dndsr_config_ch14}

以下按导出schema原样列出长默认值，参数表通过D编号引用。它们是默认参考，不是经过网格运行验证的推荐设置。

### D1 euler：outputControl.consoleMainOutputFormat

```json
[
  "=== Step {termBold}[{step:4d}]   ",
  "res {termBold}{termRed}{resRel:.3e}{termReset}   ",
  "t,dT,dTaumin,CFL,nFix {termGreen}[{tSimu:.3e},{curDtImplicit:.3e},{curDtMin:.3e},{CFLNow:.3e},[alphaInc({nLimInc},{alphaMinInc}), betaRec({nLimBeta},{minBeta}), alphaRes({nLimAlpha},{minAlpha})]]{termReset}   ",
  "Time[{telapsed:.3f}] recTime[{trec:.3f}] rhsTime[{trhs:.3f}] commTime[{tcomm:.3f}] limTime[{tLim:.3f}] limTimeA[{tLimiterA:.3f}] limTimeB[{tLimiterB:.3f}]"
]
```

### D2 euler：outputControl.consoleMainOutputFormatInternal

```json
[
  "\t Internal === Step [{step:4d},{iStep:2d},{iter:4d}]   ",
  "res {termRed}{resRel:.3e}{termReset}   ",
  "t,dT,dTaumin,CFL,nFix {termGreen}[{tSimu:.3e},{curDtImplicit:.3e},{curDtMin:.3e},{CFLNow:.3e},[alphaInc({nLimInc},{alphaMinInc:.3g}), betaRec({nLimBeta},{minBeta:.3g}), alphaRes({nLimAlpha},{minAlpha:.3g})]]{termReset}   ",
  "Time[{telapsedM:.3f}] recTime[{trecM:.3f}] rhsTime[{trhsM:.3f}] commTime[{tcommM:.3f}] limTime[{tLimM:.3f}] limTimeA[{tLimiterA:.3f}] limTimeB[{tLimiterB:.3f}]"
]
```

### D3 euler：outputControl.logfileOutputTitles

```json
[
  "step",
  "iStep",
  "iterAll",
  "iter",
  "tSimu",
  "res",
  "curDtImplicit",
  "curDtMin",
  "CFLNow",
  "nLimInc",
  "alphaMinInc",
  "nLimBeta",
  "minBeta",
  "nLimAlpha",
  "minAlpha",
  "tWall",
  "telapsed",
  "trec",
  "trhs",
  "tcomm",
  "tLim",
  "tLimiterA",
  "tLimiterB",
  "fluxWall",
  "CL",
  "CD",
  "AoA",
  "uMin",
  "uMax"
]
```

## 15 配套算例与schema索引 {#dndsr_config_ch15}

所有路径均相对于仓库根目录；导出文档中的链接指向本机源码。算例包含网格和输出路径，使用前应检查文件是否存在。

| 入口 | schema | 算例目录 |
| --- | --- | --- |
| `euler.exe` | [euler_schema.json](../../cases/euler/2D/euler_schema.json) | [cases/euler/2D](../../cases/euler/2D) |
| `euler2D.exe` | [euler2D_schema.json](../../cases/euler/2D/euler2D_schema.json) | [cases/euler/2D](../../cases/euler/2D) |
| `euler3D.exe` | [euler3D_schema.json](../../cases/euler/3D/euler3D_schema.json) | [cases/euler/3D](../../cases/euler/3D) |
| `eulerSA.exe` | [eulerSA_schema.json](../../cases/euler/2D/SA/eulerSA_schema.json) | [cases/euler/2D/SA](../../cases/euler/2D/SA) |
| `eulerSA3D.exe` | [eulerSA3D_schema.json](../../cases/euler/3D/SA/eulerSA3D_schema.json) | [cases/euler/3D/SA](../../cases/euler/3D/SA) |
| `euler2EQ.exe` | [euler2EQ_schema.json](../../cases/euler/2D/2EQ/euler2EQ_schema.json) | [cases/euler/2D/2EQ](../../cases/euler/2D/2EQ) |
| `euler2EQ3D.exe` | [euler2EQ3D_schema.json](../../cases/euler/3D/2EQ/euler2EQ3D_schema.json) | [cases/euler/3D/2EQ](../../cases/euler/3D/2EQ) |
| `eulerEX.exe` | [eulerEX_schema.json](../../cases/euler/2D/EX/eulerEX_schema.json) | [cases/euler/2D/EX](../../cases/euler/2D/EX) |
| `eulerEX3D.exe` | [eulerEX3D_schema.json](../../cases/euler/3D/EX/eulerEX3D_schema.json) | [cases/euler/3D/EX](../../cases/euler/3D/EX) |
| `acm2D.exe` | [acm2D_schema.json](../../cases/acm/2D/acm2D_schema.json) | [cases/acm/2D](../../cases/acm/2D) |
| `acm3D.exe` | [acm3D_schema.json](../../cases/acm/3D/acm3D_schema.json) | [cases/acm/3D](../../cases/acm/3D) |
| `acmSA2D.exe` | [acmSA2D_schema.json](../../cases/acm/2D/SA/acmSA2D_schema.json) | [cases/acm/2D/SA](../../cases/acm/2D/SA) |
| `acm2EQ2D.exe` | [acm2EQ2D_schema.json](../../cases/acm/2D/2EQ/acm2EQ2D_schema.json) | [cases/acm/2D/2EQ](../../cases/acm/2D/2EQ) |
| `acmSA3D.exe` | [acmSA3D_schema.json](../../cases/acm/3D/SA/acmSA3D_schema.json) | [cases/acm/3D/SA](../../cases/acm/3D/SA) |
| `acm2EQ3D.exe` | [acm2EQ3D_schema.json](../../cases/acm/3D/2EQ/acm2EQ3D_schema.json) | [cases/acm/3D/2EQ](../../cases/acm/3D/2EQ) |
| `acmVariable2D.exe` | [acmVariable2D_schema.json](../../cases/acm/2D/variable_density/acmVariable2D_schema.json) | [cases/acm/2D/variable_density](../../cases/acm/2D/variable_density) |
| `acmVariable3D.exe` | [acmVariable3D_schema.json](../../cases/acm/3D/variable_density/acmVariable3D_schema.json) | [cases/acm/3D/variable_density](../../cases/acm/3D/variable_density) |
| `ncfv_euler2D.exe` | [ncfv_euler2D_schema.json](../../cases/ncfv_euler/2D/ncfv_euler2D_schema.json) | [cases/ncfv_euler/2D](../../cases/ncfv_euler/2D) |
| `ncfv_euler3D.exe` | [ncfv_euler3D_schema.json](../../cases/ncfv_euler/3D/ncfv_euler3D_schema.json) | [cases/ncfv_euler/3D](../../cases/ncfv_euler/3D) |

建议从以下具体算例开始：

- [Euler二维NS](../../cases/euler/2D/euler2D_config.json)
- [Euler三维NS](../../cases/euler/3D/euler3D_config.json)
- [ACM三维SST/BDF2圆柱](../../cases/acm/3D/acm3D_cylinder_Re3000_coarse_SST_BDF2_LUSGS.json)
- [ACMVariable二维](../../cases/acm/2D/variable_density/acmVariable2D.json)
- [NCFV二维](../../cases/ncfv_euler/2D/ncfv_euler2D.json)
- [EulerP周期脚本](../../test/EulerP/test_solver.py)

## 16 来源、范围与核对记录 {#dndsr_config_ch16}

本版本逐一调用19个原生入口的`--emit-schema`，全部与仓库配套schema一致。完整表覆盖Euler 372项、ACM 146项、ACMVariable 127项、NCFV 102项，共747项族内参数路径；跨族相同名字仍各自计数。Euler状态联合格式、边界联合条目、Geom透明对象和EulerP字典在正文补充，因此747不是所有展开子键的总数。

核对包括：参数路径无遗漏、每项中文解释存在、默认参考/枚举/数值上下界来自配套schema、19个入口均列入、源码链接存在、Word和PDF导出及页面检查。说明书核对不代表所有算例网格已读取或长时间求解已验证。

主要实现来源：

- [Euler入口与加载](../../src/Euler/SingleBlockApp.hpp)、[Euler求解器配置](../../src/Euler/EulerSolver.hpp)、[Euler时间/收敛实现](../../src/Euler/EulerSolver.hxx)
- [Euler物性/模型配置](../../src/Euler/EulerEvaluatorSettings.hpp)、[状态换算](../../src/Euler/Physics/PhysicsProperties.hpp)、[Euler边界](../../src/Euler/EulerBC.hpp)
- [CFV重构配置](../../src/CFV/VRSettings.hpp)、[Geom网格选项](../../src/Geom/Mesh/Mesh.hpp)、[共享ODE](../../src/Solver/ODE.hpp)
- [常密度ACM配置](../../src/ACM/ACMConfig.hpp)、[ACM时间推进配置](../../src/ACM/ACMTime.hpp)、[ACM收敛实现](../../src/ACM/ACMTime.cpp)
- [ACMVariable配置](../../src/ACMVariable/ACMConfig.hpp)、[变密度时间推进](../../src/ACMVariable/ACMTime.cpp)
- [NCFV配置](../../src/NCFV/NCFVConfig.hpp)、[NCFV校验](../../src/NCFV/NCFVConfig.cpp)、[NCFV推进](../../src/NCFV/NCFVSolver.cpp)
- [EulerP高层接口](../../python/DNDSR/EulerP/EulerP_Solver.py)、[EulerP物性](../../src/EulerP/EulerP_Physics.hpp)、[EulerP边界](../../src/EulerP/EulerP_BC.hpp)、[EulerP Evaluator配置](../../src/EulerP/EulerP_Evaluator.cpp)
- [原有求解器拆分指南](../guides/solver_split_zh.md)、[基础配置指南](solver_config.md)、[ACM专项说明](acm_json_configuration_manual_zh.md)
