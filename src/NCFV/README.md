# NCFV — Node Center Finite Volume Method

`NCFV` 全称为 **Node Center Finite Volume Method**，是 DNDSR 内并列的
独立三阶格点型有限体积求解器模块。它只复用 DNDSR 的 CGNS
读取、网格分区、MPI 数组、JSON 配置和 Euler 黎曼通量，不修改原有
`Geom`、`CFV`、`Euler` 或其他求解器的数据结构与离散流程。数学推导和
路线分析见
[`docs/reports/DNDSR_高效三阶格点型有限体积法_数学推导_MPI设计_可行性报告.pdf`](../../docs/reports/DNDSR_高效三阶格点型有限体积法_数学推导_MPI设计_可行性报告.pdf)。

## 1. 唯一的对偶控制体几何定义

第一版只接受 O1 原始网格。所有构造点都由原始角点的坐标算术平均定义：

\[
\boldsymbol x_e=\frac{\boldsymbol x_i+\boldsymbol x_j}{2},\qquad
\boldsymbol x_f=\frac1{N_f}\sum_{a\in V(f)}\boldsymbol x_a,\qquad
\boldsymbol x_c=\frac1{N_c}\sum_{a\in V(c)}\boldsymbol x_a.
\]

这里的面点和体点不是面积质心或体积质心。对于四边形、六面体、棱柱和
金字塔，两者通常不同；上述顶点等权平均就是算法定义。

- 二维节点份额：对每个拓扑链 `(cell, edge, node)` 构造微三角形
  `conv(node, edge-average, cell-average)`。
- 二维内部对偶面：原始边对应相邻单元中的线段
  `conv(edge-average, cell-average)`；物理边界用
  `conv(node, edge-average)` 闭合。
- 三维节点份额：对每个拓扑链 `(cell, face, edge, node)` 构造微四面体
  `conv(node, edge-average, face-average, cell-average)`。
- 三维内部对偶面：每条原始边对应所有
  `conv(edge-average, face-average, cell-average)` 微三角形的并；物理边界
  使用 `conv(node, edge-average, face-average)`。

初始化阶段保存 cell/face/edge 构造点坐标、节点对偶体零至二阶矩、边宏面
有向面积、边界片及压缩积分权重。对每个节点还检查
\(\sum_e B_{ie}\boldsymbol A_e+\sum_b\boldsymbol A_b=0\) 的几何闭合。

## 2. 两种 JSON 可切换算法

两种方法共用同一套对偶拓扑、二次零均值最小二乘重构、限制器、Euler
通量和 SSPRK3 时间推进，只替换体/面的积分实现。

### `EfficientDifferential`

设一个线段、三角形或四面体微单纯形有 \(n\) 个构造顶点
\(\boldsymbol r_r\)，以原始节点 \(i\) 为展开点，令
\(\boldsymbol d_r=\boldsymbol r_r-\boldsymbol x_i\)、
\(\boldsymbol D=\sum_r\boldsymbol d_r\)。构造点梯度按与坐标完全相同的
顶点系数插值，
\(\boldsymbol g(\boldsymbol r_r)=\sum_m C_{rm}\boldsymbol g_m\)。
将二阶项用全微分

\[
\boldsymbol H_i\boldsymbol d_r
=\boldsymbol g(\boldsymbol r_r)-\boldsymbol g_i+O(h^2)
\]

替换后，单纯形积分可压缩为

\[
\int_K u\,dK=|K|u_i+\sum_m\boldsymbol\ell_{im}^{K}\!\cdot\boldsymbol g_m,
\]

\[
\boldsymbol\ell_{im}^{K}
=\frac{|K|}{n}\boldsymbol D\,\delta_{im}
+\frac{|K|}{2n(n+1)}\left[
\boldsymbol D\sum_r(C_{rm}-\delta_{im})
+\sum_r\boldsymbol d_r(C_{rm}-\delta_{im})\right].
\]

这些权重在初始化时逐微单纯形累加。运行期只存和通信节点点值与一阶梯度；
体积分、左右宏面平均和中心物理通量都由压缩权重计算。每条原始边只调用
一次黎曼求解器形成宏面耗散。该模式不创建、使用或保存任何体、内部面、
边界面的 Gauss 点坐标。

### `TraditionalQuadrature`

保存完整二次重构系数，并把每个对偶微体/微面映射到 DNDSR 已有规则：

- 二维体：`Tri3` Hammer 型规则；二维面：`Line2` Gauss-Legendre 规则；
- 三维体：`Tet4` 单纯形规则；三维面：`Tri3` Hammer 型规则。

体积分点用于初始化对偶体矩和零均值二次基；内部面、边界面积分点在每次
残差计算中评价左右二次多项式并调用数值通量。默认 `quadratureOrder=4`，
足以覆盖三阶空间离散所需的二次多项式积分。

## 3. 重构、通量与 MPI

节点对偶体均值采用完整二次基做加权 SVD 最小二乘。高效模式只保存伪逆的
前 `dimension` 行，因此所得一阶梯度仍来自完整二次问题；传统模式保存
全部 5（二维）或 9（三维）行。模板按原始节点边图逐环扩展，并检查秩与
条件数。

两种模式均按马润之硕士论文式（3-34）使用对偶控制体的分方向半跨度：

\[
\Delta x_i=(x_{\max,i}-x_{\min,i})/2,\quad
\Delta y_i=(y_{\max,i}-y_{\min,i})/2,\quad
\Delta z_i=(z_{\max,i}-z_{\min,i})/2.
\]

令 \(H_i=\operatorname{diag}(\Delta x_i,\Delta y_i,\Delta z_i)\)，
基函数使用 \(\boldsymbol\xi=H_i^{-1}(\boldsymbol x-\boldsymbol x_i)\)，
再减去目标控制体上的体均值；邻居积分也必须使用目标节点的 \(H_i\)。
二次矩按 \(H_i^{-1}M_{2,i}H_i^{-1}\) 缩放，含所有交叉项。
高效模式从完整伪逆截取一次系数后，按
\(\nabla u_i=H_i^{-1}\boldsymbol a_{1,i}\) 还原物理梯度。
传统模式在积分点及限制器采样点使用同一套尺度，并对基函数求物理导数。
代码中纯二次项仍采用 \(\xi_d^2/2\) 的系数约定，不影响归一化及多项式空间。

初始化期间直接从每个微单纯形的构造顶点累积节点相对坐标极值，
缓存 `lowerOffset`、`upperOffset`、`referenceLengths`，不需要积分点。
即使 `retainMicroGeometry=false` 也保留这些缓存；二维只检查 x/y 方向，
未使用的 z 尺度设为 1。退化的有效方向直接报错，不静默退回体积尺度。
普通节点的参考长度随节点 owner/ghost 映射交换；全平移周期节点先在
节点相对坐标中对各份额取 min/max 并集，再计算完整控制体半跨度，
不能对周期盒两端的绝对坐标直接取包围盒。

`lengthScale=V_i^(1/d)` 仅保留给距离权重及原有 CFL/壁面尺度用途，
不再用于重构基函数。此修改不改变距离权重公式、重构模板目标大小、
积分规则或 JSON 模式名称；秩和条件数检查仍基于归一化后的加权矩阵。
重启只读取物理状态，参考长度随网格重新初始化，无需更改重启文件格式。

MPI 分工如下：

1. 原始边以相邻单元 owner 的最小 rank 唯一拥有；跨分区的 edge/face
   全局编号通过规范化顶点键的分布式哈希目录补齐。
2. 节点均值、梯度/二次系数、点值按 DNDSR 节点映射 pull。
3. 边 owner 计算唯一一份积分通量；其他 rank pull ghost 边通量。
4. 节点 owner 按有向关联号 \(B_{ie}=\pm1\) 累加边通量和本地边界通量，
   再除以节点对偶体积。

这样跨分区边的两端严格使用同一份通量，不需要对残差执行 MPI push-sum。

高效模式限制器严格约束论文式（3-94）--（3-100）的宏界面平均值，而不是
原始边中点值。对于界面 \(S_{ij}\) 的 \(i\) 侧，先用预计算权重得到

\[
\bar U_i^{ij}=U_i+S_{ij}^{-1}\sum_m\boldsymbol\omega^i_{ij,m}
\cdot\nabla U_m,
\]

再用端点格点值 \(U_i,U_j\) 构成逐分量上下界。所得单个锚点系数
\(\phi_i\) 作用于该侧完整权重和；不能把和式内每个共享梯度永久乘以各自
节点的系数，否则不再等价于论文式（3-96）。格点值恢复仍使用未限制梯度；
限制系数只在该侧界面均值、物理通量梯度积分和黏性界面梯度中生效。
跨分区节点计算限制器时，会在初始化阶段额外构造所需 ghost 原始边的宏面
权重；关闭限制器时不保存这部分权重。

## 4. 构建与运行

模块目录为 `src/NCFV/`，C++ 命名空间为 `DNDS::NCFV`，库目标为 `ncfv`，
程序目标为 `NCFV`，可执行文件为 `build/app/NCFV.exe`。

```bash
cmake -S . -B build
cmake --build build -t NCFV -j8

cd build
./app/NCFV.exe ../cases/NCFV/NCFV.json
./app/NCFV.exe ../cases/NCFV/NCFV_traditional.json
mpirun --oversubscribe -np 4 ./app/NCFV.exe ../cases/NCFV/NCFV.json
```

模式只由下列 JSON 字段切换：

```json
"algorithm": {
    "mode": "EfficientDifferential"
}
```

改为 `"TraditionalQuadrature"` 即进入普通三阶数值积分路径。二维示例为
`cases/NCFV/NCFV.json` 和 `NCFV_traditional.json`，三维示例为
`NCFV_3d.json`。命令行也支持 JSON pointer 覆盖及配置 schema 输出：

```bash
./app/NCFV.exe --emit-schema
./app/NCFV.exe ../cases/NCFV/NCFV.json \
  -k /algorithm/mode -v TraditionalQuadrature
```

## 5. 黏性、边界、初场与 I/O

历史更名说明：VertexFV → NCFV 的更名本身不改变高效/传统模式或重启数据结构；
后续分方向归一化更新见第 3 节。
新配置及默认输出使用 `data/out/NCFV/`。更名前已完成的计算、重启和视频仍
保留在 `data/out/vertexFV/`，已发布报告的文件名也不变。
历史收敛分析脚本默认继续读取归档结果。重启时可在新配置中将
`io.restartInput` 指向原重启前缀，网格路径则使用 `cases/NCFV/` 下的新位置。
旧源码目录、头文件和程序目标不再作为接口保留。

层流 Navier–Stokes 通量复用 `Euler::Gas` 的守恒量梯度转换、牛顿应力与
傅里叶热传导内核，积分通量为 `F_inviscid - F_viscous`。黏度模型可选
`Constant`、`Sutherland`、`DensityProportional`（此时参数为运动黏度）。
`gasConstant`、`prandtlNumber`、参考温度和 Sutherland 常数必须与输入场
采用同一套量纲。传统模式在面积分点求二次重构的解析梯度；高效模式用
构造顶点插值系数压缩梯度积分，无 Gauss 点。左右重构迹的跳跃只作为黏性
惩罚项加入平均梯度，不代替真实法向导数。

内部黏性面按论文式（4-153）--（4-170）先计算原始变量算术平均
\(\widetilde q=(q_L+q_R)/2\)，再以守恒变量形式构造 dGRP 梯度

\[
\overline{\nabla U}=\frac12
(\overline{\nabla U_L}+\overline{\nabla U_R})
+\frac{\bar U_R-\bar U_L}{2\Delta\widetilde x}\,\boldsymbol n,
\qquad
\Delta\widetilde x=\frac{\min(\Omega_i,\Omega_j)}{S_{ij}},
\]

然后在 \(\widetilde q\) 处用链式法则转换为原始变量梯度。宏面黏性通量乘
标量面积 \(S_{ij}=\sum_k S_{ij,k}\)，而不是合成有向面积向量的模。

`physics.boundaryZones` 按区分大小写的 CGNS 名称配置：`FarField`、
`SlipWall`、`Symmetry`、`NoSlipAdiabaticWall`、`NoSlipIsothermalWall`、
`SupersonicInlet`、`SupersonicOutlet`、`PressureOutlet`。远场采用指定外侧
状态的黎曼通量，压力出口使用给定静压；尚非全特征无反射边界。
无滑移壁可设置 `wallVelocity`，等温壁还需 `wallTemperature`；
`strongState=true` 在初始时刻和每个 RK 子步施加入口/壁面节点条件。
角点处入口优先于壁面，同优先级按区 ID 确定。建议启用
`requireBoundaryZoneCoverage=true` 防止漏配。
高效模式边界无粘通量按论文第 4.3.2 节先在边界格点调用边界通量求解器，
再把通量线性插值到边中点和面顶点并精确积分；不会先插值守恒量再调用
非线性边界通量函数。

`time.useCFLTimeStep=true` 时，`dt_i=CFL*V_i/sum(lambda_conv+lambda_visc)`。
`useLocalTimeStep=true` 为稳态局部伪时间，输出时间记为各步最小步长的累加；
非定常物理时间必须设置 `false`，对全域取 MPI 最小步长。`endTime` 控制
终止时刻并截短最后一步；`iterations` 是本次最多推进的额外步数。

初场顺序是统一状态 → `initialField.boxes` → `planes` → `expressions`
→ `nodeFile`；重启输入优先于这些新初场。节点文件为 CSV 或空白分隔文本：

```text
original_node,rho,u,v,w,p
0,1,1,1,0,1
1,1,1,1,0,1
```

节点编号是 `node2nodeOrig` 的分区无关原始编号；二维去掉 w。
`nodeFileIndexBase` 可为 0 或 1。`nodeFileVariables=Conservative` 时字段为
`rho,rhou,rhov,[rhow],rhoE`。输入值解释为节点对偶体均值；一般光滑解析场
不能将点值直接当作高阶均值。ExprTk 程序可使用 `x[3]`、`globalNode`、
`UPrim[dimension+2]`，设置 `inRegion=1` 并返回 0。

VTK 使用原始网格输出分布式 `.pvtu`/`.vtu`，密度、压力、温度、Mach、
能量、速度和局部步长作为节点数据；这些量由对偶均值转换，非逐点精确值。
`outputInterval`/`restartInterval` 控制频率，`writeInitial`/`writeFinal`/
`writeFinalRestart` 控制首末输出。
H5 重启使用原始节点编号重分配，支持更换 MPI 进程数；JSON 重启逐 rank
存储并校验原始节点次序，要求相同分区。`restartInput` 不含 `.dnds.h5` 或
`.dir` 后缀。完整运行配置会写入 `outputPrefix.resolved.json`。

可运行的黏性示例为 `cases/NCFV/NCFV_viscous.json`。

## 6. 周期等熵涡

`cases/NCFV/NCFV_iv40.json` 使用用户提供的三棱柱网格及论文
式 (3-71)、(3-72)。`prepare_iv40.py` 将旧 CGNS ElementRange 边界转换为
读取器支持的 PointRange/FaceCenter，所有坐标及连接保持原值。

三个周期长度为 `[10,10,4]`，六个边界均配置 `Periodic`。周期节点的原始
对偶体份额以体积加权合并，重构模板使用周期边图和最近平移像；坐标平均、
微体/微面仍在各自原始坐标系构建。高效点值恢复也按完整周期控制体合并。
边通量分布式计算后，将同一周期节点各份额的积分残差相加，再除以完整
控制体积。第一版复制全局周期目录并使用 `MPI_Allgatherv` 交换状态；适合
本验证规模，尚不能作为大规模弱扩展实现。当前仅支持三维全平移周期盒，
要求各方向有足够网格层数、对面节点匹配且关闭限制器。

解析涡的初场在高效模式使用精确一阶导数及预计算全微分权重转换成均值；
传统模式对解析守恒量做体数值积分。两者均非直接把节点点值赋给均值。
`β=5`、中心 `(5,5)`、背景 `(rho,u,v,w,p)=(1,1,1,0,1)`，完整周期为 10；
配置推进至论文三维算例的 `t=2`。黏性及限制器关闭，使用全局物理步长。
`solution.diagnostics.csv` 给出恢复密度点值相对解析解的体积加权误差、
总守恒量和熵偏差；逐 rank `.nodes.rankNNNN.csv` 保存原始编号、坐标、
对偶体份额体积、均值、恢复点值和解析点值，便于独立复核。

```bash
cd build
mpirun --oversubscribe -np 4 ./app/NCFV.exe ../cases/NCFV/NCFV_iv40.json
```

## 7. 测试与当前边界

```bash
cmake --build build -t ncfv_unit_tests -j8
ctest --test-dir build -R '^ncfv_' --output-on-failure
```

`ncfv_test_geometry` 检查线、三角形和四面体全微分公式对任意二次多项式
达到机器精度。`ncfv_test_parallel` 在三维 Hex 网格上同时检查坐标平均、
几何闭合、两种积分存储不变量、1/2/4/8-rank 拓扑以及自由流残差；还在
完整 MPI 对偶宏面上用含交叉项的任意二次多项式分别验证格点恢复权重、
左右界面平均权重和通量矩阵权重，并验证跨分区限制器确实约束宏面平均值。

I/O 测试还验证非均匀 CSV 初场、一步推进、VTK/H5 输出、状态回读及强
无滑移壁条件。当前支持 O1 二维/三维 Euler 和层流黏性流动；O2 曲边、
旋转周期、湍流模型和隐式推进未实现。现有精确积分与自由流测试不是整套
非线性 Navier–Stokes 三阶收敛的证明，仍需系统网格加密与黏性制造解验证。
