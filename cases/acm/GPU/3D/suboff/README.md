# SUBOFF：GPU SST/BDF2

用户指定保持链接任务的 SST/BDF2 工况，先补齐独立 GPU 副本再运行。
冻结的 `src/ACM`、`src/NCFV`、对应 CPU 入口与四个冻结二进制均保留，418 项哈希检查通过。

正式配置：[suboff_SST_BDF2.json](suboff_SST_BDF2.json)。相对路径以 `build-gpu-solvers/` 为工作目录。
网格副本为 `data/mesh/SUBOFF_external3914184_20261104.cgns`，附件未修改。
SHA-256：`76251a5c33b2ed41436a857b9b46b34ff3129df6f46cf72d972475b0feb1e085`。

## 工况

| 参数 | 值 |
| --- | --- |
| 体单元 / 节点 | 3914184 / 1538575 |
| 棱柱 / 四面体 / 金字塔 | 2654017 / 1259772 / 395 |
| 来流 / 常密度 | U=1 / rho=1 |
| 分子动力黏度 / 名义 Re_L | 3.630083418e-7 / 12000000 |
| ACM beta² / alpha | 4 / 0.5 |
| 闭合 / 通量 | KOmegaSST / Roe |
| 初始及远场 k / omega | 8.333524834710744e-11 / 0.2295684113865932 |
| 物理推进 | BE 启动，之后 BDF2；dt=0.001，20000 步，共 20 时间单位 |
| 内迭代 | BDF2DualTimeLUSGS，最多 30 次，松弛 0.7 |
| 伪时间 | 局部 CFL=40，最大伪时间步 0.5 |
| 重构 | maxOrder=2、intOrder=4，变分 SOR 1 次，重置系数，LocalExtrema |
| 保存间隔 | 每 100 步及末步，VTK-HDF 流动和湍流场 |

入口 INLET：给定速度；出口 OUTLET：给定压力；SUBMARINE：无滑移；
BOTTOM、ICE_WALL、SIDE_NEG、SIDE_POS：对称/无穿透。边界映射遵循参考配置。
网格全部体单元的采样 Jacobian 为正；该检查不等同于完整网格质量/离散误差认证。

参考配置明确设置 `dualTimeRequireConvergence=false`，所以耗尽 30 次内迭代会继续推进。
每个物理步的六方程缺陷与 `converged` 标志写入日志。达到总步数不自动表示内层或稳态收敛。
实现与 CPU 六变量算法对照；2026-11-10 已同步 main `88635e1` 的 SST 修正和隐式输运耦合，
最新又同步 main `c62e501` 的显式各向同性 k 应力，当前 CPU 快照由
`frozen/active_cpu_snapshot.txt` 选择。SST 默认存储静压，旧续算文件通过
`pressureIsModified=true` 转换；新静压文件应设为 false。验证记录见
`docs/dev/gpu_main_sync_c62e501_20261110.md`。Anderson/continuation/matrix-free GMRES 尚不在 CUDA 范围，
选择时明确拒绝执行。单卡定常续算与检查点工具继续保留，完整 SUBOFF 计算保持暂停于
最后保存的 15850 步。同步验证及未解决问题见 `docs/dev/gpu_main_sync_20261110.md`。

## GPU 范围

SST 闭合、流动/湍流输运、限制、FD 线性化、4×4 流动 LU-SGS、2×2 湍流隐式块、
真实线性残差、非线性修正、BDF 历史和正值保护均在 GPU。初始壁距/静态几何在初始化阶段处理。
CUDA Graph 复用固定的重构及 LU-SGS 本地依赖顺序，未改变 SOR/SGS 算法。
单卡时间步没有场数据主存中转；本机双卡的 MPI halo 仍通过固定页主存中转，单独计数。
保存输出时下载流场。现有 VTK-HDF 流场不包含完整两级 BDF 历史与重构历史，不能视为精确续算快照。

## 验证与运行

~~~bash
cmake --build build-gpu-solvers --target acm2EQ_gpu3D -j3
venv/bin/python scripts/verify_gpu_sst_bdf2.py
venv/bin/python scripts/profile_gpu_residency.py --sst --mpi
venv/bin/python scripts/run_suboff_gpu.py --detach
~~~

正式输出目录：`data/out/GPU/SUBOFF_SST_BDF2_20261104/`。

- `status.json`：后台进程、物理步进度、时间、最近缺陷与收敛标志。
- `run.log`：网格初始化、设备、全部物理步及实际通信计数。
- `fields/`：周期输出与最终 VTK-HDF 流场。
- `verification/summary.json`：11 种配置 × 单/双卡 × CPU/host/CUDA，66 项全部通过；最大尺度误差 4.3884577697469487e-13。
- `verification_extra/summary.json`：额外 12 项，多次线性残差修正及强刚性衰减历史的 BE 回退均通过，最大尺度误差 4.490480868570054e-9；`validation_summary.json` 汇总全部 78 项。
- `profile/residency_audit.json`：实际 Nsight 轨迹，无步内分配、无 halo 之外的场传输。
- `profile/acm_sst_memcheck.log`：Compute Sanitizer，0 errors。
- 原有显式路径另做 120 项回归对照，全部通过，最大尺度误差 6.38378239159465e-16；目录见 `manifest.json` 的 `explicit_regression`。
- `mesh_audit.json`、`manifest.json`、`bin/`、`gpu_source_at_launch.tar.gz`：输入与程序留档。

后台管理器在失败时记录错误，不更换物理模型或转用 CPU；结束时检查所有指定物理步及最终场有限性。
是否已经完成以运行状态和原始日志为准。

## 从第 4100 步改为 dt=0.01

`suboff_SST_BDF2_restart4100_dt0p01.json` 从原两进程输出的第 4100 步
（t=4.1）导入速度、压力、k、omega 及密度加权 SST 变量，保持原两进程划分。
读取器检查单元类型、连接及顺序；原配置和原输出不覆盖。

VTKHDF 不含完整 BDF/重构历史，所以必须显式设置
`restartSettings.resetBDFHistory=true`，并给出 `physicalTime=4.1`。
第一新步为 BE（第 4101 步，t=4.11），之后为固定 dt=0.01 的 BDF2。
重构系数重新建立；这不是原 dt=0.001 轨迹的逐位连续恢复。

保留原终止物理时间 t=20，因此追加 1590 步，最终绝对步号为 5690。
每 10 步输出一次，仍对应物理时间间隔 0.1；另外保存导入初场和末场。
内迭代上限仍为 30，允许未达到内迭代收敛条件时推进。

~~~bash
venv/bin/python scripts/verify_gpu_field_restart.py \
  --output data/out/GPU/SUBOFF_SST_BDF2_restart4100_dt0p01_20261105/verification
venv/bin/python scripts/run_suboff_gpu.py --detach --ranks 2 \
  --config cases/acm/GPU/3D/suboff/suboff_SST_BDF2_restart4100_dt0p01.json \
  --output data/out/GPU/SUBOFF_SST_BDF2_restart4100_dt0p01_20261105
~~~

新目录中的 `status.json` 区分绝对步号、追加步数和物理时间。
验证覆盖单/双进程、显式缺失历史拒绝、导入场完全一致、BE/BDF2 顺序、
改变步长后的输出时间以及 host/CUDA 场结果对照。

### 后续设置：100 次内迭代，总步号 10000

用户随后将内迭代上限设为 100，并指定总计算步数为 10000。
`suboff_SST_BDF2_restart4110_dt0p01_inner100_end10000.json` 从最新完整输出
第 4110 步（t=4.2）续算，保持 dt=0.01，追加 5890 步至总步号 10000、t=63.1。
`nSteps` 表示本段追加步数，因此配置值是 5890。
仍以一步 BE 重置缺失的历史，随后使用 BDF2，保存间隔保持 10 步。

~~~bash
venv/bin/python scripts/run_suboff_gpu.py --detach --ranks 2 \
  --config cases/acm/GPU/3D/suboff/suboff_SST_BDF2_restart4110_dt0p01_inner100_end10000.json \
  --output data/out/GPU/SUBOFF_SST_BDF2_restart4110_dt0p01_inner100_end10000_20261105
~~~

各阶段配置、日志和流场均保留；`data/out/GPU/SUBOFF_active_run.json` 指向当前阶段。

### 最新设置：恢复 30 次内迭代，总步号保持 10000

`suboff_SST_BDF2_restart4130_dt0p01_inner30_end10000.json` 按用户要求将上限恢复为 30，
从上一阶段最新完整输出第 4130 步（t=4.4）导入初场。
dt 仍为 0.01，追加 5870 步至总步号 10000、t=63.1。
缺失的历史仍用一步 BE 重新启动，之后恢复 BDF2；所有既有输出保留。

### 单卡续算：第 6890 步起，保持 3914184 单元

`suboff_SST_BDF2_restart6890_singleGPU_dt0p01_inner30_end10000.json` 从双卡最后完整保存的
第 6890 步（t=32.0）迁移到单 MPI 进程、物理 GPU 1，追加 3110 步至 10000。
dt=0.01、内迭代上限 30、SST 和全部空间参数保持原值。场文件没有 BDF 历史，
因此第 6891 步用 BE，之后恢复 BDF2。

原来的三次容量二进制在两卡合计占用约 113 GiB，不能原样放入一张 80 GB 卡。
本算例为三维二次重构，以 `-DDNDS_GPU_ACM_MAXIMUM_MODES=9` 编译，
缩减未使用的多项式容量，所有实际使用的双精度系数、积分点和数值算法保持原值。
默认容量仍为 19；要计算三维三次重构，需要重新配置为 19 并重编 GPU 目标。

`scripts/remap_acm_vtkhdf_restart.py` 使用同一网格的单进程输出作排列模板，
验证点坐标完全相同、单元类型及每个单元的完整顶点集合一致，建立双射后
精确重排全部 CellData，不使用流场插值。转换报告、单元映射及原二进制保存在
`data/out/GPU/SUBOFF_single_gpu_migration_20261106/`。初始化模板与正式求解都使用
OMP_NUM_THREADS=8；改变网格排列或线程设置时应重新生成并验证模板。

~~~bash
cmake -S . -B build-gpu-solvers -DDNDS_GPU_ACM_MAXIMUM_MODES=9
cmake --build build-gpu-solvers --target acm2EQ_gpu3D -j4
venv/bin/python scripts/run_suboff_gpu.py --detach --ranks 1 --gpu-device 1 \
  --config cases/acm/GPU/3D/suboff/suboff_SST_BDF2_restart6890_singleGPU_dt0p01_inner30_end10000.json \
  --output data/out/GPU/SUBOFF_SST_BDF2_singleGPU_restart6890_dt0p01_inner30_end10000_20261106
~~~

正式续算前应完成小网格容量 19/9 与 host/CUDA 对照；正式启动后再次核对初始输出场
与转换文件完全相同，并确认单卡推进到 BE/BDF2 两步及首次新流场输出。

### 定常局部时间步：从第 10000 步结果继续至空间残差收敛

`suboff_SST_steady_LUSGS_local_singleGPU_restart10000_damped.json` 使用单卡最后保存的
第 10000 步、t=63.1 流场，选择 `ImplicitEulerLUSGS`，移除全部物理时间导数和
BDF 历史项。每次定常迭代重新计算各单元的局部 CFL 步长，执行一次伪瞬态
LU-SGS 修正；SST 源项 Jacobian 和所有数值更新继续在 GPU 上执行。
`physicalTimeStep` 仅为旧配置格式保留，定常路径不使用它，物理时间保持 63.1。

`steadyUntilConverged=true`、`nSteps=0` 表示不以预设迭代数作为完成条件。
六个空间方程分别以本次求解初始 RMS 为固定基准，要求残差小于
`max(初始残差 × 1e-6, 对应绝对阈值)`，并连续满足 20 次。
绝对阈值依次为 `[1e-8,1e-8,1e-8,1e-8,1e-10,1e-3]`（u,v,w,p,k,omega）。
全局停止条件不会用每步重置的初始残差，也不会把步数耗尽或拒绝更新视为收敛。
固定 CFL=0.5、修正阻尼=0.3，每轮重新计算各单元的局部伪时间步。
关闭残差接受过滤和自适应 CFL，允许定常伪时间响应中的暂时残差波动；
只有六个实际空间残差满足上述固定目标才视为收敛。
每 50 次迭代及真正收敛时保存场；输出文件名记录定常迭代号，series 的物理时间保持不变。

~~~bash
venv/bin/python scripts/run_suboff_steady_gpu.py --detach --gpu-device 1 \
  --config cases/acm/GPU/3D/suboff/suboff_SST_steady_LUSGS_local_singleGPU_restart10000_damped.json \
  --output data/out/GPU/SUBOFF_SST_steady_LUSGS_local_singleGPU_restart10000_damped_20261109
~~~

控制器记录 `status.json`、`residual_history.csv`、固定参考与目标残差。
仅在程序报告收敛、连续窗口内全部六个残差再次核验通过、最终保存场有效时标记 completed；
程序异常或未收敛退出标记 failed，不把预算消耗完当作成功。
小网格检查程序为 `scripts/verify_gpu_steady_sst.py`，覆盖 host/CUDA 更新对照、
物理时间步长无关性、无上限迭代与连续收敛窗口、空间残差回溯和 BDF2 回归。

### 保存场 SST 残差诊断

`DNDS_GPU_SST_HOST_AUDIT_FILE=/绝对路径/spatial_audit.json` 配合 `--backend host`
只对导入场求一次 CPU 参考空间残差，统计六方程 RMS、每个方程残差最大的十个单元，
以及已经达到湍流下限但 RHS 仍为负的单元数和残差贡献。该入口不推进状态，
不进入 CUDA 驻留分支；完整网格初始化仍需要主存和预处理时间。
用 `-k /restartSettings/flowFile -v '"/绝对路径/检查点.vtkhdf"'` 指定待查场，
同步设置 `restartSettings/completedSteps`，并设置 `outputSettings/interval=0`。

`DNDS_GPU_SST_AUDIT_FILE=/绝对路径/cells.bin` 配合 `--backend cuda` 则直接诊断
设备离散算子，必须有足够显存加载完整驻留工作区。该版本输出头为三个 uint64
`[version=1, ownedCells, columns=49]`，随后是逐单元 float64 数据。
`scripts/analyze_gpu_sst_audit.py` 检查格式并生成带列名的摘要 JSON。

`scripts/verify_gpu_sst_closure.py` 从生产头文件提取实际设备函数，验证均匀衰减、
近壁剪切、刚体旋转、正负交叉扩散和壁面 omega 的解析值，并检查非正交扩散
原反例被有界面电导消除。该保护仅约束扩散，不证明含源项的完整更新保持正性。
2026-11-13 的独立 GPU 修复还包含最小二乘 SST 梯度及分离的对流入流状态；
`scripts/audit_gpu_sst_model.py` 执行实际生产面通量和隐式面核，验证四类入流边界、
体积通量与重构速度方向不一致、局部极值和非正交线性场。
最初冻结 CPU 不含这些修复。用户随后授权以 `dcf4662` 为基线同步纯 CPU SST，
2026-11-16 的新面电导算法仅修改独立 `ACMGPU` host 和 CUDA，默认对照为这两条路径；
纯 CPU 保持 2026-11-13 冻结算法，显式选择 CPU 时仅作历史比较，不能要求更新结果逐项相同。
`scripts/audit_suboff_sst_local.py` 可独立复算保存场的局部源项与分子扩散；
它对扭曲面的几何处理是近似的，应以完整参考/设备残差诊断确认实际残差。

### 2026-11-15 双卡定常续算

`suboff_SST_steady_local_restart33000_dualGPU_spatial_fixed_20261115.json`
从第 33000 步静压检查点继续，保留完整 3914184 单元、二次 CFV、局部 CFL=0.5、
阻尼 0.3、每步一次修正和三次 LU-SGS 扫描，以及原有六方程绝对目标和连续 20 次判据。
两卡空闲显存不对称，因此采用三个等规模 MPI 分区，rank 0 放在 GPU0，rank 1、2
放在 GPU1。单元数分别为 1304754、1304719、1304711；该分配不保证显存或耗时严格按比例。

原单卡检查点保持不变。`scripts/remap_acm_vtkhdf_restart.py` 按目标网格模板重排，
逐一验证唯一顶点坐标、单元类型和完整顶点集合；所有单元场直接重排，没有插值。
目标模板和正式运行均使用 OMP_NUM_THREADS=8、同一网格及 METIS 设置。
三进程小网格的精确初场导入、设备映射和 host/CUDA 定常更新对照通过，
24 个场数组的最大缩放误差为 3.21e-13 以下。

~~~bash
venv/bin/python scripts/run_suboff_gpu_test.py \
  --config cases/acm/GPU/3D/suboff/suboff_SST_steady_local_restart33000_dualGPU_spatial_fixed_20261115.json \
  --output data/out/GPU/SUBOFF_SST_steady_local_restart33000_dualGPU_spatial_fixed_20261115 \
  --rank-devices 0 1 1 --device-min-free-mib 28672 53248
~~~

队列要求 GPU0 至少空闲 28 GiB、GPU1 至少空闲 52 GiB，为分区 halo 和设备工作区
预留空间；两卡分别满足才启动，不能仅将空闲显存相加。显存检查属于启动预检，
不保证其他进程随后不会申请显存。所有流动及 SST 数值核仍在 GPU 执行；
当前 MPI halo 传输使用设备打包与固定页主存中转。输出每 100 步及收敛终步保存。
部署状态以 `data/out/GPU/SUBOFF_active_run.json` 指向的 status.json 为准；
小网格验证通过不代表完整 SUBOFF 已经收敛。

### 2026-11-16 SST 有界面扩散

独立 `ACMGPU` host 与 CUDA 将原来的精确极值开关替换为连续状态差驱动的面电导：
`c = clamp(g/dq, 0, 2*|n|²/|dn|)`，其中 g 为非正交修正后的法向导数，
dq 为两侧单元/边界幽灵状态差，dn 为中心连线的法向投影。dq=0 时显式通量严格为零，
隐式系数取两点值。扩散通量直接计算为 `diffusivity*dq*c`，避免大切向梯度重新投影时的
相消误差；隐式块冻结并使用同一个 c，包括边界幽灵状态导数。内部面仍使用同一共享通量。
这会限制反向及过大的非正交修正；强偏斜网格上的任意仿射场不保证完全保持原通量。
该限制不证明对流、源项及完整耦合更新的正性或收敛性。

原第 45100 步完整 3914184 单元检查点的只读复算中，处于下限且 RHS 为负的
k/omega 单元数由 67/5 变为 0/0。原场尚未适应该算子，k RMS 从 0.02752 升至 0.05208、
omega RMS 从 1.1071e7 升至 4.1216e7，不能据此报告收敛改善。
二维/三维、单/双进程的 80 项 host/CUDA 对照、32 项实际 CUDA 面核检查、
三进程双 GPU 精确重排续算和显存访问检查均通过。纯 CPU 保持冻结快照，
完整 GPU 正式续算未因这些测试自动启动。记录位于
`data/out/GPU/SST_bounded_flux_fix_20261116/validation_summary.json`。

### 2026-11-16 入口湍流参数对比

`suboff_SST_steady_local_restart45300_Tu0p1_dualGPU_20261116.json` 仍从原 GPU 第 45300 步
完整场恢复，保持双卡三分区、定常局部 CFL=0.5、松弛 0.3 和原六方程收敛目标。
只改变入口/远场湍流值及未使用检查点时的初始值：

| 参数 | 原值 | 新值 |
| --- | --- | --- |
| k | 8.333524834710744e-11 | 1.5e-6 |
| omega | 0.2295684113865932 | 1.1478420569329661 |
| 入口湍流强度 | 0.0007453646% | 0.1% |
| omega × 艇长 / U | 1 | 5 |
| 名义入口 nu_t/nu=k/(nu*omega) | 0.001 | 3.5999173 |

取艇长 L=4.356、U=1，按 `k=1.5*(Tu*U)^2`、`omega=5*U/L` 换算。
原始 SST 的低湍流范围使用近似计算域长度；该网格域长约为 5 倍艇长，原 omega×域长/U≈5，
因此不能因其 omega×艇长/U=1 就宣称原设置错误。
新值用于参数敏感性研究，不保证解决不收敛，也不代表原参考工况的实测入口条件。

[Spalart–Rumsey 2007](https://ntrs.nasa.gov/api/citations/20070035069/downloads/20070035069.pdf)
区分入口和艇体附近环境湍流，并讨论低 omega 及衰减可能造成的收敛问题；
[NASA CFL3D](https://nasa.github.io/CFL3D/Cfl3dv6/cfl3dv6_new.html) 给出湍流强度换算和
环境参数表；[SUBOFF 风洞实验](https://www.mdpi.com/2077-1312/12/9/1576) 报告约 0.1% 来流强度，
其偏航、介质和雷诺数条件与本任务不同。
[TMR SST](https://tmbwg.github.io/turbmodels/sst.html) 给出的壁面系数为 `10*6/0.075=800`，本次保持不变。

模型仍为原 SST1994，没有加入 SST-sust 源项，也未将全域下限提高至环境湍流值。
新入口值仍会衰减，不能视为全域恒定 Tu=0.1%。导入的旧场不重置；压力、速度和湍流变量均保留。
GPU 小网格验证覆盖三进程双卡、旧场导入完全一致和新入口条件下的十次定常更新，
不构成完整 SUBOFF 网格的收敛验证。文献审查及测试记录在
`data/out/GPU/SUBOFF_SST_turbulence_parameter_study_20261116/`。

### 2026-11-17 UUID 队列与初始化内存核查

`suboff_SST_steady_local_restart45300_Tu0p1_dualGPU_20261117.json` 从相同的完整第 45300 步
检查点恢复，数值设置与前一天的 Tu=0.1% 配置相同，输出使用新目录。
前一天的任务在初始化阶段被 OOM 终止，没有完成新的迭代。不能将此终止视为数值发散。

队列支持 `--rank-device-uuids UUID0 UUID1 UUID1`，与 `--rank-devices` 互斥。
`--device-min-free-mib 28672 53248` 在 UUID 模式按不同 UUID 首次出现的顺序解释；
每次检查重新解析可见编号，缺卡时状态为 `waiting_for_gpu_access`，不会退化为单卡运行。
`--minimum-host-free-gib 145` 同时检查账号 cgroup 余量及系统可用主存。
本地服务上限为 145 GiB，账号当时上限为 160 GiB；这属于资源配置，不保证初始化一定能容纳。
运行资源、GPU 可见编号和访问权限会变化，应以当前队列状态为准。

初始化仍保留 CPU 网格、重构及工作区，同时构建完整的主存上传包，直到 GPU 构造完成才释放。
三进程日志中的 `payloadGiB` 合计约 58.27 GiB，仅为部分静态数组的小计，不是实测主存峰值；
11 月 15 日和 16 日该小计相同，最近的面扩散修改没有新增按网格规模分配的数组。
独立核查中，三进程各 32 次 resident 构造/销毁及 256 次定常更新后，追踪的设备和锁页分配归零；
64 单元真实网格的 300 次 CUDA 定常更新在 `memcheck --leak-check full` 下报告零泄漏、零错误。
主存 Valgrind 的 8 次小网格更新未比 MPI 初始化基线增加泄漏记录。
这些小规模结果不代替完整网格峰值测量，也不构成收敛或所有异常路径无泄漏的证明。
原始场、硬件采样、计算日志及本地内存审计记录保存在忽略的 `data/out/GPU/`，不随源码发布。
