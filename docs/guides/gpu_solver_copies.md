# ACM / NCFV 显存常驻 GPU 显式求解器

项目先更新到 3e7cb94，保留既有本地修改，再冻结 CPU 并创建 GPU 副本。
当前快照为 frozen/ACM_NCFV_CPU_20261103_3e7cb94/，由 frozen/active_cpu_snapshot.txt 选择。
原 CPU 模块和四个二进制没有为 CUDA 修改。
`GPUACM` 分支保存当前源码与配置。重新克隆后使用
`python3 scripts/verify_cpu_freeze.py --sources-only` 核对冻结源码；
默认完整校验还要求原开发机器上的二进制与压缩快照，详见 `frozen/README.md`。

## 实际运行范围

| 程序 | 运行入口 | GPU 数值模块 |
| --- | --- | --- |
| acm_gpu2D.exe / acm_gpu3D.exe | src/ACMGPU/ACMResident.hxx | src/GPU/ResidentACM.cu |
| acm2EQ_gpu2D.exe / acm2EQ_gpu3D.exe | src/ACMGPU/ACMResident.hxx | src/GPU/ResidentACMSST.cuh |
| ncfv_euler_gpu2D.exe / ncfv_euler_gpu3D.exe | src/NCFVGPU/NCFVResident.cpp | src/GPU/ResidentNCFV.cu |

时间步内的整个数值路径在 GPU 执行：重构、限制器、边界状态、周期变换、黏性与无黏通量、
残差装配、CFL、归约和 SSPRK3。状态、系数、梯度、限制因子、残差、步长与历史数组驻留显存。
CPU 只负责配置、网格/静态几何及矩阵预处理、CUDA/MPI 调度、少量控制标量和文件读写。
完整输出场只在输出/重启时下载；多卡另交换 halo 子集。
原 CPU 空间残差入口在 CUDA 模式下会拒绝执行。

ACM 支持常密度层流二维/三维，Roe/Rusanov，一阶、Green-Gauss、变分重构，
至三次多项式，SOR/Jacobi 与固定点收敛控制，LocalExtrema 限制器，现有定常边界族，
黏性、局部/全局 CFL 和显式 SSPRK3。保留 CPU 的变分原地顺序和每步四次残差评估，
避免改变重构温启动历史。

NCFV 支持二维/三维 Roe，LS/SVDLS/变分重构，快速微分与传统积分，
连续/二值 Barth、压力/方向约束、点值恢复限制、共享梯度顺序/并行调度及正负预算，
二值/连续/熵约束通量修正，理想气体层流黏性及热通量、现有边界、CFL 和 SSPRK3 非物理状态重试。
模板选择与几何矩阵在初始化期间生成，运行中的系数求解和限制在 GPU。

当前提供上述**完整 GPU 显式路径**，没有宣称 CPU 全功能等价。
ACM 通用隐式积分、其他 RANS 模型、特征 WBAP/CWBAP、变分重构 GMRES、Green-Gauss 增广变分泛函、
自适应稳态 CFL 尚未迁移；选择它们会明确报错。
NCFV 非 Roe 黎曼求解器尚未迁移；诊断用特殊中点/参考通量接口和涡量诊断 CSV 不在当前 GPU 范围。
NCFV 初值与重启读入通过初始化阶段的 CPU I/O，最终重启文件可照常保存。
ACM VTK-HDF 读入、输出沿用现有格式；变分系数历史不属于该重启格式。

2026-11-04 新增保守六变量 `KOmegaSST` + `BDF2DualTimeLUSGS` 路径，三维已完成 CPU 对照。
设备上计算 SST F1/F2、涡黏性、输运/源项、壁面 omega、重构限制、源项与通量 Jacobian、
局部 LU-SGS、真实线性残差、非线性修正及两级 BDF 历史；第一步 BE，之后 BDF2，
湍流历史违反正值下界时逐方程改用 BE，与冻结 CPU 算法一致。
重构与 LU-SGS 的固定本地依赖调度在初始化时捕获 CUDA Graph，时间步内重复使用。
通用 GMRES、其他闭合模型以及远端较新 CPU 的 Anderson/continuation 加速未迁移。
SST 的连续性输运使用同一 ACM Roe/Rusanov 面体积通量，流动应力使用分子黏度加 SST 涡黏度。
本次新增结果见 `cases/acm/GPU/3D/suboff/README.md`，不能将此前显式版本的验收记录当作 SST 验收。

## 多 GPU 通信

设备内核打包/解包 halo；CUDA-aware MPI 可直接接收设备缓冲区。
普通 MPI 使用初始化时分配的固定页主存通信缓冲区中转，CPU 仅搬运载荷。
重构、限制器、通量、残差、因子合并和时间更新始终在 GPU，不调用 CPU CFD 算法。
单卡时间步内只有少量控制标量跨主机/设备，所有场数据留在显存。
多卡会额外搬运 halo 子集，并单独统计中转字节；不宣称多卡通信完全避免主存。
本机跨 NUMA 的手写 IPC/NCCL 强制直连未通过稳定性验收，已从生产实现删除。
每个 rank 报告 GPU_halo_bytes、MPI_staging_bytes（或 packed_MPI_staging_bytes）和输出字节。

## 编译与运行

本机为 A100（SM80）、CUDA 13.2，其他机器需改为实际编译器和架构。

~~~bash
cmake -S . -B build-gpu-solvers -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DMPI_C_COMPILER=/usr/bin/mpicc -DMPI_CXX_COMPILER=/usr/bin/mpicxx \
  -DDNDS_BUILD_GPU_SOLVERS=ON -DDNDS_USE_CUDA=OFF \
  -DDNDS_USE_CANTERA=OFF -DDNDS_USE_CCACHE=OFF \
  -DDNDS_BUILD_TESTS=ON \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.2/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=80
cmake --build build-gpu-solvers --target all_gpu_solvers gpu_flux_verify gpu_resident_halo_verify -j8
~~~

DNDS_BUILD_GPU_SOLVERS 是独立开关，普通 CPU 构建默认关闭；
DNDS_USE_CUDA 控制既有 EulerP/数组设备后端，无需开启。
配置路径相对启动目录解析，建议从 build-gpu-solvers 启动。

~~~bash
cd build-gpu-solvers
OMP_NUM_THREADS=1 mpirun -np 1 ./app/acm_gpu2D.exe --gpu-device 1
OMP_NUM_THREADS=1 mpirun -np 2 ./app/ncfv_euler_gpu3D.exe

# 显式 CPU 对照。
mpirun -np 1 ./app/ncfv_euler_gpu3D.exe --backend host

# 复用黎曼配置，覆盖输出路径以区分 CPU 结果。
OMP_NUM_THREADS=1 mpirun -np 2 ./app/ncfv_euler_gpu3D.exe \
  ../cases/ncfv_euler/3D/riemann_barth_400/RP1.json \
  -k /io/outputPrefix -v '"../data/out/GPU/RP1/flow"' \
  -k /io/vtkSeriesName -v '"../data/out/GPU/RP1/series"' \
  -k /io/restartPrefix -v '"../data/out/GPU/RP1/restart"'
~~~

默认后端为 cuda。--gpu-device auto 按节点内 rank 选设备，也可指定编号或设置 CUDA_VISIBLE_DEVICES。
没有 GPU、内核错误或不支持的配置会退出，不静默回退 CPU。
大网格需为初始化保留足够主存；尚未给出加速比或 400×400×10 全时段 GPU 性能数据。

## 可复现验证

从项目根目录运行：

~~~bash
venv/bin/python scripts/verify_gpu_solvers.py --device 1 --mpi
venv/bin/python scripts/verify_gpu_solvers.py --transport-only
venv/bin/python scripts/profile_gpu_residency.py --mpi
venv/bin/python scripts/check_gpu_residency.py data/out/GPU/resident_profile_实际时间
python3 scripts/verify_cpu_freeze.py
~~~

数值脚本分别运行冻结 CPU、副本 host 和 resident CUDA，比较 ACM VTK-HDF 流场、
NCFV 每次 ASCII VTU 输出和 MPI 分区，阈值为 5e-10。日志必须实际出现 resident 入口。
脚本不导入 DNDSR Python 绑定，只依赖 numpy、h5py。

通信专项在两套缓冲区间连续 2000 次改变数据宽度，逐值检查双卡 halo。
外部 Nsight Systems 在 NVTX 的 numerical step 区间捕获真实 CUDA API、内核与拷贝。
审计单独统计 MPI halo transport 区间内的通信中转，拒绝其他大块场数据传输、
步内设备分配或没有实际内核的轨迹。单卡通信中转必须为零。
Compute Sanitizer 对两类三维程序分别做内存检查。
验收独立于求解器打印的计数；原始轨迹、日志与 JSON 结果保存在 data/out/GPU/。

最初的仅通量批量 CUDA 实现已删除运行接入，不作为当前验收依据。
FluxBatch 的 2048 组随机/特征值碰撞对照保留为独立数学核检查，
生产时间步直接使用设备函数，不经过整批上传和下载。

## 2026-11-04 验收与交付

75 个配置组合、225 项冻结 CPU / host / CUDA 对照全部通过，最大尺度误差为
4.042846138528678e-14。复用了同一 CPU 基线、相同数值内核的 45 个单进程组合；
通信更新后重新执行全部 30 个双进程组合。各条汇总保留实际日志路径。

- 数值汇总：data/out/GPU/validation_20261104_003731/summary.json。
- 六份 Nsight 原始轨迹与独立审计：data/out/GPU/resident_profile_20261104_003732/。
- 单卡两类程序的时间步场传输均为零；所有轨迹的步内设备分配和 MPI 之外的场传输均为零。
- 多卡通信中转单独计量：四步内每 rank，ACM 为 958464 字节，NCFV 为 657600 字节。
- ACM / NCFV 三维 Compute Sanitizer 均为 0 errors。
- 两套缓冲区共 2000 次不同数据宽度的双卡 halo 检查通过。
- 原 CPU 冻结的 418 项源码/产物哈希匹配。

独立交付目录为 data/out/GPU/resident_explicit_20261104_004303/，包含四个程序、GPU 源码包、
数值汇总、Nsight 审计与 SHA-256 manifest.json。程序使用本机现有 MPI/CUDA/项目依赖；
运行时仍从 build-gpu-solvers 目录解析算例相对路径。
该交付为全 GPU 显式数值版本；上述未迁移算法和 400×400×10 全时段结果/影片尚未完成。

## SUBOFF SST/BDF2 计算

用户附件网格的副本为 `data/mesh/SUBOFF_external3914184_20261104.cgns`。
正式配置为 `cases/acm/GPU/3D/suboff/suboff_SST_BDF2.json`，使用 3914184 个体单元，
Re=12000000、来流 1、rho=1、mu=3.630083418e-7、dt=0.001、20000 步、最多 30 次内迭代。
配置按用户指定保留参考工况；允许内迭代未收敛时推进，结果不自动表示收敛。

~~~bash
venv/bin/python scripts/verify_gpu_sst_bdf2.py
venv/bin/python scripts/profile_gpu_residency.py --sst --mpi
venv/bin/python scripts/run_suboff_gpu.py --detach
~~~

本次任务日志、网格审计、运行状态、流场、原始 Nsight 轨迹、内存检查及二进制备份位于
`data/out/GPU/SUBOFF_SST_BDF2_20261104/`。正式推进状态以 `status.json` / `run.log` 为准，
不把启动进程或通过小网格验证等同于完成 20000 步。
