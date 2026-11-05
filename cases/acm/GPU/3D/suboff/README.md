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
实现与冻结本地 CPU 六变量算法对照，远端较新 CPU 的 Anderson/continuation 加速未启用。

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
