# GPU 程序来源与运行架构

CPU 基线为 3e7cb9413dc26b7836f85213fcde2954ec6cc1b1 加更新时保留的本地改动。
快照由 frozen/active_cpu_snapshot.txt 指定；原 CPU 模块和四个可执行文件保持冻结。
scripts/verify_cpu_freeze.py 校验源码与产物，共 418 项。

ACMGPU、NCFVGPU 从该基线独立复制，当前 CUDA 入口为 RunResident。
设备数值实现是 ResidentACM.cu、ResidentNCFV.cu。显式时间步内的状态、重构、
限制器、边界、黏性/无黏通量、残差、步长和 SSPRK3 阶段均在 GPU。
跨卡 halo 在 GPU 打包/解包；支持 CUDA-aware MPI，其余 MPI 使用固定页主存通信缓冲区中转。
主存中转仅搬运 halo 载荷，所有 CFD 数值计算仍在 GPU；单卡时间步没有场数据中转。
CPU 只负责配置、网格/静态算子预处理、调度和文件读写。

原 FluxBatch 混合求解路径已从空间残差中删除。
FluxBatch.cu 仅保留设备选择和独立通量核验证接口；生产时间步不调用其批量上传/下载。
显式指定 --backend host 才使用副本的 CPU 对照路径；CUDA 入口禁止回退。

当前范围为完整 GPU 显式路径，以及保守六变量 KOmegaSST + BDF2DualTimeLUSGS 路径。
SST 扩展的闭合、输运、壁面、隐式线性化/求解、正值保护、BDF 历史均在设备上；
固定重构及 SGS 调度使用初始化捕获的 CUDA Graph。三维 CPU 对照与独立 Nsight/memcheck 记录位于
data/out/GPU/SUBOFF_SST_BDF2_20261104/。冻结 CPU 保持不变。
通用隐式/GMRES、其他湍流模型、特征 WBAP/CWBAP、重构 GMRES 等未迁移配置明确报错。
构建、范围和独立 Nsight 验证见 docs/guides/gpu_solver_copies.md。
