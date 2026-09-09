# 二次多项式重构试验运行记录

日期：2026-09-09。生产模块无改动；仅新增独立诊断及后处理程序。

1. 先执行 `CCACHE_DISABLE=1 cmake --build build -t NCFV -j6`，全部 81 项构建步骤成功。构建完成后重新编译并链接诊断，保证没有使用构建中间状态的库。
2. 诊断可执行文件：`/tmp/ncfv-quadratic.uq2LTo/quadratic_reconstruction_probe`。SHA256：`e07564be0272cb536f55d12dde039999fff9155345691e2b0465daeb20d73ca1`。生产源文件及最终 NCFV 库哈希见 `audit.json`。
3. 所有 MPI 运行的工作目录为项目 `build/`，环境为 `OMP_NUM_THREADS=1`，使用 `mpirun --bind-to none`。四套网格对应进程数为 2、4、8、16。
4. 高效模式读取 `cases/NCFV/t2_cfl05_thesis_20260907/ivN.json`，传统模式读取 `cases/NCFV/traditional_t2_cfl05_20260908/ivN.json`，N=10、20、40、80。原 JSON 均未修改，实际诊断覆盖配置保存在各运行 `metrics.json` 中。
5. 输出根目录为 `data/out/NCFV/quadratic_reconstruction_20260909/`，下设 `efficient/ivN/` 和 `traditional/ivN/`。每次运行的日志为根目录下 `efficient_ivN.log` 或 `traditional_ivN.log`。八次 MPI 调用均正常退出，返回码为 0。
6. 每次运行仅调用一次 `ComputeCoefficients` 和一次 `RecoverPointValues`；未调用 `Solver::Run`、残差或通量计算。微四面体几何保留用于独立解析积分核对，高效算法求积点存储计数始终为零。
7. 每个节点 CSV 存储原始节点 ID、坐标、对偶体积、边界标记、五个初始平均值及独立参考均值、五个重构点值和十五个一阶导数。后处理不加载 DNDSR Python 绑定，逐点以 NumPy 长精度矩阵/Hessian 公式重算参考值。
8. 原始平均值及解析积分使用当前生成的同一套微四面体顶点；参考积分公式和体积行列式独立计算。因此可检查几何矩量和积分权重一致性，但不能凭此单独证明 CGNS 输入拓扑本身正确。

这是重构准确性测试，耗时包含网格读取、几何构造、矩阵构造、解析参考积分和 CSV 输出；不同规模任务并行执行。此处耗时不能作为两种完整求解算法的性能比较。
