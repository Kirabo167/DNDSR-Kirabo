# ACM / NCFV CPU 冻结快照

冻结基准为 `54c33894d1a9eed7e56d59fd816d9eebb5820a2b` 加冻结时的工作区改动，
准确时间和文件数见 `metadata.json`。本次没有创建 Git 提交。

- `source.tar.gz`：冻结时 Git 跟踪文件及未忽略的本地源码、配置；包含未提交的 CPU 修改。
- `working_tree.patch`、`git_status.txt`：冻结前的改动及状态记录。
- `bin/`：当时已编译的 `acm2D.exe`、`acm3D.exe`、`ncfv_euler2D.exe`、`ncfv_euler3D.exe`。
- `CMakeCache.txt`、`*.ldd.txt`：构建参数和共享库依赖路径。
- `cpu_sources.sha256`：CPU 模块、入口和原有算例源码校验。
- `artifacts.sha256`：冻结产物校验。

在项目根目录运行 `python3 scripts/verify_cpu_freeze.py` 校验。
恢复源码时解压到**新目录**，避免覆盖当前工作区。

此快照不打包系统共享库、外部依赖构建、Python 环境或未跟踪的网格及计算结果；
运行二进制仍需当前机器上的 MPI/HDF5 等依赖。所有 JSON 路径相对运行目录解析，
在 `build/` 或 `build-gpu-solvers/` 中使用原来的案例命令即可。

后续 GPU 工作在 `src/ACMGPU`、`src/NCFVGPU` 和 `src/GPU` 中进行。
原有 CPU 模块保持冻结时内容，校验脚本会报告后续改变。
