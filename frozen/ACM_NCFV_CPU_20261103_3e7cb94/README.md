# 更新后的 ACM / NCFV CPU 基线

用户要求先更新本地项目，再编写 GPU 版本。当前项目已快进到
`3e7cb9413dc26b7836f85213fcde2954ec6cc1b1`，更新前的本地修改已恢复。
此目录冻结更新后的 CPU 源码与重新编译的四个程序，作为 GPU 数值对照。
旧快照 `frozen/ACM_NCFV_CPU_20261103/` 继续保留。

- `source.tar.gz`：当前 CPU 项目源码、配置及相关未提交文件；排除 GPU 副本、历史快照和大型运行输出。
- `working_tree.patch`、`git_status.txt`：更新并恢复本地修改后的状态。
- `bin/`：重新编译的 ACM、NCFV 二维/三维 CPU 可执行程序。
- `metadata.json`、`CMakeCache.txt`、`*.ldd.txt`：基线提交、构建及动态库依赖信息。
- `cpu_sources.sha256`、`artifacts.sha256`：CPU 源码及快照产物校验。

项目根目录运行 `python3 scripts/verify_cpu_freeze.py` 校验当前基线。
源码恢复时应解压到新目录。系统共享库、外部依赖构建和未跟踪网格不包含在快照中，
二进制依赖本机现有 MPI、HDF5 等库；可执行文件从 `build/` 运行，配置路径规则不变。

更新前所有 636 个本地改动文件另备份于
`data/out/project_update_backup_20261103_175332/`。恢复 stash 时没有冲突，stash 已消费。
