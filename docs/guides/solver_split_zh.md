# 求解器拆分、配置迁移与上游对比 {#solver_split_zh}

本文记录 2026-10-09 的求解器组织方式。CPU 求解器按 **Euler/CFV、ACM、NCFV Euler**
三类组织，每类提供独立的二维和三维可执行目标。ACM 常密度和变密度分别保留自己的
实现；Euler 保留上游的各个模型目标。`EulerP` 的 Python/CUDA 路径维持原有组织。

## 1. 可执行文件和模型

Linux 可执行文件位于所选构建目录的 `app/` 下，文件名带 `.exe`；CMake 目标不带后缀。

| 目标 | `solver.type` | `solver.model` | 网格维度 | 流场变量数 |
| --- | --- | --- | --- | --- |
| `euler` | `Euler` | `NS` | 2D | 5 |
| `euler2D` | `Euler` | `NS_2D` | 2D | 4 |
| `euler3D` | `Euler` | `NS_3D` | 3D | 5 |
| `eulerSA` / `eulerSA3D` | `Euler` | `NS_SA` / `NS_SA_3D` | 2D / 3D | 6 |
| `euler2EQ` / `euler2EQ3D` | `Euler` | `NS_2EQ` / `NS_2EQ_3D` | 2D / 3D | 7 |
| `eulerEX` / `eulerEX3D` | `Euler` | `NS_EX` / `NS_EX_3D` | 2D / 3D | JSON 指定的动态长度 |
| `acm2D` / `acm3D` | `ACM` | `ConstantDensity2D` / `ConstantDensity3D` | 2D / 3D | 4 |
| `acmVariable2D` / `acmVariable3D` | `ACMVariable` | `VariableDensity2D` / `VariableDensity3D` | 2D / 3D | 5 |
| `ncfv_euler2D` / `ncfv_euler3D` | `ncfv_euler` | `IdealGas` | 2D / 3D | 4 / 5 |

Euler 和 ACM 的 `solver.discretization` 为 `CFV`，NCFV Euler 为 `NCFV`。
`ACM` 是常密度三维入口的兼容名称；旧的单一 `NCFV` 构建目标已经由两个
`ncfv_euler*` 目标替代。`euler` 的统一分派入口已移除，运行 ACM 或 NCFV 时应选择对应程序。

`euler` 保持上游 NS 模型的二维几何、三个速度分量、五个守恒变量约定；
`euler2D` 使用两个速度分量和四个守恒变量。二维网格不能作为互换两个入口的充分条件。
常密度 ACM 始终存储 `[u,v,w,p]`，变密度 ACM 始终存储 `[rho,mx,my,mz,p]`。

## 2. 按需要构建

从项目根目录配置，然后构建所需目标：

```bash
cmake --preset release-test
cmake --build build --target euler2D acm2D ncfv_euler2D --parallel 4

# 每类的聚合目标
cmake --build build --target all_euler --parallel 4
cmake --build build --target all_acm --parallel 4
cmake --build build --target all_ncfv_euler --parallel 4

# 全部求解器；等价的 build preset 为 solvers
cmake --build --preset solvers --parallel 4
```

默认构建包含 `euler`；其他求解器按指定目标构建。`acm`、`ncfv_euler` 构建 preset
分别构建该类全部变体。公共网格、MPI 和数值工具复用；ACM/NCFV 的二维、三维实例化
分别建库，因此构建一个维度不需要另一维度的求解器实例化，也不链接其他求解器家族。

拆分允许只编译所需模块，但不会自动限制同时运行的编译任务。内存紧张时先降低
`--parallel`，必要时使用 `--parallel 1`，并检查 LTO、优化选项和单个编译单元的大小。
本次没有测量并报告编译内存降幅。

EulerEX 的反应流功能需要 Cantera：

```bash
cmake --preset reactive-test
cmake --build build-reactive --target all_euler --parallel 4
```

`eulerEX` 和 `eulerEX3D` 也可在 Cantera 关闭时编译，但该构建不能运行
`reactiveFlow.enabled=true` 的配置。`eulerState` 保留，`canteraConstVolTrajectory`
和 `cantera_Test` 在启用 Cantera 时可用。

## 3. cases 目录与路径

```text
cases/
├── euler/
│   ├── 2D/                 euler、euler2D；SA/、2EQ/、EX/ 子目录
│   ├── 3D/                 euler3D；SA/、2EQ/、EX/ 子目录
│   └── defaults/           共享的 Euler 默认参数说明
├── acm/
│   ├── 2D/                 常密度；variable_density/ 子目录
│   └── 3D/                 常密度；variable_density/ 子目录
└── ncfv_euler/
    ├── 2D/                 二维算例和 schema
    └── 3D/                 三维算例、网格、manifest 和 diagnostics/
```

schema 位于对应模型的算例目录中。旧 `cases/NCFV` 的网格和诊断工具移至
`cases/ncfv_euler/3D`。目录清单还见仓库的 `cases/README.md`。

网格、输出和重启路径仍相对于**启动时的工作目录**，通常为 `build/` 或 `build-reactive/`；
移动 JSON 到新子目录不会改变这个约定。`$schema` 路径相对于 JSON 文件自身。
机制文件还可由 `DNDS_MECH_PATH` / `CANTERA_DATA` 定位。

```bash
(cd build && ./app/euler2D.exe ../cases/euler/2D/euler2D_config.json)
(cd build && ./app/acm2D.exe ../cases/acm/2D/acm2D.json)
(cd build && mpirun -np 4 ./app/ncfv_euler3D.exe \
    ../cases/ncfv_euler/3D/NCFV_periodic_hex_iv10.json)
```

## 4. JSON 元数据与默认参数

维护的配置带有与可执行文件匹配的 `solver` 对象。二维 NCFV 的示例片段为：

```json
{
    "solver": {
        "type": "ncfv_euler",
        "discretization": "NCFV",
        "model": "IdealGas",
        "fieldNVariables": 4
    },
    "dimension": 2
}
```

这只是元数据片段，完整可运行配置还需网格、物理参数等字段。
NCFV 的变量数为 `dimension + 2`；固定长度模型要求 `fieldNVariables` 与模型匹配。
EulerEX 优先使用 JSON 中的动态长度，也保留 `eulerEX.exe 14 config.json` 的上游位置参数。

Euler 从指定配置所在目录读取 `<目标>_default_config.json`（存在时），再合并 case 和
`-k/-v` 覆盖；默认文件不存在时，从该模型的内存默认参数开始。程序不会在启动时
覆盖已有默认文件。上游会先写出默认 JSON，所以比较结果时必须核对最终合并参数，
不能只比较用户 case 文件。

ACM/ACMVariable 使用完整单文件配置，不搜索相邻默认文件；其 loader 会补齐已实现的
兼容字段。NCFV 从 `Configuration` 默认值合并配置，并补齐边界和初场数组元素的默认项。
加载行为及主要配置段见 [配置手册](../solver-guide/solver_config.md)。

本次迁移补齐了 ExprTk 的 `stateType`、常密度 ACM 输出/重启字段、旧默认模板参数和状态长度。
11 份历史 NCFV 配置中的已禁用 `Roe_M2` 改为受支持的 `Roe`。旧计时或误差结果仍对应
原来的通量设置，比较新配置前应重新计算。

## 5. 校验与 schema 更新

`--check-config` 检查类型、参数和模型匹配，不读取网格，也不生成求解结果。
它不证明外部输入文件齐全或算例已收敛。

```bash
(cd build && ./app/acm2D.exe ../cases/acm/2D/acm2D.json --check-config)
(cd build && ./app/ncfv_euler3D.exe --emit-schema)
python3 cases/validate_configs.py --quiet
python3 scripts/check_solver_cases.py --build-dir build \
    --reactive-build-dir build-reactive
```

完整配置检查需要对应入口均已构建。没有 Cantera 构建时，审计脚本会将反应流配置列为
需要另一构建的项目，而不是按非反应流运行。schema 更新脚本需要九个 Cantera Euler
入口；另外六个 ACM/NCFV 入口可来自同一构建或第二参数指定的 CPU 构建：

```bash
cmake --build --preset schemas --parallel 4
bash cases/update_schemas.sh build-reactive build
# 仅生成和检查，保留现有 schema
DNDS_SCHEMA_CHECK_ONLY=1 bash cases/update_schemas.sh build-reactive build
```

新配置 CTest 依赖全部求解器入口；仅构建 `all_unit_tests` 不会构建这些程序：

```bash
cmake --build build --target all_solvers all_unit_tests --parallel 4
ctest --test-dir build -L solver_config --output-on-failure
ctest --test-dir build -R '^(acm|ncfv|euler)_' --output-on-failure
```

这两个配置工具只读取 JSON 或调用 C++ 程序，不导入 DNDSR Python 扩展；运行 pytest
之前仍须按 [构建指南](building.md) 重建并安装 pybind11 模块。

## 6. 与上游的关系与验证范围

2026-10-09 核对的上游 main 为
[`080b221dc6cd291aaafa5cb8af6e037c83f89607`](https://github.com/CFDLAB-THU/DNDSR/tree/080b221dc6cd291aaafa5cb8af6e037c83f89607)。
本次拆分未改动 Euler 的通量、残差、时间推进实现。这次比较中，`EulerSolver.hxx`、
`EulerSolver_Init.hxx`、`EulerSolver_PrintData.hxx`、`EulerEvaluator.hxx`、
`EulerEvaluator_EvaluateRHS.hxx`、`EulerEvaluator_EvaluateDt.hxx` 和 `EulerEvaluator.hpp`
与该上游版本逐字一致。

本地在拆分前就已有其他差异：Wilcox 生产项限幅使用 `config.productionLimit`
（默认仍为 20）；三维 `normWBAP` 使用对应维度的多项式范数；周期信息复制和重排
等共享网格实现也与该上游不同。这些依赖可能影响相应算例，不能把上述文件一致性
表述为全部求解结果与上游严格一致。

截至本次验证：CPU 构建生成 17 个可执行文件，29 项 ACM/NCFV/Euler 测试通过；
Cantera 构建生成 12 个可执行文件，15 项 Euler/反应流测试通过。其中 Euler 相关测试
共 22 次执行。142 份算例/默认配置通过原生加载和 schema 校验，11 项错误模型/维度
拒绝检查通过，9 个独立入口的双进程单步小算例通过。二维 Euler 入口的短算例使用
均匀流保持检查。这些是本地功能与回归验证，没有完成同条件上游逐步场解/残差对照。

本机还有 67 份配置显式指定的源网格不可用。保留了原网格名称，没有换用其他网格
改变其物理问题；大型工程算例未全部推进到收敛。网格 manifest 是数据说明文件，
不作为求解器配置校验。提交到 Git 的文件包含常用周期网格；107 MiB 的
`periodic_tet_iv80.cgns` 及更大的加密棱柱网格仍为本地生成数据。需要 Tet4 N=80 网格时，
先构建 CGNS 依赖，再运行：

```bash
python3 cases/ncfv_euler/3D/diagnostics/generate_periodic_tet_hex_meshes.py \
    --topologies tet --sizes 80
```

本机 OpenMPI 在图形设备探测阶段曾挂起，验证时采用以下环境设置；其他机器应按
实际 MPI 环境选择：

```bash
env -u DISPLAY HWLOC_COMPONENTS=-gl OMPI_MCA_btl=self,vader OMPI_MCA_pml=ob1 \
    ctest --test-dir build -L solver_config --output-on-failure
```

hwloc 对图形探测的说明见 [官方环境变量文档](https://www.open-mpi.org/projects/hwloc/doc/v2.14.0/envvar.html)。
