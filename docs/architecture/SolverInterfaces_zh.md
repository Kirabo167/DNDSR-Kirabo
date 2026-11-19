# DNDSR 模块接口

理解一个算例，可以从可执行文件一路跟到它读取的字段。两条主路径共用 DNDS 的 MPI、数组和序列化基础设施。

| 路径 | 网格与场 | 算子和推进 |
|---|---|---|
| Euler、ACM、ACMVariable | Geom 提供分区及邻接，CFV 在单元上组织重构和 DOF | 各求解器的 evaluator 计算通量与 RHS，solver 管理时间推进及隐式求解。EulerP 有独立的设备及 Python 接口。 |
| NCFV | Geom 提供原始网格，NCFV 构造以节点为中心的对偶控制体 | NCFV 管理节点重构、空间残差和时间推进。 |
| Cauchy | NativeGeom 提供实体空间，NativeCFV 准备几何与重构。DNDS Runtime/Algebra 提供字段存储、访问作用域及 halo。 | Cauchy 的 Domain 调度离散和推进，调用 Physics 的热力学和 RANS 点算子。Simulation 管理多域和重启。 |

## 新增前端或配置项

1. 在主程序的 `argparse::ArgumentParser` 上调用 `DNDS::Config::AddSolverArguments(parser)`。它注册 `--check-config`、`--emit-schema` 以及按出现顺序配对的 `-k/-v`。模型、默认算例和重启选项由前端决定。配置检查应在读取网格之前返回。[参数声明](../../src/DNDS/Config/SolverArguments.hpp)
2. 在处理覆盖的那一层调用 `DNDS::Config::ForEachJsonOverride(keys, values, syntax, receiver)`。回调收到 JSON Pointer 和带类型的值。同一个键可以出现多次，后一次覆盖前一次。数组下标和路径中的 `~0`、`~1` 遵循 JSON Pointer 规则。[覆盖值解析](../../src/DNDS/Config/JsonOverrides.hpp)
3. 在模块的配置结构中声明默认值、字段和约束。ACM 两种密度模型共用完整算例读取及旧参数名映射。NCFV 在合并默认值后应用覆盖。Euler 保留算例旁的默认文件、JSONC 及历史字符串转义。Cauchy 在解析域配置后覆盖已有路径并记录来源。跨前端传入字符串时可统一写成 `-v '"Roe"'`。[ACM 兼容入口](../../src/ACMCommon/CaseInput.hpp)

## 为 Cauchy 增加方程块

EquationSchema 描述逻辑块及分量。EquationDofBinding 把逻辑块映射到物理字段及分量偏移。一个流动块与一个湍流块可以共享一行存储，也可以绑定到两个布局各异的字段。进入算子时通过 ReadOwned、ReadWithGhost 或 WriteOwned 获取访问作用域，在作用域内把所需块解析成视图，再把视图传给单元循环。[方程字段接口](../../src/Cauchy/State/EquationDofs.hpp)

遍历物理存储时，用 `DofBundle::BlockAt(i)` 按插入顺序访问，用 `Name(i)` 取得标签。按标签配置的调用方可继续使用 `Block(name)`。编号只在对应的 bundle 内有意义，逻辑块经过绑定后才能确定其物理编号。向 bundle 追加字段可能使先前取得的 C++ 字段引用失效，应在完成注册后取得引用。Runtime 和 EquationDofs 管理访问检查、字段布局、所有权及 ghost 的发布。[DOF 容器](../../src/DNDS/Algebra/DofBundle.hpp)

## ACM 残差报告

`MeasureResidualNorms<N>` 读取本进程拥有的单元，只对 N 个分量的平方和做一次全局求和。每个分量的 RMS 为平方和除以全局拥有单元数后开方。前四项合成流动 RMS，其余活动分量合成湍流 RMS。空分区也参加归约，ghost 不计入分子或分母。分离推进的湍流输运仍提供自己的方程残差。调用方决定传入空间 RHS 还是带时间项的缺陷，收敛判据由时间推进模块管理。[残差统计](../../src/ACM/ACMResidualNorms.hpp)
