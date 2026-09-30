# NCFV 高效与传统积分：三维 Taylor–Green 涡精度验证

## 问题与参照

- 周期区域：`[-π,π]^3`，使用 `tgv_hex_8/16/32/64.cgns` 四档网格。
- 物理参数：`Re=ρ₀U₀L/μ=1600`，`ρ₀=U₀=L=1`，`μ=1/1600`，
  `M₀=0.1`，`γ=1.4`，`Pr=0.71`；常黏度理想气体 Navier–Stokes。
- 初始场：`u=sin(x)cos(y)cos(z)`，`v=-cos(x)sin(y)cos(z)`，`w=0`；
  `p=1/(γM₀²)+(cos(2x)+cos(2y))(cos(2z)+2)/16`，`ρ=p/[1/(γM₀²)]`。
- 空间算法只切换 `EfficientDifferential` 与 `TraditionalQuadrature`；
  均使用 `SVDLeastSquares` 二次重构、Roe 通量、SSPRK3、不使用限制器。
- 时间步：8³、16³、32³、64³ 分别使用 0.01、0.005、0.0025、0.00125，
  步数分别为 800、1600、3200、6400，物理终点均为 `t=8`。
- 参考：[Dairay 等人的 Incompact3d Re=1600、512³ DNS 数据](https://zenodo.org/records/2577239)。
  本地副本在 `data/reference/taylor_green/TGV_Re1600.dat`。它求解不可压缩
  Navier–Stokes，因此与本算例的低马赫可压缩解存在模型差异。参考 `t=8`
  的动能 `E=0.098289593974`、涡量平方积分 `Z=8.29592515527`、
  耗散率 `ε=0.0103729838977`。

## 误差定义

在每张周期网格上去除正边界的重复节点，计算
`E=⟨|u|²/2⟩`。通过周期 FFT 求速度导数并计算
`Z=⟨|∇×u|²/2⟩`、`ε_ω=2μZ`。报告 `|E-E_DNS|/E_DNS` 与
`|Z-Z_DNS|/Z_DNS`；同时报告同网格两种算法的速度均方根差。
这两个相对误差包含空间、时间和低马赫模型差异，不能单独解释为严格的
空间截断误差。后处理脚本：
`cases/NCFV/diagnostics/analyze_taylor_green.py`。

## 运行状态

完整结果计算中。每个算例的 `run.log`、`timing.txt`、最终 VTK 和
`solution.pvtu.series` 位于
`data/out/NCFV/taylor_green/t8/hex<N>/<efficient|traditional>/`。
分析脚本仅接受 VTK 序列记录的最终时间确为 `t=8` 的文件。

| N | 高效算法 `E(t=8)` | 传统算法 `E(t=8)` | 状态 |
| ---: | ---: | ---: | --- |
| 8 | 1.8382018899e-7 | 1.28661128e-6 | 两者完成；均严重欠分辨 |
| 16 | 0.00269105094 | 0.00982288863 | 两者完成；动能相对 DNS 误差分别为 97.26% 和 90.01% |
| 32 | — | — | 运行中 |
| 64 | — | — | 高效运行中 |

以上中间数据不是最终精度结论。全部算例完成后运行：

```bash
venv/bin/python cases/NCFV/diagnostics/analyze_taylor_green.py
```
