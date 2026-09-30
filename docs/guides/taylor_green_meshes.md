# 三维 Taylor–Green 涡周期网格

本仓库的 `scripts/generate_taylor_green_meshes.py` 在 `[-π, π]^3` 上生成均匀
Cartesian 六面体网格，以 CGNS 非结构格式保存。每个文件包含 `HEXA_8` 体单元、
六组 `QUAD_4` 边界面及 `FaceCenter` 边界记录。配对顺序与 NCFV 的
`mesh.periodicBoundaryPairs` 一致：

| 方向 | 负侧（main） | 正侧（donor） | 平移 |
| --- | --- | --- | --- |
| x | `bc-2` | `bc-2-1` | `[2π, 0, 0]` |
| y | `bc-3` | `bc-3-1` | `[0, 2π, 0]` |
| z | `bc-4` | `bc-4-1` | `[0, 0, 2π]` |

已生成的网格位于 `data/mesh/taylor_green/`：

| 文件 | 六面体单元 | 节点 | 单方向间距 `2π/N` | 用途 |
| --- | ---: | ---: | ---: | --- |
| `tgv_hex_8.cgns` | 512 | 729 | 0.785398 | 冒烟检查 |
| `tgv_hex_16.cgns` | 4,096 | 4,913 | 0.392699 | 初步收敛检查 |
| `tgv_hex_32.cgns` | 32,768 | 35,937 | 0.196350 | 收敛检查 |
| `tgv_hex_64.cgns` | 262,144 | 274,625 | 0.098175 | 公开基准最低等效分辨率 |
| `tgv_hex_128.cgns` | 2,097,152 | 2,146,689 | 0.049087 | 二维解析场空间算子加密验证 |
| `tgv_hex_256.cgns` | 16,777,216 | 16,974,593 | 0.024544 | 高效方法限制为 2 层节点环后可运行；传统模式内存不足 |

这些 CGNS 文件位于仓库忽略的 `data/mesh/` 下。重新生成或增加分辨率：

```bash
venv/bin/python scripts/generate_taylor_green_meshes.py --sizes 8 16 32 64
# 需要单独生成 128³ 网格时（目标文件不能已存在）：
venv/bin/python scripts/generate_taylor_green_meshes.py --sizes 128
```

脚本不会覆盖已有文件；若只想重建某档，先由使用者自行处理对应输出文件。
从 `build/` 启动求解器时，网格路径例如
`../data/mesh/taylor_green/tgv_hex_32.cgns`。NCFV 的周期长度应设为
`[6.283185307179586, 6.283185307179586, 6.283185307179586]`，并按上表配置
六个 `Periodic` 边界区和有序配对。

网格本身没有初始流场。当前 NCFV 的 `isentropicVortex` 初场是等熵平移涡，
不是 Taylor–Green 涡；运行 TGV 仍需按目标雷诺数、马赫数和黏度模型设置
相应初场及物性。这里的 `64³` 指每方向 64 个一阶六面体单元；对高阶方法，
应以自由度等效分辨率与公开结果比较。

## 公开验证资料

- [HiOCFD5 Taylor–Green 涡 Re=1600 基准](https://how5.cenaero.be/node/11)：
  三周期立方体，要求 Cartesian 等效分辨率 `64³`、`128³`、`256³`；验证量包括
  动能耗散率、涡量平方积分和能谱。其[完整说明 PDF](https://how5.cenaero.be/sites/how5.cenaero.be/files/2025-01/BS1_TaylorGreenVortexRe1600_0.pdf)
  给出初始条件及结果提交约定。
- [Nektar++ 开源教程](https://doc.nektar.info/tutorials/5.6.0/incns/taylor-green-vortex/incns-taylor-green-vortexch1.html)：
  提供 `TGV64_mesh.msh` 和 `TGV128_mesh.msh` 的 Gmsh 网格以及求解配置。
  这些文件面向 Nektar++，不能直接替代本目录中的 DNDSR CGNS 文件。
