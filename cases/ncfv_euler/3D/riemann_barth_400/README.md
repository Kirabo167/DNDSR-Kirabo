# 二维四象限黎曼问题：NCFV + Barth–Jespersen 限制器

四组初值逐项采用用户提供的图片，原图保存在同目录
[`initial_states.png`](initial_states.png)，表中每项为 `[rho,u,v,p]`。
三维薄层求解增加 `w=0`，配置内的原始变量顺序为 `[rho,u,v,w,p]`。
特别保留 RP3、RP4 图片中的速度 `0.1`。

| 算例 | 左上：x<0.5，y≥0.5 | 右上：x≥0.5，y≥0.5 | 左下：x<0.5，y<0.5 | 右下：x≥0.5，y<0.5 | 终止时间 |
|---|---|---|---|---|---|
| RP1 | `[0.5323,1.206,0,0.3]` | `[1.5,0,0,1.5]` | `[0.138,1.206,1.206,0.029]` | `[0.5323,0,1.206,0.3]` | 0.25 |
| RP2 | `[0.5065,0.8939,0,0.35]` | `[1.1,0,0,1.1]` | `[1.1,0.8939,0.8939,1.1]` | `[0.5065,0,0.8939,0.35]` | 0.25 |
| RP3 | `[0.5197,-0.6259,0.1,0.4]` | `[1,0.1,0.1,1]` | `[0.8,0.1,0.1,0.4]` | `[0.5197,0.1,-0.6259,0.4]` | 0.30 |
| RP4 | `[1.0222,-0.6179,0.1,1]` | `[0.5313,0.1,0.1,0.4]` | `[0.8,0.1,0.1,1]` | `[1,0.1,0.8276,1]` | 0.20 |

## 网格与数值设置

计算域为 `[0,1] × [0,1] × [0,0.025]`，采用 `400 × 400 × 10`
个均匀 HEXA_8 单元，共 1,600,000 个单元和 1,768,811 个节点。
三个方向间距均为 0.0025。网格文件为
`data/mesh/riemann_barth_400x400x10.cgns`。

`XMIN/XMAX/YMIN/YMAX` 使用 `SupersonicOutlet` 的内状态复制实现零梯度透射；
`ZMIN/ZMAX` 使用静止滑移边界，使初始无展向速度的流场作为二维薄层推进。
这里的 `SupersonicOutlet` 是求解器的边界模式名称，实际使用不要求所有边界
流动都处于超声速。气体为 γ=1.4 的无黏理想气体。

使用 `SVDLeastSquares` 二次重构、`Roe` 黎曼通量和 SSPRK3 时间推进；
`reconstruction.enableLimiter=true` 开启每节点一个标量的 Barth–Jespersen
限制器。四组配置统一采用 `EfficientDifferential` 积分模式。
时间步由全域最小 CFL 步长确定，CFL=0.5，最大时间步为 `1e-4`；
`useLocalTimeStep=false`，终止时刻由 `endTime` 截断至表中时间。

有限体积初值采用各节点对偶控制体内的**精确守恒量体积均值**。先将图片
每个象限的 `[rho,u,v,w,p]` 转为 `[rho,rhou,rhov,rhow,rhoE]`，其中
`rhoE=p/(gamma-1)+rho*(u*u+v*v+w*w)/2`，再按对偶体与象限的相交体积
混合全部五个守恒分量。跨 `x=0.5` 或 `y=0.5` 的界面节点使用两侧各
50% 的守恒状态；交线节点使用四象限各25%；其余节点保持相应象限状态。
物理边界处的对偶体按计算域裁剪。

初始化CSV保存在 `data/mesh/riemann_barth_400_initial/RP*.csv`，包含全部
1,768,811 个节点，使用 x 最快变化的零基 CGNS 原节点编号。
`initialField.nodeFileVariables=Conservative`，并要求文件覆盖全部节点。
配置保留的 ExprTk 表达式用于逐项查阅照片初态；执行顺序为表达式赋值后
再完整加载CSV，因此最终用于重构的状态是精确体积均值。
每个CSV的 `.metadata.json` 保存 γ=1.4、网格/CSV哈希、象限守恒状态与
全域五个守恒量的解析/离散积分校验；全尺寸网格的积分相对误差约为 `1e-15`。

## 运行与保存

从项目根目录生成网格、构建当前工作树的求解器，然后使用统一执行脚本：

```bash
venv/bin/python cases/ncfv_euler/3D/riemann_barth_400/generate_mesh.py
venv/bin/python cases/ncfv_euler/3D/riemann_barth_400/generate_dual_mean_initial.py \
  --mesh data/mesh/riemann_barth_400x400x10.cgns \
  --output-dir data/mesh/riemann_barth_400_initial
cmake --build build -t ncfv_euler3D -j8
venv/bin/python cases/ncfv_euler/3D/riemann_barth_400/run_cases.py \
  --ranks 64 --render
```

网格已生成时使用 `generate_mesh.py --verify-only` 校验；生成脚本拒绝覆盖现有
网格及其元数据。执行脚本依次运行四组算例，为每组保存日志、精确命令、
配置/网格/初值CSV/可执行文件的 SHA-256、退出码和耗时，并检查最终可视化时间、
HDF5 重启时间以及保存配置中的限制器开关。已有非空结果目录会被拒绝覆盖。
`--skip-completed` 会校验输入哈希和最终结果后跳过已有完成记录。
`--binary` 可选择保存的可执行文件快照，`--case-directory` 可选择固定的
四组输入配置目录。运行脚本设置 `mpi_yield_when_idle=1`。

本次正式批处理已脱离终端在后台持续运行，启动命令与PID保存在输出根目录的
`launch.json`，整体进度见 `pipeline_status.json` 和 `pipeline.log`。
执行时固定使用 `inputs/` 中的二进制与配置副本。该账户的继承CPU配额经
`cpu.max` 核验为16核；本次64个MPI进程共享该配额。后续新运行可采用
`--ranks 16`，减少超额调度和通信开销。

单独直接运行一个算例时，从 `build/` 启动，例如：

```bash
cmake --build build -t ncfv_euler3D -j8
cd build
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  mpirun --bind-to none -np 32 ./app/ncfv_euler3D.exe \
  ../cases/ncfv_euler/3D/riemann_barth_400/RP1.json
```

另外三组更换为 `RP2.json`、`RP3.json`、`RP4.json`。
所有相对路径均以求解器运行时工作目录 `build/` 为基准。

每组结果保存到 `data/out/NCFV/riemann_barth_400/RP*/`：

- `solution_*.pvtu` 及分区 VTU：初始、每 30 步和最终的完整三维场；
- `series.pvtu.series`：带真实物理时间的可视化序列；
- `restart_*.dnds.h5`：每 500 步和最终的可重启场；
- `solution.resolved.json`：求解器实际采用的配置；
- `solver.log` 和 `run_manifest.json`：运行日志及自动校验记录。

按步输出在自适应 CFL 下不是等物理时间间隔。`render_results.py` 读取序列
中的真实时间，在 `z=0.0125` 提取二维切片，按统一绝对物理时间轴选取最近
的前一张已保存快照，不作时间插值；算例达到各自终止时刻后保持最终画面。
脚本逐个 VTU 分区读取，检查完整体网格中的密度和压力均为正且有限，并记录
同一 `(x,y)` 在展向不同节点处的数值变化。原始三维快照和最终重启文件保留。
每组预计保存约100张以上的三维快照，实际数量以结果序列为准。

四组计算完成后，也可单独运行后处理：

```bash
EGL_PLATFORM=surfaceless venv/bin/python \
  cases/ncfv_euler/3D/riemann_barth_400/render_results.py \
  --root data/out/NCFV/riemann_barth_400 --workers 2
```

后处理输出到 `data/out/NCFV/riemann_barth_400/videos/`：

- `RP*_density_pressure.mp4`：每组密度、压力双面板视频；
- `RP1_RP4_density_overview.mp4`：四组密度的 2×2 对照视频；
- `RP*_final_density_pressure.png` 和 `RP1_RP4_final_density_overview.png`：最终图；
- `validation.json`：正值、有限值和展向变化检查；
- `render_manifest.json`：统一色标、帧时间及原始快照索引映射；
- `slices/` 和 `frames/`：缓存的二维切片和视频画面。

视频默认 24 fps、演化部分 12 秒、最终画面额外停留 1 秒。

## 40 × 40 × 4 网格稳定性试算记录

以下仅为先行稳定性检查，**不是本次 400 × 400 × 10 网格的计算结果**。
采用 `riemann_barth_pilot40x40x4.cgns`，区域为
`[0,1] × [0,1] × [0,0.1]`，共 6,400 个单元、8,405 个节点；
保留上述 ExprTk 节点赋值初场，以 8 MPI 进程、CFL=0.5、最大时间步
`1e-3` 运行。`run_pilots.py` 仅通过 CLI 覆盖配置，并关闭 VTK、
重启和诊断场输出。下表时间由每步日志中的实际 `dt` 累加核验。

| 积分模式 | 算例 | 完成步数 | 达到时间 | 总耗时 / 秒 | 状态 |
|---|---|---:|---:|---:|---|
| TraditionalQuadrature | RP1 | 324 | 0.25 | 133.042 | 正常到终时 |
| TraditionalQuadrature | RP2 | 427 | 0.25 | 435.389 | 正常到终时 |
| TraditionalQuadrature | RP3 | 369 | 0.30 | 275.137 | 正常到终时 |
| TraditionalQuadrature | RP4 | 287 | 0.20 | 40.884 | 正常到终时 |
| EfficientDifferential | RP1 | 0 | 0 | 1.216 | 第一步前非物理恢复锚点 |
| EfficientDifferential | RP2 | 427 | 0.25 | 114.222 | 正常到终时 |
| EfficientDifferential | RP3 | 369 | 0.30 | 20.464 | 正常到终时 |
| EfficientDifferential | RP4 | 287 | 0.20 | 16.051 | 正常到终时 |

通过的七组均退出码为 0，记录的残差有限，并通过求解器每个 SSPRK3
阶段的守恒状态物理性检查。总耗时包含初始化，期间其他大网格任务的
并发竞争不同，因此本表不能用作两种模式的性能对比。

高效模式 RP1 在 1 MPI 进程下也复现相同的首步前失败，报错位置为
`EfficientSurfaceMean → PreservePhysical`：恢复锚点
`rho=0.09930701285241307`、`rhoE=0.18843831847997253`、
`p=-0.020780201970915384`。该检查发生于时间推进前，降低 CFL 不能
改变这个初始锚点；本记录对应直接节点赋值初场，不包含后续精确对偶
体积均值初场的试验。随后按有限体积定义修正了初始状态的体积平均，
并完成下表四组高效模式验证。

日志及完整命令、网格/配置/可执行文件哈希保存于
`data/out/NCFV/riemann_barth_400/pilots/`，汇总分别为
`summary_np8_cfl05_traditional_fixed.json` 和
`summary_np8_cfl05_efficient_fixed.json`。

## 精确对偶守恒均值的四组先行验证

同一 `40 × 40 × 4` 网格采用上述守恒量体积平均、`EfficientDifferential`、
Barth限制器、4 MPI进程、CFL=0.5、最大时间步 `1e-3`，四组均通过每个
SSPRK3阶段的物理性检查，残差有限，退出码为0。下表终时直接读取最终
HDF5重启文件，并与逐步日志时间累加交叉核验。这些是小网格稳定性证据；
400 × 400 × 10 正式计算结果由各组正式运行清单独立记录。

| 算例 | 步数 | HDF5终时 | 日志时间累加 | 耗时 / 秒 |
|---|---:|---:|---:|---:|
| RP1 | 324 | 0.25 | 0.249999999999947 | 289.633 |
| RP2 | 427 | 0.25 | 0.250000000000042 | 43.473 |
| RP3 | 369 | 0.30 | 0.299999999999997 | 37.570 |
| RP4 | 287 | 0.20 | 0.199999999999983 | 29.275 |

耗时包含初始化，RP1试验期间还存在大网格初始化并发竞争，不能据此比较
各组性能。每组独立命令、CLI覆盖项、CSV/网格/配置/程序哈希、日志和
最终重启场保存在 `pilots/RP*_exact_dual_means_np4/`。

`generate_dual_mean_initial.py` 从CGNS坐标推断尺寸，验证均匀笛卡尔网格
及x最快变化的原始节点顺序，再生成精确对偶均值；生产网格每组有8800个
两象限混合节点和11个四象限混合节点。RP1先行试验使用的初版生成器为
`generate_rp1_pilot_dual_means.py`；通用生成器对同一RP1小网格的输出与
其逐项最大差异 `3.8e-15`，所有守恒分量相符至双精度舍入误差。
