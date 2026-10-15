# 二维四象限黎曼问题：400×400×5 六面体网格

初值取自对话 `01a13db8-9a59-70d0-bdcf-0c5c9f7d9c95` 已核对的照片。
下表变量顺序为 `[rho,u,v,p]`，三维薄层的展向初速度为零。

| 算例 | 左下 | 右下 | 左上 | 右上 | 终止时间 |
|---|---|---|---|---|---|
| RP1 | `[0.138,1.206,1.206,0.029]` | `[0.5323,0,1.206,0.3]` | `[0.5323,1.206,0,0.3]` | `[1.5,0,0,1.5]` | 0.25 |
| RP2 | `[1.1,0.8939,0.8939,1.1]` | `[0.5065,0,0.8939,0.35]` | `[0.5065,0.8939,0,0.35]` | `[1.1,0,0,1.1]` | 0.25 |
| RP3 | `[0.8,0.1,0.1,0.4]` | `[0.5197,0.1,-0.6259,0.4]` | `[0.5197,-0.6259,0.1,0.4]` | `[1,0.1,0.1,1]` | 0.30 |
| RP4 | `[0.8,0.1,0.1,1]` | `[1,0.1,0.8276,1]` | `[1.0222,-0.6179,0.1,1]` | `[0.5313,0.1,0.1,0.4]` | 0.20 |

域为 `[0,1] × [0,1] × [0,0.0125]`，各方向间距为 0.0025。
共有 800,000 个 HEXA_8 单元、964,806 个节点。
网格侧边为零梯度透射，上下展向边界为 SlipWall。
采用原对话最终的 EfficientDifferential 积分模式、SVDLeastSquares 二次重构、
Roe 通量、SSPRK3、γ=1.4，启用单系数 Barth–Jespersen 限制器。
全域 CFL=0.5，最大时间步 1e-4，不使用局部时间步。

`generate_dual_mean_initial.py` 将每个四象限原始状态先转为守恒量，再按
节点对偶体积与象限的交叠体积求精确均值。跨间断面的节点使用混合守恒均值。
该脚本检查全域守恒积分、正密度与正压力，输出 CSV 与带哈希的元数据。

使用本机项目 Python 环境，无需 DNDSR Python 扩展：

```bash
PY=/home/mrz/.venvs/DNDSR-Kirabo/bin/python
CASE=cases/ncfv_euler/3D/riemann_barth_400x400x5
cmake --build build -t ncfv_euler3D -j8
$PY "$CASE/generate_mesh.py"
$PY "$CASE/generate_dual_mean_initial.py"
$PY "$CASE/run_cases.py" --ranks 64 --jobs 4 --render
```

生成器拒绝覆盖已有网格、初值和结果。已有完整结果可用
`run_cases.py --skip-completed --render` 核验后跳过。求解器从 `build/` 运行。
批处理固定使用结果目录 `inputs/` 中的二进制和配置副本，并记录输入哈希。
四组各 64 个 MPI 进程并发，总计 256 进程。2026-10-15 核实当前主机名为
`cpu704`，账户上级 cgroup 的 `cpu.max=6400000 100000`，即共享 64 核配额；
进程数不会提高配额，且同账户其他计算也共享它。实际运行资源见
`data/out/NCFV/riemann_barth_400x400x5/runtime_resources.json`。

结果保存在 `data/out/NCFV/riemann_barth_400x400x5/`：

- `pipeline_status.json`、`pipeline.log`、`launch.json`：总状态、日志和后台 PID。
- `RP*/run_manifest.json`、`solver.log`：命令、输入哈希、退出码和进度。
- `RP*/solution_*.pvtu` 及 VTU：初始、每 30 步和最终完整场。
- `RP*/series.pvtu.series`：各快照的物理时间。
- `RP*/restart_*.dnds.h5`：每 500 步及最终重启文件。
- `videos/RP*_density_pressure.mp4`：四组密度/压力并排视频。
- `videos/RP1_RP4_density_overview.mp4`：四组密度对照视频。
- `videos/*final*.png`、`validation.json`、`render_manifest.json`：终态图、
  全场正密度/压力检查、展向变化统计及时间映射。

四组都到达指定终止时间，且输出与重启时间核验通过后才开始渲染。
视频在 z=0.00625 取切片，各帧使用固定全局颜色范围。横轴为绝对物理时间，
保存快照之间保持前一状态，各组终止后保持最终状态，不做时间插值。
`pipeline_status.json` 中的 `complete` 表示整条计算和视频流程成功完成。

## RP1 第 1290 步恢复与正性保护

原 RP1 于 2026-10-15 16:17 中止。增加非零 rank 的异常输出后，
从第 1290 步复现确认：第 1296 步后 rank 55 的重构点锚压力成为
`-0.00011012659305826841`，`EfficientSurfaceMean` 的正性保护因锚点自身
非物理而报错。Barth 系数只限制界面修正量，不能修复此前的负压点恢复值。
修复在启用限制器的高效模式下，将非物理点恢复值向守恒均值凸组合收缩。
物理点值、守恒均值、Roe 通量及 CFL 均保持原有设置。

最后原生 H5 检查点为第 1000 步，最后完整 VTK 为第 1290 步。
`recover_snapshot_restart.py` 核验全部 964,806 个节点后，将 VTK 中的
密度和守恒总能量直接保留、动量由密度乘速度恢复，写入独立 H5。
此恢复仅引入浮点舍入，压力往返误差最大约 `6.7e-16`，不裁剪状态。
恢复的物理时间为 `0.09980282772547822`。

`RP1_resume_1290.json` 为续算配置，输出到
`RP1/resume_1290_positive_anchors/`，每 30 步同时保存 VTK 与原生 H5。
`resume_rp1.py` 保存初次失败的 manifest、启动续算、完成后合并时序，
再等待 RP2–RP4 完成，接管最终核验和视频生成。
原 VTK、H5 和初次日志均保留。进度见 RP1 的 `run_manifest.json` 中
`active_solver_log` 指向的日志，以及根目录 `RP1_recovery_status.json`。
