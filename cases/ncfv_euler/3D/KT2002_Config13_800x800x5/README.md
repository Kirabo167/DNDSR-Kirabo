# Kurganov–Tadmor（2002）Configuration 13：五方法，800×800×5

参考文献：A. Kurganov and E. Tadmor, *Solution of two-dimensional Riemann problems for gas dynamics without Riemann problem solvers*, Numerical Methods for Partial Differential Equations 18 (2002), 584–608，DOI：10.1002/num.10025。初值见印刷第601页（附件PDF第18页），参考图3.13见第602页（PDF第19页），终止时间 t=0.3。当前RP1对应该文Configuration 3，不是本算例。

## 物理问题与初值

理想气体γ=1.4，初始间断位于x=y=0.5。下表采用原始变量顺序；w始终仅在初值设为零，重构和时间推进不强制z导数或w为零。

| 象限 | ρ | u | v | w | p |
|---|---:|---:|---:|---:|---:|
| 右上，1 | 1 | 0 | -0.3 | 0 | 1 |
| 左上，2 | 2 | 0 | 0.3 | 0 | 1 |
| 左下，3 | 1.0625 | 0 | 0.8145 | 0 | 0.4 |
| 右下，4 | 0.5313 | 0 | 0.4276 | 0 | 0.4 |

`generate_initial.py` 复用已验证的对偶体象限交叠积分：先将四象限原始变量转为守恒变量，再按精确体积分数平均。检查总积分、正密度与正压力，并保存输入哈希，避免在初始间断处直接平均压力造成能量不一致。

生产网格复用320万个HEXA_8单元：Δx=Δy=0.00125，Δz=0.01，域为[0,1]×[0,1]×[0,0.05]。z两面周期，x、y四面常值外推。独立周期节点为801×801×5。论文第III节使用400×400网格点，本轮800×800是原始六面体单元数，二者的离散自由度及算法并不相同。

## 五种方法与时间推进

保持上一轮冻结可执行文件及限制参数，包含SG_BJ_PRES、SG_SIGN_BUDGET、FC_IDP_BINARY、FC_IDP_CONT、LLF_FIRST_ORDER。使用EfficientDifferential、SVDLeastSquares、SSPRK3；高阶候选通量Roe。LLF对照的实际通量是一阶LLF，但冻结程序仍计算未采用的高阶候选。

本轮CFL=0.475，与文献数值设置一致；最大步长1e-4，目标t=0.3。VTK每200步、检查点每100步，另写初始及最终状态。新算例全部从其自身初始场开始，不读取RP1检查点。论文采用二维central-upwind算法；本轮为三维挤出网格上的NCFV，保持这些差别的明确记录。

## 验证与排队

`run_suite.py` 首先核验五份配置及同一可执行文件哈希，再以4 MPI逐组运行40×40×5预检至t=0.3。预检z总厚度1.0，保持Δz/Δx=8，与生产网格纵横比一致；因此不将小网格的层差数值当作大网格预测。要求程序正常退出、时间达到目标、周期独立节点完整、保存状态密度压力有限且为正，并核对限制参数；额外记录越界相关极值、z层差、w和非物理步重试次数。z层差是诊断量，不因保正检查通过而视为已消除。

五组预检均完成，非物理步重试均为零。密度最大z层差依次约8.61e-4、7.12e-4、9.03e-2、1.04e-1、3.77e-15，说明当前高阶方案仍不能严格保持二维性，而且不同RP中方法之间的相对表现可能改变。

预检后自动启动生产作业：每组128 MPI、单进程单线程，最多两组同时运行，固定Linux CPU 0–127和128–255，合计256个不同物理核。其余三组按顺序排队。每个节点采用同一限制策略，不使用额外的z导数修正。启动后保存逐rank实际CPU亲和性及全局物理核无重叠证明。

输出根目录：`data/out/NCFV/KT2002_Config13_800x800x5_np128_20261030/`。`suite_status.json`为总状态，`preflight/results.json`为小网格证据，`physical_affinity_validation.json`为核绑定证据，五个同名目录保存场数据、日志和检查点。完成后自动输出真实五层的物理量分布、层差、统一色标25等值级对比及指标CSV。只有退出码为零且最终检查点达到t=0.3并通过数值检查，才标记为completed。

SIGTERM发送给本控制器会停止它启动的MPI进程组；已有状态文件时拒绝重复启动。没有修改或提交C++源码。

## 五组视频

`render_five_videos_25contours.py`读取五组完整VTK时序，在z=0.02截面生成各组密度/压力并排视频，以及五列、密度上排/压力下排的对照视频。每个物理量在五组及全部保存时刻间共用固定色标和25个内部等值等级，低值蓝、高值红。视频覆盖t=0～0.3，24 fps、504帧，末态停留1秒；保存时刻之间保持前一个状态，不进行时间插值。

```bash
/home/mrz/.venvs/DNDSR-Kirabo/bin/python \
  cases/ncfv_euler/3D/KT2002_Config13_800x800x5/render_five_videos_25contours.py \
  --root data/out/NCFV/KT2002_Config13_800x800x5_np128_20261030 \
  --output data/out/NCFV/KT2002_Config13_800x800x5_np128_20261030/videos_five_cases_t03_25contours_20261031 \
  --renderer cases/ncfv_euler/3D/riemann_barth_400x400x5/render_results.py \
  --workers 4 --z 0.02
```

渲染逐分片检查保存场的密度、压力是否有限且为正，保留切片缓存和统计。每段视频完整解码并核对帧数、分辨率后才写入最终文件；`video_status.json`记录完成情况和SHA256，`render_manifest.json`记录时间映射、色标及等值等级。已完成的目标目录拒绝重复生成。该流程不修改求解器或场文件。
