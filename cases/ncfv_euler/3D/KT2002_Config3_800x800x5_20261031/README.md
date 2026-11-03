# Kurganov–Tadmor Configuration 3：800×800×5，五方法

本轮从t=0计算至文献图3.3对应的t=0.3，与此前只算至t=0.25的RP1结果分别保存。使用当前编译程序和显式`stencilSelectionStrategy=Original`；该名称指当前代码的逐圈反射分组策略，不代表恢复2026-10-26冻结源码中的旧选点实现。没有修改求解器源码。

四象限顺序为左下、右下、左上、右上，变量为`[rho,u,v,w,p]`：

| 象限 | 状态 |
|---|---|
| 左下 | `[0.138,1.206,1.206,0,0.029]` |
| 右下 | `[0.5323,0,1.206,0,0.3]` |
| 左上 | `[0.5323,1.206,0,0,0.3]` |
| 右上 | `[1.5,0,0,0,1.5]` |

复用精确对偶体守恒均值初值及800×800×5六面体网格，域为[0,1]×[0,1]×[0,0.05]，Δz=0.01。z两面周期，x/y四面常值外推。EfficientDifferential、SVDLeastSquares、SSPRK3，γ=1.4，CFL=0.475，最大时间步1e-4。

五组分别为`SG_BJ_PRES`（Barth＋压力约束）、`SG_SIGN_BUDGET`（正负预算）、`FC_IDP_BINARY`（完整通量二元回退与比熵下限）、`FC_IDP_CONT`（完整通量连续限制与比熵下限）、`LLF_FIRST_ORDER`（一阶LLF实际通量，高阶候选仍计算）。限制参数与Config13相同，模板选项显式设置为当前Original。

`run_suite.py`冻结当前可执行文件、NCFV源码、CMakeCache和配置，先逐组进行40×40×5、4 MPI、t=0.3预检。预检厚度1.0，保持生产网格的Δz/Δx=8；要求正常退出、检查点时间正确、节点完整、密度压力有限且为正，并记录展向差异和拒步次数。保正通过不等于振荡和展向误差已经消除。

正式计算每组128 MPI、每进程单线程；两个槽分别为Linux CPU 0–127和128–255，合计256个不同物理核，不使用SMT兄弟线程。其余三组排队自动接续，输出实际逐rank绑定证据。

VTK每250步、检查点每500步保存，并保存初始/最终场。按约30个场快照、16个检查点/组估计总输出约113 GiB；启动要求140 GiB可用，运行时不足10 GiB停止自身作业。无旧结果删除操作。

输出根目录为`data/out/NCFV/KT2002_Config3_800x800x5_np128_20261031/`。总进度见`suite_status.json`，小网格验证见`preflight/results.json`；各组`solver.log`和`run_status.json`记录物理时间与耗时。完成后从最终检查点输出五层分布、层差和统一色标比较；25等值级视频可用本目录的`render_five_videos_25contours.py`生成。

```bash
/home/mrz/.venvs/DNDSR-Kirabo/bin/python \
  cases/ncfv_euler/3D/KT2002_Config3_800x800x5_20261031/run_suite.py
```

已有状态文件时拒绝重复启动。向控制器发送SIGTERM会停止它启动的进程组，保留已有结果。
