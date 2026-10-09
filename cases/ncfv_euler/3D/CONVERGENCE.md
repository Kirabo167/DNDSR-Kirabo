# 高效模式等熵涡网格收敛算例

方法名称为 **NCFV — Node Center Finite Volume Method**，三维可执行文件为
`ncfv_euler3D.exe`。下文的误差和
进程数记录来自更名前已完成的计算，历史数据路径保持不变。这里的新运行命令
使用更名后的配置，新生成的结果写入 `data/out/NCFV/`；后处理脚本默认仍读取
`data/out/vertexFV/` 中的历史基准，不会自动切换到新结果。

当前配置中的已禁用 `Roe_M2` 已迁移为 `Roe`；以下历史误差和计时仍对应旧通量设置，
比较新配置时须重新计算。

本组验证使用用户提供的 iv10、iv20、iv40、iv80 三维三棱柱网格。
四套基准算例保持相同物理参数、高效积分模式、CFL=0.5、
步长上限 0.01，并全部推进到物理时间 t=2。
iv40 沿用此前的视频基准算例，不混用传统积分的结果。

## 输入网格

使用 prepare_iv40.py source.cgns destination.cgns 创建读取器兼容副本。
脚本名称虽然包含 iv40，也适用于本组其余三个输入。
只把旧 CGNS 边界 ElementRange 转换为 PointRange/FaceCenter，
不修改坐标和体/面连接；输出的 .manifest.json 记录数组校验值。
分析脚本会回读兼容副本并验证这些校验值。

## 运行

从 build/ 运行，例如：

    OMP_NUM_THREADS=1 mpirun -np 2 ./app/ncfv_euler3D.exe ../cases/ncfv_euler/3D/NCFV_iv10.json
    OMP_NUM_THREADS=1 mpirun -np 4 ./app/ncfv_euler3D.exe ../cases/ncfv_euler/3D/NCFV_iv20.json
    OMP_NUM_THREADS=1 mpirun -np 16 ./app/ncfv_euler3D.exe ../cases/ncfv_euler/3D/NCFV_iv80.json

实际三套新增计算并发启动，进程数分别为 2、4、16。
iv40 基准为先前的 4 进程结果。
各自完整输出保存在 data/out/vertexFV/convergence/iv{10,20,80}/。
不要直接重跑覆盖已有结果；需要复算时，先在配置中指定新的输出、
VTK series 和 restart 前缀。

时间误差检查使用 NCFV_iv20_halfdt.json、
NCFV_iv40_halfdt.json，CFL=0.25、步长上限 0.005，其余设置不变。
这些复算不替代基准网格收敛表，只用于检查时间误差敏感性。

## 后处理

从仓库根目录运行：

    MPLCONFIGDIR=/tmp/ncfv-convergence-mpl \
    venv/bin/python cases/ncfv_euler/3D/analyze_vortex_convergence.py
    xelatex -interaction=nonstopmode -halt-on-error \
      -output-directory=docs/reports docs/reports/vertexfv_convergence_report.tex

分析不导入 DNDSR 的 Python 扩展，而直接读取原始节点 owner 的 CSV。
脚本只有在全部四套基准和两套半时间步算例完成 t=2 后才输出最终报告数据；
不会把未完成结果混进收敛表。主要验证：

- 原始节点唯一拥有、输出坐标和输入网格匹配；
- 解析密度由独立公式重算，不只信任求解器的 rho_exact 列；
- 使用恢复节点点值计算体积加权 L1/L2/Linf，与求解器全局汇总交叉核对；
- 各控制体份额体积为正、总和为 400，周期副本保守变量一致；
- 最终密度与由保守均值计算的压力为正，记录质量、动量和总能量漂移；
- 基准算例物理、积分、重构、初场和时间配置相同，Gauss 点存储计数为零。

收敛阶采用 h=(400/N_periodic)^(1/3) 计算，另存名义 h=10/n 的阶数。
实际非结构网格自由度不是严格八倍增加，不把加密比强制写为 2。
metrics.json 同时保留面内尺度、初始化误差和两个步长解之间的范数差。
论文原程序的绝对误差不是验收阈值，目标阶数也不会被写成实测阶数。

输出：

- docs/reports/vertexfv_convergence_report.pdf：中文验证报告；
- docs/reports/vertexfv_convergence/convergence.csv：完整误差与收敛阶表；
- docs/reports/vertexfv_convergence/metrics.json：可复核指标；
- docs/reports/vertexfv_convergence/convergence.png：误差收敛曲线；
- docs/reports/vertexfv_convergence/density_grid_comparison.png：最终密度与误差图。
