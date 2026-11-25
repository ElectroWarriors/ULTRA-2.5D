# README
## v1 - 2025.08.23
* 线长（平移、旋转），重叠，边界约束，跨光罩面积约束基本完成
* 优化参数接口（自动调整策略）初步完成
* 测例`case`效果良好，但`chiplet 1`的旋转方向有不合理之处：
<img src="../../celldata/case_8c/placer/log_20250823-121745-draw/QQ2025823-142133-HD.gif" alt="QQ2025823-142133-HD" style="zoom:50%;" />

## v2 - 2025.08.30
* 更新线长模型，新增加flip功能，通过`strategy.json`控制
* 更新`strategy.json`脚本及其C++接口，更清楚的含义

## v2.1 - 2025.09.04
* 新增legalization功能（可选：仅水平/仅垂直/水平垂直的约束），并更新回db
* 新增添加cross reticle chiplets的功能，以减小cross reticle对critical nets的signal integrity的影响


## v2.2 - 2025.09.14
* 修改cross reticle chiplets为EMIB

## v2.3 -> v3 - 2025.09.18
* 新增routability估计的功能
* 新增根据routability进行legalize以解决拥塞的功能

