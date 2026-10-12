# 项目入口

当前版本 v6.6.0。演示入口为 `run_demo.py`，单任务入口为 `predict.py`。
当前源码位于 `src/promoter_atlas/`，实际训练配置为 `configs/v6.6.json`。
本轮冻结源码和日志位于 `training/v6.6.0/`，评估记录位于 `evaluation/v6.6.0/`，数据定义与来源见 `data/DATA_SOURCE_MANIFEST.json`。

修改时保留本轮冻结源码、配置和校验清单。新的训练使用独立配置与输出目录，记录实际数据和模型哈希；指标取自对应评估文件。历史版入口见 `models/versions/`，当前演示只加载 v6.6.0。
