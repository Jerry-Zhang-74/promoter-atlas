# 输入示例

`esb_demo.tsv`保存8条人类启动子的400 bp局部窗口、2000 bp上游窗口和源记录编号。`fragment_demo.tsv`保存HEK293公开数据训练集的3条片段，长度和原记录见`fragment_demo_provenance.tsv`。

双窗口输入用于E/S/B与54组织表达预测；单片段输入用于HEK293活性预测。组织输出的数值是预测log2(1+mean_tpm)，原始表达表保存在data/processed/labels/build/。演示结果由随包模型计算。
