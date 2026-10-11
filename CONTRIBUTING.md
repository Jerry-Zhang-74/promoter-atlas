# 参与贡献 · Contributing

[中文](#中文) · [English](#english)

## 中文

欢迎提交问题、复现记录和改进建议。较大的改动可先通过 Issue 讨论，附上使用版本、操作系统、Python/PyTorch 版本和最小复现步骤。

修改推理流程后，运行 `python scripts/verify_release.py`。涉及模型或数据的改动，请同时更新来源、配置、E/S/B 分轴结果和评估条件。
示例检查结果放在软件验证记录中，实验评估放在相应研究报告中。

密钥与私人样本保存在本地，大文件通过 Release 管理。提交说明使用简短、具体的描述。贡献内容采用项目的 MIT 许可证。

---

## English

Questions, reproduction reports and improvements are welcome. For a substantial change, open an issue with the release version, operating system, Python/PyTorch versions and a minimal example.

After changing inference, run `python scripts/verify_release.py`. For model or data changes, update the sources, configuration, per-axis E/S/B results and evaluation conditions.
Record example checks with software verification results and experimental evaluations in the relevant research report.

Keep credentials and private samples locally, and manage large files through Releases. Use short, specific commit messages. Contributions use the project's MIT license.
