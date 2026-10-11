# 发布文件与来源清单 · Release and provenance manifests

## 中文

这些 JSON 清单把发布文件、原始研究记录和当前目录连接起来。下载程序根据其中的文件名、大小与校验值取得准确的资产；版本索引则帮助读者从公开版本号查到原始模型名称、训练源码和评估记录。

### 下载与模型身份

| 清单 | 内容与使用者 |
|---|---|
| [`WEIGHTS_MANIFEST.json`](WEIGHTS_MANIFEST.json) | 当前模型的分片文件、文件 SHA-256、张量名称及内容 SHA-256；由权重下载器和加载器读取 |
| [`TRAINING_ASSETS.json`](TRAINING_ASSETS.json) | 训练归档分组、压缩分片和解包成员的大小及 SHA-256；由数据恢复器读取 |
| [`UPSTREAM_DATA.json`](UPSTREAM_DATA.json) | 来源文件的下载 URL、传输压缩方式、文件大小和内容 SHA-256；由上游下载器读取 |

整文件 SHA-256 标识具体文件；张量状态 SHA-256 标识模型参数内容。下载和核对命令见[脚本说明](../../scripts/README.md)。

### 版本与文件沿革

| 清单 | 可以查询什么 |
|---|---|
| [`MODEL_VERSION_REGISTRY.json`](MODEL_VERSION_REGISTRY.json) | 公开版本与历史模型名称的映射、模型路线及对应材料 |
| [`HISTORICAL_RELEASES.json`](HISTORICAL_RELEASES.json) | 历史 Release 标签、模型记录、归档名称、大小与校验值 |
| [`HISTORICAL_FILE_TIMES.json`](HISTORICAL_FILE_TIMES.json) | 发布整理时采集的源文件修改和创建时间，时区为 Asia/Shanghai |
| [`SOURCE_HISTORY_MANIFEST.json`](SOURCE_HISTORY_MANIFEST.json) | 原始路径 `path`、当前仓库路径 `public_path`、源文件和公开文件指纹，以及文档修订记录 |
| [`PATH_MAP.json`](PATH_MAP.json) | 历史项目相对路径到当前目录的映射，用于数据恢复和历史脚本工作区 |
| [`LAYOUT_MIGRATION.json`](LAYOUT_MIGRATION.json) | 2026-10-11 目录迁移时的旧公开路径、新路径、文件指纹与源文件时间 |
| [`PUBLIC_EXPORT_PROVENANCE.json`](PUBLIC_EXPORT_PROVENANCE.json) | 首次公开导出所依据的竞赛交付快照及文件指纹 |

查找一份旧记录时，先用 `PATH_MAP.json` 定位当前文件，再用对应来源清单核对内容。文件时间描述本地源文件的时间属性；训练起止时间见该轮日志，GitHub 发布日期见 Release。各类时间按各自来源保存。

## English

These JSON manifests connect released assets, original research records and the current layout. Downloaders use filenames, sizes and hashes to retrieve precise assets; version indexes link public version numbers to original model names, training source and evaluation records.

### Downloads and model identity

| Manifest | Contents and consumers |
|---|---|
| [`WEIGHTS_MANIFEST.json`](WEIGHTS_MANIFEST.json) | Current-model shards, file SHA-256 values, tensor names and content hashes; read by the weight downloader and loader |
| [`TRAINING_ASSETS.json`](TRAINING_ASSETS.json) | Training archive groups, compressed segments and extracted-member sizes and hashes; read by the restorer |
| [`UPSTREAM_DATA.json`](UPSTREAM_DATA.json) | Source URLs, transport compression, file sizes and content hashes; read by the upstream downloader |

A file SHA-256 identifies a particular file; the tensor-state SHA-256 identifies the model's parameter content. Download and checking commands are described in the [script guide](../../scripts/README.md#english).

### Versions and file history

| Manifest | What it records |
|---|---|
| [`MODEL_VERSION_REGISTRY.json`](MODEL_VERSION_REGISTRY.json) | Public-to-historical model-name mappings, model routes and associated materials |
| [`HISTORICAL_RELEASES.json`](HISTORICAL_RELEASES.json) | Historical Release tags, model records, archive names, sizes and hashes |
| [`HISTORICAL_FILE_TIMES.json`](HISTORICAL_FILE_TIMES.json) | Source-file modification and creation times captured during publication, in Asia/Shanghai time |
| [`SOURCE_HISTORY_MANIFEST.json`](SOURCE_HISTORY_MANIFEST.json) | Original `path`, current `public_path`, source and public hashes, and documentation revisions |
| [`PATH_MAP.json`](PATH_MAP.json) | Original project-relative paths mapped to the current layout, used for data restoration and historical workspaces |
| [`LAYOUT_MIGRATION.json`](LAYOUT_MIGRATION.json) | Old and new public paths, file hashes and source times from the 2026-10-11 layout migration |
| [`PUBLIC_EXPORT_PROVENANCE.json`](PUBLIC_EXPORT_PROVENANCE.json) | The competition delivery snapshot and file hashes used for the initial public export |

To find an old record, use `PATH_MAP.json` to locate the current file, then consult the relevant source manifest for its identity. File times describe local source-file attributes; training timestamps come from run logs, and GitHub publication dates come from Releases. Each timestamp retains its own source.
