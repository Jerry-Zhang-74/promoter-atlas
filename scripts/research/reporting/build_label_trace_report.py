#!/usr/bin/env python3
"""Build a concise Chinese HTML report from the completed label-trace export.

Reads existing audit results and small source records. Does not re-scan GTEx,
rewrite exported data tables, or alter any training input.
"""
import base64
import csv
import hashlib
import html
import json
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import url2pathname

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports/label_trace_20261001"
WORKBOOK = ROOT / "outputs/01a0f783-d754-7803-abcf-34aa761a6a6b/全库标签与原始数据索引.xlsx"
FIGURE = ROOT / "figures/labels/candidate_e_es_scatter_main_density.png"
GCT = ROOT / "data/raw/GTEx_Analysis_v10_RNASeQCv2.4.2_gene_tpm.gct.gz"
ANNOTATION = ROOT / "data/raw/GTEx_Analysis_v10_Annotations_SampleAttributesDS.txt"
EXPRESSION = ROOT / "labels/build/gene_expression_table.tsv"
LABELS = ROOT / "labels/build/gene_expression_label_table.tsv"
PROMOTERS = ROOT / "labels/build/promoter_label_table.tsv"
ANCHOR_ORDER = ["ACTB", "GAPDH", "PGK1", "UBC", "EEF1A1", "RPL8", "RPLP0", "B2M"]
TABLE_FIELDS = ["gene_symbol", "gene_id", "Score_E", "Score_S", "Score_B", "E_raw", "S_raw", "B_raw"]


def esc(value):
    return html.escape(str(value), quote=True)


def read_rows(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        yield from csv.DictReader(handle, delimiter="\t")


def local_link(path, label):
    return f'<a href="{esc(Path(path).resolve().as_uri())}">{esc(label)}</a>'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def selected_tsv_lines(path, line_numbers):
    wanted = set(line_numbers)
    found = {}
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        header = next(csv.reader([handle.readline()], delimiter="\t"))
        for line_number, text in enumerate(handle, 2):
            if line_number in wanted:
                found[line_number] = dict(zip(header, next(csv.reader([text], delimiter="\t"))))
            if line_number >= max(wanted):
                break
    assert set(found) == wanted
    return found


class AnchorTableReader(HTMLParser):
    def __init__(self):
        super().__init__()
        self.rows = {}
        self.gene = self.field = None
        self.buffer = []
        self.local_links = []
        self.image_sources = []

    def handle_starttag(self, tag, attributes):
        attributes = dict(attributes)
        if tag == "a" and attributes.get("href", "").startswith("file:"):
            self.local_links.append(attributes["href"])
        if tag == "img":
            self.image_sources.append(attributes.get("src", ""))
        if tag == "tr" and "data-gene" in attributes:
            self.gene = attributes["data-gene"]
            self.rows[self.gene] = {}
        if tag == "td" and self.gene is not None:
            self.field = attributes.get("data-field")
            self.buffer = []

    def handle_data(self, text):
        if self.gene is not None and self.field is not None:
            self.buffer.append(text)

    def handle_endtag(self, tag):
        if tag == "td" and self.gene is not None and self.field is not None:
            self.rows[self.gene][self.field] = "".join(self.buffer)
            self.field = None
        if tag == "tr":
            self.gene = None


def build():
    provenance = json.loads((OUT / "provenance.json").read_text(encoding="utf-8"))
    label_stats = provenance["label_verification"]
    source_stats = provenance["gct_verification"]
    promoter_stats = provenance["promoter_mapping"]
    geometry = json.loads((OUT / "plot_geometry.json").read_text(encoding="utf-8"))
    anchors = {row["gene_symbol"]: row for row in read_rows(OUT / "named_anchor_labels.tsv")}
    assert set(anchors) == set(ANCHOR_ORDER)
    actb = anchors["ACTB"]
    gene = actb["gene_id"]
    gct_row = next(row for row in read_rows(OUT / "raw_gct_gene_rows.tsv") if row["gene_id"] == gene)
    tissue_row = next(row for row in read_rows(OUT / "tissue_mean_tpm_used_for_labels.tsv") if row["gene_id"] == gene)
    tissue_name = "Adipose - Subcutaneous"
    tissue = next(row for row in read_rows(OUT / "tissue_metadata.tsv") if row["tissue"] == tissue_name)
    sample = next(row for row in read_rows(OUT / "sample_tissue_mapping.tsv") if row["tissue"] == tissue_name)
    promoter = next(row for row in read_rows(OUT / "promoter_gene_label_mapping.tsv") if row["gene_id"] == gene)
    annotation_columns = ANNOTATION.open(encoding="utf-8").readline().rstrip("\r\n").split("\t")
    annotation_tissue_column = annotation_columns.index("SMTSD") + 1
    expression_columns = EXPRESSION.open(encoding="utf-8").readline().rstrip("\r\n").split("\t")
    expression_mean_column = expression_columns.index("mean_tpm") + 1
    expression_line = int(gct_row["expression_first_line_1based"]) + int(tissue["tissue_column_index_1based"]) - 1
    annotation_line = int(sample["sample_annotation_line_1based"])
    actual_annotation = selected_tsv_lines(ANNOTATION, [annotation_line])[annotation_line]
    assert actual_annotation["SAMPID"] == sample["sample_id"] and actual_annotation["SMTSD"] == tissue_name
    actual_expression = selected_tsv_lines(EXPRESSION, [expression_line])[expression_line]
    assert actual_expression["gene_id"] == gene and actual_expression["tissue"] == tissue_name
    assert float(actual_expression["mean_tpm"]) == float(tissue_row[tissue_name])
    actual_labels = selected_tsv_lines(LABELS, [int(actb["label_source_line_1based"])])[int(actb["label_source_line_1based"])]
    assert actual_labels["gene_id"] == gene
    promoter_line_numbers = [int(value) for value in promoter["source_promoter_lines_1based"].split("|")]
    actual_promoters = selected_tsv_lines(PROMOTERS, promoter_line_numbers)
    for source_row in actual_promoters.values():
        assert source_row["record_id"] == promoter["record_id"] and source_row["gene_id"].split(".")[0] == gene
        for field in ["Score_E", "Score_S", "Score_B", "Score_HKG"]:
            assert source_row[field] == actual_labels[field] == actb[field]
    raw_cells = source_stats["gct_gene_rows"] * source_stats["gct_sample_columns"]
    tissue_cells = source_stats["gct_gene_rows"] * len(provenance["tissues"])
    image_bytes = FIGURE.read_bytes()
    image_hash = hashlib.sha256(image_bytes).hexdigest()
    assert image_hash == provenance["reference_figure_sha256"] == provenance["attachment_sha256"]
    assert label_stats["label_audit_all_pass"]
    assert source_stats["mean_tpm_mismatch_cells"] == source_stats["median_tpm_mismatch_cells"] == 0
    assert geometry["sum_hexagon_counts"] == label_stats["plotted_gene_count"]
    image_uri = "data:image/png;base64," + base64.b64encode(image_bytes).decode("ascii")
    head_names = ["基因", "gene_id", "Score_E", "Score_S", "Score_B", "E_raw", "S_raw", "B_raw"]
    table_head = "".join(f"<th>{esc(name)}</th>" for name in head_names)
    table_rows = "".join(
        f'<tr data-gene="{esc(anchors[name]["gene_id"])}">' +
        "".join(f'<td data-field="{esc(field)}">{esc(anchors[name][field])}</td>' for field in TABLE_FIELDS) + "</tr>"
        for name in ANCHOR_ORDER)
    links = [
        ("全库基因标签", [("gene_labels_gtex.tsv", "58,988 基因完整标签与重算值"),
                           ("gene_labels_all_database.tsv", "79,852 基因并集，含缺失标签原因")]),
        ("启动子映射", [("promoter_gene_label_mapping.tsv", "507,365 启动子、继承分数及两窗口来源行号")]),
        ("每组织 TPM", [("tissue_mean_tpm_used_for_labels.tsv", "用于标签的 58,988 × 54 组织均值"),
                        ("tissue_mean_tpm_before_duplicate_collapse.tsv", "合并重复 gene_id 前的组织均值"),
                        ("tissue_median_tpm_source_not_used_in_labels.tsv", "组织中位数（保留核验，未用于标签）")]),
        ("逐行追溯", [("raw_gct_gene_rows.tsv", "GCT 行号、版本基因 ID、逐行核验"),
                      ("sample_tissue_mapping.tsv", "样本 ID → GCT 列号 → 注释行号 → 组织"),
                      ("duplicate_resolution.tsv", "45 个重复基因的逐组织合并记录")]),
        ("校验与方法", [("source_manifest.tsv", "来源绝对路径与 SHA256"),
                        ("label_recompute_audit.tsv", "17 类标签字段核验"),
                        ("provenance.json", "完整公式、范围与限制"),
                        ("plot_geometry.json", "图范围与 hexbin 参数")]),
    ]
    file_sections = "".join(f'<div class="file-group"><h3>{esc(title)}</h3><ul>' +
                            "".join(f"<li>{local_link(OUT / filename, label)}</li>" for filename, label in items) +
                            "</ul></div>" for title, items in links)
    document = f'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>全库标签与原始数据追溯</title>
<style>
:root{{--ink:#18333f;--muted:#526874;--teal:#147f7a;--line:#dce8ec;--bg:#f3f7f8}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:16px/1.7 "Microsoft YaHei","Noto Sans CJK SC",system-ui,sans-serif}}
main{{max-width:1120px;margin:32px auto;padding:36px 44px;background:#fff;border:1px solid var(--line);border-radius:16px}}
h1{{font-size:30px;line-height:1.35;margin:4px 0 12px}}h2{{font-size:21px;margin:34px 0 12px;border-top:1px solid var(--line);padding-top:23px}}h3{{font-size:16px;margin:0 0 6px}}
p{{margin:8px 0 14px}}a{{color:#176b8e;text-underline-offset:3px}}.eyebrow{{font-size:13px;color:var(--muted);letter-spacing:.08em}}.lead{{color:var(--muted);max-width:900px}}
.actions{{display:flex;flex-wrap:wrap;gap:10px;margin:18px 0 22px}}.button{{display:inline-block;border:1px solid var(--line);padding:8px 14px;border-radius:7px;text-decoration:none;font-weight:600}}.button.primary{{background:var(--teal);color:#fff;border-color:var(--teal)}}
.stats{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}}.stat{{background:#f0f7f7;border:1px solid var(--line);padding:14px;border-radius:9px}}.stat strong{{display:block;font-size:28px;line-height:1.25;font-variant-numeric:tabular-nums}}.stat span{{font-size:13px;color:var(--muted)}}
.note{{font-size:14px;color:var(--muted)}}figure{{margin:25px 0 0}}figure img{{width:100%;height:auto;display:block}}figcaption{{font-size:13px;color:var(--muted);margin:10px 0}}
.table-wrap{{overflow-x:auto}}table{{border-collapse:collapse;width:100%;font-size:13px;font-variant-numeric:tabular-nums}}th,td{{padding:9px 8px;border-bottom:1px solid var(--line);white-space:nowrap;text-align:right}}th{{background:#edf5f6;font-weight:600}}td:first-child,th:first-child,td:nth-child(2),th:nth-child(2){{text-align:left}}tbody tr:nth-child(even){{background:#f8fbfb}}
ul,ol{{padding-left:23px;margin:10px 0}}li{{margin:6px 0}}code{{font-family:Consolas,monospace;font-size:.88em;overflow-wrap:anywhere;background:#eef4f5;padding:1px 4px;border-radius:3px}}.example{{background:#f4f8f9;border-left:4px solid var(--teal);padding:16px 22px}}.example ol{{margin:0}}.files{{display:grid;grid-template-columns:1fr 1fr;gap:20px 30px}}.file-group li{{font-size:14px}}.path{{font-size:12px;color:var(--muted);word-break:break-all}}.check{{padding:14px 18px;background:#eef8f2;border:1px solid #cfe7d8;border-radius:8px}}footer{{font-size:12px;color:var(--muted);border-top:1px solid var(--line);padding-top:16px;margin-top:28px}}
@media(max-width:760px){{main{{margin:0;border:0;border-radius:0;padding:22px 18px}}h1{{font-size:25px}}.stats{{grid-template-columns:repeat(2,1fr)}}.files{{grid-template-columns:1fr}}.stat strong{{font-size:25px}}}}
@media print{{body{{background:#fff}}main{{margin:0;border:0;padding:0;max-width:none}}.actions{{display:none}}h2{{break-after:avoid}}tr,.stat,.file-group{{break-inside:avoid}}figure{{break-inside:avoid}}a{{color:inherit}}}}
</style></head><body><main>
<div class="eyebrow">PROMOTER ATLAS · 数据追溯 · 2026-10-01</div>
<h1>全库标签与原始数据从哪里来</h1>
<p class="lead">范围为附图对应的人类 Candidate E 标签体系。每个分数都保留基因 ID、原始文件位置与计算路径；启动子通过所属基因继承这些标签。</p>
<div class="actions"><a class="button primary" href="{esc(WORKBOOK.resolve().as_uri())}">打开全库 Excel 索引</a><a class="button" href="#files">查看完整数据文件</a><a class="button" href="#actb">看 ACTB 追溯示例</a></div>
<div class="stats"><div class="stat"><strong>{label_stats['gene_label_count']:,}</strong><span>GTEx 标签基因，全量保留</span></div><div class="stat"><strong>{label_stats['all_database_union_gene_count']:,}</strong><span>GTEx 与启动子基因并集</span></div><div class="stat"><strong>{label_stats['plotted_gene_count']:,}</strong><span>进入附图的基因</span></div><div class="stat"><strong>{promoter_stats['unique_promoter_records']:,}</strong><span>唯一启动子记录</span></div></div>
<p class="note">并集中 {label_stats['promoter_only_genes_without_GTEx_label']:,} 个基因有启动子记录但无 GTEx 标签；GTEx 标签库内 {label_stats['nonplotted_GTEx_gene_count']:,} 个基因未进入图。启动子中 {promoter_stats['promoters_with_gene_labels']:,} 条匹配基因标签，{promoter_stats['promoters_without_gene_labels']:,} 条保留缺失状态。</p>
<figure><img src="{image_uri}" alt="原始 E–S 基因密度图，标出 ACTB、GAPDH、PGK1、UBC、EEF1A1、RPL8、RPLP0、B2M"><figcaption>原图直接内嵌；其 SHA256 与用户附件完全相同。每个六边形表示基因数：{geometry['min_count']}–{geometry['max_count']} 个，{geometry['occupied_hexagons']:,} 个非空格合计 {geometry['sum_hexagon_counts']:,} 基因。绘图区自动定界，无额外 E/S 数值截取。</figcaption></figure>

<h2>八个标注基因的真实标签</h2>
<div class="table-wrap"><table id="anchor-table"><thead><tr>{table_head}</tr></thead><tbody>{table_rows}</tbody></table></div>
<p class="note">以上为现存标签表中的六位小数值；原始指标与重新计算的高精度值并列保存在完整 TSV 中。</p>

<h2>公式与图的筛选规则</h2>
<ul><li><strong>E_raw</strong>：54 个组织的 <code>log₂(1 + 组织 mean_TPM)</code> 等权平均。</li>
<li><strong>B_raw</strong>：组织 mean_TPM ≥ 1 的组织数 ÷ 54。</li>
<li><strong>S_raw</strong>：仅在可检出组织中计算上述 log 值的负中位绝对偏差（−MAD，无缩放系数）；需同时满足 <code>B_raw ≥ 0.30</code> 且可检出组织数 ≥ 3。</li>
<li><strong>Score_E / S / B</strong>：全库稳定序数排名 <code>rank / (58,988 − 1)</code>，rank 从 0 起。相同原始值按 gene_id 顺序拆开排名；因此 B_raw 同为 1 的基因仍可能有不同 Score_B。</li>
<li><strong>缺失与入图</strong>：未定义的 S 先按 −∞ 参与全库排序，随后 Score_S 留空。图只纳入 E/S 非空且 <code>S_defined_main=1</code> 的基因；故图中 Score_S 数据范围为 {label_stats['plotted_data_range']['Score_S'][0]:.6f}–1，不能解释为绝对稳定性的百分比。</li></ul>
<p class="note">45 个去版本后重复的基因逐组织合并：同时存在 0 和正数时取正数最大值，否则取算术均值。导出忠实保留历史规则，未改变原始标签。</p>

<h2>全量核验覆盖到哪一层</h2>
<div class="check"><strong>原始数据与标签核验全部通过。</strong><br>
已遍历原 GCT 的 {source_stats['gct_gene_rows']:,} 行基因 × {source_stats['gct_sample_columns']:,} 个样本（{raw_cells:,} 个 TPM 值），按 54 组织独立核对 {tissue_cells:,} 个均值及同量中位数；19,616 样本全部映射到组织，组织样本数完全一致。历史表保留六位小数，比较容差为 5.001 × 10<sup>−7</sup>。全库 17 类指标、分数、计数和排名均通过重新计算。</div>
<p><strong>TPM 宽表是各组织的样本汇总均值。</strong>样本级原始 TPM 仍在本地 GCT；样本列与组织的对应关系另有索引。导出没有把每个启动子的标签称为独立实验测量：GTEx 在这里提供的是基因级表达汇总。</p>

<h2 id="actb">一个可逐步核实的 ACTB 示例</h2>
<div class="example"><ol>
<li><strong>定位基因。</strong><code>{esc(gene)}</code> 对应 GCT ID <code>{esc(gct_row['gtex_gene_id_versioned'])}</code>，见 {local_link(OUT / 'raw_gct_gene_rows.tsv', 'GCT 行索引')}。</li>
<li><strong>定位样本单元格。</strong>GCT 解压文本第 <strong>{int(gct_row['gct_file_line_1based']):,} 行、第 {sample['gct_column_1based']} 列</strong>对应样本 <code>{esc(sample['sample_id'])}</code>。其 {local_link(ANNOTATION, 'SampleAttributes 注释')}在第 {int(sample['sample_annotation_line_1based']):,} 行，<code>SMTSD</code> 为第 {annotation_tissue_column} 列，组织是 <code>{esc(tissue_name)}</code>。这是一个样本坐标示例，组织均值由该组织全部样本计算。</li>
<li><strong>定位组织均值。</strong>该组织共 {tissue['sample_count']} 个样本，ACTB 均值为 <strong>{esc(tissue_row[tissue_name])} TPM</strong>；位于历史 {local_link(EXPRESSION, 'gene_expression_table.tsv')}第 {expression_line:,} 行、第 {expression_mean_column} 列 <code>mean_tpm</code>。其余 53 个组织见 {local_link(OUT / 'tissue_mean_tpm_used_for_labels.tsv', '用于标签的组织 TPM 宽表')}。</li>
<li><strong>从组织值到标签。</strong>ACTB 的 54/54 组织可检出；E_raw={esc(actb['E_raw'])}、S_raw={esc(actb['S_raw'])}、B_raw={esc(actb['B_raw'])}。全库排名后 E/S/B 为 <strong>{esc(actb['Score_E'])} / {esc(actb['Score_S'])} / {esc(actb['Score_B'])}</strong>，见 {local_link(LABELS, '原标签表')}第 {int(actb['label_source_line_1based']):,} 行。</li>
<li><strong>连接启动子。</strong>例如 <code>{esc(promoter['record_id'])}</code>（{esc(promoter['chrom'])}，{esc(promoter['strand'])} 链，TSS {int(promoter['tss_1based']):,}）以同一个 gene_id 继承以上分数。其 up200 / up2000 记录位于 {local_link(PROMOTERS, '原启动子标签表')}第 {esc(promoter['source_promoter_lines_1based'].replace('|', ' / '))} 行；参见 {local_link(OUT / 'promoter_gene_label_mapping.tsv', '启动子映射')}。此处 GENCODE 基因版本为 <code>{esc(promoter['gene_id_versioned'])}</code>，与 GTEx 版本分别保留，关联键使用去版本后的 gene_id。</li>
</ol></div>
<p class="note">所有行号和列号均从 1 起，包含表头；GCT 行号指 gzip 解压后的文本，前 3 行为元信息及列名。样本级单个 TPM 未在本例中另行抄写，可按上述坐标在原文件核实。</p>

<h2 id="files">打开完整数据与来源</h2><div class="files">{file_sections}</div>
<h3>本地原始文件</h3>
<ul><li>{local_link(GCT, 'GTEx v10 基因 × 样本 TPM 原始 GCT（gzip）')}<div class="path">{esc(GCT.resolve())}</div></li>
<li>{local_link(ANNOTATION, 'GTEx v10 SampleAttributes 样本注释')}<div class="path">{esc(ANNOTATION.resolve())}</div></li></ul>
<p class="note">复查顺序：先用 {local_link(OUT / 'source_manifest.tsv', 'source_manifest')} 的绝对路径与 SHA256 确认文件身份，再沿 gene_id → GCT 行号 / 样本列号 → 注释组织 → 表内均值 → 标签 → promoter record_id 追溯。{local_link(OUT / 'provenance.json', 'provenance.json')}保留现存官方数据链接与完整方法。</p>

<h2>已知边界</h2>
<p>历史 GTEx 聚合脚本、当时的软件版本及确切下载时间未完整保留。本次已用现存 GCT 和样本注释对全部组织汇总值完成独立核对；文件修改时间不充当下载时间。基因表达标签关联到启动子后，仍不能当作每个启动子或 TSS 的独立活性测量。</p>
<footer>仅使用已完成的本地导出生成此报告；本次报告构建没有重新扫描 2.26 GB GCT，也没有改写数据表或训练输入。原图已内嵌；其他链接指向本机绝对路径。完整校验记录见 {local_link(OUT / 'report_delivery_manifest.json', '报告交付清单')}。</footer>
</main></body></html>'''
    report = OUT / "report.html"
    report.write_text(document, encoding="utf-8")
    parsed = AnchorTableReader()
    parsed.feed(report.read_text(encoding="utf-8"))
    assert len(parsed.rows) == 8
    master_anchors = {row["gene_id"]: row for row in read_rows(OUT / "gene_labels_gtex.tsv") if row["is_named_anchor"] == "1"}
    for anchor in anchors.values():
        for field in TABLE_FIELDS:
            assert parsed.rows[anchor["gene_id"]][field] == master_anchors[anchor["gene_id"]][field], (anchor["gene_id"], field)
    for field in ["Score_E", "Score_S", "Score_B", "Score_HKG"]:
        assert promoter[field] == actb[field], field
    for uri in parsed.local_links:
        linked_path = Path(url2pathname(urlparse(uri).path))
        assert linked_path.exists() or linked_path == OUT / "report_delivery_manifest.json", linked_path
    assert len(parsed.image_sources) == 1
    embedded_bytes = base64.b64decode(parsed.image_sources[0].split(",", 1)[1])
    assert hashlib.sha256(embedded_bytes).hexdigest() == image_hash
    manifest = {
        "report_file": str(report.resolve()), "report_sha256": digest(report), "report_bytes": report.stat().st_size,
        "builder_file": str(Path(__file__).resolve()), "builder_sha256": digest(Path(__file__)),
        "embedded_figure_sha256": image_hash, "figure_matches_user_attachment": True,
        "workbook_link": str(WORKBOOK.resolve()), "workbook_exists_at_report_build": WORKBOOK.exists(),
        "checks": {"html_anchor_rows": 8, "fields_compared_per_anchor": len(TABLE_FIELDS),
                   "all_html_anchor_fields_match_full_gene_table": True, "ACTB_promoter_scores_match_gene": True,
                   "absolute_local_file_links_checked": len(parsed.local_links),
                   "embedded_image_decoded_bytes_match_source": True,
                   "ACTB_annotation_expression_label_and_promoter_line_pointers_checked_in_source_files": True,
                   "existing_source_audits_all_pass": True, "original_GCT_rescanned_for_report": False,
                   "source_or_exported_data_tables_modified": False},
        "ACTB_example": {"gene_id": gene, "gct_line_1based": int(gct_row["gct_file_line_1based"]),
                         "gct_sample_column_1based": int(sample["gct_column_1based"]), "sample_id": sample["sample_id"],
                         "annotation_line_1based": int(sample["sample_annotation_line_1based"]),
                         "annotation_tissue_column_1based": annotation_tissue_column,
                         "tissue": tissue_name, "sample_count": int(tissue["sample_count"]),
                         "tissue_mean_tpm": tissue_row[tissue_name], "expression_line_1based": expression_line,
                         "expression_mean_column_1based": expression_mean_column,
                         "label_line_1based": int(actb["label_source_line_1based"]),
                         "promoter_record_id": promoter["record_id"],
                         "promoter_source_lines_1based": promoter["source_promoter_lines_1based"]},
    }
    (OUT / "report_delivery_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"report": str(report), "html_anchor_fields_checked": 8 * len(TABLE_FIELDS),
                      "image_exact_match": True, "workbook_exists": WORKBOOK.exists(), "bytes": report.stat().st_size}, ensure_ascii=False))


if __name__ == "__main__":
    build()
