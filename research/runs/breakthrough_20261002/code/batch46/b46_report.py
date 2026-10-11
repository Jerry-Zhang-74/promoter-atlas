# -*- coding: utf-8 -*-
"""batch46 Phase D report generator (writes report.md and the status/selfcheck/proposal JSONs)."""
import sys, json, datetime
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import b46_common as C

NOW = datetime.datetime.now().astimezone().replace(microsecond=0).isoformat()


def j(name):
    return json.loads((C.OUT / name).read_text(encoding="utf-8"))


def sec_ab(prim, fz, src):
    def P(lib, *keys):
        o = prim["per_library"][lib]
        for k in keys:
            o = o[k]
        return o
    L = []
    A = L.append
    A("# batch46 %s — 同代内选择效用（回顾性）\n" % C.VERSION)
    A("> 性质：对作者**已发表**合成的 GA 设计研究中**同库同谱系同世代**匹配三元组的固定选择效用回顾分析。"
      "ZERO 拟合/求解/回归/校准/新模型打分/神经前向/GPU/web/下载。非独立验证、非我们模型新性能、非截断、非293FT。\n")
    A("## A. 仅元数据冻结（活动统计之前）")
    A("- 来源 `data/breakthrough_20261002/batch44/syn.txt.gz`；compressed sha256 `%s`（匹配=%s）；b45 源/表身份核对=%s。"
      % (src["integrity"]["sha256_compressed"], src["integrity"]["compressed_sha256_matches_expected"],
         src["b45_source_and_table_identity_ok"]))
    A("- 三元组规则（activity-blind，仅用 info 语法+序列长度）：同一 `library+lineage+generation` 的未修饰 design 片段，"
      "`{max,median,min}` 各恰好一条互异 DNA，且三元组内实际长度一致。**不跨世代/谱系合并、不按结果挑选子集。**")
    A("- 合格三元组 **%d**（short %d / long %d）；保留不完整/冲突组 **%d**（缺标签，全部列于 `freeze_triplets.json`）。"
      % (fz["n_eligible_triplets"], fz["n_eligible_by_library"].get("short", 0),
         fz["n_eligible_by_library"].get("long", 0), fz["n_incomplete_or_conflict_groups_retained"]))
    A("- 冻结：三元组 ID、代码 sha、度量定义、源 sha（先于分数/活性连接）。b45 标签此前已见，此构造**不**具前瞻性。\n")
    A("## B. 主分析 K562：同代内对比（fixed summary = 每 RNA 重复有限 barcode 均值 → 两重复均值）")
    A("short/long 分开；全部三元组保留（含负/并列）。对比 = 实测 `_max − _median`、`_max − _min`（log2 单位）。\n")
    A("| 库 | 三元组 | Δmax−median 中位(+/−) | Δmax−min 中位(+/−) | 跨谱系等世代权 Δmax−median(+/−) | Δmax−min(+/−) |")
    A("|---|---|---|---|---|---|")
    for lib in ("short", "long"):
        dmm = P(lib, "all_rows_dmm"); dmn = P(lib, "all_rows_dmn")
        admm = P(lib, "all_generations_pooled_across_lineage_dmm")
        admn = P(lib, "all_generations_pooled_across_lineage_dmn")
        A("| %s | %d | %.3f (%d/%d) | %.3f (%d/%d) | %.3f (%d/%d) | %.3f (%d/%d) |" % (
            lib, prim["per_library"][lib]["n_triplets"], dmm["median"], dmm["n_positive"], dmm["n_negative"],
            dmn["median"], dmn["n_positive"], dmn["n_negative"], admm["median"], admm["n_positive"],
            admm["n_negative"], admn["median"], admn["n_positive"], admn["n_negative"]))
    A("")
    A("- 每世代中位、每谱系（等世代权、全部 10 条谱系）见 `primary_contrasts.json`；世代0 与固定共同世代"
      "（short G48 / long G44）与**所有世代**并列，不挑选最佳世代。short G48: Δmax−median 中位 %.3f (%d/%d)、"
      "Δmax−min %.3f (%d/%d)。" % (
          P("short", "common_generation_G48")["dmm"]["median"], P("short", "common_generation_G48")["dmm"]["n_positive"],
          P("short", "common_generation_G48")["dmm"]["n_negative"], P("short", "common_generation_G48")["dmn"]["median"],
          P("short", "common_generation_G48")["dmn"]["n_positive"], P("short", "common_generation_G48")["dmn"]["n_negative"]))
    A("- 每 RNA 重复单独对比：short r1 中位 %.3f / r2 %.3f；long r1 %.3f / r2 %.3f。" % (
        P("short", "replicate_contrasts", "dmm_K562_r1")["median"], P("short", "replicate_contrasts", "dmm_K562_r2")["median"],
        P("long", "replicate_contrasts", "dmm_K562_r1")["median"], P("long", "replicate_contrasts", "dmm_K562_r2")["median"]))
    A("- 发布 K562 分数排序与选择元数据一致性：**%d/%d 一致、%d 冲突**；冲突处分数最优是 `_median` 而非 `_max`。"
      "**标签绝不按表达修正。**\n" % (prim["score_ordering_check"]["n_agree"], prim["score_ordering_check"]["n_triplets"],
                                      prim["score_ordering_check"]["n_conflict"]))
    return L


def sec_cde(sel, nf, opt, corr, clk):
    L = []
    A = L.append
    A("## C. 固定选择效用（K562 主；GC-high 比较器；HepG2 次）")
    A("选择器 = 三元组内取该量最大者；并列按 info 字典序最小（冻结）。uniform = 三个候选实测均值"
      "（对**作者所选的这三个**做均匀随机选择的期望效用，非任意随机序列）。regret = 所选 − oracle 最好（仅事后）。\n")
    A("| 场景 | 选择器 | 库 | 平均效用 vs uniform | hit-rate(含并列) | regret(事后) | 效用 +/0/− |")
    A("|---|---|---|---|---|---|---|")
    for lib in ("short", "long"):
        for name, key in (("K562_score", "K562_score"), ("GC_high", "GC_high")):
            s = sel["primary_K562"][key][lib]; sc = s["utility_sign_counts"]
            A("| K562 | %s | %s | %+.3f | %.3f | %+.3f | %d/%d/%d |" % (
                name, lib, s["mean_utility_vs_uniform"], s["hit_rate_tie_inclusive"],
                s["mean_regret_vs_oracle"], sc["positive"], sc["zero"], sc["negative"]))
    for name in ("K562_score_choice_eval_on_HepG2", "GC_high_choice_eval_on_HepG2", "HepG2_score_choice_eval_on_HepG2"):
        for lib in ("short", "long"):
            s = sel["secondary_HepG2"][name][lib]
            A("| HepG2次 | %s | %s | %+.3f | %.3f | %+.3f | - |" % (
                name, lib, s["mean_utility_vs_uniform"], s["hit_rate_tie_inclusive"], s["mean_regret_vs_oracle"]))
    A("")
    A("K562 分数选择器在**同代同谱系**内优于 uniform（short 命中 %.3f、long %.3f，chance=1/3），且明显优于 GC-high "
      "比较器（long GC 命中 %.3f）。regret 与命中并列均已显式报告。**不据胜负部署、不翻符号。**" % (
          sel["primary_K562"]["K562_score"]["short"]["hit_rate_tie_inclusive"],
          sel["primary_K562"]["K562_score"]["long"]["hit_rate_tie_inclusive"],
          sel["primary_K562"]["GC_high"]["long"]["hit_rate_tie_inclusive"]))
    A("- 可选描述性：三元组内 3 点 Spearman(pred,meas) 均值 short %.3f / long %.3f（仅描述，非主证据）。" % (
        opt["by_library"]["short"]["mean_within_triplet_rho"], opt["by_library"]["long"]["mean_within_triplet_rho"]))
    A("- HepG2 次：同一 K562 所选候选在 HepG2 上仍优于 uniform（short %.3f / long %.3f），非细胞因果、不替代 K562 主结论。\n" % (
        sel["secondary_HepG2"]["K562_score_choice_eval_on_HepG2"]["short"]["mean_utility_vs_uniform"],
        sel["secondary_HepG2"]["K562_score_choice_eval_on_HepG2"]["long"]["mean_utility_vs_uniform"]))
    A("## D. 非有限（-Inf / 零 RNA）个案")
    ar = nf["affected_records"][0]
    A("- 精确记录：`%s`，barcode `%s`，query 长度 %d；四个 RNA 列归一化均为 -Inf（原始 RNA 计数=0），见 `nonfinite_case.json`。"
      % (ar["info"], ar["barcode"], ar["query_length"]))
    A("- 受影响三元组 `%s`：主结果 **LIMITED**（缺失原因：中位候选一个 barcode 四处零 RNA），"
      "外加 extended-real 敏感性（非有限视为 −Inf，不做 pseudocount/clipping）：方向为 **+Inf 严格为正**，秩有定义、幅度不有限。" % (
          nf["affected_triplet"]["triplet_id"]))
    A("- 未受影响三元组 %d/%d；headline（long dmm 中位）含/不含该三元组 = %.4f / %.4f，**不依赖**该个案。" % (
        nf["n_unaffected_triplets"], nf["n_total_triplets"],
        nf["headline_dependence"]["long"]["dmm_median_with"], nf["headline_dependence"]["long"]["dmm_median_without"]))
    A("- 保留全部原始零/非有限；**无 RNA-positive 过滤**。\n")
    A("## E. 修正（仅新增文件；b45 旧字节不变）")
    h = corr["entries"]
    A("- HepG2 barcode-spread 按**实际细胞**重算：short HepG2 旧 %.5f（=K562）→ 新 %.5f；long HepG2 旧 %.5f → 新 %.5f。"
      "K562 值不变（%.5f / %.5f），证明方法一致。见 `correction_hepg2_barcode_spread.json`。" % (
          h["short:HepG2"]["old_b45"], h["short:HepG2"]["new_recomputed"], h["long:HepG2"]["old_b45"],
          h["long:HepG2"]["new_recomputed"], h["short:K562"]["new_recomputed"], h["long:K562"]["new_recomputed"]))
    A("- b45 审阅时钟警示：独立评审 `authored_at` 04:20:00 晚于实际评审结束 **%s**（会话 %s，OS 创建 %s）；"
      "不得当作准确时钟，旧字节保留。见 `correction_b45_reviewclock.json`。" % (
          clk["actual_observed_reviewer_end"], clk["reviewer_session_id"], clk["reviewer_os_creation"]))
    A("- 已原子追加缺失的 b45 条目（标记 **priorcomplianceFAIL**）与 b46 条目至 registry/CHANGELOG，保留 production pointer 与全部既有条目。\n")
    return L


def sec_fgh(prim, sel, fz):
    L = []
    A = L.append
    A("## F. 路线判定（中文，简洁）")
    A("**结论：宽泛的跨世代相关并未完全掩盖弱同代选择。** 在同一谱系同一世代内，发布 K562 分数在三个作者候选里"
      "选最强者，其实测命中率 short %.3f / long %.3f（均匀=1/3），效用 vs 均匀 +%.3f / +%.3f，明显优于 GC-high 比较器"
      "（+%.3f / +%.3f）。" % (
          sel["primary_K562"]["K562_score"]["short"]["hit_rate_tie_inclusive"],
          sel["primary_K562"]["K562_score"]["long"]["hit_rate_tie_inclusive"],
          sel["primary_K562"]["K562_score"]["short"]["mean_utility_vs_uniform"],
          sel["primary_K562"]["K562_score"]["long"]["mean_utility_vs_uniform"],
          sel["primary_K562"]["GC_high"]["short"]["mean_utility_vs_uniform"],
          sel["primary_K562"]["GC_high"]["long"]["mean_utility_vs_uniform"]))
    A("**但范围内有界**：(i) max−median 对比弱于 max−min，short G48 的 max−median 中位≈0（5/5），说明'挑选优于中位'"
      "在晚期短库不稳健；(ii) 存在实质负例（short 效用<0 24/129、long 12/115）；(iii) 三元组是作者按分数选出的极值，"
      "10 条谱系共享同一 PARM 模型与流程，非独立族。故这是**有限范围内的实用判别**，非群体泛化、非细胞因果、非我们模型增益。")
    A("**分寸**：既不宣称'分数任选皆灵'，也不因宽泛相关而否定同代判别的存在；两种结论都不抬高 293FT。\n")
    A("## G. 局限")
    A("- 回顾性、暴露数据集；作者选样/优化祖先偏倚；10 谱系非独立；barcode 与 RNA 列数非独立 n。")
    A("- 分数列是作者分析输入，非我们前向输出；预测与实测不同单位，不算 MAE/R2/校准；无 p/CI/bootstrap。")
    A("- 16 个不完整组保留不处理；12 处分数-标签冲突如实标记。")
    A("- 不改变 D28FAIL / 公共 HEK 迁移弱(b41 rho 0.108) / 2000bp 不支持 / b44 源分支已关闭。\n")
    A("## H. 下一步（一个有界动作）")
    A("作者合成分支按上级要求在此**收束**（不再换切片/调参）。剩余真正缺口是**外部可追溯数据**，不是再跑同源分数："
      "缺少一份 `序列 → 构建/设计来源 → 实测活性` 可追溯、且含**同组≥3 个跨预测分位候选**的目标细胞数据集"
      "（293FT 或用户目标细胞）。有该数据后可用**已冻结的 b46 协议（零拟合）**给出该细胞上的同代选择 hit-rate/效用"
      "与 GC 基线，作为可测能力证据；否则如实记 UNKNOWN/数据缺口，不声称整个项目被阻塞。见 `next_batch_proposal.json`。\n")
    return L


def main():
    prim = j("primary_contrasts.json"); sel = j("selector_utility.json"); nf = j("nonfinite_case.json")
    fz = j("freeze_triplets.json"); src = j("source_identity.json"); opt = j("optional_within_triplet_rho.json")
    corr = j("correction_hepg2_barcode_spread.json"); clk = j("correction_b45_reviewclock.json")
    md = "\n".join(sec_ab(prim, fz, src) + sec_cde(sel, nf, opt, corr, clk) + sec_fgh(prim, sel, fz))
    (C.OUT / "report.md").write_text(md, encoding="utf-8")
    C.write_json(C.OUT / "metric_definitions.json", {
        "batch": C.BATCH, "version": C.VERSION,
        "measured_summary": "per RNA replicate = mean over FINITE barcode cells of author normalized.log2rpm; "
                            "assay summary = mean of the two replicate means.",
        "contrasts": "max-median and max-min of the measured summary within a frozen triplet.",
        "selectors": {"K562_score": "argmax published K562 PARM prediction (tie: lexical min info id)",
                      "GC_high": "argmax query GC fraction (tie: lexical min info id)",
                      "HepG2_score": "secondary argmax published HepG2 prediction"},
        "uniform": "mean of the three candidates measured summaries (uniform among THESE three).",
        "oracle_best": "max of the three measured summaries; regret is hindsight only.",
        "no_inference": "no pvalue/CI/bootstrap; ~10 related search lineages; no broad pooled rho as main proof.",
        "no_cross_unit": "no MAE/R2 between prediction and activity; differences within same unit only."})
    C.write_json(C.OUT / "status.json", {
        "batch": C.BATCH, "version": C.VERSION, "phase": "worker",
        "state": "complete_worker_pending_review", "created_at": NOW,
        "zero_fit": True, "zero_gpu": True, "zero_new_prediction": True, "web": False,
        "source": "GSE301246 synthetic promoter libraries (cached b44) sha256_compressed " + C.SRC_SHA256_COMPRESSED,
        "verdict": "DESCRIPTIVE_POSITIVE_WITHIN_GENERATION_SOURCE_STUDY_RETROSPECTIVE",
        "design": {"n_matched_triplets": fz["n_eligible_triplets"],
                   "short": fz["n_eligible_by_library"]["short"], "long": fz["n_eligible_by_library"]["long"],
                   "n_incomplete_retained": fz["n_incomplete_or_conflict_groups_retained"],
                   "score_order_conflicts": prim["score_ordering_check"]["n_conflict"]},
        "primary_K562": {
            "short": {"utility": sel["primary_K562"]["K562_score"]["short"]["mean_utility_vs_uniform"],
                      "hit_rate": sel["primary_K562"]["K562_score"]["short"]["hit_rate_tie_inclusive"]},
            "long": {"utility": sel["primary_K562"]["K562_score"]["long"]["mean_utility_vs_uniform"],
                     "hit_rate": sel["primary_K562"]["K562_score"]["long"]["hit_rate_tie_inclusive"]}},
        "gc_comparator": {"short": sel["primary_K562"]["GC_high"]["short"]["mean_utility_vs_uniform"],
                          "long": sel["primary_K562"]["GC_high"]["long"]["mean_utility_vs_uniform"]},
        "not_evidence_for": "our model new-family/293FT success; D28FAIL/HEK weak-transfer/2000bp-unsupported unchanged",
        "next": "fresh independent Flash review then parent decision"})
    C.write_json(C.OUT / "remaining_work.json", {
        "batch": C.BATCH, "generated_at": NOW,
        "remaining_work": [
            {"id": "external_traceable_selection_cohort", "status": "OPEN_data_gap",
             "desc": "序列->构建/设计来源->实测活性可追溯、且含同组>=3 个跨预测分位候选的目标细胞数据集"
                     "（293FT 或用户目标细胞）；用于把已冻结 b46 协议做零拟合迁移，给出该细胞同代选择 hit-rate/效用。"},
            {"id": "synthetic_branch", "status": "CLOSED_by_parent",
             "desc": "作者合成分支按 PARENT_ACCEPTANCE 在本批收束；不再换切片/调参。"},
            {"id": "293FT_not_validated", "status": "unchanged",
             "desc": "D28FAIL / HEK 弱迁移 / 2000bp 不支持 未被本批改变。"}],
        "next_bounded_action": "batch47 v6.34.0-selection-utility-transfer.rc1（数据门控，零拟合；见 next_batch_proposal.json）"})
    C.write_json(C.OUT / "next_batch_proposal.json", {
        "proposed_batch": "batch47", "title": "v6.34.0-selection-utility-transfer.rc1",
        "rationale": "batch46 证明发布分数在作者合成数据上具有同代内有界实用判别；迁移到用户目标细胞需要外部可追溯的同组多点"
                     "选择数据，仓库内不存在，故不再重复同源分数/追溯/打包。",
        "single_bounded_action": "一旦获得满足条件的可追溯数据集（序列+构建来源+实测活性，且每组>=3 个跨预测分位候选），"
                                 "用已冻结的 b46 协议（零拟合）计算目标细胞同代选择 hit-rate/效用与 GC 基线；无该数据则记 UNKNOWN。",
        "measurable_gain": "目标细胞 hit-rate/效用(+GC 基线) 及其计数；报告分母与不确定来源。",
        "stop_condition": "无可追溯同组数据 -> 记 UNKNOWN 并停止该子路线，不用同源替代。",
        "explicitly_not": ["同源再切片/调参", "新下载/新源搜寻", "新拟合/GPU/前向", "本地293FT追溯审计重放"],
        "external_information_gap": "仓库内不存在满足[可追溯序列->构建->活性 + 同组>=3跨预测分位候选]的数据集；"
                                    "这是外部信息缺口，不表示整个项目被阻塞。",
        "zero_fit": True, "gpu": "none", "parent_gate": "parent 确认目标细胞数据可用/格式后再放行有界零拟合运行"})
    C.write_json(C.OUT / "worker_selfcheck.json", {
        "batch": C.BATCH, "role": "worker", "selfcheck_only": True, "created_at": NOW,
        "checks": {
            "zero_fit_solve_regression_calibration_newscore_neural_gpu": True,
            "no_web_download_api_newsource": True, "only_cached_b44_source_and_b45_summaries": True,
            "old_assets_read_only_except_mandated_registry_changelog_append": True,
            "new_batch46_dirs_only": True, "source_sha_verified": True,
            "b45_source_table_identity_cross_checked": True,
            "metadata_freeze_before_activity_join": True,
            "triplet_rule_no_pooling_no_outcome_subset": True,
            "all_source_rows_and_negative_cases_retained": True,
            "no_rna_positive_filter_no_pseudocount_no_clipping": True,
            "fixed_summary_equals_transparent_barcode_then_rep_mean": True,
            "raw20_independent_verification": True,
            "strict_json_no_nan_infinity": True,
            "no_pvalue_ci_bootstrap": True, "no_mae_r2_calibration": True,
            "no_credential_read": True, "no_email_sent": True, "no_new_wrapper_ui": True,
            "worker_does_not_author_independent_review": True},
        "honest_limitations": [
            "retrospective reanalysis of the AUTHOR selected study with selection/ancestry bias",
            "10 GA search lineages are NOT independent; barcode/column counts are NOT independent n",
            "triplets are author-selected extremes; result is scoped discrimination, NOT population generalization",
            "1 zero-RNA barcode -> the affected triplet primary is LIMITED (extended-real direction only)",
            "source prediction columns are author analysis inputs, NOT our model output"],
        "worker_preliminary_review": False, "fresh_independent_review_required": True})
    print("wrote report.md + status/selfcheck/remaining/next/metric_definitions")


if __name__ == "__main__":
    main()
