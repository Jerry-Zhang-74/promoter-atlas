"""Read local cell-line measurements without changing or re-gating the source data.

CSV values are retained verbatim, including ####, zeros and percentages. FCS
files contribute acquisition metadata only; events are not biological replicates.
Project annotations are explicitly distinguished from laboratory sample metadata.
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import io
import json
import math
import re
import statistics
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = Path('E:/国创赛-启阶')
DEFAULT_OUT = ROOT / 'reports/cellline_v1_20261001'


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def file_info(path):
    return {'path': path.as_posix(), 'bytes': path.stat().st_size, 'sha256': sha256(path)}


def region_sha256(path, start, count):
    h = hashlib.sha256()
    with path.open('rb') as handle:
        handle.seek(start)
        while count:
            block = handle.read(min(count, 1024 * 1024))
            if not block:
                raise ValueError(f'Truncated data region: {path}')
            h.update(block)
            count -= len(block)
    return h.hexdigest()


def line_evidence(path, needle):
    lines = path.read_text(encoding='utf-8-sig').splitlines()
    hits = [{'line_1based': i, 'text': line} for i, line in enumerate(lines, 1) if needle in line]
    if not hits:
        raise ValueError(f'Missing annotation evidence: {path}: {needle}')
    return {'source_path': path.as_posix(), 'matches': hits}


def literal_assignment(path, name):
    tree = ast.parse(path.read_text(encoding='utf-8-sig'))
    node = next(n for n in tree.body if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == name for t in n.targets))
    return ast.literal_eval(node.value), {'source_path': path.as_posix(),
        'line_start_1based': node.lineno, 'line_end_1based': node.end_lineno, 'assignment': name}


def csv_records(path):
    raw = path.read_bytes()
    encoding = 'utf-8-sig' if raw.startswith(b'\xef\xbb\xbf') else 'utf-8'
    text = raw.decode(encoding, errors='strict')
    assert text.encode(encoding) == raw
    reader = csv.reader(io.StringIO(text, newline=''))
    records, previous = [], 0
    for row in reader:
        records.append({'record': len(records) + 1, 'line_start': previous + 1,
                        'line_end': reader.line_num, 'values': row})
        previous = reader.line_num
    header_i = next(i for i, r in enumerate(records)
                    if r['values'] and r['values'][0].strip() == '试管名称：')
    return encoding, records, header_i


def time_metadata(path, experiment):
    refs = []
    result = {'experiment_name': experiment, 'time_flags': []}
    clone = re.search(r'ROSA26\s*-\s*(\d+)', experiment, re.I)
    result['clone_reported'] = 'ROSA26-' + clone.group(1) if clone else ''
    date = re.search(r'(20\d{6})', experiment)
    result['acquisition_date_reported'] = datetime.strptime(date.group(1), '%Y%m%d').date().isoformat() if date else ''
    for key, pattern in [('days_after_transfection_reported', r'转染后\s*DAY\s*(\d+)'),
                         ('days_after_sort_reported', r'分选后\s*DAY\s*(\d+)')]:
        header_match = re.search(pattern, experiment, re.I)
        name_match = re.search(pattern, path.name, re.I)
        chosen = header_match or name_match
        result[key] = int(chosen.group(1)) if chosen else ''
        refs.append({'field': key, 'value': result[key], 'evidence':
                     'CSV record 1 column 2' if header_match else ('filename' if name_match else 'unavailable')})
        if header_match and name_match and header_match.group(1) != name_match.group(1):
            result['time_flags'].append(key + '_header_filename_conflict')
    result['sort_round_reported'] = 2 if '第二次分选' in experiment else ''
    remaining = re.sub(r'(转染后|分选后)\s*DAY\s*\d+', '', experiment, flags=re.I)
    result['unqualified_day_in_experiment'] = ';'.join(re.findall(r'DAY\s*(\d+)', remaining, re.I))
    if result['unqualified_day_in_experiment']:
        result['time_flags'].append('unqualified_day_in_experiment_do_not_assume_transfection_day')
    if '分选 DAY13' in experiment:
        result['time_flags'].append('header_sort_DAY13_vs_filename_after_sort_DAY5_requires_confirmation')
    if result['sort_round_reported'] == 2:
        result['time_flags'].append('second_sort_population_not_same_sort_reference_as_D5_D7')
    result['time_evidence'] = refs
    return result


def parse_value(raw):
    if raw.strip() == '####':
        return '', 'unavailable_marker', ''
    if not raw.strip():
        return '', 'blank', ''
    token = raw.strip()
    percent = token.endswith('%')
    numeric = token[:-1] if percent else token
    if not re.fullmatch(r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?', numeric):
        return '', 'non_numeric', ''
    return numeric, 'observed', 'percent' if percent else ''


def statistic(header):
    h = header.strip()
    if h == '体积(微升)：':
        return 'sample_volume', '', '', 'uL'
    if ' Median ' in h:
        gate, channel = h.split(' Median ', 1)
        return 'median', gate.strip(), channel, 'instrument_units'
    for suffix, stat, unit in [(' %父群', 'percent_parent', 'percent'), (' %总数', 'percent_total', 'percent'),
                               (' 颗粒数/μL(V)', 'event_density', 'events_per_uL'), (' 颗粒数', 'event_count', 'events')]:
        if h.endswith(suffix):
            return stat, h[:-len(suffix)].strip(), '', unit
    if h == 'All Events':
        return 'event_count', 'All Events', '', 'events'
    return 'unclassified_exported_statistic', '', '', ''


def fcs_metadata(path):
    with path.open('rb') as handle:
        header = handle.read(58)
        if not header.startswith(b'FCS'):
            raise ValueError(f'Invalid FCS header: {path}')
        start, end = int(header[10:18]), int(header[18:26])
        handle.seek(start)
        raw = handle.read(end - start + 1)
    delimiter = chr(raw[0])
    text = raw[1:].decode('utf-8', errors='strict').rstrip('\x00 ')
    parts, buf, i = [], [], 0
    while i < len(text):
        if text[i] == delimiter:
            if i + 1 < len(text) and text[i + 1] == delimiter:
                buf.append(delimiter)
                i += 2
                continue
            parts.append(''.join(buf))
            buf = []
        else:
            buf.append(text[i])
        i += 1
    if buf:
        parts.append(''.join(buf))
    if len(parts) % 2:
        raise ValueError(f'Odd number of FCS TEXT tokens: {path}')
    pairs = list(zip(parts[::2], parts[1::2]))
    meta = dict(pairs)
    par, total = int(meta['$PAR']), int(meta['$TOT'])
    data_start, data_end = int(meta['$BEGINDATA']), int(meta['$ENDDATA'])
    params = [{key: meta.get(f'$P{n}{key}', '') for key in ['N', 'S', 'B', 'R', 'E', 'G', 'V']}
              | {'index_1based': n} for n in range(1, par + 1)]
    result = {'fcs_version': header[:6].decode('ascii'), 'text_encoding': 'utf-8 strict',
              'text_start_byte_0based': start, 'text_end_byte_0based_inclusive': end,
              'data_start_byte_0based': data_start, 'data_end_byte_0based_inclusive': data_end,
              'metadata': meta, 'metadata_pairs': pairs, 'parameters': params,
              'events': total, 'event_data_loaded': False,
              'event_data_sha256': region_sha256(path, data_start, data_end - data_start + 1)}
    if meta['$DATATYPE'] == 'F':
        result['event_data_length_check'] = (data_end - data_start + 1 == total * par * 4)
        assert result['event_data_length_check'], path
    return result


def normalized_fcs_date(raw):
    match = re.fullmatch(r'(\d{1,2})-(\d{1,2})月-(\d{4})', raw)
    if match:
        return datetime(int(match[3]), int(match[2]), int(match[1])).date().isoformat()
    try:
        return datetime.strptime(raw, '%d-%b-%Y').date().isoformat()
    except ValueError:
        return ''


def snapgene_metadata(path):
    raw = path.read_bytes()
    pos, packets, features, notes = 0, [], [], {}
    result = {}
    while pos < len(raw):
        tag, size = raw[pos], int.from_bytes(raw[pos+1:pos+5], 'big')
        end = pos + 5 + size
        if end > len(raw):
            raise ValueError(f'Truncated SnapGene packet: {path}: {pos}')
        payload = raw[pos+5:end]
        packets.append({'tag': tag, 'header_byte_0based': pos, 'payload_bytes': size})
        if tag == 0:
            seq = payload[1:]
            result.update(sequence_length=len(seq), sequence_sha256=hashlib.sha256(seq).hexdigest(),
                          sequence_packet_byte_0based=pos, sequence_flags_byte=payload[0])
        elif tag == 10:
            xml = ET.fromstring(payload.decode('utf-8'))
            features = [{'attributes': f.attrib, 'segments': [s.attrib for s in f.findall('Segment')],
                         'xml_path': f'Features/Feature[{n}]', 'packet_header_byte_0based': pos}
                        for n, f in enumerate(xml.findall('Feature'), 1)]
        elif tag == 6:
            xml = ET.fromstring(payload.decode('utf-8'))
            notes = {element.tag: {'text': element.text, 'attributes': element.attrib}
                     for element in xml}
        pos = end
    result.update(packets=packets, features=features, notes=notes,
                  candidate_sequence_assignment='unavailable: no ACTB/GAPDH candidate feature labels; do not assign the vector to measured promoter samples')
    return result


def write_tsv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter='\t')
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def diagnostic_tables(observations, sample_records, out):
    """Create traceable ratios, collapsing tube rows before any diagnostic fit."""
    cells = defaultdict(dict)
    for observation in observations:
        if observation['source_kind'] == 'csv':
            cells[(observation['source_id'], observation['source_record_1based'])][observation['source_column_name'].strip()] = observation

    def get(sample, column):
        return cells[(sample['source_id'], sample['source_record_1based'])][column]

    by_source = defaultdict(list)
    for sample in sample_records:
        if sample['clone_reported'] == 'ROSA26-4':
            by_source[sample['source_id']].append(sample)
    ready, summaries, denominator_records = [], [], []
    for sid, samples in by_source.items():
        wt = [sample for sample in samples if sample['sample_annotation'] == 'WT']
        denominator_cells = [get(sample, 'V1L Median B525-A') for sample in wt]
        assert len(wt) == 3 and all(cell['value_status'] == 'observed' for cell in denominator_cells)
        denominator = statistics.mean(float(cell['numeric_value']) for cell in denominator_cells)
        assert denominator > 0
        denominator_ids = ';'.join(cell['observation_id'] for cell in denominator_cells)
        denominator_records.append({'source_id': sid, 'source_path': samples[0]['source_path'],
            'wt_tubes': [sample['tube_name'] for sample in wt],
            'wt_negative_gate_median_b525_values': [cell['raw_value'] for cell in denominator_cells],
            'wt_negative_gate_mean_b525': denominator, 'raw_observation_ids': denominator_ids,
            'rule': 'arithmetic mean of same CSV WT V1L Median B525-A values',
            'definition_evidence_id': 'fluorescence_definition'})
        for sample in samples:
            if not sample['sample_annotation'] or sample['sample_annotation'] == 'WT' or sample['events'] == 0:
                continue
            numerator = get(sample, 'V1R Median B525-A')
            positive = get(sample, 'V1R %父群')
            assert numerator['value_status'] == positive['value_status'] == 'observed'
            value = float(numerator['numeric_value'])
            relative = value / denominator
            vleft, vright = [float(get(sample, label + ' 颗粒数')['numeric_value']) for label in ['V1L', 'V1R']]
            parent_candidates = []
            for gate in ['All Events', 'P1', 'P2']:
                count = float(get(sample, gate if gate == 'All Events' else gate + ' 颗粒数')['numeric_value'])
                if count > 0 and count == vleft + vright and abs(100 * vright / count - float(positive['numeric_value'])) <= .00501:
                    parent_candidates.append(gate)
            row = {key: sample.get(key, '') for key in ['source_id', 'source_path', 'source_record_1based',
                'tube_name', 'sample_annotation', 'promoter_family_reported', 'upstream_bp_reported',
                'mapping_status', 'mapping_evidence_ids', 'cell_line_reported', 'clone_reported',
                'acquisition_date_reported', 'days_after_transfection_reported', 'days_after_sort_reported',
                'sort_round_reported', 'technical_repeat_index', 'repeat_type']}
            row.update(numerator_gate='V1R', numerator_channel='B525-A', numerator_statistic='Median',
                numerator_raw=value, numerator_observation_id=numerator['observation_id'],
                wt_negative_mean=denominator, denominator_observation_ids=denominator_ids,
                wt_denominator_source_records_1based=';'.join(str(w['source_record_1based']) for w in wt),
                wt_denominator_tube_names=';'.join(w['tube_name'] for w in wt),
                events_count=sample['events'], v1l_events_count=int(vleft), v1r_events_count=int(vright),
                relative_gfp=relative, log10_relative_gfp=math.log10(relative) if relative > 0 else '',
                gfp_positive_percent_parent=float(positive['numeric_value']),
                positive_rate_observation_id=positive['observation_id'],
                parent_gate_inferred_from_counts=';'.join(parent_candidates),
                parent_gate_status='arithmetic_consistency_only_not_original_gate_geometry',
                independent_biological_replicates='', candidate_sequence_id='',
                use_scope='diagnostic_only;link_sequence_and_confirm_batch_before_supervised_fit',
                quality_flags=sample['quality_flags'] + ';user_reports_no_additional_independent_repeats')
            ready.append(row)

    grouped = defaultdict(list)
    for row in ready:
        grouped[(row['source_id'], row['sample_annotation'])].append(row)
    for (sid, sample), rows in grouped.items():
        row = {key: rows[0][key] for key in ['source_id', 'source_path', 'sample_annotation',
            'promoter_family_reported', 'upstream_bp_reported', 'mapping_status', 'mapping_evidence_ids',
            'cell_line_reported', 'clone_reported', 'acquisition_date_reported',
            'days_after_transfection_reported', 'days_after_sort_reported', 'sort_round_reported',
            'wt_negative_mean', 'denominator_observation_ids', 'wt_denominator_source_records_1based',
            'wt_denominator_tube_names', 'use_scope']}
        ratios = [r['relative_gfp'] for r in rows]
        row.update(n_exported_tube_rows=len(rows), independent_biological_replicates='',
            numerator_observation_ids=';'.join(r['numerator_observation_id'] for r in rows),
            source_records_1based=';'.join(str(r['source_record_1based']) for r in rows),
            source_tube_names=';'.join(r['tube_name'] for r in rows),
            events_count_by_tube=';'.join(str(r['events_count']) for r in rows),
            v1l_events_count_by_tube=';'.join(str(r['v1l_events_count']) for r in rows),
            v1r_events_count_by_tube=';'.join(str(r['v1r_events_count']) for r in rows),
            relative_gfp=statistics.mean(ratios), log10_relative_gfp=math.log10(statistics.mean(ratios)),
            within_export_dispersion_sd=statistics.stdev(ratios) if len(ratios) > 1 else '',
            gfp_positive_percent_parent=statistics.mean(r['gfp_positive_percent_parent'] for r in rows),
            parent_gate_inferred_from_counts=';'.join(sorted({r['parent_gate_inferred_from_counts'] for r in rows})),
            candidate_sequence_id='',
            repeat_interpretation='exported tube rows collapsed; documentation calls them technical repeats; user reports no additional independent repeats',
            quality_flags=';'.join(sorted({r['quality_flags'] for r in rows})))
        summaries.append(row)
    write_tsv(out / 'measurements_ready.tsv', ready)
    write_tsv(out / 'diagnostic_candidate_summary.tsv', summaries)
    return {'definition': 'candidate V1R Median B525-A divided by arithmetic mean of WT V1L Median B525-A in the same CSV',
        'r4_only': True, 'r10_excluded': 'uncertain Tube identities, G5/G20 row-order conflict, cell-line/time ambiguity',
        'tube_rows': len(ready), 'candidate_timepoint_rows': len(summaries),
        'denominators': denominator_records,
        'parent_gate_inferred_from_counts': dict(Counter(r['parent_gate_inferred_from_counts'] for r in summaries)),
        'scope': 'diagnostic measurements; no biological n assigned; no cross-time independence or expression stability claim'}


def sequence_endpoint_tables(out):
    constructs_path = out / 'construct_sequences.tsv'
    if not constructs_path.exists():
        return {'status': 'awaiting_separate_construct_sequence_extraction'}
    with constructs_path.open(encoding='utf-8-sig', newline='') as handle:
        constructs = {row['construct_id']: row for row in csv.DictReader(handle, delimiter='\t')}
    with (out / 'diagnostic_candidate_summary.tsv').open(encoding='utf-8-sig', newline='') as handle:
        summary = list(csv.DictReader(handle, delimiter='\t'))
    endpoint, time_diagnostics, excluded = [], [], []
    for row in summary:
        construct_id = {'CAV': 'CMVfull', 'CMV': 'CMVcore'}.get(row['sample_annotation'], row['sample_annotation'])
        construct = constructs.get(construct_id)
        if not construct or construct['eligible_exploratory_sequence_fit'] != '1':
            excluded.append({'source_id': row['source_id'], 'sample_annotation': row['sample_annotation'],
                'construct_id': construct_id, 'reason': construct['sequence_status'] if construct else 'sequence_unavailable'})
            continue
        sequence = construct['sequence']
        assert hashlib.sha256(sequence.encode('ascii')).hexdigest() == construct['sequence_sha256']
        linked = row | {'construct_id': construct_id, 'family': construct['family'],
            'candidate_sequence_id': construct_id,
            'sequence_length_bp': construct['sequence_length_bp'], 'sequence_sha256': construct['sequence_sha256'],
            'sequence': sequence, 'sequence_source_path': construct['source_path'],
            'sequence_source_sha256': construct['source_sha256'],
            'sequence_feature_ranges_1based_inclusive': construct['feature_ranges_1based_inclusive'],
            'sequence_mapping_evidence': construct['mapping_evidence'],
            'assay_to_plasmid_barcode_verified': construct['assay_to_plasmid_barcode_verified'],
            'control_alias_evidence_id': 'control_alias' if construct_id.startswith('CMV') else '',
            'target_name': 'log10_GFP_positive_median_divided_by_same_batch_WT_negative_median',
            'target_value': row['log10_relative_gfp'], 'split_group_construct': construct_id,
            'split_group_family': construct['family'],
            'use_scope': 'exploratory_construct_diagnostic;assay_to_plasmid_identity_not_barcode_verified;no_independent_biological_n'}
        if row['days_after_sort_reported'] == '28':
            assert row['parent_gate_inferred_from_counts'] == 'P2'
            linked['dataset_role'] = 'D28_second_sort_primary_diagnostic_endpoint'
            linked['endpoint_eligible'] = 1
            endpoint.append(linked)
        else:
            assert row['parent_gate_inferred_from_counts'] == 'All Events'
            linked['dataset_role'] = 'time_diagnostic_only_different_parent_gate_and_population'
            linked['endpoint_eligible'] = 0
            time_diagnostics.append(linked)
    assert len({row['construct_id'] for row in endpoint}) == len(endpoint)
    write_tsv(out / 'construct_endpoints_D28.tsv', endpoint)
    write_tsv(out / 'construct_time_diagnostics.tsv', time_diagnostics)
    return {'status': 'linked_for_exploratory_diagnostic_only', 'construct_source': file_info(constructs_path),
        'D28_construct_endpoint_rows': len(endpoint), 'other_time_diagnostic_rows': len(time_diagnostics),
        'D28_unique_constructs': [row['construct_id'] for row in endpoint], 'excluded': excluded,
        'control_alias_evidence_id': 'control_alias',
        'replicate_rule': 'one endpoint per construct, no biological replicate count inferred from tubes or cell events',
        'target_scope': 'positive-gate signal in a second-sort population; not an unbiased all-cell activity or independent validation score'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source-project', type=Path, default=DEFAULT_SOURCE)
    ap.add_argument('--out', type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    project, out = args.source_project.resolve(), args.out.resolve()
    if out == project or project in out.parents:
        raise ValueError('Extraction outputs must be outside the source project')
    data = project / 'data'
    workflow = project / 'PROJECT_WORKFLOW.md'
    analysis = project / 'work/flow_analysis'
    evidence = {
        'cell_line_r4': line_evidence(workflow, '细胞系统一写'),
        'technical_repeats': line_evidence(workflow, 'n = 3 是技术重复'),
        'fluorescence_definition': line_evidence(workflow, '主指标 = 样本 GFP+ 门'),
        'obsolete_csv_extraction': line_evidence(workflow, '原来取错了通道和行'),
        'series_annotation': line_evidence(project / 'pictures/图片索引.md', 'ACTB 系列（A1'),
        'candidate_length_annotation': line_evidence(project / 'work/plan_extract.txt', 'A1 为上游 100 bp'),
        'control_alias': line_evidence(analysis / 'mkfig14_strength.py', "LABEL = {'CAV'"),
    }
    tubes, evidence['r4_tubes_extended_legacy_script'] = literal_assignment(analysis / 'mkfig15_v4_snr.py', 'TUBES')
    current_tubes, evidence['r4_tubes_current_script'] = literal_assignment(analysis / 'mkfig15_v7_relative.py', 'TUBES')
    file_groups, evidence['r4_files_and_wt_current_script'] = literal_assignment(analysis / 'mkfig15_v7_relative.py', 'FILES')
    assert all(tubes[key] == val for key, val in current_tubes.items())
    r4_files = {name: {'day': day, 'wt': wt} for day, name, wt in file_groups}
    package_path = analysis / 'data_package.json'
    package = json.loads(package_path.read_text(encoding='utf-8'))
    sample_definitions = {}
    for n, entry in enumerate(package['sample_map']):
        key = entry['sample'].upper()
        sample_definitions[key] = entry
        evidence['sample_' + key] = {'source_path': package_path.as_posix(),
            'json_path': f'$.sample_map[{n}]', 'reported_annotation': entry,
            'status': 'project_annotation_not_sequence_verified; obsolete timecourse numeric fields in this source were not used'}

    def annotate(raw_name, csv_name=''):
        canonical, group_index, repeat_type, refs = '', '', '', []
        mapping_status = 'unmapped'
        if csv_name in r4_files:
            if raw_name in r4_files[csv_name]['wt']:
                canonical = 'WT'
                group_index = r4_files[csv_name]['wt'].index(raw_name) + 1
                refs.append('r4_files_and_wt_current_script')
                mapping_status = 'current_project_analysis_annotation'
            else:
                for sample, members in tubes.items():
                    if raw_name in members:
                        canonical, group_index = sample, members.index(raw_name) + 1
                        current = sample in current_tubes
                        refs.append('r4_tubes_current_script' if current else 'r4_tubes_extended_legacy_script')
                        mapping_status = 'current_project_analysis_annotation' if current else 'legacy_project_analysis_annotation_requires_confirmation'
                        break
            if canonical:
                repeat_type = 'technical_same_tube_reported'
                refs.append('technical_repeats')
        if not canonical:
            # This alias rule is documented in prepare_package_json.py, not a
            # de novo interpretation of the A/G filename prefixes.
            clean = re.sub(r'\s*-\s*\d+$', '', raw_name).strip().upper()
            if clean in sample_definitions:
                canonical = clean
                mapping_status = 'project_sample_annotation_only'
                refs.append('sample_alias_rule')
        definition = sample_definitions.get(canonical, {})
        if definition:
            refs.append('sample_' + canonical)
        return {'sample_annotation': canonical, 'mapping_status': mapping_status,
                'promoter_family_reported': definition.get('series', ''),
                'upstream_bp_reported': definition.get('tss_upstream_bp') or '',
                'technical_repeat_index': group_index, 'repeat_type': repeat_type,
                'biological_replicate_id': '', 'candidate_sequence_id': '',
                'mapping_evidence_ids': ';'.join(refs)}

    evidence['sample_alias_rule'] = line_evidence(analysis / 'prepare_package_json.py', 'clean = re.sub')
    sources, observations, sample_records = [], [], []
    source_counter = 0
    for path in sorted(data.glob('*.csv'), key=lambda p: p.name.lower()):
        source_counter += 1
        sid = f'S{source_counter:03d}'
        enc, records, hi = csv_records(path)
        header = records[hi]['values']
        experiment = records[0]['values'][1]
        timing = time_metadata(path, experiment)
        info = file_info(path) | {'source_id': sid, 'kind': 'csv', 'encoding': enc,
            'encoding_roundtrip_exact': True, 'header_record_1based': records[hi]['record'],
            'header_line_1based': records[hi]['line_start'],
            'columns': [{'column_1based': i, 'name': value} for i, value in enumerate(header, 1)],
            'sample_rows': len(records) - hi - 1, 'time_metadata': timing}
        sources.append(info)
        for r in records[hi+1:]:
            values = r['values']
            if not values:
                continue
            assert len(values) == len(header), (path, r['record'])
            ann = annotate(values[0], path.name)
            flags = list(timing['time_flags'])
            if values[3] == '0':
                flags.append('zero_events_exported_empty_tube_not_observed_zero_fluorescence')
            if timing['clone_reported'] == 'ROSA26-10':
                flags.append('r10_sample_manifest_missing;do_not_align_rows_across_time')
            base = {'source_id': sid, 'source_kind': 'csv', 'source_path': path.as_posix(),
                'source_sha256': info['sha256'], 'source_record_1based': r['record'],
                'source_line_start_1based': r['line_start'], 'source_line_end_1based': r['line_end'],
                'tube_name': values[0], 'sample_id_raw': values[1], **ann,
                'cell_line_reported': 'HEK293T' if timing['clone_reported'] == 'ROSA26-4' else '',
                'cell_line_evidence_id': 'cell_line_r4' if timing['clone_reported'] == 'ROSA26-4' else '',
                'clone_reported': timing['clone_reported'], 'experiment_name': experiment,
                'acquisition_date_reported': timing['acquisition_date_reported'],
                'days_after_transfection_reported': timing['days_after_transfection_reported'],
                'days_after_sort_reported': timing['days_after_sort_reported'],
                'sort_round_reported': timing['sort_round_reported'],
                'time_metadata_location': f'source_inventory.json sources[{sid}].time_metadata',
                'quality_flags': ';'.join(flags)}
            sample_records.append(base | {'events': int(values[3]), 'source_filename': path.name})
            for col, (name, value) in enumerate(zip(header, values), 1):
                if col < 3:
                    continue
                numeric, status, raw_unit = parse_value(value)
                stat, gate, channel, unit = statistic(name)
                observations.append(base | {'source_column_1based': col, 'source_column_name': name,
                    'metadata_key': '', 'raw_value': value, 'numeric_value': numeric,
                    'value_status': status, 'statistic': stat, 'gate': gate, 'channel': channel,
                    'unit': raw_unit or unit})

    for path in sorted(data.glob('*.fcs'), key=lambda p: p.name.lower()):
        source_counter += 1
        sid = f'S{source_counter:03d}'
        fcs = fcs_metadata(path)
        meta = fcs['metadata']
        raw_name = meta.get('TBNM', '') or Path(meta.get('$FIL', path.name)).stem.replace('_[5 mL Tubes] Data Source - 1', '')
        date = normalized_fcs_date(meta.get('$DATE', ''))
        matches = [r for r in sample_records if r['tube_name'] == raw_name
                   and r['acquisition_date_reported'] == date and r['events'] == fcs['events']]
        info = file_info(path) | {'source_id': sid, 'kind': 'fcs', **fcs,
             'sample_name_reported': raw_name, 'sample_name_metadata_key': 'TBNM' if 'TBNM' in meta else '$FIL',
             'acquisition_date_iso': date,
             'csv_matches': [{'source_id': r['source_id'], 'record_1based': r['source_record_1based'],
                             'source_path': r['source_path'], 'matched_on': ['sample_name', '$DATE', '$TOT=All Events']}
                             for r in matches]}
        sources.append(info)
        if len(matches) == 1:
            base = {k: v for k, v in matches[0].items() if k not in ['events', 'source_filename']}
            base['quality_flags'] += ';linked_csv_same_name_date_events_not_independent_replicate'
        else:
            base = annotate(raw_name)
            base.update(tube_name=raw_name, acquisition_date_reported=date,
                        quality_flags='event_count_is_acquisition_metadata_not_biological_n;clone_and_experimental_time_unverified')
        base.update(source_id=sid, source_kind='fcs_metadata', source_path=path.as_posix(),
                    source_sha256=info['sha256'], source_record_1based='', source_line_start_1based='',
                    source_line_end_1based='', source_column_1based='', source_column_name='',
                    metadata_key='$TOT', raw_value=meta['$TOT'], numeric_value=str(fcs['events']),
                    value_status='observed', statistic='acquisition_event_count', gate='All Events',
                    channel='', unit='events')
        observations.append(base)

    for path in sorted(data.glob('*.dna')):
        source_counter += 1
        sources.append(file_info(path) | {'source_id': f'S{source_counter:03d}', 'kind': 'snapgene',
                                         **snapgene_metadata(path)})

    supporting = sorted({Path(v['source_path']) for v in evidence.values()}, key=str)
    instrument_batches = Counter((s['metadata'].get('$CYT'), s['acquisition_date_iso'])
                                 for s in sources if s['kind'] == 'fcs')
    fcs_data_groups = defaultdict(list)
    for source in sources:
        if source['kind'] == 'fcs':
            fcs_data_groups[source['event_data_sha256']].append(source['source_id'])
    missing = [
        {'id': 'candidate_sequences', 'required': 'Exact sequence and orientation for every candidate; genome build, transcript/TSS, coordinates, inserted downstream bases and junctions.',
         'evidence': 'Source sample_map gives family/upstream length only. The .dna feature packet has no ACTB/GAPDH candidate labels.'},
        {'id': 'r10_tube_manifest', 'required': 'Tube5–Tube19 promoter/construct identities separately for every ROSA26-10 time point; verify G5/G20 row-order change.',
         'evidence': 'D5 CSV record 16=g20 and record 19=g5; D10 and D14 record 16=G5/g5, record 19=G20/g20; D7 records 16/19=Tube14/Tube17.'},
        {'id': 'r10_time', 'required': 'Confirm D5/D7 definitions, actual transfection/sort dates and cell line for clone ROSA26-10.',
         'evidence': 'D5 filename says after-sort DAY5; record 1 column 2 says 分选 DAY13. D7 header DAY15 is unqualified. HEK293T documentation explicitly points to ROSA26-4 only.'},
        {'id': 'r4_batch_timeline', 'required': 'Confirm whether August D28 and July D5/D7 belong to the same transfection population; record both sorting dates.',
         'evidence': 'July 4 transfection DAY12 and July 6 DAY14 imply June 22; August 24 DAY43 implies July 12. August header explicitly says second sorting. Do not silently reconcile these clocks.'},
        {'id': 'biological_replicates', 'required': 'Independent culture/transfection IDs, clone lineage, passage, harvest date and biological replicate count; technical tube repeats cannot substitute.',
         'evidence_id': 'technical_repeats'},
        {'id': 'fcs_batch_context', 'required': 'Clone, days after transfection/sorting and repeat relationships for the June 16 Sony acquisitions; identify any same-sample repeated exports.',
         'evidence': 'FCS TEXT contains instrument/date/events/channels; no clone or experimental day is assigned from file naming.'},
        {'id': 'gates_and_calibration', 'required': 'Original gate hierarchy/coordinates for each CSV, detector calibration and compensation settings across sessions; same-batch WT association.',
         'evidence': 'CSV columns name gates but do not encode gate geometry. CytoFLEX FCS metadata records USCOMP=False. No new gates or normalized labels are produced.'},
        {'id': 'reported_annotations', 'required': 'Confirm sample definitions and the extended Tube→candidate map against lab records, including A6, A20, G1–G20, CAV/CMV controls.',
         'evidence': 'Project scripts and notes document mappings, but they are not sequence-verified sample manifests. Legacy numeric summary fields were not reused.'},
    ]
    inventory = {'schema_version': 1, 'created_at': datetime.now(timezone(timedelta(hours=8))).isoformat(),
        'source_project': project.as_posix(), 'source_project_modified': False,
        'extraction_policy': {'csv_all_exported_quantitative_cells': True,
            'raw_strings_retained': True, 'percent_numeric_values_are_percentage_points': True,
            'unavailable_marker': '####', 'zero_event_tubes_retained_and_flagged': True,
            'fcs_event_data_loaded': False, 'new_gates_or_normalized_targets_computed': False,
            'events_are_not_biological_replicates': True,
            'annotations_are_reported_project_metadata_not_sequence_verification': True},
        'summary': {'files_by_kind': dict(Counter(s['kind'] for s in sources)),
            'csv_sample_rows': len(sample_records), 'csv_zero_event_rows': sum(r['events'] == 0 for r in sample_records),
            'raw_observation_rows': len(observations), 'value_status_counts': dict(Counter(r['value_status'] for r in observations)),
            'fcs_instrument_date_batches': [{'instrument': k[0], 'date': k[1], 'files': v} for k, v in instrument_batches.items()],
            'fcs_with_unique_csv_match': sum(len(s.get('csv_matches', [])) == 1 for s in sources if s['kind'] == 'fcs')},
        'sources': sources, 'supporting_sources': [file_info(p) for p in supporting],
        'identical_fcs_event_data_groups': [{'event_data_sha256': key, 'source_ids': values}
            for key, values in fcs_data_groups.items() if len(values) > 1],
        'annotation_evidence': evidence, 'missing_metadata': missing,
        'sample_records': sample_records,
        'user_clarification': {'statement': '没复测', 'interpretation_from_parent_thread':
            'No additional independent repeats; never interpret the three exported tubes as three biological replicates.',
            'source': 'user clarification relayed by main thread on 2026-10-01; not an instrument field'}}
    fields = ['observation_id', 'source_id', 'source_kind', 'source_path', 'source_sha256',
        'source_record_1based', 'source_line_start_1based', 'source_line_end_1based',
        'source_column_1based', 'source_column_name', 'metadata_key', 'tube_name', 'sample_id_raw',
        'raw_value', 'numeric_value', 'value_status', 'statistic', 'gate', 'channel', 'unit',
        'sample_annotation', 'mapping_status', 'promoter_family_reported', 'upstream_bp_reported',
        'mapping_evidence_ids', 'cell_line_reported', 'cell_line_evidence_id', 'clone_reported',
        'experiment_name', 'acquisition_date_reported', 'days_after_transfection_reported',
        'days_after_sort_reported', 'sort_round_reported', 'time_metadata_location',
        'technical_repeat_index', 'repeat_type', 'biological_replicate_id', 'candidate_sequence_id', 'quality_flags']
    out.mkdir(parents=True, exist_ok=True)
    raw_path = out / 'observations_raw.tsv'
    temporary = raw_path.with_suffix('.tsv.tmp')
    with temporary.open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter='\t', extrasaction='raise')
        writer.writeheader()
        for index, row in enumerate(observations, 1):
            row['observation_id'] = f'O{index:06d}'
            writer.writerow(row)
    temporary.replace(raw_path)
    inventory['diagnostic_normalization'] = diagnostic_tables(observations, sample_records, out)
    inventory['construct_endpoint_link'] = sequence_endpoint_tables(out)
    output_names = ['observations_raw.tsv', 'measurements_ready.tsv', 'diagnostic_candidate_summary.tsv']
    if inventory['construct_endpoint_link']['status'] == 'linked_for_exploratory_diagnostic_only':
        output_names += ['construct_endpoints_D28.tsv', 'construct_time_diagnostics.tsv']
        inventory['missing_metadata'][0]['current_resolution'] = (
            'A separate construct_sequences.tsv now supplies eligible donor features; '
            'assay-to-plasmid barcode/sequence identity remains unverified. '
            'CMVcore is excluded because its supplied feature length conflicts with the reported assay construct.')
    inventory['outputs'] = {name: file_info(out / name) for name in output_names}
    temporary = out / 'source_inventory.json.tmp'
    temporary.write_text(json.dumps(inventory, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(out / 'source_inventory.json')
    print(json.dumps(inventory['summary'], ensure_ascii=False, indent=2))


if __name__ == '__main__':
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    main()
