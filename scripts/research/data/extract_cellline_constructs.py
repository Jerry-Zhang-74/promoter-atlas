"""Read supplied SnapGene donor files without changing their sequence or annotations.

Coordinates in the source feature XML are 1-based inclusive. Assay-to-file links
use the documented A/G naming and explicit feature lengths, and remain marked
as inferred where a sample ID is absent from the source filename.
"""
import csv
import hashlib
import json
import re
import struct
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = Path('E:/BIO/promoter_truncation')
OUT = ROOT / 'reports/cellline_v1_20261001'


def read_snapgene(path):
    blob = path.read_bytes()
    position = 0
    sequence = None
    features = []
    magic = False
    while position < len(blob):
        if len(blob) - position < 5:
            raise ValueError(f'Truncated packet header: {path}')
        packet = blob[position]
        length = struct.unpack('>I', blob[position + 1:position + 5])[0]
        end = position + 5 + length
        if end > len(blob):
            raise ValueError(f'Truncated packet body: {path}')
        data = blob[position + 5:end]
        position = end
        if packet == 9:
            magic = data.startswith(b'SnapGene')
        elif packet == 0:
            if sequence is not None:
                raise ValueError(f'Multiple DNA packets: {path}')
            sequence = data[1:].decode('ascii').upper()
            if not re.fullmatch('[ACGTRYSWKMBDHVN]+', sequence):
                raise ValueError(f'Invalid DNA alphabet: {path}')
        elif packet == 10:
            for feature in ET.fromstring(data).findall('Feature'):
                features.append({
                    'name': feature.get('name', ''), 'type': feature.get('type', ''),
                    'directionality': feature.get('directionality', 'unspecified'),
                    'segments_1based_inclusive': [seg.get('range') for seg in feature.findall('Segment')],
                })
    if not magic or sequence is None:
        raise ValueError(f'Not a complete SnapGene DNA file: {path}')
    return sequence, features, hashlib.sha256(blob).hexdigest()


def extract_feature(sequence, feature):
    chunks = []
    for interval in feature['segments_1based_inclusive']:
        start, end = map(int, interval.split('-'))
        if not 1 <= start <= len(sequence) or not 1 <= end <= len(sequence):
            raise ValueError('Feature outside molecule')
        chunks.append(sequence[start-1:end] if start <= end else sequence[start-1:] + sequence[:end])
    seq = ''.join(chunks)
    if feature['directionality'] == '2':
        seq = seq.translate(str.maketrans('ACGTRYSWKMBDHVN', 'TGCAYRSWMKVHDBN'))[::-1]
    return seq


def write_table(path, rows):
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter='\t')
        writer.writeheader()
        writer.writerows(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    root = SOURCE / 'plasmid'
    requests = []
    for short, gene in [('A', 'ACTB'), ('G', 'GAPDH')]:
        prefix = f'AP77_GG_{gene}' + ('_mut' if gene == 'ACTB' else '') + '_Assembly'
        for length in (100, 200, 300, 400):
            construct = short + str(length // 100)
            requests.append((construct, gene, length, root/'pool_2_length'/f'{prefix}_{construct}.dna',
                             [gene + '_promoter'], 'filename_and_feature_length'))
        requests.append((short+'20', gene, 2000, root/'pool_2_length'/f'{prefix}.dna',
                         [gene + '_promoter'], 'inferred_from_gene_and_feature_length'))
        requests.append((short+'5', gene, 500,
                         root/('pool_2_length' if gene == 'ACTB' else 'pool_1_SCP')/f'AP77-{gene}.dna',
                         ['ACTB_-0.5k~-1' if gene == 'ACTB' else 'GAPDH_v4'],
                         'inferred_from_gene_and_feature_length'))
    requests.append(('CMVfull', 'CMV', 584, root/'pool_1_SCP'/'AP77-CMV - complete.dna',
                     ['CMV enhancer', 'CMV promoter'], 'inferred_from_name_and_combined_features'))
    requests.append(('CMVcore', 'CMV', 200, root/'pool_1_SCP'/'AP77-CMV.dna',
                     ['CMV SP1'], 'conflict_feature_length_81_vs_documented_200'))
    rows, source_rows, fasta = [], [], []
    sequences = {}
    for construct, family, expected_length, path, names, mapping in requests:
        molecule, features, file_hash = read_snapgene(path)
        chosen = []
        for name in names:
            matches = [feature for feature in features if feature['name'] == name]
            if len(matches) != 1:
                raise ValueError(f'Expected unique feature {name}: {path}')
            chosen.append(matches[0])
        sequence = ''.join(extract_feature(molecule, feature) for feature in chosen)
        intervals = ';'.join(','.join(f['segments_1based_inclusive']) for f in chosen)
        valid = len(sequence) == expected_length
        row = {
            'construct_id': construct, 'family': family, 'documented_length_bp': expected_length,
            'sequence_length_bp': len(sequence), 'sequence_sha256': hashlib.sha256(sequence.encode()).hexdigest(),
            'source_path': str(path), 'source_sha256': file_hash, 'source_molecule_length_bp': len(molecule),
            'feature_names': ';'.join(names), 'feature_ranges_1based_inclusive': intervals,
            'feature_directionality': ';'.join(f['directionality'] for f in chosen),
            'mapping_evidence': mapping, 'eligible_exploratory_sequence_fit': int(valid),
            'assay_to_plasmid_barcode_verified': 0,
            'sequence_status': 'supplied_donor_feature' if valid else 'length_conflict_exclude_from_fit',
            'sequence': sequence,
        }
        rows.append(row)
        source_rows.append({'path': str(path), 'sha256': file_hash, 'molecule_length_bp': len(molecule),
                            'features': features})
        sequences[construct] = sequence
        if valid:
            fasta.append(f'>{construct}\n{sequence}\n')
    comparisons = []
    for family in ('A', 'G'):
        full = sequences[family+'20']
        for label in (family+str(n) for n in (1, 2, 3, 4, 5)):
            seq = sequences[label]
            comparisons.append({'construct_id': label, 'reference': family+'20',
                                'comparison': 'same_length_suffix',
                                'mismatching_positions_1based': [i+1 for i,(a,b) in enumerate(zip(seq,full[-len(seq):])) if a != b]})
    write_table(OUT/'construct_sequences.tsv', rows)
    (OUT/'construct_sequences.fasta').write_text(''.join(fasta), encoding='ascii')
    (OUT/'construct_sequence_provenance.json').write_text(json.dumps({
        'format': 'SnapGene packets; feature XML uses 1-based inclusive coordinates',
        'source_directory': str(SOURCE), 'sources': source_rows, 'nested_sequence_comparisons': comparisons,
        'limitations': [
            'File-to-assay correspondence is inferred from project naming, donor vector type and feature length; no sample barcode link was found.',
            'A5 differs by one base from the last 500 bp of the supplied mutated A20 donor; actual supplied sequence is retained.',
            'The AP77-CMV candidate has an 81 bp CMV SP1 feature; historical assay documentation calls CMVcore 200 bp. Excluded pending resolution.',
            'No explicit 600 bp A6 donor was identified; no A6 sequence was synthesized from another construct.',
            'Unstranded promoter features are exported in stored molecule orientation, with the original directionality retained.',
            'These are supplied plasmid design files, not proof of sequence validation of the assayed clones.',
        ],
    }, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'rows': len(rows), 'eligible': sum(r['eligible_exploratory_sequence_fit'] for r in rows),
                      'construct_lengths': {r['construct_id']: r['sequence_length_bp'] for r in rows},
                      'comparisons': comparisons}, ensure_ascii=False))


if __name__ == '__main__':
    main()
