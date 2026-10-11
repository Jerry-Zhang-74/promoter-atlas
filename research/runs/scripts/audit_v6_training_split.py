"""Measure existing split limitations without changing the resumed experiment."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from train_promoter_atlas_v6 import INPUT, MpraDataset

ROOT = Path(__file__).resolve().parents[2]


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    df = pd.read_csv(INPUT, sep='\t', usecols=['species', 'gene_id', 'split', 'label_mask', 'Score_E', 'ortholog_E'])
    df['split'] = df.split.fillna('train')
    df['gene_id'] = df.gene_id.str.split('.').str[0]
    result = {'training_input_sha256': sha256(INPUT), 'training_rows': len(df)}
    groups = df.groupby(['species', 'gene_id']).split.nunique()
    result['within_species_genes_spanning_splits'] = int((groups > 1).sum())
    assert result['within_species_genes_spanning_splits'] == 0
    human = df[(df.species == 'homo_sapiens') & (df.label_mask == 1)].drop_duplicates('gene_id')
    htest = set(human.loc[human.split == 'test', 'gene_id'])
    hval = set(human.loc[human.split == 'val', 'gene_id'])
    orth = pd.read_csv(ROOT / 'features/multispecies/ortholog_human_to_species.tsv', sep='\t')
    orth['human_gene_id'] = orth.human_gene_id.str.split('.').str[0]
    exposed_test, exposed_val, rows = set(), set(), []
    for species, grp in df[(df.species != 'homo_sapiens') & (df.split == 'train') & df.ortholog_E.notna()].groupby('species'):
        col = species + '_gene'
        pairs = orth[['human_gene_id', col]].dropna().copy()
        pairs[col] = pairs[col].str.split('.').str[0]
        pairs = pairs.merge(human[['gene_id', 'Score_E']].rename(columns={
            'gene_id': 'human_gene_id', 'Score_E': 'donor_score_E'}), on='human_gene_id', how='inner')
        donor_map = grp.merge(pairs, left_on='gene_id', right_on=col, how='inner')
        donor_map = donor_map[np.isclose(donor_map.ortholog_E, donor_map.donor_score_E, rtol=1e-7, atol=1e-8)]
        donors = set(donor_map.human_gene_id)
        exposed_test |= donors & htest
        exposed_val |= donors & hval
        rows.append({'species': species, 'train_rows_with_test_donor': int(donor_map.human_gene_id.isin(htest).sum()),
                     'test_donor_genes': len(donors & htest), 'val_donor_genes': len(donors & hval)})
    result.update({'human_test_labeled_genes': len(htest), 'human_test_genes_with_ortholog_training_label': len(exposed_test),
                   'human_val_labeled_genes': len(hval), 'human_val_genes_with_ortholog_training_label': len(exposed_val),
                   'ortholog_label_overlap_by_species': rows})
    mpra = MpraDataset(str(ROOT / 'features/mpra/mpra_supervised.tsv'), seed=20260912)
    d = mpra.df
    test = d.split_v2 == 'test'
    train = d.split_v2 == 'train'
    train_families = set(d.loc[train, 'family_v2'])
    result['mpra_test_rows_in_train_families'] = int((test & d.family_v2.isin(train_families)).sum())
    result['mpra_test_n'] = int(test.sum())
    result['mpra_core_sha256'] = sha256(ROOT / 'features/mpra/mpra_supervised.tsv')
    result['interpretation'] = ('The frozen split is disjoint within each species but does not group orthologs across species. '
        'Scores describe this historical experiment, not a leakage-free estimate on unseen ortholog groups. '
        'MPRA small families can span splits; epoch 0 also used a different split whose exact assignment is unavailable.')
    out = ROOT / 'reports/rounds/round04_split_audit.json'
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
