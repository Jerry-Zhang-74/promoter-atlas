import hashlib,importlib.util,io,json,tarfile,tempfile,unittest
from pathlib import Path
from promoter_atlas.layout import canonical_path,project_path,ROOT

class LayoutTests(unittest.TestCase):
    def test_canonical_and_legacy_data_roots(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);old='labels/build/example.tsv'
            expected=root/'data/processed/labels/build/example.tsv'
            self.assertEqual(Path(project_path(root,old)),expected)
            legacy=root/old;legacy.parent.mkdir(parents=True);legacy.write_text('old')
            self.assertEqual(Path(project_path(root,old)),legacy)
            expected.parent.mkdir(parents=True);expected.write_text('organized')
            self.assertEqual(Path(project_path(root,old)),expected)

    def test_all_frozen_members_have_unique_destinations(self):
        m=json.loads((ROOT/'artifacts/manifests/TRAINING_ASSETS.json').read_text('utf-8'))
        paths={}
        for g in m['groups']:
            for f in g['files']:
                destination=canonical_path(ROOT,f['path'])
                if destination in paths:self.assertEqual(paths[destination],f['path'])
                paths[destination]=f['path']
                self.assertEqual(destination,ROOT/f['repository_path'])
        self.assertEqual(len(paths),1265)

    def test_archive_restoration_and_existing_file_protection(self):
        import zstandard
        spec=importlib.util.spec_from_file_location('restore',ROOT/'scripts/restore_training_assets.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        body=b'gene_id\tScore_E\nexample\t0.5\n';name='labels/build/example.tsv'
        buf=io.BytesIO()
        with tarfile.open(fileobj=buf,mode='w') as tf:
            item=tarfile.TarInfo(name);item.size=len(body);tf.addfile(item,io.BytesIO(body))
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);archive=root/'fixture.tar.zst';archive.write_bytes(zstandard.ZstdCompressor().compress(buf.getvalue()))
            group={'files':[{'path':name,'bytes':len(body),'sha256':hashlib.sha256(body).hexdigest()}]}
            out=root/'project';out.mkdir()
            self.assertEqual(module.restore(group,[archive],out),1)
            restored=out/'data/processed/labels/build/example.tsv'
            self.assertEqual(restored.read_bytes(),body)
            self.assertEqual(module.restore(group,[archive],out),1)
            restored.write_bytes(b'changed')
            with self.assertRaises(FileExistsError):module.restore(group,[archive],out)

    def test_path_escape_rejected(self):
        for path in ['../outside','/absolute','C:/absolute','labels/../../outside']:
            with self.assertRaises(ValueError):canonical_path(ROOT,path)

if __name__=='__main__':unittest.main()
