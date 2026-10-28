"""Check product relationships without relying on UI wording."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from product_data import enrich_payload
ROOT=Path(__file__).resolve().parent.parent

class ProductDataTest(unittest.TestCase):
    def load(self):
        return json.loads((ROOT/'knowledge/catalog.json').read_text()), json.loads((ROOT/'quiz/data/questions.json').read_text())
    def test_all_source_fields_are_preserved(self):
        catalog,bank=self.load();snapshot=copy.deepcopy(bank);enrich_payload(ROOT,catalog,bank)
        self.assertEqual(bank,snapshot)
        for original,enriched in zip(bank['questions'],catalog['question_bank']):
            for k,v in original.items():self.assertEqual(enriched[k],v)
            self.assertTrue(enriched['topics'])
        self.assertEqual(len(catalog['question_bank']),224)
    def test_paths_and_prerequisites_validate(self):
        catalog,bank=self.load();enrich_payload(ROOT,catalog,bank)
        self.assertEqual(len(catalog['learning_map']['paths']),3)
        self.assertGreaterEqual(len(catalog['learning_map']['experiments']),12)
    def test_invalid_metadata_fails_loudly(self):
        for changes in ({'prerequisites':{'policy-gradient':['policy-gradient']}},{'question_links':{'Q999':['policy-gradient']}}):
            with tempfile.TemporaryDirectory() as folder:
                root=Path(folder);(root/'knowledge').mkdir()
                meta=json.loads((ROOT/'knowledge/learning-map.json').read_text());meta.update(changes)
                (root/'knowledge/learning-map.json').write_text(json.dumps(meta))
                catalog,bank=self.load()
                with self.assertRaises(ValueError):enrich_payload(root,catalog,bank)
if __name__=='__main__':unittest.main()
