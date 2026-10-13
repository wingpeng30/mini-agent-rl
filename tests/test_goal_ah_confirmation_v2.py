"""Integrity checks for the bounded confirmation repair, without model calls."""
import json
import sys
import unittest
from collections import Counter
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from prepare_v133_goal_ah import ROOT,OUT,rows,sha
import prepare_v133_goal_ah_confirmation as c

class ConfirmationV2(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.path=OUT/'confirmation-repair-v2'
        cls.manifest=json.loads((cls.path/'final-manifest.json').read_text(encoding='utf8'))
        cls.cases=rows(cls.path/'final-source-cases.jsonl')

    def test_fixed_counts_and_no_training_oracle_changes(self):
        self.assertEqual(Counter(x['source'] for x in self.cases),{'musique':32,'2wiki':32})
        self.assertEqual(len(self.cases),64)
        self.assertEqual(self.manifest['supplemented'],28)
        self.assertEqual(self.manifest['hotpot_used'],0)
        self.assertTrue(self.manifest['original_gold_unchanged'])
        self.assertFalse(self.manifest['model_filtering'])
        self.assertEqual(self.manifest['training_changes'],0)

    def test_frozen_assets_and_original_labels(self):
        for name,digest in self.manifest['files'].items():self.assertEqual(sha(ROOT/name),digest)
        originals={x['id']:x for x in rows(self.path/'kept-v1-cases.jsonl')}
        for n in ['musique','2wiki']:originals.update({x['id']:x for x in rows(self.path/f'candidate-pool-{n}.jsonl')})
        for x in self.cases:
            for key in ['question','answer','answer_aliases','chain']:self.assertEqual(x[key],originals[x['id']][key])

    def test_each_support_exactly_preserved_in_actual_corpus(self):
        corpus=json.loads((self.path/'final-confirm-corpus.json').read_text(encoding='utf8'))
        self.assertEqual(len(corpus),2048)
        lookup={x['title']:x['text'] for x in corpus}
        for x in self.cases:
            for d in x['docs']:
                if d['support']:self.assertEqual(lookup[d['title']],d['text'])

    def test_new_two_hop_quotes_and_mutual_entity_isolation(self):
        lookup={x['id']:x for x in self.cases}; seen=set()
        for x in self.cases:
            entities=c.case_entities(x);self.assertFalse(entities&seen);seen.update(entities)
        for x in rows(self.path/'source-review-main.jsonl'):
            if not x['accepted']:continue
            docs={d['title']:d['text'] for d in lookup[x['case_id']]['docs'] if d['support']}
            self.assertEqual({q['relation_index'] for q in x['quotes']},{0,1})
            for q in x['quotes']:self.assertIn(q['quote'],docs[q.get('title',q.get('document_title'))])

    def test_historical_isolation_and_approval_pending(self):
        audit=json.loads((self.path/'final-isolation.json').read_text(encoding='utf8'))
        self.assertTrue(audit['passed']);self.assertEqual(audit['overlaps'],[])
        self.assertTrue(audit['new_candidates_exclude_all_v1_pools'])
        self.assertFalse((OUT/'gpu-approval.json').exists())

if __name__=='__main__':unittest.main()
