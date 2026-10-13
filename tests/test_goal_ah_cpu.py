"""CPU tests for data integrity, isolation gates and action accounting."""
import copy
import json
import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import prepare_v133_goal_ah as h
import run_v133_goal_ah_entity_relation as runner
import assess_v133_goal_ah as assess

class GoalAHCPU(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.proposals=h.rows(h.OUT/'material-proposals.jsonl')
        cls.reviews=h.rows(h.OUT/'material-review-main.jsonl')

    def test_reviewed_quota_and_real_failure_coverage(self):
        material,gate=h.material_gate(self.proposals,self.reviews)
        self.assertTrue(gate['passed']);self.assertEqual(len(material),64)
        self.assertEqual(gate['real_error_question_count'],16)

    def test_cannot_accept_hidden_or_fabricated_quote(self):
        reviews=copy.deepcopy(self.reviews)
        r=next(r for r in reviews if r['accepted']);r['source_quotes'][0]['quote']='NOT IN ACTUAL TOOL BODY'
        with self.assertRaisesRegex(ValueError,'verbatim'):h.material_gate(self.proposals,reviews)

    def test_cannot_change_category_to_fill_quota(self):
        reviews=copy.deepcopy(self.reviews)
        r=next(r for r in reviews if r['accepted']);r['corrected_target']['verdict']='invented'
        with self.assertRaisesRegex(ValueError,'category'):h.material_gate(self.proposals,reviews)

    def test_duplicate_review_rejected(self):
        with self.assertRaises(ValueError):h.material_gate(self.proposals,self.reviews+[self.reviews[0]])

    def test_shortfall_is_terminal_gate(self):
        reviews=copy.deepcopy(self.reviews)
        for r in reviews:
            if r['corrected_target']['verdict']=='entity_mismatch':r['accepted']=False
        _,gate=h.material_gate(self.proposals,reviews)
        self.assertFalse(gate['passed']);self.assertEqual(gate['shortages']['entity_mismatch'],16)
        self.assertFalse(gate['gpu_authorized'])

    def test_fourth_repeated_search_counted_once_with_both_reasons(self):
        trace=[dict(action='search',query=q) for q in ['A','B','C',' A ']]
        invalid=runner.broad_invalid(trace)
        self.assertEqual(invalid,[dict(call_index=3,reasons=['repeated_search','search_over_budget'])])

    def test_actual_training_evidence_preserved_and_budget(self):
        material={r['judgment_id']:r for r in h.rows(h.OUT/'judgments-reviewed.jsonl')}
        tokens=h.rows(h.OUT/'judgment-training-inputs.jsonl')
        self.assertEqual(len(tokens),64)
        for t in tokens:
            self.assertLessEqual(t['input_tokens'],2048);self.assertLessEqual(t['target_tokens'],128)
            self.assertEqual(t['prompt_attention_mask'],[1]*len(t['prompt_token_ids']))
            original=h.documents(material[t['judgment_id']]['actual_prompt'])
            actual=h.documents(t['rendered_input'])
            self.assertEqual([(d['document_id'],d['text']) for d in original],[(d['document_id'],d['text']) for d in actual])

    def test_no_evaluation_ids_in_either_training_loss(self):
        train={r['id'] for r in h.rows(h.Z/'training-inputs.jsonl')}
        judge={r['case_id'] for r in h.rows(h.OUT/'judgments-reviewed.jsonl')}
        cfg=json.loads((h.OUT/'training-freeze.json').read_text(encoding='utf8'))
        for spec in cfg['evaluation']:
            evaluation={r['case_id'] for r in h.rows(h.ROOT/spec['cases'])}
            self.assertFalse(evaluation&train);self.assertFalse(evaluation&judge)

    def test_mechanism_is_bounded_and_incomplete_search_is_legal(self):
        probes=h.rows(h.OUT/'mechanism-probes.jsonl')
        self.assertEqual(len(probes),16)
        for p in probes:
            self.assertTrue(p['training_use_forbidden'])
            if p['expected_action']=='search':self.assertGreater(p['remaining_searches'],0)

    def test_product_gain_does_not_override_bad_or_protection(self):
        base=dict(success=35,bad=22,legacy_invalid=0,broad_invalid=7)
        candidate=dict(success=39,bad=23,legacy_invalid=0,broad_invalid=7)
        self.assertFalse(assess.qualification(candidate,base,{},0,True)['passed'])
        candidate['bad']=22
        self.assertTrue(assess.qualification(candidate,base,{'Y':(34,33)},1,True)['passed'])
        self.assertFalse(assess.qualification(candidate,base,{'Y':(34,32)},1,True)['passed'])
        self.assertFalse(assess.qualification(candidate,base,{},2,True)['passed'])

    def test_continuation_gain_cannot_be_attributed_to_judgment(self):
        a=dict(success=39,entity_relation_mismatch=5,evidence_complete_rate=.6)
        b=dict(a)
        self.assertFalse(assess.mechanism_b_over_a(b,a,12,11)['passed'])
        b['success']=40
        self.assertTrue(assess.mechanism_b_over_a(b,a,12,11)['passed'])

    def test_release_requires_both_seeds_and_quality(self):
        results={42:{'passed':False},43:{'passed':True}}
        self.assertFalse(assess.release_gate(results,True)['publish'])
        results[42]['passed']=True
        self.assertFalse(assess.release_gate(results,False)['publish'])
        self.assertEqual(assess.release_gate(results,True)['selected_seed'],42)

if __name__=='__main__':unittest.main()
