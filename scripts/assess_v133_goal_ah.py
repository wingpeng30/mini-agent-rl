"""Pure CPU gates. Semantic review must use the same frozen rubric for all models."""
from collections import Counter
import statistics

CATEGORIES={'success','factual_error','unsupported_by_visible_evidence','missing_required_information','ambiguous'}

def metrics(raw,review):
    if len(raw)!=len(review) or {r['case_id'] for r in raw}!={r['case_id'] for r in review}:
        raise ValueError('One reviewed judgment per actual trajectory required')
    if any(r['category'] not in CATEGORIES for r in review):raise ValueError('Unknown review category')
    if any(r['category']=='ambiguous' for r in review):raise ValueError('Unresolved ambiguity: no qualification gate')
    count=Counter(r['category'] for r in review)
    return dict(success=count['success'],bad=count['factual_error']+count['unsupported_by_visible_evidence'],
        factual_error=count['factual_error'],unsupported=count['unsupported_by_visible_evidence'],
        em=sum(r['em'] for r in raw),f1=statistics.mean(r['f1'] for r in raw),
        entity_relation_mismatch=sum(bool(r['entity_mismatch'] or r['relation_mismatch']) for r in review),
        evidence_complete_rate=statistics.mean(bool(r['evidence_complete']) for r in review),
        legacy_invalid=sum(r['legacy_invalid_actions'] for r in raw),
        broad_invalid=sum(len(r['broad_invalid_actions']) for r in raw),
        searches=statistics.mean(r['searches'] for r in raw),
        model_calls=statistics.mean(r['model_calls'] for r in raw),
        mean_latency=statistics.mean(r['elapsed_seconds'] for r in raw),
        median_latency=statistics.median(r['elapsed_seconds'] for r in raw))

def qualification(candidate,baseline,regressions,lost_protected,engineering):
    checks=dict(success=candidate['success']>=baseline['success']+4,bad=candidate['bad']<=baseline['bad'],
        legacy_invalid=candidate['legacy_invalid']<=baseline['legacy_invalid'],
        broad_invalid=candidate['broad_invalid']<=baseline['broad_invalid'],
        regressions=all(after>=before-1 for before,after in regressions.values()),
        protected=lost_protected<=1,engineering=bool(engineering))
    return dict(passed=all(checks.values()),checks=checks)

def mechanism_b_over_a(b,a,b_probe,a_probe):
    checks=dict(product_success=b['success']>=a['success']+1,probe_correct=b_probe>=a_probe+1,
        mismatch=b['entity_relation_mismatch']<=a['entity_relation_mismatch'],
        completeness=b['evidence_complete_rate']>=a['evidence_complete_rate'])
    return dict(passed=all(checks.values()),checks=checks)

def release_gate(seed_results,confirmation_quality_passed):
    # Seed 42 is fixed for release; no choosing the seed based on confirmation.
    passed=confirmation_quality_passed and set(seed_results)=={42,43} and all(seed_results[s]['passed'] for s in [42,43])
    return dict(publish=bool(passed),selected_seed=42 if passed else None)
