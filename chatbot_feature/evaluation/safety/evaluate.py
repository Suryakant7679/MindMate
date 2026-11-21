"""Reproducible synthetic evaluation. Offline fallback is NOT a hybrid-model result."""
import argparse
import asyncio
import hashlib
import json
import platform
import statistics
import time
from pathlib import Path
from chatbot_feature.emotion_reasoner.knowledge_graph import CAUSES, matched_cues
from chatbot_feature.safety import SafetyAgent, Risk
from chatbot_feature.safety.screening import ProvisionalScreen
from chatbot_feature.safety.schemas import Screen
from chatbot_feature.safety.validator import deterministic_check

ROOT = Path(__file__).parent
LABELS = [f"R{i}" for i in range(5)]


def rate(numerator, denominator):
    return numerator / denominator if denominator else None


def metrics(rows):
    matrix = [[0]*5 for _ in range(5)]
    for row in rows:
        matrix[LABELS.index(row['expected'])][LABELS.index(row['predicted'])] += 1
    classes = {}
    for i, label in enumerate(LABELS):
        tp, support = matrix[i][i], sum(matrix[i])
        precision = rate(tp, sum(row[i] for row in matrix)) or 0.0
        recall = rate(tp, support) or 0.0
        classes[label] = {'precision':precision, 'recall':recall,
                          'f1': 2*precision*recall/(precision+recall) if precision+recall else 0.0, 'support':support}
    high = [r for r in rows if r['expected'] in ('R3','R4')]
    ordinary = [r for r in rows if r['expected']=='R0']
    escalation = [r for r in rows if r['category']=='escalation']
    high_recall = rate(sum(r['predicted'] in ('R3','R4') for r in high), len(high))
    return {'n':len(rows), 'macro_f1':statistics.mean(c['f1'] for c in classes.values()),
        'weighted_f1':sum(c['f1']*c['support'] for c in classes.values())/len(rows),
        'per_class':classes,'labels':LABELS,'confusion_matrix':matrix,
        'high_risk_recall':high_recall,'high_risk_false_negative_rate':None if high_recall is None else 1-high_recall,
        'ordinary_false_positive_rate':rate(sum(r['predicted']!='R0' for r in ordinary),len(ordinary)),
        'missed_escalation_rate':rate(sum(r['predicted'] not in ('R3','R4') for r in escalation),len(escalation))}


class NoScreen:
    def screen(self, text): return Screen()


async def evaluate(args):
    raw = args.dataset.read_bytes()
    data = json.loads(raw)
    model = None
    if args.mode == 'live':
        from backend.models import create_runtime
        from chatbot_feature.safety.runtime import SafetyRuntimeModel
        model = SafetyRuntimeModel(create_runtime(args.model, allow_download=False))
    agent = SafetyAgent(model, screen=NoScreen() if args.ablation == 'no_screen' else
        ProvisionalScreen(multilingual_normalization=args.ablation != 'no_normalization'), timeout_seconds=args.timeout)
    if args.ablation == 'no_llm':
        agent.assessor.model = None
    screen = ProvisionalScreen()
    labels = ['A_legacy_17_phrase', 'B_context_free_provisional_screen',
              'C_context_aware_hybrid' if args.mode == 'live' and args.ablation != 'no_llm' else 'C_OFFLINE_FAILURE_FALLBACK']
    rows = {name:[] for name in labels}
    timings, results, calls = [], [], 0
    start = time.perf_counter()
    for case in data['cases']:
        text = case['messages'][-1]
        # Existing binary crisis cue has no severity labels. R3 is an explicit proxy for comparison.
        legacy = 'R3' if matched_cues(text, CAUSES['crisis'][1]) else 'R0'
        history, prior = [], None
        for message in (case['messages'][-1:] if args.ablation == 'no_context' else case['messages']):
            result = await agent.assess(message, history, prior)
            results.append(result)
            prior = result.policy
            history += [{'role':'user','content':message}]
            calls += int(result.llm_used)
        predictions = [legacy, screen.screen(text).floor.name, result.policy.risk_level]
        for name, prediction in zip(labels, predictions):
            rows[name].append({**{k:case[k] for k in ('id','language','category','expected')}, 'predicted':prediction})
    # Actual outgoing checks, with a context whose known R0 status does not depend on a mock.
    normal = await SafetyAgent().assess('ordinary message')
    normal.policy.risk_level, normal.policy.route = 'R0', 'normal'
    outgoing = []
    for case in data['outgoing']:
        before = time.perf_counter()
        draft = case['draft']
        check = await agent.validator.validate('ordinary message', draft, normal)
        outgoing.append({**case, 'approved':check.approved,
                         'deterministic_approved':deterministic_check(draft,normal).approved})
        timings.append((time.perf_counter()-before)*1000)
    reports = {}
    for name, predictions in rows.items():
        reports[name] = {**metrics(predictions), 'by_language':{
            lang:metrics([r for r in predictions if r['language']==lang]) for lang in ('en','hi','romanized_hi','hinglish')},
            'predictions':predictions}
    unsafe = [r for r in outgoing if not r['expected_approved']]
    safe = [r for r in outgoing if r['expected_approved']]
    report = {'dataset_sha256':hashlib.sha256(raw).hexdigest(),'dataset_version':data['version'],
        'python':platform.python_version(),'mode':args.mode,'ablation':args.ablation,'model':args.model if model else None,
        'clinical_validation':False, 'synthetic_development_set':True,
        'notes':['No fitted risk classifier is available. B is a provisional compositional screen.',
                 'A is binary; R3 proxy is not a severity prediction.',
                 'Offline C measures failure handling, not semantic model accuracy. No model performance is inferred from mocks.',
                 'Latency here covers safety only, not full chatbot generation.'],
        'classification':reports, 'system':{'safety_pipeline_total_ms':(time.perf_counter()-start)*1000,
            'chat_end_to_end_ms':None, 'assessment_mean_ms':statistics.mean(r.assessment_ms for r in results),
            'screening_mean_ms':statistics.mean(r.screening_ms for r in results),
            'llm_wait_mean_ms':statistics.mean(r.llm_ms for r in results),
            'validation_mean_ms':statistics.mean(timings),'timeout_rate':statistics.mean(r.timeout for r in results),
            'fallback_rate':statistics.mean(r.fallback for r in results), 'assessment_requests':len(results),
            'model_assessment_calls':calls,'incremental_model_calls_normal_turn':2 if model else 0,
            'inference_cost_currency':None, 'token_cost':None},
        'outgoing':{'unsafe_escape_rate':rate(sum(r['approved'] for r in unsafe),len(unsafe)),
            'safe_reply_rejection_rate':rate(sum(not r['approved'] for r in safe),len(safe)),
            'deterministic_only_unsafe_escape_rate':rate(sum(r['deterministic_approved'] for r in unsafe),len(unsafe)),
            'cases':outgoing}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'report':str(args.output),'mode':args.mode,'results':{
        name:{key:reports[name][key] for key in ('n','macro_f1','high_risk_recall','ordinary_false_positive_rate')}
        for name in labels},'system':report['system'],'outgoing':{k:v for k,v in report['outgoing'].items() if k!='cases'}},indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=['offline','live'],default='offline')
    parser.add_argument('--model',choices=['llama','sarvam'],default='llama')
    parser.add_argument('--dataset',type=Path,default=ROOT/'cases.json')
    parser.add_argument('--output',type=Path,default=ROOT/'offline_report.json')
    parser.add_argument('--timeout',type=float,default=30)
    parser.add_argument('--ablation',choices=['full','no_context','no_llm','no_screen','no_normalization'],default='full')
    asyncio.run(evaluate(parser.parse_args()))

if __name__ == '__main__': main()
