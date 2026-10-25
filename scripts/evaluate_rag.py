"""Run held-out synthetic cases; retain raw answers for explicit manual review."""
import json,re,sys,time,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from rag import RAGPrototype,ROOT
p=RAGPrototype();cases=json.loads((ROOT/'evaluation/control.json').read_text());rows=[]
def facts_ok(text,checks):return all(re.search(c,text,re.I|re.S) for c in checks)
for case in cases:
 start=time.monotonic();docs,info=p.retriever.search(case['query'],case['history'],semantic=False);baseline_ms=(time.monotonic()-start)*1000
 baseline=docs[0]['content'] if info['lexical_score']>=.10 else 'В базе нет ответа.'
 result=p.ask(case['query'],case['history'])
 row=dict(case=case,result=result,baseline=dict(answer=baseline,ids=info['retrieved_ids'],latency_ms=baseline_ms))
 row['rag_hit']=bool(set(case['expected_ids'])&set(result['retrieval']['retrieved_ids'])) if case['group'] in ('faq','dialogue') else None
 row['baseline_hit']=bool(set(case['expected_ids'])&set(info['retrieved_ids'])) if case['group'] in ('faq','dialogue') else None
 row['rag_facts_match']=bool(facts_ok(result['answer'],case['checks'])) if case['checks'] else None
 row['baseline_facts_match']=bool(facts_ok(baseline,case['checks'])) if case['checks'] else None
 rows.append(row)
 (ROOT/'evaluation/raw_results.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2))
 print(case['id'],case['group'],result['mode'],result['retrieval']['mode'],result['latency_ms'],'ms',row['rag_hit'],row['rag_facts_match'],flush=True)
 time.sleep(.3)
qa=[r for r in rows if r['case']['group'] in ('faq','dialogue')];risk=[r for r in rows if r['case']['group']=='risk'];lat=[r['result']['latency_ms'] for r in rows];gen=[r['result']['latency_ms'] for r in qa]
summary={'cases':len(rows),'qa_cases':len(qa),'control_sha256':hashlib.sha256((ROOT/'evaluation/control.json').read_bytes()).hexdigest(),'rag_recall_at5':sum(r['rag_hit'] for r in qa)/len(qa),'baseline_recall_at5':sum(r['baseline_hit'] for r in qa)/len(qa),'rag_regex_fact_coverage':sum(r['rag_facts_match'] for r in qa)/len(qa),'baseline_regex_fact_coverage':sum(r['baseline_facts_match'] for r in qa)/len(qa),'risk_detected':sum(r['result']['mode']=='copilot' for r in risk),'risk_cases':len(risk),'latency_all_p50_ms':float(np.percentile(lat,50)),'latency_all_p95_ms':float(np.percentile(lat,95)),'latency_qa_p50_ms':float(np.percentile(gen,50)),'latency_qa_p95_ms':float(np.percentile(gen,95)),'baseline_p95_ms':float(np.percentile([r['baseline']['latency_ms'] for r in rows],95)),'usage':{k:sum(r['result']['usage'][k] for r in rows) for k in ('embedding_tokens','prompt_tokens','completion_tokens','total_tokens')},'fallbacks':sum(r['result']['mode']=='fallback' for r in rows),'note':'Regex checks measure expected fact coverage, not independently verified factual correctness. Manual review required.'}
(ROOT/'evaluation/summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2));print(json.dumps(summary,ensure_ascii=False),flush=True)
