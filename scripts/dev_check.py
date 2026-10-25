import sys,json,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from rag import RAGPrototype,ROOT
p=RAGPrototype();cases=json.loads((ROOT/'evaluation/dev.json').read_text());hits=0
for c in cases:
 docs,info=p.retriever.search(c['query']);ok=c['expected_id'] in info['retrieved_ids'];hits+=ok
 print(c['query'],ok,info['retrieved_ids'],flush=True)
for q,h in [('Какие продукты есть?',[]),('Как оформить дебетовую карту?',[]),('А какие документы нужны?',[{'role':'user','text':'Расскажи о карте Старт'},{'role':'assistant','text':'Старт — детская карта под контролем родителя.'}]),('Как заблокировать карту?',[]),('Какой сегодня курс доллара?',[])]:
 result=p.ask(q,h);print(json.dumps({'query':q,**result},ensure_ascii=False),flush=True)
print('DEV_RECALL',hits,len(cases))
