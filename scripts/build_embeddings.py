"""Build a versioned vector artifact. Authorization stays in the process environment."""
import json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from rag import ROOT,content_hash,search_text
from giga_client import GigaClient
articles=json.loads((ROOT/'knowledge_base.json').read_text());client=GigaClient();vectors=[];tokens=0
for a in articles:
 # Articles deliberately short. API token usage is retained to verify the 512-token bound.
 v,t=client.embed([search_text(a)],time.monotonic()+25)
 if t>512:raise RuntimeError(f'Article {a["id"]} exceeds embedding context; split it before indexing')
 vectors.extend(v);tokens+=t
 print(a['id'],t,flush=True)
out={'model':client.embedding_model,'kb_hash':content_hash(articles),'ids':[a['id'] for a in articles],'vectors':vectors,'usage_tokens':tokens}
(ROOT/'data/kb_vectors.json').write_text(json.dumps(out,separators=(',',':')))
print('INDEX_READY',len(vectors),'TOKENS',tokens)
