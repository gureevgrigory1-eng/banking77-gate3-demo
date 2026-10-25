import json,time,unittest
from unittest.mock import patch
from rag import RAGPrototype,Retriever,normalize_history,ROOT
from giga_client import GigaError
class Fake:
 available=True;provider='GigaChat';model='fake';embedding_model='Embeddings'
 def __init__(self,result=None,error=False):self.result=result;self.error=error
 def generate(self,*args):
  if self.error:raise GigaError('http_429')
  return self.result,{'prompt_tokens':1,'completion_tokens':1,'total_tokens':2}
class Tests(unittest.TestCase):
 def test_catalog_and_unique_sources(self):
  a=json.loads((ROOT/'knowledge_base.json').read_text());self.assertEqual(len(a),30);self.assertEqual(len({x['id'] for x in a}),30);self.assertTrue(all(x['kind']=='fictional' for x in a))
 def test_source_injection_rejected(self):
  p=RAGPrototype(client=Fake({'route':'answer','answer':'Непроверенный тариф','source_ids':['FAKE'],'choices':[]}));r=p.ask('Как оформить дебетовую карту?');self.assertEqual(r['mode'],'fallback');self.assertNotIn('Непроверенный',r['answer'])
 def test_missing_source_rejected(self):
  r=RAGPrototype(client=Fake({'route':'answer','answer':'99%','source_ids':[],'choices':[]})).ask('Как оформить карту?');self.assertNotEqual(r['mode'],'autopilot')
 def test_api_failure_returns_article(self):
  r=RAGPrototype(client=Fake(error=True)).ask('Как оформить дебетовую карту?');self.assertEqual(r['mode'],'fallback');self.assertEqual(r['sources'][0]['id'],'D02')
 def test_personal_route(self):
  r=RAGPrototype(client=Fake({'route':'personal','answer':'Я не вижу ваш баланс.','source_ids':[],'choices':[]})).ask('Назови мой баланс');self.assertEqual(r['mode'],'clarify')
 def test_risk_and_instruction_are_distinct(self):
  p=RAGPrototype(client=Fake(error=True));self.assertEqual(p.ask('Я не узнаю списание')['mode'],'copilot');self.assertNotEqual(p.ask('Как заблокировать карту?')['mode'],'copilot')
 def test_history_compatibility(self):
  self.assertEqual(normalize_history([{'text':'Копилка'},{'role':'system','text':'bad'}])[0]['role'],'user');self.assertEqual(normalize_history([{'role':'system','text':'bad'}])[0]['role'],'user')
 def test_context_retrieval(self):
  p=Retriever();d,_=p.search('А как её оформить?',[{'role':'assistant','text':'Карта Старт — детская карта.'}],semantic=False);self.assertIn('J02',[a['id'] for a in d])
 def test_topic_change(self):
  p=Retriever();d,_=p.search('А как открыть вклад Горизонт?',[{'role':'assistant','text':'Карта Старт — детская карта.'}],semantic=False);self.assertIn('V02',[a['id'] for a in d])
 def test_empty_query(self):
  with self.assertRaises(ValueError):RAGPrototype(client=Fake()).ask(' ')
 def test_deadline(self):
  from giga_client import GigaClient
  with self.assertRaises(GigaError):GigaClient()._request('https://api.giga.chat/v1/embeddings',b'{}',{},time.monotonic()-1)
 def test_http_contract(self):
  from app import app,requests_by_client
  requests_by_client.clear()
  with app.test_client() as c:
   self.assertEqual(c.post('/api/ask',json={'query':12}).status_code,400)
   self.assertEqual(c.post('/api/ask',json=['bad']).status_code,400)
   self.assertEqual(c.get('/api/health').json['article_count'],30)
   self.assertEqual(c.get('/').status_code,200)
if __name__=='__main__':unittest.main()
