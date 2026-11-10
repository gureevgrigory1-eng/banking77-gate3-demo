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
  a=json.loads((ROOT/'knowledge_base.json').read_text());self.assertEqual(len(a),32);self.assertEqual(len({x['id'] for x in a}),32);self.assertTrue(all(x['kind']=='fictional' for x in a))
 def test_source_injection_rejected(self):
  p=RAGPrototype(client=Fake({'route':'answer','answer':'Непроверенный тариф','source_ids':['FAKE'],'choices':[]}));r=p.ask('Как оформить дебетовую карту?');self.assertEqual(r['mode'],'copilot');self.assertNotIn('Непроверенный',r['answer'])
 def test_missing_source_rejected(self):
  r=RAGPrototype(client=Fake({'route':'answer','answer':'99%','source_ids':[],'choices':[]})).ask('Как оформить карту?');self.assertNotEqual(r['mode'],'autopilot')
 def test_api_failure_does_not_return_unverified_article(self):
  r=RAGPrototype(client=Fake(error=True)).ask('Как оформить дебетовую карту?');self.assertEqual(r['mode'],'copilot');self.assertEqual(r['handoff_status'],'prepared');self.assertEqual(r['sources'][0]['id'],'G06');self.assertNotIn('Чтобы оформить',r['answer']);self.assertIn('оператор не подключён',r['answer'])
 def test_personal_route(self):
  r=RAGPrototype(client=Fake({'route':'personal','answer':'Ваш баланс 999 ₽.','source_ids':[],'choices':[]})).ask('Назови мой баланс');self.assertEqual(r['mode'],'copilot');self.assertNotIn('999',r['answer'])
 def test_unavailable_route_cannot_assert_unsupported_bank_policy(self):
  r=RAGPrototype(client=Fake({'route':'unavailable','answer':'У нас нет такой услуги.','source_ids':[],'choices':[]})).ask('Есть ли у вас неизвестная услуга?')
  self.assertEqual(r['mode'],'copilot');self.assertNotIn('У нас нет',r['answer'])
 def test_known_missing_terms_do_not_reuse_other_transfer_tariff(self):
  p=RAGPrototype(client=Fake({'route':'answer','answer':'Комиссия 0,5%.','source_ids':['G03'],'choices':[]}))
  r=p.ask('Какая комиссия за перевод себе в другой банк через СБП?')
  self.assertEqual(r['retrieval']['mode'],'scope_rule');self.assertNotIn('0,5%',r['answer'])
  self.assertEqual(p.ask('Есть ли переводы в криптовалюте?')['retrieval']['mode'],'scope_rule')
 def test_deposit_followup_does_not_promise_same_terms(self):
  p=RAGPrototype(client=Fake({'route':'answer','answer':'Да, на тех же условиях.','source_ids':['V01'],'choices':[]}))
  history=[{'role':'user','text':'На сколько открывается вклад Горизонт?'},{'role':'assistant','text':'Вклад открывается на 6 месяцев.','route':'answer'}]
  r=p.ask('А её можно продлить?',history)
  self.assertEqual(r['retrieval']['mode'],'policy_rule');self.assertIn('Автоматического продления',r['answer']);self.assertNotIn('на тех же условиях',r['answer'])
 def test_risk_and_instruction_are_distinct(self):
  p=RAGPrototype(client=Fake(error=True));self.assertEqual(p.ask('Я не узнаю списание')['retrieval']['mode'],'safety_rule');self.assertNotEqual(p.ask('Как заблокировать карту?')['retrieval']['mode'],'safety_rule')
 def test_credentials_and_child_card_loss_get_relevant_actions(self):
  p=RAGPrototype(client=Fake(error=True))
  password=p.ask('Кто-то знает мой пароль от банковского приложения.')
  self.assertEqual(password['mode'],'copilot');self.assertIn('пароль',password['answer']);self.assertNotIn('Если карта потеряна',password['answer'])
  child=p.ask('А если карта ребёнка потерялась?')
  self.assertEqual(child['mode'],'copilot');self.assertEqual(child['sources'][0]['id'],'J04');self.assertIn('Заблокировать',child['answer'])
 def test_sensitive_values_redacted_from_operator_summary(self):
  p=RAGPrototype(client=Fake(error=True))
  r=p.ask('Я сообщил код из СМС 123456 и номер карты 1234 5678 9012 3456 постороннему.')
  self.assertEqual(r['mode'],'copilot');self.assertNotIn('123456',r['operator_summary']);self.assertNotIn('1234 5678 9012 3456',r['operator_summary'])
 def test_history_compatibility(self):
  self.assertEqual(normalize_history([{'text':'Копилка'},{'role':'system','text':'bad'}])[0]['role'],'user');self.assertEqual(normalize_history([{'role':'system','text':'bad'}])[0]['role'],'user')
  self.assertEqual(normalize_history([{'role':'assistant','text':'Уточните','route':'clarify'}])[0]['route'],'clarify')
 def test_three_unresolved_clarifications_prepare_handoff(self):
  clarification={'route':'clarify','answer':'Уточните, что вы имеете в виду?','source_ids':[],'choices':['Доставка','Потеря']}
  history=[]
  for index in range(3):
   history.extend([{'role':'user','text':f'Вопрос {index}'},{'role':'assistant','text':'Уточните','route':'clarify'}])
  self.assertEqual(RAGPrototype(client=Fake(clarification)).ask('Я имею в виду другое',history)['mode'],'copilot')
  self.assertEqual(RAGPrototype(client=Fake(clarification)).ask('Я имею в виду другое',history[2:])['mode'],'clarify')
 def test_clear_answer_after_clarifications_is_not_escalated(self):
  history=[{'role':'user','text':'Вопрос'},{'role':'assistant','text':'Уточните','route':'clarify'}]*3
  response={'route':'answer','answer':'Есть три карты.','source_ids':['G01'],'choices':[]}
  self.assertEqual(RAGPrototype(client=Fake(response)).ask('Какие продукты есть?',history)['mode'],'autopilot')
 def test_fourth_clarification_on_model_failure_prepares_handoff(self):
  history=[{'role':'user','text':'Вопрос'},{'role':'assistant','text':'Уточните','route':'unavailable'}]*3
  self.assertEqual(RAGPrototype(client=Fake(error=True)).ask('Непонятный запрос',history)['mode'],'copilot')
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
   self.assertEqual(c.get('/api/health').json['article_count'],32)
   page=c.get('/')
   self.assertEqual(page.status_code,200)
   self.assertIn('Онлайн-помощник',page.data.decode())
   self.assertNotIn('Учебный прототип',page.data.decode())
if __name__=='__main__':unittest.main()
