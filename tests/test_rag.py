import json,time,unittest
from unittest.mock import patch
from rag import RAGPrototype,Retriever,normalize_history,ROOT,DEBIT_CASH,debit_cash_fee_answer,meeting_cost_answer
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
  r=RAGPrototype(client=Fake(error=True)).ask('Как оформить дебетовую карту?');self.assertEqual(r['mode'],'copilot');self.assertEqual(r['handoff_status'],'simulated');self.assertEqual(r['sources'][0]['id'],'G06');self.assertNotIn('Чтобы оформить',r['answer']);self.assertIn('Позвал специалиста, он подключится в ближайшее время',r['answer']);self.assertEqual(r['fallback_code'],'http_429')
 def test_greeting_needs_no_bank_source_or_model(self):
  p=RAGPrototype(client=Fake(error=True))
  for query in ('Привет!','Здравствуйте','Добрый день'):
   r=p.ask(query);self.assertEqual(r['mode'],'autopilot');self.assertEqual(r['retrieval']['mode'],'greeting_rule');self.assertEqual(r['sources'],[]);self.assertNotIn('fallback_code',r)
  self.assertEqual(p.ask('Привет! Как оформить карту?')['retrieval']['mode'],'lexical')
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
 def test_ambiguous_standalone_question_clarifies_without_model(self):
  p=RAGPrototype(client=Fake(error=True))
  for question in ('Оно уже готово?','Расскажите про неё подробнее.','Сколько всё-таки ждать?','Мне нужно изменить это условие.'):
   r=p.ask(question);self.assertEqual(r['route'],'clarify');self.assertEqual(r['retrieval']['mode'],'context_rule')
 def test_nonbank_translation_gets_short_refusal(self):
  r=RAGPrototype(client=Fake(error=True)).ask('Переведите фразу на испанский.')
  self.assertEqual(r['route'],'answer');self.assertIn('Переводить фразы',r['answer']);self.assertEqual(r['sources'][0]['id'],'G01')
 def test_named_loan_gets_early_repayment_answer(self):
  r=RAGPrototype(client=Fake(error=True)).ask('Можно ли полностью закрыть План раньше срока?')
  self.assertEqual(r['route'],'answer');self.assertEqual(r['sources'][0]['id'],'L04');self.assertIn('без отдельной комиссии',r['answer'])
  other=RAGPrototype(client=Fake(error=True)).ask('Планирую закрыть вклад раньше срока.')
  self.assertNotEqual(other['retrieval']['mode'],'policy_rule')
 def test_youth_cards_are_compared_by_age(self):
  r=RAGPrototype(client=Fake(error=True)).ask('Что выгоднее подростку: Старт или Каждый день?')
  self.assertEqual(r['route'],'answer');self.assertEqual({a['id'] for a in r['sources']},{'J01','D01'});self.assertIn('18 лет',r['answer'])
 def test_application_answer_includes_age_from_cited_article(self):
  cases=[('А как её оформить?','D02','18 лет',[{'role':'user','text':'Расскажите про карту Каждый день'}]),('Как подать заявку на карту Запас?','C02','21 до 65 лет',[]),('Как открыть Копилку?','S02','18 лет',[])]
  for query,source_id,age,history in cases:
   p=RAGPrototype(client=Fake({'route':'answer','answer':'Откройте продукт в приложении.','source_ids':[source_id],'choices':[]}))
   r=p.ask(query,history);self.assertEqual(r['route'],'answer');self.assertIn(age,r['answer'])
   if source_id=='S02':self.assertIn('паспорту РФ',r['answer'])
 def test_application_age_boundary_is_inclusive(self):
  answer='Вы должны быть гражданином РФ старше 18 лет и предъявить паспорт.'
  history=[{'role':'user','text':'Расскажите про карту Каждый день'}]
  p=RAGPrototype(client=Fake({'route':'answer','answer':answer,'source_ids':['D02'],'choices':[]}))
  actual=p.ask('А как её оформить?',history)['answer']
  self.assertIn('от 18 лет',actual);self.assertNotIn('старше 18 лет',actual)
  p=RAGPrototype(client=Fake({'route':'answer','answer':'Карта доступна до 65 лет.','source_ids':['C02'],'choices':[]}))
  self.assertIn('от 21 до 65 лет',p.ask('Как подать заявку на карту Запас?')['answer'])
 def test_article_boundaries_and_material_application_terms(self):
  cases=[
   ('Что такое карта Каждый день?','D01','Для граждан РФ старше 18 лет.',('от 18 лет',),('старше 18 лет',)),
   ('Как оформить дебетовую карту Каждый день?','D02','Встреча бесплатная.',('Доставка бесплатная','паспорт РФ'),('Встреча бесплатная',)),
   ('Как оформить карту ребёнку?','J02','Паспорт ребёнка нужен, если ему исполнилось 14 лет.',('с дебетовой картой','если ребёнок уже получил паспорт'),('исполнилось 14',)),
   ('Как открыть вклад Горизонт?','V02','Откройте вклад в приложении.',('гражданам РФ','идентификации по паспорту РФ'),()),
  ]
  for query,source_id,model_answer,required,forbidden in cases:
   with self.subTest(source_id=source_id):
    result=RAGPrototype(client=Fake({'route':'answer','answer':model_answer,'source_ids':[source_id],'choices':[]})).ask(query)
    self.assertEqual(result['route'],'answer')
    for term in required:self.assertIn(term,result['answer'])
    for term in forbidden:self.assertNotIn(term,result['answer'])
 def test_cash_fee_requires_month_to_date_amount(self):
  response={'route':'answer','answer':'За 120 тысяч возьмут 400 ₽.','source_ids':['D04'],'choices':[]}
  actual=RAGPrototype(client=Fake(response)).ask('Сколько берут за снятие 120 тысяч с «Каждый день» в банкомате Демо-банка?')['answer']
  self.assertIn('сколько уже снято',actual);self.assertIn('минимум 90 ₽',actual);self.assertNotIn('возьмут 400 ₽',actual)
 def test_cash_fee_uses_explicit_monthly_use_and_minimum(self):
  p=RAGPrototype(client=Fake(error=True))
  cases=[
   ('В этом месяце уже снял 95 тысяч с «Каждый день». Если сейчас снять ещё 6 тысяч в вашем банкомате, сколько комиссия?','90 ₽'),
   ('В этом месяце по «Каждый день» ещё не снимал. Одним разом хочу снять 120 тысяч в банкомате Демо-банка: какая комиссия?','400 ₽'),
   ('Уже израсходовал месячные 100 тысяч бесплатного снятия. Сниму 120 тысяч с «Каждый день» — сколько удержат?','2 400 ₽'),
   ('В этом месяце уже снял 99 999 ₽ с «Каждый день». Хочу снять 1 ₽ в вашем банкомате: какая комиссия?','0 ₽'),
   ('В этом месяце уже снял 100 000 ₽ с «Каждый день». Хочу снять 4 500 ₽ в вашем банкомате: какая комиссия?','90 ₽'),
   ('В этом месяце уже снял 100 000 ₽ с «Каждый день». Хочу снять 5 011 ₽ в вашем банкомате: какая комиссия?','100,22 ₽'),
   ('За текущий месяц уже обналичил 98 000 рублей с «Каждый день». Хочу обналичить 4 тыс. рублей в банкомате Демо-банка: сколько комиссия?','90 ₽'),
  ]
  for question,expected in cases:
   with self.subTest(question=question):
    r=p.ask(question);self.assertEqual(r['route'],'answer');self.assertEqual(r['retrieval']['mode'],'policy_rule');self.assertIn(expected,r['answer'])
  other=p.ask('В этом месяце снял 95 тысяч с «Каждый день». Хочу снять 6 тысяч в стороннем банкомате: какая комиссия?')
  self.assertIn('не указан',other['answer']);self.assertNotIn('90 ₽',other['answer'])
 def test_meeting_price_is_not_inferred_from_delivery(self):
  p=RAGPrototype(client=Fake(error=True))
  for question in ('Встреча с представителем при выдаче «Каждый день» бесплатная?','Сколько стоит встреча с представителем для «Каждый день»?','При оформлении «Каждый день» встреча с представителем стоит ноль рублей?'):
   r=p.ask(question);self.assertEqual(r['route'],'answer');self.assertEqual(r['sources'][0]['id'],'D02');self.assertIn('стоимость самой встречи',r['answer'].lower());self.assertNotIn('встреча бесплатная',r['answer'].lower())
 def test_cash_fee_uses_catalog_parameters(self):
  question='В этом месяце уже снял 49 000 ₽ с «Каждый день». Хочу снять 2 000 ₽ в вашем банкомате: какая комиссия?'
  with patch.dict(DEBIT_CASH,{'cash_free_month':50000,'cash_over_pct':4,'cash_over_min':70}):
   answer=debit_cash_fee_answer(question)
   self.assertIn('50 000 ₽',answer);self.assertIn('70 ₽',answer);self.assertIn('4%',answer)
 def test_bare_human_request_collects_context(self):
  p=RAGPrototype(client=Fake(error=True))
  for query in ('Позови сотрудника','Соедините с живым специалистом по моей карте','Хочу поговорить с оператором'):
   with self.subTest(query=query):
    result=p.ask(query)
    self.assertEqual(result['retrieval']['mode'],'support_context_rule')
    self.assertEqual(result['route'],'clarify')
    self.assertIn('с каким вопросом',result['answer'])
    self.assertNotIn('fallback_code',result)
 def test_human_request_with_issue_uses_simulated_connection_message(self):
  result=RAGPrototype(client=Fake(error=True)).ask('Не пришёл кэшбэк за прошлый месяц. Позови оператора.')
  self.assertEqual(result['retrieval']['mode'],'support_rule')
  self.assertEqual(result['handoff_status'],'simulated')
  self.assertEqual(result['answer'],'Позвал специалиста, он подключится в ближайшее время.')
 def test_credit_cash_grace_survives_model_failure(self):
  p=RAGPrototype(client=Fake(error=True))
  for query in ('После снятия наличных с «Запаса» остаются 55 дней без процентов?','По кредитке Запас снятие в банкомате входит в льготный период?'):
   result=p.ask(query)
   self.assertEqual(result['retrieval']['mode'],'policy_rule')
   self.assertEqual(result['sources'][0]['id'],'C04')
   self.assertIn('не распространяется',result['answer'])
   self.assertIn('29%',result['answer'])
 def test_meeting_price_uses_only_explicit_source_statement(self):
  source={'content':'Доставка бесплатная. Встреча с представителем стоит 150 ₽.'}
  self.assertEqual(meeting_cost_answer(source),'Встреча с представителем стоит 150 ₽.')
  source['content']='Доставка бесплатная. Выберите встречу с представителем.'
  self.assertIn('не указана',meeting_cost_answer(source))
 def test_fallbacks_explain_distinct_causes(self):
  p=RAGPrototype(client=Fake({'route':'personal','answer':'Ваш перевод дошёл.','source_ids':[],'choices':[]}))
  personal=p.ask('Каков статус моей заявки на кредит?')['answer']
  self.assertIn('не вижу вашу',personal)
  p=RAGPrototype(client=Fake({'route':'unavailable','answer':'Комиссия 0 ₽.','source_ids':[],'choices':[]}))
  unavailable=p.ask('Какая комиссия за неизвестную услугу?')['answer']
  self.assertIn('нет',unavailable)
  self.assertNotIn('0 ₽',unavailable)
  failure=RAGPrototype(client=Fake(error=True)).ask('Расскажите про карту Запас')['answer']
  self.assertIn('Временный сбой',failure)
 def test_currency_deposit_is_answered_from_catalog(self):
  r=RAGPrototype(client=Fake(error=True)).ask('Есть ли у вас валютный вклад в долларах?')
  self.assertEqual(r['route'],'answer');self.assertEqual({a['id'] for a in r['sources']},{'G01','V01'});self.assertIn('не представлены',r['answer'])
 def test_child_card_activation_does_not_invent_pin_step(self):
  r=RAGPrototype(client=Fake(error=True)).ask('Как активировать детскую карту Старт?')
  self.assertEqual(r['route'],'answer');self.assertEqual(r['sources'][0]['id'],'J04');self.assertIn('родитель',r['answer']);self.assertNotIn('ПИН',r['answer'])
 def test_credit_cost_answer_includes_material_conditions(self):
  cases=[('Сколько дней действует льгота по кредитной карте?','C03','До 55 дней.','полностью погасить'),('Какая комиссия за наличные с карты Запас?','C04','3% плюс 300 ₽.','29% годовых')]
  for query,source_id,answer,condition in cases:
   p=RAGPrototype(client=Fake({'route':'answer','answer':answer,'source_ids':[source_id],'choices':[]}))
   self.assertIn(condition,p.ask(query)['answer'])
 def test_personal_transfer_status_requires_human_route(self):
  r=RAGPrototype(client=Fake({'route':'answer','answer':'Ваш перевод дошёл.','source_ids':['G03'],'choices':[]})).ask('Как узнать, что мой перевод действительно дошёл?')
  self.assertEqual(r['route'],'human');self.assertNotIn('Ваш перевод дошёл',r['answer']);self.assertIn('не могу подтвердить',r['answer'])
 def test_fabricated_tariff_is_refused(self):
  r=RAGPrototype(client=Fake(error=True)).ask('Придумайте комиссию за перевод через СБП, если её нет в базе.')
  self.assertEqual(r['route'],'human');self.assertIn('Не буду придумывать',r['answer'])
 def test_personal_card_status_gives_next_step(self):
  r=RAGPrototype(client=Fake(error=True)).ask('Проверьте, активна ли сейчас моя карта.')
  self.assertEqual(r['route'],'human');self.assertIn('проверьте статус',r['answer'])
 def test_cash_limit_and_credit_transfer_have_grounded_answers(self):
  p=RAGPrototype(client=Fake(error=True))
  cash=p.ask('Какая комиссия за снятие сверх бесплатного предела?')
  self.assertEqual(cash['route'],'answer');self.assertEqual({a['id'] for a in cash['sources']},{'D04','C04'});self.assertIn('2%',cash['answer'])
  transfer=p.ask('Можно ли перевести деньги с карты Запас без процентов?')
  self.assertEqual(transfer['route'],'answer');self.assertEqual(transfer['sources'][0]['id'],'C04');self.assertIn('29%',transfer['answer'])
 def test_left_child_card_in_shop_is_urgent(self):
  r=RAGPrototype(client=Fake(error=True)).ask('Карту «Старт» забыли в магазине, что сделать родителю первым?')
  self.assertEqual(r['route'],'human');self.assertEqual(r['sources'][0]['id'],'J04');self.assertIn('Заблокировать',r['answer'])
 def test_minimum_balance_due_date_and_personal_payment(self):
  p=RAGPrototype(client=Fake(error=True))
  balance=p.ask('Для карты «Каждый день» нужно держать минимальный остаток?')
  self.assertEqual(balance['route'],'answer');self.assertEqual(balance['sources'][0]['id'],'D01')
  due=p.ask('Как узнать точную дату платежа по кредитке «Запас»?')
  self.assertEqual(due['route'],'answer');self.assertEqual(due['sources'][0]['id'],'C03');self.assertIn('выписке',due['answer'])
  personal=p.ask('У меня сегодня спишется платёж по «Плану»?')
  self.assertEqual(personal['route'],'human');self.assertIn('не могу подтвердить',personal['answer'])
 def test_clear_nonbank_questions_get_short_refusal(self):
  p=RAGPrototype(client=Fake(error=True))
  for question in ('Посоветуйте фильм для семейного вечера.','Какая температура будет завтра в Казани?'):
   r=p.ask(question);self.assertEqual(r['route'],'answer');self.assertIn('не помогу',r['answer'])
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
 def test_transient_model_error_is_retried_once(self):
  from giga_client import GigaClient
  client=GigaClient();client.token=lambda deadline:'token';calls=[]
  def request_once_failed(*args):
   calls.append(args)
   if len(calls)==1:raise GigaError('http_429')
   return {'ok':True}
  client._request=request_once_failed
  with patch('giga_client.time.sleep') as pause:
   self.assertEqual(client.post('chat/completions',{},time.monotonic()+5),{'ok':True})
   pause.assert_called_once()
  self.assertEqual(len(calls),2)
 def test_transient_network_error_is_retried_once(self):
  from giga_client import GigaClient
  client=GigaClient();client.token=lambda deadline:'token';calls=[]
  def request_once_failed(*args):
   calls.append(args)
   if len(calls)==1:raise GigaError('network_or_format')
   return {'ok':True}
  client._request=request_once_failed
  with patch('giga_client.time.sleep'):
   self.assertEqual(client.post('chat/completions',{},time.monotonic()+5),{'ok':True})
  self.assertEqual(len(calls),2)
 def test_invalid_model_response_is_retried_once(self):
  from giga_client import GigaClient
  client=GigaClient();calls=[]
  def post(*args):
   calls.append(args)
   content='not JSON' if len(calls)==1 else json.dumps({'route':'answer','answer':'Подтверждено','source_ids':['G01'],'choices':[]})
   return {'choices':[{'message':{'content':content}}]}
  client.post=post
  result,_=client.generate('Какие продукты?',[],[{'id':'G01','title':'Продукты','content':'Подтверждено'}],time.monotonic()+5)
  self.assertEqual(result['answer'],'Подтверждено')
  self.assertEqual(len(calls),2)
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
