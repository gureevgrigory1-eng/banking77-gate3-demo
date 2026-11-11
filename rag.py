"""Lexical RAG with grounded GigaChat evidence selection; optional vector index."""
import hashlib,json,re,time
from pathlib import Path
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from giga_client import GigaClient,GigaError
ROOT=Path(__file__).resolve().parent
NOTE='Информация относится к продуктам Демо-банка. Операции со счетами в этом чате недоступны.'
ALIASES={'debit':['каждый день','дебетов','дебетовая','зарплатн'],'credit':['запас','кредитн','кредитка'],'junior':['старт','детск','ребен','ребён','школьн','подрост'],'savings':['копилк','накопительн'],'deposit':['горизонт','вклад','депозит'],'loan':['план','кредит налич','потребительск']}
RISK=re.compile(r'мошенн|украл|украден|потерял.{0,25}карт|потеряна.{0,15}карт|потерял.{0,15}телефон|не\s+(?:узнаю|совершал|делал|проводил|подтверждал|покупал)|незнаком.{0,20}(?:плат|спис|перев)|неизвестн.{0,20}(?:плат|спис|перев)|без моего|подозрительн.{0,20}(?:спис|перев|операц)|(?:сообщил|передал|назвал|отправил).{0,30}(?:код|парол|cvv|пин)|(?:списали|сняли).{0,25}(?:дважды|два раза)|(?:переведи|отправь|перечисли)\s+(?:мне\s+)?(?:деньги|\d)|(?:оформи|выдай)\s+(?:мне\s+)?кредит|(?:заблокируй|разблокируй).{0,20}карт|(?:закрой|открой).{0,20}(?:сч[её]т|вклад)|stolen|fraud|unrecognised|unauthorized',re.I)
CARD_LOSS=re.compile(r'(?:карт.{0,40}(?:потер|украд|пропал)|(?:потер|украд|пропал).{0,40}карт)',re.I)
CREDENTIAL_RISK=re.compile(r'(?:кто-то|посторон|незнаком|мошенн).{0,45}(?:парол|пин|cvv|код)|(?:парол|пин|cvv|код).{0,45}(?:знает|узнал|передал|назвал|сообщил|ут[её]к|посторон)',re.I)
GREETING=re.compile(r'^(?:привет(?:ствую)?|здравствуйте|здравствуй(?:те)?|доброе утро|добрый день|добрый вечер)[\s!?.]*$',re.I)

def risk_kind(query):
 if CREDENTIAL_RISK.search(query):return 'credentials'
 if CARD_LOSS.search(query):return 'child_card_loss' if re.search(r'детск|реб[её]н|старт',query,re.I) else 'card_loss'
 if RISK.search(query):return 'other'
 return None

def redact_summary(text):
 text=re.sub(r'\b(?:\d[ -]?){12,18}\d\b','[номер карты скрыт]',text)
 text=re.sub(r'\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b','[email скрыт]',text)
 text=re.sub(r'((?:код(?: из смс)?|cvv|пин)[^\d\n]{0,15})\d{3,6}\b',r'\1[значение скрыто]',text,flags=re.I)
 return text

def unsupported_scope(query):
 text=query.lower()
 if 'криптовалют' in text and re.search(r'перевод|плат[её]ж|обмен',text):return 'Условия операций с криптовалютой не описаны в базе знаний.'
 if 'сбп' in text and re.search(r'себе|самому себе|между своими',text) and re.search(r'друг.{0,12}банк|ин.{0,12}банк',text):
  return 'Условия перевода себе в другой банк по СБП не описаны в базе знаний.'
 return None

def needs_context(query,history):
 if history or product_mentions(query) or re.search(r'карт|сч[её]т|вклад|кредит|комисс|перевод|плат[её]ж|тариф|заявк',query,re.I):return False
 return bool(re.search(r'\b(оно|она|он|е[её]|это|этот|эту|этой|него|не[её]|ней|там)\b|\b(?:сколько|долго).{0,20}ждать\b',query,re.I))

def outside_bank_scope(query):
 return bool(re.search(r'\bперевед(?:ите|и)\b.{0,45}\b(?:фраз[уы]|предложени[ея]|текст)\b.{0,45}\bна\s+[а-яё]+',query,re.I))

def asks_loan_early_repayment(query):
 return bool(re.search(r'\bплан\b|кредит налич|потребительск',query,re.I)) and bool(re.search(r'закры|погас',query,re.I)) and bool(re.search(r'раньше|досроч|срок',query,re.I))

def asks_youth_card_comparison(query):
 text=query.lower()
 return 'старт' in text and 'каждый день' in text and bool(re.search(r'подрост|реб[её]н|дет|выгод|сравн|отлич',text))

def asks_personal_transfer_status(query):
 return bool(re.search(r'\bмо[йяюеиё]\b|\bмоего\b|\bсво[йяюеи]\b',query,re.I) and re.search(r'перевод|плат[её]ж',query,re.I) and re.search(r'дош[её]л|зачисл|поступ|получ|статус',query,re.I))

def asks_personal_card_status(query):
 return bool(re.search(r'\bмо[йяюеиё]\b|\bмоего\b|\bсво[йяюеи]\b',query,re.I) and re.search(r'карт',query,re.I) and re.search(r'проверь|активн|статус',query,re.I) and not re.search(r'как\s+активир',query,re.I))

def asks_fabricated_terms(query):
 return bool(re.search(r'придум|выдум|наугад',query,re.I) and re.search(r'комисс|тариф|ставк|услов',query,re.I))

def asks_deposit_extension(query,history):
 return bool(re.search(r'продл|пролонг',query,re.I) and 'deposit' in product_mentions(query+' '+ ' '.join(x['text'] for x in history)))
def normalize_history(history):
 result=[]
 for x in (history or [])[-6:]:
  if not isinstance(x,dict) or not isinstance(x.get('text'),str):continue
  role=x.get('role') if x.get('role') in ('user','assistant') else 'user'
  item={'role':role,'text':x['text'][:1200]}
  if role=='assistant' and x.get('route') in ('answer','clarify','unavailable','personal','risk','human'):
   item['route']=x['route']
  result.append(item)
 return result
def clarification_streak(history):
 count=0
 for item in reversed(history):
  if item['role']!='assistant':continue
  if item.get('route') not in ('clarify','unavailable'):break
  count+=1
 return count
def content_hash(articles):return hashlib.sha256(json.dumps(articles,ensure_ascii=False,sort_keys=True).encode()).hexdigest()
def search_text(a):return a['title']+' '+a['aliases']+' '+a['content']
def product_mentions(text):return [p for p,words in ALIASES.items() if any(w in text.lower() for w in words)]
class Retriever:
 def __init__(self,root=ROOT,client=None):
  self.root=Path(root); self.articles=json.loads((self.root/'knowledge_base.json').read_text()); self.by_id={a['id']:a for a in self.articles}; self.client=client or GigaClient()
  texts=[search_text(a) for a in self.articles]
  self.word=TfidfVectorizer(ngram_range=(1,2),sublinear_tf=True); self.w=self.word.fit_transform(texts)
  self.char=TfidfVectorizer(analyzer='char_wb',ngram_range=(3,5),sublinear_tf=True); self.c=self.char.fit_transform(texts)
  self.vectors=None; path=self.root/'data/kb_vectors.json'
  if path.exists():
   data=json.loads(path.read_text())
   if data.get('kb_hash')==content_hash(self.articles) and data.get('model')==self.client.embedding_model and data.get('ids')==[a['id'] for a in self.articles]:
    self.vectors=np.asarray(data['vectors'],dtype=np.float32)
    self.vectors/=np.maximum(np.linalg.norm(self.vectors,axis=1,keepdims=True),1e-8)
 def contextual_query(self,query,history):
  mentions=product_mentions(query)
  if not mentions and (re.search(r'\b(е[её]|его|это[тй]?|он[ао]?|так[а-я]*|них|эти)\b',query.lower()) or len(query.split())<=6):
   for item in reversed(history):
    prior=product_mentions(item['text'])
    if len(prior)==1:return query+' '+next(a['title'].split(':')[0] for a in self.articles if a['product']==prior[0])
    if len(prior)>1:break
  return query
 def search(self,query,history=None,deadline=None,semantic=True):
  q=self.contextual_query(query,history or []); deadline=deadline or time.monotonic()+25
  scores=.45*(self.w@self.word.transform([q]).T).toarray().ravel()+.55*(self.c@self.char.transform([q]).T).toarray().ravel()
  lex=np.argsort(-scores); ranked=lex; mode='lexical'; tokens=0
  if semantic and self.vectors is not None and self.client.available:
   try:
    vectors,tokens=self.client.embed([q],min(deadline,time.monotonic()+7))
    v=np.asarray(vectors[0],dtype=np.float32); v/=max(np.linalg.norm(v),1e-8)
    sims=self.vectors@v; dense=np.argsort(-sims)
    fusion=np.zeros(len(scores))
    for order in (lex,dense):
     for rank,index in enumerate(order):fusion[index]+=1/(60+rank+1)
    ranked=np.argsort(-fusion); mode='hybrid'
   except (GigaError,ValueError,KeyError,TypeError):mode='lexical_fallback'
  top=list(ranked[:5])
  if re.search(r'какие.{0,15}(продукт|услуг)|каталог|что.{0,12}(предлага|есть у вас)',query.lower()):
   idx=next(i for i,a in enumerate(self.articles) if a['id']=='G01'); top=[idx]+[i for i in top if i!=idx]; top=top[:5]
  return [self.articles[i] for i in top],{'mode':mode,'embedding_tokens':tokens,'lexical_score':round(float(max(scores)),4),'query':q,'retrieved_ids':[self.articles[i]['id'] for i in top]}
class RAGPrototype:
 def __init__(self,root=ROOT,client=None):
  self.remote=client or GigaClient(); self.retriever=Retriever(root,self.remote)
 def result(self,answer,mode='autopilot',sources=None,**extra):
  return dict(mode=mode,answer=answer,sources=sources or [],note=NOTE,**extra)
 def specialist(self,query,history,answer,reason,sources=None,draft='Проверить запрос клиента и помочь через защищённый канал.'):
  summary=redact_summary('\n'.join([x['text'] for x in history if x['role']=='user'][-2:]+[query]))
  answer=answer.rstrip()+' Специалист подключён в демо-режиме; для реальной помощи используйте поддержку в приложении.'
  return self.result(answer,mode='copilot',sources=sources or [self.retriever.by_id['G06']],operator_summary=summary,draft=draft,reason=reason,route='human',handoff_status='simulated')
 def handoff(self,query,history,kind='other'):
  source=self.retriever.by_id['G05']; support=self.retriever.by_id['G06']
  sources=[source,support]
  if kind=='child_card_loss':
   sources=[self.retriever.by_id['J04'],support]
   answer='Если потеряна детская карта «Старт», родителю нужно открыть её в приложении: «Старт» → «Настройки» → «Заблокировать», затем обратиться в поддержку через «Помощь» → «Чат» для перевыпуска. В этом чате карта не блокируется.'
  elif kind=='credentials':
   answer='Если пароль, ПИН или код стал известен постороннему, прекратите общение с ним, смените пароль через официальный канал и срочно обратитесь в поддержку через «Помощь» → «Чат» в приложении. Не сообщайте новые коды или пароль в этом чате. Здесь я не могу защитить ваш доступ.'
  elif kind=='card_loss':
   answer='Если карта потеряна или украдена, сразу откройте её в приложении и выберите «Настройки» → «Заблокировать», затем обратитесь в поддержку через «Помощь» → «Чат». В этом чате карта не блокируется.'
  else:
   answer='Я не могу выполнить действие или проверить личную операцию в этом чате. Если вы заметили неизвестное списание или подозреваете мошенничество, заблокируйте карту в приложении и обратитесь в поддержку через «Помощь» → «Чат». Не сообщайте ПИН, CVV, пароль и коды из СМС.'
  return self.specialist(query,history,answer,'Инцидент или запрос действия требует защищённого канала.',sources,'Нужно проверить обстоятельства обращения через защищённый канал. Действия со счётом не выполнены.')
 def safe_fallback(self,query,history,reason):
  support=self.retriever.by_id['G06']
  return self.specialist(query,history,'Не могу надёжно ответить на этот вопрос автоматически. Обратитесь в поддержку через «Помощь» → «Чат» в приложении.',reason,[support],'Проверить вопрос по актуальным условиям и ответить клиенту.')
 def clarification_handoff(self,query,history):
  return self.specialist(query,history,'Вопрос остаётся неясным после нескольких уточнений. Обратитесь в поддержку через «Помощь» → «Чат» в приложении.','После трёх уточнений вопрос остался неоднозначным.',[self.retriever.by_id['G06']],'Пожалуйста, уточните запрос клиента и помогите с решением.')
 def ask(self,query,history=None):
  query=query.strip()[:1000]; history=normalize_history(history); started=time.monotonic(); deadline=started+25
  if not query:raise ValueError('Введите вопрос')
  usage={'embedding_tokens':0,'prompt_tokens':0,'completion_tokens':0,'total_tokens':0}
  kind=risk_kind(query)
  if GREETING.fullmatch(query):
   out=self.result('Здравствуйте! Помогу разобраться с картами, сбережениями, кредитом и переводами Демо-банка. О чём хотите узнать?',reason='Приветствие.',route='answer'); retrieval={'mode':'greeting_rule','retrieved_ids':[]}
  elif kind:
   out=self.handoff(query,history,kind); retrieval={'mode':'safety_rule','retrieved_ids':[]}
  elif needs_context(query,history):
   answer='Уточните, пожалуйста, что именно должно быть готово: карта, заявка или перевод?' if re.search(r'готов',query,re.I) else 'Уточните, пожалуйста, о каком продукте или действии вы хотите узнать подробнее?'
   out=self.result(answer,mode='clarify',clarifying_choices=['Карта','Заявка','Перевод'] if re.search(r'готов',query,re.I) else ['Карта','Счёт или вклад','Кредит'],reason='В вопросе нет названия предмета; нужен контекст.',route='clarify'); retrieval={'mode':'context_rule','retrieved_ids':[]}
  elif outside_bank_scope(query):
   out=self.result('Я помогаю с продуктами и обслуживанием Демо-банка. Переводить фразы на другие языки не могу.',sources=[self.retriever.by_id['G01']],reason='Вопрос вне области банковской помощи.',route='answer'); retrieval={'mode':'scope_rule','retrieved_ids':['G01']}
  elif asks_fabricated_terms(query):
   out=self.specialist(query,history,'Не буду придумывать комиссию или условие, которого нет в подтверждённых материалах. Уточните действующие условия через «Помощь» → «Чат» в приложении.','Запрошено неподтверждённое условие.',[self.retriever.by_id['G03'],self.retriever.by_id['G06']]); retrieval={'mode':'terms_rule','retrieved_ids':['G03','G06']}
  elif asks_personal_transfer_status(query):
   out=self.specialist(query,history,'Я не вижу ваш перевод и не могу подтвердить его зачисление. Проверьте сведения о переводе в приложении; если нужно подтвердить статус, обратитесь в поддержку через «Помощь» → «Чат».','Для проверки конкретного перевода нужны личные данные.',[self.retriever.by_id['G03'],self.retriever.by_id['G06']]); retrieval={'mode':'personal_rule','retrieved_ids':['G03','G06']}
  elif asks_personal_card_status(query):
   out=self.specialist(query,history,'Я не вижу состояние вашей карты. Откройте её в приложении и проверьте статус; если он неясен, обратитесь в поддержку через «Помощь» → «Чат».','Для проверки состояния конкретной карты нужны личные данные.',[self.retriever.by_id['G05'],self.retriever.by_id['G06']]); retrieval={'mode':'personal_rule','retrieved_ids':['G05','G06']}
  elif unsupported_scope(query):
   out=self.safe_fallback(query,history,unsupported_scope(query)); retrieval={'mode':'scope_rule','retrieved_ids':[]}
  elif asks_loan_early_repayment(query):
   out=self.result('Да. Кредит «План» можно полностью или частично погасить досрочно без отдельной комиссии. В приложении откройте «План» → «Досрочное погашение», выберите сумму и дату, пополните счёт и подтвердите заявку. Проценты рассчитываются за фактическое пользование долгом; в этом чате я не подаю заявку и не подтверждаю погашение.',sources=[self.retriever.by_id['L04']],reason='Ответ по условиям досрочного погашения кредита «План».',route='answer'); retrieval={'mode':'policy_rule','retrieved_ids':['L04']}
  elif asks_youth_card_comparison(query):
   out=self.result('Если подростку ещё нет 18 лет, из этих двух карт ему подходит «Старт»: она доступна с 6 до 17 лет и управляется родителем. «Каждый день» доступна с 18 лет. Обслуживание обеих карт бесплатное. У «Старта» нет кэшбэка, а «Каждый день» даёт 1% за обычные покупки, до 1000 ₽ в месяц. Выбор зависит от возраста и того, нужен ли родительский контроль.',sources=[self.retriever.by_id['J01'],self.retriever.by_id['D01']],reason='Сравнение двух подтверждённых продуктов.',route='answer'); retrieval={'mode':'comparison_rule','retrieved_ids':['J01','D01']}
  elif asks_deposit_extension(query,history):
   sources=[self.retriever.by_id[i] for i in ('V01','V02','V03')]
   out=self.result('Автоматического продления вклада «Горизонт» нет. В конце срока сумма и проценты поступают на выбранный счёт. Если захотите открыть новый вклад, проверьте его условия в приложении на тот момент: прежнюю ставку и другие условия я не могу обещать.',sources=sources,reason='Ответ по подтверждённым условиям вклада.',route='answer')
   retrieval={'mode':'policy_rule','retrieved_ids':[a['id'] for a in sources]}
  elif re.search(r'оператор|живого человека|позови.{0,15}сотрудник',query,re.I):
   out=self.specialist(query,history,'Подключаю специалиста для дальнейшей помощи.','Клиент запросил специалиста.',[self.retriever.by_id['G06']],'Клиент просит помощь специалиста.'); retrieval={'mode':'support_rule','retrieved_ids':['G06']}
  else:
   docs,retrieval=self.retriever.search(query,history,deadline); usage['embedding_tokens']=retrieval['embedding_tokens']
   try:
    if not self.remote.available:raise GigaError('not_configured')
    data,gen_usage=self.remote.generate(query,history,docs,deadline); usage.update({k:gen_usage.get(k,0) for k in ('prompt_tokens','completion_tokens','total_tokens')})
    ids=data['source_ids']; allowed={a['id']:a for a in docs}
    if any(i not in allowed for i in ids) or (data['route']=='answer' and not ids):raise GigaError('invalid_citations')
    sources=[allowed[i] for i in dict.fromkeys(ids)]
    if data['route'] in ('risk','human'):out=self.handoff(query,history,kind or 'other')
    elif data['route'] in ('clarify','unavailable') and clarification_streak(history)>=3:out=self.clarification_handoff(query,history)
    elif data['route'] in ('unavailable','personal'):out=self.safe_fallback(query,history,'Личные сведения или условия отсутствуют в подтверждённой базе.')
    else:out=self.result(data['answer'],mode='autopilot' if data['route']=='answer' else 'clarify',sources=sources,clarifying_choices=data['choices'][:3],reason='Ответ по условиям продуктов.' if data['route']=='answer' else 'Нужно уточнение или недоступны личные сведения.',route=data['route'])
   except (GigaError,KeyError,TypeError,ValueError) as exc:
    if clarification_streak(history)>=3:out=self.clarification_handoff(query,history)
    else:out=self.safe_fallback(query,history,'Автоматический ответ временно недоступен; сведения из поиска не проверены.')
    out['fallback_code']=str(exc) if isinstance(exc,GigaError) else 'invalid_response'
  out.update(retrieval=retrieval,usage=usage,latency_ms=round((time.monotonic()-started)*1000),kb_version=self.retriever.articles[0]['version'])
  return out
