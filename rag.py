"""Lexical RAG with grounded GigaChat evidence selection; optional vector index."""
import hashlib,json,re,time
from pathlib import Path
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from giga_client import GigaClient,GigaError
ROOT=Path(__file__).resolve().parent
NOTE='Учебный Демо-банк: все продукты и условия вымышлены. Реальные операции не выполняются.'
ALIASES={'debit':['каждый день','дебетов','дебетовая','зарплатн'],'credit':['запас','кредитн','кредитка'],'junior':['старт','детск','ребен','ребён','школьн','подрост'],'savings':['копилк','накопительн'],'deposit':['горизонт','вклад','депозит'],'loan':['план','кредит налич','потребительск']}
RISK=re.compile(r'мошенн|украл|украден|потерял.{0,25}карт|потеряна.{0,15}карт|потерял.{0,15}телефон|не\s+(?:узнаю|совершал|делал|проводил|подтверждал|покупал)|незнаком.{0,20}(?:плат|спис|перев)|неизвестн.{0,20}(?:плат|спис|перев)|без моего|подозрительн.{0,20}(?:спис|перев|операц)|(?:сообщил|передал|назвал|отправил).{0,30}(?:код|парол|cvv|пин)|(?:списали|сняли).{0,25}(?:дважды|два раза)|(?:переведи|отправь|перечисли)\s+(?:мне\s+)?(?:деньги|\d)|(?:оформи|выдай)\s+(?:мне\s+)?кредит|(?:заблокируй|разблокируй).{0,20}карт|(?:закрой|открой).{0,20}(?:сч[её]т|вклад)|stolen|fraud|unrecognised|unauthorized',re.I)
def normalize_history(history):
 return [{'role':x.get('role','user') if x.get('role','user') in ('user','assistant') else 'user','text':x['text'][:1200]} for x in (history or [])[-6:] if isinstance(x,dict) and isinstance(x.get('text'),str)]
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
 def handoff(self,query,history):
  source=self.retriever.by_id['G05']; support=self.retriever.by_id['G06']
  return self.result('Такой случай нужно проверить через официальный канал вашего банка. При потере карты или подозрении на мошенничество используйте блокировку в приложении и свяжитесь с банком через приложение или номер на обороте карты. Не сообщайте ПИН, CVV и коды из СМС. В этом демо я могу только подготовить описание для сотрудника; оператор не подключён и операции не выполнены.',mode='copilot',sources=[source,support],operator_summary='\n'.join([x['text'] for x in history if x['role']=='user'][-2:]+[query]),draft='Нужна проверка обстоятельств обращения через защищённый канал банка. Действия со счётом в демо не выполнены.',reason='Инцидент или запрос действия требует защищённого канала.')
 def ask(self,query,history=None):
  query=query.strip()[:1000]; history=normalize_history(history); started=time.monotonic(); deadline=started+25
  if not query:raise ValueError('Введите вопрос')
  usage={'embedding_tokens':0,'prompt_tokens':0,'completion_tokens':0,'total_tokens':0}
  if RISK.search(query) and not re.search(r'^(как |что делать,? если |что делать если |инструкция)',query.lower()):
   out=self.handoff(query,history); retrieval={'mode':'safety_rule','retrieved_ids':[]}
  elif re.search(r'оператор|живого человека|позови.{0,15}сотрудник',query,re.I):
   out=self.result('В учебном приложении: «Помощь» → «Чат» → попросить сотрудника. В этом прототипе реальной очереди операторов нет. Могу подготовить описание обращения; для связи используйте официальный канал вашего банка.',mode='copilot',sources=[self.retriever.by_id['G06']],operator_summary=query,draft='Клиент просит помощь сотрудника.',reason='Явный запрос сотрудника.'); retrieval={'mode':'support_rule','retrieved_ids':['G06']}
  else:
   docs,retrieval=self.retriever.search(query,history,deadline); usage['embedding_tokens']=retrieval['embedding_tokens']
   try:
    if not self.remote.available:raise GigaError('not_configured')
    data,gen_usage=self.remote.generate(query,history,docs,deadline); usage.update({k:gen_usage.get(k,0) for k in ('prompt_tokens','completion_tokens','total_tokens')})
    ids=data['source_ids']; allowed={a['id']:a for a in docs}
    if any(i not in allowed for i in ids) or (data['route']=='answer' and not ids):raise GigaError('invalid_citations')
    sources=[allowed[i] for i in dict.fromkeys(ids)]
    if data['route'] in ('risk','human'):out=self.handoff(query,history)
    else:out=self.result(data['answer'],mode='autopilot' if data['route']=='answer' else 'clarify',sources=sources,clarifying_choices=data['choices'][:3],reason='Ответ по найденным учебным статьям.' if data['route']=='answer' else 'Уточнение или ограничение доступных сведений.',route=data['route'])
   except (GigaError,KeyError,TypeError,ValueError) as exc:
    if retrieval['lexical_score']>=.10:
     out=self.result('Генерация ответа сейчас недоступна. Ниже найденная учебная статья:\n\n'+docs[0]['content'],mode='fallback',sources=[docs[0]],reason='Резервный режим: текст статьи без генерации.',fallback_code=str(exc) if isinstance(exc,GigaError) else 'invalid_response')
    else:out=self.result('В учебной базе не нашлось достаточно подходящих сведений. Уточните продукт: карта Каждый день, Запас, Старт, счёт Копилка, вклад Горизонт или кредит План.',mode='clarify',reason='Нет подтверждённого ответа.',route='unavailable')
  out.update(retrieval=retrieval,usage=usage,latency_ms=round((time.monotonic()-started)*1000),kb_version=self.retriever.articles[0]['version'])
  return out
