import streamlit as st
import pandas as pd
import numpy as np
from collections import Counter
from itertools import combinations
from math import factorial

st.set_page_config(page_title='Attribution Lab', layout='wide')
st.title('Attribution Lab — перша транзакція')
st.caption('Синтетичні дані. Атрибуція не є доказом причинності.')
seed = pd.DataFrame([
 ['A>B>C>D','0,7,20,30',30,1,100],['A>B>C','0,7,20',30,1,40],
 ['A>C>D','0,15,29',30,1,60],['A>B','0,7',30,0,120],
 ['A>C','0,15',30,0,80],['B>D','0,9',10,1,35],['D','0',1,1,20],
 ['D','0',1,0,60],['A','0',30,0,100],['B','0',30,0,50],
 ['C','0',30,0,40],['','',30,0,100],['','',30,1,10]],
 columns=['path','touch_days','end_day','converted','count'])
st.markdown('A — залучення; B — retention; C — відкриття картки; D — sales. Один рядок — група однакових шляхів. converted: 1 = транзакція, 0 = без транзакції. Дні від початку спостереження.')
df=st.data_editor(seed,num_rows='dynamic',use_container_width=True)
h=st.slider('Time Decay: напіврозпад (днів)',1,90,7)
rows=[]
try:
 for _,r in df.iterrows():
  p=[x.strip() for x in str(r.path).split('>') if x.strip()]; t=[float(x) for x in str(r.touch_days).split(',') if x.strip()]; e=float(r.end_day); y=int(r.converted); n=float(r['count'])
  assert len(p)==len(t) and y in (0,1) and n>=0 and all(0<=x<=e for x in t) and t==sorted(t) and not set(p)&{'START','CONV','NULL'}
  rows.append((p,t,e,y,n))
except Exception:
 st.error('Перевірте шляхи, дні, converted та count. START, CONV, NULL — зарезервовані назви.'); st.stop()
channels=sorted({c for p,*_ in rows for c in p}); k=len(channels)
if not k or k>12: st.error('Потрібно від 1 до 12 кампаній.'); st.stop()
# Time decay: each converting path contributes its count, split across touches.
td=Counter(); edges=Counter(); groups=Counter(); wins=Counter()
for p,t,e,y,n in rows:
 s=frozenset(p); groups[s]+=n; wins[s]+=n*y
 if y and p:
  a=np.array(t)-e; w=2**((a-a.max())/h); w=w/w.sum()
  for c,v in zip(p,w): td[c]+=n*v
 chain=['START']+p+['CONV' if y else 'NULL']
 for a,b in zip(chain,chain[1:]): edges[a,b]+=n
states=['START']+channels; P={}
for a in states:
 total=sum(v for (x,b),v in edges.items() if x==a)
 P[a]={b:v/total for (x,b),v in edges.items() if x==a and total}
# Removal redirects incoming probability to NULL (no bypass or renormalization).
def probability(removed=None):
 active=[a for a in states if a!=removed]; idx={a:i for i,a in enumerate(active)}
 Q=np.zeros((len(active),len(active))); b=np.zeros(len(active))
 for a in active:
  for dest,v in P[a].items():
   if dest=='CONV': b[idx[a]]+=v
   elif dest in idx: Q[idx[a],idx[dest]]+=v
 return float(np.linalg.lstsq(np.eye(len(active))-Q,b,rcond=None)[0][0])
base=probability(); removal={c:max(0,base-probability(c)) for c in channels}
# Shapley game: exact observed coalition conversion rate; missing coalitions
# use pooled rates of observed subsets (explicit demonstration assumption).
def value(s):
 if groups[s]>0:return wins[s]/groups[s]
 subs=[a for a in groups if a<=s and groups[a]>0]; den=sum(groups[a] for a in subs)
 return sum(wins[a] for a in subs)/den if den else 0.
sh={}
for c in channels:
 others=[x for x in channels if x!=c]; score=0.
 for j in range(k):
  for combo in combinations(others,j):
   s=frozenset(combo); score+=factorial(j)*factorial(k-j-1)/factorial(k)*(value(s|{c})-value(s))
 sh[c]=score

def pct(d,c):
 total=sum(d.values()); return 100*d.get(c,0)/total if total>0 else np.nan
out=pd.DataFrame({'Time Decay %':[pct(td,c) for c in channels],'Markov %':[pct(removal,c) for c in channels],'Shapley (п.п.)':[100*sh[c] for c in channels]},index=channels)
st.subheader('Порівняння моделей')
a,b,c=st.columns(3); a.metric('Конверсія Markov',f'{base:.2%}'); b.metric('Транзакції',f'{sum(y*n for p,t,e,y,n in rows):g}'); c.metric('Без рекламного контакту',f'{sum(y*n for p,t,e,y,n in rows if not p):g}')
st.dataframe(out.round(2)); st.bar_chart(out[['Time Decay %','Markov %']]); st.bar_chart(out[['Shapley (п.п.)']])
st.caption('Shapley показано в процентних пунктах приросту над базовою коаліцією, а не як частку транзакцій. Від’ємні внески збережено. Для зіставлення часток нижче використано нормалізацію лише якщо сумарний приріст позитивний.')
if sum(sh.values())>0:
 compare=out[['Time Decay %','Markov %']].copy(); compare['Shapley %']=[pct(sh,c) for c in channels]; st.bar_chart(compare); st.dataframe(compare.round(2))
else: compare=out
st.subheader('Діагностика Markov')
st.dataframe(pd.DataFrame({'Без кампанії':[probability(c) for c in channels],'Падіння, п.п.':[100*removal[c] for c in channels]},index=channels))
st.write('Матриця переходів',pd.DataFrame(P).T.fillna(0).round(3))
st.subheader('Діагностика Shapley')
st.write(f'Спостережувані коаліції: {sum(v>0 for v in groups.values())} із {2**k}. Базова конверсія: {value(frozenset()):.2%}. Конверсія повної коаліції: {value(frozenset(channels)):.2%}.')
st.warning('Для неспостережуваних коаліцій використано об’єднану конверсію спостережуваних підмножин. Це демонстраційна імпутація, не причинна оцінка. Markov використовує модель першого порядку та перенаправлення видалених переходів у неконверсію. Time Decay не розподіляє органічні транзакції. Для production потрібні повні конверсійні та неконверсійні шляхи, однаковий горизонт спостереження та контрольні експерименти.')
st.download_button('Експорт результатів CSV',compare.to_csv().encode('utf-8-sig'),'attribution_results.csv','text/csv')
