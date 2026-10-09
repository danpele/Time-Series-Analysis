r"""
build_chapter11.py -- Capitolul 11 (Foundation models pentru serii de timp), EN + RO dintr-o singură sursă
=========================================================================================================
Capitol de studiu individual: curs scris pentru a fi citit singur (exemple rezolvate, slide-uri de interpretare,
recapitulări, autoevaluare cu răspunsuri), fără seminar. Text ⟦english||română⟧; cifrele @{cheie} vin din
Quantlets/Ch_11/ch11_numbers.json (generate_all_charts.py). Nicio cifră nu este scrisă de mînă.
Material refolosit și reformulat: vechiul capitol 11 TSA (LLM și foundation models) și MFM, capitolul 14.
Ieșire:
  EN/Courses/chapter11_foundation_models_time_series.tex
  RO/Cursuri/capitol11_foundation_models_serii_timp.tex
Rulare:
  python3 Quantlets/Ch_11/generate_all_charts.py
  python3 latex/build_chapter11.py && python3 latex/tsa_build.py compile 11
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tsa_build import Deck, Values, table, photo   # noqa: E402
from tsa_build import items as _items   # noqa: E402
from ch11_common import REFS, T, bib, date, finalize, load, month   # noqa: E402


def items(*xs):
    return _items(*[x[0] if isinstance(x, tuple) and not x[1] else x for x in xs])


N = load()
V = Values()
D = Deck(11, 'lecture', refs=REFS)
C = 'https://commons.wikimedia.org/wiki/File:'


def ql(folder):
    return f'\\quantlet{{{folder.replace("_", chr(92) + "_")}}}{{\\qlurl{{{folder}}}}}'


def chart(title, fig, folder, bullets, h='0.6\\textheight', size='footnotesize'):
    body = (f'\\begin{{center}}\n\\includegraphics[width=0.97\\textwidth,height={h},keepaspectratio]{{{fig}.pdf}}\n'
            f'\\end{{center}}\n\\vspace{{-0.25cm}}\n' + items(*bullets) + '\n' + ql(folder))
    D.frame(title, body, size)


def chart_alone(title, fig, folder, bullets, parts=('',), h='0.88\\textheight', size='small', notes=()):
    """The chart alone on a frame, as wide as the slide allows, so that its text stays legible ((1/n)); its bullets, its
    interpretation (`notes`) and the Quantlet link on the last frame ((n/n)). `parts`: trim boxes (left bottom right top,
    in bp) that show one chart file in pieces, one piece per frame."""
    n = len(parts) + 1
    for i, trim in enumerate(parts, 1):
        g = (f'trim={trim},clip,' if trim else '') + f'width=0.96\\paperwidth,height={h},keepaspectratio'
        D.frame(f'{title} ({i}/{n})', f'\\centering\n\\makebox[\\textwidth][c]{{\\includegraphics[{g}]{{{fig}.pdf}}}}', size)
    D.frame(f'{title} ({n}/{n})', items(*bullets, *notes) + '\n' + ql(folder), size)


def interp(title, bullets, size='small'):
    D.frame(T(f'Interpreting {title[0]}', f'Interpretarea {title[1]}'), items(*bullets), size)


PH = {
    'dalles': ('ch11_google_dalles_2011.jpg', C + 'Google_Data_Center,_The_Dalles.jpg',
               T('Photo', 'Foto') + ': Visitor7 (2011); CC BY-SA 3.0; Wikimedia Commons'),
    'attention': ('ch11_attention_figure_2017.png', C + 'Attention_Is_All_You_Need_-_Encoder-decoder_Architecture.png',
                  T('Figure', 'Figura') + ': Vaswani et al.\\ (2017), Google; CC BY-SA 4.0; Wikimedia Commons'),
    'spheres': ('ch11_amazon_spheres_2018.jpg', C + 'Amazon_Spheres_2018.jpg',
                T('Photo', 'Foto') + ': Andy Li (2018); CC0; Wikimedia Commons'),
    'salesforce': ('ch11_salesforce_tower_2020.jpg', C + 'Salesforce_Tower_2020.jpg',
                   T('Photo', 'Foto') + ': Saggittarius A (2020); CC BY-SA 4.0; Wikimedia Commons'),
    'pdf': ('ch4_portile_de_fier_ii.jpg', C + 'Por\\%C8\\%9Bile_de_Fier_II_(01).jpg',
            T('Photo', 'Foto') + ': Nenea hartia (2016); CC BY-SA 4.0; Wikimedia Commons'),
}


def ph(key, cap, h='0.46\\textheight'):
    f, url, cred = PH[key]
    return photo(f, cap, url, cred, h=h)


def two(left, right, wl='0.38', wr='0.6'):
    return (f'\\begin{{columns}}[T]\n\\begin{{column}}{{{wl}\\textwidth}}\n{left}\n\\end{{column}}\n'
            f'\\begin{{column}}{{{wr}\\textwidth}}\n{right}\n\\end{{column}}\n\\end{{columns}}')


# =============================================================================
# CIFRE
# =============================================================================
P = V.put
EX = N['ex']
for i, (u, z, t) in enumerate(zip(EX['u'], EX['z'], EX['tok'])):
    P(f'u{i}', u, 1)
    P(f'z{i}', z, 3)
    V.raw(f'tok{i}', str(t))
V.raw('u.first', month(EX['u_dates'][0]))
V.raw('u.last', month(EX['u_dates'][-1]))
P('u.s', EX['s'], 2)
P('u.width', EX['width'], 4)
for i, w in enumerate(EX['att_w']):
    P(f'att{i}', w, 2)
P('att.out', EX['att_out'], 1)
P('pin.y', EX['pin_y'], 1)
for i, q in enumerate(EX['pin_q']):
    P(f'pin.q{i}', q, 1)
P('pin.1', EX['pin']['0.1'], 2)
P('pin.5', EX['pin']['0.5'], 2)
P('pin.9', EX['pin']['0.9'], 2)
P('pin.sum', 2 * sum(EX['pin'].values()) / 3, 3)
P('crps', EX['crps'], 3)
P('crps0', EX['crps0'], 3)
P('crpsw', EX['crps_wide'], 3)
P('ex.mae', EX['mase_mae'], 2)
P('ex.scale', EX['mase_scale'], 2)
P('ex.mase', EX['mase'], 1)

TK = N['tokens']
P('tk.s', TK['s'], 1)
P('tk.zmin', TK['zmin'], 2)
P('tk.zmax', TK['zmax'], 2)
V.raw('tk.ntok', str(TK['ntok']))
V.raw('tk.bins', str(TK['bins']))
PT = N['patching']
V.raw('pt.n', str(PT['n']))
V.raw('pt.P', str(PT['P']))
V.raw('pt.np', str(PT['patches']))

SE = N['series']
for k, v in SE.items():
    V.int(f'se.{k}.n', v['n'])
V.raw('se.load.end', date(SE['load']['end']))

FL = N['fan_load']
V.raw('fl.origin', date(FL['origin']))
P('fl.mn', FL['mase_naive'], 2)
P('fl.mb', FL['mase_bolt'], 2)
P('fl.cov', 100 * FL['cov_bolt'], 0)
for i, q in enumerate(FL['q_24']):
    P(f'fl.q{i}', q, 2)
P('fl.y', FL['y_24'], 2)
P('fl.p1', FL['pin_24']['0.1'], 3)
P('fl.p5', FL['pin_24']['0.5'], 3)
P('fl.p9', FL['pin_24']['0.9'], 3)
FI = N['fan_ip']
V.raw('fi.origin', month(FI['origin']))
V.raw('fi.end', month(FI['end']))
P('fi.me', FI['mase_ets'], 2)
P('fi.mb', FI['mase_bolt'], 2)
P('fi.mn', FI['mase_naive'], 2)
P('fi.ce', 100 * FI['cov_ets'], 0)
P('fi.cb', 100 * FI['cov_bolt'], 0)
FE = N['fan_eurron']
V.raw('fe.origin', date(FE['origin']))
V.raw('fe.end', date(FE['end']))
P('fe.last', FE['last'], 4)
P('fe.med', FE['med_bolt'], 4)
P('fe.wb', FE['width_bolt'], 3)
P('fe.wr', FE['width_rw'], 3)
P('fe.mb', FE['mae_bolt'], 4)
P('fe.mr', FE['mae_rw'], 4)

B = N['bench']
KEYS = ['load', 'ip', 'gdp', 'infl', 'unemp', 'eurron', 'retail']
MOD = {'ETS': 'ets', 'ARIMA': 'arima', 'Chronos-Bolt tiny': 'bt', 'Chronos-Bolt small': 'bs', 'Chronos-2': 'c2',
       'Seasonal naive': 'sn'}
for k in KEYS:
    V.raw(f'b.{k}.n', str(B[k]['_n']))
    for m, a in MOD.items():
        P(f'b.{k}.{a}.w', B[k][m]['rel_wql'], 2)
        P(f'b.{k}.{a}.m', B[k][m]['rel_mase'], 2)
        P(f'b.{k}.{a}.c', 100 * B[k][m]['cov80'], 0)
for m, a in MOD.items():
    P(f'gm.{a}.w', B['_gm'][m]['rel_wql'], 2)
    P(f'gm.{a}.m', B['_gm'][m]['rel_mase'], 2)
    P(f'gm.{a}.c', 100 * B['_gm'][m]['cov80'], 0)
    V.raw(f'gm.{a}.wins', str(B['_gm'][m]['wins']))
V.raw('nor', str(sum(B[k]['_n'] for k in KEYS)))
INF = N['infos']
V.raw('ar.load', f"({','.join(map(str, INF['load']['order']))})")
V.raw('ar.ip', f"({','.join(map(str, INF['ip']['order']))})({','.join(map(str, INF['ip']['sorder'][:3]))})" + '$_{12}$')

H = N['horizon']
for m, a in MOD.items():
    P(f'h.{a}.1', H[m]['h1'], 2)
    P(f'h.{a}.24', H[m]['h24'], 2)
    P(f'h.{a}.48', H[m]['h48'], 2)
    P(f'h.{a}.all', H[m]['all'], 2)

CX = N['context']
for m, a in (('Chronos-Bolt tiny', 'bt'), ('Chronos-Bolt small', 'bs'), ('Chronos-2', 'c2')):
    for c, v in zip(CX['contexts'], CX['mase'][m]):
        P(f'cx.{a}.{c}', v, 2)
P('cx.naive', CX['naive'], 2)
V.raw('cx.n', str(CX['n']))

PP = N['prepost']
for k, v in PP['Chronos-Bolt small'].items():
    for part in ('before', 'after'):
        P(f'pp.{k}.bs.{part}', v[f'Chronos-Bolt small|{part}'], 2)
        P(f'pp.{k}.ets.{part}', v[f'ETS|{part}'], 2)
        V.raw(f'pp.{k}.n.{part}', str(v[f'n|{part}']))
for k, v in PP['Chronos-2'].items():
    for part in ('before', 'after'):
        P(f'pq.{k}.c2.{part}', v[f'Chronos-2|{part}'], 2)
        P(f'pq.{k}.ets.{part}', v[f'ETS|{part}'], 2)
        V.raw(f'pq.{k}.n.{part}', str(v[f'n|{part}']))

SZ = N['size']
for m, a in (('Chronos-Bolt tiny', 'bt'), ('Chronos-Bolt small', 'bs'), ('Chronos-2', 'c2')):
    P(f'sz.{a}', SZ[m]['params'] / 1e6, 0)
    P(f'sec.{a}', 1000 * INF['load']['seconds'][m], 0)
P('sec.ets', 1000 * INF['load']['seconds']['ETS'], 0)

def de(v):
    v = int(round(v))
    return 'de ' if v >= 20 and (v % 100 >= 20 or v % 100 == 0) else ''


for key, v in (('nor', sum(B[k]['_n'] for k in KEYS)), ('se.load', SE['load']['n']), ('se.gdp', SE['gdp']['n']),
               ('sz.bt', SZ['Chronos-Bolt tiny']['params'] / 1e6), ('sz.bs', SZ['Chronos-Bolt small']['params'] / 1e6),
               ('sz.c2', SZ['Chronos-2']['params'] / 1e6), ('pt.np', PT['patches']), ('b.load', B['load']['_n']),
               ('cx', CX['n']), ('pq.b', PP['Chronos-2']['eurron']['n|before']), ('pq.a', PP['Chronos-2']['eurron']['n|after'])):
    V.raw(f'{key}.de', de(v))

P('sec.arima', 1000 * INF['load']['seconds']['ARIMA'], 0)

# =============================================================================
# DESCHIDERE
# =============================================================================
D.frame(T("Today's question and route", 'Întrebarea de azi și traseul'), items(
    (T('\\textbf{Question}: can one large model, pretrained on many series, forecast a series it has never seen as well as a model built for that series?',
       '\\textbf{Întrebarea}: poate un singur model mare, pre-antrenat pe multe serii, să prognozeze o serie pe care nu a văzut-o la fel de bine ca un model construit pentru acea serie?'),
     [T('until now (Chapters 0--10): one model per series, estimated on that series (ETS, ARIMA, GARCH, VAR)', 'pînă acum (capitolele 0--10): un model pentru fiecare serie, estimat pe acea serie (ETS, ARIMA, GARCH, VAR)'),
      T('Chapter 9: global machine-learning models, trained on the series of one data set', 'Capitolul 9: modele globale de învățare automată, antrenate pe seriile unui singur set de date')]),
    (T('\\textbf{Route} of the chapter (self-study)', '\\textbf{Traseul} capitolului (studiu individual)'),
     [T('the idea of a foundation model; Transformers and attention in brief', 'ideea de foundation model; Transformer-ul și atenția, pe scurt'),
      T('from numbers to tokens: scaling, quantisation, patching; the main model families', 'de la numere la tokeni: scalare, cuantizare, patching; principalele familii de modele'),
      T('probabilistic forecasts and their scores; an honest test on Romanian and US series', 'prognozele probabiliste și scorurile lor; un test corect pe serii din România și din SUA'),
      T('leakage and contamination; when these models help and when they do not', 'leakage-ul și contaminarea; situațiile în care aceste modele ajută și cele în care nu ajută')])))

D.frame(T('Learning outcomes', 'Rezultatele învățării'), items(
    T('Define a foundation model, zero-shot forecasting and fine-tuning', 'Definiți un foundation model, prognoza zero-shot și fine-tuning-ul'),
    T('Explain how a series becomes the input of a Transformer: mean scaling, quantisation, patching', 'Explicați cum devine o serie intrarea unui Transformer: scalarea prin medie, cuantizarea, patching-ul'),
    T('Describe Chronos, TimesFM, Moirai, Lag-Llama and TimeGPT and say how they differ', 'Descrieți Chronos, TimesFM, Moirai, Lag-Llama și TimeGPT și spuneți prin ce diferă'),
    T('Score a quantile forecast with the pinball loss, the CRPS, MASE and the coverage of an interval', 'Evaluați o prognoză cuantilă cu pierderea pinball, CRPS, MASE și acoperirea unui interval'),
    T('Run an open model on a CPU, compare it fairly with seasonal naive, ETS and ARIMA and recognise contamination', 'Rulați un model deschis pe un CPU, comparați-l corect cu prognoza sezonieră naivă, ETS și ARIMA și recunoașteți contaminarea')))

D.frame(T('Prerequisites for Today', 'Noțiuni necesare azi'), items(
    (T('Forecasting benchmarks and their evaluation (Chapters 0 and 4)', 'Reperele de prognoză și evaluarea lor (capitolele 0 și 4)'),
     [T('seasonal naive $\\hat y_{T+h} = y_{T+h-m}$: the forecast repeats the value of the same season one cycle earlier', 'prognoza sezonieră naivă $\\hat y_{T+h} = y_{T+h-m}$: prognoza repetă valoarea din același sezon al ciclului anterior'),
      T('$T$: the last observed period; $h$: the horizon; $m$: the season length (12 for monthly data, 168 for hourly data with a weekly cycle); the hat marks a forecast', '$T$: ultima perioadă observată; $h$: orizontul; $m$: lungimea sezonului (12 pentru date lunare, 168 pentru date orare cu ciclu săptămînal); notația $\hat{\ }$ indică o prognoză'),
      T('ETS; time-series cross-validation with a rolling origin; MAE, RMSE, MASE; the Diebold--Mariano test', 'ETS; validarea încrucișată cu origine mobilă; MAE, RMSE, MASE; testul Diebold--Mariano')]),
    (T('ARIMA and seasonal ARIMA (Chapters 3 and 4); machine learning basics (Chapter 9)', 'ARIMA și ARIMA sezonier (capitolele 3 și 4); noțiuni de bază de învățare automată (Capitolul 9)'),
     [T('training, validation and test samples; a neural network as a flexible function with many parameters', 'eșantioanele de antrenare, validare și test; o rețea neuronală ca funcție flexibilă cu mulți parametri'),
      T('global models: one model trained on many series', 'modelele globale: un model antrenat pe multe serii')]),
    (T('Quantiles: $q_\\tau$ is the value below which the variable falls with probability $\\tau$', 'Cuantilele: $q_\\tau$ este valoarea sub care variabila se află cu probabilitatea $\\tau$'),
     [T('$\\tau \\in (0, 1)$: the level; $q_{0.5}$ is the median, $q_{0.1}$ and $q_{0.9}$ bound a central 80\\% interval', '$\\tau \\in (0, 1)$: nivelul; $q_{0.5}$ este mediana, $q_{0.1}$ și $q_{0.9}$ delimitează un interval central de 80\\%')])), size='footnotesize')

# =============================================================================
# 1. IDEEA
# =============================================================================
D.section('The foundation-model idea', 'Ideea de foundation model')

D.frame(T('Three ways to forecast many series', 'Trei moduri de a prognoza multe serii'), items(
    (T('\\textbf{Local} models', 'Modele \\textbf{locale}'),
         [T('one model per series, estimated on that series only', 'cîte un model pentru fiecare serie, estimat doar pe acea serie'),
          T('ETS, ARIMA: few parameters, interpretable; 10\\,000 series mean 10\\,000 estimations', 'ETS, ARIMA: puțini parametri, interpretabile; 10\\,000 de serii înseamnă 10\\,000 de estimări')]),
    (T('\\textbf{Global} models (Chapter 9)', 'Modele \\textbf{globale} (Capitolul 9)'),
         [T('one model trained on all the series of a data set', 'un model antrenat pe toate seriile unui set de date'),
          T('example: DeepAR (\\refSalinas), a recurrent network trained on all products of a retailer', 'exemplu: DeepAR (\\refSalinas), o rețea recurentă antrenată pe toate produsele unui comerciant')]),
    (T('\\textbf{Foundation} models', 'Modele de tip \\textbf{foundation}'),
         [T('one model pretrained once on a very broad corpus of series from many domains', 'un model pre-antrenat o singură dată pe un corpus foarte larg de serii din multe domenii'),
          T('then used on new series and new data sets, without re-estimation', 'folosit apoi pe serii noi și seturi de date noi, fără reestimare'),
          T('the term comes from \\refBommasani: models trained on broad data at scale, adaptable to many tasks', 'termenul provine de la \\refBommasani: modele antrenate pe date largi, la scară mare, adaptabile la multe sarcini')])))

D.frame(T('Zero-shot, fine-tuning, in-context', 'Zero-shot, fine-tuning și prognoza din context'), items(
    (T('\\textbf{Context}', '\\textbf{Contextul}'),
         [T('the recent history $y_{T-C+1}, \\dots, y_T$ passed to the model at forecast time ($C$ = context length)', 'istoricul recent $y_{T-C+1}, \\dots, y_T$ transmis modelului în momentul prognozei ($C$ = lungimea contextului)')]),
    (T('\\textbf{Zero-shot} forecasting', 'Prognoza \\textbf{zero-shot}'),
         [T('the pretrained weights stay fixed', 'ponderile pre-antrenate rămîn fixe'),
          T('only the context of the new series is used', 'se folosește doar contextul seriei noi'),
          T('no parameter is estimated on the target series: forecasting takes a fraction of a second', 'niciun parametru nu se estimează pe seria-țintă: prognoza durează o fracțiune de secundă')]),
    (T('\\textbf{Fine-tuning}', '\\textbf{Fine-tuning}'),
         [T('further training of the pretrained weights on the target data', 'antrenarea suplimentară a ponderilor pre-antrenate pe datele-țintă'),
          T('can help on unusual series; needs a validation sample and more computation', 'poate ajuta la serii neobișnuite; cere un eșantion de validare și mai mult calcul')]),
    (T('The output is usually a set of \\textbf{quantiles} $\\hat q_{0.1}, \\dots, \\hat q_{0.9}$ for each future step, not a single number', 'Rezultatul este de obicei o mulțime de \\textbf{cuantile} $\\hat q_{0{,}1}; \\dots; \\hat q_{0{,}9}$ pentru fiecare pas viitor, nu un singur număr'), [])))

D.frame(T('Pretraining at scale', 'Pre-antrenarea la scară mare'), two(
    ph('dalles', T('A Google data centre, The Dalles, Oregon', 'Un centru de date Google, The Dalles, Oregon')),
    items(
        (T('Pretraining corpora mix public series from many domains', 'Corpusurile de pre-antrenare amestecă serii publice din multe domenii'),
         [T('energy, transport, retail, weather, web traffic, finance and economics', 'energie, transport, comerț, vreme, trafic web, finanțe și economie'),
          T('plus synthetic series: random mixtures of real series and series drawn from Gaussian processes (Chronos)', 'plus serii sintetice: amestecuri aleatoare de serii reale și serii generate din procese gaussiene (Chronos)')]),
        (T('Scale: TimesFM was pretrained on about $10^{11}$ time points (\\refDas)', 'Scara: TimesFM a fost pre-antrenat pe circa $10^{11}$ puncte (\\refDas)'), []),
        (T('Pretraining is done once, by the developer, on many graphics processors (GPU)', 'Pre-antrenarea se face o singură dată, de dezvoltator, pe multe procesoare grafice (GPU)'),
         [T('the user only runs the model: the small versions run on an ordinary CPU', 'utilizatorul doar rulează modelul: versiunile mici rulează pe un CPU obișnuit')])), wl='0.42', wr='0.56'))

D.recap(('The foundation-model idea', 'ideea de foundation model'), [
    T('Local, global and foundation models differ in what the model is trained on', 'Modelele locale, globale și de tip foundation diferă prin datele pe care se antrenează'),
    T('Zero-shot: fixed weights, only the context of the new series', 'Zero-shot: ponderi fixe, doar contextul seriei noi'),
    T('The output is a set of quantiles: a probabilistic forecast', 'Rezultatul este o mulțime de cuantile: o prognoză probabilistă')])

# =============================================================================
# 2. TRANSFORMER
# =============================================================================
D.section('Transformers in brief', 'Transformer-ul, pe scurt')

D.frame(T('The Transformer (Vaswani et al., 2017)', 'Transformer-ul (Vaswani et al., 2017)'), two(
    ph('attention', T('The original diagram: encoder (left), decoder (right)', 'Diagrama originală: encoder (stînga), decoder (dreapta)'), h='0.62\\textheight'),
    items(
        (T('\\refVaswani: a network for sequences built on \\textbf{attention}, without recurrence', '\\refVaswani: o rețea pentru secvențe construită pe \\textbf{atenție}, fără recurență'),
         [T('designed for translation; behind every large language model (LLM)', 'creată pentru traducere; stă la baza oricărui model mare de limbaj (LLM)')]),
        (T('Input: a sequence of \\textbf{tokens}, each turned into a vector (an embedding)', 'Intrarea: o secvență de \\textbf{tokeni}, fiecare transformat într-un vector (embedding)'), []),
        (T('\\textbf{Encoder--decoder} architecture', 'Arhitectura \\textbf{encoder--decoder}'),
             [T('the \\textbf{encoder} reads the whole input', '\\textbf{encoder-ul} citește toată intrarea'),
              T('the \\textbf{decoder} produces the output step by step', '\\textbf{decoder-ul} produce ieșirea pas cu pas'),
              T('\\textbf{decoder-only} models (GPT, TimesFM, Lag-Llama) forecast the next token from all previous ones', 'modelele \\textbf{doar cu decoder} (GPT, TimesFM, Lag-Llama) prognozează tokenul următor din toți cei anteriori')]),
        (T('Position information is added to each vector, since attention ignores order', 'Fiecărui vector i se adaugă informația despre poziție, deoarece atenția ignoră ordinea'), [])), wl='0.34', wr='0.64'))

D.frame(T('Self-attention (1/2): the formula', 'Self-attention (1/2): formula'), items(
    (T('Each input vector $x_i$ produces three vectors, by three linear maps learnt in training', 'Fiecare vector de intrare $x_i$ produce trei vectori, prin trei transformări liniare învățate la antrenare'),
     [T('a query $q_i = W_Q x_i$ (what position $i$ looks for), a key $k_i = W_K x_i$ (what position $i$ offers), a value $v_i = W_V x_i$ (what position $i$ passes on)', 'o interogare $q_i = W_Q x_i$ (ce caută poziția $i$), o cheie $k_i = W_K x_i$ (ce oferă poziția $i$), o valoare $v_i = W_V x_i$ (ce transmite poziția $i$)'),
      T('$W_Q$, $W_K$, $W_V$: weight matrices learnt in training, the same for every position', '$W_Q$, $W_K$, $W_V$: matrice de ponderi învățate la antrenare, aceleași pentru toate pozițiile')]),
    (T('The output at position $i$ is a weighted average of the values of all positions', 'Ieșirea în poziția $i$ este o medie ponderată a valorilor tuturor pozițiilor'),
     [T('$\\displaystyle \\text{out}_i = \\sum_j w_{ij} v_j, \\qquad w_{ij} = \\frac{\\exp(q_i^\\top k_j/\\sqrt{d})}{\\sum_l \\exp(q_i^\\top k_l/\\sqrt{d})}$', '$\\displaystyle \\text{out}_i = \\sum_j w_{ij} v_j, \\qquad w_{ij} = \\frac{\\exp(q_i^\\top k_j/\\sqrt{d})}{\\sum_l \\exp(q_i^\\top k_l/\\sqrt{d})}$')]),
    (T('Notation', 'Notațiile'),
     [T('$j, l$: indices that run over all input positions; $w_{ij}$: the weight that position $i$ gives to position $j$', '$j, l$: indici care parcurg toate pozițiile de intrare; $w_{ij}$: ponderea pe care poziția $i$ o dă poziției $j$'),
      T('$q_i^\\top k_j$: the scalar product of query and key, large when the two vectors point in the same direction (similarity score)', '$q_i^\\top k_j$: produsul scalar dintre interogare și cheie, mare cînd cei doi vectori au aceeași direcție (scorul de similaritate)'),
      T('$d$: the length of the key vectors; dividing by $\\sqrt{d}$ keeps the scores of moderate size', '$d$: lungimea vectorilor-cheie; împărțirea la $\\sqrt{d}$ menține scorurile la o mărime moderată'),
      T('the exponential makes every weight positive and the denominator makes the weights of position $i$ sum to 1', 'exponențiala face toate ponderile pozitive, iar numitorul face ca ponderile poziției $i$ să aibă suma 1')])), size='footnotesize')

D.frame(T('Self-attention (2/2): matrix form and intuition', 'Self-attention (2/2): forma matricială și intuiția'), items(
    (T('Matrix form of the same computation: $\\text{softmax}(QK^\\top/\\sqrt{d})\\,V$', 'Forma matricială a aceluiași calcul: $\\text{softmax}(QK^\\top/\\sqrt{d})\\,V$'),
     [T('$Q$, $K$, $V$: matrices whose rows are the vectors $q_i$, $k_i$, $v_i$; $QK^\\top$ holds all the scores $q_i^\\top k_j$', '$Q$, $K$, $V$: matrice ale căror linii sînt vectorii $q_i$, $k_i$, $v_i$; $QK^\\top$ conține toate scorurile $q_i^\\top k_j$'),
      T('softmax: applied row by row, it turns the scores into the weights $w_{ij}$ of the previous slide', 'softmax: aplicată pe fiecare linie, transformă scorurile în ponderile $w_{ij}$ de pe slide-ul anterior')]),
    (T('Intuition for series: today\'s step can look back at the same hour last week and give it a large weight', 'Intuiția pentru serii: pasul de azi se poate uita la aceeași oră de săptămîna trecută și îi poate da o pondere mare'),
     [T('the weights are learnt from data, not fixed in advance as the lags of an AR($p$)', 'ponderile sînt învățate din date, nu fixate dinainte, ca lagurile unui AR($p$)')]),
    (T('Cost: $n$ inputs need $n^2$ comparisons', 'Costul: $n$ intrări cer $n^2$ comparații'),
     [T('a context of 2048 values gives about 4 million comparisons in each attention layer', 'un context de 2048 de valori dă circa 4 milioane de comparații în fiecare strat de atenție')])))

D.frame(T('Worked example: attention weights', 'Exemplu rezolvat: ponderile atenției'), items(
    (T('One query and three keys; scores $q^\\top k_j/\\sqrt{d}$ = 2, 1, 0; values $v_j$ = 10, 20, 30', 'O interogare și trei chei; scorurile $q^\\top k_j/\\sqrt{d}$ = 2, 1, 0; valorile $v_j$ = 10, 20, 30'), []),
    (T('Step 1: exponentiate: $e^2$, $e^1$, $e^0$', 'Pasul 1: exponențiala: $e^2$, $e^1$, $e^0$'), []),
    (T('Step 2: divide by the sum: weights @{att0}, @{att1}, @{att2} (they sum to 1)', 'Pasul 2: împărțirea la sumă: ponderile @{att0}, @{att1}, @{att2} (au suma 1)'), []),
    (T('Step 3: output $= \\sum_j w_j v_j$ = @{att.out}', 'Pasul 3: ieșirea $= \\sum_j w_j v_j$ = @{att.out}'),
     [T('the highest score gets most of the weight, but every value contributes', 'scorul cel mai mare primește cea mai mare parte din pondere, dar fiecare valoare contribuie')]),
    T('A foundation model stacks many such layers, each with several attention heads', 'Un foundation model suprapune multe astfel de straturi, fiecare cu mai multe capete de atenție')))

D.recap(('Transformers', 'Transformer-ul'), [
    T('Attention: each step is a learnt weighted average of all steps', 'Atenția: fiecare pas este o medie ponderată, învățată, a tuturor pașilor'),
    T('Encoder--decoder (Chronos) and decoder-only (TimesFM, Lag-Llama) designs', 'Arhitecturi encoder--decoder (Chronos) și arhitecturi doar cu decoder (TimesFM, Lag-Llama)'),
    T('The cost grows with the square of the number of inputs', 'Costul crește cu pătratul numărului de intrări')])

# =============================================================================
# 3. TOKENI
# =============================================================================
D.section('From numbers to tokens', 'De la numere la tokeni')

D.frame(T('The problem: a language model reads words, a series has numbers', 'Problema: un model de limbaj citește cuvinte, o serie are numere'), items(
    (T('Series differ in units and levels: load in GW, an index around 100, an exchange rate around 5', 'Seriile diferă prin unități și niveluri: consumul în GW, un indice în jur de 100, un curs de schimb în jur de 5'),
     [T('one model for all of them needs a common scale', 'un singur model pentru toate are nevoie de o scală comună')]),
    (T('Three solutions used in practice', 'Trei soluții folosite în practică'),
     [T('\\textbf{scaling + quantisation}: each value becomes one token from a fixed vocabulary (Chronos)', '\\textbf{scalare + cuantizare}: fiecare valoare devine un token dintr-un vocabular fix (Chronos)'),
      T('\\textbf{patching}: groups of consecutive values become one input vector (TimesFM, Moirai, Chronos-Bolt)', '\\textbf{patching}: grupuri de valori consecutive devin un vector de intrare (TimesFM, Moirai, Chronos-Bolt)'),
      T('\\textbf{digits as text}: a general LLM reads the numbers as strings (\\refGruver)', '\\textbf{cifrele ca text}: un LLM general citește numerele ca șiruri de caractere (\\refGruver)')]),
    T('In all cases the forecasts are transformed back to the original units', 'În toate cazurile, prognozele se transformă înapoi în unitățile inițiale')))

D.frame(T('Mean scaling and quantisation (Chronos) (1/2)', 'Scalarea prin medie și cuantizarea (Chronos) (1/2)'), items(
    (T('\\textbf{Mean scaling}', '\\textbf{Scalarea prin medie}'),
         [T('each value of the context is divided by the mean absolute value of the context', 'fiecare valoare a contextului se împarte la media valorilor absolute ale contextului'),
          T('$\\tilde x_t = x_t / s, \\qquad s = \\frac{1}{C}\\sum_{t=1}^{C} |x_t|$', '$\\tilde x_t = x_t / s, \\qquad s = \\frac{1}{C}\\sum_{t=1}^{C} |x_t|$'),
          T('$x_t$: the $t$-th value of the context, $t = 1, \\dots, C$; $C$: the context length; $s$: the scale; $\\tilde x_t$: the scaled value', '$x_t$: a $t$-a valoare a contextului, $t = 1, \\dots, C$; $C$: lungimea contextului; $s$: scala; $\\tilde x_t$: valoarea scalată'),
          T('the scaled context has values of order one, whatever the units', 'contextul scalat are valori de ordinul unității, oricare ar fi unitățile')]),
    (T('\\textbf{Quantisation}', '\\textbf{Cuantizarea}'),
         [T('each scaled value is replaced by the number of a bin', 'fiecare valoare scalată este înlocuită cu numărul unui interval'),
          T('$[-15, 15]$ is divided into 4093 equal bins of width @{u.width}; the token of $\\tilde x_t$ is the number of the nearest bin centre', 'intervalul $[-15, 15]$ este împărțit în 4093 de intervale egale, de lățime @{u.width}; tokenul lui $\\tilde x_t$ este numărul celui mai apropiat centru'),
          T('plus special tokens (padding, end of sequence)', 'plus tokeni speciali (completare, sfîrșit de secvență)')])), size='footnotesize')

D.frame(T('Mean scaling and quantisation (Chronos) (2/2)', 'Scalarea prin medie și cuantizarea (Chronos) (2/2)'), items(
    (T('Training: a language model (T5) learns to predict the next token', 'Antrenarea: un model de limbaj (T5) învață să prognozeze tokenul următor'),
     [T('loss: the cross-entropy $-\\log \\hat p(z_{t+1})$, where $z_{t+1}$ is the true next token and $\\hat p(z_{t+1})$ the probability the model gave it', 'funcția de pierdere: cross-entropy $-\\log \\hat p(z_{t+1})$, unde $z_{t+1}$ este tokenul următor efectiv, iar $\\hat p(z_{t+1})$ probabilitatea atribuită lui de model'),
      T('it is 0 when that probability is 1 and grows as the probability falls', 'este 0 cînd această probabilitate este 1 și crește cînd probabilitatea scade')]),
    (T('Forecast in three steps', 'Prognoza, în trei pași'),
     [T('sample many token paths from the model', 'se eșantionează din model multe traiectorii de tokeni'),
      T('map each token back to its bin centre and multiply by $s$', 'fiecare token se înlocuiește cu centrul intervalului său și se înmulțește cu $s$'),
      T('read the quantiles of the sampled values at each future step', 'se citesc cuantilele valorilor eșantionate la fiecare pas viitor')]),
    T('Limit: scaled values outside $[-15, 15]$ cannot be produced', 'Limita: valorile scalate în afara intervalului $[-15, 15]$ nu pot fi produse')))

D.frame(T('Worked example: scaling and tokens', 'Exemplu rezolvat: scalarea și tokenii'), items(
    (T('Context: the Romanian unemployment rate (\\%, seasonally adjusted, Eurostat), @{u.first} to @{u.last}', 'Contextul: rata șomajului din România (\\%, ajustată sezonier, Eurostat), @{u.first}--@{u.last}'),
     [T('$x$ = @{u0}, @{u1}, @{u2}, @{u3}, @{u4}, @{u5}', '$x$ = @{u0}, @{u1}, @{u2}, @{u3}, @{u4}, @{u5}')]),
    (T('Step 1: scale $s$ = mean of $|x_t|$ = @{u.s}', 'Pasul 1: scala $s$ = media lui $|x_t|$ = @{u.s}'), []),
    (T('Step 2: scaled values $x_t/s$ = @{z0}, @{z1}, @{z2}, @{z3}, @{z4}, @{z5}', 'Pasul 2: valorile scalate $x_t/s$ = @{z0}, @{z1}, @{z2}, @{z3}, @{z4}, @{z5}'), []),
    (T('Step 3: nearest bin centres (0 to 4092): tokens @{tok0}, @{tok1}, @{tok2}, @{tok3}, @{tok4}, @{tok5}', 'Pasul 3: cele mai apropiate centre (de la 0 la 4092): tokenii @{tok0}, @{tok1}, @{tok2}, @{tok3}, @{tok4}, @{tok5}'),
     [T('equal values give equal tokens; a change of 0.1 points moves the token by several bins', 'valorile egale dau tokeni egali; o variație de 0,1 puncte mută tokenul cu mai multe intervale')]),
    T('The model never sees the level 6.6\\%: only the shape relative to the scale', 'Modelul nu vede niciodată nivelul de 6,6\\%: doar forma relativă la scală')))

chart_alone(T('Scaling and quantisation on real data', 'Scalarea și cuantizarea pe date reale'), 'tsa_ch11_tokens', 'TSA_ch11_tokens_scores', [
    T('Romanian industrial production (Eurostat, 2021 = 100), the last 96 months; scale $s$ = @{tk.s}', 'Producția industrială a României (Eurostat, 2021 = 100), ultimele 96 de luni; scala $s$ = @{tk.s}'),
    T('Right: the scaled context and its tokens on a coarse grid of @{tk.bins} bins (Chronos uses 4093)', 'Dreapta: contextul scalat și tokenii lui pe o grilă grosieră de @{tk.bins} de intervale (Chronos folosește 4093)')], notes=[
    T('After scaling, all values lie between @{tk.zmin} and @{tk.zmax}: a few per cent of the range $[-15, 15]$', 'După scalare, toate valorile sînt între @{tk.zmin} și @{tk.zmax}: cîteva procente din intervalul $[-15, 15]$'),
    T('On the coarse grid the 96 months use only @{tk.ntok} tokens: the April 2020 fall (the lockdown) is one separate token', 'Pe grila grosieră, cele 96 de luni folosesc doar @{tk.ntok} tokeni: căderea din aprilie 2020 (carantina) este un token separat'),
    T('With 4093 bins the grid is fine enough to keep the seasonal shape', 'Cu 4093 de intervale, grila este suficient de fină pentru a păstra forma sezonieră'),
    T('One extreme value inflates $s$ and shrinks all the other scaled values: outliers in the context matter', 'O valoare extremă mărește $s$ și micșorează toate celelalte valori scalate: valorile aberante din context contează')])

D.frame(T('Patching', 'Patching-ul'), items(
    (T('\\textbf{Patch}', '\\textbf{Patch}'),
         [T('a block of $P$ consecutive values, mapped to one input vector by a small network', 'un bloc de $P$ valori consecutive, transformat într-un vector de intrare de o rețea mică'),
          T('PatchTST (\\refNie): ``a time series is worth 64 words\'\'', 'PatchTST (\\refNie): „o serie de timp valorează cît 64 de cuvinte”')]),
    (T('Two gains', 'Două avantaje'),
     [T('fewer inputs: 2048 values with $P = 16$ give 128 patches, so attention needs $128^2$ instead of $2048^2$ comparisons', 'mai puține intrări: 2048 de valori cu $P = 16$ dau 128 de patch-uri, deci atenția cere $128^2$ în loc de $2048^2$ comparații'),
      T('each input carries a local shape (a slope, a peak), not a single noisy number', 'fiecare intrare poartă o formă locală (o pantă, un vîrf), nu un singur număr zgomotos')]),
    (T('Used by TimesFM (input patches of 32), Moirai (patch size by frequency), Chronos-Bolt and Chronos-2 (16)', 'Folosit de TimesFM (patch-uri de intrare de 32), Moirai (mărimea patch-ului după frecvență), Chronos-Bolt și Chronos-2 (16)'),
     [T('the output can also be a whole patch of future values at once, which is faster than token by token', 'și ieșirea poate fi un patch întreg de valori viitoare, ceea ce este mai rapid decît token cu token')])))

chart(T('Patching the Romanian electricity load', 'Patching-ul consumului de energie electrică al României'), 'tsa_ch11_patching', 'TSA_ch11_tokens_scores', [
    T('@{pt.n} hourly values (two weeks of June 2026, ENTSO-E) cut into @{pt.np} patches of @{pt.P} values', '@{pt.n} valori orare (două săptămîni din iunie 2026, ENTSO-E), împărțite în @{pt.np} @{pt.np.de}patch-uri de cîte @{pt.P} valori')],
    h='0.62\\textheight')

interp(('patching', 'patching-ului'), [
    T('A day has 24 hours, so a patch of 16 covers two thirds of a day: patches do not align with the daily cycle', 'O zi are 24 de ore, deci un patch de 16 acoperă două treimi dintr-o zi: patch-urile nu se aliniază cu ciclul zilnic'),
    T('Attention links patches: the morning rise of Monday can be matched with that of the previous Monday', 'Atenția leagă patch-urile: creșterea de dimineață de luni poate fi pusă în legătură cu cea de lunea trecută'),
    T('A weekly cycle (168 hours) is visible only if the context covers at least one week: 11 patches', 'Un ciclu săptămînal (168 de ore) este vizibil doar dacă contextul acoperă cel puțin o săptămînă: 11 patch-uri')])

D.recap(('From numbers to tokens', 'de la numere la tokeni'), [
    T('Mean scaling removes units; quantisation turns values into tokens (Chronos)', 'Scalarea prin medie elimină unitățile; cuantizarea transformă valorile în tokeni (Chronos)'),
    T('Patching groups values: shorter sequences, local shapes', 'Patching-ul grupează valorile: secvențe mai scurte, forme locale'),
    T('The forecast is transformed back to the original units', 'Prognoza se transformă înapoi în unitățile inițiale')])

# =============================================================================
# 4. FAMILII
# =============================================================================
D.section('The main model families', 'Principalele familii de modele')

D.frame(T('Chronos, Chronos-Bolt and Chronos-2 (Amazon)', 'Chronos, Chronos-Bolt și Chronos-2 (Amazon)'), two(
    ph('spheres', T('The Amazon Spheres, Seattle', 'Amazon Spheres, Seattle'), h='0.62\\textheight'),
    items(
        (T('\\textbf{Chronos} (\\refAnsari)', '\\textbf{Chronos} (\\refAnsari)'),
             [T('T5 encoder--decoder on scaled and quantised values', 'T5 encoder--decoder pe valori scalate și cuantizate'),
              T('samples token paths', 'eșantionează traiectorii de tokeni')]),
        (T('\\textbf{Chronos-Bolt} (November 2024)', '\\textbf{Chronos-Bolt} (noiembrie 2024)'),
             [T('patches of 16 values and a direct output of the quantiles 0.1, \\dots, 0.9', 'patch-uri de 16 valori și ieșirea directă a cuantilelor 0,1; \\dots; 0,9'),
              T('sizes used here: tiny @{sz.bt} and small @{sz.bs} million parameters; context up to 2048 values', 'dimensiunile folosite aici: tiny, @{sz.bt} @{sz.bt.de}milioane, și small, @{sz.bs} @{sz.bs.de}milioane de parametri; context de pînă la 2048 de valori')]),
        (T('\\textbf{Chronos-2} (\\refAnsariB, October 2025)', '\\textbf{Chronos-2} (\\refAnsariB, octombrie 2025)'),
             [T('@{sz.c2} million parameters, 21 quantile levels, covariates and groups of related series', '@{sz.c2} @{sz.c2.de}milioane de parametri, 21 de niveluri de cuantile, covariate și grupuri de serii înrudite')]),
        (T('Open weights (Apache-2.0 licence); Python package \\texttt{chronos-forecasting}; all run on a CPU', 'Ponderi deschise (licența Apache-2.0); pachetul Python \\texttt{chronos-forecasting}; toate rulează pe un CPU'), [])), wl='0.32', wr='0.66'), size='footnotesize')

D.frame(T('TimesFM (Google) and Lag-Llama', 'TimesFM (Google) și Lag-Llama'), items(
    (T('\\textbf{TimesFM} (\\refDas)', '\\textbf{TimesFM} (\\refDas)'),
         [T('decoder-only Transformer, input patches of 32, output patches of 128 values', 'Transformer doar cu decoder, patch-uri de intrare de 32 și de ieșire de 128 de valori'),
          T('pretrained on real series (Google Trends, Wikipedia page views) and synthetic ones; about 200 million parameters', 'pre-antrenat pe serii reale (Google Trends, accesări de pagini Wikipedia) și sintetice; circa 200 de milioane de parametri'),
          T('open weights (Apache-2.0); later versions add quantile outputs', 'ponderi deschise (Apache-2.0); versiunile ulterioare adaugă ieșiri cuantile')]),
    (T('\\textbf{Lag-Llama} (\\refRasul)', '\\textbf{Lag-Llama} (\\refRasul)'),
         [T('decoder-only, built on the Llama architecture of language models', 'doar cu decoder, construit pe arhitectura Llama a modelelor de limbaj'),
          T('inputs: lagged values chosen by frequency (for example 7 days, 12 months, 24 hours) plus calendar features', 'intrările: valorile seriei la laguri alese după frecvență (de exemplu 7 zile, 12 luni, 24 de ore), plus variabile de calendar'),
          T('output: the parameters of a Student-$t$ distribution for the next step; sampling gives paths', 'ieșirea: parametrii unei distribuții Student-$t$ pentru pasul următor; eșantionarea dă traiectorii'),
          T('small and open (Apache-2.0); meant to be fine-tuned', 'mic și deschis (Apache-2.0); gîndit pentru fine-tuning')])), size='footnotesize')

D.frame(T('Moirai (Salesforce) and TimeGPT', 'Moirai (Salesforce) și TimeGPT'), two(
    ph('salesforce', T('Salesforce Tower, San Francisco', 'Salesforce Tower, San Francisco'), h='0.62\\textheight'),
    items(
        (T('\\textbf{Moirai} (\\refWoo)', '\\textbf{Moirai} (\\refWoo)'),
             [T("``universal'' forecaster", 'model de prognoză „universal”'),
              T('any number of variables in one sequence; patch size chosen by frequency', 'orice număr de variabile într-o singură secvență; mărimea patch-ului aleasă după frecvență'),
              T('output: a mixture of distributions; corpus LOTSA, about 27 billion observations', 'ieșirea: un amestec de distribuții; corpusul LOTSA, circa 27 de miliarde de observații'),
              T('open weights under a non-commercial licence (CC BY-NC 4.0)', 'ponderi deschise, cu licență necomercială (CC BY-NC 4.0)')]),
        (T('\\textbf{TimeGPT} (\\refGarza)', '\\textbf{TimeGPT} (\\refGarza)'),
             [T('closed model, available only through a paid API', 'model închis, disponibil doar printr-un API cu plată'),
              T('the weights and the training data cannot be inspected; the version can change; the data leave your computer', 'ponderile și datele de antrenare nu pot fi inspectate; versiunea se poate schimba; datele pleacă de pe calculatorul dumneavoastră'),
              T('not used in this course: the results could not be reproduced', 'nu îl folosim în acest curs: rezultatele nu ar putea fi reproduse')])), wl='0.3', wr='0.68'), size='footnotesize')

D.frame(T('The families side by side', 'Comparația familiilor de modele'), table(
    'lllll', T('\\textbf{Model}', '\\textbf{Model}') + ' & ' + T('\\textbf{Architecture}', '\\textbf{Arhitectura}') + ' & '
    + T('\\textbf{Input}', '\\textbf{Intrarea}') + ' & ' + T('\\textbf{Output}', '\\textbf{Ieșirea}') + ' & ' + T('\\textbf{Access}', '\\textbf{Acces}'),
    ['Chronos (2024) & encoder--decoder (T5) & ' + T('tokens (bins)', 'tokeni (intervale)') + ' & ' + T('sampled paths', 'traiectorii eșantionate') + ' & Apache-2.0',
     'Chronos-Bolt (2024) & encoder--decoder & ' + T('patches of 16', 'patch-uri de 16') + ' & ' + T('9 quantiles', '9 cuantile') + ' & Apache-2.0',
     'Chronos-2 (2025) & encoder & ' + T('patches + covariates', 'patch-uri + covariate') + ' & ' + T('21 quantiles', '21 de cuantile') + ' & Apache-2.0',
     'TimesFM (2024) & decoder & ' + T('patches of 32', 'patch-uri de 32') + ' & ' + T('patches of 128', 'patch-uri de 128') + ' & Apache-2.0',
     'Moirai (2024) & encoder & ' + T('patches, any variables', 'patch-uri, orice variabile') + ' & ' + T('mixture', 'amestec') + ' & CC BY-NC 4.0',
     'Lag-Llama (2023) & decoder (Llama) & ' + T('lagged values', 'valori la laguri fixe') + ' & Student-$t$ & Apache-2.0',
     'TimeGPT (2023) & ' + T('not disclosed', 'nepublicată') + ' & -- & ' + T('points, intervals', 'puncte, intervale') + ' & ' + T('paid API', 'API cu plată')],
    size='scriptsize') + items(
    T('A further line of work feeds the digits to general LLMs (\\refGruver); \\refTan show that the language-model part adds cost but no accuracy', 'O altă direcție dă cifrele unor LLM-uri generale (\\refGruver); \\refTan arată că partea de model de limbaj adaugă cost, dar nu și precizie')),
    size='footnotesize')

D.recap(('The model families', 'familiile de modele'), [
    T('Chronos: tokens; Chronos-Bolt and Chronos-2: patches and direct quantiles', 'Chronos: tokeni; Chronos-Bolt și Chronos-2: patch-uri și cuantile directe'),
    T('TimesFM and Lag-Llama: decoder-only; Moirai: any number of variables', 'TimesFM și Lag-Llama: doar cu decoder; Moirai: orice număr de variabile'),
    T('Open weights make results reproducible; a paid API does not', 'Ponderile deschise fac rezultatele reproductibile; un API cu plată, nu')])

# =============================================================================
# 5. SCORURI
# =============================================================================
D.section('Probabilistic forecasts and their scores', 'Prognozele probabiliste și scorurile lor')

D.frame(T('Quantile forecasts', 'Prognozele cuantile'), items(
    (T('A \\textbf{probabilistic forecast} gives the distribution of $y_{T+h}$ given the data up to $T$', 'O \\textbf{prognoză probabilistă} dă distribuția lui $y_{T+h}$, condiționată de datele pînă la $T$'),
     [T('foundation models report it through quantiles $\\hat q_\\tau$ at levels $\\tau = 0.1, \\dots, 0.9$', 'foundation models o raportează prin cuantile $\\hat q_\\tau$ la nivelurile $\\tau = 0{,}1; \\dots; 0{,}9$')]),
    (T('Point forecast: the median $\\hat q_{0.5}$', 'Prognoza punctuală: mediana $\\hat q_{0{,}5}$'), []),
    (T('Central 80\\% interval: $[\\hat q_{0.1}, \\hat q_{0.9}]$; a \\textbf{fan chart} draws several such bands', 'Intervalul central de 80\\%: $[\\hat q_{0{,}1}, \\hat q_{0{,}9}]$; un \\textbf{fan chart} desenează mai multe astfel de benzi'), []),
    (T('Two qualities to check', 'Două calități de verificat'),
     [T('\\textbf{calibration}: the 80\\% interval contains about 80\\% of the outcomes', '\\textbf{calibrarea}: intervalul de 80\\% conține circa 80\\% din valorile observate'),
      T('\\textbf{sharpness}: among calibrated forecasts, narrower is better (\\refGR)', '\\textbf{precizia} (sharpness): dintre prognozele calibrate, cea mai îngustă este mai bună (\\refGR)')]),
    (T('For ETS and ARIMA we use Normal quantiles $\\hat q_\\tau = \\hat y_{T+h} + z_\\tau \\hat\\sigma_h$', 'Pentru ETS și ARIMA folosim cuantilele distribuției Normale, $\\hat q_\\tau = \\hat y_{T+h} + z_\\tau \\hat\\sigma_h$'),
     [T('$z_\\tau$: the $\\tau$ quantile of the standard Normal distribution ($z_{0.9} = 1.28$); $\\hat\\sigma_h$: the estimated standard deviation of the $h$-step forecast error', '$z_\\tau$: cuantila de nivel $\\tau$ a distribuției Normale standard ($z_{0.9} = 1.28$); $\\hat\\sigma_h$: abaterea standard estimată a erorii de prognoză la orizontul $h$')])), size='footnotesize')

D.frame(T('The pinball loss', 'Pierderea pinball'), items(
    (T('The \\textbf{pinball} (quantile) loss scores one forecast quantile $\\hat q_\\tau$ against the outcome $y$', 'Pierderea \\textbf{pinball} (cuantilă) evaluează o cuantilă prognozată $\\hat q_\\tau$ față de valoarea observată $y$'),
     [T('$L_\\tau(y, \\hat q_\\tau) = \\tau\\,(y - \\hat q_\\tau)$ if $y \\ge \\hat q_\\tau$, and $(1 - \\tau)(\\hat q_\\tau - y)$ otherwise', '$L_\\tau(y, \\hat q_\\tau) = \\tau\\,(y - \\hat q_\\tau)$ dacă $y \\ge \\hat q_\\tau$ și $(1 - \\tau)(\\hat q_\\tau - y)$ altfel')]),
    (T('Notation', 'Notațiile'),
     [T('$\\tau$: the quantile level; $\\hat q_\\tau$: the forecast quantile; $y$: the value observed later', '$\\tau$: nivelul cuantilei; $\\hat q_\\tau$: cuantila prognozată; $y$: valoarea observată ulterior')]),
    (T('Reading', 'Interpretarea'),
     [T('the loss is 0 when $\\hat q_\\tau = y$ and grows linearly with the distance between them', 'pierderea este 0 cînd $\\hat q_\\tau = y$ și crește liniar cu distanța dintre ele'),
      T('an outcome above the quantile is weighted by $\\tau$, one below by $1 - \\tau$', 'o valoare observată peste cuantilă primește ponderea $\\tau$, una sub cuantilă, ponderea $1 - \\tau$'),
      T('its expected value is smallest at the true quantile: reporting the true quantile is the best strategy', 'valoarea ei așteptată este minimă în cuantila adevărată: raportarea cuantilei adevărate este strategia optimă')])))

D.frame(T('The CRPS', 'Scorul CRPS'), items(
    (T('The \\textbf{CRPS} (continuous ranked probability score, \\refMW) scores a whole forecast distribution against the outcome $y$', '\\textbf{CRPS} (continuous ranked probability score, scorul de probabilitate continuu pe ranguri, \\refMW) evaluează o întreagă distribuție prognozată față de valoarea observată $y$'),
     [T('$\\text{CRPS}(F, y) = \\int (F(z) - \\mathbf{1}\\{z \\ge y\\})^2\\,dz = 2\\int_0^1 L_\\tau(y, F^{-1}(\\tau))\\,d\\tau$', '$\\text{CRPS}(F, y) = \\int (F(z) - \\mathbf{1}\\{z \\ge y\\})^2\\,dz = 2\\int_0^1 L_\\tau(y, F^{-1}(\\tau))\\,d\\tau$')]),
    (T('Notation', 'Notațiile'),
     [T('$F$: the forecast cumulative distribution function (CDF); $z$: the integration variable, running over all possible values', '$F$: funcția de repartiție prognozată; $z$: variabila de integrare, care parcurge toate valorile posibile'),
      T('$\\mathbf{1}\\{z \\ge y\\}$: the indicator, 1 if $z \\ge y$ and 0 otherwise (the CDF of a forecast that puts all mass on $y$)', '$\\mathbf{1}\\{z \\ge y\\}$: indicatorul, 1 dacă $z \\ge y$ și 0 altfel (funcția de repartiție a unei prognoze care pune toată masa în $y$)'),
      T('$F^{-1}(\\tau)$: the $\\tau$ quantile of $F$; the second form is the pinball loss averaged over all levels', '$F^{-1}(\\tau)$: cuantila de nivel $\\tau$ a lui $F$; a doua formă este pierderea pinball mediată pe toate nivelurile')]),
    (T('Reading', 'Interpretarea'),
     [T('$\\text{CRPS} \\ge 0$, in the units of $y$; 0 only for a forecast that puts all its mass on the outcome; smaller is better', '$\\text{CRPS} \\ge 0$, în unitățile lui $y$; 0 doar pentru o prognoză care pune toată masa în valoarea observată; mai mic înseamnă mai bine'),
      T('for a point forecast it equals $|y - \\hat y|$: the CRPS generalises the MAE', 'pentru o prognoză punctuală este $|y - \\hat y|$: CRPS generalizează MAE')])), size='footnotesize')

D.frame(T('The weighted quantile loss (WQL)', 'Pierderea cuantilă ponderată (WQL)'), items(
    (T('\\textbf{WQL} (weighted quantile loss)', '\\textbf{WQL} (weighted quantile loss, pierderea cuantilă ponderată)'),
         [T('the CRPS approximated with the 9 levels 0.1, \\dots, 0.9 and divided by the size of the series', 'CRPS aproximat cu cele 9 niveluri 0,1; \\dots; 0,9 și împărțit la mărimea seriei'),
          T('$\\text{WQL} = \\frac{1}{9}\\sum_{\\tau}\\frac{2\\sum_t L_\\tau(y_t, \\hat q_{\\tau,t})}{\\sum_t |y_t|}$', '$\\text{WQL} = \\frac{1}{9}\\sum_{\\tau}\\frac{2\\sum_t L_\\tau(y_t, \\hat q_{\\tau,t})}{\\sum_t |y_t|}$')]),
    (T('Notation', 'Notațiile'),
     [T('$\\tau$: runs over the 9 levels; $t$: runs over all forecast steps of all origins', '$\\tau$: parcurge cele 9 niveluri; $t$: parcurge toți pașii de prognoză de la toate originile'),
      T('$\\hat q_{\\tau,t}$: the forecast $\\tau$ quantile for step $t$; $y_t$: the outcome at step $t$', '$\\hat q_{\\tau,t}$: cuantila de nivel $\\tau$ prognozată pentru pasul $t$; $y_t$: valoarea observată la pasul $t$')]),
    (T('Reading', 'Interpretarea'),
     [T('the denominator $\\sum_t |y_t|$ removes the units: series with different levels can be compared', 'numitorul $\\sum_t |y_t|$ elimină unitățile: se pot compara serii cu niveluri diferite'),
      T('0 for a perfect forecast; smaller is better; used by Chronos and GIFT-Eval', '0 pentru o prognoză perfectă; mai mic înseamnă mai bine; folosit de Chronos și de GIFT-Eval')])))

chart(T('The pinball loss and the CRPS', 'Pierderea pinball și CRPS'), 'tsa_ch11_scores', 'TSA_ch11_tokens_scores', [
    T('Left: $L_\\tau$ against $y - \\hat q$ for $\\tau$ = 0.1, 0.5, 0.9. Right: the forecast N(0, 1), the outcome $y = 1$', 'Stînga: $L_\\tau$ în funcție de $y - \\hat q$ pentru $\\tau$ = 0,1; 0,5; 0,9. Dreapta: prognoza N(0, 1), valoarea observată $y = 1$'),
    T('N($\\mu$, $\\sigma^2$): the Normal distribution with mean $\\mu$ and variance $\\sigma^2$', 'N($\\mu$, $\\sigma^2$): distribuția Normală cu media $\\mu$ și varianța $\\sigma^2$')],
    h='0.58\\textheight')

interp(('the scores', 'scorurilor'), [
    T('At $\\tau = 0.9$ an outcome above the quantile costs 9 times more than one below: the 0.9 quantile should be exceeded only 10\\% of the time', 'La $\\tau = 0{,}9$, o valoare peste cuantilă costă de 9 ori mai mult decît una sub ea: cuantila de 0,9 ar trebui depășită doar în 10\\% din cazuri'),
    T('At $\\tau = 0.5$ the loss is half the absolute error: the median is scored like a point forecast', 'La $\\tau = 0{,}5$, pierderea este jumătate din eroarea absolută: mediana este evaluată ca o prognoză punctuală'),
    T('CRPS of N(0, 1): @{crps0} if $y = 0$, @{crps} if $y = 1$; N(0, 4) at $y = 1$ gives @{crpsw}: a wider band is penalised even when it covers the outcome', 'CRPS pentru N(0, 1): @{crps0} dacă $y = 0$, @{crps} dacă $y = 1$; N(0, 4) în $y = 1$ dă @{crpsw}: o bandă mai largă este penalizată chiar dacă acoperă valoarea'),
    T('The CRPS rewards calibration and sharpness at the same time', 'CRPS răsplătește în același timp calibrarea și precizia')])

D.frame(T('Worked example: pinball losses', 'Exemplu rezolvat: pierderile pinball'), items(
    (T('Forecast of next month\'s unemployment rate: $\\hat q_{0.1}$ = @{pin.q0}, $\\hat q_{0.5}$ = @{pin.q1}, $\\hat q_{0.9}$ = @{pin.q2}; outcome $y$ = @{pin.y}', 'Prognoza ratei șomajului pentru luna viitoare: $\\hat q_{0{,}1}$ = @{pin.q0}, $\\hat q_{0{,}5}$ = @{pin.q1}, $\\hat q_{0{,}9}$ = @{pin.q2}; valoarea observată $y$ = @{pin.y}'), []),
    (T('$\\tau = 0.1$: $y$ is above, loss $0.1 \\cdot (7.2 - 6.5)$ = @{pin.1}', '$\\tau = 0{,}1$: $y$ este deasupra, pierderea $0{,}1 \\cdot (7{,}2 - 6{,}5)$ = @{pin.1}'), []),
    (T('$\\tau = 0.5$: $y$ is above, loss $0.5 \\cdot (7.2 - 7.0)$ = @{pin.5}', '$\\tau = 0{,}5$: $y$ este deasupra, pierderea $0{,}5 \\cdot (7{,}2 - 7{,}0)$ = @{pin.5}'), []),
    (T('$\\tau = 0.9$: $y$ is below, loss $(1 - 0.9) \\cdot (7.6 - 7.2)$ = @{pin.9}', '$\\tau = 0{,}9$: $y$ este dedesubt, pierderea $(1 - 0{,}9) \\cdot (7{,}6 - 7{,}2)$ = @{pin.9}'), []),
    (T('Approximate CRPS with these three levels: $\\frac{2}{3}$ (sum of losses) = @{pin.sum} percentage points', 'CRPS aproximat cu aceste trei niveluri: $\\frac{2}{3}$ (suma pierderilor) = @{pin.sum} puncte procentuale'),
     [T('the outcome lies inside the 80\\% interval [6.5; 7.6]: it counts as covered in the coverage score (two slides on)', 'valoarea observată este în intervalul de 80\\% [6,5; 7,6]: în scorul de acoperire (două slide-uri mai departe) contează ca acoperită')])))

D.frame(T('Point accuracy: the MASE', 'Precizia punctuală: MASE'), items(
    (T('\\textbf{MASE} (mean absolute scaled error, \\refHK)', '\\textbf{MASE} (mean absolute scaled error, eroarea absolută medie scalată, \\refHK)'),
         [T('the MAE of the forecast divided by the in-sample MAE of the seasonal naive', 'MAE al prognozei împărțit la MAE în eșantion al prognozei sezoniere naive'),
          T('$\\text{MASE} = \\dfrac{\\frac1h\\sum_{j=1}^{h}|y_{T+j} - \\hat y_{T+j}|}{\\frac{1}{T-m}\\sum_{t=m+1}^{T}|y_t - y_{t-m}|}$', '$\\text{MASE} = \\dfrac{\\frac1h\\sum_{j=1}^{h}|y_{T+j} - \\hat y_{T+j}|}{\\frac{1}{T-m}\\sum_{t=m+1}^{T}|y_t - y_{t-m}|}$')]),
    (T('Notation', 'Notațiile'),
     [T('numerator: the mean absolute error of the forecasts $\\hat y_{T+j}$ over the $h$ steps after the origin $T$', 'numărătorul: eroarea absolută medie a prognozelor $\\hat y_{T+j}$ pe cei $h$ pași de după originea $T$'),
      T('denominator: the mean absolute error of the seasonal naive forecast $y_{t-m}$ inside the sample $t = m+1, \\dots, T$; $m$: the season length', 'numitorul: eroarea absolută medie a prognozei sezoniere naive $y_{t-m}$ în interiorul eșantionului, $t = m+1, \\dots, T$; $m$: lungimea sezonului')]),
    (T('Reading', 'Interpretarea'),
     [T('below 1: better than the in-sample seasonal naive; above 1: worse; free of units', 'sub 1: mai bun decît prognoza sezonieră naivă în eșantion; peste 1: mai slab; fără unități'),
      T('example: MAE @{ex.mae}, in-sample naive MAE @{ex.scale}, so MASE = @{ex.mase}', 'exemplu: MAE @{ex.mae}, MAE naiv în eșantion @{ex.scale}, deci MASE = @{ex.mase}')])), size='footnotesize')

D.frame(T('Coverage and relative scores', 'Acoperirea și scorurile relative'), items(
    (T('\\textbf{Coverage}', '\\textbf{Acoperirea}'),
         [T('the share of outcomes inside the 80\\% interval', 'proporția valorilor observate din intervalul de 80\\%'),
          T('$\\text{Cov} = \\frac1n\\sum_{i=1}^{n} \\mathbf{1}\\{\\hat q_{0.1}^{(i)} \\le y_i \\le \\hat q_{0.9}^{(i)}\\}$; $n$: the number of forecasts; $\\hat q_{\\tau}^{(i)}$: the quantile of forecast $i$; $\\mathbf{1}\\{\\cdot\\}$: 1 if the outcome is inside, 0 otherwise', '$\\text{Cov} = \\frac1n\\sum_{i=1}^{n} \\mathbf{1}\\{\\hat q_{0.1}^{(i)} \\le y_i \\le \\hat q_{0.9}^{(i)}\\}$; $n$: numărul de prognoze; $\\hat q_{\\tau}^{(i)}$: cuantila prognozei $i$; $\\mathbf{1}\\{\\cdot\\}$: 1 dacă valoarea observată este în interval, 0 altfel'),
          T('target 80\\%; 7 of 10 inside means 70\\%: intervals somewhat too narrow', 'ținta este 80\\%; 7 din 10 în interval înseamnă 70\\%: intervale puțin prea înguste')]),
    (T('\\textbf{Relative scores}', '\\textbf{Scorurile relative}'),
         [T("a model's MASE or WQL divided by that of the seasonal naive on the same origins", 'MASE sau WQL al unui model împărțit la cel al prognozei sezoniere naive, pe aceleași origini'),
          T('below 1: the model beats the seasonal naive', 'sub 1: modelul este mai bun decît prognoza sezonieră naivă'),
          T('averaged over the $S$ series with the geometric mean $(r_1 r_2 \\cdots r_S)^{1/S}$, $r_k$: the ratio for series $k$; so 2 and 0.5 cancel out', 'mediate pe cele $S$ serii cu media geometrică $(r_1 r_2 \\cdots r_S)^{1/S}$, $r_k$: raportul seriei $k$; astfel 2 și 0,5 se compensează')])))

D.recap(('Scores', 'scorurile'), [
    T('Pinball loss for each quantile; CRPS for the whole distribution; WQL as its scaled approximation', 'Pierderea pinball pentru fiecare cuantilă; CRPS pentru întreaga distribuție; WQL ca aproximare scalată'),
    T('MASE for the median; coverage for calibration', 'MASE pentru mediană; acoperirea pentru calibrare'),
    T('Report scores relative to the seasonal naive forecast', 'Raportați scorurile relativ la prognoza sezonieră naivă')])

# =============================================================================
# 6. ZERO-SHOT PE DATE REALE
# =============================================================================
D.section('Zero-shot forecasts on real data', 'Prognoze zero-shot pe date reale')

# the 4 x 2 grid is shown in two pieces: rows 1-2 (load, industrial production, GDP, inflation), then rows 3-4
chart_alone(T('Seven series for the test', 'Șapte serii pentru test'), 'tsa_ch11_series', 'TSA_ch11_tokens_scores', [
        (T('Romania (ENTSO-E)', 'România (ENTSO-E)'), [T('hourly electricity load', 'consumul orar de energie electrică')]),
        (T('Romania (Eurostat)', 'România (Eurostat)'),
         [T('industrial production', 'producția industrială'),
          T('real GDP, not seasonally adjusted', 'PIB-ul real, neajustat sezonier'),
          T('HICP inflation; unemployment rate', 'inflația IAPC; rata șomajului')]),
        (T('EUR/RON (BNR reference rate)', 'EUR/RON (cursul de referință BNR)'), []),
        (T('US retail sales (FRED RSXFSN)', 'Vînzările cu amănuntul din SUA (FRED RSXFSN)'), [])],
    parts=('0 305 0 0', '0 0 0 302'), notes=[
    T('Strong seasonality: load (daily and weekly cycles), industrial production and US retail sales (yearly), GDP (quarterly)', 'Sezonalitate puternică: consumul (cicluri zilnice și săptămînale), producția industrială și vînzările din SUA (anuale), PIB-ul (trimestrial)'),
    T('Close to a random walk: EUR/RON and the unemployment rate; noisy and short-lived: monthly inflation', 'Aproape de un mers aleator: EUR/RON și rata șomajului; zgomotoasă și cu memorie scurtă: inflația lunară'),
    T('Breaks: the 2020 pandemic (industrial production, GDP, retail); the 2022 inflation surge', 'Rupturi: pandemia din 2020 (producția industrială, PIB-ul, vînzările); creșterea inflației din 2022'),
    T('Lengths from @{se.gdp.n} quarters (GDP) to @{se.load.n} hours (load): very different amounts of context', 'Lungimi de la @{se.gdp.n} @{se.gdp.de}trimestre (PIB) la @{se.load.n} @{se.load.de}ore (consumul): cantități foarte diferite de context')])

chart_alone(T('Romanian electricity load, 48 hours ahead', 'Consumul de energie electrică al României, prognoză pe 48 de ore'), 'tsa_ch11_fan_load', 'TSA_ch11_zero_shot', [
    T('Chronos-Bolt small, zero-shot, context of 2048 hours (85 days); origin @{fl.origin}, 00:00; bands: 50\\% and 80\\% central intervals', 'Chronos-Bolt small, zero-shot, context de 2048 de ore (85 de zile); originea @{fl.origin}, ora 00:00; benzi: intervalele centrale de 50\\% și 80\\%')], notes=[
    T('The model reproduces the daily shape (night trough, evening peak) without any estimation on this series', 'Modelul reproduce forma zilnică (minimul de noapte, vîrful de seară) fără nicio estimare pe această serie'),
    T('MASE: Chronos-Bolt @{fl.mb}, seasonal naive @{fl.mn}: better than repeating last week', 'MASE: Chronos-Bolt @{fl.mb}, prognoza sezonieră naivă @{fl.mn}: mai bun decît repetarea săptămînii trecute'),
    T('But the actual load stays above the forecast all day: only @{fl.cov}\\% of the 48 hours fall in the 80\\% band', 'Dar consumul efectiv rămîne peste prognoză toată ziua: doar @{fl.cov}\\% din cele 48 de ore cad în banda de 80\\%'),
    T('The model has no temperature or calendar input: a change in the weather that the past 85 days do not announce cannot be anticipated', 'Modelul nu primește temperatura sau calendarul: o schimbare a vremii pe care ultimele 85 de zile nu o anunță nu poate fi anticipată'),
    T('One origin proves little: the systematic test follows', 'O singură origine dovedește puțin: urmează testul sistematic')])

D.frame(T('Worked example: scoring one hour', 'Exemplu rezolvat: evaluarea unei ore'), items(
    (T('Hour 24 of the load forecast (@{fl.origin}, 23:00): $\\hat q_{0.1}$ = @{fl.q0}, $\\hat q_{0.5}$ = @{fl.q1}, $\\hat q_{0.9}$ = @{fl.q2} GW; outcome @{fl.y} GW', 'Ora 24 a prognozei consumului (@{fl.origin}, ora 23:00): $\\hat q_{0{,}1}$ = @{fl.q0}, $\\hat q_{0{,}5}$ = @{fl.q1}, $\\hat q_{0{,}9}$ = @{fl.q2} GW; valoarea observată @{fl.y} GW'), []),
    (T('All three quantiles are below the outcome, so each loss is $\\tau\\,(y - \\hat q_\\tau)$', 'Toate cele trei cuantile sînt sub valoarea observată, deci fiecare pierdere este $\\tau\\,(y - \\hat q_\\tau)$'),
     [T('$\\tau = 0.1$: @{fl.p1}; \\ $\\tau = 0.5$: @{fl.p5}; \\ $\\tau = 0.9$: @{fl.p9} GW', '$\\tau = 0{,}1$: @{fl.p1}; \\ $\\tau = 0{,}5$: @{fl.p5}; \\ $\\tau = 0{,}9$: @{fl.p9} GW')]),
    (T('The outcome is above $\\hat q_{0.9}$: coverage indicator 0 for this hour', 'Valoarea observată este peste $\\hat q_{0{,}9}$: indicatorul de acoperire este 0 pentru această oră'), []),
    T('Averaged over many hours and origins, these losses give the WQL of the next section', 'Mediate pe multe ore și origini, aceste pierderi dau WQL din secțiunea următoare')))

chart_alone(T('Romanian industrial production, 12 months ahead', 'Producția industrială a României, prognoză pe 12 luni'), 'tsa_ch11_fan_ip', 'TSA_ch11_zero_shot', [
    T('Origin @{fi.origin}, forecasts to @{fi.end}; Chronos-Bolt small (zero-shot) and ETS (estimated on the series), both with 80\\% bands', 'Originea @{fi.origin}, prognoze pînă în @{fi.end}; Chronos-Bolt small (zero-shot) și ETS (estimat pe serie), ambele cu benzi de 80\\%')], notes=[
    T('Both models copy the seasonal pattern (the August and December dips) and keep the recent level', 'Ambele modele copiază tiparul sezonier (scăderile din august și decembrie) și păstrează nivelul recent'),
    T('MASE: Chronos-Bolt @{fi.mb}, ETS @{fi.me}, seasonal naive @{fi.mn}: on this origin the simplest forecast wins', 'MASE: Chronos-Bolt @{fi.mb}, ETS @{fi.me}, prognoza sezonieră naivă @{fi.mn}: pentru această origine cîștigă cea mai simplă prognoză'),
    T('The weak production of 2026 is not announced by the context; the 80\\% bands cover @{fi.cb}\\% (Chronos-Bolt) and @{fi.ce}\\% (ETS) of the months', 'Producția slabă din 2026 nu este anunțată de context; benzile de 80\\% acoperă @{fi.cb}\\% (Chronos-Bolt) și @{fi.ce}\\% (ETS) din luni'),
    T('The bands of Chronos-Bolt are narrower than those of ETS at long horizons', 'Benzile Chronos-Bolt sînt mai înguste decît cele ale ETS la orizonturi lungi')])

chart_alone(T('EUR/RON, 20 days ahead', 'EUR/RON, prognoză pe 20 de zile'), 'tsa_ch11_fan_eurron', 'TSA_ch11_zero_shot', [
    T('BNR reference rate; origin @{fe.origin}, last value before it @{fe.last}; Chronos-Bolt small against the random walk with Normal bands', 'Cursul de referință BNR; originea @{fe.origin}, ultima valoare dinaintea ei @{fe.last}; Chronos-Bolt small comparat cu mersul aleator cu benzi Normale')], notes=[
    T('The median after 20 days is @{fe.med}, practically the last value @{fe.last}: the model has learnt to behave like a random walk', 'Mediana după 20 de zile este @{fe.med}, practic ultima valoare, @{fe.last}: modelul a învățat să se comporte ca un mers aleator'),
    T('Width of the 80\\% band at 20 days: @{fe.wb} (Chronos-Bolt) and @{fe.wr} (random walk): almost the same uncertainty', 'Lățimea benzii de 80\\% la 20 de zile: @{fe.wb} (Chronos-Bolt) și @{fe.wr} (mersul aleator): aproape aceeași incertitudine'),
    T('MAE over the 20 days: @{fe.mb} against @{fe.mr}: a tie in practice', 'MAE pe cele 20 de zile: @{fe.mb} față de @{fe.mr}: practic egalitate'),
    T('A model cannot extract what the past does not contain: exchange rates stay close to unpredictable (Chapter 1)', 'Un model nu poate extrage ce trecutul nu conține: cursurile de schimb rămîn aproape imprevizibile (Capitolul 1)')])

D.recap(('Zero-shot forecasts', 'prognozele zero-shot'), [
    T('Without estimation, the model copies cycles and levels from the context', 'Fără estimare, modelul copiază ciclurile și nivelurile din context'),
    T('It cannot anticipate what the context does not show (heat, a downturn)', 'Nu poate anticipa ce contextul nu arată (căldura, o recesiune)'),
    T('On a random walk it returns a random walk', 'Pentru un mers aleator, prognoza lui coincide practic cu cea a mersului aleator')])

# =============================================================================
# 7. EVALUARE
# =============================================================================
D.section('An honest evaluation', 'O evaluare corectă')

D.frame(T('The test protocol', 'Protocolul testului'), items(
    (T('\\textbf{Rolling origin} (Chapter 4)', '\\textbf{Origine mobilă} (Capitolul 4)'),
         [T('at each origin every model forecasts $h$ steps from the data up to the origin only', 'la fiecare origine, fiecare model prognozează $h$ pași doar din datele pînă la origine'),
          T('load: $h$ = 48 hours, weekly origins July 2025 to June 2026; monthly series: $h$ = 12, monthly origins from 2016; GDP: $h$ = 4, from 2012; EUR/RON: $h$ = 20 days, every 20 days from 2016', 'consumul: $h$ = 48 de ore, origini săptămînale din iulie 2025 pînă în iunie 2026; seriile lunare: $h$ = 12, origini lunare din 2016; PIB-ul: $h$ = 4, din 2012; EUR/RON: $h$ = 20 de zile, la fiecare 20 de zile din 2016'),
          T('@{nor} origins in total; context: all the past, at most 2048 observations', '@{nor} @{nor.de}origini în total; contextul: tot trecutul, cel mult 2048 de observații')]),
    (T('\\textbf{Benchmarks}, re-estimated at every origin', '\\textbf{Reperele}, reestimate la fiecare origine'),
     [T('seasonal naive (the naive forecast for unemployment and EUR/RON)', 'prognoza sezonieră naivă (prognoza naivă pentru șomaj și EUR/RON)'),
      T('ETS with damped trend and additive seasonality; ARIMA with orders chosen once by AIC before the first origin (load: ARIMA@{ar.load} on weekly differences)', 'ETS cu trend amortizat și sezonalitate aditivă; ARIMA cu ordinele alese o singură dată după AIC, înainte de prima origine (consumul: ARIMA@{ar.load} pe diferențele săptămînale)')]),
    (T('\\textbf{Foundation models}, zero-shot', '\\textbf{Foundation models}, zero-shot'),
         [T('Chronos-Bolt tiny and small, Chronos-2', 'Chronos-Bolt tiny și small, Chronos-2')]),
    T('Same origins, same context, same horizon, same scores: MASE, WQL, coverage', 'Aceleași origini, același context, același orizont, aceleași scoruri: MASE, WQL, acoperirea')), size='footnotesize')

D.frame(T('Results: WQL relative to the seasonal naive', 'Rezultate: WQL relativ la prognoza sezonieră naivă'), table(
    'lcccccc', T('\\textbf{Series}', '\\textbf{Seria}') + ' & ' + T('\\textbf{Origins}', '\\textbf{Origini}') + ' & ETS & ARIMA & Bolt tiny & Bolt small & Chronos-2',
    [T(lab_en, lab_ro) + f' & @{{b.{k}.n}} & @{{b.{k}.ets.w}} & @{{b.{k}.arima.w}} & @{{b.{k}.bt.w}} & @{{b.{k}.bs.w}} & @{{b.{k}.c2.w}}'
     for k, lab_en, lab_ro in [('load', 'RO load (hourly)', 'Consumul RO (orar)'), ('ip', 'RO industrial production', 'Producția industrială RO'),
                               ('gdp', 'RO real GDP', 'PIB-ul real RO'), ('infl', 'RO inflation', 'Inflația RO'),
                               ('unemp', 'RO unemployment', 'Șomajul RO'), ('eurron', 'EUR/RON (daily)', 'EUR/RON (zilnic)'),
                               ('retail', 'US retail sales', 'Vînzările cu amănuntul SUA')]]
    + ['\\midrule ' + T('Geometric mean', 'Media geometrică') + ' & & @{gm.ets.w} & @{gm.arima.w} & @{gm.bt.w} & @{gm.bs.w} & @{gm.c2.w}'],
    size='scriptsize') + items(
    T('Below 1: better than the seasonal naive; each value is the WQL of the model divided by that of the seasonal naive on the same origins', 'Sub 1: mai bun decît prognoza sezonieră naivă; fiecare valoare este WQL al modelului împărțit la cel al prognozei sezoniere naive pe aceleași origini')),
    size='footnotesize')

chart(T('MASE and WQL by series and model', 'MASE și WQL pe serii și modele'), 'tsa_ch11_benchmark', 'TSA_ch11_benchmark', [
    T('Ratios to the seasonal naive forecast (log scale; the dashed line is 1)', 'Rapoarte față de prognoza sezonieră naivă (scală logaritmică; linia punctată este 1)')],
    h='0.66\\textheight')

interp(('the benchmark', 'testului'), [
    T('Hourly load: the foundation models win clearly (WQL ratio @{b.load.c2.w} for Chronos-2, @{b.load.bs.w} for Bolt small, @{b.load.arima.w} for ARIMA); ETS with a daily season only fails (@{b.load.ets.w})', 'Consumul orar: foundation models cîștigă clar (raportul WQL @{b.load.c2.w} pentru Chronos-2, @{b.load.bs.w} pentru Bolt small, @{b.load.arima.w} pentru ARIMA); ETS, doar cu sezonul zilnic, eșuează (@{b.load.ets.w})'),
    T('Romanian GDP (short quarterly series): ARIMA is best (@{b.gdp.arima.w}); Chronos-Bolt is worse than the seasonal naive (@{b.gdp.bs.w})', 'PIB-ul României (serie trimestrială scurtă): ARIMA este cel mai bun (@{b.gdp.arima.w}); Chronos-Bolt este mai slab decît prognoza sezonieră naivă (@{b.gdp.bs.w})'),
    T('Unemployment and EUR/RON: all models are close to 1: nothing beats the naive forecast by much', 'Șomajul și EUR/RON: toate modelele sînt aproape de 1: nimic nu bate cu mult prognoza naivă'),
    T('Geometric mean of WQL: Chronos-2 @{gm.c2.w}, ARIMA @{gm.arima.w}, Bolt small @{gm.bs.w}, Bolt tiny @{gm.bt.w}, ETS @{gm.ets.w}; Chronos-2 is best on @{gm.c2.wins} of 7 series', 'Media geometrică a WQL: Chronos-2 @{gm.c2.w}, ARIMA @{gm.arima.w}, Bolt small @{gm.bs.w}, Bolt tiny @{gm.bt.w}, ETS @{gm.ets.w}; Chronos-2 este cel mai bun pe @{gm.c2.wins} din 7 serii'),
    T('A well-specified ARIMA is as good as the small foundation models; a single winner on all series does not exist', 'Un ARIMA bine specificat este la fel de bun ca foundation models mici; un cîștigător unic pe toate seriile nu există')], size='footnotesize')

chart(T('Coverage of the 80\\% intervals', 'Acoperirea intervalelor de 80\\%'), 'tsa_ch11_coverage', 'TSA_ch11_benchmark', [
    T('Share of outcomes inside $[\\hat q_{0.1}, \\hat q_{0.9}]$, all origins and steps; the dashed line is the target 0.8', 'Proporția valorilor observate din $[\\hat q_{0{,}1}, \\hat q_{0{,}9}]$, pe toate originile și toți pașii; linia punctată este ținta de 0,8')],
    h='0.66\\textheight')

interp(('the coverage', 'acoperirii'), [
    T('Average coverage: seasonal naive @{gm.sn.c}\\%, ETS @{gm.ets.c}\\%, ARIMA @{gm.arima.c}\\%, Bolt small @{gm.bs.c}\\%, Chronos-2 @{gm.c2.c}\\%', 'Acoperirea medie: prognoza sezonieră naivă @{gm.sn.c}\\%, ETS @{gm.ets.c}\\%, ARIMA @{gm.arima.c}\\%, Bolt small @{gm.bs.c}\\%, Chronos-2 @{gm.c2.c}\\%'),
    T('Chronos-2 has the best WQL but intervals that are too narrow: sharp, not fully calibrated', 'Chronos-2 are cel mai bun WQL, dar intervale prea înguste: precis, dar nu complet calibrat'),
    T('US retail sales: every model covers well below 80\\% (from @{b.retail.c2.c}\\% for Chronos-2 to @{b.retail.bt.c}\\% for Bolt tiny): the 2020 shock and the 2021 jump surprise all of them', 'Vînzările din SUA: fiecare model acoperă mult sub 80\\% (de la @{b.retail.c2.c}\\% pentru Chronos-2 la @{b.retail.bt.c}\\% pentru Bolt tiny): șocul din 2020 și saltul din 2021 îi surprind pe toți'),
    T('ETS and ARIMA bands assume Normal errors and constant variance: too wide in calm periods, too narrow in crises', 'Benzile ETS și ARIMA presupun erori Normale și varianță constantă: prea largi în perioadele calme, prea înguste în crize')])

chart(T('Romanian load: error by horizon', 'Consumul României: eroarea în funcție de orizont'), 'tsa_ch11_horizon', 'TSA_ch11_benchmark', [
    T('Mean absolute error (GW) of the median forecast for each of the 48 hours ahead, over @{b.load.n} weekly origins', 'Eroarea absolută medie (GW) a prognozei mediane pentru fiecare dintre cele 48 de ore, pe @{b.load.n} @{b.load.de}origini săptămînale')],
    h='0.62\\textheight')

interp(('the error by horizon', 'erorii în funcție de orizont'), [
    T('First hour: ARIMA @{h.arima.1} and Chronos-2 @{h.c2.1} GW, far better than the seasonal naive @{h.sn.1}: the last observation matters most', 'Prima oră: ARIMA @{h.arima.1} și Chronos-2 @{h.c2.1} GW, mult mai bine decît prognoza sezonieră naivă, @{h.sn.1}: ultima observație contează cel mai mult'),
    T('After one day ARIMA is back to the naive level (@{h.arima.24} against @{h.sn.24} GW): its correction fades out', 'După o zi, ARIMA revine la nivelul prognozei naive (@{h.arima.24} față de @{h.sn.24} GW): corecția lui se stinge'),
    T('Chronos-2 keeps an advantage over the whole horizon (@{h.c2.48} GW at 48 hours against @{h.sn.48})', 'Chronos-2 își păstrează avantajul pe tot orizontul (@{h.c2.48} GW la 48 de ore, față de @{h.sn.48})'),
    T('ETS is good for a few hours and then the worst: a daily season cannot represent the weekend', 'ETS este bun cîteva ore, apoi cel mai slab: un sezon zilnic nu poate reprezenta weekendul')])

chart_alone(T('How much context is needed?', 'Lungimea necesară a contextului'), 'tsa_ch11_context', 'TSA_ch11_context_contamination', [
    T('MASE on Romanian load for contexts of 96 to 2048 hours; @{cx.n} weekly origins (the last half year); the dashed line is the seasonal naive', 'MASE pentru consumul României cu contexte de 96 pînă la 2048 de ore; @{cx.n} @{cx.de}origini săptămînale (ultima jumătate de an); linia punctată este prognoza sezonieră naivă')], notes=[
    T('With 96 hours (4 days) every model is worse than the seasonal naive: Chronos-Bolt small @{cx.bs.96}, Chronos-2 @{cx.c2.96}, naive @{cx.naive}', 'Cu 96 de ore (4 zile), fiecare model este mai slab decît prognoza sezonieră naivă: Chronos-Bolt small @{cx.bs.96}, Chronos-2 @{cx.c2.96}, naiv @{cx.naive}'),
    T('From 192 hours (more than one week) on, the weekly cycle is in the context and the error drops', 'De la 192 de ore (peste o săptămînă), ciclul săptămînal intră în context și eroarea scade'),
    T('Chronos-2 keeps improving up to 1344 hours (@{cx.c2.1344}); Chronos-Bolt flattens at 672 hours (@{cx.bs.672})', 'Chronos-2 se îmbunătățește pînă la 1344 de ore (@{cx.c2.1344}); Chronos-Bolt se stabilizează de la 672 de ore (@{cx.bs.672})'),
    T('Rule: give the model several full seasonal cycles; a zero-shot model only repeats what it sees', 'Regula: dați modelului mai multe cicluri sezoniere complete; un model zero-shot doar repetă ce vede')])

chart(T('Size and accuracy', 'Dimensiunea și precizia'), 'tsa_ch11_size', 'TSA_ch11_context_contamination', [
    T('Geometric mean over the seven series of the WQL ratio to the seasonal naive, against the number of parameters', 'Media geometrică, pe cele șapte serii, a raportului WQL față de prognoza sezonieră naivă, în funcție de numărul de parametri')],
    h='0.62\\textheight')

interp(('size and cost', 'dimensiunii și a costului'), [
    T('Parameters: Bolt tiny @{sz.bt} million, Bolt small @{sz.bs} million, Chronos-2 @{sz.c2} million; ETS and ARIMA: a few per series', 'Parametri: Bolt tiny @{sz.bt} @{sz.bt.de}milioane, Bolt small @{sz.bs} @{sz.bs.de}milioane, Chronos-2 @{sz.c2} @{sz.c2.de}milioane; ETS și ARIMA: cîțiva pe serie'),
    T('Accuracy improves with size, but ARIMA (@{gm.arima.w}) sits between the small and the large model', 'Precizia crește cu dimensiunea, dar ARIMA (@{gm.arima.w}) se situează între modelul mic și cel mare'),
    T('Time per load forecast on a laptop CPU: Bolt small @{sec.bs} ms, Chronos-2 @{sec.c2} ms, ARIMA @{sec.arima} ms, ETS @{sec.ets} ms', 'Timpul pentru o prognoză a consumului pe procesorul unui laptop: Bolt small @{sec.bs} ms, Chronos-2 @{sec.c2} ms, ARIMA @{sec.arima} ms, ETS @{sec.ets} ms'),
    T('Zero-shot inference is cheap; the cost was paid once, in pretraining', 'Inferența zero-shot este ieftină; costul a fost plătit o singură dată, la pre-antrenare')])

D.frame(T('Landmark benchmarks', 'Benchmark-uri de referință'), items(
    (T('\\textbf{M3} (\\refMH)', '\\textbf{M3} (\\refMH)'),
         [T('3003 series', '3003 serii'),
          T('sophisticated methods did not beat simple ones on average', 'metodele sofisticate nu le-au bătut în medie pe cele simple')]),
    (T('\\textbf{M4} (\\refMSA)', '\\textbf{M4} (\\refMSA)'),
         [T('100\\,000 series', '100\\,000 de serii'),
          T('combinations and a hybrid of exponential smoothing and a neural network won', 'au cîștigat combinațiile și un hibrid de netezire exponențială și rețea neuronală')]),
    (T('\\textbf{Monash archive} (\\refMonash)', '\\textbf{Arhiva Monash} (\\refMonash)'),
         [T('public data sets with fixed horizons and benchmark results', 'seturi publice de date cu orizonturi fixe și rezultate de referință')]),
    (T('\\textbf{GIFT-Eval} (\\refAksu)', '\\textbf{GIFT-Eval} (\\refAksu)'),
         [T('many domains and frequencies, MASE and CRPS, a public leaderboard of foundation models', 'multe domenii și frecvențe, MASE și CRPS, un clasament public al foundation models'),
          T('separates the test data from a pretraining corpus, to limit contamination', 'separă datele de test de un corpus de pre-antrenare, pentru a limita contaminarea')]),
    (T('Critical studies: simple linear models beat early Transformer forecasters (\\refZeng); LLM parts add no accuracy (\\refTan)', 'Studii critice: modele liniare simple au bătut primele modele de prognoză cu Transformer (\\refZeng); componentele LLM nu adaugă precizie (\\refTan)'), [])),
    size='footnotesize')

D.frame(T('When foundation models help, and when not', 'Utilitatea foundation models: avantaje și limite'), items(
    (T('\\textbf{Help}', '\\textbf{Ajută}'),
     [T('many series, little time per series; strong and regular seasonality; long contexts (load, sales, traffic)', 'multe serii, puțin timp pentru fiecare; sezonalitate puternică și regulată; contexte lungi (consum, vînzări, trafic)'),
      T('a quick, strong benchmark before building a dedicated model; probabilistic output at no extra cost', 'un reper rapid și puternic înaintea unui model dedicat; ieșire probabilistă fără cost suplimentar')]),
    (T('\\textbf{Do not help}', '\\textbf{Nu ajută}'),
     [T('series close to a random walk (exchange rates, returns): nothing to learn from the past', 'serii apropiate de un mers aleator (cursuri de schimb, randamente): nu este nimic de învățat din trecut'),
      T('short series (a few dozen quarters) where a careful ARIMA is better', 'serii scurte (cîteva zeci de trimestre), unde un ARIMA atent specificat este mai bun'),
      T('when drivers matter (temperature, prices, policy) and are not in the context', 'cînd contează factorii explicativi (temperatura, prețurile, politicile), iar aceștia nu sînt în context'),
      T('when the model must be explained parameter by parameter, or a causal question is asked', 'cînd modelul trebuie explicat parametru cu parametru sau se pune o întrebare cauzală')]),
    T('Always compare with seasonal naive, ETS and ARIMA on the same origins; test differences with Diebold--Mariano (\\refDM)', 'Comparați întotdeauna cu prognoza sezonieră naivă, ETS și ARIMA pe aceleași origini; testați diferențele cu Diebold--Mariano (\\refDM)')),
    size='footnotesize')

D.recap(('The evaluation', 'evaluarea'), [
    T('Chronos-2 is best on average, especially for hourly load; ARIMA is close and wins on GDP', 'Chronos-2 este cel mai bun în medie, mai ales pentru consumul orar; ARIMA este aproape și cîștigă la PIB'),
    T('On random-walk-like series no model beats the naive forecast', 'Pe seriile apropiate de un mers aleator, niciun model nu bate prognoza naivă'),
    T('Intervals of foundation models tend to be too narrow; context must cover the seasons', 'Intervalele foundation models tind să fie prea înguste; contextul trebuie să acopere sezoanele')])

# =============================================================================
# 8. LEAKAGE ȘI CONTAMINARE
# =============================================================================
D.section('Leakage and contamination', 'Leakage-ul și contaminarea')

D.frame(T('Leakage in your own pipeline', 'Leakage-ul în propriul cod'), items(
    (T('\\textbf{Leakage}', '\\textbf{Leakage}'),
         [T('information from the test period reaches the forecasts (\\refHew)', 'informația din perioada de test ajunge în prognoze (\\refHew)')]),
    (T('Typical mistakes', 'Greșeli tipice'),
     [T('scaling or standardising with the mean of the whole sample', 'scalarea sau standardizarea cu media întregului eșantion'),
      T('choosing ARIMA orders, the context length or the model version after looking at test errors', 'alegerea ordinelor ARIMA, a lungimii contextului sau a versiunii modelului după ce ați văzut erorile de test'),
      T('using revised data (GDP) as if it had been known at the origin', 'folosirea datelor revizuite (PIB) ca și cum ar fi fost cunoscute la origine'),
      T('random train/test splits of a time series (Chapter 9)', 'împărțirea aleatoare a unei serii de timp în antrenare și test (Capitolul 9)')]),
    (T('In this chapter: each context ends at the origin; ARIMA orders were fixed before the first origin; Chronos scales by the context only', 'În acest capitol: fiecare context se termină la origine; ordinele ARIMA au fost fixate înaintea primei origini; Chronos scalează doar cu contextul'),
     [T('remaining issue: Eurostat GDP and production are the latest vintage, not real-time data', 'problema rămasă: PIB-ul și producția de la Eurostat sînt ultima versiune revizuită, nu date în timp real')])), size='footnotesize')

D.frame(T('Contamination of benchmarks', 'Contaminarea benchmark-urilor'), items(
    (T('\\textbf{Contamination}', '\\textbf{Contaminarea}'),
         [T('the test series (or series highly correlated with it) were in the pretraining corpus', 'seria de test (sau serii puternic corelate cu ea) a fost în corpusul de pre-antrenare'),
          T('the score then measures memory, not forecasting: an optimistic bias', 'scorul măsoară atunci memoria, nu prognoza: o distorsiune optimistă')]),
    (T('Chronos reports two benchmarks: in-domain (data sets used in training) and zero-shot (data sets not used) (\\refAnsari)', 'Chronos raportează două benchmark-uri: în domeniu (seturi folosite la antrenare) și zero-shot (seturi nefolosite) (\\refAnsari)'),
     [T('public series such as M4 or US macro data are likely in many corpora; FRED is public, so is Eurostat', 'serii publice precum M4 sau datele macro din SUA sînt probabil în multe corpusuri; FRED este public, la fel și Eurostat')]),
    (T('Remedies', 'Remedii'),
     [T('test only on data published after the release date of the weights', 'testați doar pe date publicate după data lansării ponderilor'),
      T('live benchmarks: forecasts submitted before the outcome exists', 'benchmark-uri live: prognoze depuse înainte ca rezultatul să existe'),
      T('report the training corpus and the release date of every model you use', 'raportați corpusul de antrenare și data lansării fiecărui model folosit')])), size='footnotesize')

chart(T('A contamination check: before and after the release', 'O verificare a contaminării: înainte și după lansare'), 'tsa_ch11_prepost', 'TSA_ch11_context_contamination', [
    T('WQL ratio to the seasonal naive for origins before (hatched) and after (solid) the release of the weights; ETS on the same origins as control', 'Raportul WQL față de prognoza sezonieră naivă pentru originile dinainte (hașurat) și de după (plin) lansarea ponderilor; ETS pe aceleași origini, ca termen de comparație')],
    h='0.66\\textheight')

interp(('the contamination check', 'verificării contaminării'), [
    T('Chronos-2 on EUR/RON: @{pq.eurron.c2.before} before its release (@{pq.eurron.n.before} origins), @{pq.eurron.c2.after} after (@{pq.eurron.n.after} origins): the advantage disappears', 'Chronos-2 pe EUR/RON: @{pq.eurron.c2.before} înainte de lansare (@{pq.eurron.n.before} @{pq.b.de}origini), @{pq.eurron.c2.after} după (@{pq.eurron.n.after} @{pq.a.de}origini): avantajul dispare'),
    T('Chronos-2 on load: @{pq.load.c2.before} before, @{pq.load.c2.after} after: the gain is real, also on data it could not have seen', 'Chronos-2 pe consum: @{pq.load.c2.before} înainte, @{pq.load.c2.after} după: cîștigul este real, și pe date pe care nu le-a putut vedea'),
    T('Chronos-Bolt small, US retail: @{pp.retail.bs.before} before, @{pp.retail.bs.after} after, while ETS is stable (@{pp.retail.ets.before}, @{pp.retail.ets.after})', 'Chronos-Bolt small, vînzările din SUA: @{pp.retail.bs.before} înainte, @{pp.retail.bs.after} după, în timp ce ETS este stabil (@{pp.retail.ets.before}; @{pp.retail.ets.after})'),
    T('Caution: 3 to 35 origins after the release are few; the after period also differs (2025--2026); a drop is a warning, not a proof', 'Atenție: 3--35 de origini după lansare sînt puține; și perioada de după diferă (2025--2026); o scădere este un semnal de alarmă, nu o dovadă'),
    T('Chronos-2 was released in October 2025: its monthly results above use only origins before that date', 'Chronos-2 a fost lansat în octombrie 2025: rezultatele lui lunare de mai sus folosesc doar origini dinaintea acestei date')], size='footnotesize')

D.recap(('Leakage and contamination', 'leakage-ul și contaminarea'), [
    T('Leakage is under your control: use only data available at each origin', 'Leakage-ul este sub controlul dumneavoastră: folosiți doar datele disponibile la fiecare origine'),
    T('Contamination is not: the pretraining corpus may contain your series', 'Contaminarea nu este: corpusul de pre-antrenare poate conține seria dumneavoastră'),
    T('Check results on data released after the model', 'Verificați rezultatele pe date publicate după lansarea modelului')])

# =============================================================================
# 9. AI
# =============================================================================
D.section('Possible contribution of AI', 'Contribuția posibilă a AI')

D.frame(T('Possible contribution of AI', 'Contribuția posibilă a AI'), items(
    (T('\\textbf{Code}', '\\textbf{Cod}'),
         [T('a first draft of a rolling-origin backtest, of the pinball loss and the WQL, of a fan chart', 'o primă versiune a unui backtest cu origine mobilă, a pierderii pinball și a WQL, a unui fan chart')]),
    (T('\\textbf{Reading}', '\\textbf{Lectură}'),
         [T('a summary of a model paper (architecture, corpus, licence, release date)', 'un rezumat al articolului unui model (arhitectura, corpusul, licența, data lansării)')]),
    (T('\\textbf{Exploration}', '\\textbf{Explorare}'),
         [T('run several open models on many Romanian series from Eurostat or INS', 'rularea mai multor modele deschise pe multe serii românești de la Eurostat sau INS')]),
    (T('Example prompt', 'Exemplu de prompt'),
     [T('\\aiprompt{Write Python code that downloads Romanian monthly industrial production (Eurostat sts\\_inpr\\_m, not seasonally adjusted), forecasts 12 months ahead from monthly origins since 2016 with amazon/chronos-bolt-small (package chronos-forecasting, CPU), ETS and the seasonal naive, and reports MASE, the weighted quantile loss and the coverage of the 80\\% interval.}',
        '\\aiprompt{Write Python code that downloads Romanian monthly industrial production (Eurostat sts\\_inpr\\_m, not seasonally adjusted), forecasts 12 months ahead from monthly origins since 2016 with amazon/chronos-bolt-small (package chronos-forecasting, CPU), ETS and the seasonal naive, and reports MASE, the weighted quantile loss and the coverage of the 80\\% interval.}')])))

D.frame(T('Checks you must run', 'Verificări necesare'), items(
    T('The package and the model names: \\texttt{BaseChronosPipeline}, \\texttt{amazon/chronos-bolt-small}; an invented class or model name fails at once', 'Pachetul și numele modelelor: \\texttt{BaseChronosPipeline}, \\texttt{amazon/chronos-bolt-small}; o clasă sau un nume de model inventat eșuează imediat'),
    T('The context ends at the origin: no scaling, ordering or tuning with test data', 'Contextul se termină la origine: nicio scalare, ordonare sau calibrare cu date de test'),
    T('The quantile levels: Chronos-Bolt gives only 0.1 to 0.9; a ``1\\% quantile\'\' from it is an extrapolation', 'Nivelurile cuantilelor: Chronos-Bolt dă doar 0,1--0,9; o „cuantilă de 1\\%” obținută din el este o extrapolare'),
    T('The benchmarks: seasonal naive, ETS and ARIMA on the same origins; scores relative to the naive forecast', 'Reperele: prognoza sezonieră naivă, ETS și ARIMA pe aceleași origini; scoruri relative la prognoza naivă'),
    T('The release date of the weights against the test period; claims of ``state of the art\'\' need a source', 'Data lansării ponderilor comparată cu perioada de test; afirmațiile de tip „cel mai bun model” au nevoie de o sursă'),
    T('Every cited paper: it must exist; check the arXiv number or the DOI', 'Fiecare articol citat: trebuie să existe; verificați numărul arXiv sau DOI-ul')))

# =============================================================================
# REZUMAT
# =============================================================================
D.section('Summary', 'Rezumat')

D.frame(T('Key takeaways', 'Idei de reținut'), items(
    T('A foundation model is pretrained once on many series and forecasts new series zero-shot, from the context only', 'Un foundation model este pre-antrenat o singură dată pe multe serii și prognozează serii noi zero-shot, doar din context'),
    T('Series become Transformer inputs by scaling, quantisation (Chronos) or patching (TimesFM, Moirai, Chronos-Bolt)', 'Seriile devin intrări ale unui Transformer prin scalare, cuantizare (Chronos) sau patching (TimesFM, Moirai, Chronos-Bolt)'),
    T('Outputs are quantiles: score them with the pinball loss, the CRPS or WQL, and check coverage', 'Ieșirile sînt cuantile: evaluați-le cu pierderea pinball, CRPS sau WQL și verificați acoperirea'),
    T('On our seven series: large gains on hourly load, none on random-walk-like series, losses on short quarterly GDP; ARIMA remains a strong benchmark', 'Pe cele șapte serii: cîștiguri mari la consumul orar, niciunul la seriile apropiate de un mers aleator, pierderi la PIB-ul trimestrial scurt; ARIMA rămîne un reper puternic'),
    T('Contamination can inflate published scores: test on data released after the model', 'Contaminarea poate face scorurile publicate să pară mai bune decît sînt: testați pe date publicate după lansarea modelului')))

D.frame(T('Key formulas', 'Formule de reținut'), '{\\renewcommand{\\arraystretch}{1.4}' + table(
    'll', T('\\textbf{Quantity}', '\\textbf{Mărimea}') + ' & ' + T('\\textbf{Formula}', '\\textbf{Formula}'),
    [T('Attention', 'Atenția') + ' & $\\text{softmax}(QK^\\top/\\sqrt{d})\\,V$',
     T('Mean scaling', 'Scalarea prin medie') + ' & $\\tilde x_t = x_t/s$, \\quad $s = \\frac1C\\sum_{t=1}^{C}|x_t|$',
     T('Pinball loss', 'Pierderea pinball') + ' & $L_\\tau(y, \\hat q_\\tau) = \\max\\{\\tau(y - \\hat q_\\tau),\\ (\\tau - 1)(y - \\hat q_\\tau)\\}$',
     'CRPS & $\\int (F(z) - \\mathbf{1}\\{z \\ge y\\})^2\\,dz = 2\\int_0^1 L_\\tau(y, F^{-1}(\\tau))\\,d\\tau$',
     'WQL & $\\frac{1}{9}\\sum_{\\tau} 2\\sum_t L_{\\tau}(y_t, \\hat q_{\\tau,t}) \\big/ \\sum_t |y_t|$',
     'MASE & $\\frac1h\\sum_j |y_{T+j} - \\hat y_{T+j}| \\Big/ \\frac{1}{T-m}\\sum_{t>m}|y_t - y_{t-m}|$',
     T('Coverage', 'Acoperirea') + ' & $\\frac1n\\sum_i \\mathbf{1}\\{\\hat q_{0.1}^{(i)} \\le y_i \\le \\hat q_{0.9}^{(i)}\\}$, ' + T('target 0.8', 'ținta 0,8')],
    size='footnotesize') + '}')

D.frame(T('Self-assessment (1/2)', 'Autoevaluare (1/2)'), items(
    (T('\\textbf{Question}: what is the difference between zero-shot forecasting and fine-tuning?', '\\textbf{Întrebare}: care este diferența dintre prognoza zero-shot și fine-tuning?'),
     [T('\\textbf{Answer}: zero-shot keeps the pretrained weights fixed and uses only the context; fine-tuning trains the weights further on the target data', '\\textbf{Răspuns}: zero-shot păstrează fixe ponderile pre-antrenate și folosește doar contextul; fine-tuning-ul antrenează în continuare ponderile pe datele-țintă')]),
    (T('\\textbf{Question}: a context is 10, 20, 30. What are $s$ and the scaled values in Chronos?', '\\textbf{Întrebare}: un context este 10, 20, 30. Cît sînt $s$ și valorile scalate în Chronos?'),
     [T('\\textbf{Answer}: $s = 20$; scaled values 0.5, 1, 1.5', '\\textbf{Răspuns}: $s = 20$; valorile scalate 0,5; 1; 1,5')]),
    (T('\\textbf{Question}: $\\hat q_{0.9} = 5$ and $y = 6$. What is the pinball loss?', '\\textbf{Întrebare}: $\\hat q_{0{,}9} = 5$ și $y = 6$. Cît este pierderea pinball?'),
     [T('\\textbf{Answer}: $0.9 \\cdot (6 - 5) = 0.9$', '\\textbf{Răspuns}: $0{,}9 \\cdot (6 - 5) = 0{,}9$')]),
    (T('\\textbf{Question}: how many patches of 16 does a context of 512 values give?', '\\textbf{Întrebare}: cîte patch-uri de 16 dă un context de 512 valori?'),
     [T('\\textbf{Answer}: 32', '\\textbf{Răspuns}: 32')])))

D.frame(T('Self-assessment (2/2)', 'Autoevaluare (2/2)'), items(
    (T('\\textbf{Question}: an 80\\% interval covers 95\\% of the outcomes. Is this good?', '\\textbf{Întrebare}: un interval de 80\\% acoperă 95\\% din valorile observate. Este un rezultat bun?'),
     [T('\\textbf{Answer}: no: the intervals are too wide (not sharp); the CRPS penalises this', '\\textbf{Răspuns}: nu: intervalele sînt prea largi (lipsește precizia); CRPS penalizează acest lucru')]),
    (T('\\textbf{Question}: why can a foundation model not beat the random walk on EUR/RON?', '\\textbf{Întrebare}: de ce nu poate un foundation model să bată mersul aleator pe EUR/RON?'),
     [T('\\textbf{Answer}: the changes of the rate are close to unpredictable from the past; no model can extract information that the context does not contain', '\\textbf{Răspuns}: variațiile cursului sînt aproape imprevizibile pe baza trecutului; niciun model nu poate extrage informație pe care contextul nu o conține')]),
    (T('\\textbf{Question}: a model released in 2025 scores very well on test data from 2018--2023. What do you check?', '\\textbf{Întrebare}: un model lansat în 2025 obține scoruri foarte bune pe date de test din 2018--2023. Ce verificați?'),
     [T('\\textbf{Answer}: whether those series were in its pretraining corpus; repeat the test on data published after the release', '\\textbf{Răspuns}: dacă acele serii au fost în corpusul lui de pre-antrenare; repetați testul pe date publicate după lansare')]),
    (T('\\textbf{Idea for a project}: compare Chronos-Bolt, Chronos-2 and ARIMA on all Romanian monthly series of one Eurostat data set, before and after the release dates', '\\textbf{Idee de proiect}: comparați Chronos-Bolt, Chronos-2 și ARIMA pe toate seriile lunare ale României dintr-un set de date Eurostat, înainte și după datele de lansare'), []),
    T('Next: Chapter 12, spectral analysis', 'Urmează: Capitolul 12, analiza spectrală')))

D.references(bib())

if __name__ == '__main__':
    finalize(D.write(V))
