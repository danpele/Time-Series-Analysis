r"""
build_chapter10.py -- Capitolul 10 (Modele în spațiul stărilor, filtrul Kalman și modele Markov switching), EN + RO
==================================================================================================================
Text ⟦english||română⟧; cifrele @{cheie} vin din Quantlets/Ch_10/ch10_numbers.json (generate_all_charts.py).
Nicio cifră nu este scrisă de mînă. Capitol nou (nu există un deck TSA anterior).
Surse: Durbin și Koopman (2012), Harvey (1989), Hamilton (1989, 1994, cap. 13 și 22), Kim și Nelson (1999),
Huang și Petukhina (2022).
Ieșire:
  EN/Courses/chapter10_state_space_kalman_markov_switching.tex
  RO/Cursuri/capitol10_spatiul_starilor_kalman_markov_switching.tex
Rulare:
  python3 Quantlets/Ch_10/generate_all_charts.py
  python3 latex/build_chapter10.py && python3 latex/tsa_build.py compile 10
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tsa_build import Deck, Values, table, photo   # noqa: E402
from tsa_build import items as _items   # noqa: E402
from ch10_common import QLURL, REFS, T, bib, finalize, load, month, quarter   # noqa: E402


def items(*xs):
    return _items(*[x[0] if isinstance(x, tuple) and not x[1] else x for x in xs])


N = load()
V = Values()
D = Deck(10, 'lecture', refs=REFS)
C = 'https://commons.wikimedia.org/wiki/File:'


def ql(folder):
    return f'\\quantlet{{{folder.replace("_", chr(92) + "_")}}}{{\\qlurl{{{folder}}}}}'


def chart(title, fig, folder, bullets, h='0.6\\textheight', size='footnotesize'):
    body = (f'\\begin{{center}}\n\\includegraphics[width=0.97\\textwidth,height={h},keepaspectratio]{{{fig}.pdf}}\n'
            f'\\end{{center}}\n\\vspace{{-0.25cm}}\n' + items(*bullets) + '\n' + ql(folder))
    D.frame(title, body, size)


def interp(title, bullets, size='small'):
    D.frame(T(f'Interpreting {title[0]}', f'Interpretarea {title[1]}'), items(*bullets), size)


PH = {
    'kalman': ('ch10_kalman_2007.jpg', C + 'ETH-BIB-Kalman,_Rudolf_E._(1930-2016)-HK_04-01925.jpg',
               T('Photo', 'Foto') + ': ETH-Bibliothek Zürich, Bildarchiv (2007); CC BY-SA 4.0; Wikimedia Commons'),
    'apollo': ('ch10_apollo8_navigation_1968.jpg', C + 'Apollo_8_Lovell_at_Guidance_and_Navigation_station.jpg',
               T('Photo', 'Foto') + ': NASA (1968); ' + T('public domain', 'domeniu public') + '; Wikimedia Commons'),
    'nber': ('ch10_nber_offices_2022.jpg', C + 'National_Bureau_of_Economic_Research_offices.jpg',
             T('Photo', 'Foto') + ': Astrophobe (2022); CC BY-SA 4.0; Wikimedia Commons'),
}


def ph(key, cap, h='0.46\\textheight'):
    f, url, cred = PH[key]
    return photo(f, cap, url, cred, h=h)


def two(left, right, wl='0.4', wr='0.58'):
    return (f'\\begin{{columns}}[T]\n\\begin{{column}}{{{wl}\\textwidth}}\n{left}\n\\end{{column}}\n'
            f'\\begin{{column}}{{{wr}\\textwidth}}\n{right}\n\\end{{column}}\n\\end{{columns}}')


# =============================================================================
# CIFRE
# =============================================================================
P = V.put
EX = N['ex']
for r in EX['rows']:
    t = r['t']
    for k in ('y', 'a', 'P', 'F', 'K', 'v', 'af', 'Pf'):
        P(f'ex{t}.{k}', r[k], 3 if k in ('K',) else 2 if k in ('P', 'F', 'Pf', 'af', 'a') else 0)
P('ex.anext', EX['a_next'], 2)
P('ex.Pnext', EX['P_next'], 2)
P('ex.ssP', EX['ss_P'], 2)
P('ex.ssK', EX['ss_K'], 3)

SI = N['sim']
P('sim.r001', SI['q0.01']['acf1_dy'], 2)
P('sim.r1', SI['q1']['acf1_dy'], 2)
AR = N['arma']
for k in ('phi1', 'phi2', 'mu', 's2', 'c'):
    P(f'ar.{k}', AR[k], 3)
P('ar.lls', AR['ll_sarimax'], 4)
P('ar.llk', AR['ll_kalman'], 4)
V.raw('ar.n', str(AR['n']))

NI = N['nile']
P('ni.se', NI['s2_eps'], 0)
P('ni.sn', NI['s2_eta'], 0)
P('ni.q', NI['q'], 3)
P('ni.ll', NI['loglik'], 1)
P('ni.ssK', NI['ss_K'], 3)
P('ni.ssP', NI['ss_P'], 0)
P('ni.K2', NI['K2'], 3)
P('ni.K5', NI['K5'], 3)
P('ni.K10', NI['K10'], 3)
P('ni.alast', NI['a_last'], 0)
P('ni.sdlast', NI['sd_last'], 0)
P('ni.ymean', NI['y_mean'], 0)
P('ni.a98', NI['a1898'], 0)
P('ni.y99', NI['y1899'], 0)
P('ni.v99', NI['v1899'], 0)
P('ni.a99', NI['a1899'], 0)
P('ni.a05', NI['af_1905'], 0)
SMN = N['sm']
P('smn.se', SMN['s2_eps'], 0)
P('smn.sn', SMN['s2_eta'], 0)
GA = N['gain']
P('ga.ses', GA['alpha_ses'], 3)
P('ga.qa', GA['q_from_alpha'], 3)
P('ga.gap20', GA['gap20'], 2)
P('ga.gap5', GA['gap5'], 2)
V.raw('ga.nclose', str(GA['n_close']))
SMO = N['smooth']
for k in ('sd_f_mid', 'sd_s_mid', 's1897', 'f1897', 's1900', 'f1900'):
    P(f'smo.{k}', SMO[k], 0)
LI = N['lik']
P('li.lr001', LI['lr_q001'], 1)
P('li.lr1', LI['lr_q1'], 1)
P('li.s2', LI['s2_conc'], 0)
DG = N['diag']
P('dg.lb', DG['lb10'], 1)
P('dg.lbp', DG['lb10_p'], 2)
P('dg.jb', DG['jb'], 2)
P('dg.jbp', DG['jb_p'], 2)
P('dg.acf1', DG['acf1'], 2)
V.raw('dg.nout', str(DG['n_out']))
V.raw('dg.ne', str(DG['n_e']))
V.raw('dg.levy', str(DG['lev_min_year']))
P('dg.lev', DG['lev_min'], 2)
V.raw('dg.obsy', str(DG['obs_min_year']))
P('dg.obs', DG['obs_min'], 2)
MI = N['miss']
P('mi.gap', MI['sd_gap_mid'], 0)
P('mi.full', MI['sd_full_mid'], 0)
P('mi.f1', MI['f1'], 0)
P('mi.sdf1', MI['sd_f1'], 0)
P('mi.sdf30', MI['sd_f30'], 0)

TR = N['trend']
P('tr.st', TR['s2_trend'], 3)
P('tr.sar', TR['s2_ar'], 2)
P('tr.phi1', TR['phi1'], 2)
P('tr.phi2', TR['phi2'], 2)
P('tr.lam', TR['lam_ratio'], 0)
V.raw('tr.n', str(TR['n']))
V.raw('tr.first', quarter(TR['first']))
V.raw('tr.last', quarter(TR['last']))
P('tr.avg', TR['avg_growth'], 1)
GP = N['gap']
for k in ('sd_uc', 'sd_hp', 'sd_ham', 'c_uc_hp', 'c_uc_ham', 'c_hp_ham', 'last_uc', 'last_hp', 'last_ham', 'peak_uc',
          'peak_hp', 'peak_ham', 'uc2020', 'hp2020', 'ham_bsum'):
    P(f'gp.{k}', GP[k], 2 if k.startswith('c_') or k == 'ham_bsum' else 1)
V.raw('gp.lastd', quarter(GP['last_d']))
V.raw('gp.peakd', quarter(GP['peak_uc_d']))
RT = N['rt']
for k in ('rev_hp', 'rev_uc', 'c_hp', 'c_uc', 'hp_rt_2008', 'hp_fin_2008', 'uc_f_2008', 'uc_s_2008'):
    P(f'rt.{k}', RT[k], 2 if k.startswith('c_') else 1)
TV = N['tvp']
P('tv.ols', TV['beta_ols'], 2)
P('tv.min', TV['b_min'], 2)
V.raw('tv.mind', month(TV['b_min_d']))
P('tv.max', TV['b_max'], 2)
V.raw('tv.maxd', month(TV['b_max_d']))
P('tv.last', TV['b_last'], 2)
P('tv.sdlast', TV['sd_last'], 2)
P('tv.sdb', TV['sd_beta'], 3)
P('tv.lr', TV['lr'], 1)
V.int('tv.n', TV['n'])
FM = N['dfm']
for k, v in FM['loadings'].items():
    P(f'fm.{k}', v, 2)
P('fm.phi1', FM['phi1'], 2)
P('fm.phi2', FM['phi2'], 2)
P('fm.rec', FM['mean_rec'], 1)
P('fm.exp', FM['mean_exp'], 1)
P('fm.covid', FM['covid_min'], 0)
P('fm.last', FM['last'], 1)
V.raw('fm.lastd', month(FM['last_d']))
P('fm.neg', 100 * FM['share_neg_rec'], 0)
P('fm.corr', FM['corr_avg'], 2)

MU = N['msus']
for tag, s in (('ha', MU['ham']), ('hb', MU['ham_b']), ('ex', MU['ext'])):
    P(f'{tag}.p11', s['p11'], 3)
    P(f'{tag}.p22', s['p22'], 3)
    P(f'{tag}.d1', s['dur1'], 1)
    P(f'{tag}.d2', s['dur2'], 1)
    P(f'{tag}.m1', s['const_1'], 2)
    P(f'{tag}.m2', s['const_2'], 2)
    P(f'{tag}.ll', s['loglik'], 2)
    P(f'{tag}.erg', 100 * s['ergodic1'], 0)
P('ex.s', MU['ext']['sigma2'] ** 0.5, 2)
P('hacon', 100 * MU['ham_conc']['concord'], 0)
P('hahit', 100 * MU['ham_conc']['hit'], 0)
P('hbhit', 100 * MU['ham_b_conc']['hit'], 0)
P('excon', 100 * MU['ext_conc']['concord'], 0)
P('exhit', 100 * MU['ext_conc']['hit'], 0)
P('exfalse', 100 * MU['ext_conc']['false'], 0)
P('exconf', 100 * MU['ext_conc_f']['concord'], 0)
P('exhitf', 100 * MU['ext_conc_f']['hit'], 0)
P('ms.p20', MU['p2020q2'], 2)
P('ms.post', MU['post_max_after2021'], 2)
P('ms.g20', MU['g2020q2'], 1)
V.raw('ms.nrec', str(MU['n_rec_q']))
MF = N['msf']
for k in ('f2008q1', 's2008q1', 'f2008q3', 's2008q3', 'f2008q4', 's2008q4', 'g2008q3', 'g2008q4'):
    P(f'mf.{k}', MF[k], 2)
V.raw('mf.ff', quarter(MF['first_f50']))
V.raw('mf.fs', quarter(MF['first_s50']))
PI = N['pit']
P('pi.d1', PI['sv']['dur1'], 0)
P('pi.d2', PI['sv']['dur2'], 0)
P('pi.s1', PI['sv']['sigma2_1'] ** 0.5, 2)
P('pi.s2', PI['sv']['sigma2_2'] ** 0.5, 2)
V.raw('pi.sw', quarter(PI['sv_switch']))
P('pi.con', 100 * PI['sv_conc']['concord'], 0)
P('pi.false', 100 * PI['sv_conc']['false'], 0)
P('pi.m1', PI['all']['const_1'], 1)
P('pi.d2all', PI['all']['dur2'], 0)
MR = N['msro']
P('mr.d1', MR['dur1'], 1)
P('mr.d2', MR['dur2'], 1)
P('mr.m1', MR['const_1'], 2)
P('mr.m2', MR['const_2'], 2)
P('mr.s1', MR['sd1'], 1)
P('mr.s2', MR['sd2'], 1)
P('mr.share', 100 * MR['share_vol'], 0)
P('mr.plast', MR['p_last'], 2)
SPL = MR['spells']
for i, (a, b) in enumerate(SPL):
    V.raw(f'mr.sp{i}', quarter(a) + '--' + quarter(b) if a != b else quarter(a))
VO = N['vol']
P('vo.s1', VO['sd1_ann'], 0)
P('vo.s2', VO['sd2_ann'], 0)
P('vo.d1', VO['dur1'], 0)
P('vo.d2', VO['dur2'], 0)
P('vo.m1', VO['const_1'], 2)
P('vo.m2', VO['const_2'], 2)
P('vo.ga', VO['garch_a'], 2)
P('vo.gb', VO['garch_b'], 2)
P('vo.gab', VO['garch_ab'], 2)
P('vo.corr', VO['corr'], 2)
P('vo.share', 100 * VO['share_turb'], 0)
P('vo.erg', 100 * VO['ergodic1'], 0)
TY = list(VO['top_years'].items())
for i, (y, s) in enumerate(TY[:3]):
    V.raw(f'vo.y{i}', y)
    P(f'vo.ys{i}', 100 * s, 0)
ME = N['mem']
for k in ('acf_data_1', 'acf_data_26', 'acf_data_52', 'acf_sim_1', 'acf_sim_26', 'acf_sim_52', 'd_data', 'd_sim',
          'd_sim_sd', 'garch_ab_sim', 'garch_ab_sim_min', 'd_iid'):
    P(f'me.{k}', ME[k], 2)
V.raw('me.reps', str(ME['reps']))
V.raw('me.repsg', str(ME['reps_garch']))

for _k, _v in list(V.items()):                 # negative numbers: a real minus sign in text and in math mode
    if isinstance(_v, str) and _v.startswith('⁅-'):
        V[_k] = '⁅\\ensuremath{-}' + _v[2:]

# =============================================================================
# DESCHIDERE
# =============================================================================
D.frame(T("Today's question and route", 'Întrebarea de azi și traseul'), items(
    (T('\\textbf{Question}: how do we measure what we cannot observe directly: the true level of a noisy series, the output gap, a changing beta, a recession?',
       '\\textbf{Întrebarea}: cum măsurăm ceea ce nu observăm direct: nivelul real al unei serii zgomotoase, deviația PIB de la potențial, un beta care se schimbă, o recesiune?'),
     [T('the answer: write the unobserved quantity as a \\textbf{state} and let the data update our estimate of it, period by period',
        'răspunsul: scriem mărimea neobservată ca o \\textbf{stare} și lăsăm datele să ne actualizeze estimarea ei, perioadă cu perioadă')]),
    (T('\\textbf{Route} of the chapter', '\\textbf{Traseul} capitolului'),
     [T('the state space form: measurement and transition equations; local level, local linear trend, ARMA, time-varying regression',
        'forma în spațiul stărilor: ecuația de măsurare și ecuația de tranziție; local level, local linear trend, ARMA, regresie cu parametri variabili'),
      T('the Kalman filter by hand; smoothing; likelihood and estimation; missing values', 'filtrul Kalman de mînă; netezirea; verosimilitatea și estimarea; valorile lipsă'),
      T('trend and cycle: Romanian GDP, the HP filter and its critique; dynamic factors and nowcasting', 'trend și ciclu: PIB-ul României, filtrul HP și critica lui; factori dinamici și nowcasting'),
      T('Markov switching: recessions, Romanian growth regimes, volatility regimes', 'Markov switching: recesiuni, regimuri ale creșterii economice din România, regimuri de volatilitate')]),
    T('We build on Chapter 0 (exponential smoothing), Chapter 2 (ARMA), Chapter 5 (GARCH) and Chapter 8 (long memory)',
      'Pornim de la Capitolul 0 (netezirea exponențială), Capitolul 2 (ARMA), Capitolul 5 (GARCH) și Capitolul 8 (memoria lungă)')))

D.frame(T('Learning outcomes', 'Rezultatele învățării'), items(
    T('Write a model in state space form: local level, local linear trend, AR(2), a regression with a time-varying coefficient', 'Scrieți un model în forma în spațiul stărilor: local level, local linear trend, AR(2), o regresie cu un coeficient variabil în timp'),
    T('Run the Kalman filter by hand and explain the Kalman gain', 'Aplicați de mînă filtrul Kalman și explicați cîștigul Kalman'),
    T('Show that simple exponential smoothing is the steady state of the local level filter', 'Arătați că netezirea exponențială simplă este starea de echilibru a filtrului pentru modelul local level'),
    T('Estimate a state space model by maximum likelihood, smooth it, and handle missing values and forecasts', 'Estimați un model în spațiul stărilor prin verosimilitate maximă, netezați-l și tratați valorile lipsă și prognozele'),
    T('Decompose GDP into trend and cycle and judge the HP filter against the critique of Hamilton (2018)', 'Descompuneți PIB-ul în trend și ciclu și evaluați filtrul HP prin prisma criticii lui Hamilton (2018)'),
    T('Estimate a Markov-switching model and read its regime probabilities, durations and pitfalls', 'Estimați un model Markov switching și interpretați probabilitățile regimurilor, duratele și capcanele lui')))

D.frame(T('Reading and tools', 'Bibliografie și instrumente'), items(
    (T('Textbook: \\refHP, the chapter on state space models and the Kalman filter', 'Manual: \\refHP, capitolul despre modelele în spațiul stărilor și filtrul Kalman'),
     [T('state space: \\refDK\\ (the Nile example of this chapter comes from their book); \\refHarvey; \\refSS, Ch.~6', 'spațiul stărilor: \\refDK\\ (exemplul Nilului din acest capitol provine din cartea lor); \\refHarvey; \\refSS, cap.~6'),
      T('regime switching: \\refHamMS; \\refHamilton, Ch.~13 (Kalman filter) and Ch.~22 (regime changes); \\refKN', 'schimbări de regim: \\refHamMS; \\refHamilton, cap.~13 (filtrul Kalman) și cap.~22 (schimbări de regim); \\refKN')]),
    (T('Python Quantlets of this chapter: \\href{' + QLURL + '}{Quantlets/Ch\\_10}', 'Quantlet-urile Python ale capitolului: \\href{' + QLURL + '}{Quantlets/Ch\\_10}'),
     [T('a Kalman filter written out in \\texttt{numpy}; \\texttt{statsmodels}: \\texttt{UnobservedComponents}, \\texttt{SARIMAX}, \\texttt{DynamicFactor}, \\texttt{MarkovRegression}, \\texttt{MarkovAutoregression}',
        'un filtru Kalman scris explicit în \\texttt{numpy}; \\texttt{statsmodels}: \\texttt{UnobservedComponents}, \\texttt{SARIMAX}, \\texttt{DynamicFactor}, \\texttt{MarkovRegression}, \\texttt{MarkovAutoregression}')]),
    T('Lecture notebook: \\href{\\colaburl{notebooks/EN/chapter10_lecture_notebook.ipynb}}{open in Google Colab}',
      'Notebook-ul cursului: \\href{\\colaburl{notebooks/EN/chapter10_lecture_notebook.ipynb}}{deschideți în Google Colab}'),
    T('Video course: \\quantinar{Kalman Filter}{https://quantinar.com/course/42/methodology}',
      'Curs video: \\quantinar{Kalman Filter}{https://quantinar.com/course/42/methodology}')))

# =============================================================================
# 1. STĂRI ASCUNSE
# =============================================================================
D.section('Hidden states', 'Stări ascunse')

D.frame(T('Rudolf Kálmán and the Apollo navigation', 'Rudolf Kálmán și navigația misiunilor Apollo'), two(
    ph('kalman', T('Rudolf E. Kálmán (1930--2016)', 'Rudolf E. Kálmán (1930--2016)'), h='0.25\\textheight')
    + '\\\\[1mm]' + ph('apollo', T('Apollo 8 (1968): Jim Lovell at the guidance and navigation station', 'Apollo 8 (1968): Jim Lovell la stația de ghidare și navigație'), h='0.21\\textheight'),
    items((T('\\refKalman: a recursive estimate of a hidden state from noisy measurements', '\\refKalman: o estimare recursivă a unei stări ascunse din măsurători zgomotoase'),
           [T('at each step: predict the state, then correct the prediction with the new measurement', 'la fiecare pas: prezicem starea, apoi corectăm predicția cu noua măsurătoare'),
            T('the correction weight, the \\textbf{Kalman gain}, balances the two sources of uncertainty', 'ponderea corecției, \\textbf{cîștigul Kalman}, echilibrează cele două surse de incertitudine')]),
          (T('The filter was adapted at NASA Ames for the navigation of the Apollo spacecraft', 'Filtrul a fost adaptat la NASA Ames pentru navigația navelor Apollo'),
           [T('state: position and velocity; measurements: star sightings and radar', 'starea: poziția și viteza; măsurătorile: observații ale stelelor și radar')]),
          T('In economics the state is a trend, a cycle, a coefficient or a regime; the measurements are the published data', 'În economie starea este un trend, un ciclu, un coeficient sau un regim; măsurătorile sînt datele publicate'))), size='footnotesize')

D.frame(T('Hidden states in economics and finance', 'Stări ascunse în economie și finanțe'), table(
    '>{\\raggedright\\arraybackslash}p{3.6cm}>{\\raggedright\\arraybackslash}p{3.6cm}>{\\raggedright\\arraybackslash}p{3.4cm}', T('\\textbf{Observed series}', '\\textbf{Seria observată}') + ' & ' + T('\\textbf{Hidden state}', '\\textbf{Starea ascunsă}') + ' & ' + T('\\textbf{Model in this chapter}', '\\textbf{Modelul din acest capitol}'),
    [T('Nile flow, monthly inflation', 'Debitul Nilului, inflația lunară') + ' & ' + T('the underlying level', 'nivelul de fond') + ' & local level',
     T('real GDP', 'PIB real') + ' & ' + T('trend and output gap', 'trendul și deviația PIB') + ' & ' + T('unobserved components', 'componente neobservate'),
     T('stock index returns', 'randamentele unui indice bursier') + ' & ' + T('a time-varying beta', 'un beta variabil în timp') + ' & ' + T('time-varying regression', 'regresie cu parametri variabili'),
     T('several activity indicators', 'mai mulți indicatori de activitate') + ' & ' + T('one common factor', 'un factor comun') + ' & ' + T('dynamic factor model', 'model cu factori dinamici'),
     T('GDP growth, returns', 'creșterea PIB, randamentele') + ' & ' + T('recession or expansion; calm or turbulent', 'recesiune sau expansiune; calm sau agitat') + ' & Markov switching'],
    size='footnotesize') + items(
    T('The first four have a \\textbf{continuous} state and are filtered by the Kalman filter', 'Primele patru au o stare \\textbf{continuă} și sînt filtrate cu filtrul Kalman'),
    T('The last has a \\textbf{discrete} state (a regime) and is filtered by the Hamilton filter', 'Ultimul are o stare \\textbf{discretă} (un regim) și este filtrat cu filtrul Hamilton'),
    T('Both filters do the same thing: predict the state, then update the prediction with $y_t$', 'Ambele filtre fac același lucru: prezic starea, apoi actualizează predicția cu $y_t$')))

# =============================================================================
# 2. FORMA ÎN SPAȚIUL STĂRILOR
# =============================================================================
D.section('The state space form', 'Forma în spațiul stărilor')

D.frame(T('The state space form', 'Forma în spațiul stărilor'), items(
    (T('\\textbf{Measurement equation}: $y_t = Z_t\\alpha_t + \\varepsilon_t$, \\quad $\\varepsilon_t \\sim N(0, H)$', '\\textbf{Ecuația de măsurare}: $y_t = Z_t\\alpha_t + \\varepsilon_t$, \\quad $\\varepsilon_t \\sim N(0, H)$'),
     [T('$y_t$: the observation (here a scalar); $\\alpha_t$: the \\textbf{state vector} ($m \\times 1$), not observed; $Z_t$: a $1 \\times m$ row of known numbers',
        '$y_t$: observația (aici un scalar); $\\alpha_t$: \\textbf{vectorul de stare} ($m \\times 1$), neobservat; $Z_t$: un rînd $1 \\times m$ de numere cunoscute')]),
    (T('\\textbf{Transition equation}: $\\alpha_{t+1} = T\\alpha_t + R\\eta_t$, \\quad $\\eta_t \\sim N(0, Q)$', '\\textbf{Ecuația de tranziție}: $\\alpha_{t+1} = T\\alpha_t + R\\eta_t$, \\quad $\\eta_t \\sim N(0, Q)$'),
     [T('the state evolves as a first-order vector autoregression (VAR(1), Chapter 6); $T$ is $m \\times m$', 'starea evoluează ca o autoregresie vectorială de ordinul 1 (VAR(1), Capitolul 6); $T$ este $m \\times m$'),
      T('$R$: maps the shocks $\\eta_t$ into the state; $H$, $Q$: the variances of the measurement and of the state shocks', '$R$: transmite șocurile $\\eta_t$ în stare; $H$, $Q$: varianțele șocurilor de măsurare și ale șocurilor stării'),
      T('initial state: $\\alpha_1 \\sim N(a_1, P_1)$, $a_1$, $P_1$: its mean and variance; $\\varepsilon_t$, $\\eta_s$ and $\\alpha_1$ mutually independent', 'starea inițială: $\\alpha_1 \\sim N(a_1, P_1)$, $a_1$, $P_1$: media și varianța ei; $\\varepsilon_t$, $\\eta_s$ și $\\alpha_1$ sînt independente între ele')]),
    (T('The \\textbf{system matrices} $Z_t, H, T, R, Q$ contain the parameters $\\theta$; they are estimated by maximum likelihood', '\\textbf{Matricele sistemului} $Z_t, H, T, R, Q$ conțin parametrii $\\theta$; aceștia se estimează prin verosimilitate maximă'),
     [T('the same form covers ARMA, exponential smoothing, trend--cycle models, regressions with moving coefficients and factor models \\refDK', 'aceeași formă acoperă ARMA, netezirea exponențială, modelele trend--ciclu, regresiile cu coeficienți mobili și modelele factoriale \\refDK')])))

D.frame(T('The local level model', 'Modelul local level'), items(
    (T('\\textbf{Local level} (random walk plus noise): $y_t = \\mu_t + \\varepsilon_t$, \\quad $\\mu_{t+1} = \\mu_t + \\eta_t$', '\\textbf{Local level} (mers aleator plus zgomot): $y_t = \\mu_t + \\varepsilon_t$, \\quad $\\mu_{t+1} = \\mu_t + \\eta_t$'),
     [T('state $\\alpha_t = \\mu_t$, the level; $Z = T = R = 1$, $H = \\sigma^2_\\varepsilon$, $Q = \\sigma^2_\\eta$', 'starea $\\alpha_t = \\mu_t$, nivelul; $Z = T = R = 1$, $H = \\sigma^2_\\varepsilon$, $Q = \\sigma^2_\\eta$'),
      T('$\\varepsilon_t$: measurement noise that disappears next period; $\\eta_t$: a permanent shift of the level', '$\\varepsilon_t$: zgomot de măsurare care dispare în perioada următoare; $\\eta_t$: o deplasare permanentă a nivelului')]),
    (T('\\textbf{Signal-to-noise ratio} $q = \\sigma^2_\\eta/\\sigma^2_\\varepsilon$: how much of the movement is permanent', '\\textbf{Raportul semnal--zgomot} $q = \\sigma^2_\\eta/\\sigma^2_\\varepsilon$: cîtă parte din mișcare este permanentă'),
     [T('$q = 0$: a constant level (white noise around a mean); $q \\to \\infty$: a pure random walk', '$q = 0$: un nivel constant (zgomot alb în jurul unei medii); $q \\to \\infty$: un mers aleator pur')]),
    (T('Reduced form: $\\Delta y_t = \\eta_{t-1} + \\varepsilon_t - \\varepsilon_{t-1}$ is an MA(1) with $\\rho(1) = -1/(q+2)$', 'Forma redusă: $\\Delta y_t = \\eta_{t-1} + \\varepsilon_t - \\varepsilon_{t-1}$ este un MA(1) cu $\\rho(1) = -1/(q+2)$'),
     [T('so the local level model is an ARIMA(0,1,1) (Chapter 3) with a negative MA coefficient', 'deci modelul local level este un ARIMA(0,1,1) (Capitolul 3) cu un coeficient MA negativ'),
      T('$\\gamma(0) = \\sigma^2_\\eta + 2\\sigma^2_\\varepsilon$ and $\\gamma(1) = -\\sigma^2_\\varepsilon$ give $\\rho(1)$', '$\\gamma(0) = \\sigma^2_\\eta + 2\\sigma^2_\\varepsilon$ și $\\gamma(1) = -\\sigma^2_\\varepsilon$ dau $\\rho(1)$')])))

chart(T('Local level and local linear trend paths', 'Traiectorii local level și local linear trend'), 'tsa_ch10_ss_examples', 'TSA_ch10_state_space_examples', [
    T('Left: the same shocks $\\varepsilon_t, \\eta_t \\sim N(0, 1)$ with $\\sigma^2_\\varepsilon = 1$ and $q = 0.01;\\ 0.1;\\ 1$ (thin: $y_t$, thick: $\\mu_t$). Right: a local linear trend, whose slope $\\beta_t$ is itself a random walk',
      'Stînga: aceleași șocuri $\\varepsilon_t, \\eta_t \\sim N(0, 1)$, cu $\\sigma^2_\\varepsilon = 1$ și $q = 0{,}01;\\ 0{,}1;\\ 1$ (subțire: $y_t$, gros: $\\mu_t$). Dreapta: un local linear trend, a cărui pantă $\\beta_t$ este ea însăși un mers aleator')],
    h='0.66\\textheight')

interp(('the simulated paths', 'traiectoriilor simulate'), [
    (T('Small $q$: the level hardly moves and $y_t$ looks like noise around a mean; large $q$: the level wanders and $y_t$ follows it', '$q$ mic: nivelul abia se mișcă, iar $y_t$ arată ca un zgomot în jurul unei medii; $q$ mare: nivelul rătăcește, iar $y_t$ îl urmează'),
     [T('first-difference autocorrelation: @{sim.r001} for $q = 0.01$ (theory $-1/2.01$) and @{sim.r1} for $q = 1$ (theory $-1/3$)', 'autocorelația primei diferențe: @{sim.r001} pentru $q = 0{,}01$ (teoretic $-1/2{,}01$) și @{sim.r1} pentru $q = 1$ (teoretic $-1/3$)')]),
    (T('The local linear trend has a slope that changes slowly: periods of fast and slow growth', 'Local linear trend are o pantă care se schimbă lent: perioade de creștere rapidă și lentă'),
     [T('this is the trend model we will use for GDP', 'acesta este modelul de trend pe care îl vom folosi pentru PIB')]),
    T('The estimation problem: from $y_t$ alone, how much is level and how much is noise? The Kalman filter answers it for given variances; the likelihood chooses the variances',
      'Problema de estimare: doar din $y_t$, cît este nivel și cît este zgomot? Filtrul Kalman răspunde pentru varianțe date; verosimilitatea alege varianțele')])

D.frame(T('The local linear trend', 'Modelul local linear trend'), items(
    (T('\\textbf{Local linear trend}: $y_t = \\mu_t + \\varepsilon_t$, \\quad $\\mu_{t+1} = \\mu_t + \\beta_t + \\eta_t$, \\quad $\\beta_{t+1} = \\beta_t + \\zeta_t$', '\\textbf{Local linear trend}: $y_t = \\mu_t + \\varepsilon_t$, \\quad $\\mu_{t+1} = \\mu_t + \\beta_t + \\eta_t$, \\quad $\\beta_{t+1} = \\beta_t + \\zeta_t$'),
     [T('state $\\alpha_t = (\\mu_t, \\beta_t)\'$: the level and the slope (the growth rate)', 'starea $\\alpha_t = (\\mu_t, \\beta_t)\'$: nivelul și panta (rata de creștere)'),
      T('$Z = (1,\\ 0)$, \\quad $T = \\begin{pmatrix}1 & 1\\\\ 0 & 1\\end{pmatrix}$, \\quad $Q = \\mathrm{diag}(\\sigma^2_\\eta, \\sigma^2_\\zeta)$', '$Z = (1,\\ 0)$, \\quad $T = \\begin{pmatrix}1 & 1\\\\ 0 & 1\\end{pmatrix}$, \\quad $Q = \\mathrm{diag}(\\sigma^2_\\eta, \\sigma^2_\\zeta)$')]),
    (T('Special cases', 'Cazuri particulare'),
     [T('$\\sigma^2_\\eta = \\sigma^2_\\zeta = 0$: a deterministic linear trend $\\mu_1 + \\beta_1 t$ (Chapter 3)', '$\\sigma^2_\\eta = \\sigma^2_\\zeta = 0$: un trend liniar determinist $\\mu_1 + \\beta_1 t$ (Capitolul 3)'),
      T('$\\sigma^2_\\zeta = 0$: a random walk with drift $\\beta$', '$\\sigma^2_\\zeta = 0$: un mers aleator cu deriva $\\beta$'),
      T('$\\sigma^2_\\eta = 0$: the \\textbf{smooth trend} (integrated random walk), the model behind the HP filter (Section 5)', '$\\sigma^2_\\eta = 0$: \\textbf{trendul neted} (mers aleator integrat), modelul din spatele filtrului HP (secțiunea 5)')]),
    T('Forecasts: $\\hat y_{T+h} = \\hat\\mu_T + h\\hat\\beta_T$, the state space version of Holt\'s method (Chapter 0)', 'Prognozele: $\\hat y_{T+h} = \\hat\\mu_T + h\\hat\\beta_T$, versiunea în spațiul stărilor a metodei Holt (Capitolul 0)')))

D.frame(T('ARMA models in state space form', 'Modele ARMA în forma în spațiul stărilor'), items(
    (T('\\textbf{AR(2)}: $y_t = \\phi_1y_{t-1} + \\phi_2y_{t-2} + u_t$; state $\\alpha_t = (y_t, y_{t-1})\'$', '\\textbf{AR(2)}: $y_t = \\phi_1y_{t-1} + \\phi_2y_{t-2} + u_t$; starea $\\alpha_t = (y_t, y_{t-1})\'$'),
     [T('$y_t = (1,\\ 0)\\,\\alpha_t$ ($H = 0$), \\quad $\\alpha_{t+1} = \\begin{pmatrix}\\phi_1 & \\phi_2\\\\ 1 & 0\\end{pmatrix}\\alpha_t + \\begin{pmatrix}1\\\\ 0\\end{pmatrix}u_{t+1}$', '$y_t = (1,\\ 0)\\,\\alpha_t$ ($H = 0$), \\quad $\\alpha_{t+1} = \\begin{pmatrix}\\phi_1 & \\phi_2\\\\ 1 & 0\\end{pmatrix}\\alpha_t + \\begin{pmatrix}1\\\\ 0\\end{pmatrix}u_{t+1}$'),
      T('$T$ is the companion matrix of Chapter 2; its eigenvalues are the inverse AR roots', '$T$ este matricea companion din Capitolul 2; valorile ei proprii sînt inversele rădăcinilor AR')]),
    (T('\\textbf{ARMA(1,1)}: $y_t = \\phi y_{t-1} + u_t + \\theta u_{t-1}$ with $\\alpha_t = (y_t, \\theta u_t)\'$', '\\textbf{ARMA(1,1)}: $y_t = \\phi y_{t-1} + u_t + \\theta u_{t-1}$, cu $\\alpha_t = (y_t, \\theta u_t)\'$'),
     [T('$T = \\begin{pmatrix}\\phi & 1\\\\ 0 & 0\\end{pmatrix}$, $R = (1,\\ \\theta)\'$; any ARMA$(p,q)$ fits with $m = \\max(p, q+1)$', '$T = \\begin{pmatrix}\\phi & 1\\\\ 0 & 0\\end{pmatrix}$, $R = (1,\\ \\theta)\'$; orice ARMA$(p,q)$ se scrie cu $m = \\max(p, q+1)$')]),
    (T('Why it matters: the Kalman filter gives the \\textbf{exact} Gaussian likelihood of an ARMA model, with missing values allowed', 'Importanța practică: filtrul Kalman dă verosimilitatea gaussiană \\textbf{exactă} a unui model ARMA, chiar cu valori lipsă'),
     [T('the stationary start: $P_1$ solves $P = TPT\' + RQR\'$, i.e.\\ $\\mathrm{vec}(P_1) = (I - T \\otimes T)^{-1}\\mathrm{vec}(RQR\')$', 'pornirea staționară: $P_1$ rezolvă $P = TPT\' + RQR\'$, adică $\\mathrm{vec}(P_1) = (I - T \\otimes T)^{-1}\\mathrm{vec}(RQR\')$'),
      T('this is how \\texttt{statsmodels} \\texttt{SARIMAX} estimates the ARIMA models of Chapters 2--4', 'așa estimează \\texttt{statsmodels} \\texttt{SARIMAX} modelele ARIMA din Capitolele 2--4')])))

D.frame(T('Worked example: AR(2) for US GDP growth', 'Exemplu rezolvat: AR(2) pentru creșterea PIB din SUA'), items(
    (T('Data: US real GDP growth, quarter on quarter, 1947--2019 ($T = @{ar.n}$, FRED GDPC1)', 'Date: creșterea PIB real al SUA, trimestru față de trimestru, 1947--2019 ($T = @{ar.n}$, FRED GDPC1)'),
     [T('AR(2) with a constant by \\texttt{SARIMAX}: $\\hat\\phi_1 = @{ar.phi1}$, $\\hat\\phi_2 = @{ar.phi2}$, $\\hat\\sigma^2 = @{ar.s2}$, mean $\\hat\\mu = @{ar.mu}\\%$ per quarter',
        'AR(2) cu termen liber estimat cu \\texttt{SARIMAX}: $\\hat\\phi_1 = @{ar.phi1}$, $\\hat\\phi_2 = @{ar.phi2}$, $\\hat\\sigma^2 = @{ar.s2}$, media $\\hat\\mu = @{ar.mu}\\%$ pe trimestru')]),
    (T('Our own filter, with the state space matrices of the previous slide and the stationary $P_1$', 'Filtrul nostru, cu matricele din slide-ul anterior și cu $P_1$ staționar'),
     [T('log-likelihood: Kalman @{ar.llk}; \\texttt{SARIMAX} @{ar.lls}: the same number', 'log-verosimilitatea: Kalman @{ar.llk}; \\texttt{SARIMAX} @{ar.lls}: același număr')]),
    T('The prediction errors $v_t$ of the filter are the one-step forecast errors of the AR(2); the likelihood is built from them (Section 4)', 'Erorile de predicție $v_t$ ale filtrului sînt erorile de prognoză la un pas ale AR(2); verosimilitatea se construiește din ele (secțiunea 4)')))

D.frame(T('Regression with time-varying parameters', 'Regresie cu parametri variabili în timp'), items(
    (T('\\textbf{Time-varying parameter (TVP) regression}: $y_t = \\alpha + \\beta_t x_t + \\varepsilon_t$, \\quad $\\beta_{t+1} = \\beta_t + \\eta_t$', '\\textbf{Regresia cu parametri variabili în timp} (TVP): $y_t = \\alpha + \\beta_t x_t + \\varepsilon_t$, \\quad $\\beta_{t+1} = \\beta_t + \\eta_t$'),
     [T('state $(\\alpha, \\beta_t)\'$; the row $Z_t = (1,\\ x_t)$ \\textbf{changes over time}; $Q = \\mathrm{diag}(0, \\sigma^2_\\eta)$', 'starea $(\\alpha, \\beta_t)\'$; rîndul $Z_t = (1,\\ x_t)$ \\textbf{se schimbă în timp}; $Q = \\mathrm{diag}(0, \\sigma^2_\\eta)$'),
      T('$\\sigma^2_\\eta = 0$: ordinary least squares (OLS), computed recursively', '$\\sigma^2_\\eta = 0$: metoda celor mai mici pătrate (OLS), calculată recursiv')]),
    (T('Examples', 'Exemple'),
     [T('a market beta that rises in crises; the pass-through of the exchange rate to inflation; the persistence of inflation', 'un beta de piață care crește în crize; transmiterea cursului de schimb în inflație; persistența inflației')]),
    T('Better than a rolling window: no window length to choose, every observation is used, and $\\beta_t$ comes with a standard error', 'Mai bună decît o fereastră mobilă: nu alegem lungimea ferestrei, folosim toate observațiile, iar $\\beta_t$ vine cu o eroare standard')))

D.frame(T('Exponential smoothing as a state space model', 'Netezirea exponențială ca model în spațiul stărilor'), items(
    (T('Chapter 0: simple exponential smoothing (SES) updates a level $\\ell_t = \\alpha y_t + (1-\\alpha)\\ell_{t-1}$ and forecasts $\\hat y_{t+1} = \\ell_t$', 'Capitolul 0: netezirea exponențială simplă (SES) actualizează un nivel $\\ell_t = \\alpha y_t + (1-\\alpha)\\ell_{t-1}$ și prognozează $\\hat y_{t+1} = \\ell_t$'),
     [T('equivalently $\\ell_t = \\ell_{t-1} + \\alpha(y_t - \\ell_{t-1})$: old level plus a share $\\alpha$ of the surprise', 'echivalent $\\ell_t = \\ell_{t-1} + \\alpha(y_t - \\ell_{t-1})$: nivelul vechi plus o fracțiune $\\alpha$ din surpriză')]),
    (T('\\refMuth: SES gives the optimal forecasts when the data follow the local level model', '\\refMuth: SES dă prognozele optime atunci cînd datele urmează modelul local level'),
     [T('Section 3 shows it: $\\alpha$ is the steady-state Kalman gain', 'secțiunea 3 o arată: $\\alpha$ este cîștigul Kalman de echilibru')]),
    T('ETS models (error, trend, seasonal) are state space models with a single source of error \\refHKOS; the models here have separate shocks for each component',
      'Modelele ETS (eroare, trend, sezonalitate) sînt modele în spațiul stărilor cu o singură sursă de eroare \\refHKOS; modelele de aici au șocuri separate pentru fiecare componentă')))

D.recap(('The state space form', 'forma în spațiul stărilor'), [
    T('$y_t = Z_t\\alpha_t + \\varepsilon_t$, $\\alpha_{t+1} = T\\alpha_t + R\\eta_t$: a hidden VAR(1) observed with noise', '$y_t = Z_t\\alpha_t + \\varepsilon_t$, $\\alpha_{t+1} = T\\alpha_t + R\\eta_t$: un VAR(1) ascuns, observat cu zgomot'),
    T('Local level: $q = \\sigma^2_\\eta/\\sigma^2_\\varepsilon$; reduced form ARIMA(0,1,1)', 'Local level: $q = \\sigma^2_\\eta/\\sigma^2_\\varepsilon$; forma redusă ARIMA(0,1,1)'),
    T('Local linear trend, ARMA and TVP regression are other choices of $Z_t, T, Q$', 'Local linear trend, ARMA și regresia TVP sînt alte alegeri ale lui $Z_t, T, Q$'),
    T('The Kalman likelihood of an AR(2) equals the exact ARMA likelihood', 'Verosimilitatea Kalman a unui AR(2) este egală cu verosimilitatea ARMA exactă')])

# =============================================================================
# 3. FILTRUL KALMAN
# =============================================================================
D.section('The Kalman filter', 'Filtrul Kalman')

D.frame(T('The idea of the filter', 'Ideea filtrului'), items(
    (T('Notation: $Y_t = (y_1, \\dots, y_t)$; we track two numbers (vectors) for the state', 'Notație: $Y_t = (y_1, \\dots, y_t)$; urmărim două mărimi (vectori) pentru stare'),
     [T('prediction: $a_t = E(\\alpha_t \\mid Y_{t-1})$ with variance $P_t$', 'predicția: $a_t = E(\\alpha_t \\mid Y_{t-1})$, cu varianța $P_t$'),
      T('filtered estimate: $a_{t|t} = E(\\alpha_t \\mid Y_t)$ with variance $P_{t|t}$', 'estimarea filtrată: $a_{t|t} = E(\\alpha_t \\mid Y_t)$, cu varianța $P_{t|t}$')]),
    (T('Each period has two steps', 'Fiecare perioadă are doi pași'),
     [T('\\textbf{update}: $y_t$ arrives; correct $a_t$ by a share of the surprise $v_t = y_t - Z_ta_t$', '\\textbf{actualizarea}: sosește $y_t$; corectăm $a_t$ cu o fracțiune din surpriza $v_t = y_t - Z_ta_t$'),
      T('\\textbf{prediction}: push the filtered state through the transition equation to get $a_{t+1}$', '\\textbf{predicția}: trecem starea filtrată prin ecuația de tranziție și obținem $a_{t+1}$')]),
    T('With Gaussian errors, $a_{t|t}$ is the exact conditional mean; without normality it is still the best \\textbf{linear} estimate \\refDK', 'Cu erori gaussiene, $a_{t|t}$ este media condiționată exactă; fără normalitate rămîne cea mai bună estimare \\textbf{liniară} \\refDK'),
    T('Start: $a_1, P_1$; for a non-stationary state we use a \\textbf{diffuse} start ($P_1$ very large: we know nothing)', 'Pornirea: $a_1, P_1$; pentru o stare nestaționară folosim o pornire \\textbf{difuză} ($P_1$ foarte mare: nu știm nimic)')))

D.frame(T('The two steps in formulas', 'Cei doi pași în formule'), items(
    (T('\\textbf{Update} (when $y_t$ is observed)', '\\textbf{Actualizarea} (cînd $y_t$ este observat)'),
     [T('prediction error $v_t = y_t - Z_ta_t$, with variance $F_t = Z_tP_tZ_t\' + H$', 'eroarea de predicție $v_t = y_t - Z_ta_t$, cu varianța $F_t = Z_tP_tZ_t\' + H$'),
      T('\\textbf{Kalman gain} $K_t = P_tZ_t\'F_t^{-1}$', '\\textbf{cîștigul Kalman} $K_t = P_tZ_t\'F_t^{-1}$'),
      T('$a_{t|t} = a_t + K_tv_t$, \\quad $P_{t|t} = P_t - K_tZ_tP_t$', '$a_{t|t} = a_t + K_tv_t$, \\quad $P_{t|t} = P_t - K_tZ_tP_t$')]),
    (T('\\textbf{Prediction}', '\\textbf{Predicția}'),
     [T('$a_{t+1} = Ta_{t|t}$, \\quad $P_{t+1} = TP_{t|t}T\' + RQR\'$', '$a_{t+1} = Ta_{t|t}$, \\quad $P_{t+1} = TP_{t|t}T\' + RQR\'$')]),
    (T('\\textbf{Local level}: $F_t = P_t + \\sigma^2_\\varepsilon$, \\quad $K_t = P_t/(P_t + \\sigma^2_\\varepsilon)$', '\\textbf{Local level}: $F_t = P_t + \\sigma^2_\\varepsilon$, \\quad $K_t = P_t/(P_t + \\sigma^2_\\varepsilon)$'),
     [T('$a_{t+1} = a_t + K_t(y_t - a_t)$, \\quad $P_{t+1} = P_t(1 - K_t) + \\sigma^2_\\eta$', '$a_{t+1} = a_t + K_t(y_t - a_t)$, \\quad $P_{t+1} = P_t(1 - K_t) + \\sigma^2_\\eta$')]),
    T('Missing $y_t$: skip the update ($a_{t|t} = a_t$, $P_{t|t} = P_t$) and predict', 'Lipsește $y_t$: sărim peste actualizare ($a_{t|t} = a_t$, $P_{t|t} = P_t$) și facem predicția')))

D.frame(T('The Kalman gain', 'Cîștigul Kalman'), items(
    (T('Local level: $K_t = \\dfrac{P_t}{P_t + \\sigma^2_\\varepsilon}$ is the weight of the new observation, between 0 and 1', 'Local level: $K_t = \\dfrac{P_t}{P_t + \\sigma^2_\\varepsilon}$ este ponderea noii observații, între 0 și 1'),
     [T('$a_{t|t} = (1 - K_t)\\,a_t + K_t\\,y_t$: a weighted average of the prediction and the observation', '$a_{t|t} = (1 - K_t)\\,a_t + K_t\\,y_t$: o medie ponderată a predicției și a observației')]),
    (T('Two uncertainties compete', 'Două incertitudini concurează'),
     [T('noisy measurements (large $\\sigma^2_\\varepsilon$): small $K_t$, trust the model', 'măsurători zgomotoase ($\\sigma^2_\\varepsilon$ mare): $K_t$ mic, avem încredere în model'),
      T('uncertain state (large $P_t$, e.g.\\ at the start or after missing data): large $K_t$, trust the data', 'stare incertă ($P_t$ mare, de exemplu la început sau după valori lipsă): $K_t$ mare, avem încredere în date')]),
    (T('The variance always falls with an observation: $P_{t|t} = (1 - K_t)P_t \\le P_t$', 'Varianța scade întotdeauna după o observație: $P_{t|t} = (1 - K_t)P_t \\le P_t$'),
     [T('and grows by $\\sigma^2_\\eta$ in the prediction step: the two forces settle at a steady state', 'și crește cu $\\sigma^2_\\eta$ în pasul de predicție: cele două forțe se echilibrează într-o stare de echilibru')]),
    T('The same formula as the weighting of two independent estimates by their precisions $1/P_t$ and $1/\\sigma^2_\\varepsilon$', 'Aceeași formulă ca la ponderarea a două estimări independente după precizia lor, $1/P_t$ și $1/\\sigma^2_\\varepsilon$')))

TBW = T('$t$ & $y_t$ & $a_t$ & $P_t$ & $F_t$ & $K_t$ & $v_t$ & $a_{t|t}$ & $P_{t|t}$', '$t$ & $y_t$ & $a_t$ & $P_t$ & $F_t$ & $K_t$ & $v_t$ & $a_{t|t}$ & $P_{t|t}$')
ROWS = [f'{t} & @{{ex{t}.y}} & @{{ex{t}.a}} & @{{ex{t}.P}} & @{{ex{t}.F}} & @{{ex{t}.K}} & @{{ex{t}.v}} & @{{ex{t}.af}} & @{{ex{t}.Pf}}' for t in (1, 2, 3)]

D.frame(T('Worked example: three periods by hand (1/2)', 'Exemplu rezolvat: trei perioade de mînă (1/2)'), items(
    (T('Local level with $\\sigma^2_\\varepsilon = 4$, $\\sigma^2_\\eta = 1$ ($q = 0.25$); prior $a_1 = 10$, $P_1 = 4$; data $y = (12, 11, 14)$', 'Local level cu $\\sigma^2_\\varepsilon = 4$, $\\sigma^2_\\eta = 1$ ($q = 0{,}25$); a priori $a_1 = 10$, $P_1 = 4$; datele $y = (12, 11, 14)$'),
     [T('$t = 1$: $F_1 = 4 + 4 = 8$, $K_1 = 4/8 = 0.5$, $v_1 = 12 - 10 = 2$', '$t = 1$: $F_1 = 4 + 4 = 8$, $K_1 = 4/8 = 0{,}5$, $v_1 = 12 - 10 = 2$'),
      T('$a_{1|1} = 10 + 0.5 \\cdot 2 = 11$, $P_{1|1} = 4(1 - 0.5) = 2$; prediction $a_2 = 11$, $P_2 = 2 + 1 = 3$', '$a_{1|1} = 10 + 0{,}5 \\cdot 2 = 11$, $P_{1|1} = 4(1 - 0{,}5) = 2$; predicția $a_2 = 11$, $P_2 = 2 + 1 = 3$')]),
    (T('$t = 2$: $F_2 = 3 + 4 = 7$, $K_2 = 3/7 = @{ex2.K}$, $v_2 = 11 - 11 = 0$', '$t = 2$: $F_2 = 3 + 4 = 7$, $K_2 = 3/7 = @{ex2.K}$, $v_2 = 11 - 11 = 0$'),
     [T('no surprise: the level stays at $a_{2|2} = 11$, but the uncertainty falls to $P_{2|2} = 3 \\cdot 4/7 = @{ex2.Pf}$', 'nicio surpriză: nivelul rămîne $a_{2|2} = 11$, dar incertitudinea scade la $P_{2|2} = 3 \\cdot 4/7 = @{ex2.Pf}$'),
      T('prediction $a_3 = 11$, $P_3 = @{ex2.Pf} + 1 = @{ex3.P}$', 'predicția $a_3 = 11$, $P_3 = @{ex2.Pf} + 1 = @{ex3.P}$')]),
    T('The gain falls from 0.5 to @{ex2.K}: the filter learns about the level and relies less on each new observation', 'Cîștigul scade de la 0,5 la @{ex2.K}: filtrul învață despre nivel și se bazează mai puțin pe fiecare observație nouă')))

D.frame(T('Worked example: three periods by hand (2/2)', 'Exemplu rezolvat: trei perioade de mînă (2/2)'), table(
    'ccccccccc', TBW, ROWS, size='footnotesize') + items(
    (T('$t = 3$: $K_3 = @{ex3.P}/@{ex3.F} = @{ex3.K}$, $v_3 = 14 - 11 = 3$, $a_{3|3} = 11 + @{ex3.K} \\cdot 3 = @{ex3.af}$', '$t = 3$: $K_3 = @{ex3.P}/@{ex3.F} = @{ex3.K}$, $v_3 = 14 - 11 = 3$, $a_{3|3} = 11 + @{ex3.K} \\cdot 3 = @{ex3.af}$'),
     [T('forecast for $t = 4$: $a_4 = @{ex.anext}$ with $P_4 = @{ex.Pnext}$; the forecast of $y_4$ has variance $P_4 + \\sigma^2_\\varepsilon$', 'prognoza pentru $t = 4$: $a_4 = @{ex.anext}$, cu $P_4 = @{ex.Pnext}$; prognoza lui $y_4$ are varianța $P_4 + \\sigma^2_\\varepsilon$')]),
    (T('The gain keeps falling towards the steady state $\\bar K = @{ex.ssK}$ ($\\bar P = @{ex.ssP}$, next slide)', 'Cîștigul continuă să scadă spre starea de echilibru $\\bar K = @{ex.ssK}$ ($\\bar P = @{ex.ssP}$, slide-ul următor)'),
     [T('the surprise of 3 moves the level by only $@{ex3.K} \\cdot 3$: part of it is treated as noise', 'surpriza de 3 mută nivelul doar cu $@{ex3.K} \\cdot 3$: o parte din ea este tratată ca zgomot')])), size='small')

D.frame(T('Steady state and exponential smoothing', 'Starea de echilibru și netezirea exponențială'), items(
    (T('For the local level model, $P_t$ converges to the solution of $\\bar P = \\bar P(1 - \\bar K) + \\sigma^2_\\eta$ with $\\bar K = \\bar P/(\\bar P + \\sigma^2_\\varepsilon)$', 'Pentru modelul local level, $P_t$ converge la soluția ecuației $\\bar P = \\bar P(1 - \\bar K) + \\sigma^2_\\eta$, cu $\\bar K = \\bar P/(\\bar P + \\sigma^2_\\varepsilon)$'),
     [T('solution: $\\bar P = \\sigma^2_\\varepsilon\\,\\dfrac{q + \\sqrt{q^2 + 4q}}{2}$, \\quad $\\bar K = \\dfrac{\\bar P}{\\bar P + \\sigma^2_\\varepsilon}$', 'soluția: $\\bar P = \\sigma^2_\\varepsilon\\,\\dfrac{q + \\sqrt{q^2 + 4q}}{2}$, \\quad $\\bar K = \\dfrac{\\bar P}{\\bar P + \\sigma^2_\\varepsilon}$'),
      T('example: $q = 0.25$ gives $\\bar P/\\sigma^2_\\varepsilon = (0.25 + \\sqrt{1.0625})/2 = @{ex.ssP}/4$ and $\\bar K = @{ex.ssK}$', 'exemplu: $q = 0{,}25$ dă $\\bar P/\\sigma^2_\\varepsilon = (0{,}25 + \\sqrt{1{,}0625})/2 = @{ex.ssP}/4$ și $\\bar K = @{ex.ssK}$')]),
    (T('In the steady state: $a_{t+1} = a_t + \\bar K(y_t - a_t) = \\bar K y_t + (1 - \\bar K)a_t$', 'În starea de echilibru: $a_{t+1} = a_t + \\bar K(y_t - a_t) = \\bar K y_t + (1 - \\bar K)a_t$'),
     [T('this is SES with $\\alpha = \\bar K$: exponential smoothing is the Kalman filter of a local level model \\refMuth', 'aceasta este SES cu $\\alpha = \\bar K$: netezirea exponențială este filtrul Kalman al unui model local level \\refMuth')]),
    (T('Inverse map: an SES weight $\\alpha$ corresponds to $q = \\alpha^2/(1 - \\alpha)$', 'Relația inversă: o pondere SES $\\alpha$ corespunde lui $q = \\alpha^2/(1 - \\alpha)$'),
     [T('the state space version adds what SES lacks: a likelihood for $q$, forecast intervals, missing values', 'versiunea în spațiul stărilor adaugă ce îi lipsește SES: o verosimilitate pentru $q$, intervale de prognoză, valori lipsă')])))

chart(T('The Nile: the Kalman filter of the local level model', 'Nilul: filtrul Kalman pentru modelul local level'), 'tsa_ch10_nile_filter', 'TSA_ch10_kalman_filter', [
    T('Annual flow at Aswan, 1871--1970 ($10^8$ m$^3$, statsmodels); variances by maximum likelihood with a diffuse start: $\\hat\\sigma^2_\\varepsilon = @{ni.se}$, $\\hat\\sigma^2_\\eta = @{ni.sn}$, $\\hat q = @{ni.q}$, the estimates of \\refDK',
      'Debitul anual la Aswan, 1871--1970 ($10^8$ m$^3$, statsmodels); varianțele prin verosimilitate maximă, cu pornire difuză: $\\hat\\sigma^2_\\varepsilon = @{ni.se}$, $\\hat\\sigma^2_\\eta = @{ni.sn}$, $\\hat q = @{ni.q}$, estimările din \\refDK')],
    h='0.58\\textheight')

interp(('the Nile filter', 'filtrului pentru Nil'), [
    (T('Most of the year-to-year movement is noise: $\\hat q = @{ni.q}$, so each flow moves the level by about a quarter of the surprise ($\\bar K = @{ni.ssK}$)', 'Cea mai mare parte a mișcării de la un an la altul este zgomot: $\\hat q = @{ni.q}$, deci fiecare debit mută nivelul cu aproximativ un sfert din surpriză ($\\bar K = @{ni.ssK}$)'),
     [T('\\texttt{statsmodels} \\texttt{UnobservedComponents(\'llevel\')} gives @{smn.se} and @{smn.sn}: the same model, a slightly different start', '\\texttt{statsmodels} \\texttt{UnobservedComponents(\'llevel\')} dă @{smn.se} și @{smn.sn}: același model, o pornire puțin diferită')]),
    (T('1899: the flow is @{ni.y99}, the predicted level @{ni.a98}; the surprise $v = @{ni.v99}$ lowers the level only to @{ni.a99}', '1899: debitul este @{ni.y99}, nivelul prezis @{ni.a98}; surpriza $v = @{ni.v99}$ coboară nivelul doar la @{ni.a99}'),
     [T('the filter needs several low years to accept the new level (@{ni.a05} in 1905): the first Aswan dam was completed in 1902', 'filtrul are nevoie de mai mulți ani slabi pentru a accepta noul nivel (@{ni.a05} în 1905): primul baraj de la Aswan a fost finalizat în 1902')]),
    T('In 1970 the level is @{ni.alast} with standard error @{ni.sdlast}, well below the sample mean @{ni.ymean}', 'În 1970 nivelul este @{ni.alast}, cu eroarea standard @{ni.sdlast}, mult sub media eșantionului, @{ni.ymean}')])

chart(T('The gain converges; SES reproduces the filter', 'Cîștigul converge; SES reproduce filtrul'), 'tsa_ch10_gain_ses', 'TSA_ch10_kalman_filter', [
    T('Left: $K_t$ of the Nile filter; right: the Kalman one-step forecasts $a_{t}$ and SES with $\\alpha = \\bar K$, started at the first observation',
      'Stînga: $K_t$ al filtrului pentru Nil; dreapta: prognozele Kalman la un pas $a_{t}$ și SES cu $\\alpha = \\bar K$, pornită de la prima observație')],
    h='0.66\\textheight')

interp(('the gain', 'cîștigului'), [
    (T('$K_t$ starts at @{ni.K2} (after the diffuse start), @{ni.K5} in year 5, and is within 0.001 of $\\bar K = @{ni.ssK}$ after @{ga.nclose} years', '$K_t$ pornește de la @{ni.K2} (după pornirea difuză), @{ni.K5} în anul 5 și ajunge la mai puțin de 0,001 de $\\bar K = @{ni.ssK}$ după @{ga.nclose} ani'),
     [T('the forecasts differ by @{ga.gap5} in year 5 and by less than @{ga.gap20} after 20 years', 'prognozele diferă cu @{ga.gap5} în anul 5 și cu mai puțin de @{ga.gap20} după 20 de ani')]),
    (T('SES fitted directly by least squares (Chapter 0) gives $\\hat\\alpha = @{ga.ses}$, i.e.\\ $q = \\alpha^2/(1-\\alpha) = @{ga.qa}$', 'SES estimată direct prin cele mai mici pătrate (Capitolul 0) dă $\\hat\\alpha = @{ga.ses}$, adică $q = \\alpha^2/(1-\\alpha) = @{ga.qa}$'),
     [T('close to the likelihood answer: two criteria, one model', 'aproape de răspunsul verosimilității: două criterii, un singur model')]),
    T('The state space view explains SES: $\\alpha$ is large when the level moves a lot relative to the noise', 'Perspectiva spațiului stărilor explică SES: $\\alpha$ este mare cînd nivelul se mișcă mult în raport cu zgomotul')])

D.recap(('The Kalman filter', 'filtrul Kalman'), [
    T('Update: $v_t = y_t - Z_ta_t$, $F_t = Z_tP_tZ_t\' + H$, $K_t = P_tZ_t\'F_t^{-1}$, $a_{t|t} = a_t + K_tv_t$', 'Actualizarea: $v_t = y_t - Z_ta_t$, $F_t = Z_tP_tZ_t\' + H$, $K_t = P_tZ_t\'F_t^{-1}$, $a_{t|t} = a_t + K_tv_t$'),
    T('Prediction: $a_{t+1} = Ta_{t|t}$, $P_{t+1} = TP_{t|t}T\' + RQR\'$', 'Predicția: $a_{t+1} = Ta_{t|t}$, $P_{t+1} = TP_{t|t}T\' + RQR\'$'),
    T('The gain weighs the data against the model; it settles at a steady state', 'Cîștigul cîntărește datele în raport cu modelul; se stabilizează într-o stare de echilibru'),
    T('Local level in steady state = SES with $\\alpha = \\bar K$; Nile: $\\bar K = @{ni.ssK}$', 'Local level în starea de echilibru = SES cu $\\alpha = \\bar K$; Nilul: $\\bar K = @{ni.ssK}$')])

# =============================================================================
# 4. NETEZIRE, VEROSIMILITATE, VALORI LIPSĂ
# =============================================================================
D.section('Smoothing, likelihood and missing data', 'Netezire, verosimilitate și valori lipsă')

D.frame(T('Smoothing: using the whole sample', 'Netezirea: folosim tot eșantionul'), items(
    (T('\\textbf{Filtering} uses $Y_t$ (the past and the present): the real-time estimate', '\\textbf{Filtrarea} folosește $Y_t$ (trecutul și prezentul): estimarea în timp real'),
     [T('\\textbf{smoothing} uses $Y_n$ (the whole sample): $\\hat\\alpha_t = E(\\alpha_t \\mid Y_n)$, with variance $V_t \\le P_{t|t}$', '\\textbf{netezirea} folosește $Y_n$ (tot eșantionul): $\\hat\\alpha_t = E(\\alpha_t \\mid Y_n)$, cu varianța $V_t \\le P_{t|t}$')]),
    (T('\\textbf{Rauch--Tung--Striebel} smoother \\refRTS: one backward pass after the filter, from $t = n-1$ to 1', 'Netezitorul \\textbf{Rauch--Tung--Striebel} \\refRTS: o trecere înapoi după filtru, de la $t = n-1$ la 1'),
     [T('$n$: the sample size; $J_t$: the smoother gain, the weight of the revision coming from the future', '$n$: volumul eșantionului; $J_t$: cîștigul netezitorului, ponderea revizuirii care vine din viitor'),
      T('$J_t = P_{t|t}T\'P_{t+1}^{-1}$, \\quad $\\hat\\alpha_t = a_{t|t} + J_t(\\hat\\alpha_{t+1} - a_{t+1})$', '$J_t = P_{t|t}T\'P_{t+1}^{-1}$, \\quad $\\hat\\alpha_t = a_{t|t} + J_t(\\hat\\alpha_{t+1} - a_{t+1})$'),
      T('$V_t = P_{t|t} + J_t(V_{t+1} - P_{t+1})J_t\'$; at $t = n$ smoothed and filtered coincide', '$V_t = P_{t|t} + J_t(V_{t+1} - P_{t+1})J_t\'$; la $t = n$ estimarea netezită și cea filtrată coincid')]),
    T('Use: the filtered state for decisions in real time and forecasting; the smoothed state for history (dating breaks, the past output gap)', 'Utilizare: starea filtrată pentru decizii în timp real și prognoză; starea netezită pentru istorie (datarea rupturilor, deviația PIB din trecut)')))

chart(T('The Nile: filtered against smoothed level', 'Nilul: nivelul filtrat și nivelul netezit'), 'tsa_ch10_nile_smooth', 'TSA_ch10_smoothing_likelihood', [
    T('Filtered level $a_{t|t}$ and smoothed level $\\hat\\alpha_t$ with its 90\\% band; the same maximum-likelihood variances', 'Nivelul filtrat $a_{t|t}$ și nivelul netezit $\\hat\\alpha_t$, cu banda de 90\\%; aceleași varianțe de verosimilitate maximă')],
    h='0.58\\textheight')

interp(('the smoothed level', 'nivelului netezit'), [
    (T('The smoothed level reacts to the 1898--1899 drop \\textbf{before} it happens in the data: in 1897 it is already @{smo.s1897}, against @{smo.f1897} filtered', 'Nivelul netezit reacționează la scăderea din 1898--1899 \\textbf{înainte} ca ea să apară în date: în 1897 este deja @{smo.s1897}, față de @{smo.f1897} filtrat'),
     [T('it knows the future: useful for history, impossible in real time', 'folosește informație din viitor: util pentru istorie, imposibil în timp real')]),
    (T('In 1900: smoothed @{smo.s1900}, filtered @{smo.f1900}; the smoothed level adapts faster because it sees the low years after 1900', 'În 1900: netezit @{smo.s1900}, filtrat @{smo.f1900}; nivelul netezit se adaptează mai repede, pentru că vede anii slabi de după 1900'),
     [T('standard error in 1920: @{smo.sd_s_mid} smoothed against @{smo.sd_f_mid} filtered', 'eroarea standard în 1920: @{smo.sd_s_mid} netezit, față de @{smo.sd_f_mid} filtrat')]),
    T('The smoothed path shows a step near 1899 more clearly than any moving average of Chapter 0', 'Traiectoria netezită arată o treaptă în jurul anului 1899 mai clar decît orice medie mobilă din Capitolul 0')])

D.frame(T('The likelihood: prediction-error decomposition', 'Verosimilitatea: descompunerea erorilor de predicție'), items(
    (T('The joint density factorises into one-step conditional densities: $p(y_1, \\dots, y_n) = \\prod_t p(y_t \\mid Y_{t-1})$', 'Densitatea comună se descompune în densități condiționate la un pas: $p(y_1, \\dots, y_n) = \\prod_t p(y_t \\mid Y_{t-1})$'),
     [T('with Gaussian errors, $y_t \\mid Y_{t-1} \\sim N(Z_ta_t, F_t)$: the filter delivers the mean and the variance', 'cu erori gaussiene, $y_t \\mid Y_{t-1} \\sim N(Z_ta_t, F_t)$: filtrul dă media și varianța')]),
    (T('\\textbf{Prediction-error decomposition}', '\\textbf{Descompunerea erorilor de predicție}'),
     [T('$\\log L(\\theta) = -\\dfrac{n}{2}\\log 2\\pi - \\dfrac12\\sum_{t=1}^{n}\\left(\\log F_t + \\dfrac{v_t^2}{F_t}\\right)$', '$\\log L(\\theta) = -\\dfrac{n}{2}\\log 2\\pi - \\dfrac12\\sum_{t=1}^{n}\\left(\\log F_t + \\dfrac{v_t^2}{F_t}\\right)$'),
      T('$\\theta$: the parameters (the variances of the model); $v_t$, $F_t$: the prediction errors and their variances, from the filter', '$\\theta$: parametrii (varianțele modelului); $v_t$, $F_t$: erorile de predicție și varianțele lor, date de filtru'),
      T('one run of the filter gives $\\log L$ for one value of $\\theta$; a numerical optimiser searches over $\\theta$', 'o rulare a filtrului dă $\\log L$ pentru o valoare a lui $\\theta$; un optimizator numeric caută după $\\theta$')]),
    T('Diffuse start: the first $d$ terms (here $d = 1$ for the level) are left out, since $F_t$ is huge there \\refDK', 'Pornirea difuză: primii $d$ termeni (aici $d = 1$ pentru nivel) sînt omiși, deoarece acolo $F_t$ este uriaș \\refDK'),
    T('The same idea as the exact ARMA likelihood of Chapter 2, now for any state space model', 'Aceeași idee ca verosimilitatea ARMA exactă din Capitolul 2, acum pentru orice model în spațiul stărilor')))

D.frame(T('Maximum likelihood in practice', 'Verosimilitatea maximă în practică'), items(
    (T('Optimise over $\\log\\sigma^2$ (variances stay positive); several starting values, since the surface can be flat', 'Optimizăm după $\\log\\sigma^2$ (varianțele rămîn pozitive); mai multe valori de pornire, deoarece suprafața poate fi plată'),
     [T('a variance estimated at zero is common and meaningful: that component is deterministic', 'o varianță estimată la zero este frecventă și are sens: acea componentă este deterministă')]),
    (T('\\textbf{Concentrating}: for the local level, write $\\sigma^2_\\eta = q\\sigma^2_\\varepsilon$; for fixed $q$, $\\hat\\sigma^2_\\varepsilon = \\frac{1}{n-1}\\sum_t v_t^2/F_t$ in closed form', '\\textbf{Concentrarea}: pentru local level scriem $\\sigma^2_\\eta = q\\sigma^2_\\varepsilon$; pentru $q$ fixat, $\\hat\\sigma^2_\\varepsilon = \\frac{1}{n-1}\\sum_t v_t^2/F_t$ are formă închisă'),
     [T('the profile log-likelihood is then a function of $q$ alone (next chart)', 'log-verosimilitatea profil devine atunci o funcție doar de $q$ (graficul următor)')]),
    (T('Standard errors from the numerical Hessian; model choice by AIC and BIC; tests on the standardised prediction errors $e_t = v_t/\\sqrt{F_t}$', 'Erorile standard din hessiana numerică; alegerea modelului după AIC și BIC; teste pe erorile de predicție standardizate $e_t = v_t/\\sqrt{F_t}$'),
     [T('\\texttt{statsmodels}: \\texttt{UnobservedComponents}, \\texttt{SARIMAX}, \\texttt{DynamicFactor} all use this machinery', '\\texttt{statsmodels}: \\texttt{UnobservedComponents}, \\texttt{SARIMAX}, \\texttt{DynamicFactor} folosesc toate acest mecanism')])))

chart(T('Profile likelihood and the choice of $q$', 'Verosimilitatea profil și alegerea lui $q$'), 'tsa_ch10_likelihood', 'TSA_ch10_smoothing_likelihood', [
    T('Left: profile log-likelihood of the Nile local level model against $q$ (log scale); right: filtered levels for $q = 0.001$, the ML value and $q = 1$',
      'Stînga: log-verosimilitatea profil a modelului local level pentru Nil, în funcție de $q$ (scară logaritmică); dreapta: nivelurile filtrate pentru $q = 0{,}001$, valoarea ML și $q = 1$')],
    h='0.66\\textheight')

interp(('the profile likelihood', 'verosimilității profil'), [
    (T('A clear maximum at $\\hat q = @{ni.q}$, with $\\hat\\sigma^2_\\varepsilon = @{li.s2}$ from the closed form', 'Un maxim clar la $\\hat q = @{ni.q}$, cu $\\hat\\sigma^2_\\varepsilon = @{li.s2}$ din forma închisă'),
     [T('likelihood-ratio statistic against $q = 0.001$ (an almost constant level): @{li.lr001}; against $q = 1$: @{li.lr1}', 'statistica raportului de verosimilitate față de $q = 0{,}001$ (un nivel aproape constant): @{li.lr001}; față de $q = 1$: @{li.lr1}')]),
    (T('Small $q$: the level is almost flat and misses the drop after 1899; large $q$: the level chases every flood', '$q$ mic: nivelul este aproape plat și nu surprinde scăderea de după 1899; $q$ mare: nivelul urmărește fiecare viitură'),
     [T('the likelihood picks the compromise that predicts best one step ahead', 'verosimilitatea alege compromisul care prezice cel mai bine la un pas')]),
    T('Choosing $q$ is the same as choosing the SES weight $\\alpha$ in Chapter 0, now with a statistical criterion', 'Alegerea lui $q$ este echivalentă cu alegerea ponderii SES $\\alpha$ din Capitolul 0, acum cu un criteriu statistic')])

D.frame(T('Diagnostics', 'Diagnosticare'), items(
    (T('If the model is right, the standardised prediction errors $e_t = v_t/\\sqrt{F_t}$ are i.i.d. $N(0, 1)$', 'Dacă modelul este corect, erorile de predicție standardizate $e_t = v_t/\\sqrt{F_t}$ sînt i.i.d. $N(0, 1)$'),
     [T('the checks of Chapters 1--2: the ACF and the Ljung--Box test (independence), Jarque--Bera (normality), a plot over time (outliers)', 'verificările din Capitolele 1--2: ACF și testul Ljung--Box (independență), Jarque--Bera (normalitate), graficul în timp (valori extreme)')]),
    (T('\\textbf{Auxiliary residuals} \\refDK: the smoothed disturbances $\\hat\\varepsilon_t$ and $\\hat\\eta_t$, standardised', '\\textbf{Reziduurile auxiliare} \\refDK: perturbațiile netezite $\\hat\\varepsilon_t$ și $\\hat\\eta_t$, standardizate'),
     [T('a large $\\hat\\varepsilon_t$: an \\textbf{outlier} in one observation', 'un $\\hat\\varepsilon_t$ mare: o \\textbf{valoare extremă} într-o singură observație'),
      T('a large $\\hat\\eta_t$: a \\textbf{break} in the level, a permanent shift', 'un $\\hat\\eta_t$ mare: o \\textbf{ruptură} în nivel, o deplasare permanentă')]),
    T('Computed by a backward recursion (the disturbance smoother) after the filter', 'Se calculează printr-o recursie înapoi (netezitorul perturbațiilor) după filtru')))

chart(T('Diagnostics of the Nile model', 'Diagnosticarea modelului pentru Nil'), 'tsa_ch10_diagnostics', 'TSA_ch10_smoothing_likelihood', [
    T('Left: standardised one-step prediction errors $e_t$; right: standardised auxiliary residual of the level, $\\hat\\eta_t$ (a shock between year $t$ and $t+1$); dashed: $\\pm 1.96$',
      'Stînga: erorile de predicție la un pas standardizate $e_t$; dreapta: reziduul auxiliar standardizat al nivelului, $\\hat\\eta_t$ (un șoc între anul $t$ și $t+1$); linie punctată: $\\pm 1{,}96$')],
    h='0.66\\textheight')

interp(('the diagnostics', 'diagnosticării'), [
    (T('Prediction errors: Ljung--Box $Q(10) = @{dg.lb}$ (p = @{dg.lbp}), Jarque--Bera @{dg.jb} (p = @{dg.jbp}); @{dg.nout} of @{dg.ne} outside $\\pm 1.96$', 'Erorile de predicție: Ljung--Box $Q(10) = @{dg.lb}$ (p = @{dg.lbp}), Jarque--Bera @{dg.jb} (p = @{dg.jbp}); @{dg.nout} din @{dg.ne} în afara intervalului $\\pm 1{,}96$'),
     [T('no evidence against the model in the one-step errors', 'nicio dovadă împotriva modelului în erorile la un pas')]),
    (T('The level residual has its minimum in @{dg.levy} (@{dg.lev}): the shift of the level between 1898 and 1899', 'Reziduul nivelului are minimul în @{dg.levy} (@{dg.lev}): deplasarea nivelului între 1898 și 1899'),
     [T('the observation residual has its minimum in @{dg.obsy} (@{dg.obs}): a single very dry year, an outlier', 'reziduul observației are minimul în @{dg.obsy} (@{dg.obs}): un singur an foarte secetos, o valoare extremă')]),
    T('The same break as in Chapter 8 (spurious long memory of the Nile), now located by the model itself', 'Aceeași ruptură ca în Capitolul 8 (memoria lungă aparentă a Nilului), acum localizată de model însuși')])

D.frame(T('Missing values and forecasting', 'Valori lipsă și prognoză'), items(
    (T('A missing $y_t$: no update; $a_{t+1} = Ta_t$, $P_{t+1} = TP_tT\' + RQR\'$', 'Un $y_t$ lipsă: fără actualizare; $a_{t+1} = Ta_t$, $P_{t+1} = TP_tT\' + RQR\'$'),
     [T('the likelihood simply skips that term; nothing is interpolated by hand', 'verosimilitatea omite pur și simplu acel termen; nu interpolăm nimic de mînă')]),
    (T('Forecasting $h$ steps ahead = treating $y_{n+1}, \\dots, y_{n+h}$ as missing', 'Prognoza la $h$ pași = tratarea lui $y_{n+1}, \\dots, y_{n+h}$ ca valori lipsă'),
     [T('local level: $\\hat y_{n+h} = a_{n+1}$ for every $h$, with variance $P_{n+1} + (h-1)\\sigma^2_\\eta + \\sigma^2_\\varepsilon$', 'local level: $\\hat y_{n+h} = a_{n+1}$ pentru orice $h$, cu varianța $P_{n+1} + (h-1)\\sigma^2_\\eta + \\sigma^2_\\varepsilon$')]),
    (T('Uses: irregular data (holidays), series that start at different dates, mixed frequencies, the ragged edge of a nowcast', 'Utilizări: date neregulate (sărbători), serii care încep la date diferite, frecvențe mixte, marginea neregulată a unui nowcast'),
     [T('the smoother fills the gaps with $\\hat\\alpha_t$ and the corresponding standard error', 'netezitorul umple golurile cu $\\hat\\alpha_t$ și cu eroarea standard corespunzătoare')])))

chart(T('Missing years and forecasts of the Nile', 'Ani lipsă și prognoze pentru Nil'), 'tsa_ch10_missing', 'TSA_ch10_missing_data', [
    T('Left: 1891--1910 and 1931--1950 removed (the experiment of \\refDK), smoothed level with 90\\% band; right: forecasts for 1971--2000 with 90\\% intervals for $y$',
      'Stînga: anii 1891--1910 și 1931--1950 eliminați (experimentul din \\refDK), nivelul netezit cu banda de 90\\%; dreapta: prognozele pentru 1971--2000, cu intervale de 90\\% pentru $y$')],
    h='0.66\\textheight')

interp(('the missing data', 'valorilor lipsă'), [
    (T('Inside a gap the smoothed level is a straight line between the two edges; its standard error grows to @{mi.gap} in 1900, against @{mi.full} with all data', 'În interiorul unui gol nivelul netezit este o dreaptă între cele două margini; eroarea standard crește la @{mi.gap} în 1900, față de @{mi.full} cu toate datele'),
     [T('the band widens towards the middle of the gap and narrows near observed years', 'banda se lărgește spre mijlocul golului și se îngustează lîngă anii observați')]),
    (T('Forecasts are flat at @{mi.f1}: a random-walk level has no predictable direction', 'Prognozele sînt constante, la @{mi.f1}: un nivel de tip mers aleator nu are o direcție previzibilă'),
     [T('standard error of $y$: @{mi.sdf1} one year ahead, @{mi.sdf30} after 30 years', 'eroarea standard a lui $y$: @{mi.sdf1} la un an, @{mi.sdf30} după 30 de ani')]),
    T('The same code handles missing data, forecasts and estimation: one of the main practical strengths of state space models', 'Același cod tratează valorile lipsă, prognozele și estimarea: unul dintre principalele avantaje practice ale modelelor în spațiul stărilor')])

D.recap(('Smoothing, likelihood and missing data', 'netezire, verosimilitate și valori lipsă'), [
    T('Filtered: real time; smoothed (RTS): the whole sample, smaller variance', 'Filtrat: timp real; netezit (RTS): tot eșantionul, varianță mai mică'),
    T('$\\log L = -\\frac12\\sum(\\log 2\\pi + \\log F_t + v_t^2/F_t)$, maximised numerically', '$\\log L = -\\frac12\\sum(\\log 2\\pi + \\log F_t + v_t^2/F_t)$, maximizată numeric'),
    T('Diagnostics on $e_t$; auxiliary residuals find outliers and breaks (Nile 1898--1899)', 'Diagnosticare pe $e_t$; reziduurile auxiliare găsesc valori extreme și rupturi (Nilul, 1898--1899)'),
    T('Missing data and forecasts: skip the update step', 'Valori lipsă și prognoze: sărim peste pasul de actualizare')])

# =============================================================================
# 5. COMPONENTE NEOBSERVATE, TREND ȘI CICLU
# =============================================================================
D.section('Trend, cycle and common factors', 'Trend, ciclu și factori comuni')

D.frame(T('Unobserved components models', 'Modele cu componente neobservate'), items(
    (T('\\textbf{Structural time series} or \\textbf{unobserved components} (UC) model \\refHarvey: $y_t = \\mu_t + \\psi_t + \\gamma_t + \\varepsilon_t$', 'Modelul \\textbf{structural de serii de timp} sau \\textbf{cu componente neobservate} (UC) \\refHarvey: $y_t = \\mu_t + \\psi_t + \\gamma_t + \\varepsilon_t$'),
     [T('trend $\\mu_t$ (local level or local linear trend), cycle $\\psi_t$, seasonal $\\gamma_t$, irregular $\\varepsilon_t$', 'trend $\\mu_t$ (local level sau local linear trend), ciclu $\\psi_t$, sezonalitate $\\gamma_t$, componenta neregulată $\\varepsilon_t$'),
      T('the decomposition of Chapter 0, but each component is a stochastic process with its own variance', 'descompunerea din Capitolul 0, dar fiecare componentă este un proces stochastic cu propria varianță')]),
    (T('The \\textbf{cycle}: a stationary AR(2) $\\psi_t = \\phi_1\\psi_{t-1} + \\phi_2\\psi_{t-2} + \\kappa_t$, or a damped stochastic cycle with period $2\\pi/\\lambda_c$', '\\textbf{Ciclul}: un AR(2) staționar $\\psi_t = \\phi_1\\psi_{t-1} + \\phi_2\\psi_{t-2} + \\kappa_t$ sau un ciclu stochastic amortizat cu perioada $2\\pi/\\lambda_c$'),
     [T('$\\phi_1$, $\\phi_2$: the AR coefficients of the cycle; $\\kappa_t$: the cycle shock; $\\lambda_c$: the frequency of the cycle, in radians', '$\\phi_1$, $\\phi_2$: coeficienții AR ai ciclului; $\\kappa_t$: șocul ciclului; $\\lambda_c$: frecvența ciclului, în radiani'),
      T('for log GDP, $\\psi_t$ is the \\textbf{output gap}: the percentage deviation of output from its trend (potential)', 'pentru logaritmul PIB, $\\psi_t$ este \\textbf{deviația PIB} (output gap): abaterea procentuală a producției de la trend (potențial)')]),
    T('In Python: \\texttt{UnobservedComponents(y, level=\'smooth trend\', autoregressive=2)}; the whole state is estimated by the Kalman filter and smoother',
      'În Python: \\texttt{UnobservedComponents(y, level=\'smooth trend\', autoregressive=2)}; toată starea se estimează cu filtrul și netezitorul Kalman')))

D.frame(T('The HP filter is a state space smoother', 'Filtrul HP este un netezitor în spațiul stărilor'), items(
    (T('\\refHPf: the trend $\\tau_t$ minimises $\\sum_t(y_t - \\tau_t)^2 + \\lambda\\sum_t(\\Delta^2\\tau_t)^2$; $\\lambda = 1600$ for quarterly data', '\\refHPf: trendul $\\tau_t$ minimizează $\\sum_t(y_t - \\tau_t)^2 + \\lambda\\sum_t(\\Delta^2\\tau_t)^2$; $\\lambda = 1600$ pentru date trimestriale'),
     [T('$\\Delta^2\\tau_t = \\tau_t - 2\\tau_{t-1} + \\tau_{t-2}$: the change in the slope of the trend; $\\lambda \\ge 0$: the smoothing parameter', '$\\Delta^2\\tau_t = \\tau_t - 2\\tau_{t-1} + \\tau_{t-2}$: variația pantei trendului; $\\lambda \\ge 0$: parametrul de netezire'),
      T('large $\\lambda$: a straight-line trend; $\\lambda = 0$: the trend is the series itself', '$\\lambda$ mare: un trend liniar; $\\lambda = 0$: trendul este chiar seria')]),
    (T('This is exactly the \\textbf{smoothed} trend of a UC model: smooth trend ($\\Delta^2\\mu_{t+1} = \\zeta_t$) plus white noise, with $\\lambda = \\sigma^2_\\varepsilon/\\sigma^2_\\zeta$ \\refHJ', 'Este exact trendul \\textbf{netezit} al unui model UC: trend neted ($\\Delta^2\\mu_{t+1} = \\zeta_t$) plus zgomot alb, cu $\\lambda = \\sigma^2_\\varepsilon/\\sigma^2_\\zeta$ \\refHJ'),
     [T('the HP filter imposes the signal-to-noise ratio and a white-noise gap; the UC model estimates both', 'filtrul HP impune raportul semnal--zgomot și o deviație de tip zgomot alb; modelul UC le estimează pe amîndouă')]),
    T('Being a smoother, HP uses future data: its last values are revised as new quarters arrive', 'Fiind un netezitor, HP folosește date viitoare: ultimele lui valori se revizuiesc pe măsură ce sosesc trimestre noi')))

D.frame(T('Hamilton (2018): why you should never use the HP filter', 'Hamilton (2018): argumentele împotriva filtrului HP'), items(
    (T('\\refHamHP\\ lists three problems', '\\refHamHP\\ enumeră trei probleme'),
     [T('the HP gap of a random walk has cycles that are not in the data: spurious dynamics', 'deviația HP a unui mers aleator are cicluri care nu există în date: dinamică aparentă'),
      T('the end of the sample is treated differently from the middle: large revisions, misleading real-time gaps', 'sfîrșitul eșantionului este tratat altfel decît mijlocul: revizuiri mari, deviații înșelătoare în timp real'),
      T('$\\lambda = 1600$ is a convention, far from the value a likelihood would choose', '$\\lambda = 1600$ este o convenție, departe de valoarea pe care ar alege-o o verosimilitate')]),
    (T('His alternative, the \\textbf{regression filter}: OLS of $y_{t+h}$ on $1, y_t, y_{t-1}, y_{t-2}, y_{t-3}$ with $h = 8$ quarters', 'Alternativa lui, \\textbf{filtrul de regresie}: OLS a lui $y_{t+h}$ pe $1, y_t, y_{t-1}, y_{t-2}, y_{t-3}$, cu $h = 8$ trimestre'),
     [T('the cycle is the residual: what could not be predicted two years earlier; one-sided by construction', 'ciclul este reziduul: ceea ce nu putea fi prezis cu doi ani înainte; unilateral prin construcție')]),
    T('No method is the truth: we compare the UC model, HP and the Hamilton filter on Romanian GDP', 'Nicio metodă nu dă valoarea adevărată: comparăm modelul UC, HP și filtrul Hamilton pe PIB-ul României')))

chart(T('Romanian GDP: trend', 'PIB-ul României: trendul'), 'tsa_ch10_ro_trend', 'TSA_ch10_trend_cycle', [
    T('$100 \\times \\log$ of real GDP, quarterly, seasonally and calendar adjusted, chain-linked volumes (Eurostat namq\\_10\\_gdp), @{tr.first}--@{tr.last} ($T = @{tr.n}$); UC model: smooth trend plus AR(2) cycle; HP with $\\lambda = 1600$',
      '$100 \\times \\log$ din PIB-ul real, trimestrial, ajustat sezonier și cu numărul de zile lucrătoare, volume înlănțuite (Eurostat namq\\_10\\_gdp), @{tr.first}--@{tr.last} ($T = @{tr.n}$); modelul UC: trend neted plus ciclu AR(2); HP cu $\\lambda = 1600$')],
    h='0.58\\textheight')

interp(('the trend', 'trendului'), [
    (T('Average growth @{tr.avg}\\% per year, with a slow start (the 1997--1999 recession), a boom until 2008, a stagnation and a recovery after 2013', 'Creștere medie de @{tr.avg}\\% pe an, cu un început lent (recesiunea din 1997--1999), un boom pînă în 2008, o stagnare și o revenire după 2013'),
     [T('the two trends are almost identical: the difference between methods lies in the gap, not in the trend', 'cele două trenduri sînt aproape identice: diferența dintre metode se află în deviație, nu în trend')]),
    (T('UC estimates: $\\hat\\sigma^2_\\zeta = @{tr.st}$ (trend slope), cycle AR(2) with $\\hat\\phi_1 = @{tr.phi1}$, $\\hat\\phi_2 = @{tr.phi2}$, $\\hat\\sigma^2_\\kappa = @{tr.sar}$; the irregular variance is estimated at zero', 'Estimările UC: $\\hat\\sigma^2_\\zeta = @{tr.st}$ (panta trendului), ciclul AR(2) cu $\\hat\\phi_1 = @{tr.phi1}$, $\\hat\\phi_2 = @{tr.phi2}$, $\\hat\\sigma^2_\\kappa = @{tr.sar}$; varianța componentei neregulate este estimată la zero'),
     [T('the ratio cycle variance / trend variance is @{tr.lam}, far below the 1600 that HP imposes: a more flexible trend', 'raportul dintre varianța ciclului și varianța trendului este @{tr.lam}, mult sub valoarea 1600 impusă de HP: un trend mai flexibil')]),
    T('A flexible trend absorbs part of the boom: the UC gap will be smaller than the HP gap', 'Un trend flexibil absoarbe o parte din boom: deviația UC va fi mai mică decît deviația HP')])

chart(T('The Romanian output gap: three methods', 'Deviația PIB a României: trei metode'), 'tsa_ch10_output_gap', 'TSA_ch10_trend_cycle', [
    T('Gap in \\% of trend: UC cycle (smoothed), HP gap ($\\lambda = 1600$) and the Hamilton filter ($h = 8$, four lags), the first 11 quarters lost', 'Deviația în \\% din trend: ciclul UC (netezit), deviația HP ($\\lambda = 1600$) și filtrul Hamilton ($h = 8$, patru laguri), primele 11 trimestre se pierd')],
    h='0.58\\textheight')

interp(('the output gaps', 'deviațiilor PIB'), [
    (T('All three see the overheating before 2008 (peaks: UC @{gp.peak_uc}\\% in @{gp.peakd}, HP @{gp.peak_hp}\\%, Hamilton @{gp.peak_ham}\\%) and the 2020 pandemic (UC @{gp.uc2020}\\%, HP @{gp.hp2020}\\%)', 'Toate cele trei metode surprind supraîncălzirea dinainte de 2008 (maxime: UC @{gp.peak_uc}\\% în @{gp.peakd}, HP @{gp.peak_hp}\\%, Hamilton @{gp.peak_ham}\\%) și pandemia din 2020 (UC @{gp.uc2020}\\%, HP @{gp.hp2020}\\%)'),
     [T('correlations: UC--HP @{gp.c_uc_hp}, UC--Hamilton @{gp.c_uc_ham}, HP--Hamilton @{gp.c_hp_ham}', 'corelații: UC--HP @{gp.c_uc_hp}, UC--Hamilton @{gp.c_uc_ham}, HP--Hamilton @{gp.c_hp_ham}')]),
    (T('Standard deviations: UC @{gp.sd_uc}, HP @{gp.sd_hp}, Hamilton @{gp.sd_ham}: the Hamilton gap also contains every surprise of two years', 'Abaterile standard: UC @{gp.sd_uc}, HP @{gp.sd_hp}, Hamilton @{gp.sd_ham}: deviația Hamilton conține și toate surprizele din doi ani'),
     [T('the four regression coefficients sum to @{gp.ham_bsum}: GDP is close to a random walk with drift', 'cei patru coeficienți ai regresiei au suma @{gp.ham_bsum}: PIB-ul este aproape de un mers aleator cu derivă')]),
    T('Latest gap (@{gp.lastd}): UC @{gp.last_uc}\\%, HP @{gp.last_hp}\\%, Hamilton @{gp.last_ham}\\%: the sign agrees, the size does not', 'Ultima deviație (@{gp.lastd}): UC @{gp.last_uc}\\%, HP @{gp.last_hp}\\%, Hamilton @{gp.last_ham}\\%: semnul coincide, mărimea nu')])

chart(T('Real time against hindsight', 'Timp real și retrospectivă'), 'tsa_ch10_realtime', 'TSA_ch10_trend_cycle', [
    T('Left: the HP gap computed with all data (two-sided) and the last value of the HP gap computed with the data available at each date (one-sided, from 2004); right: the UC cycle, filtered and smoothed',
      'Stînga: deviația HP calculată cu toate datele (bilaterală) și ultima valoare a deviației HP calculate cu datele disponibile la fiecare dată (unilaterală, din 2004); dreapta: ciclul UC, filtrat și netezit')],
    h='0.66\\textheight')

interp(('the revisions', 'revizuirilor'), [
    (T('In 2008Q3 the real-time HP gap was @{rt.hp_rt_2008}\\%; with hindsight it is @{rt.hp_fin_2008}\\%: the overheating was invisible in real time', 'În T3 2008 deviația HP în timp real era @{rt.hp_rt_2008}\\%; retrospectiv este @{rt.hp_fin_2008}\\%: supraîncălzirea era invizibilă în timp real'),
     [T('the UC model shows the same problem: filtered @{rt.uc_f_2008}\\%, smoothed @{rt.uc_s_2008}\\%', 'modelul UC arată aceeași problemă: filtrat @{rt.uc_f_2008}\\%, netezit @{rt.uc_s_2008}\\%')]),
    (T('Root mean square revision since 2004: HP @{rt.rev_hp} points, UC @{rt.rev_uc} points; correlation real time--final: HP @{rt.c_hp}, UC @{rt.c_uc}', 'Revizuirea medie pătratică din 2004: HP @{rt.rev_hp} puncte, UC @{rt.rev_uc} puncte; corelația timp real--final: HP @{rt.c_hp}, UC @{rt.c_uc}'),
     [T('the end-point problem of \\refHamHP; the UC model at least reports its own uncertainty', 'problema capătului de eșantion din \\refHamHP; modelul UC își raportează cel puțin propria incertitudine')]),
    T('For policy in real time, use the filtered (one-sided) estimate and its standard error, never the last point of a two-sided filter', 'Pentru politica economică în timp real folosiți estimarea filtrată (unilaterală) și eroarea ei standard, niciodată ultimul punct al unui filtru bilateral')])

chart(T('A time-varying beta: the BET and the euro area', 'Un beta variabil în timp: BET și zona euro'), 'tsa_ch10_tvp_beta', 'TSA_ch10_tvp_regression', [
    T('Weekly log returns of the BET on the Euro Stoxx 50 (EODHD), 2005--2026 ($T = @{tv.n}$ weeks)',
      'Randamentele logaritmice săptămînale ale BET pe Euro Stoxx 50 (EODHD), 2005--2026 ($T = @{tv.n}$ de săptămîni)'),
    T('Random-walk beta by maximum likelihood, Kalman smoothed, with a 90\\% band; 52-week rolling OLS; constant OLS',
      'Beta de tip mers aleator prin verosimilitate maximă, netezit Kalman, cu bandă de 90\\%; OLS pe ferestre mobile de 52 de săptămîni; OLS constant')],
    h='0.55\\textheight')

interp(('the time-varying beta', 'coeficientului beta variabil'), [
    (T('The constant OLS beta is @{tv.ols}; the smoothed beta peaks at @{tv.max} (@{tv.maxd}) and falls to @{tv.min} (@{tv.mind})', 'Beta OLS constant este @{tv.ols}; beta netezit atinge @{tv.max} (@{tv.maxd}) și coboară la @{tv.min} (@{tv.mind})'),
     [T('in the 2008 crisis the Romanian market moved almost one for one with the euro area; later it decoupled', 'în criza din 2008 piața românească s-a mișcat aproape unu la unu cu zona euro; ulterior s-a decuplat')]),
    (T('Likelihood ratio against a constant beta: @{tv.lr} (one restriction, a variance on the boundary): strong evidence of variation', 'Raportul de verosimilitate față de un beta constant: @{tv.lr} (o restricție, o varianță la limită): dovezi puternice de variație'),
     [T('weekly standard deviation of the beta shocks @{tv.sdb}; latest beta @{tv.last} with standard error @{tv.sdlast}', 'abaterea standard săptămînală a șocurilor lui beta @{tv.sdb}; ultimul beta @{tv.last}, cu eroarea standard @{tv.sdlast}')]),
    T('The rolling window jumps when a single extreme week enters or leaves it; the Kalman beta moves only as much as the likelihood allows', 'Estimarea pe fereastră mobilă se modifică brusc cînd o singură săptămînă extremă intră sau iese din fereastră; beta Kalman se mișcă doar atît cît permite verosimilitatea')])

D.frame(T('Dynamic factor models and nowcasting', 'Modele cu factori dinamici și nowcasting'), items(
    (T('\\textbf{Dynamic factor model}: $y_{it} = \\lambda_if_t + u_{it}$, \\quad $f_t = \\phi_1f_{t-1} + \\phi_2f_{t-2} + \\eta_t$', '\\textbf{Model cu factori dinamici}: $y_{it} = \\lambda_if_t + u_{it}$, \\quad $f_t = \\phi_1f_{t-1} + \\phi_2f_{t-2} + \\eta_t$'),
     [T('many series $y_{1t}, \\dots, y_{Nt}$ driven by one unobserved common factor $f_t$ (the state) with loadings $\\lambda_i$', 'multe serii $y_{1t}, \\dots, y_{Nt}$ conduse de un factor comun neobservat $f_t$ (starea), cu încărcările $\\lambda_i$'),
      T('$u_{it}$: the part of series $i$ not explained by the factor; $\\phi_1$, $\\phi_2$: the AR coefficients of the factor', '$u_{it}$: partea seriei $i$ neexplicată de factor; $\\phi_1$, $\\phi_2$: coeficienții AR ai factorului'),
      T('\\refSW: a coincident index of US activity from four monthly indicators', '\\refSW: un indice coincident al activității din SUA, din patru indicatori lunari')]),
    (T('\\textbf{Nowcasting} \\refGRS: estimate the current quarter before GDP is published', '\\textbf{Nowcasting} \\refGRS: estimăm trimestrul curent înainte ca PIB-ul să fie publicat'),
     [T('monthly data arrive at different dates: the latest months of some series are missing (the \\textbf{ragged edge})', 'datele lunare sosesc la date diferite: ultimele luni ale unor serii lipsesc (\\textbf{marginea neregulată})'),
      T('the Kalman filter treats them as missing values and updates the factor with whatever has arrived', 'filtrul Kalman le tratează ca valori lipsă și actualizează factorul cu ce a sosit deja')]),
    T('Central banks use such models to nowcast GDP; \\texttt{statsmodels} has \\texttt{DynamicFactor} and \\texttt{DynamicFactorMQ}', 'Băncile centrale folosesc astfel de modele pentru nowcasting-ul PIB; \\texttt{statsmodels} are \\texttt{DynamicFactor} și \\texttt{DynamicFactorMQ}')))

chart(T('A common factor of US activity', 'Un factor comun al activității din SUA'), 'tsa_ch10_dfm', 'TSA_ch10_dynamic_factor', [
    T('Monthly growth of industrial production, payroll employment, real income less transfers and real manufacturing and trade sales (FRED), standardised',
      'Creșterea lunară a producției industriale, a numărului de salariați, a venitului real fără transferuri și a vînzărilor reale din industrie și comerț (FRED), standardizate'),
    T('One AR(2) factor estimated on 1967--2019 and run to @{fm.lastd}; the last two months of income and sales set to missing',
      'Un factor AR(2) estimat pe 1967--2019 și rulat pînă în @{fm.lastd}; ultimele două luni ale venitului și vînzărilor sînt tratate ca lipsă')],
    h='0.52\\textheight')

interp(('the common factor', 'factorului comun'), [
    (T('Loadings: employment @{fm.PAYEMS}, industrial production @{fm.INDPRO}, sales @{fm.CMRMTSPL}, income @{fm.W875RX1}; factor AR(2) with $\\hat\\phi_1 = @{fm.phi1}$, $\\hat\\phi_2 = @{fm.phi2}$', 'Încărcări: salariați @{fm.PAYEMS}, producția industrială @{fm.INDPRO}, vînzări @{fm.CMRMTSPL}, venit @{fm.W875RX1}; factorul AR(2) cu $\\hat\\phi_1 = @{fm.phi1}$, $\\hat\\phi_2 = @{fm.phi2}$'),
     [T('correlation with the simple average of the four series: @{fm.corr}', 'corelația cu media simplă a celor patru serii: @{fm.corr}')]),
    (T('Mean of the factor: @{fm.rec} in NBER recession months, @{fm.exp} in expansions; negative in @{fm.neg}\\% of recession months', 'Media factorului: @{fm.rec} în lunile de recesiune NBER, @{fm.exp} în expansiuni; negativ în @{fm.neg}\\% din lunile de recesiune'),
     [T('April 2020: @{fm.covid} standard units, off the scale of the chart', 'aprilie 2020: @{fm.covid} unități standard, în afara scalei graficului')]),
    T('Latest value (@{fm.lastd}): @{fm.last}, computed although two of the four series are not yet available', 'Ultima valoare (@{fm.lastd}): @{fm.last}, calculată deși două dintre cele patru serii nu sînt încă disponibile')])

D.recap(('Trend, cycle and common factors', 'trend, ciclu și factori comuni'), [
    T('UC models: trend + cycle + seasonal + irregular, each with its own variance', 'Modelele UC: trend + ciclu + sezonalitate + componentă neregulată, fiecare cu varianța ei'),
    T('HP = smoother of a UC model with imposed $\\lambda$; Hamilton (2018): spurious cycles and end-point revisions', 'HP = netezitorul unui model UC cu $\\lambda$ impus; Hamilton (2018): cicluri aparente și revizuiri la capătul eșantionului'),
    T('Romanian output gap: the methods agree on the sign, not on the size; real-time gaps are revised heavily', 'Deviația PIB a României: metodele coincid ca semn, nu ca mărime; deviațiile în timp real se revizuiesc puternic'),
    T('TVP regression and dynamic factors are state space models too', 'Regresia TVP și factorii dinamici sînt și ei modele în spațiul stărilor')])

# =============================================================================
# 6. MARKOV SWITCHING
# =============================================================================
D.section('Markov switching', 'Modele Markov switching')

D.frame(T('Hamilton (1989) and the dating of recessions', 'Hamilton (1989) și datarea recesiunilor'), two(
    ph('nber', T('The NBER, Cambridge (Massachusetts), whose committee dates US recessions', 'NBER, Cambridge (Massachusetts), al cărui comitet datează recesiunile din SUA'), h='0.36\\textheight'),
    items((T('The National Bureau of Economic Research (NBER) dates US recessions by a committee, months after the fact', 'National Bureau of Economic Research (NBER) datează recesiunile din SUA printr-un comitet, la cîteva luni după eveniment'),
           [T('a recession is a regime: GDP behaves differently while it lasts', 'o recesiune este un regim: PIB-ul se comportă altfel cît timp durează')]),
          (T('\\refHamMS: let the mean of GDP growth depend on an unobserved regime $S_t \\in \\{1, 2\\}$ that follows a Markov chain', '\\refHamMS: media creșterii PIB depinde de un regim neobservat $S_t \\in \\{1, 2\\}$, care urmează un lanț Markov'),
           [T('the data then tell us the probability of a recession in each quarter, without a committee', 'datele ne spun apoi probabilitatea unei recesiuni în fiecare trimestru, fără comitet')]),
          T('The state is now \\textbf{discrete}; the Kalman filter is replaced by the Hamilton filter', 'Starea este acum \\textbf{discretă}; filtrul Kalman este înlocuit de filtrul Hamilton'))), size='footnotesize')

D.frame(T('The Markov-switching model', 'Modelul Markov switching'), items(
    (T('\\textbf{Switching mean}: $y_t = \\mu_{S_t} + \\varepsilon_t$, \\quad $\\varepsilon_t \\sim N(0, \\sigma^2_{S_t})$, \\quad $S_t \\in \\{1, \\dots, k\\}$', '\\textbf{Medie cu schimbare de regim}: $y_t = \\mu_{S_t} + \\varepsilon_t$, \\quad $\\varepsilon_t \\sim N(0, \\sigma^2_{S_t})$, \\quad $S_t \\in \\{1, \\dots, k\\}$'),
     [T('\\textbf{Markov chain}: $p_{ij} = \\Pr(S_t = j \\mid S_{t-1} = i)$, constant over time; the rows of the transition matrix sum to 1', '\\textbf{Lanțul Markov}: $p_{ij} = \\Pr(S_t = j \\mid S_{t-1} = i)$, constante în timp; rîndurile matricei de tranziție însumează 1')]),
    (T('Hamilton\'s MS-AR(4): $y_t - \\mu_{S_t} = \\sum_{j=1}^{4}\\phi_j(y_{t-j} - \\mu_{S_{t-j}}) + \\varepsilon_t$', 'MS-AR(4) al lui Hamilton: $y_t - \\mu_{S_t} = \\sum_{j=1}^{4}\\phi_j(y_{t-j} - \\mu_{S_{t-j}}) + \\varepsilon_t$'),
     [T('the AR dynamics stay the same; only the mean switches', 'dinamica AR rămîne aceeași; doar media se schimbă')]),
    (T('Variants: switching variance (calm and turbulent markets), switching AR coefficients, $k = 3$ regimes', 'Variante: varianță cu schimbare de regim (piețe calme și agitate), coeficienți AR cu schimbare de regim, $k = 3$ regimuri'),
     [T('in Python: \\texttt{MarkovRegression}, \\texttt{MarkovAutoregression} (\\texttt{statsmodels})', 'în Python: \\texttt{MarkovRegression}, \\texttt{MarkovAutoregression} (\\texttt{statsmodels})')]),
    T('A nonlinear model: the forecast depends on the probability of each regime', 'Un model neliniar: prognoza depinde de probabilitatea fiecărui regim')))

D.frame(T('Transition matrix, durations and ergodic probabilities', 'Matricea de tranziție, durate și probabilități ergodice'), items(
    (T('Two regimes: $\\mathbf{P} = \\begin{pmatrix}p_{11} & 1 - p_{11}\\\\ 1 - p_{22} & p_{22}\\end{pmatrix}$', 'Două regimuri: $\\mathbf{P} = \\begin{pmatrix}p_{11} & 1 - p_{11}\\\\ 1 - p_{22} & p_{22}\\end{pmatrix}$'),
     [T('$h$-step transitions: $\\mathbf{P}^h$', 'tranzițiile în $h$ pași: $\\mathbf{P}^h$')]),
    (T('\\textbf{Expected duration} of regime $i$: the time spent there is geometric, $\\Pr(D = d) = p_{ii}^{d-1}(1 - p_{ii})$, so $E(D) = 1/(1 - p_{ii})$', '\\textbf{Durata așteptată} a regimului $i$: timpul petrecut acolo este geometric, $\\Pr(D = d) = p_{ii}^{d-1}(1 - p_{ii})$, deci $E(D) = 1/(1 - p_{ii})$'),
     [T('$p_{11} = 0.75$: recessions last $1/0.25 = 4$ quarters on average; $p_{22} = 0.95$: expansions last 20 quarters', '$p_{11} = 0{,}75$: recesiunile durează în medie $1/0{,}25 = 4$ trimestre; $p_{22} = 0{,}95$: expansiunile durează 20 de trimestre')]),
    (T('\\textbf{Ergodic} (long-run) probabilities: $\\pi_1 = \\dfrac{1 - p_{22}}{2 - p_{11} - p_{22}}$, the share of time spent in regime 1', 'Probabilitățile \\textbf{ergodice} (pe termen lung): $\\pi_1 = \\dfrac{1 - p_{22}}{2 - p_{11} - p_{22}}$, fracțiunea de timp petrecută în regimul 1'),
     [T('example: $\\pi_1 = 0.05/0.30 = 1/6$: one quarter in six is a recession quarter', 'exemplu: $\\pi_1 = 0{,}05/0{,}30 = 1/6$: un trimestru din șase este trimestru de recesiune')])))

D.frame(T('The Hamilton filter', 'Filtrul Hamilton'), items(
    (T('Track $\\xi_{t|t} = \\Pr(S_t = j \\mid Y_t)$, the \\textbf{filtered probabilities}, for each regime $j$', 'Urmărim $\\xi_{t|t} = \\Pr(S_t = j \\mid Y_t)$, \\textbf{probabilitățile filtrate}, pentru fiecare regim $j$'),
     [T('\\textbf{prediction}: $\\Pr(S_t = j \\mid Y_{t-1}) = \\sum_i p_{ij}\\Pr(S_{t-1} = i \\mid Y_{t-1})$', '\\textbf{predicția}: $\\Pr(S_t = j \\mid Y_{t-1}) = \\sum_i p_{ij}\\Pr(S_{t-1} = i \\mid Y_{t-1})$'),
      T('\\textbf{densities}: $f_j(y_t) = \\varphi\\bigl((y_t - \\mu_j)/\\sigma_j\\bigr)/\\sigma_j$, the likelihood of $y_t$ in each regime', '\\textbf{densitățile}: $f_j(y_t) = \\varphi\\bigl((y_t - \\mu_j)/\\sigma_j\\bigr)/\\sigma_j$, verosimilitatea lui $y_t$ în fiecare regim'),
      T('$\\varphi$: the standard Normal density; $\\mu_j$, $\\sigma_j$: the mean and standard deviation of regime $j$', '$\\varphi$: densitatea distribuției Normale standard; $\\mu_j$, $\\sigma_j$: media și abaterea standard a regimului $j$'),
      T('\\textbf{update} (Bayes): $\\Pr(S_t = j \\mid Y_t) = \\dfrac{\\Pr(S_t = j \\mid Y_{t-1})\\,f_j(y_t)}{\\sum_i\\Pr(S_t = i \\mid Y_{t-1})\\,f_i(y_t)}$', '\\textbf{actualizarea} (Bayes): $\\Pr(S_t = j \\mid Y_t) = \\dfrac{\\Pr(S_t = j \\mid Y_{t-1})\\,f_j(y_t)}{\\sum_i\\Pr(S_t = i \\mid Y_{t-1})\\,f_i(y_t)}$')]),
    (T('The denominator is $p(y_t \\mid Y_{t-1})$: the log-likelihood is $\\sum_t\\log p(y_t \\mid Y_{t-1})$, as for the Kalman filter', 'Numitorul este $p(y_t \\mid Y_{t-1})$: log-verosimilitatea este $\\sum_t\\log p(y_t \\mid Y_{t-1})$, ca la filtrul Kalman'),
     [T('the same predict--update cycle, with probabilities instead of means and variances', 'același ciclu predicție--actualizare, cu probabilități în locul mediilor și varianțelor')]),
    T('Start: the ergodic probabilities $\\pi_j$', 'Pornirea: probabilitățile ergodice $\\pi_j$')))

D.frame(T('Smoothed probabilities and estimation', 'Probabilități netezite și estimare'), items(
    (T('\\textbf{Smoothed probabilities} $\\Pr(S_t = j \\mid Y_n)$: a backward pass \\refKim, the analogue of the RTS smoother', '\\textbf{Probabilitățile netezite} $\\Pr(S_t = j \\mid Y_n)$: o trecere înapoi \\refKim, analogul netezitorului RTS'),
     [T('used to date regimes in history; the filtered probabilities are what a forecaster knew at time $t$', 'folosite pentru datarea istorică a regimurilor; probabilitățile filtrate sînt ceea ce știa un prognozator la momentul $t$')]),
    (T('Estimation: maximum likelihood by numerical optimisation, or the EM algorithm \\refHamEM', 'Estimarea: verosimilitate maximă prin optimizare numerică sau algoritmul EM \\refHamEM'),
     [T('EM: expectation (smoothed probabilities given $\\theta$) and maximisation (weighted means, variances, transition counts), repeated', 'EM: așteptarea (probabilitățile netezite pentru $\\theta$ dat) și maximizarea (medii, varianțe și numărători de tranziții ponderate), repetate')]),
    (T('The likelihood has \\textbf{several local maxima}: use many random starting values and compare the solutions', 'Verosimilitatea are \\textbf{mai multe maxime locale}: folosiți multe valori de pornire aleatoare și comparați soluțiile'),
     [T('label switching: regime 1 and regime 2 can swap; name them by their means or variances', 'schimbarea etichetelor: regimurile 1 și 2 își pot schimba locul; denumiți-le după medii sau varianțe')])))

D.frame(T('Hamilton\'s specification on today\'s data', 'Specificația lui Hamilton pe datele de azi'), table(
    '>{\\raggedright\\arraybackslash}p{5.6cm}cc', T('\\textbf{MS-AR(4), US real GDP growth, 1951Q2--1984Q4}', '\\textbf{MS-AR(4), creșterea PIB real al SUA, T2 1951--T4 1984}') + ' & ' + T('\\textbf{Maximum A}', '\\textbf{Maximul A}') + ' & ' + T('\\textbf{Maximum B}', '\\textbf{Maximul B}'),
    [T('mean, regime 1 / regime 2 (\\% per quarter)', 'media, regimul 1 / regimul 2 (\\% pe trimestru)') + ' & @{ha.m1} / @{ha.m2} & @{hb.m1} / @{hb.m2}',
     '$p_{11}$, $p_{22}$ & @{ha.p11}, @{ha.p22} & @{hb.p11}, @{hb.p22}',
     T('expected duration, regime 1 / 2 (quarters)', 'durata așteptată, regimul 1 / 2 (trimestre)') + ' & @{ha.d1} / @{ha.d2} & @{hb.d1} / @{hb.d2}',
     T('log-likelihood', 'log-verosimilitatea') + ' & @{ha.ll} & @{hb.ll}',
     T('NBER recession quarters found (smoothed prob. $> 0.5$)', 'trimestre de recesiune NBER găsite (prob. netezită $> 0{,}5$)') + ' & @{hahit}\\% & @{hbhit}\\%'],
    size='footnotesize') + items(
    T('Maximum A is Hamilton\'s answer: a recession regime of about four quarters that matches the NBER dates', 'Maximul A este răspunsul lui Hamilton: un regim de recesiune de aproximativ patru trimestre, care se potrivește cu datările NBER'),
    T('Maximum B has a slightly higher likelihood but a different meaning: isolated sharp quarters', 'Maximul B are o verosimilitate puțin mai mare, dar alt înțeles: trimestre izolate cu scăderi bruște'),
    T('A likelihood difference of @{hb.ll} against @{ha.ll} cannot decide; economic meaning and robustness checks must', 'O diferență de verosimilitate de @{hb.ll} față de @{ha.ll} nu poate decide; decid înțelesul economic și verificările de robustețe')))

chart(T('US recessions from GDP growth', 'Recesiunile din SUA din creșterea PIB'), 'tsa_ch10_ms_us', 'TSA_ch10_markov_gdp', [
    T('Two regimes with switching mean, constant variance, estimated on @{ms.nrec} NBER recession quarters out of 1947Q2--2019Q4 (FRED GDPC1, USREC); probabilities after 2019 computed with the same parameters',
      'Două regimuri cu medie variabilă și varianță constantă, estimate pe T2 1947--T4 2019, eșantion cu @{ms.nrec} de trimestre de recesiune NBER (FRED GDPC1, USREC); probabilitățile de după 2019 sînt calculate cu aceiași parametri')],
    h='0.66\\textheight')

interp(('the recession probabilities', 'probabilităților de recesiune'), [
    (T('Regime means: @{ex.m1}\\% (recession) and @{ex.m2}\\% (expansion) per quarter, $\\hat\\sigma = @{ex.s}$; $\\hat p_{11} = @{ex.p11}$, $\\hat p_{22} = @{ex.p22}$', 'Mediile regimurilor: @{ex.m1}\\% (recesiune) și @{ex.m2}\\% (expansiune) pe trimestru, $\\hat\\sigma = @{ex.s}$; $\\hat p_{11} = @{ex.p11}$, $\\hat p_{22} = @{ex.p22}$'),
     [T('expected durations: @{ex.d1} quarters of recession, @{ex.d2} quarters of expansion; ergodic share of recession @{ex.erg}\\%', 'durate așteptate: @{ex.d1} trimestre de recesiune, @{ex.d2} trimestre de expansiune; ponderea ergodică a recesiunii @{ex.erg}\\%')]),
    (T('Agreement with the NBER: @{excon}\\% of quarters; @{exhit}\\% of NBER recession quarters detected, @{exfalse}\\% false alarms', 'Concordanța cu NBER: @{excon}\\% din trimestre; @{exhit}\\% din trimestrele de recesiune NBER detectate, @{exfalse}\\% alarme false'),
     [T('the model misses the mild recessions (2001) and catches the deep ones', 'modelul nu detectează recesiunile ușoare (2001), dar le detectează pe cele profunde')]),
    T('2020Q2 ($@{ms.g20}\\%$): probability @{ms.p20}, then back to expansion; after 2021 the probability never exceeds @{ms.post}', 'T2 2020 ($@{ms.g20}\\%$): probabilitatea @{ms.p20}, apoi revenire la expansiune; după 2021 probabilitatea nu depășește @{ms.post}')])

chart(T('Filtered against smoothed probabilities: 2008', 'Probabilități filtrate și netezite: 2008'), 'tsa_ch10_ms_filtered', 'TSA_ch10_markov_gdp', [
    T('Predicted, filtered and smoothed probability of the recession regime, 2005--2012, same model as on the previous chart', 'Probabilitatea prezisă, filtrată și netezită a regimului de recesiune, 2005--2012, același model ca în graficul anterior')],
    h='0.56\\textheight')

interp(('the real-time probabilities', 'probabilităților în timp real'), [
    (T('The NBER recession started in 2008Q1; the smoothed probability crosses 0.5 in @{mf.fs}, the filtered one only in @{mf.ff}', 'Recesiunea NBER a început în T1 2008; probabilitatea netezită trece de 0,5 în @{mf.fs}, cea filtrată abia în @{mf.ff}'),
     [T('2008Q1: filtered @{mf.f2008q1}, smoothed @{mf.s2008q1}; 2008Q3 (growth @{mf.g2008q3}\\%): filtered @{mf.f2008q3}, smoothed @{mf.s2008q3}', 'T1 2008: filtrată @{mf.f2008q1}, netezită @{mf.s2008q1}; T3 2008 (creștere @{mf.g2008q3}\\%): filtrată @{mf.f2008q3}, netezită @{mf.s2008q3}')]),
    (T('Concordance with the NBER using only filtered probabilities: @{exconf}\\% of quarters, @{exhitf}\\% of recession quarters', 'Concordanța cu NBER folosind doar probabilitățile filtrate: @{exconf}\\% din trimestre, @{exhitf}\\% din trimestrele de recesiune'),
     [T('real-time dating is harder: \\refCP\\ compare the methods on real-time data', 'datarea în timp real este mai grea: \\refCP\\ compară metodele pe date în timp real')]),
    T('Never judge a model\'s real-time skill from its smoothed probabilities: they use the future', 'Nu judecați niciodată capacitatea unui model în timp real după probabilitățile netezite: ele folosesc viitorul')])

D.frame(T('Checks: which regimes did the model find?', 'Verificări: regimurile găsite de model'), items(
    (T('\\textbf{Switching variance} on 1947--2019: the regimes become high and low volatility, not recession and expansion', '\\textbf{Varianță cu schimbare de regim} pe 1947--2019: regimurile devin volatilitate mare și volatilitate mică, nu recesiune și expansiune'),
     [T('standard deviations @{pi.s2} and @{pi.s1}; durations @{pi.d2} and @{pi.d1} quarters; the switch to the calm regime in @{pi.sw}', 'abateri standard @{pi.s2} și @{pi.s1}; durate de @{pi.d2} și @{pi.d1} de trimestre; trecerea la regimul calm în @{pi.sw}'),
      T('this is the ``Great Moderation\'\' of the mid-1980s; agreement with the NBER falls to @{pi.con}\\% (@{pi.false}\\% false alarms)', 'aceasta este „Marea Moderație” de la mijlocul anilor 1980; concordanța cu NBER scade la @{pi.con}\\% (@{pi.false}\\% alarme false)')]),
    (T('\\textbf{Including 2020} in the estimation: one regime is the single quarter 2020Q2 (mean @{pi.m1}\\%), the other lasts @{pi.d2all} quarters', '\\textbf{Includerea anului 2020} în estimare: un regim este doar trimestrul T2 2020 (media @{pi.m1}\\%), celălalt durează @{pi.d2all} de trimestre'),
     [T('one extreme observation can capture a whole regime', 'o singură observație extremă poate forma singură un regim')]),
    T('Checklist: plot the probabilities against known events; compare several starting values and samples; prefer the specification with a clear economic meaning', 'Lista de verificări: reprezentați probabilitățile alături de evenimente cunoscute; comparați mai multe valori de pornire și eșantioane; preferați specificația cu un înțeles economic clar')))

chart(T('Romanian GDP growth regimes', 'Regimuri ale creșterii PIB în România'), 'tsa_ch10_ms_ro', 'TSA_ch10_markov_gdp', [
    T('Quarter-on-quarter growth of Romanian real GDP (Eurostat), two regimes with switching mean and variance; smoothed probability of the volatile regime', 'Creșterea trimestrială a PIB-ului real al României (Eurostat), două regimuri cu medie și varianță variabile; probabilitatea netezită a regimului volatil')],
    h='0.66\\textheight')

interp(('the Romanian regimes', 'regimurilor din România'), [
    (T('Stable regime: mean @{mr.m2}\\% per quarter, standard deviation @{mr.s2}; volatile regime: mean @{mr.m1}\\%, standard deviation @{mr.s1}', 'Regimul stabil: media @{mr.m2}\\% pe trimestru, abaterea standard @{mr.s2}; regimul volatil: media @{mr.m1}\\%, abaterea standard @{mr.s1}'),
     [T('expected durations: @{mr.d2} quarters stable, @{mr.d1} quarters volatile; @{mr.share}\\% of the quarters are volatile', 'durate așteptate: @{mr.d2} trimestre în regimul stabil, @{mr.d1} trimestre în cel volatil; @{mr.share}\\% din trimestre sînt volatile')]),
    (T('Volatile spells: @{mr.sp0} (transition and the 1997--1999 recession), @{mr.sp1}, @{mr.sp2} (the crisis and the austerity of 2010), @{mr.sp3} (the pandemic)', 'Episoade volatile: @{mr.sp0} (tranziția și recesiunea din 1997--1999), @{mr.sp1}, @{mr.sp2} (criza și austeritatea din 2010), @{mr.sp3} (pandemia)'),
     [T('latest probability of the volatile regime: @{mr.plast}', 'ultima probabilitate a regimului volatil: @{mr.plast}')]),
    T('With 125 quarters a mean-only model is unstable; regimes of volatility and growth come together in an emerging economy', 'Cu 125 de trimestre un model doar cu medie variabilă este instabil; într-o economie emergentă regimurile de volatilitate și de creștere vin împreună')])

chart(T('Calm and turbulent stock markets', 'Piețe bursiere calme și agitate'), 'tsa_ch10_vol_regimes', 'TSA_ch10_volatility_regimes', [
    T('Weekly log returns of the S\\&P 500, 2000--2026 (EODHD); two regimes with switching mean and variance; bottom: the volatility implied by the regime probabilities against a GARCH(1,1) (Chapter 5), both annualised',
      'Randamentele logaritmice săptămînale ale S\\&P 500, 2000--2026 (EODHD); două regimuri cu medie și varianță variabile; jos: volatilitatea implicată de probabilitățile regimurilor comparată cu un GARCH(1,1) (Capitolul 5), ambele anualizate')],
    h='0.66\\textheight')

interp(('the volatility regimes', 'regimurilor de volatilitate'), [
    (T('Calm regime: volatility @{vo.s2}\\% per year, mean @{vo.m2}\\% per week; turbulent: @{vo.s1}\\% per year, mean @{vo.m1}\\% per week', 'Regimul calm: volatilitate de @{vo.s2}\\% pe an, media @{vo.m2}\\% pe săptămînă; agitat: @{vo.s1}\\% pe an, media @{vo.m1}\\% pe săptămînă'),
     [T('durations: @{vo.d2} weeks calm, @{vo.d1} weeks turbulent; turbulent in @{vo.share}\\% of the weeks (ergodic @{vo.erg}\\%)', 'durate: @{vo.d2} de săptămîni calme, @{vo.d1} săptămîni agitate; agitat în @{vo.share}\\% din săptămîni (ergodic @{vo.erg}\\%)')]),
    (T('Most turbulent years: @{vo.y0} (@{vo.ys0}\\% of the weeks), @{vo.y1} (@{vo.ys1}\\%), @{vo.y2} (@{vo.ys2}\\%)', 'Cei mai agitați ani: @{vo.y0} (@{vo.ys0}\\% din săptămîni), @{vo.y1} (@{vo.ys1}\\%), @{vo.y2} (@{vo.ys2}\\%)'),
     [T('correlation with the GARCH(1,1) volatility: @{vo.corr}; GARCH: $\\hat\\alpha = @{vo.ga}$, $\\hat\\beta = @{vo.gb}$', 'corelația cu volatilitatea GARCH(1,1): @{vo.corr}; GARCH: $\\hat\\alpha = @{vo.ga}$, $\\hat\\beta = @{vo.gb}$')]),
    T('Two views of the same clustering: GARCH moves volatility continuously, the regime model jumps between two levels', 'Două perspective asupra aceluiași volatility clustering: GARCH modifică volatilitatea continuu, modelul cu regimuri trece brusc de la un nivel la altul')])

D.frame(T('Regimes, GARCH persistence and long memory', 'Regimuri, persistența GARCH și memoria lungă'), items(
    (T('\\refLL: shifts in the level of variance make a GARCH look almost integrated ($\\alpha + \\beta \\approx 1$)', '\\refLL: schimbările nivelului varianței fac un GARCH să pară aproape integrat ($\\alpha + \\beta \\approx 1$)'),
     [T('the near-IGARCH estimates of Chapter 5 may reflect regimes, not one very persistent process', 'estimările aproape IGARCH din Capitolul 5 pot reflecta regimuri, nu un singur proces foarte persistent'),
      T('\\refHSus: ARCH within regimes (SWARCH) lowers the persistence', '\\refHSus: ARCH în interiorul regimurilor (SWARCH) reduce persistența')]),
    (T('\\refDI: rare regime switches produce a slowly decaying ACF and $\\hat d > 0$ (Chapter 8, spurious long memory)', '\\refDI: schimbările rare de regim produc o ACF care scade lent și $\\hat d > 0$ (Capitolul 8, memoria lungă aparentă)'),
     [T('three models, one fact: volatility shocks persist; the next chart simulates the fitted regime model to see how far regimes alone go', 'trei modele, un singur fapt: șocurile volatilității persistă; graficul următor simulează modelul de regimuri estimat, pentru a vedea cît explică doar regimurile')])))

chart(T('Regimes imitate memory', 'Regimurile imită memoria'), 'tsa_ch10_regimes_memory', 'TSA_ch10_volatility_regimes', [
    T('ACF of weekly $|r_t|$ of the S\\&P 500; mean ACF of @{me.reps} series simulated from the fitted two-regime model (constant variance within each regime, no GARCH); the same returns in random order',
      'ACF a lui $|r_t|$ săptămînal pentru S\\&P 500; ACF medie a @{me.reps} de serii simulate din modelul cu două regimuri estimat (varianță constantă în fiecare regim, fără GARCH); aceleași randamente în ordine aleatoare')],
    h='0.56\\textheight')

interp(('the simulated regimes', 'regimurilor simulate'), [
    (T('Data: ACF of $|r_t|$ @{me.acf_data_1} at lag 1, @{me.acf_data_26} at 26 weeks; regime model: @{me.acf_sim_1} and @{me.acf_sim_26}; random order: none', 'Datele: ACF a lui $|r_t|$ @{me.acf_data_1} la lagul 1, @{me.acf_data_26} la 26 de săptămîni; modelul cu regimuri: @{me.acf_sim_1} și @{me.acf_sim_26}; ordinea aleatoare: nimic'),
     [T('the regimes explain most of the short-run clustering, less of the long tail', 'regimurile explică cea mai mare parte a volatility clustering-ului pe termen scurt, mai puțin din coada lungă a ACF')]),
    (T('Local Whittle $\\hat d$ of $|r_t|$: @{me.d_data} in the data, @{me.d_sim} (SD @{me.d_sim_sd}) in the simulated regime series, @{me.d_iid} in random order', 'Estimatorul Whittle local $\\hat d$ pentru $|r_t|$: @{me.d_data} în date, @{me.d_sim} (SD @{me.d_sim_sd}) în seriile simulate cu regimuri, @{me.d_iid} în ordine aleatoare'),
     [T('a GARCH(1,1) fitted to the simulated series gives $\\alpha + \\beta = @{me.garch_ab_sim}$ on average (@{me.repsg} series), although there is no GARCH in them', 'un GARCH(1,1) estimat pe seriile simulate dă în medie $\\alpha + \\beta = @{me.garch_ab_sim}$ (@{me.repsg} serii), deși ele nu conțin GARCH')]),
    T('Persistence and long memory in volatility are consistent with regimes: compare models by likelihood and forecasts, not by one statistic', 'Persistența și memoria lungă a volatilității sînt compatibile cu regimurile: comparați modelele după verosimilitate și prognoze, nu după o singură statistică')])

D.recap(('Markov switching', 'modelele Markov switching'), [
    T('$y_t = \\mu_{S_t} + \\varepsilon_t$, $S_t$ a Markov chain; durations $1/(1 - p_{ii})$; ergodic $\\pi_1 = (1-p_{22})/(2-p_{11}-p_{22})$', '$y_t = \\mu_{S_t} + \\varepsilon_t$, $S_t$ un lanț Markov; durate $1/(1 - p_{ii})$; ergodic $\\pi_1 = (1-p_{22})/(2-p_{11}-p_{22})$'),
    T('Hamilton filter: predict with $\\mathbf{P}$, update with Bayes; Kim smoother for history', 'Filtrul Hamilton: predicție cu $\\mathbf{P}$, actualizare cu Bayes; netezitorul Kim pentru istorie'),
    T('US GDP: the model recovers the deep NBER recessions; filtered probabilities react later', 'PIB-ul SUA: modelul regăsește recesiunile NBER profunde; probabilitățile filtrate reacționează mai tîrziu'),
    T('Several maxima, outliers and variance breaks can change what a ``regime\'\' means', 'Mai multe maxime, valorile extreme și rupturile de varianță pot schimba înțelesul unui „regim”'),
    T('Regimes reproduce volatility clustering, GARCH persistence and apparent long memory', 'Regimurile reproduc volatility clustering, persistența GARCH și memoria lungă aparentă')])

# =============================================================================
# 7. AI
# =============================================================================
D.section('Possible contribution of AI', 'Contribuția posibilă a AI')

D.frame(T('Possible contribution of AI', 'Contribuția posibilă a AI'), items(
    T('\\textbf{Code}: a first draft of a Kalman filter, of the matrices of a new model, of a Markov-switching fit with many starting values', '\\textbf{Cod}: o primă versiune a unui filtru Kalman, a matricelor unui model nou, a unei estimări Markov switching cu multe valori de pornire'),
    T('\\textbf{Explanation}: a second derivation of the gain, of the steady state, of the Hamilton filter', '\\textbf{Explicații}: o a doua derivare a cîștigului, a stării de echilibru, a filtrului Hamilton'),
    T('\\textbf{Exploration}: output gaps for many EU countries; regimes in many markets; a nowcast with many indicators', '\\textbf{Explorare}: deviații PIB pentru multe țări din UE; regimuri pe multe piețe; un nowcast cu mulți indicatori'),
    (T('Example prompt', 'Exemplu de prompt'),
     [T('\\aiprompt{Write Python code that downloads Romanian quarterly real GDP from Eurostat (namq\\_10\\_gdp, Q.CLV10\\_MEUR.SCA.B1GQ.RO), fits UnobservedComponents with a smooth trend and an AR(2) cycle, compares the smoothed and the filtered cycle with the HP gap and with the Hamilton (2018) regression filter, and reports the revisions of the last eight quarters.}',
        '\\aiprompt{Write Python code that downloads Romanian quarterly real GDP from Eurostat (namq\\_10\\_gdp, Q.CLV10\\_MEUR.SCA.B1GQ.RO), fits UnobservedComponents with a smooth trend and an AR(2) cycle, compares the smoothed and the filtered cycle with the HP gap and with the Hamilton (2018) regression filter, and reports the revisions of the last eight quarters.}')])))

D.frame(T('Checks you must run', 'Verificări necesare'), items(
    T('Simulate from known parameters and check that the code recovers them (the Nile values of \\refDK\\ are a good test)', 'Simulați din parametri cunoscuți și verificați că acest cod îi regăsește (valorile pentru Nil din \\refDK\\ sînt un test bun)'),
    T('Filtered or smoothed? Real-time claims need filtered estimates; an AI answer that uses \\texttt{smoothed} for a ``forecast\'\' is wrong', 'Filtrat sau netezit? Afirmațiile despre timp real cer estimări filtrate; un răspuns AI care folosește \\texttt{smoothed} pentru o „prognoză” este greșit'),
    T('Durations: $1/(1 - p_{ii})$, not $p_{ii}/(1 - p_{ii})$; check the convention of the transition matrix (rows or columns) in the library', 'Duratele: $1/(1 - p_{ii})$, nu $p_{ii}/(1 - p_{ii})$; verificați convenția matricei de tranziție (rînduri sau coloane) în bibliotecă'),
    T('Markov switching: several starting values, label switching, outliers that capture a regime', 'Markov switching: mai multe valori de pornire, schimbarea etichetelor, valori extreme care formează singure un regim'),
    T('The HP filter at the end of the sample: do not read the last gap as a real-time estimate', 'Filtrul HP la capătul eșantionului: nu interpretați ultima deviație ca o estimare în timp real'),
    T('Every cited reference: it must exist; check the DOI', 'Fiecare referință citată: trebuie să existe; verificați DOI-ul')))

# =============================================================================
# REZUMAT
# =============================================================================
D.section('Summary', 'Rezumat')

D.frame(T('Key takeaways', 'Idei de reținut'), items(
    T('State space form: a hidden state $\\alpha_t$ follows a VAR(1) and is measured with noise; ARMA, ETS, UC, TVP and factor models fit in it', 'Forma în spațiul stărilor: o stare ascunsă $\\alpha_t$ urmează un VAR(1) și este măsurată cu zgomot; ARMA, ETS, UC, TVP și modelele factoriale se scriu astfel'),
    T('The Kalman filter predicts and updates; the gain weighs data against model; the local level filter in steady state is SES', 'Filtrul Kalman prezice și actualizează; cîștigul cîntărește datele în raport cu modelul; filtrul local level în starea de echilibru este SES'),
    T('Smoothing for history, filtering for real time; the likelihood comes from the prediction errors; missing data are easy', 'Netezirea pentru istorie, filtrarea pentru timp real; verosimilitatea provine din erorile de predicție; valorile lipsă sînt ușor de tratat'),
    T('Output gaps depend on the method and are revised heavily at the end of the sample', 'Deviațiile PIB depind de metodă și se revizuiesc puternic la capătul eșantionului'),
    T('Markov switching: regime probabilities, durations $1/(1 - p_{ii})$; check what the regimes mean', 'Markov switching: probabilitățile regimurilor, durate $1/(1 - p_{ii})$; verificați ce înseamnă regimurile')))

D.frame(T('Key formulas', 'Formule de reținut'), '{\\renewcommand{\\arraystretch}{1.35}' + table(
    'll', T('\\textbf{Quantity}', '\\textbf{Mărimea}') + ' & ' + T('\\textbf{Formula}', '\\textbf{Formula}'),
    [T('State space form', 'Forma în spațiul stărilor') + ' & $y_t = Z_t\\alpha_t + \\varepsilon_t$, \\quad $\\alpha_{t+1} = T\\alpha_t + R\\eta_t$',
     T('Update', 'Actualizarea') + ' & $v_t = y_t - Z_ta_t$, \\quad $F_t = Z_tP_tZ_t\' + H$, \\quad $K_t = P_tZ_t\'F_t^{-1}$, \\quad $a_{t|t} = a_t + K_tv_t$',
     T('Prediction', 'Predicția') + ' & $a_{t+1} = Ta_{t|t}$, \\quad $P_{t+1} = TP_{t|t}T\' + RQR\'$',
     T('Local level steady state', 'Starea de echilibru local level') + ' & $\\bar P = \\sigma^2_\\varepsilon(q + \\sqrt{q^2 + 4q})/2$, \\quad $\\alpha_{SES} = \\bar K = \\bar P/(\\bar P + \\sigma^2_\\varepsilon)$',
     T('Log-likelihood', 'Log-verosimilitatea') + ' & $-\\frac12\\sum_t(\\log 2\\pi + \\log F_t + v_t^2/F_t)$',
     T('HP filter', 'Filtrul HP') + ' & $\\min\\sum(y_t - \\tau_t)^2 + \\lambda\\sum(\\Delta^2\\tau_t)^2$, \\quad $\\lambda = \\sigma^2_\\varepsilon/\\sigma^2_\\zeta$',
     T('Markov chain', 'Lanțul Markov') + ' & $E(D_i) = 1/(1 - p_{ii})$, \\quad $\\pi_1 = (1 - p_{22})/(2 - p_{11} - p_{22})$',
     T('Hamilton filter', 'Filtrul Hamilton') + ' & $\\Pr(S_t = j \\mid Y_t) \\propto f_j(y_t)\\sum_i p_{ij}\\Pr(S_{t-1} = i \\mid Y_{t-1})$'],
    size='footnotesize') + '}')

D.frame(T('Self-assessment', 'Autoevaluare'), items(
    (T('\\textbf{Question}: local level with $P_t = 3$, $\\sigma^2_\\varepsilon = 1$. What is the Kalman gain?', '\\textbf{Întrebare}: local level cu $P_t = 3$, $\\sigma^2_\\varepsilon = 1$. Cît este cîștigul Kalman?'),
     [T('\\textbf{Answer}: $K_t = 3/(3 + 1) = 0.75$: the new observation gets weight 0.75', '\\textbf{Răspuns}: $K_t = 3/(3 + 1) = 0{,}75$: noua observație primește ponderea 0,75')]),
    (T('\\textbf{Question}: an SES model has $\\alpha = 0.5$. Which local level model is behind it?', '\\textbf{Întrebare}: un model SES are $\\alpha = 0{,}5$. Ce model local level se află în spatele lui?'),
     [T('\\textbf{Answer}: $q = \\alpha^2/(1 - \\alpha) = 0.25/0.5 = 0.5$', '\\textbf{Răspuns}: $q = \\alpha^2/(1 - \\alpha) = 0{,}25/0{,}5 = 0{,}5$')]),
    (T('\\textbf{Question}: $p_{11} = 0.9$ in a two-regime model. How long does regime 1 last on average?', '\\textbf{Întrebare}: $p_{11} = 0{,}9$ într-un model cu două regimuri. Cît durează în medie regimul 1?'),
     [T('\\textbf{Answer}: $1/(1 - 0.9) = 10$ periods', '\\textbf{Răspuns}: $1/(1 - 0{,}9) = 10$ perioade')]),
    T('Next: Chapters 11--14 for self-study; Chapter 15, review', 'Urmează: Capitolele 11--14, pentru studiu individual; Capitolul 15, recapitularea')))

D.references(bib())

if __name__ == '__main__':
    finalize(D.write(V))
