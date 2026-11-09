"""Build the interactive pages of Chapter 6 (VAR models and Granger causality): interactive/ch6.html (EN) and interactive/ch6_ro.html (RO).

One block per chart slide of the lecture body, in the order of the slides: the slide on the left, its interactive version on
the right. The result tables of the lecture (VAR coefficients, Granger tests, spillover table) and its worked examples (the
VAR(1), the Granger F test, the orthogonalised responses, the FEVD) get blocks of their own (EXTRA), placed by slide; the
chart blocks also carry live tables and step-by-step calculations. Logic in interactive/ch6.js, inputs in
interactive/ch6_data.js (Quantlets/Ch_06/export_interactive.py), slide images in interactive/slides/.
"""
import os
import re

from interactive_common import slider, select, play, charts

N = 6

QL = 'https://github.com/danpele/Time-Series-Analysis/tree/main/Quantlets/Ch_06'
TEX = {'ro': 'RO/Cursuri/capitol6_modele_var_cauzalitate_granger', 'en': 'EN/Courses/chapter6_var_models_granger_causality'}

# (anchor, chart file, section key); order of the slides
BLOCKS = [('ro-macro', 'tsa_ch6_ro_macro', 'ms'), ('ccf', 'tsa_ch6_ccf', 'ms'),
          ('var-sim', 'tsa_ch6_var_sim', 'var'),
          ('roots', 'tsa_ch6_roots', 'stab'),
          ('ic', 'tsa_ch6_ic', 'est'),
          ('resid', 'tsa_ch6_resid', 'diag'),
          ('granger-sim', 'tsa_ch6_granger_sim', 'gr'),
          ('irf-ro', 'tsa_ch6_irf_ro', 'irf'), ('irf-order', 'tsa_ch6_irf_order', 'irf'), ('market-irf', 'tsa_ch6_market_irf', 'irf'),
          ('fevd-ro', 'tsa_ch6_fevd_ro', 'fevd'), ('spillover', 'tsa_ch6_spillover', 'fevd'),
          ('forecast-ro', 'tsa_ch6_forecast_ro', 'fc'), ('oos', 'tsa_ch6_oos', 'fc'),
          ('us-data', 'tsa_ch6_us_data', 'svar'), ('sw-irf', 'tsa_ch6_sw_irf', 'svar')]

# blocks of the result tables and worked examples: (anchor, frame title RO, frame title EN, section key)
EXTRA = [('wx-var1', 'Exemplu rezolvat: un VAR(1)', 'Worked example: a VAR(1)', 'var'),
         ('var-table', 'VAR(2) românesc estimat', 'The estimated Romanian VAR(2)', 'est'),
         ('granger-wx', 'Exemplu rezolvat: este creșterea PIB-ului o cauză Granger pentru ROBOR?', 'Worked example: does GDP growth Granger-cause ROBOR?', 'gr'),
         ('granger-ro', 'Cauzalitatea Granger în VAR(2) românesc', 'Granger causality in the Romanian VAR(2)', 'gr'),
         ('granger-mkt', 'Piețe: cauzalitatea Granger de la New York către Frankfurt și București', 'Markets: Granger causality from New York to Frankfurt and Bucharest', 'gr'),
         ('wx-orth', 'Exemplu rezolvat: răspunsuri ortogonalizate', 'Worked example: orthogonalised responses', 'irf'),
         ('wx-fevd', 'Descompunerea varianței erorii de prognoză', 'Forecast error variance decomposition', 'fevd'),
         ('spill-table', 'Tabelul de spillover: S&P 500, DAX, BET', 'Spillover table: S&P 500, DAX, BET', 'fevd')]

PFX = {'ro-macro': 'rm', 'ccf': 'cc', 'wx-var1': 'w1', 'var-sim': 'vs', 'roots': 'ro', 'ic': 'ic', 'var-table': 'vt', 'resid': 're', 'granger-wx': 'gw',
       'granger-ro': 'gr', 'granger-mkt': 'gm', 'granger-sim': 'gs', 'wx-orth': 'wo', 'irf-ro': 'ir', 'irf-order': 'io', 'market-irf': 'mi', 'wx-fevd': 'wf',
       'fevd-ro': 'fv', 'spill-table': 'st', 'spillover': 'sp', 'forecast-ro': 'fc', 'oos': 'oo', 'us-data': 'us', 'sw-irf': 'sw'}

SECTIONS = {
 'ro': dict(ms='De la o serie la mai multe', var='Modelul VAR(p)', stab='Staționaritate și forma companion', est='Estimare și alegerea numărului de laguri',
            diag='Diagnosticarea reziduurilor', gr='Cauzalitatea Granger', irf='Funcții de răspuns la impuls', fevd='Descompunerea varianței erorii de prognoză',
            fc='Prognoza cu un VAR', svar='VAR-uri structurale și studiul de caz Stock–Watson'),
 'en': dict(ms='From one series to several', var='The VAR(p) model', stab='Stationarity and the companion form', est='Estimation and lag selection',
            diag='Residual diagnostics', gr='Granger causality', irf='Impulse responses', fevd='Forecast error variance decomposition',
            fc='Forecasting with a VAR', svar='Structural VARs and the Stock–Watson case study'),
}

L = {
 'ro': dict(file='ch6_ro.html', html_lang='ro', site='../index_ro.html#chapters', course='Serii de timp',
   h1='Capitolul 6 · Modele VAR și cauzalitate Granger', back='Site-ul cursului', pdf_btn='PDF-ul cursului', ql='Quantlets',
   slide='Slide-ul', ask='Întrebare pentru sală', answer='Răspuns',
   note='În stînga este slide-ul din curs, în dreapta aceeași figură sau același tabel, calculat în browser din aceleași date. Butonul ▶ modifică parametrul pas cu pas; '
        'casetele „Exemplu rezolvat” refac calculul cu cifrele capitolului.',
   src='Date trimestriale pentru România, T1 2005 – T2 2026: creșterea PIB-ului real (Eurostat, seria din curs), inflația anuală IAPC și ROBOR 3M (medii trimestriale). '
       'Randamente logaritmice zilnice în %, 6.433 de zile comune, 6 ianuarie 2000 – 18 septembrie 2026: S&P 500, DAX și BET (EODHD). Statele Unite: FRED (GDPCTPI, UNRATE, '
       'FEDFUNDS), T1 1960 – T2 2026. VAR-urile, testele, răspunsurile la impuls, FEVD și indicele de spillover se reestimează în browser prin OLS; stările inițiale sînt '
       'cele din curs. Benzile bootstrap ale răspunsurilor românești folosesc extragerile seminței capitolului (500 de replicări); simularea VAR(1) folosește extragerile '
       'Normale ale aceleiași semințe. Variantele care nu apar în curs (alt ordin p, altă ordine Cholesky, alt eșantion, alte ferestre, alte efecte în simulare) sînt rulări '
       'suplimentare ale codului capitolului, cu aceleași date și aceeași sămînță; indicele de spillover cu alte ferestre sau alt orizont se calculează din 5 în 5 zile.'),
 'en': dict(file='ch6.html', html_lang='en', site='../index.html#chapters', course='Time Series Analysis',
   h1='Chapter 6 · VAR Models and Granger Causality', back='Course site', pdf_btn='Lecture PDF', ql='Quantlets',
   slide='Slide', ask='Question for the room', answer='Answer',
   note='The slide of the lecture is on the left, the same figure or table on the right, computed in the browser from the same data. The ▶ button moves the parameter step by step; '
        'the “Worked example” boxes redo the calculation with the numbers of the chapter.',
   src='Quarterly data for Romania, 2005Q1 – 2026Q2: real GDP growth (Eurostat, the series of the lecture), 12-month HICP inflation and ROBOR 3M (quarterly means). '
       'Daily log returns in %, 6,433 common days, 6 January 2000 – 18 September 2026: S&P 500, DAX and BET (EODHD). United States: FRED (GDPCTPI, UNRATE, FEDFUNDS), '
       '1960Q1 – 2026Q2. The VARs, tests, impulse responses, FEVDs and the spillover index are re-estimated in the browser by OLS; the initial states are those of the '
       'lecture. The bootstrap bands of the Romanian responses use the draws of the chapter’s seed (500 replications); the simulated VAR(1) uses the Normal draws of the same '
       'seed. Variants that do not appear in the lecture (another lag order p, another Cholesky ordering, another sample, other windows, other effects in the simulation) are '
       'extra runs of the chapter code with the same data and seed; the spillover index with other windows or another horizon is computed every 5 days.'),
}

CSS = ('<style>.c6x{margin:.6rem 0 .3rem;padding:.55rem .8rem;border:1px solid #D5DEEC;border-left:4px solid #17A2B8;border-radius:6px;background:#F7FAFD;min-width:0}'
       '.c6x h4{margin:0 0 .2rem;font-size:.95rem;color:#1A3A6E}.c6x .c6f{overflow-x:auto;overflow-y:hidden;max-width:100%;font-size:.92rem}'
       '.c6x ol{margin:.3rem 0 0 1.25rem;padding:0;font-size:.92rem}.c6x li{margin:.18rem 0}'
       '.c6t{max-width:100%;overflow-x:auto;margin:.55rem 0 .3rem}.c6t table{border-collapse:collapse;font-size:.86rem;margin:0;border-top:2px solid #1A3A6E;border-bottom:2px solid #1A3A6E}'
       '.c6t th,.c6t td{padding:.24rem .55rem;text-align:left;white-space:nowrap;vertical-align:top}.c6t thead th{border-bottom:1px solid #1A3A6E}'
       '.c6t tbody tr:nth-child(even){background:#F5F7FB}.c6t .r{text-align:right}.c6cap{font-size:.84rem;margin:.15rem 0 .3rem;color:#1F2A44;white-space:normal}.c6x .ctl{max-width:360px;margin:.15rem 0 .2rem}</style>')


def tbl(i):
    return f'<div class="tbl c6t" id="{i}"></div>'


def wx(i, title, formula, ctl=''):
    """Worked-example box; `ctl` holds the controls that only drive the example (they stay out of the chart's controls bar)."""
    return f'<div class="c6x"><h4>{title}</h4>{ctl}<div class="c6f">\\[{formula}\\]</div><ol id="{i}"></ol></div>'


def widgets(lang):
    ro = lang == 'ro'
    s = lambda a, b: a if ro else b   # noqa: E731
    lec = s(' (curs)', ' (lecture)')
    W = s('Exemplu rezolvat', 'Worked example')
    vname = {'g': s('creșterea PIB', 'GDP growth'), 'pi': s('inflația', 'inflation'), 'i': 'ROBOR 3M'}
    ords = [('gpi', s('g, π, i', 'g, π, i') + lec), ('gip', 'g, i, π'), ('pgi', 'π, g, i'), ('pig', 'π, i, g'), ('igp', 'i, g, π'), ('ipg', 'i, π, g')]
    ends = [('2026-04-01', s('T2 2026', '2026Q2') + lec), ('2019-10-01', s('T4 2019', '2019Q4')),
            ('2014-10-01', s('T4 2014', '2014Q4'))]
    vgen = {'g': s('creșterii PIB', 'GDP growth'), 'pi': s('inflației', 'inflation'), 'i': 'ROBOR 3M'}
    pairs = [(f'{i}-{j}', (f'{vname[a]} la șocul {vgen[b]}' if ro else f'{vname[a]} to a {vgen[b]} shock') + (lec if (i, j) == (2, 1) else ''))
             for i, a in enumerate(('g', 'pi', 'i')) for j, b in enumerate(('g', 'pi', 'i'))]
    return {
     'ro-macro': (slider('rm-y0', s('Fereastra începe în', 'Window starts in'), 2000, 2022, 1, 2005) + slider('rm-y1', s('Fereastra se termină în', 'Window ends in'), 2008, 2026, 1, 2026) + play('rm'),
                  CSS + charts('rm-chart') + tbl('rm-tb')),
     'ccf': (slider('cc-y0', s('Eșantioanele încep în (România: cel mai devreme în 2005)', 'Samples start in (Romania: 2005 at the earliest)'), 2000, 2024, 1, 2000)
             + select('cc-t', s('ROBOR și inflația (dreapta)', 'ROBOR and inflation (right)'), [('lev', s('niveluri', 'levels') + lec), ('d1', s('diferențe de ordinul întîi', 'first differences'))], 'lev')
             + play('cc'), charts('cc-chart') + tbl('cc-tb')),
     'wx-var1': (slider('w1-a12', 'a₁₂', -0.4, 0.6, 0.05, 0.2) + slider('w1-a21', 'a₂₁', 0, 1.2, 0.05, 0.3) + slider('w1-h', s('Orizontul h', 'Horizon h'), 1, 12, 1, 2) + play('w1'),
                 wx('w1-ws', W, r'\hat{\mathbf Y}_{T+h}=\mathbf c+\mathbf A\hat{\mathbf Y}_{T+h-1},\quad \boldsymbol\mu=(\mathbf I-\mathbf A)^{-1}\mathbf c,\quad '
                    r'\lambda^2-\mathrm{tr}(\mathbf A)\lambda+\det(\mathbf A)=0,\quad \boldsymbol\Sigma(h)=\sum_{s=0}^{h-1}\mathbf A^{s}\boldsymbol\Sigma\mathbf A^{s\prime}')
                 + charts('w1-chart') + tbl('w1-tb')),
     'var-sim': (slider('vs-a', 'a₂₁', 0, 1.2, 0.05, 0.3) + slider('vs-r', s('Corelația șocurilor ρ', 'Correlation of the shocks ρ'), -0.9, 0.9, 0.1, 0.5)
                 + slider('vs-h', s('Orizontul h (calculul de mai jos)', 'Horizon h (calculation below)'), 0, 10, 1, 5) + play('vs'),
                 charts('vs-chart') + wx('vs-ws', W, r'\boldsymbol\Phi_h=\mathbf A\boldsymbol\Phi_{h-1}=\mathbf A^h,\qquad \boldsymbol\Phi_h\mathbf e_1=w_1\lambda_1^h\mathbf v_1+w_2\lambda_2^h\mathbf v_2')
                 + tbl('vs-tb')),
     'roots': (slider('ro-p', s('Ordinul p al VAR-ului românesc', 'Lag order p of the Romanian VAR'), 1, 6, 1, 2) + slider('ro-e', s('Eșantionul se termină în', 'Sample ends in'), 2012, 2026, 1, 2026) + play('ro'),
               charts('ro-chart') + tbl('ro-tb')),
     'ic': (slider('ic-m', 'p<sub>max</sub>', 4, 8, 1, 6) + slider('ic-e', s('Eșantionul se termină în', 'Sample ends in'), 2012, 2026, 1, 2026)
            + slider('ic-p', s('p în calculul de mai jos', 'p in the calculation below'), 0, 6, 1, 2) + play('ic'),
            charts('ic-chart') + tbl('ic-tb') + wx('ic-ws', W, r'\mathrm{IC}(p)=\ln\det\tilde{\boldsymbol\Sigma}(p)+c_T\,\frac{pK^2}{T},\qquad c_T=2\ (\mathrm{AIC}),\ \ 2\ln\ln T\ (\mathrm{HQ}),\ \ \ln T\ (\mathrm{BIC})')),
     'var-table': (slider('vt-p', s('Ordinul p', 'Lag order p'), 1, 4, 1, 2) + select('vt-e', s('Eșantionul se termină în', 'Sample ends in'), ends, '2026-04-01') + play('vt'),
                   tbl('vt-tb')),
     'resid': (slider('re-p', s('Ordinul p', 'Lag order p'), 1, 4, 1, 2)
               + select('re-d', s('Trimestrele de criză', 'Crisis quarters'), [('none', s('fără dummy', 'no dummies') + lec), ('dum', s('dummy pentru T1 2009 și T2 2020', 'dummies for 2009Q1 and 2020Q2'))], 'none')
               + play('re'),
               charts('re-chart') + tbl('re-tb') + wx('re-ws', W, r'Q_h=T^2\sum_{j=1}^{h}\frac{1}{T-j}\,\mathrm{tr}\big(\hat{\mathbf C}_j^{\prime}\hat{\mathbf C}_0^{-1}\hat{\mathbf C}_j\hat{\mathbf C}_0^{-1}\big)\sim\chi^2\big(K^2(h-p)\big)',
                                                     slider('re-h', s('Lagurile h ale testului portmanteau', 'Lags h of the portmanteau test'), 3, 16, 1, 8))),
     'granger-wx': (select('gw-c', s('Cauza', 'Cause'), [('0', vname['g'] + lec), ('1', vname['pi']), ('2', vname['i'])], '0')
                    + select('gw-e', s('Efectul', 'Effect'), [('0', vname['g']), ('1', vname['pi']), ('2', vname['i'] + lec)], '2')
                    + slider('gw-p', s('Ordinul p', 'Lag order p'), 1, 4, 1, 2) + play('gw'),
                    wx('gw-ws', W, r'F=\frac{(RSS_R-RSS_U)/p}{RSS_U/(T-Kp-1)}\sim F(p,\,T-Kp-1),\qquad W=pF\sim\chi^2(p)')),
     'granger-ro': (slider('gr-p', s('Ordinul p', 'Lag order p'), 1, 6, 1, 2)
                    + select('gr-s', s('Sistemul', 'System'), [('tri', s('trei variabile', 'three variables') + lec), ('bi', s('două variabile: perechea testată', 'two variables: the tested pair'))], 'tri')
                    + select('gr-e', s('Eșantionul se termină în', 'Sample ends in'), ends, '2026-04-01') + play('gr'),
                    tbl('gr-tb') + charts('gr-chart')),
     'granger-mkt': (slider('gm-p', s('Ordinul p', 'Lag order p'), 1, 10, 1, 4) + slider('gm-y', s('Eșantionul începe în', 'Sample starts in'), 2000, 2020, 1, 2000) + play('gm'),
                     tbl('gm-tb') + charts('gm-chart')),
     'granger-sim': (select('gs-c', s('Efectul lui z asupra lui y', 'Effect of z on y'), [('0.0', '0'), ('0.4', s('0,4', '0.4')), ('0.8', s('0,8', '0.8') + lec)], '0.8')
                     + select('gs-p', s('Ordinul p al VAR-ului testat', 'Lag order p of the tested VAR'), [('1', '1'), ('2', '2' + lec), ('4', '4')], '2')
                     + select('gs-a', s('Nivelul de semnificație', 'Significance level'), [('0.01', '1%'), ('0.05', '5%' + lec), ('0.1', '10%')], '0.05') + play('gs'),
                     charts('gs-chart') + tbl('gs-tb')),
     'wx-orth': (slider('wo-r', s('Corelația șocurilor ρ', 'Correlation of the shocks ρ'), -0.9, 0.9, 0.1, 0.5)
                 + select('wo-o', s('Ordinea Cholesky', 'Cholesky ordering'), [('y1', '(y₁, y₂)' + lec), ('y2', '(y₂, y₁)')], 'y1')
                 + slider('wo-h', s('Orizontul h', 'Horizon h'), 0, 8, 1, 1) + play('wo'),
                 wx('wo-ws', W, r'\boldsymbol\Sigma=\mathbf P\mathbf P^{\prime},\qquad \boldsymbol\Theta_0=\mathbf P,\qquad \boldsymbol\Theta_h=\boldsymbol\Phi_h\mathbf P=\mathbf A\boldsymbol\Theta_{h-1}')
                 + charts('wo-chart')),
     'irf-ro': (select('ir-o', s('Ordinea Cholesky', 'Cholesky ordering'), ords, 'gpi') + slider('ir-p', s('Ordinul p (benzi doar pentru p = 2)', 'Lag order p (bands for p = 2 only)'), 1, 4, 1, 2)
                + select('ir-j', s('Răspunsul din tabel și din calcul', 'Response in the table and the calculation'), pairs, '2-1')
                + slider('ir-h', s('Orizontul h al calculului', 'Horizon h of the calculation'), 0, 12, 1, 4) + play('ir'),
                charts('ir-chart') + tbl('ir-tb') + wx('ir-ws', W, r'\theta_{jk,h}=\sum_{m=1}^{K}\phi_{jm,h}\,p_{mk},\qquad \boldsymbol\Theta_h=\boldsymbol\Phi_h\mathbf P,\quad \boldsymbol\Sigma=\mathbf P\mathbf P^{\prime}')),
     'irf-order': (slider('io-p', s('Ordinul p', 'Lag order p'), 1, 4, 1, 2) + select('io-e', s('Eșantionul se termină în', 'Sample ends in'), ends, '2026-04-01') + play('io'),
                   charts('io-chart') + wx('io-ws', W, r'\boldsymbol\Theta_0=\mathbf P\ \ (\text{Cholesky}),\qquad \boldsymbol\psi_k(0)=\boldsymbol\Sigma\mathbf e_k/\sqrt{\sigma_{kk}}\ \ (\text{generalised})')),
     'market-irf': (slider('mi-p', s('Ordinul p', 'Lag order p'), 1, 8, 1, 4) + slider('mi-y', s('Eșantionul începe în', 'Sample starts in'), 2000, 2020, 1, 2000) + play('mi'),
                    charts('mi-chart') + tbl('mi-tb') + wx('mi-ws', W, r'\boldsymbol\psi_{S}(0)=\frac{\boldsymbol\Sigma\mathbf e_{S}}{\sqrt{\sigma_{SS}}},\qquad \psi_{j}(0)=\frac{\sigma_{jS}}{\sqrt{\sigma_{SS}}}=\rho_{jS}\,\sqrt{\sigma_{jj}},\quad S=\text{S\&P 500}')),
     'wx-fevd': (slider('wf-r', s('Corelația șocurilor ρ', 'Correlation of the shocks ρ'), -0.9, 0.9, 0.1, 0.5)
                 + select('wf-o', s('Ordinea Cholesky', 'Cholesky ordering'), [('y1', '(y₁, y₂)' + lec), ('y2', '(y₂, y₁)')], 'y1')
                 + select('wf-v', s('Variabila', 'Variable'), [('0', 'y₁'), ('1', 'y₂' + lec)], '1') + slider('wf-h', s('Orizontul h', 'Horizon h'), 1, 10, 1, 2) + play('wf'),
                 wx('wf-ws', W, r'\omega_{jk}(h)=\frac{\sum_{s=0}^{h-1}\theta_{jk,s}^2}{\sum_{s=0}^{h-1}\sum_{m=1}^{K}\theta_{jm,s}^2}') + charts('wf-chart')),
     'fevd-ro': (select('fv-o', s('Ordinea Cholesky', 'Cholesky ordering'), ords, 'gpi') + slider('fv-p', s('Ordinul p', 'Lag order p'), 1, 4, 1, 2)
                 + slider('fv-h', s('Orizontul h al calculului', 'Horizon h of the calculation'), 1, 12, 1, 12) + play('fv'),
                 charts('fv-chart') + tbl('fv-tb') + wx('fv-ws', W, r'\omega_{i\pi}(h)=\frac{\sum_{s=0}^{h-1}\theta_{i\pi,s}^2}{\sum_{s=0}^{h-1}\big(\theta_{ig,s}^2+\theta_{i\pi,s}^2+\theta_{ii,s}^2\big)}')),
     'spill-table': (slider('st-h', s('Orizontul H (zile)', 'Horizon H (days)'), 1, 20, 1, 10) + slider('st-p', s('Ordinul p', 'Lag order p'), 1, 8, 1, 4)
                     + select('st-y', s('Eșantionul', 'Sample'), [('all', '2000–2026' + lec), ('2008', s('anul 2008', 'the year 2008')), ('2017', s('anul 2017', 'the year 2017')), ('2020', s('anul 2020', 'the year 2020'))], 'all')
                     + play('st'),
                     tbl('st-tb') + wx('st-ws', W, r'S(H)=\frac{100}{K}\sum_{j\ne k}\tilde\omega_{jk}(H),\qquad ' + s(r'\text{spre}_k=100\sum_{j\ne k}\tilde\omega_{jk},\quad \text{de la}_j=100\sum_{k\ne j}\tilde\omega_{jk},\quad \text{net}_k=\text{spre}_k-\text{de la}_k', r'\text{to}_k=100\sum_{j\ne k}\tilde\omega_{jk},\quad \text{from}_j=100\sum_{k\ne j}\tilde\omega_{jk},\quad \text{net}_k=\text{to}_k-\text{from}_k'))
                     + charts('st-chart')),
     'spillover': (slider('sp-w', s('Fereastra W (zile)', 'Window W (days)'), 100, 500, 50, 200) + slider('sp-h', s('Orizontul H (zile)', 'Horizon H (days)'), 5, 20, 1, 10)
                   + slider('sp-d', s('Tabelul de mai jos: fereastra care se încheie pe', 'Table below: the window ending on'), 0, 6233, 1, 0) + play('sp'),
                   charts('sp-chart') + tbl('sp-tb')),
     'forecast-ro': (slider('fc-p', s('Ordinul p', 'Lag order p'), 1, 4, 1, 2) + slider('fc-o', s('Originea prognozei', 'Forecast origin'), 63, 105, 1, 105)
                     + select('fc-v', s('Variabila din calcul', 'Variable of the calculation'), [('0', vname['g']), ('1', vname['pi'] + lec), ('2', vname['i'])], '1') + play('fc'),
                     charts('fc-chart') + tbl('fc-tb') + wx('fc-ws', W, r'\hat{\mathbf Y}_{T+1|T}=\mathbf c+\mathbf A_1\mathbf Y_T+\mathbf A_2\mathbf Y_{T-1},\qquad \hat Y_{j,T+h|T}\pm1{,}96\sqrt{\sigma_{jj}(h)},\quad \boldsymbol\Sigma(h)=\sum_{s=0}^{h-1}\boldsymbol\Phi_s\boldsymbol\Sigma\boldsymbol\Phi_s^{\prime}'.replace('1{,}96', '1{,}96' if ro else '1.96'))),
     'oos': (select('oo-c', s('Tabelul și barele evidențiate: țara și orizontul', 'Table and highlighted bars: country and horizon'), [('ro-1', s('România, h = 1', 'Romania, h = 1') + lec), ('ro-4', s('România, h = 4', 'Romania, h = 4')),
                                                                                          ('us-2', s('SUA, h = 2', 'US, h = 2')), ('us-4', s('SUA, h = 4', 'US, h = 4')), ('us-8', s('SUA, h = 8', 'US, h = 8'))], 'ro-1')
             + slider('oo-y', s('Evaluarea țării din tabel începe în (cealaltă țară: cu tot atîția ani mai tîrziu)', 'Evaluation of the table’s country starts in (the other country: as many years later)'), 2015, 2022, 1, 2015) + play('oo'),
             charts('oo-chart') + tbl('oo-tb')),
     'us-data': (slider('us-y0', s('Fereastra începe în', 'Window starts in'), 1960, 2015, 1, 1960) + slider('us-y1', s('Fereastra se termină în', 'Window ends in'), 1970, 2026, 1, 2000) + play('us'),
                 charts('us-chart') + tbl('us-tb')),
     'sw-irf': (select('sw-e', s('Eșantionul se termină în', 'Sample ends in'), [('2000-10-01', s('T4 2000', '2000Q4') + lec), ('2007-10-01', s('T4 2007', '2007Q4')),
                                                                                ('2019-10-01', s('T4 2019', '2019Q4')), ('2026-04-01', s('T2 2026', '2026Q2'))], '2000-10-01')
                + slider('sw-p', s('Ordinul p (benzi doar pentru curs)', 'Lag order p (bands for the lecture only)'), 1, 6, 1, 4) + play('sw'),
                charts('sw-chart') + tbl('sw-tb')),
    }


TEXT_JS = {
 'ro': r"""(function () {
const MON = ['ianuarie', 'februarie', 'martie', 'aprilie', 'mai', 'iunie', 'iulie', 'august', 'septembrie', 'octombrie', 'noiembrie', 'decembrie'];
const dmy = d => `${+d.slice(8, 10)} ${MON[+d.slice(5, 7) - 1]} ${d.slice(0, 4)}`;
const q = d => `T${Math.floor((+d.slice(5, 7) - 1) / 3) + 1} ${d.slice(0, 4)}`;
const de = n => { const r = n % 100; return (r === 0 && n > 0) || r >= 20 ? ' de' : ''; };
const V = { g: 'creșterea PIB', pi: 'inflația', i: 'ROBOR 3M', u: 'șomajul', R: 'dobînda federală' };
const yes = b => (b ? 'da' : 'nu');
const list = a => (a.length ? a.join(', ') : 'niciuna');
window.CH6_TEXT = {
 dmy, q, my: d => `${MON[+d.slice(5, 7) - 1]} ${d.slice(0, 4)}`,
 v: V, vg: { g: 'creșterii PIB', pi: 'inflației', i: 'ROBOR 3M', u: 'șomajului', R: 'dobînzii federale' }, sym: { g: 'g', pi: 'π', i: 'i', u: 'u', R: 'R' },
 L: { var: 'Variabila', mean: 'Media', sd: 'Abaterea standard', re: 'Partea reală', im: 'Partea imaginară', mod: 'Modulul', half: 'Timpul de înjumătățire (perioade)',
   ldet: 'ln det Σ̃(p)', ncoef: 'Coeficienți', const: 'termen liber', reg: 'Regresorul', eq: 'ecuația', skew: 'Asimetria', kurt: 'Kurtosisul', big: 'Cel mai mare reziduu',
   cause: 'Cauza', effect: 'Efectul', df: 'Grade de libertate', bi: 'fără z (VAR cu x, y), %', tri: 'cu z (VAR cu x, y, z), %', resp: 'răspunsul', lo: 'banda: 5%', hi: 'banda: 95%',
   day: 'ziua', gen: 'generalizat', chf: 'Cholesky, S&P 500 primul', chl: 'Cholesky, S&P 500 ultimul', tofrom: 'Spre ← de la (%)', fromo: 'de la ceilalți', too: 'spre ceilalți',
   index: 'indice', net: 'net (spre − de la)', q: 'Trimestrul', rmseRw: 'RMSE mers aleator' },
 rm1: (a, b, n, c, c1, c2) => `${a} – ${b} (${n}${de(n)} trimestre): corelația inflație–ROBOR 3M ${c}; creșterea PIB cu inflația ${c1}, cu ROBOR 3M ${c2}.`,
 rm2: (p, dp, i, di) => `În fereastră: inflația maximă ${p}% (${dp}), ROBOR 3M maxim ${i}% (${di}).`,
 rmCap: (a, b, n) => `Medii trimestriale, ${a} – ${b}, ${n}${de(n)} trimestre; în dreapta, matricea corelațiilor (cu aldine: |r| ≥ 0,5).`,
 cc1: (d, n, b) => `Randamente zilnice de la ${dmy(d)}: T = ${n}${de(n)} zile, banda ±1,96/√T = ±${b}.`,
 cc2: (m, r0, r1, rm, o) => `${m}: ${r0} în aceeași zi, ${r1} la k = 1 (S&P 500 de ieri), ${rm} la k = −1 (${m} de ieri cu S&P 500 de azi); în afara benzii la k = ${list(o)}.`,
 cc3: (t, y, k, r, r4m, r4, b) => `ROBOR cu inflația${t === 'd1' ? ' (diferențe)' : ''}, de la ${y}: maximul ${r} la k = ${k}; ${r4m} la k = −4 și ${r4} la k = 4 (banda pentru seriile trimestriale: ±${b}).`,
 ccCap: b => `Corelațiile încrucișate pentru k = −3, …, 3; cu aldine: în afara benzii ±${b}.`,
 w1s1: (a12, a21, f1, f2) => `Ŷ<sub>T+1</sub> = c + A Y<sub>T</sub>: y<sub>1</sub>: 1,2 + 0,5·5 + ${a12}·2 = <b>${f1}</b>; y<sub>2</sub>: 0,6 + ${a21}·5 + 0,4·2 = <b>${f2}</b>.`,
 w1s2: (f1, f2, a12, a21, g1, g2) => `Ŷ<sub>T+2</sub> = c + A Ŷ<sub>T+1</sub>: 1,2 + 0,5·${f1} + ${a12}·${f2} = <b>${g1}</b>; 0,6 + ${a21}·${f1} + 0,4·${f2} = <b>${g2}</b>.`,
 w1s3: (h, f1, f2, w1, w2, v1, v2) => `Repetînd pînă la h = ${h}: Ŷ<sub>T+${h}</sub> = (${f1}; ${f2})′; varianțele erorii σ<sub>11</sub>(${h}) = ${v1}, σ<sub>22</sub>(${h}) = ${v2}, deci intervalele de 95% sînt ±${w1} și ±${w2}.`,
 w1s4: (t, ac, d, dq, l, m, st) => `Valorile proprii: tr(A) = 0,5 + 0,4 = ${t}, det(A) = 0,5·0,4 − ${ac} = ${d}; ` + (dq === null ? `discriminantul este negativ: λ = ${l[0]} ± ${l[1]}i, ` : `discriminantul tr² − 4 det = ${dq}, λ = (${t} ± √${dq})/2: λ<sub>1</sub> = ${l[0]}, λ<sub>2</sub> = ${l[1]}; `) + `cel mai mare modul ${m}: ${st ? 'VAR stabil' : '<b>VAR instabil</b> (modul ≥ 1)'}.`,
 w1s5: (a12, a21, d, m1, m2) => `Media: det(I − A) = 0,5·0,6 − ${a12}·${a21} = ${d}; μ<sub>1</sub> = (0,6·1,2 + ${a12}·0,6)/${d} = <b>${m1}</b>, μ<sub>2</sub> = (${a21}·1,2 + 0,5·0,6)/${d} = <b>${m2}</b>.`,
 w1s5x: d => `Media: det(I − A) = ${d}; pentru un VAR instabil μ = (I − A)⁻¹c nu este media procesului: prognozele nu converg.`,
 w1o1: (m, a, b) => `Cel mai mare modul al valorilor proprii ${m}: prognozele se apropie de μ = (${a}; ${b})′, cu o viteză dată de λ<sub>1</sub>.`,
 w1o2: m => `Cel mai mare modul al valorilor proprii ${m} ≥ 1: VAR instabil, prognozele nu converg spre o medie.`,
 w1o3: (a, b) => `Pentru h = 2: σ<sub>11</sub>(2) = ${a}, σ<sub>22</sub>(2) = ${b} (Σ + AΣA′).`,
 vs1: (a, r, m1, m2, u1, u2, c) => `a<sub>21</sub> = ${a}, ρ = ${r}: mediile de selecție ${m1} și ${m2} (μ = ${u1} și ${u2}); corelația de selecție a lui y<sub>1</sub> și y<sub>2</sub> ${c}.`,
 vs2: (l1, l2, h, p1, p2, lh) => `Valorile proprii ${l1} și ${l2}; la h = ${h}: răspunsul lui y<sub>1</sub> ${p1}, al lui y<sub>2</sub> ${p2}, iar λ<sub>1</sub><sup>${h}</sup> = ${lh}.`,
 vss1: (h, a, b, a21, c, d) => `Φ<sub>${h}</sub>e<sub>1</sub> = A Φ<sub>${h - 1}</sub>e<sub>1</sub>: din (${a}; ${b})′ obținem 0,5·${a} + 0,2·${b} = <b>${c}</b> și ${a21}·${a} + 0,4·${b} = <b>${d}</b>.`,
 vss2: (c1, l1, c2, l2, d1, d2) => `Descompunerea după vectorii proprii (Φ<sub>h</sub> = V diag(λ<sub>1</sub><sup>h</sup>, λ<sub>2</sub><sup>h</sup>) V⁻¹): Φ<sub>11</sub>(h) = ${c1}·${l1}<sup>h</sup> + ${c2}·${l2}<sup>h</sup>, Φ<sub>21</sub>(h) = ${d1}·${l1}<sup>h</sup> + ${d2}·${l2}<sup>h</sup>.`,
 vss3: (h, w, l1, lh, ap, ex) => `La h = ${h}: ${w} · ${l1}<sup>${h}</sup> = ${w} · ${lh}; cu a doua componentă, Φ<sub>11</sub>(${h}) = <b>${ap}</b> (direct: ${ex}). Răspunsul nu este λ<sub>1</sub><sup>${h}</sup> = ${lh}: acesta este doar factorul cu care scade componenta dominantă; ponderea ei în e<sub>1</sub> este ${w}.`,
 ro1: (p, e, n, k, m, st, hl) => `VAR(${p}) românesc, T1 2005 – ${e} (${n}${de(n)} observații folosite): ${k} valori proprii, cel mai mare modul ${m}, ${st ? 'stabil' : '<b>instabil</b>'}` + (hl ? `; un șoc scade la jumătate după ${hl} trimestre (ln 0,5 / ln ${m}).` : '.'),
 ro2: (u, m, u20, m2) => `VAR(4) american: ${u} (${u}<sup>20</sup> = ${u20} după 20 de trimestre); VAR(4) al piețelor: ${m} (${m}<sup>2</sup> = ${m2} după 2 zile).`,
 roCap: (p, k) => `Cele ${k} valori proprii ale matricei companion a VAR(${p}) românesc (3${p > 1 ? '·' + p : ''} × 3${p > 1 ? '·' + p : ''}), în ordinea modulului.`,
 ic1: (m, T, e, a, h, b) => `p = 0, …, ${m} pe eșantionul comun de T = ${T}${de(T)} trimestre (pînă în ${e}): AIC alege p = ${a}, HQ p = ${h}, BIC p = ${b}.`,
 ic2: (a1, a2, a5, ba) => `AIC: ${a1} pentru p = 1, ${a2} pentru p = 2` + (a5 === null ? '' : `, ${a5} pentru p = 5`) + `; minimul AIC la p = ${ba}.`,
 icCap: T => `Pe același eșantion de T = ${T}${de(T)} trimestre; cu aldine, minimul fiecărui criteriu.`,
 ics1: (T, m, p, K, np) => `Eșantionul comun: T = ${T} (primele ${m} trimestre sînt păstrate pentru laguri); p = ${p}, K = ${K}: pK² = ${np} coeficienți ai lagurilor.`,
 ics2: (p, ld) => `Ajustarea: ln det Σ̃(${p}) = ${ld}, din reziduurile OLS ale celor trei ecuații, cu împărțire la T.`,
 ics3: (np, T, pen, lt, llt) => `Penalizarea pe unitatea lui c<sub>T</sub>: pK²/T = ${np}/${T} = ${pen}; c<sub>T</sub> = 2 (AIC), 2 ln ln T = ${llt} (HQ), ln T = ${lt} (BIC).`,
 ics4: (ld, pa, a, ph, h, pb, b) => `AIC = ${ld} + ${pa} = <b>${a}</b>; HQ = ${ld} + ${ph} = <b>${h}</b>; BIC = ${ld} + ${pb} = <b>${b}</b>.`,
 vtCap: (n, a, b, k) => `OLS pe ${n}${de(n)} trimestre (${a} – ${b}); statisticile t între paranteze; ${k}${de(k)} coeficienți; cu aldine: |t| ≥ 2.`,
 vt1: (g, p, i, c) => `Abaterile standard ale reziduurilor: ${g} (g), ${p} (π), ${i} (i); corelația reziduurilor lui π și i: ${c}.`,
 vt2: (b, t, pp, r2, n, k, df) => `În ecuația ROBOR, coeficientul lui g<sub>t−1</sub> este ${b} (t = ${t}); coeficientul lui π<sub>t−1</sub> în ecuația inflației ${pp}; R² al ecuației PIB ${r2}. Fiecare ecuație are ${df}${de(df)} grade de libertate reziduale (${n}${de(n)} observații).`,
 re1: (a, qa, b, qb, c, qc) => `Cele mai mari reziduuri: PIB ${a} (${qa}), inflația ${b} (${qb}), ROBOR ${c} (${qc}).`,
 re2: (h, qh, df, p, q12, df12, p12) => `Portmanteau: Q<sub>${h}</sub> = ${qh} (χ²(${df}), ${p}); Q<sub>12</sub> = ${q12} (χ²(${df12}), ${p12}).`,
 re3: (s, df, p, d) => `Jarque–Bera multivariat: ${s} (χ²(${df}), ${p})` + (d ? '; cu dummy pentru cele două trimestre de criză.' : '.'),
 reNote: (h, qh, df, p, q12, df12, p12, jb, pj) => `Portmanteau Q<sub>${h}</sub> = ${qh} (${df}${de(df)} grade de libertate, ${p}); Q<sub>12</sub> = ${q12} (${df12}, ${p12}); Jarque–Bera ${jb} (${pj}). Cu aldine: kurtosis peste 4.`,
 res1: (n, K, h, p, df) => `T = ${n} reziduuri, K = ${K}, h = ${h}, p = ${p}: gradele de libertate K²(h − p) = ${K * K}·${h - p} = ${df}.`,
 res2: (t, q) => `Termenii T²/(T − j)·tr(Ĉ<sub>j</sub>′Ĉ<sub>0</sub>⁻¹Ĉ<sub>j</sub>Ĉ<sub>0</sub>⁻¹) pentru j = 1, 2, 3: ${t.join('; ')}; …; suma pînă la h: Q = <b>${q}</b>.`,
 res3: (q, df, p, r) => `p-value = P(χ²(${df}) > ${q}) = <b>${p}</b>${r ? ': respingem ipoteza că reziduurile sînt zgomot alb la 5%.' : ': la 5% testul nu separă autocorelația reziduurilor de zgomot.'}`,
 gws1: (e, c, n, p, k, r) => `Ecuația lui ${e}<sub>t</sub>, cu T = ${n}, ${k} regresori (constanta și ${p} laguri ale celor trei variabile): RSS<sub>U</sub> = <b>${r}</b>.`,
 gws2: (c, p, k, r) => `Fără cele ${p} laguri ale lui ${c} (${k} regresori): RSS<sub>R</sub> = <b>${r}</b>.`,
 gws3: (rr, ru, p, d2, a, b, F) => `F = [(${rr} − ${ru})/${p}] / [${ru}/${d2}] = ${a} / ${b} = <b>${F}</b>.`,
 gws4: (p, d2, c, pv, r) => `Grade de libertate (${p}, ${d2}); valoarea critică de 5%: ${c}; p-value ${pv}, ${r ? 'sub' : 'peste'} 0,05.`,
 gws5: (p, F, W, pw) => `Versiunea Wald: W = pF = ${p}·${F} = ${W}, p-value din χ²(${p}): ${pw}.`,
 gws6: (r, c, e) => r ? `Decizia: respingem H<sub>0</sub>: trecutul variabilei ${c} ajută la prognoza variabilei ${e}, dat fiind trecutul celorlalte două.`
   : `Decizia: nu respingem H<sub>0</sub>: în acest VAR, lagurile variabilei ${c} nu îmbunătățesc prognoza variabilei ${e} într-un mod separat de zgomot.`,
 gw1: (c, e, p, F, d1, d2, pv) => `${c} → ${e}, VAR(${p}): F = ${F}, F(${d1}, ${d2}), ${pv}.`,
 grCap: (s, p, e, c) => `Teste F(${p}, ·) în VAR(${p}) ${s === 'bi' ? 'cu două variabile (perechea testată)' : 'cu trei variabile'}, T1 2005 – ${e}; valoarea critică de 5%: ${c}; cu aldine: p &lt; 0,05.`,
 gr1: s => `Semnificative la 5%: ${list(s)}.`,
 gr2: (F1, p1, F2, p2) => `Inflația → creșterea PIB: F = ${F1} (${p1}); ROBOR 3M → inflația: F = ${F2} (${p2}).`,
 gmCap: (p, n, y) => `Randamente logaritmice zilnice din ${y}, VAR(${p}), T = ${n}${de(n)} zile; cu aldine: p &lt; 0,05.`,
 gm1: (b, s, d, p) => `Coeficientul S&P 500 de ieri în ecuația BET: ${b} (SE ${s}); în ecuația DAX: ${d} (VAR(${p})).`,
 gm2: (F1, p1, F2, p2) => `BET → S&P 500: F = ${F1} (${p1}); DAX → S&P 500: F = ${F2} (${p2}).`,
 gs1: (c, p, a, rows) => `Efectul lui z asupra lui y ${c}, VAR(${p}), nivelul ${a}%: respingeri fără z / cu z: ` + rows.map(([T, b, t]) => `T = ${T}: ${b}% / ${t}%`).join('; ') + '.',
 gsCap: (c, p, a, R) => `Procentul din cele ${R} de simulări în care testul F respinge „x nu cauzează pe y în sens Granger” la ${a}% (efectul lui z asupra lui y ${c}, VAR(${p})).`,
 wos1: (r, p21, p22) => `Cholesky în ordinea (y<sub>1</sub>, y<sub>2</sub>): p<sub>11</sub> = √σ<sub>11</sub> = 1, p<sub>21</sub> = σ<sub>21</sub>/p<sub>11</sub> = ${p21}, p<sub>22</sub> = √(1 − ${r}²) = ${p22}.`,
 wos1b: (r, p11, p12) => `Cholesky cu y<sub>2</sub> primul: y<sub>2</sub> primește doar șocul u<sub>2</sub>, deci p<sub>21</sub> = 0, p<sub>22</sub> = 1, p<sub>12</sub> = ${p12}, p<sub>11</sub> = √(1 − ${r}²) = ${p11}.`,
 wos2: (P, a, b, c, d) => `Θ<sub>0</sub> = P = ${P}: șocul u<sub>1</sub> = 1 mișcă y<sub>1</sub> cu ${a} și y<sub>2</sub> cu ${b}; u<sub>2</sub> = 1 mișcă y<sub>1</sub> cu ${c} și y<sub>2</sub> cu ${d}.`,
 wos3: (h, a, b, c, d, e, f, g, k) => `Θ<sub>${h}</sub> = A Θ<sub>${h - 1}</sub>: coloana lui u<sub>1</sub>: 0,5·${a} + 0,2·${b} = <b>${c}</b>, 0,3·${a} + 0,4·${b} = <b>${d}</b>; coloana lui u<sub>2</sub>: 0,5·${e} + 0,2·${f} = <b>${g}</b>, 0,3·${e} + 0,4·${f} = <b>${k}</b>.`,
 wos4: (h, M) => `Θ<sub>${h}</sub> = ${M} (rîndul: variabila; coloana: șocul).`,
 wo1: (o, a, b, c, d) => `Ordinea ${o === 'y2' ? '(y₂, y₁)' : '(y₁, y₂)'}: y<sub>2</sub> la u<sub>1</sub>: ${a} la impact, ${b} după o perioadă; y<sub>1</sub> la u<sub>2</sub>: ${c} la impact, ${d} după o perioadă.`,
 ir1: (o, p, pp, i0, m, hm, i12, ns) => `Ordinea ${o}, VAR(${p}): un șoc al inflației (${pp} pp la impact) crește ROBOR cu ${i0} pp la impact și pînă la ${m} pp după ${hm} trimestre; ${i12} pp după 12` + (ns === null ? '.' : `; banda exclude zero la ${ns} din 13 orizonturi.`),
 ir2: (g1, pm, hp, p4, p12) => `Un șoc ROBOR schimbă creșterea PIB cu ${g1} pp în trimestrul următor; inflația după un șoc ROBOR: maximul ${pm} pp (la ${hp} trimestre). Inflația după propriul șoc: ${p4} pp după 4 trimestre, ${p12} după 12.`,
 ir3: p => `Benzile bootstrap sînt calculate pentru p = 2 (cu extragerile seminței capitolului); pentru p = ${p} se arată doar răspunsurile.`,
 irCap: (r, s) => `Răspunsul variabilei ${r} la un șoc al ${s} de o abatere standard; cu aldine: banda de 90% exclude zero.`,
 irs1: (o, P) => `Ordinea ${o}: P = chol(Σ̂) = ${P} (rîndurile și coloanele în ordinea de estimare).`,
 irs2: (h, r, s, t, v) => `θ<sub>${r}${s},${h}</sub> = Σ<sub>m</sub> φ<sub>${r}m,${h}</sub> p<sub>m${s}</sub> = ${t.join(' + ')} = <b>${v}</b>.`,
 irs3: (o, j, s) => `Șocul lui ${s} este pe poziția ${j + 1} în ordinea (${o.join(', ')}): variabilele dinaintea lui nu reacționează la el în trimestrul șocului.`,
 io1: (p, e, a, b, g) => `VAR(${p}), pînă în ${e}: maximul răspunsului ROBOR la un șoc al inflației ${a} pp (inflația înaintea ROBOR), ${b} pp (ROBOR primul), ${g} pp (generalizat).`,
 io2: (a, b, i, c) => `Inflația după un șoc ROBOR: maximul ${a} pp, față de ${b} pp cu ROBOR primul (${i} pp deja la impact); corelația reziduurilor lui π și i: ${c}.`,
 ios1: S => `Σ̂ în ordinea (g, π, i) = ${S}.`,
 ios2: (s, p31, p21, p22, p32) => `Inflația înaintea ROBOR: p<sub>iπ</sub> = (σ<sub>iπ</sub> − p<sub>ig</sub>p<sub>πg</sub>)/p<sub>ππ</sub> = (${s} − ${p31}·${p21})/${p22} = <b>${p32}</b>; cu ROBOR înaintea inflației, răspunsul ROBOR la șocul inflației la impact este 0.`,
 ios3: (s, v, g) => `Generalizat: ψ<sub>π</sub>(0) pentru ROBOR = σ<sub>iπ</sub>/√σ<sub>ππ</sub> = ${s}/√${v} = <b>${g}</b>, fără nicio ordine.`,
 ios4: (a, g) => `Diferența ${a} față de ${g}: Cholesky scade din covarianță partea explicată de șocul PIB (așezat primul); răspunsul generalizat o păstrează pe toată.`,
 mi1: (p, y, s, d0, d1, b0, b1) => `VAR(${p}) din ${y}, șoc S&P 500 de o abatere standard (${s}%): generalizat DAX ${d0}% în aceeași zi și ${d1}% a doua zi; BET ${b0}% și ${b1}%.`,
 mi2: (d, b, c) => `Cu S&P 500 ultimul: zero în aceeași zi, apoi DAX ${d}% și BET ${b}%; corelația reziduurilor S&P 500 și DAX ${c}.`,
 mis1: (a, b, c) => `Din Σ̂: σ<sub>S&P,S&P</sub> = ${a}, σ<sub>DAX,S&P</sub> = ${b}, σ<sub>BET,S&P</sub> = ${c}.`,
 mis2: (b, s, d, c, e) => `Impactul generalizat: DAX ${b}/${s} = <b>${d}</b>; BET ${c}/${s} = <b>${e}</b> (√σ<sub>S&P,S&P</sub> = ${s}).`,
 mis3: (r, s, v) => `Echivalent: ρ<sub>DAX,S&P</sub>·√σ<sub>DAX</sub> = ${r}·${s} = ${v}; cu S&P 500 primul Cholesky dă exact aceleași răspunsuri.`,
 wfs1: (h, j, l) => `Răspunsurile lui y<sub>${j}</sub> pentru s = 0, …, ${h - 1}: ${l.join('; ')}.`,
 wfs2: (j, a, sa, b, sb, m) => `Sume de pătrate: u<sub>1</sub>: ${a.join(' + ')} = ${sa}; u<sub>2</sub>: ${b.join(' + ')} = ${sb}; varianța erorii lui y<sub>${j}</sub>: ${m}.`,
 wfs3: (j, h, a, m, s1, s2) => `ω<sub>${j}1</sub>(${h}) = ${a}/${m} = <b>${s1}%</b>; ω<sub>${j}2</sub>(${h}) = <b>${s2}%</b>; rîndul însumează 100%.`,
 wf1: (j, h, m, s, s10) => `y<sub>${j}</sub>, h = ${h}: varianța erorii ${m}; ponderea lui u<sub>1</sub> ${s}% (la h = 10: ${s10}%).`,
 fv1: (o, p, a, b, c, d) => `Ordinea ${o}, VAR(${p}): ROBOR, propriile șocuri ${a}% la un trimestru și ${b}% la 12; la 12 trimestre șocurile inflației explică ${c}%, cele ale PIB-ului ${d}%.`,
 fv2: (a, b) => `La 12 trimestre: inflația ${a}% din propriile șocuri, creșterea PIB ${b}%.`,
 fvCap: o => `Ponderile (%) șocurilor ortogonale (coloanele) în varianța erorii de prognoză, ordinea ${o}; cu aldine: orizontul din calcul.`,
 fvs1: (h, l, more, s) => `Răspunsurile ROBOR la șocul inflației, s = 0, 1, …: ${l.join('; ')}${more ? '; …' : ''}; Σ<sub>s&lt;${h}</sub> θ<sub>iπ,s</sub>² = <b>${s}</b>.`,
 fvs2: (h, d) => `Varianța erorii de prognoză la ${h} pași a ROBOR, suma pătratelor răspunsurilor la toate cele trei șocuri: <b>${d}</b>.`,
 fvs3: (h, a, b, s) => `ω<sub>iπ</sub>(${h}) = ${a}/${b} = <b>${s}%</b>.`,
 stCap: (H, p, y, n) => `FEVD generalizată la H = ${H} zile, VAR(${p}), ${y ? 'anul ' + y : '2000–2026'} (T = ${n}); rîndul j: ponderile varianței erorii de prognoză a pieței j.`,
 sts1: (H, p) => `Fiecare rînd al FEVD generalizate (H = ${H}, VAR(${p})) este rescalat să însumeze 100%; diagonala (cu aldine) este partea propriilor șocuri.`,
 sts2: (f, s, S) => `„De la ceilalți” pe rînduri: ${f.join(' + ')} = ${s}; împărțit la K = 3: indicele S = <b>${S}%</b>.`,
 sts3: rows => rows.map(([m, t, f, n]) => `${m}: spre ceilalți ${t}, de la ceilalți ${f}, net ${n}`).join('; ') + '.',
 st1: (S, a, b, f, t) => `Indicele ${S}%; S&P 500 primește ${a}% de la DAX, DAX ${b}% de la S&P 500; BET primește ${f}% și transmite ${t}%.`,
 sp1: (W, H, st, m, x, dx, mn, dmn, l, f) => `Ferestre de ${W}${de(W)} zile, H = ${H}${st > 1 ? `, calculat din ${st} în ${st} zile` : ''}: media ${m}%, maximul ${x}% (${dx}), minimul ${mn}% (${dmn}), ultima valoare ${l}%; întregul eșantion ${f}%.`,
 sp2: (a, b, c) => `Media în 2007: ${a}%; T4 2008: ${b}%; martie–iunie 2020: ${c}%.`,
 spCap: (W, a, b, H, S) => `Tabelul de spillover al ferestrei ${a} – ${b} (${W}${de(W)} zile, H = ${H}): indicele ${S}%.`,
 fc1: (o, p, l, a, b, lo, hi) => `Originea ${o}, VAR(${p}): inflația de la ${l}% la ${a}% după un an și ${b}% după doi (95%: [${lo}; ${hi}]).`,
 fc2: (i4, i8, g, lo, hi, mp, mi) => `ROBOR: ${i4}% după un an, ${i8}% după doi; creșterea PIB după un an: ${g}% pe trimestru [${lo}; ${hi}]; media VAR-ului: inflația ${mp}%, ROBOR ${mi}%.`,
 fc3: (qq, a, f, ins) => `Realizat în ${qq}: inflația ${a}%, față de prognoza ${f}%; ${ins ? 'în interiorul' : '<b>în afara</b>'} intervalului de 95%.`,
 fcCap: (p, o) => `Prognoze VAR(${p}) din ${o}, cu intervale de 95% între paranteze (incertitudinea parametrilor nu este inclusă).`,
 fcs1: (s, o, a, b) => `Ultimele observații: Y<sub>T</sub> = ${a} (${o})` + (b ? `, Y<sub>T−1</sub> = ${b}` : '') + ` în ordinea (g, π, i).`,
 fcs2: (s, e, more, v) => `Ecuația lui ${s}: Ŷ = ${e}${more ? ' + …' : ''} = <b>${v}</b>.`,
 fcs3: (s, v1, pt, hw, vH, H) => `σ<sub>${s}${s}</sub>(1) = ${v1}, interval ${pt} ± ${hw}; la ${H} pași varianța erorii crește la ${vH}.`,
 ooCap: (c, h, f) => `${c === 'ro' ? 'România' : 'Statele Unite'}, h = ${h}, ținte de la ${f}; RMSE relativ la mersul aleator (cu aldine: modelul mai precis dintre VAR și AR); DM: testul Diebold–Mariano VAR față de AR.`,
 oo1: (v, a, z, p, n) => `România, h = 1, creșterea PIB: VAR ${v}, AR ${a} (RMSE relativ la mersul aleator); DM = ${z}, ${p} (${n} prognoze).`,
 oo2: (uv, ua, rv, ra, pv, pa) => `SUA, h = 8: șomajul VAR ${uv}, AR ${ua}; dobînda federală VAR ${rv}, AR ${ra}; inflația VAR ${pv}, AR ${pa}.`,
 us1: (a, b, n, mp, mu, mr, c1, c2) => `${a} – ${b} (${n}${de(n)} trimestre): media inflației ${mp}%, a șomajului ${mu}%, a dobînzii federale ${mr}%; corelația inflație–dobîndă ${c1}, șomaj–dobîndă ${c2}.`,
 usCap: (a, b, n) => `Statistici pentru ${a} – ${b} (${n}${de(n)} trimestre).`,
 sw1: (e, p, n, r0, um, hu, pm, hp, p12, u2) => `VAR(${p}), T1 1960 – ${e} (T = ${n}): un șoc monetar de ${r0} pp crește șomajul cu pînă la ${um} pp după ${hu} trimestre (${u2} pp după 2); inflația: maximul ${pm} pp (la ${hp}), ${p12} pp după 12 trimestre.`,
 sw2: (a, b, m, lec) => `FEVD la 12 trimestre: șocurile monetare explică ${a}% din șomaj și ${b}% din inflație; cel mai mare modul ${m}` + (lec ? '; benzile de 90% sînt cele din curs.' : '; benzile bootstrap sînt arătate doar pentru specificația din curs.'),
 swCap: (p, e, n) => `Teste Granger F în VAR(${p}), T1 1960 – ${e} (T = ${n}); cu aldine: p &lt; 0,05.`,
 swNote: rows => 'FEVD la 12 trimestre (șocuri inflație / șomaj / dobîndă, %): ' + rows.map(([v, a, b, c]) => `${v}: ${a} / ${b} / ${c}`).join('; ') + '.',
};
})();
""",
 'en': r"""(function () {
const MON = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'];
const dmy = d => `${+d.slice(8, 10)} ${MON[+d.slice(5, 7) - 1]} ${d.slice(0, 4)}`;
const q = d => `${d.slice(0, 4)}Q${Math.floor((+d.slice(5, 7) - 1) / 3) + 1}`;
const V = { g: 'GDP growth', pi: 'inflation', i: 'ROBOR 3M', u: 'unemployment', R: 'fed funds rate' };
const list = a => (a.length ? a.join(', ') : 'none');
window.CH6_TEXT = {
 dmy, q, my: d => `${MON[+d.slice(5, 7) - 1]} ${d.slice(0, 4)}`,
 v: V, vg: V, sym: { g: 'g', pi: 'π', i: 'i', u: 'u', R: 'R' },
 L: { var: 'Variable', mean: 'Mean', sd: 'Standard deviation', re: 'Real part', im: 'Imaginary part', mod: 'Modulus', half: 'Half-life (periods)',
   ldet: 'ln det Σ̃(p)', ncoef: 'Coefficients', const: 'constant', reg: 'Regressor', eq: 'equation', skew: 'Skewness', kurt: 'Kurtosis', big: 'Largest residual',
   cause: 'Cause', effect: 'Effect', df: 'Degrees of freedom', bi: 'without z (VAR in x, y), %', tri: 'with z (VAR in x, y, z), %', resp: 'response', lo: 'band: 5%', hi: 'band: 95%',
   day: 'day', gen: 'generalised', chf: 'Cholesky, S&P 500 first', chl: 'Cholesky, S&P 500 last', tofrom: 'To ← from (%)', fromo: 'from others', too: 'to others',
   index: 'index', net: 'net (to − from)', q: 'Quarter', rmseRw: 'RMSE random walk' },
 rm1: (a, b, n, c, c1, c2) => `${a} – ${b} (${n} quarters): correlation of inflation and ROBOR 3M ${c}; GDP growth with inflation ${c1}, with ROBOR 3M ${c2}.`,
 rm2: (p, dp, i, di) => `In the window: highest inflation ${p}% (${dp}), highest ROBOR 3M ${i}% (${di}).`,
 rmCap: (a, b, n) => `Quarterly means, ${a} – ${b}, ${n} quarters; on the right, the correlation matrix (bold: |r| ≥ 0.5).`,
 cc1: (d, n, b) => `Daily returns from ${dmy(d)}: T = ${n} days, band ±1.96/√T = ±${b}.`,
 cc2: (m, r0, r1, rm, o) => `${m}: ${r0} on the same day, ${r1} at k = 1 (yesterday’s S&P 500), ${rm} at k = −1 (yesterday’s ${m} with today’s S&P 500); outside the band at k = ${list(o)}.`,
 cc3: (t, y, k, r, r4m, r4, b) => `ROBOR with inflation${t === 'd1' ? ' (differences)' : ''} from ${y}: largest ${r} at k = ${k}; ${r4m} at k = −4 and ${r4} at k = 4 (band for the quarterly series: ±${b}).`,
 ccCap: b => `Cross-correlations for k = −3, …, 3; bold: outside the band ±${b}.`,
 w1s1: (a12, a21, f1, f2) => `Ŷ<sub>T+1</sub> = c + A Y<sub>T</sub>: y<sub>1</sub>: 1.2 + 0.5·5 + ${a12}·2 = <b>${f1}</b>; y<sub>2</sub>: 0.6 + ${a21}·5 + 0.4·2 = <b>${f2}</b>.`,
 w1s2: (f1, f2, a12, a21, g1, g2) => `Ŷ<sub>T+2</sub> = c + A Ŷ<sub>T+1</sub>: 1.2 + 0.5·${f1} + ${a12}·${f2} = <b>${g1}</b>; 0.6 + ${a21}·${f1} + 0.4·${f2} = <b>${g2}</b>.`,
 w1s3: (h, f1, f2, w1, w2, v1, v2) => `Repeating up to h = ${h}: Ŷ<sub>T+${h}</sub> = (${f1}; ${f2})′; error variances σ<sub>11</sub>(${h}) = ${v1}, σ<sub>22</sub>(${h}) = ${v2}, so the 95% intervals are ±${w1} and ±${w2}.`,
 w1s4: (t, ac, d, dq, l, m, st) => `Eigenvalues: tr(A) = 0.5 + 0.4 = ${t}, det(A) = 0.5·0.4 − ${ac} = ${d}; ` + (dq === null ? `the discriminant is negative: λ = ${l[0]} ± ${l[1]}i, ` : `discriminant tr² − 4 det = ${dq}, λ = (${t} ± √${dq})/2: λ<sub>1</sub> = ${l[0]}, λ<sub>2</sub> = ${l[1]}; `) + `largest modulus ${m}: ${st ? 'stable VAR' : '<b>unstable VAR</b> (modulus ≥ 1)'}.`,
 w1s5: (a12, a21, d, m1, m2) => `Mean: det(I − A) = 0.5·0.6 − ${a12}·${a21} = ${d}; μ<sub>1</sub> = (0.6·1.2 + ${a12}·0.6)/${d} = <b>${m1}</b>, μ<sub>2</sub> = (${a21}·1.2 + 0.5·0.6)/${d} = <b>${m2}</b>.`,
 w1s5x: d => `Mean: det(I − A) = ${d}; for an unstable VAR μ = (I − A)⁻¹c is not the mean of the process: the forecasts do not converge.`,
 w1o1: (m, a, b) => `Largest eigenvalue modulus ${m}: the forecasts approach μ = (${a}; ${b})′ at a speed set by λ<sub>1</sub>.`,
 w1o2: m => `Largest eigenvalue modulus ${m} ≥ 1: unstable VAR, the forecasts do not converge to a mean.`,
 w1o3: (a, b) => `For h = 2: σ<sub>11</sub>(2) = ${a}, σ<sub>22</sub>(2) = ${b} (Σ + AΣA′).`,
 vs1: (a, r, m1, m2, u1, u2, c) => `a<sub>21</sub> = ${a}, ρ = ${r}: sample means ${m1} and ${m2} (μ = ${u1} and ${u2}); sample correlation of y<sub>1</sub> and y<sub>2</sub> ${c}.`,
 vs2: (l1, l2, h, p1, p2, lh) => `Eigenvalues ${l1} and ${l2}; at h = ${h}: response of y<sub>1</sub> ${p1}, of y<sub>2</sub> ${p2}, and λ<sub>1</sub><sup>${h}</sup> = ${lh}.`,
 vss1: (h, a, b, a21, c, d) => `Φ<sub>${h}</sub>e<sub>1</sub> = A Φ<sub>${h - 1}</sub>e<sub>1</sub>: from (${a}; ${b})′ we get 0.5·${a} + 0.2·${b} = <b>${c}</b> and ${a21}·${a} + 0.4·${b} = <b>${d}</b>.`,
 vss2: (c1, l1, c2, l2, d1, d2) => `Eigenvector decomposition (Φ<sub>h</sub> = V diag(λ<sub>1</sub><sup>h</sup>, λ<sub>2</sub><sup>h</sup>) V⁻¹): Φ<sub>11</sub>(h) = ${c1}·${l1}<sup>h</sup> + ${c2}·${l2}<sup>h</sup>, Φ<sub>21</sub>(h) = ${d1}·${l1}<sup>h</sup> + ${d2}·${l2}<sup>h</sup>.`,
 vss3: (h, w, l1, lh, ap, ex) => `At h = ${h}: ${w} · ${l1}<sup>${h}</sup> = ${w} · ${lh}; with the second component, Φ<sub>11</sub>(${h}) = <b>${ap}</b> (directly: ${ex}). The response is not λ<sub>1</sub><sup>${h}</sup> = ${lh}: that is only the factor by which the dominant component shrinks; its weight in e<sub>1</sub> is ${w}.`,
 ro1: (p, e, n, k, m, st, hl) => `Romanian VAR(${p}), 2005Q1 – ${e} (${n} observations used): ${k} eigenvalues, largest modulus ${m}, ${st ? 'stable' : '<b>unstable</b>'}` + (hl ? `; a shock halves after ${hl} quarters (ln 0.5 / ln ${m}).` : '.'),
 ro2: (u, m, u20, m2) => `US VAR(4): ${u} (${u}<sup>20</sup> = ${u20} after 20 quarters); market VAR(4): ${m} (${m}<sup>2</sup> = ${m2} after 2 days).`,
 roCap: (p, k) => `The ${k} eigenvalues of the companion matrix of the Romanian VAR(${p}) (3${p > 1 ? '·' + p : ''} × 3${p > 1 ? '·' + p : ''}), by modulus.`,
 ic1: (m, T, e, a, h, b) => `p = 0, …, ${m} on the common sample of T = ${T} quarters (up to ${e}): AIC picks p = ${a}, HQ p = ${h}, BIC p = ${b}.`,
 ic2: (a1, a2, a5, ba) => `AIC: ${a1} for p = 1, ${a2} for p = 2` + (a5 === null ? '' : `, ${a5} for p = 5`) + `; AIC minimum at p = ${ba}.`,
 icCap: T => `On the same sample of T = ${T} quarters; bold: the minimum of each criterion.`,
 ics1: (T, m, p, K, np) => `Common sample: T = ${T} (the first ${m} quarters are kept for the lags); p = ${p}, K = ${K}: pK² = ${np} lag coefficients.`,
 ics2: (p, ld) => `Fit: ln det Σ̃(${p}) = ${ld}, from the OLS residuals of the three equations, divided by T.`,
 ics3: (np, T, pen, lt, llt) => `Penalty per unit of c<sub>T</sub>: pK²/T = ${np}/${T} = ${pen}; c<sub>T</sub> = 2 (AIC), 2 ln ln T = ${llt} (HQ), ln T = ${lt} (BIC).`,
 ics4: (ld, pa, a, ph, h, pb, b) => `AIC = ${ld} + ${pa} = <b>${a}</b>; HQ = ${ld} + ${ph} = <b>${h}</b>; BIC = ${ld} + ${pb} = <b>${b}</b>.`,
 vtCap: (n, a, b, k) => `OLS on ${n} quarters (${a} – ${b}); t statistics in parentheses; ${k} coefficients; bold: |t| ≥ 2.`,
 vt1: (g, p, i, c) => `Residual standard deviations: ${g} (g), ${p} (π), ${i} (i); correlation of the π and i residuals: ${c}.`,
 vt2: (b, t, pp, r2, n, k, df) => `In the ROBOR equation the coefficient of g<sub>t−1</sub> is ${b} (t = ${t}); coefficient of π<sub>t−1</sub> in the inflation equation ${pp}; R² of the GDP equation ${r2}. Each equation has ${df} residual degrees of freedom (${n} observations).`,
 re1: (a, qa, b, qb, c, qc) => `Largest residuals: GDP ${a} (${qa}), inflation ${b} (${qb}), ROBOR ${c} (${qc}).`,
 re2: (h, qh, df, p, q12, df12, p12) => `Portmanteau: Q<sub>${h}</sub> = ${qh} (χ²(${df}), ${p}); Q<sub>12</sub> = ${q12} (χ²(${df12}), ${p12}).`,
 re3: (s, df, p, d) => `Multivariate Jarque–Bera: ${s} (χ²(${df}), ${p})` + (d ? '; with dummies for the two crisis quarters.' : '.'),
 reNote: (h, qh, df, p, q12, df12, p12, jb, pj) => `Portmanteau Q<sub>${h}</sub> = ${qh} (${df} degrees of freedom, ${p}); Q<sub>12</sub> = ${q12} (${df12}, ${p12}); Jarque–Bera ${jb} (${pj}). Bold: kurtosis above 4.`,
 res1: (n, K, h, p, df) => `T = ${n} residuals, K = ${K}, h = ${h}, p = ${p}: degrees of freedom K²(h − p) = ${K * K}·${h - p} = ${df}.`,
 res2: (t, q) => `The terms T²/(T − j)·tr(Ĉ<sub>j</sub>′Ĉ<sub>0</sub>⁻¹Ĉ<sub>j</sub>Ĉ<sub>0</sub>⁻¹) for j = 1, 2, 3: ${t.join('; ')}; …; the sum up to h: Q = <b>${q}</b>.`,
 res3: (q, df, p, r) => `p-value = P(χ²(${df}) > ${q}) = <b>${p}</b>${r ? ': we reject that the residuals are white noise at 5%.' : ': at 5% the test does not separate the residual autocorrelation from noise.'}`,
 gws1: (e, c, n, p, k, r) => `The equation of ${e}<sub>t</sub>, with T = ${n}, ${k} regressors (the constant and ${p} lags of the three variables): RSS<sub>U</sub> = <b>${r}</b>.`,
 gws2: (c, p, k, r) => `Without the ${p} lags of ${c} (${k} regressors): RSS<sub>R</sub> = <b>${r}</b>.`,
 gws3: (rr, ru, p, d2, a, b, F) => `F = [(${rr} − ${ru})/${p}] / [${ru}/${d2}] = ${a} / ${b} = <b>${F}</b>.`,
 gws4: (p, d2, c, pv, r) => `Degrees of freedom (${p}, ${d2}); 5% critical value: ${c}; p-value ${pv}, ${r ? 'below' : 'above'} 0.05.`,
 gws5: (p, F, W, pw) => `Wald version: W = pF = ${p}·${F} = ${W}, p-value from χ²(${p}): ${pw}.`,
 gws6: (r, c, e) => r ? `Decision: reject H<sub>0</sub>: the past of ${c} helps to forecast ${e}, given the past of the other two variables.`
   : `Decision: do not reject H<sub>0</sub>: in this VAR the lags of ${c} do not improve the forecast of ${e} in a way separated from noise.`,
 gw1: (c, e, p, F, d1, d2, pv) => `${c} → ${e}, VAR(${p}): F = ${F}, F(${d1}, ${d2}), ${pv}.`,
 grCap: (s, p, e, c) => `F(${p}, ·) tests in the ${s === 'bi' ? 'two-variable VAR(' + p + ') of the tested pair' : 'three-variable VAR(' + p + ')'}, 2005Q1 – ${e}; 5% critical value: ${c}; bold: p &lt; 0.05.`,
 gr1: s => `Significant at 5%: ${list(s)}.`,
 gr2: (F1, p1, F2, p2) => `Inflation → GDP growth: F = ${F1} (${p1}); ROBOR 3M → inflation: F = ${F2} (${p2}).`,
 gmCap: (p, n, y) => `Daily log returns from ${y}, VAR(${p}), T = ${n} days; bold: p &lt; 0.05.`,
 gm1: (b, s, d, p) => `Coefficient of yesterday’s S&P 500 in the BET equation: ${b} (SE ${s}); in the DAX equation: ${d} (VAR(${p})).`,
 gm2: (F1, p1, F2, p2) => `BET → S&P 500: F = ${F1} (${p1}); DAX → S&P 500: F = ${F2} (${p2}).`,
 gs1: (c, p, a, rows) => `Effect of z on y ${c}, VAR(${p}), level ${a}%: rejections without z / with z: ` + rows.map(([T, b, t]) => `T = ${T}: ${b}% / ${t}%`).join('; ') + '.',
 gsCap: (c, p, a, R) => `Share of the ${R} simulations in which the F test rejects “x does not Granger-cause y” at ${a}% (effect of z on y ${c}, VAR(${p})).`,
 wos1: (r, p21, p22) => `Cholesky in the order (y<sub>1</sub>, y<sub>2</sub>): p<sub>11</sub> = √σ<sub>11</sub> = 1, p<sub>21</sub> = σ<sub>21</sub>/p<sub>11</sub> = ${p21}, p<sub>22</sub> = √(1 − ${r}²) = ${p22}.`,
 wos1b: (r, p11, p12) => `Cholesky with y<sub>2</sub> first: y<sub>2</sub> receives only the shock u<sub>2</sub>, so p<sub>21</sub> = 0, p<sub>22</sub> = 1, p<sub>12</sub> = ${p12}, p<sub>11</sub> = √(1 − ${r}²) = ${p11}.`,
 wos2: (P, a, b, c, d) => `Θ<sub>0</sub> = P = ${P}: a shock u<sub>1</sub> = 1 moves y<sub>1</sub> by ${a} and y<sub>2</sub> by ${b}; u<sub>2</sub> = 1 moves y<sub>1</sub> by ${c} and y<sub>2</sub> by ${d}.`,
 wos3: (h, a, b, c, d, e, f, g, k) => `Θ<sub>${h}</sub> = A Θ<sub>${h - 1}</sub>: column of u<sub>1</sub>: 0.5·${a} + 0.2·${b} = <b>${c}</b>, 0.3·${a} + 0.4·${b} = <b>${d}</b>; column of u<sub>2</sub>: 0.5·${e} + 0.2·${f} = <b>${g}</b>, 0.3·${e} + 0.4·${f} = <b>${k}</b>.`,
 wos4: (h, M) => `Θ<sub>${h}</sub> = ${M} (row: variable; column: shock).`,
 wo1: (o, a, b, c, d) => `Order ${o === 'y2' ? '(y₂, y₁)' : '(y₁, y₂)'}: y<sub>2</sub> to u<sub>1</sub>: ${a} on impact, ${b} one period later; y<sub>1</sub> to u<sub>2</sub>: ${c} on impact, ${d} one period later.`,
 ir1: (o, p, pp, i0, m, hm, i12, ns) => `Ordering ${o}, VAR(${p}): an inflation shock (${pp} pp on impact) raises ROBOR by ${i0} pp on impact and up to ${m} pp after ${hm} quarters; ${i12} pp after 12` + (ns === null ? '.' : `; the band excludes zero at ${ns} of the 13 horizons.`),
 ir2: (g1, pm, hp, p4, p12) => `A ROBOR shock changes GDP growth by ${g1} pp in the next quarter; inflation after a ROBOR shock: largest ${pm} pp (at ${hp} quarters). Inflation after its own shock: ${p4} pp after 4 quarters, ${p12} after 12.`,
 ir3: p => `The bootstrap bands are computed for p = 2 (with the draws of the chapter’s seed); for p = ${p} only the responses are shown.`,
 irCap: (r, s) => `Response of ${r} to a one-standard-deviation ${s} shock; bold: the 90% band excludes zero.`,
 irs1: (o, P) => `Ordering ${o}: P = chol(Σ̂) = ${P} (rows and columns in the estimation order).`,
 irs2: (h, r, s, t, v) => `θ<sub>${r}${s},${h}</sub> = Σ<sub>m</sub> φ<sub>${r}m,${h}</sub> p<sub>m${s}</sub> = ${t.join(' + ')} = <b>${v}</b>.`,
 irs3: (o, j, s) => `The shock of ${s} is in position ${j + 1} of the order (${o.join(', ')}): the variables before it do not react to it in the quarter of the shock.`,
 io1: (p, e, a, b, g) => `VAR(${p}), up to ${e}: peak response of ROBOR to an inflation shock ${a} pp (inflation before ROBOR), ${b} pp (ROBOR first), ${g} pp (generalised).`,
 io2: (a, b, i, c) => `Inflation after a ROBOR shock: peak ${a} pp, against ${b} pp with ROBOR first (${i} pp already on impact); correlation of the π and i residuals: ${c}.`,
 ios1: S => `Σ̂ in the order (g, π, i) = ${S}.`,
 ios2: (s, p31, p21, p22, p32) => `Inflation before ROBOR: p<sub>iπ</sub> = (σ<sub>iπ</sub> − p<sub>ig</sub>p<sub>πg</sub>)/p<sub>ππ</sub> = (${s} − ${p31}·${p21})/${p22} = <b>${p32}</b>; with ROBOR before inflation, the impact response of ROBOR to the inflation shock is 0.`,
 ios3: (s, v, g) => `Generalised: ψ<sub>π</sub>(0) for ROBOR = σ<sub>iπ</sub>/√σ<sub>ππ</sub> = ${s}/√${v} = <b>${g}</b>, with no ordering.`,
 ios4: (a, g) => `The difference ${a} against ${g}: Cholesky removes from the covariance the part explained by the GDP shock (placed first); the generalised response keeps all of it.`,
 mi1: (p, y, s, d0, d1, b0, b1) => `VAR(${p}) from ${y}, S&P 500 shock of one standard deviation (${s}%): generalised DAX ${d0}% on the same day and ${d1}% the next day; BET ${b0}% and ${b1}%.`,
 mi2: (d, b, c) => `With the S&P 500 last: zero on the same day, then DAX ${d}% and BET ${b}%; correlation of the S&P 500 and DAX residuals ${c}.`,
 mis1: (a, b, c) => `From Σ̂: σ<sub>S&P,S&P</sub> = ${a}, σ<sub>DAX,S&P</sub> = ${b}, σ<sub>BET,S&P</sub> = ${c}.`,
 mis2: (b, s, d, c, e) => `Generalised impact: DAX ${b}/${s} = <b>${d}</b>; BET ${c}/${s} = <b>${e}</b> (√σ<sub>S&P,S&P</sub> = ${s}).`,
 mis3: (r, s, v) => `Equivalently: ρ<sub>DAX,S&P</sub>·√σ<sub>DAX</sub> = ${r}·${s} = ${v}; with the S&P 500 first, Cholesky gives exactly the same responses.`,
 wfs1: (h, j, l) => `Responses of y<sub>${j}</sub> for s = 0, …, ${h - 1}: ${l.join('; ')}.`,
 wfs2: (j, a, sa, b, sb, m) => `Sums of squares: u<sub>1</sub>: ${a.join(' + ')} = ${sa}; u<sub>2</sub>: ${b.join(' + ')} = ${sb}; error variance of y<sub>${j}</sub>: ${m}.`,
 wfs3: (j, h, a, m, s1, s2) => `ω<sub>${j}1</sub>(${h}) = ${a}/${m} = <b>${s1}%</b>; ω<sub>${j}2</sub>(${h}) = <b>${s2}%</b>; the row sums to 100%.`,
 wf1: (j, h, m, s, s10) => `y<sub>${j}</sub>, h = ${h}: error variance ${m}; share of u<sub>1</sub> ${s}% (at h = 10: ${s10}%).`,
 fv1: (o, p, a, b, c, d) => `Ordering ${o}, VAR(${p}): ROBOR, own shocks ${a}% at one quarter and ${b}% at 12; at 12 quarters inflation shocks explain ${c}%, GDP shocks ${d}%.`,
 fv2: (a, b) => `At 12 quarters: inflation ${a}% from its own shocks, GDP growth ${b}%.`,
 fvCap: o => `Shares (%) of the orthogonal shocks (columns) in the forecast error variance, ordering ${o}; bold: the horizon of the calculation.`,
 fvs1: (h, l, more, s) => `Responses of ROBOR to the inflation shock, s = 0, 1, …: ${l.join('; ')}${more ? '; …' : ''}; Σ<sub>s&lt;${h}</sub> θ<sub>iπ,s</sub>² = <b>${s}</b>.`,
 fvs2: (h, d) => `The ${h}-step forecast error variance of ROBOR, the sum of the squared responses to all three shocks: <b>${d}</b>.`,
 fvs3: (h, a, b, s) => `ω<sub>iπ</sub>(${h}) = ${a}/${b} = <b>${s}%</b>.`,
 stCap: (H, p, y, n) => `Generalised FEVD at H = ${H} days, VAR(${p}), ${y ? 'the year ' + y : '2000–2026'} (T = ${n}); row j: shares of the forecast error variance of market j.`,
 sts1: (H, p) => `Each row of the generalised FEVD (H = ${H}, VAR(${p})) is rescaled to sum to 100%; the diagonal (bold) is the part of own shocks.`,
 sts2: (f, s, S) => `“From others” by row: ${f.join(' + ')} = ${s}; divided by K = 3: the index S = <b>${S}%</b>.`,
 sts3: rows => rows.map(([m, t, f, n]) => `${m}: to others ${t}, from others ${f}, net ${n}`).join('; ') + '.',
 st1: (S, a, b, f, t) => `Index ${S}%; the S&P 500 receives ${a}% from the DAX, the DAX ${b}% from the S&P 500; the BET receives ${f}% and transmits ${t}%.`,
 sp1: (W, H, st, m, x, dx, mn, dmn, l, f) => `${W}-day windows, H = ${H}${st > 1 ? `, computed every ${st} days` : ''}: mean ${m}%, maximum ${x}% (${dx}), minimum ${mn}% (${dmn}), last value ${l}%; full sample ${f}%.`,
 sp2: (a, b, c) => `Mean in 2007: ${a}%; 2008Q4: ${b}%; March–June 2020: ${c}%.`,
 spCap: (W, a, b, H, S) => `Spillover table of the window ${a} – ${b} (${W} days, H = ${H}): index ${S}%.`,
 fc1: (o, p, l, a, b, lo, hi) => `Origin ${o}, VAR(${p}): inflation from ${l}% to ${a}% after one year and ${b}% after two (95%: [${lo}; ${hi}]).`,
 fc2: (i4, i8, g, lo, hi, mp, mi) => `ROBOR: ${i4}% after one year, ${i8}% after two; GDP growth after one year: ${g}% per quarter [${lo}; ${hi}]; mean of the VAR: inflation ${mp}%, ROBOR ${mi}%.`,
 fc3: (qq, a, f, ins) => `Realised in ${qq}: inflation ${a}%, against the forecast ${f}%; ${ins ? 'inside' : '<b>outside</b>'} the 95% interval.`,
 fcCap: (p, o) => `VAR(${p}) forecasts from ${o}, with 95% intervals in brackets (parameter uncertainty not included).`,
 fcs1: (s, o, a, b) => `Last observations: Y<sub>T</sub> = ${a} (${o})` + (b ? `, Y<sub>T−1</sub> = ${b}` : '') + ` in the order (g, π, i).`,
 fcs2: (s, e, more, v) => `Equation of ${s}: Ŷ = ${e}${more ? ' + …' : ''} = <b>${v}</b>.`,
 fcs3: (s, v1, pt, hw, vH, H) => `σ<sub>${s}${s}</sub>(1) = ${v1}, interval ${pt} ± ${hw}; at ${H} steps the error variance grows to ${vH}.`,
 ooCap: (c, h, f) => `${c === 'ro' ? 'Romania' : 'United States'}, h = ${h}, targets from ${f}; RMSE relative to the random walk (bold: the more accurate of VAR and AR); DM: Diebold–Mariano test of VAR against AR.`,
 oo1: (v, a, z, p, n) => `Romania, h = 1, GDP growth: VAR ${v}, AR ${a} (RMSE relative to the random walk); DM = ${z}, ${p} (${n} forecasts).`,
 oo2: (uv, ua, rv, ra, pv, pa) => `US, h = 8: unemployment VAR ${uv}, AR ${ua}; fed funds rate VAR ${rv}, AR ${ra}; inflation VAR ${pv}, AR ${pa}.`,
 us1: (a, b, n, mp, mu, mr, c1, c2) => `${a} – ${b} (${n} quarters): mean inflation ${mp}%, unemployment ${mu}%, fed funds rate ${mr}%; correlation inflation–rate ${c1}, unemployment–rate ${c2}.`,
 usCap: (a, b, n) => `Statistics for ${a} – ${b} (${n} quarters).`,
 sw1: (e, p, n, r0, um, hu, pm, hp, p12, u2) => `VAR(${p}), 1960Q1 – ${e} (T = ${n}): a monetary shock of ${r0} pp raises unemployment by up to ${um} pp after ${hu} quarters (${u2} pp after 2); inflation: peak ${pm} pp (at ${hp}), ${p12} pp after 12 quarters.`,
 sw2: (a, b, m, lec) => `FEVD at 12 quarters: monetary shocks explain ${a}% of unemployment and ${b}% of inflation; largest modulus ${m}` + (lec ? '; the 90% bands are those of the lecture.' : '; bootstrap bands are shown for the lecture’s specification only.'),
 swCap: (p, e, n) => `Granger F tests in the VAR(${p}), 1960Q1 – ${e} (T = ${n}); bold: p &lt; 0.05.`,
 swNote: rows => 'FEVD at 12 quarters (inflation / unemployment / rate shocks, %): ' + rows.map(([v, a, b, c]) => `${v}: ${a} / ${b} / ${c}`).join('; ') + '.',
};
})();
""",
}

QA = {
 'ro': {
  'ro-macro': ('Mutați începutul ferestrei în 2015. Cît devine corelația dintre inflație și ROBOR 3M?',
               '0,89, față de 0,66 pe 2005–2026. După 2015 cele două serii au urcat și au coborît împreună (inflația a atins 13,6% în noiembrie 2022, ROBOR 3M 8,0% în octombrie 2022), '
               'iar episodul din 2008, cînd ROBOR a ajuns la 18,2% în criza de lichiditate, a ieșit din fereastră. Corelația dintre nivelurile a două serii persistente depinde mult de eșantion '
               'și nu spune care dintre ele o precedă pe cealaltă.'),
  'ccf': ('Porniți randamentele zilnice în 2015. Este corelația dintre DAX de ieri și S&P 500 de azi (k = −1) în interiorul benzii?',
          'Nu: −0,058, în afara benzii ±0,037; pe tot eșantionul este −0,046, tot în afara benzii ±0,024. Precedența merge în principal de la New York spre Europa (0,135 la k = 1 din 2015, '
          '0,171 pe tot eșantionul), dar nu doar într-un sens: și DAX de ieri are o corelație mică, negativă, cu S&P 500 de azi. Pentru BET valoarea la k = −1 (−0,005 din 2015, −0,017 pe tot eșantionul) '
          'rămîne în bandă.'),
  'wx-var1': ('Setați a₁₂ = 0,5 și a₂₁ = 0,8. Toți coeficienții lui A sînt sub 1. Este VAR-ul stabil?',
              'Nu. tr(A) = 0,9 și det(A) = 0,5·0,4 − 0,5·0,8 = −0,20, deci λ = (0,9 ± √1,61)/2: λ₁ = 1,084 și λ₂ = −0,184. Cel mai mare modul depășește 1, iar prognozele cresc fără limită '
              '(ŷ₁ = 16,48 la h = 8). Stabilitatea depinde de valorile proprii, nu de mărimea fiecărui coeficient: efectele încrucișate a₁₂a₂₁ = 0,40 se adaugă persistenței proprii. '
              'Cu a₁₁ = 0,5 și a₂₂ = 0,4, VAR-ul este stabil doar dacă a₁₂a₂₁ < 0,30.'),
  'var-sim': ('Duceți a₂₁ la 0,6 (λ₁ = 0,8). Cît este răspunsul lui y₁ după 5 perioade, Φ₁₁(5)?',
              '0,187, nu 0,8⁵ = 0,328. Φ₁₁(h) = 0,571·0,8ʰ + 0,429·0,1ʰ: componenta dominantă intră în e₁ cu ponderea 0,571, iar raportul Φ₁₁(h)/Φ₁₁(h−1) se apropie de 0,8 abia după cîteva perioade. '
              'La fel în exemplul din curs: Φ₁₁(5) = 0,6·0,7⁵ + 0,4·0,2⁵ = 0,101, deci după 5 perioade rămîne circa 0,10 din efectul inițial; 0,7⁵ ≈ 0,17 este factorul cu care scade componenta dominantă, '
              'nu răspunsul.'),
  'roots': ('Cu p = 5, ordinul ales de AIC, cît devine cel mai mare modul și cîte valori proprii are matricea companion?',
            '0,913, din 15 valori proprii (matricea companion este 15 × 15), față de 0,869 din 6 pentru p = 2; un șoc scade la jumătate după 7,7 trimestre, față de 5,0. Lagurile suplimentare adaugă '
            'valori proprii, iar unele cu modul mare (0,874, 0,836) apar în perechi complexe, adică oscilații pe care VAR(2) nu le are. Cu 48 de coeficienți pe circa 80 de trimestre, aceste oscilații '
            'pot fi zgomot de estimare.'),
  'ic': ('Terminați eșantionul în 2019 (p<sub>max</sub> = 6). Ce ordin alege HQ?',
         'p = 5, ca AIC; BIC alege tot p = 1. Eșantionul comun are doar T = 54 de trimestre, iar ln det Σ̃(5) = −0,446 este mult mai mic decît pentru p = 2 (0,970), pentru că un VAR(5) are 48 de coeficienți '
         'pe 54 de observații și aproape reproduce datele. Penalizarea HQ, 2 ln ln T = 2,767 pe unitatea lui pK²/T, nu compensează. Pe eșantioane scurte criteriile pot alege ordine mari: '
         'trebuie verificat cîte grade de libertate rămîn.'),
  'var-table': ('Terminați eșantionul în T4 2019. Rămîne semnificativ (|t| ≥ 2) coeficientul lui g<sub>t−1</sub> în ecuația ROBOR?',
                'Da: 0,21 cu t = 3,7, față de 0,14 cu t = 3,5 pe tot eșantionul; în plus devine semnificativ și g<sub>t−2</sub> (0,14, t = 2,3). Fără 2020–2026, R² al ecuației PIB crește de la 0,16 la 0,30, '
                'pentru că trimestrul pandemiei, cu reziduul de −9,1, nu mai este în eșantion. Legătura dintre creșterea PIB și dobînda din trimestrul următor nu vine din anii pandemiei.'),
  'resid': ('Adăugați dummy pentru T1 2009 și T2 2020. Mai respinge testul Jarque–Bera normalitatea reziduurilor?',
            'Da: statistica scade de la 191 la 144 (χ²(6), p < 0,001). Dummy-urile anulează reziduurile celor două trimestre, dar rămîn alte valori extreme: ROBOR +3,5 în T4 2008, inflația −3,7 în T3 2011, '
            'PIB −5,9 în T3 2010; kurtosisul ROBOR rămîne 7,9. Testul portmanteau iese chiar mai prost (Q₈ = 88,2, p = 0,002): eliminarea a două observații nu face reziduurile un zgomot alb Normal.'),
  'granger-wx': ('Treceți la p = 4. Cît devin F și gradele de libertate pentru creșterea PIB → ROBOR 3M?',
                 'F = 8,50 cu (4, 69) grade de libertate, p < 0,001; valoarea critică de 5% scade la 2,50. Se testează acum 4 restricții: RSS<sub>R</sub> − RSS<sub>U</sub> = 64,08 − 42,93 = 21,15 se împarte la 4, '
                 'iar RSS<sub>U</sub> la T − Kp − 1 = 82 − 12 − 1 = 69. Concluzia nu depinde de p: cu p = 1, F = 10,63 (p = 0,002).'),
  'granger-ro': ('Terminați eșantionul în T4 2019 (VAR cu trei variabile). Rămîne inflația → ROBOR 3M semnificativă la 5%?',
                 'Nu: F = 2,46, p = 0,096, față de 4,74 (p = 0,011) pe tot eșantionul. În schimb, inflația → creșterea PIB devine semnificativă (F = 6,27, p = 0,004). Fără anii 2021–2023, cînd inflația a urcat la '
                 '13,6% și ROBOR a crescut odată cu ea, legătura de la inflație la ROBOR nu se mai separă de zgomot pe 58 de trimestre. Rezultatele Granger depind de eșantion, nu doar de setul de variabile.'),
  'granger-mkt': ('Porniți eșantionul în 2010. Rămîne DAX → BET semnificativă la 5%?',
                  'Nu: F = 1,1, p = 0,369 (pe 2000–2026: F = 3,2, p = 0,012). S&P 500 → BET rămîne foarte puternică (F = 22,5), iar coeficientul S&P 500 de ieri în ecuația BET scade de la 0,218 la 0,183. '
                  'Cu 4 033 de zile testul are multă putere, deci p = 0,369 arată că efectul DAX → BET, dată fiind S&P 500, este mic după 2010; nu arată că este exact zero.'),
  'granger-sim': ('Setați efectul lui z asupra lui y la 0. Ce rată de respingere are VAR-ul fără z la T = 500?',
                  '5,6%, apropiată de nivelul nominal de 5% (între 4,5% și 5,6% la toate volumele), față de 100% cu efectul 0,8 și 100% cu 0,4. x pare o cauză a lui y doar pentru că x<sub>t−1</sub> este o măsură '
                  'zgomotoasă a lui z<sub>t−2</sub>, care îl determină pe y<sub>t</sub>. Contează și ordinul: cu p = 1, chiar VAR-ul cu z respinge în 76%–100% din cazuri, pentru că un singur lag al lui z nu conține '
                  'z<sub>t−2</sub>, iar x<sub>t−1</sub> îl înlocuiește.'),
  'wx-orth': ('Puneți y₂ primul în ordinea Cholesky. Cît este răspunsul lui y₂ la șocul u₁ la impact și după o perioadă?',
              '0 la impact și 0,260 după o perioadă (0,3·0,866 + 0,4·0), față de 0,500 și 0,500 în ordinea (y₁, y₂). Cu y₂ primul, întreaga corelație de 0,5 este atribuită șocului u₂, care îl mișcă pe y₁ '
              'cu 0,500 la impact; u₁ ajunge la y₂ doar prin a₂₁, cu o perioadă întîrziere. Același VAR și aceleași date dau răspunsuri diferite: ordinea este o ipoteză.'),
  'irf-ro': ('Puneți ROBOR primul (ordinea i, g, π). Cît este răspunsul ROBOR la șocul inflației la impact și după 4 trimestre?',
             '0 la impact, prin construcție, și 0,64 pp după 4 trimestre, față de 0,23 și 0,82 pp în ordinea din curs; banda exclude zero la 9 orizonturi din 13, față de 11. Cu ROBOR primul, partea comună '
             'a celor două șocuri (corelația reziduurilor 0,25) este atribuită ROBOR. Concluzia calitativă rămîne: ROBOR crește după un șoc al inflației și rămîne mai sus ani la rînd (0,38 pp după 12 trimestre).'),
  'irf-order': ('Treceți la p = 1. Mai crește inflația după un șoc ROBOR („enigma prețurilor”) cînd inflația este înaintea ROBOR?',
                'Nu: maximul răspunsului este 0,00 pp, la impact (zero prin ordine), iar apoi răspunsul este negativ; cu p = 2 urca la 0,16 pp. Cu ROBOR primul, impactul este 0,29 pp și în VAR(1), '
                'pentru că ordinea atribuie ROBOR partea comună a șocurilor. „Enigma” depinde aici și de ordine, și de numărul de laguri: un rezultat care se schimbă cu p și cu ordinea este fragil.'),
  'market-irf': ('Porniți eșantionul în 2015. Cît devine răspunsul generalizat al BET a doua zi după un șoc S&P 500?',
                 '0,11%, față de 0,24% pe 2000–2026; răspunsul din aceeași zi rămîne 0,32%. Pentru DAX răspunsul scade de la 0,85% la 0,68% în aceeași zi și de la 0,24% la 0,16% a doua zi. Partea din ziua '
                 'următoare, care vine din orele de închidere diferite, este după 2015 de circa două ori mai mică.'),
  'wx-fevd': ('Puneți y₂ primul. Ce pondere are u₁ în varianța erorii de prognoză la 10 pași a lui y₂?',
              '10,6%, față de 45,8% în ordinea (y₁, y₂); la un pas ponderea este 0%. Cu y₂ primul, u₁ ajunge la y₂ doar prin a₂₁ = 0,3, cu o perioadă întîrziere (0,260 la h = 1), iar suma pătratelor acestor '
              'răspunsuri este 0,182 din 1,724. Varianța erorii nu se schimbă (1,72 la h = 10 în ambele ordini); se schimbă doar împărțirea ei între șocuri.'),
  'fevd-ro': ('Alegeți ordinea i, π, g (ROBOR primul). Cît explică șocurile inflației din varianța erorii de prognoză a ROBOR la 12 trimestre?',
              '27%, față de 53% în ordinea din curs; propriile șocuri ale ROBOR explică 63% în loc de 37%. La un trimestru ROBOR este explicat 100% de propriul șoc, prin construcție. Concluzia că pe termen lung '
              'dobînda este determinată mai ales de inflație se bazează deci pe ipoteza că ROBOR reacționează la inflație în același trimestru, nu invers.'),
  'spill-table': ('Alegeți anul 2020. Ce pondere din varianța erorii de prognoză a BET provine de la celelalte piețe?',
                  '49,0%, față de 14,7% pe 2000–2026; indicele urcă la 51,2% (26,4% pe tot eșantionul), iar BET transmite 45,3% în loc de 9,1%. În anul pandemiei piețele s-au mișcat împreună. Un singur an '
                  'are doar 237 de zile pentru 39 de coeficienți ai VAR(4), deci tabelul unui an este mai zgomotos; în 2017, un an calm, indicele este 24,4%, iar BET primește 13,2%.'),
  'spillover': ('Măriți fereastra la 500 de zile. Cît devine maximul indicelui și cînd apare?',
                '47,3%, pe 22 septembrie 2021, față de 57,5% pe 17 martie 2020 cu ferestre de 200 de zile. O fereastră mai lungă amestecă prăbușirea din martie 2020 cu mai multe zile calme, deci vîrful este '
                'mai mic și apare mai tîrziu; media rămîne aproape aceeași (29,1% față de 29,0%). Mărimea și data vîrfurilor depind de lungimea ferestrei: o parte din salturi este mecanică.'),
  'forecast-ro': ('Mutați originea în T4 2020. Este inflația realizată în T4 2022 în interiorul intervalului de 95% al prognozei pe 8 trimestre?',
                  'Nu: 13,2% realizat, față de o prognoză de 2,4% și intervalul [−2,3; 7,1]. VAR-ul estimat pînă în 2020 nu conține șocurile prețurilor energiei din 2021–2022, iar intervalul presupune șocuri Normale '
                  'cu varianța din trecut și ignoră incertitudinea parametrilor. Cu originea T4 2019, inflația din T4 2021 (6,4%) era încă în interval ([−2,0; 7,7]).'),
  'oos': ('Începeți evaluarea României în 2021 (h = 1). Este VAR-ul tot semnificativ mai puțin precis decît AR pentru creșterea PIB?',
          'Nu la 5%: RMSE relativ 1,06 pentru VAR și 0,76 pentru AR, dar DM = 1,74, p = 0,082, pe doar 22 de prognoze. Diferența rămîne în favoarea AR, iar VAR-ul este acum mai puțin precis chiar decît '
          'mersul aleator; cu jumătate din prognoze, testul nu o mai separă de zgomot la 5%.'),
  'us-data': ('Extindeți fereastra pînă în 2026. Cît devine media dobînzii federale?',
              '4,78%, față de 6,59% în 1960–2000; corelația șomaj–dobîndă scade de la 0,29 la 0,05. După 2008 și în 2020 dobînda a stat aproape de zero (minimul 0,06%), iar inflația a coborît pînă la −1,43%. '
              'Eșantionul extins conține un regim pe care VAR-ul din 1960–2000 nu l-a văzut.'),
  'sw-irf': ('Extindeți eșantionul pînă în T2 2026. Ce semn are răspunsul șomajului la un șoc monetar după 2 trimestre?',
             'Negativ: −0,13 pp, față de +0,05 pp în 1960–2000 (unde maximul este 0,16 pp după 8 trimestre). Șocul monetar explică acum doar 4% din varianța șomajului la 12 trimestre, față de 19%. '
             'Dobînzile aproape zero de după 2008 și pandemia schimbă sistemul, iar un VAR cu parametri constanți pe 1960–2026 amestecă regimuri diferite: semnul greșit semnalează instabilitatea modelului.'),
 },
 'en': {
  'ro-macro': ('Move the start of the window to 2015. What does the correlation between inflation and ROBOR 3M become?',
               '0.89, against 0.66 over 2005–2026. After 2015 the two series rose and fell together (inflation reached 13.6% in November 2022, ROBOR 3M 8.0% in October 2022), and the 2008 episode, '
               'when ROBOR reached 18.2% in the liquidity crisis, has left the window. The correlation between the levels of two persistent series depends strongly on the sample and does not say '
               'which of them leads the other.'),
  'ccf': ('Start the daily returns in 2015. Is the correlation between yesterday’s DAX and today’s S&P 500 (k = −1) inside the band?',
          'No: −0.058, outside the band ±0.037; over the whole sample it is −0.046, also outside the band ±0.024. The lead runs mainly from New York to Europe (0.135 at k = 1 since 2015, 0.171 over '
          'the whole sample), but not in one direction only: yesterday’s DAX also has a small negative correlation with today’s S&P 500. For the BET the value at k = −1 (−0.005 since 2015, −0.017 over '
          'the whole sample) stays inside the band.'),
  'wx-var1': ('Set a₁₂ = 0.5 and a₂₁ = 0.8. All coefficients of A are below 1. Is the VAR stable?',
              'No. tr(A) = 0.9 and det(A) = 0.5·0.4 − 0.5·0.8 = −0.20, so λ = (0.9 ± √1.61)/2: λ₁ = 1.084 and λ₂ = −0.184. The largest modulus exceeds 1 and the forecasts grow without bound '
              '(ŷ₁ = 16.48 at h = 8). Stability depends on the eigenvalues, not on the size of each coefficient: the cross effects a₁₂a₂₁ = 0.40 add to the own persistence. '
              'With a₁₁ = 0.5 and a₂₂ = 0.4, the VAR is stable only if a₁₂a₂₁ < 0.30.'),
  'var-sim': ('Move a₂₁ to 0.6 (λ₁ = 0.8). What is the response of y₁ after 5 periods, Φ₁₁(5)?',
              '0.187, not 0.8⁵ = 0.328. Φ₁₁(h) = 0.571·0.8ʰ + 0.429·0.1ʰ: the dominant component enters e₁ with weight 0.571, and the ratio Φ₁₁(h)/Φ₁₁(h−1) approaches 0.8 only after a few periods. '
              'The same holds in the lecture’s example: Φ₁₁(5) = 0.6·0.7⁵ + 0.4·0.2⁵ = 0.101, so after 5 periods about 0.10 of the initial effect is left; 0.7⁵ ≈ 0.17 is the factor by which the dominant '
              'component shrinks, not the response.'),
  'roots': ('With p = 5, the order chosen by AIC, what is the largest modulus and how many eigenvalues does the companion matrix have?',
            '0.913, out of 15 eigenvalues (the companion matrix is 15 × 15), against 0.869 out of 6 for p = 2; a shock halves after 7.7 quarters, against 5.0. Extra lags add eigenvalues, and some with '
            'a large modulus (0.874, 0.836) come in complex pairs, i.e. oscillations that the VAR(2) does not have. With 48 coefficients on about 80 quarters, these oscillations may be estimation noise.'),
  'ic': ('End the sample in 2019 (p<sub>max</sub> = 6). Which order does HQ choose?',
         'p = 5, like AIC; BIC still chooses p = 1. The common sample has only T = 54 quarters, and ln det Σ̃(5) = −0.446 is far below the value for p = 2 (0.970), because a VAR(5) has 48 coefficients '
         'for 54 observations and almost reproduces the data. The HQ penalty, 2 ln ln T = 2.767 per unit of pK²/T, does not offset it. In short samples the criteria can choose large orders: '
         'check how many degrees of freedom are left.'),
  'var-table': ('End the sample in 2019Q4. Does the coefficient of g<sub>t−1</sub> in the ROBOR equation stay significant (|t| ≥ 2)?',
                'Yes: 0.21 with t = 3.7, against 0.14 with t = 3.5 over the whole sample; in addition g<sub>t−2</sub> becomes significant (0.14, t = 2.3). Without 2020–2026 the R² of the GDP equation rises '
                'from 0.16 to 0.30, because the pandemic quarter, with its residual of −9.1, is no longer in the sample. The link between GDP growth and next quarter’s interest rate does not come from the pandemic years.'),
  'resid': ('Add dummies for 2009Q1 and 2020Q2. Does the Jarque–Bera test still reject normality of the residuals?',
            'Yes: the statistic falls from 191 to 144 (χ²(6), p < 0.001). The dummies cancel the residuals of the two quarters, but other extreme values remain: ROBOR +3.5 in 2008Q4, inflation −3.7 in 2011Q3, '
            'GDP −5.9 in 2010Q3; the kurtosis of ROBOR stays 7.9. The portmanteau test even gets worse (Q₈ = 88.2, p = 0.002): removing two observations does not make the residuals Normal white noise.'),
  'granger-wx': ('Move to p = 4. What are F and its degrees of freedom for GDP growth → ROBOR 3M?',
                 'F = 8.50 with (4, 69) degrees of freedom, p < 0.001; the 5% critical value falls to 2.50. Four restrictions are now tested: RSS<sub>R</sub> − RSS<sub>U</sub> = 64.08 − 42.93 = 21.15 is divided by 4, '
                 'and RSS<sub>U</sub> by T − Kp − 1 = 82 − 12 − 1 = 69. The conclusion does not depend on p: with p = 1, F = 10.63 (p = 0.002).'),
  'granger-ro': ('End the sample in 2019Q4 (three-variable VAR). Does inflation → ROBOR 3M stay significant at 5%?',
                 'No: F = 2.46, p = 0.096, against 4.74 (p = 0.011) over the whole sample. On the other hand, inflation → GDP growth becomes significant (F = 6.27, p = 0.004). Without 2021–2023, when inflation '
                 'climbed to 13.6% and ROBOR rose with it, the link from inflation to ROBOR is no longer separated from noise on 58 quarters. Granger results depend on the sample, not only on the set of variables.'),
  'granger-mkt': ('Start the sample in 2010. Does DAX → BET stay significant at 5%?',
                  'No: F = 1.1, p = 0.369 (over 2000–2026: F = 3.2, p = 0.012). S&P 500 → BET stays very strong (F = 22.5), and the coefficient of yesterday’s S&P 500 in the BET equation falls from 0.218 to 0.183. '
                  'With 4,033 days the test has a lot of power, so p = 0.369 shows that the DAX → BET effect, given the S&P 500, is small after 2010; it does not show that it is exactly zero.'),
  'granger-sim': ('Set the effect of z on y to 0. What rejection rate does the VAR without z have at T = 500?',
                  '5.6%, close to the nominal 5% (between 4.5% and 5.6% for all sample sizes), against 100% with the effect 0.8 and 100% with 0.4. x looks like a cause of y only because x<sub>t−1</sub> is a noisy '
                  'measure of z<sub>t−2</sub>, which drives y<sub>t</sub>. The lag order matters too: with p = 1 even the VAR with z rejects in 76%–100% of the cases, because one lag of z does not contain '
                  'z<sub>t−2</sub>, and x<sub>t−1</sub> stands in for it.'),
  'wx-orth': ('Put y₂ first in the Cholesky ordering. What is the response of y₂ to the shock u₁ on impact and one period later?',
              '0 on impact and 0.260 one period later (0.3·0.866 + 0.4·0), against 0.500 and 0.500 in the order (y₁, y₂). With y₂ first, the whole correlation of 0.5 is assigned to the shock u₂, which moves y₁ '
              'by 0.500 on impact; u₁ reaches y₂ only through a₂₁, one period later. The same VAR and the same data give different responses: the ordering is an assumption.'),
  'irf-ro': ('Put ROBOR first (order i, g, π). What is the response of ROBOR to the inflation shock on impact and after 4 quarters?',
             '0 on impact, by construction, and 0.64 pp after 4 quarters, against 0.23 and 0.82 pp in the lecture’s ordering; the band excludes zero at 9 of the 13 horizons, against 11. With ROBOR first, the '
             'common part of the two shocks (residual correlation 0.25) is assigned to ROBOR. The qualitative conclusion stays: ROBOR rises after an inflation shock and stays higher for years (0.38 pp after 12 quarters).'),
  'irf-order': ('Move to p = 1. Does inflation still rise after a ROBOR shock (the “price puzzle”) when inflation comes before ROBOR?',
                'No: the largest response is 0.00 pp, on impact (zero by the ordering), and the response is negative afterwards; with p = 2 it rose to 0.16 pp. With ROBOR first, the impact is 0.29 pp in the '
                'VAR(1) as well, because the ordering assigns the common part of the shocks to ROBOR. The “puzzle” depends here on both the ordering and the number of lags: a result that changes with p and '
                'with the ordering is fragile.'),
  'market-irf': ('Start the sample in 2015. What does the generalised response of the BET the day after an S&P 500 shock become?',
                 '0.11%, against 0.24% over 2000–2026; the same-day response stays 0.32%. For the DAX the response falls from 0.85% to 0.68% on the same day and from 0.24% to 0.16% the next day. The next-day part, '
                 'which comes from the different closing times, is about half as large after 2015.'),
  'wx-fevd': ('Put y₂ first. What share of the 10-step forecast error variance of y₂ is due to u₁?',
              '10.6%, against 45.8% in the order (y₁, y₂); at one step the share is 0%. With y₂ first, u₁ reaches y₂ only through a₂₁ = 0.3, one period later (0.260 at h = 1), and the sum of the squares of these '
              'responses is 0.182 out of 1.724. The error variance does not change (1.72 at h = 10 in both orders); only its split between the shocks does.'),
  'fevd-ro': ('Choose the order i, π, g (ROBOR first). How much of the 12-quarter forecast error variance of ROBOR do inflation shocks explain?',
              '27%, against 53% in the lecture’s ordering; ROBOR’s own shocks explain 63% instead of 37%. At one quarter ROBOR is explained 100% by its own shock, by construction. The conclusion that in the long run '
              'the interest rate is driven mainly by inflation therefore rests on the assumption that ROBOR reacts to inflation within the quarter, not the other way round.'),
  'spill-table': ('Choose the year 2020. What share of the forecast error variance of the BET comes from the other markets?',
                  '49.0%, against 14.7% over 2000–2026; the index rises to 51.2% (26.4% over the whole sample), and the BET transmits 45.3% instead of 9.1%. In the pandemic year the markets moved together. A single '
                  'year has only 237 days for the 39 coefficients of the VAR(4), so the table of one year is noisier; in 2017, a calm year, the index is 24.4% and the BET receives 13.2%.'),
  'spillover': ('Widen the window to 500 days. What does the maximum of the index become, and when does it occur?',
                '47.3%, on 22 September 2021, against 57.5% on 17 March 2020 with 200-day windows. A longer window mixes the crash of March 2020 with more calm days, so the peak is lower and comes later; '
                'the mean stays almost the same (29.1% against 29.0%). The size and date of the peaks depend on the window length: part of the jumps is mechanical.'),
  'forecast-ro': ('Move the origin to 2020Q4. Is the inflation realised in 2022Q4 inside the 95% interval of the 8-quarter forecast?',
                  'No: 13.2% realised, against a forecast of 2.4% and the interval [−2.3; 7.1]. The VAR estimated up to 2020 contains none of the energy-price shocks of 2021–2022, and the interval assumes Normal shocks '
                  'with the past variance and ignores parameter uncertainty. With the origin 2019Q4, the inflation of 2021Q4 (6.4%) was still inside the interval ([−2.0; 7.7]).'),
  'oos': ('Start the evaluation for Romania in 2021 (h = 1). Is the VAR still significantly less accurate than the AR for GDP growth?',
          'Not at 5%: relative RMSE 1.06 for the VAR and 0.76 for the AR, but DM = 1.74, p = 0.082, on only 22 forecasts. The difference still favours the AR, and the VAR is now less accurate than even the '
          'random walk; with half the forecasts, the test no longer separates it from noise at 5%.'),
  'us-data': ('Extend the window to 2026. What does the mean federal funds rate become?',
              '4.78%, against 6.59% in 1960–2000; the correlation of unemployment and the rate falls from 0.29 to 0.05. After 2008 and in 2020 the rate stayed close to zero (minimum 0.06%), and inflation went '
              'down to −1.43%. The extended sample contains a regime that the 1960–2000 VAR never saw.'),
  'sw-irf': ('Extend the sample to 2026Q2. What sign does the response of unemployment to a monetary shock have after 2 quarters?',
             'Negative: −0.13 pp, against +0.05 pp in 1960–2000 (where the peak is 0.16 pp after 8 quarters). The monetary shock now explains only 4% of the 12-quarter variance of unemployment, against 19%. '
             'The near-zero rates after 2008 and the pandemic change the system, and a VAR with constant parameters on 1960–2026 mixes different regimes: the wrong sign signals that the model is unstable.'),
 },
}

# Predict, then check: options for the question of each block, (RO, EN) pairs; key = index of the correct option.
# The correct option agrees with the readout of the block at the state the question names (verified by a node script).
PREDICT = {
 'ro-macro': (0, [('0,89: mai mare decît 0,66.', '0.89: higher than 0.66.'), ('Tot 0,66: corelația este o proprietate a seriilor, nu a ferestrei.', 'Still 0.66: the correlation is a property of the series, not of the window.'),
                  ('Sub 0,3: după 2015 dobînda nu a mai urmat inflația.', 'Below 0.3: after 2015 the rate no longer followed inflation.')]),
 'ccf': (1, [('Da: piețele europene nu pot preceda New York-ul.', 'Yes: European markets cannot lead New York.'), ('Nu: −0,058, în afara benzii ±0,037.', 'No: −0.058, outside the band ±0.037.'),
             ('Nu se poate spune: banda ±1,96/√T nu se aplică randamentelor zilnice.', 'Cannot tell: the band ±1.96/√T does not apply to daily returns.')]),
 'wx-var1': (1, [('Da: un VAR cu toți coeficienții sub 1 este stabil.', 'Yes: a VAR with all coefficients below 1 is stable.'), ('Nu: λ₁ = 1,084.', 'No: λ₁ = 1.084.'),
                 ('Da, dar converge mai încet: λ₁ = 0,95.', 'Yes, but it converges more slowly: λ₁ = 0.95.')]),
 'var-sim': (2, [('0,328 = 0,8⁵: răspunsul scade cu cea mai mare valoare proprie.', '0.328 = 0.8⁵: the response shrinks with the largest eigenvalue.'), ('0,031 = 0,5⁵: contează doar a₁₁.', '0.031 = 0.5⁵: only a₁₁ matters.'),
                 ('0,187 = 0,571·0,8⁵ + 0,429·0,1⁵.', '0.187 = 0.571·0.8⁵ + 0.429·0.1⁵.')]),
 'roots': (1, [('0,869: lagurile suplimentare nu schimbă cea mai mare valoare proprie.', '0.869: extra lags do not change the largest eigenvalue.'), ('0,913, din 15 valori proprii.', '0.913, out of 15 eigenvalues.'),
               ('Peste 1: un VAR(5) pe 80 de trimestre este nestaționar.', 'Above 1: a VAR(5) on 80 quarters is non-stationary.')]),
 'ic': (1, [('p = 2, ca pe tot eșantionul: HQ este consistent.', 'p = 2, as on the whole sample: HQ is consistent.'), ('p = 5: cu T = 54, ajustarea cîștigă în fața penalizării.', 'p = 5: with T = 54, the fit beats the penalty.'),
            ('p = 0: pe eșantioane scurte criteriile aleg modelul fără laguri.', 'p = 0: in short samples the criteria choose the model without lags.')]),
 'var-table': (1, [('Nu: fără pandemie dispare legătura dintre PIB și dobîndă.', 'No: without the pandemic the link between GDP and the rate disappears.'), ('Da: 0,21, cu t = 3,7.', 'Yes: 0.21, with t = 3.7.'),
                   ('Nu se poate estima: 21 de coeficienți pe 58 de trimestre sînt prea mulți.', 'It cannot be estimated: 21 coefficients on 58 quarters are too many.')]),
 'resid': (1, [('Nu: fără cele două crize reziduurile sînt Normale.', 'No: without the two crises the residuals are Normal.'), ('Da: 144, p < 0,001.', 'Yes: 144, p < 0.001.'),
               ('Testul nu se mai poate calcula cu variabile dummy.', 'The test can no longer be computed with dummies.')]),
 'granger-wx': (2, [('Rămîne 7,26: F nu depinde de numărul de laguri.', 'It stays 7.26: F does not depend on the number of lags.'),
                    ('Scade sub valoarea critică: mai multe laguri diluează efectul.', 'It falls below the critical value: more lags dilute the effect.'), ('F = 8,50, F(4, 69), p < 0,001.', 'F = 8.50, F(4, 69), p < 0.001.')]),
 'granger-ro': (1, [('Da: o relație Granger este o proprietate a economiei, nu a eșantionului.', 'Yes: a Granger relation is a property of the economy, not of the sample.'), ('Nu: F = 2,46, p = 0,096.', 'No: F = 2.46, p = 0.096.'),
                    ('Da, și devine mai puternică fără anii de inflație ridicată.', 'Yes, and it gets stronger without the high-inflation years.')]),
 'granger-mkt': (1, [('Da: cu mii de zile, toate legăturile sînt semnificative.', 'Yes: with thousands of days, every link is significant.'), ('Nu: F = 1,1, p = 0,369.', 'No: F = 1.1, p = 0.369.'),
                     ('Nu, și nici S&P 500 → BET nu mai este semnificativă.', 'No, and S&P 500 → BET is no longer significant either.')]),
 'granger-sim': (1, [('Tot aproape 100%: lipsa lui z produce mereu cauzalitate falsă.', 'Still close to 100%: leaving z out always creates false causality.'), ('Aproximativ 5%: 5,6%.', 'About 5%: 5.6%.'),
                     ('0%: fără efect, testul nu mai respinge niciodată.', '0%: without the effect the test never rejects.')]),
 'wx-orth': (1, [('0,500 și 0,500: ordinea nu schimbă răspunsurile.', '0.500 and 0.500: the ordering does not change the responses.'), ('0 și 0,260.', '0 and 0.260.'),
                 ('0 și 0: y₂ nu mai reacționează deloc la u₁.', '0 and 0: y₂ no longer reacts to u₁ at all.')]),
 'irf-ro': (1, [('0,23 și 0,82 pp, ca în curs.', '0.23 and 0.82 pp, as in the lecture.'), ('0 și 0,64 pp.', '0 and 0.64 pp.'),
                ('Răspunsul devine negativ: dobînda scade după șocul inflației.', 'The response turns negative: the rate falls after the inflation shock.')]),
 'irf-order': (1, [('Da, cu același maxim de 0,16 pp.', 'Yes, with the same peak of 0.16 pp.'), ('Nu: maximul este 0,00 pp (impactul), apoi răspunsul este negativ.', 'No: the peak is 0.00 pp (the impact), then the response is negative.'),
                   ('Da, și mai mult: 0,53 pp.', 'Yes, and more: 0.53 pp.')]),
 'market-irf': (1, [('0,24%: răspunsurile nu depind de eșantion.', '0.24%: the responses do not depend on the sample.'), ('0,11%.', '0.11%.'),
                    ('0: după 2015 piețele se închid la aceeași oră.', '0: after 2015 the markets close at the same time.')]),
 'wx-fevd': (1, [('45,8%, ca în cealaltă ordine: FEVD nu depinde de ordine.', '45.8%, as in the other order: the FEVD does not depend on the ordering.'), ('10,6%.', '10.6%.'),
                 ('0% la orice orizont: u₁ nu îl atinge pe y₂.', '0% at every horizon: u₁ never reaches y₂.')]),
 'fevd-ro': (1, [('53%, ca în curs.', '53%, as in the lecture.'), ('27%.', '27%.'), ('0%: cu ROBOR primul, inflația nu îl mai poate influența.', '0%: with ROBOR first, inflation can no longer affect it.')]),
 'spill-table': (1, [('Tot aproximativ 15%: BET este o piață mică, receptoare.', 'Still about 15%: the BET is a small, receiving market.'), ('49,0%.', '49.0%.'),
                     ('100%: într-o criză toate piețele sînt determinate de New York.', '100%: in a crisis every market is driven by New York.')]),
 'spillover': (1, [('57,5%, în aceeași zi: fereastra nu schimbă vîrful.', '57.5%, on the same day: the window does not change the peak.'), ('47,3%, pe 22 septembrie 2021.', '47.3%, on 22 September 2021.'),
                   ('Peste 70%: o fereastră mai lungă acumulează mai multe crize.', 'Above 70%: a longer window accumulates more crises.')]),
 'forecast-ro': (1, [('Da: un interval de 95% acoperă aproape orice valoare.', 'Yes: a 95% interval covers almost any value.'), ('Nu: 13,2% față de [−2,3; 7,1].', 'No: 13.2% against [−2.3; 7.1].'),
                     ('Da, la limită: VAR-ul anticipa creșterea inflației.', 'Yes, just: the VAR anticipated the rise in inflation.')]),
 'oos': (1, [('Da: p rămîne 0,010.', 'Yes: p stays 0.010.'), ('Nu la 5%: DM = 1,74, p = 0,082.', 'Not at 5%: DM = 1.74, p = 0.082.'),
             ('Nu: după 2021 VAR-ul devine mai precis decît AR.', 'No: after 2021 the VAR becomes more accurate than the AR.')]),
 'us-data': (1, [('Aproximativ 6,6%: media pe termen lung este stabilă.', 'About 6.6%: the long-run mean is stable.'), ('4,78%.', '4.78%.'), ('Peste 8%: inflația din 2022 a ridicat dobînzile.', 'Above 8%: the 2022 inflation raised the rates.')]),
 'sw-irf': (1, [('Pozitiv, ca în 1960–2000.', 'Positive, as in 1960–2000.'), ('Negativ: −0,13 pp.', 'Negative: −0.13 pp.'), ('Zero: șomajul nu mai reacționează la dobîndă.', 'Zero: unemployment no longer reacts to the rate.')]),
}

# no initial chart answers its question (every question asks about a changed control)
VEIL_CHARTS = ()

# Exit ticket „Verifică-te” / "Check Yourself": the questions of the slide „Autoevaluare” / "Self-assessment" and their answers.
TICKET = [
 dict(q=('Este stabil VAR(1) cu A = [0,9 0,3; 0,2 0,8]?', 'Is the VAR(1) with A = [0.9 0.3; 0.2 0.8] stable?'), key=1, opts=[
  (('Da: toți coeficienții sînt sub 1.', 'Yes: all coefficients are below 1.'),
   ('Stabilitatea depinde de valorile proprii, nu de coeficienți luați separat: aici λ₁ = 1,1.', 'Stability depends on the eigenvalues, not on the coefficients one by one: here λ₁ = 1.1.')),
  (('Nu: λ² − 1,7λ + 0,66 = 0 dă λ = 1,1 și 0,6.', 'No: λ² − 1.7λ + 0.66 = 0 gives λ = 1.1 and 0.6.'),
   ('tr(A) = 1,7 și det(A) = 0,72 − 0,06 = 0,66; λ = (1,7 ± √(2,89 − 2,64))/2 = (1,7 ± 0,5)/2.', 'tr(A) = 1.7 and det(A) = 0.72 − 0.06 = 0.66; λ = (1.7 ± √(2.89 − 2.64))/2 = (1.7 ± 0.5)/2.')),
  (('Da: valorile proprii sînt 0,9 și 0,8, elementele diagonalei.', 'Yes: the eigenvalues are 0.9 and 0.8, the diagonal elements.'),
   ('Diagonala dă valorile proprii doar pentru o matrice triunghiulară; aici a₁₂ și a₂₁ nu sînt zero.', 'The diagonal gives the eigenvalues only for a triangular matrix; here a₁₂ and a₂₁ are not zero.'))]),
 dict(q=('Un test Granger dă p = 0,30 pentru x → y. Nu are x niciun efect asupra lui y?', 'A Granger test gives p = 0.30 for x → y. Does x have no effect on y?'), key=1, opts=[
  (('Da: x nu are niciun efect asupra lui y.', 'Yes: x has no effect on y.'),
   ('Un p mare spune doar că lagurile lui x nu se separă de zgomot ca ajutor la prognoză; nu exclude efecte.', 'A large p only says that the lags of x are not separated from noise as a help to the forecast; it does not rule out effects.')),
  (('Nu putem concluziona asta: rămîn posibile efecte în aceeași perioadă, efecte lente sau o putere mică a testului.', 'We cannot conclude that: effects within the period, slow effects or low power of the test remain possible.'),
   ('Testul privește doar lagurile, în acest set de variabile și pe acest eșantion.', 'The test looks only at the lags, in this set of variables and on this sample.')),
  (('Nu: înseamnă că y îl cauzează pe x.', 'No: it means that y causes x.'),
   ('Sensul invers se testează separat, în ecuația lui x.', 'The reverse direction is tested separately, in the equation of x.'))]),
 dict(q=('Într-o FEVD Cholesky, ce pondere din varianța erorii la un pas a primei variabile provine din propriul șoc?',
         'In a Cholesky FEVD, what share of the one-step error variance of the first variable comes from its own shock?'), key=0, opts=[
  (('100%, prin construcția ordinii.', '100%, by the construction of the ordering.'),
   ('Prima variabilă nu este mișcată de celelalte șocuri ortogonale în perioada șocului: θ₁ₖ,₀ = 0 pentru k > 1.', 'The first variable is not moved by the other orthogonal shocks in the period of the shock: θ₁ₖ,₀ = 0 for k > 1.')),
  (('Depinde de corelația șocurilor.', 'It depends on the correlation of the shocks.'),
   ('Corelația este atribuită în întregime variabilei așezate prima; ea schimbă ponderile celorlalte variabile, nu pe a primei.', 'The correlation is assigned entirely to the variable placed first; it changes the shares of the other variables, not of the first.')),
  (('1/K: fiecare șoc explică o parte egală.', '1/K: each shock explains an equal part.'),
   ('Nicio formulă a FEVD nu dă ponderi egale; la un pas, ordinea Cholesky fixează rîndul primei variabile la (100%, 0, …, 0).', 'No FEVD formula gives equal shares; at one step, the Cholesky ordering fixes the row of the first variable at (100%, 0, …, 0).'))]),
]


def _predict_js():
    import json
    for i, lang in enumerate(('ro', 'en')):
        pr = {a: dict(key=k, opts=[o[i] for o in opts], **({'veil': {'out': True, 'charts': True}} if a in VEIL_CHARTS else {}))
              for a, (k, opts) in PREDICT.items()}
        tk = dict(title='Verifică-te' if lang == 'ro' else 'Check Yourself', id='check', store=f'tsa-ch6-check-{lang}',
                  qs=[dict(q=t['q'][i], key=t['key'], opts=[dict(t=o[i], why=w[i]) for o, w in t['opts']]) for t in TICKET])
        TEXT_JS[lang] += ('window.CH6_TEXT.predict = ' + json.dumps(pr, ensure_ascii=False) + ';\n'
                          + 'window.CH6_TEXT.ticket = ' + json.dumps(tk, ensure_ascii=False) + ';\n')


_predict_js()

MATHJAX = ('<script>window.MathJax = { tex: { inlineMath: [[\'\\\\(\', \'\\\\)\']], displayMath: [[\'\\\\[\', \'\\\\]\']] }, '
           'startup: { pageReady: () => MathJax.startup.defaultPageReady().then(() => document.body.setAttribute(\'data-mj\', \'done\')) } };</script>\n'
           '<script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3.2.2/es5/tex-chtml.js"></script>\n')


def _extra_pages(lang):
    """Page and title of the frames of EXTRA (the result tables and worked examples), from the .tex and the beamer .nav file."""
    from interactive_common import ROOT, frames_of, latex_text
    tex = os.path.join(ROOT, TEX[lang])
    frames = frames_of(open(tex + '.tex', encoding='utf-8').read())
    pages = re.findall(r'\\beamer@framepages \{(\d+)\}\{(\d+)\}', open(tex + '.nav', encoding='utf-8').read())
    titles = [latex_text(t) for t, _ in frames]
    res = {}
    for anchor, tro, ten, _ in EXTRA:
        want = tro if lang == 'ro' else ten
        k = titles.index(want)
        res[anchor] = (int(pages[k][0]), titles[k])
    return res


def _post(lang):
    """Insert the EXTRA blocks in slide order (with their section headings), MathJax and render their slide images."""
    import pymupdf
    import interactive_common as ic
    import sys
    sp = sys.modules[__name__]
    t, Wd = L[lang], widgets(lang)
    path = os.path.join(ic.OUT, t['file'])
    html = open(path, encoding='utf-8').read()
    viewer = '../pdfjs/web/viewer.html?file=../../' + TEX[lang] + '.pdf'
    chart_pages = ic.slide_pages(sp, lang)
    blocks = {m.group(1): m.group(0) for m in re.finditer(r'\n<section class="blk" id="([^"]+)">.*?</section>', html, flags=re.S)}
    items = [(chart_pages[c][0], a, s, blocks[a]) for a, c, s in BLOCKS]
    xp = _extra_pages(lang)
    doc = pymupdf.open(os.path.join(ic.ROOT, TEX[lang] + '.pdf'))
    for anchor, _, _, skey in EXTRA:
        pg, title = xp[anchor]
        img = f'ch6x_{anchor}_{lang}.jpg'
        doc[pg - 1].get_pixmap(dpi=130).save(os.path.join(ic.OUT, 'slides', img), jpg_quality=82)
        ctl, ch = Wd[anchor]
        q, a = QA[lang][anchor]
        items.append((pg, anchor, skey, f'''
<section class="blk" id="{anchor}">
  <h3>{title}</h3>
  <div class="pair">
    <figure class="slide"><a href="{viewer}#page={pg}" target="_blank" rel="noopener"><img src="slides/{img}" alt="{t["slide"]} {pg - 1}: {title}" loading="lazy"></a>
      <figcaption>{t["slide"]} {pg - 1}</figcaption></figure>
    <div class="live"><div class="ctls">{ctl}</div>{ch}<ul class="out" id="{PFX[anchor]}-out"></ul></div>
  </div>
  <div class="ask"><span class="ask-tag">{t["ask"]}</span> {q}</div>
  <details><summary>{t["answer"]}</summary><p>{a}</p></details>
</section>'''))
    items.sort(key=lambda x: x[0])
    body, nav, sec = [], [], None
    for _, anchor, skey, h in items:
        if skey != sec:
            sec = skey
            body.append(f'<h2 class="sec" id="s-{skey}">{SECTIONS[lang][skey]}</h2>')
            nav.append(f'<a href="#s-{skey}">{SECTIONS[lang][skey]}</a>')
        body.append(h)
    i0, i1 = html.index('<p class="toc">'), html.index('<p class="src foot">')
    html = html[:i0] + f'<p class="toc">{" · ".join(nav)}</p>\n' + ''.join(body) + '\n' + html[i1:]
    html = html.replace('</head>', MATHJAX + '</head>', 1)
    open(path, 'w', encoding='utf-8').write(html)
    print('extra blocks', lang, len(EXTRA))


if __name__ == '__main__':
    import sys
    import interactive_common
    interactive_common.build(sys.modules[__name__])
    for lg in ('en', 'ro'):
        _post(lg)
