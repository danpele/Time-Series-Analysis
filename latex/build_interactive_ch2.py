"""Build the interactive pages of Chapter 2 (ARMA models): interactive/ch2.html (EN) and interactive/ch2_ro.html (RO).

One block per chart slide of the lecture body, in the order of the slides: the slide on the left, its
interactive version on the right. Logic in interactive/ch2.js, inputs in interactive/ch2_data.js
(Quantlets/Ch_02/export_interactive.py), slide images in interactive/slides/.
The worked examples and result tables of the lecture (AR(1), characteristic roots, θ and 1/θ, ψ weights, ARMA(1,1) moments,
the ACF-PACF table, Yule-Walker, maximum likelihood, information criteria, residual diagnostics, AR(1) forecasts, the GDP
model grid, the MA(3) estimates, the GDP forecasts, the AR(1) for daily returns) get blocks of their own (EXTRA), placed by
slide. On the web lecture (build_web_lecture.py) the EXTRA blocks travel inside a chart block of the same section and ch2.js
moves each one to its frame.
"""
import os
import re
import sys
import types

from interactive_common import slider, select, play, charts

N = 2

QL = 'https://github.com/danpele/Time-Series-Analysis/tree/main/Quantlets/Ch_02'
TEX = {'ro': 'RO/Cursuri/capitol2_modele_arma', 'en': 'EN/Courses/chapter2_arma_models'}

# (anchor, chart file, section key); order of the slides
BLOCKS = [('ar1', 'tsa_ch2_ar1', 'ar'), ('ar2', 'tsa_ch2_ar2', 'ar'),
          ('ma1', 'tsa_ch2_ma1', 'ma'), ('ma1-rho', 'tsa_ch2_ma1_rho', 'ma'),
          ('psi', 'tsa_ch2_psi', 'arma'), ('patterns', 'tsa_ch2_patterns', 'arma'),
          ('estimators', 'tsa_ch2_estimators', 'est'),
          ('ic', 'tsa_ch2_ic', 'sel'), ('lb-df', 'tsa_ch2_lb_df', 'sel'),
          ('forecast-mean', 'tsa_ch2_forecast_theory', 'fc'),
          ('gdp-ident', 'tsa_ch2_gdp_ident', 'gdp'), ('gdp-diag', 'tsa_ch2_gdp_diag', 'gdp'), ('gdp-forecast', 'tsa_ch2_gdp_forecast', 'gdp'),
          ('inflation', 'tsa_ch2_inflation', 'real'), ('returns', 'tsa_ch2_returns', 'real'), ('sunspots', 'tsa_ch2_sunspots', 'real')]

# blocks of the worked examples and result tables: (anchor, frame title RO, frame title EN, section key, host chart block on the web lecture)
EXTRA = [('wx-ar1', 'Exemplu rezolvat: un AR(1)', 'Worked example: an AR(1)', 'ar', 'ar1'),
         ('ar-roots', 'Modelul AR(p) și polinomul lui caracteristic', 'The AR(p) model and its characteristic polynomial', 'ar', 'ar2'),
         ('wx-theta', 'Exemplu rezolvat: θ = 2 sau θ = 0,5?', 'Worked example: θ = 2 or θ = 0.5?', 'ma', 'ma1-rho'),
         ('wx-psi', 'Ponderile ψ și răspunsul la impuls', 'The ψ weights and the impulse response', 'arma', 'psi'),
         ('wx-arma11', 'Momentele procesului ARMA(1,1)', 'Moments of the ARMA(1,1)', 'arma', 'psi'),
         ('ident-table', 'Identificarea: tabelul ACF–PACF', 'Identification: the ACF–PACF table', 'arma', 'patterns'),
         ('wx-yw', 'Exemplu rezolvat: Yule–Walker pentru creșterea PIB-ului României', 'Worked example: Yule–Walker on Romanian GDP growth', 'est', 'estimators'),
         ('ml-lik', 'Verosimilitatea maximă', 'Maximum likelihood', 'est', 'estimators'),
         ('wx-ic', 'Criterii informaționale', 'Information criteria', 'sel', 'ic'),
         ('diag-table', 'Diagnosticarea reziduurilor', 'Residual diagnostics', 'sel', 'lb-df'),
         ('wx-fc', 'Exemplu rezolvat: prognoza unui AR(1)', 'Worked example: forecasting an AR(1)', 'fc', 'forecast-mean'),
         ('gdp-grid', 'Pasul 2: estimarea și criteriile informaționale', 'Step 2: estimation and information criteria', 'gdp', 'gdp-diag'),
         ('ma3-table', 'Interpretarea modelului MA(3) estimat', 'Interpreting the estimated MA(3)', 'gdp', 'gdp-diag'),
         ('gdp-fc-table', 'Interpretarea prognozelor PIB', 'Interpreting the GDP forecasts', 'gdp', 'gdp-forecast'),
         ('ret-table', 'Interpretarea modelului AR(1) pentru randamentele zilnice', 'Interpreting the AR(1) for daily returns', 'real', 'returns')]

PFX = {'wx-ar1': 'xa', 'ar-roots': 'xr', 'wx-theta': 'xt', 'wx-psi': 'xp', 'wx-arma11': 'xm', 'ident-table': 'it', 'wx-yw': 'xy', 'ml-lik': 'xl',
       'wx-ic': 'xi', 'diag-table': 'dt', 'wx-fc': 'xf', 'gdp-grid': 'gg', 'ma3-table': 'mq', 'gdp-fc-table': 'ft', 'ret-table': 'rx',
       'ar1': 'a1', 'ar2': 'a2', 'ma1': 'm1', 'ma1-rho': 'mr', 'psi': 'ps', 'patterns': 'pt', 'estimators': 'es', 'ic': 'ic',
       'lb-df': 'lb', 'forecast-mean': 'fc', 'gdp-ident': 'gi', 'gdp-diag': 'gd', 'gdp-forecast': 'gf', 'inflation': 'in',
       'returns': 'rt', 'sunspots': 'ss'}

SECTIONS = {
 'ro': dict(ar='Modele autoregresive', ma='Modele de medie mobilă', arma='Modele ARMA(p,q)', est='Estimarea',
            sel='Selecția modelului și diagnosticarea', fc='Prognoza cu modele ARMA', gdp='Studiu de caz: creșterea PIB-ului României',
            real='Alte serii reale'),
 'en': dict(ar='Autoregressive models', ma='Moving-average models', arma='ARMA(p,q) models', est='Estimation',
            sel='Model selection and diagnostics', fc='Forecasting with ARMA models', gdp='Case study: Romanian GDP growth',
            real='More real series'),
}

L = {
 'ro': dict(file='ch2_ro.html', html_lang='ro', site='../index_ro.html#chapters', course='Serii de timp',
   h1='Capitolul 2 · Modele ARMA', back='Site-ul cursului', pdf_btn='PDF-ul cursului', ql='Quantlets',
   slide='Slide-ul', ask='Întrebare pentru sală', answer='Răspuns',
   note='În stînga este slide-ul din curs, în dreapta aceeași figură sau același tabel, calculat în browser din aceleași date. Butonul ▶ modifică parametrul pas cu pas; '
        'casetele „Exemplu rezolvat” refac calculul cu cifrele capitolului.',
   src='Simulări: șocuri din distribuția Normală, generate cu seed-urile codului capitolului; cînd parametrii diferă de cei din curs, simularea se reface în browser. '
       'PIB: Eurostat, volume înlănțuite (2010), serie nesezonalizată, în versiunea folosită în curs. IAPC: Eurostat, pînă în august 2026. '
       'BET: EODHD; EUR/RON: cursul de referință BNR. Petele solare: setul de date din statsmodels, 1700–2008.'),
 'en': dict(file='ch2.html', html_lang='en', site='../index.html#chapters', course='Time Series Analysis',
   h1='Chapter 2 · ARMA Models', back='Course site', pdf_btn='Lecture PDF', ql='Quantlets',
   slide='Slide', ask='Question for the room', answer='Answer',
   note='The slide of the lecture is on the left, the same figure or table on the right, computed in the browser from the same data. The ▶ button moves the parameter step by step; '
        'the “Worked example” boxes redo the calculation with the numbers of the chapter.',
   src='Simulations: shocks from the Normal distribution, drawn with the seeds of the chapter code; when the parameters differ from those of the lecture, the simulation is redone in the browser. '
       'GDP: Eurostat, chain-linked volumes (2010), not seasonally adjusted, in the version used in the lecture. HICP: Eurostat, up to August 2026. '
       'BET: EODHD; EUR/RON: BNR reference rate. Sunspots: the statsmodels data set, 1700–2008.'),
}

GDP_MODELS = [f'{p}{q}' for p in range(4) for q in range(4)]


CSS = ('<style>.c2x{margin:.6rem 0 .3rem;padding:.55rem .8rem;border:1px solid #D5DEEC;border-left:4px solid #17A2B8;border-radius:6px;background:#F7FAFD;min-width:0}'
       '.c2x h4{margin:0 0 .2rem;font-size:.95rem;color:#1A3A6E}.c2x .c2f{overflow-x:auto;overflow-y:hidden;max-width:100%;font-size:.92rem}'
       '.c2x ol{margin:.3rem 0 0 1.25rem;padding:0;font-size:.92rem}.c2x li{margin:.18rem 0;overflow-wrap:anywhere}'
       '.c2t{max-width:100%;overflow-x:auto;margin:.55rem 0 .3rem}.c2t table{border-collapse:collapse;font-size:.86rem;margin:0;border-top:2px solid #1A3A6E;border-bottom:2px solid #1A3A6E}'
       '.c2t th,.c2t td{padding:.24rem .55rem;text-align:left;white-space:nowrap;vertical-align:top}.c2t thead th{border-bottom:1px solid #1A3A6E}'
       '.c2t tbody tr:nth-child(even){background:#F5F7FB}.c2t tbody tr.hi{background:#FDF0E1}.c2t .r{text-align:right}.c2cap{font-size:.84rem;margin:.15rem 0 .3rem;color:#1F2A44;white-space:normal}.c2x .ctl,.c2tc .ctl{max-width:360px;margin:.15rem 0 .2rem}.c2tc{margin:.55rem 0 0}</style>')


def tbl(i):
    return f'<div class="tbl c2t" id="{i}"></div>'


def wx(i, title, formula, ctl=''):
    """Worked-example box; `ctl` holds the controls that only drive the example (they stay out of the chart's controls bar)."""
    return f'<div class="c2x"><h4>{title}</h4>{ctl}<div class="c2f">\\[{formula}\\]</div><ol id="{i}"></ol></div>'


def _chart_widgets(lang):
    ro = lang == 'ro'
    s = lambda a, b: a if ro else b   # noqa: E731
    mod = lambda k: f'ARMA({k[0]},{k[1]})' + {'03': ' · BIC', '13': ' · AIC'}.get(k, '')   # noqa: E731
    return {
     'ar1': (slider('a1-p1', s('φ, panoul din stînga', 'φ, left panel'), -0.99, 0.99, 0.01, 0.9)
             + slider('a1-p2', s('φ, panoul din mijloc', 'φ, middle panel'), -0.99, 0.99, 0.01, 0.3)
             + slider('a1-p3', s('φ, panoul din dreapta', 'φ, right panel'), -0.99, 0.99, 0.01, -0.7) + play('a1'), charts('a1-chart')),
     'ar2': (select('a2-m', s('Exemplul', 'Example'), [('A', 'A: φ = (0,5; 0,3)' if ro else 'A: φ = (0.5, 0.3)'), ('B', 'B: φ = (1,0; −0,6)' if ro else 'B: φ = (1.0, −0.6)'),
                                                    ('C', 'C: φ = (0,6; 0,5)' if ro else 'C: φ = (0.6, 0.5)'), ('Sunspots', s('Sunspots: AR(2)-ul lui Yule', "Sunspots: Yule's AR(2)"))], 'B')
             + slider('a2-f1', 'φ₁', -2, 2, 0.001, 1.0) + slider('a2-f2', 'φ₂', -1.1, 1.1, 0.001, -0.6) + play('a2'), charts('a2-chart')),
     'ma1': (slider('m1-th', s('θ (rîndul de sus; jos: −θ)', 'θ (top row; bottom row: −θ)'), -2, 2, 0.05, 0.8) + play('m1'), charts('m1-chart')),
     'ma1-rho': (slider('mr-th', 'θ', -4, 4, 0.05, 0.5) + play('mr'), charts('mr-chart')),
     'psi': (select('ps-m', s('Modelul modificat', 'Model to change'), [(k, k) for k in ('AR(1)', 'AR(2)', 'MA(2)', 'ARMA(1,1)')], 'ARMA(1,1)')
             + slider('ps-a', 'φ₁', -2, 2, 0.05, 0.7) + slider('ps-b', 'φ₂', -1, 1, 0.05, 0) + slider('ps-c', 'θ₁', -2, 2, 0.05, 0.4)
             + slider('ps-d', 'θ₂', -2, 2, 0.05, 0) + play('ps'), charts('ps-chart')),
     'patterns': (slider('pt-T', s('Lungimea eșantionului T', 'Sample length T'), 50, 500, 10, 500) + play('pt')
                  + f'<button class="btn" id="pt-new" type="button">{s("Altă traiectorie", "New path")}</button>', charts('pt-chart')),
     'estimators': (slider('es-phi', s('φ adevărat (AR(1))', 'True φ (AR(1))'), 0, 0.98, 0.02, 0.9) + slider('es-th', s('θ adevărat (MA(1))', 'True θ (MA(1))'), -0.9, 0.9, 0.1, 0.5)
                    + select('es-T', s('Lungimea eșantionului T', 'Sample length T'), [(t, t) for t in (50, 100, 200, 400)], 100) + play('es'), charts('es-chart')),
     'ic': (slider('ic-f2', s('φ₂ adevărat (φ₁ = 0,5)', 'True φ₂ (φ₁ = 0.5)'), 0.05, 0.45, 0.05, 0.25)
            + select('ic-T', s('T în panoul din stînga', 'T in the left panel'), [(t, t) for t in (50, 100, 200, 400)], 100) + play('ic'), charts('ic-chart')),
     'lb-df': (slider('lb-m', s('Numărul de laguri m', 'Number of lags m'), 3, 20, 1, 10)
               + select('lb-a', s('Nivelul de semnificație', 'Significance level'), [('0.01', '1%'), ('0.05', '5%'), ('0.1', '10%')], '0.05') + play('lb'), charts('lb-chart')),
     'forecast-mean': (slider('fc-x', s('Ultima valoare X<sub>T</sub>', 'Last value X<sub>T</sub>'), -4, 4, 0.5, 3) + slider('fc-p1', s('φ, panoul din stînga', 'φ, left panel'), -0.95, 0.99, 0.01, 0.9)
                       + slider('fc-p2', s('φ, panoul din mijloc', 'φ, middle panel'), -0.95, 0.99, 0.01, 0.5)
                       + slider('fc-th', s('θ, panoul din dreapta', 'θ, right panel'), -0.95, 0.95, 0.05, 0.6) + play('fc'), charts('fc-chart')),
     'gdp-ident': (slider('gi-end', s('Eșantionul se termină în', 'Sample ends in'), 39, 105, 1, 105) + play('gi'), charts('gi-chart')),
     'gdp-diag': (select('gd-mod', s('Modelul', 'Model'), [(k, mod(k)) for k in GDP_MODELS], '03')
                  + select('gd-df', s('Gradele de libertate ale testului Ljung–Box', 'Degrees of freedom of the Ljung–Box test'), [('right', 'm − p − q'), ('wrong', 'm')], 'right')
                  + play('gd'), charts('gd-chart')),
     'gdp-forecast': (slider('gf-o', s('Originea prognozei', 'Forecast origin'), 0, 48, 1, 48)
                      + select('gf-lv', s('Nivelul intervalului', 'Interval level'), [('0.8', '80%'), ('0.9', '90%'), ('0.95', '95%'), ('0.99', '99%')], '0.95') + play('gf'), charts('gf-chart')),
     'inflation': (select('in-p', s('Ordinul AR p', 'AR order p'), [(str(p), f'AR({p})' + (' · BIC' if p == 2 else '')) for p in range(1, 7)], '2') + play('in'), charts('in-chart')),
     'returns': (slider('rt-y0', s('De la anul', 'From year'), 2000, 2023, 1, 2000) + slider('rt-y1', s('Pînă în anul', 'To year'), 2003, 2026, 1, 2026) + play('rt'), charts('rt-chart')),
     'sunspots': (select('ss-p', s('Ordinul AR p', 'AR order p'), [(str(p), f'AR({p})') for p in range(1, 11)], '2') + play('ss'), charts('ss-chart')),
    }


def widgets_all(lang):
    """Widgets of every block of the interactive page: the chart blocks and EXTRA (worked examples and live tables)."""
    ro = lang == 'ro'
    s = lambda a, b: a if ro else b   # noqa: E731
    W = s('Exemplu rezolvat', 'Worked example')
    d = (lambda x: x.replace('0.5', '0{,}5').replace('1.96', '1{,}96')) if ro else (lambda x: x)   # decimal comma in the RO formulas
    w = _chart_widgets(lang)
    ctl, ch = w['ar1']
    w['ar1'] = (ctl, CSS + ch)
    mods = [(k, f'ARMA({k[0]},{k[1]})') for k in GDP_MODELS]
    lev = [('0.8', '80%'), ('0.9', '90%'), ('0.95', '95%'), ('0.99', '99%')]
    end = s('Eșantionul se termină în', 'Sample ends in')
    w.update({
     'wx-ar1': (slider('xa-c', s('Termenul liber c', 'Intercept c'), -5, 5, 0.5, 2) + slider('xa-phi', 'φ', -0.95, 0.99, 0.01, 0.8) + slider('xa-s2', 'σ²', 0.25, 4, 0.25, 1)
                + slider('xa-h', s('Lagul h', 'Lag h'), 1, 10, 1, 3) + play('xa'),
                wx('xa-ws', W, d(r'\begin{gathered}X_t=c+\phi X_{t-1}+\varepsilon_t:\quad \mu=\frac{c}{1-\phi},\quad \gamma(0)=\frac{\sigma^2}{1-\phi^2},\\ \rho(h)=\phi^h,\quad h_{1/2}=\frac{\ln 0.5}{\ln|\phi|},\quad \psi_j=\phi^j\end{gathered}'))
                + tbl('xa-tb') + charts('xa-chart')),
     'ar-roots': (select('xr-m', s('Exemplul', 'Example'), [('A', 'A: φ = (0,5; 0,3)' if ro else 'A: φ = (0.5, 0.3)'), ('B', 'B: φ = (1,0; −0,6)' if ro else 'B: φ = (1.0, −0.6)'),
                                                         ('C', 'C: φ = (0,6; 0,5)' if ro else 'C: φ = (0.6, 0.5)'), ('Sunspots', s('Sunspots: AR(2)-ul lui Yule', "Sunspots: Yule's AR(2)"))], 'B')
                  + slider('xr-f1', 'φ₁', -2, 2, 0.001, 1.0) + slider('xr-f2', 'φ₂', -1.1, 1.1, 0.001, -0.6) + play('xr'),
                  wx('xr-ws', W, r'\begin{gathered}\phi(z)=1-\phi_1z-\phi_2z^2=0\ \Rightarrow\ z_{1,2}=\frac{-\phi_1\pm\sqrt{\phi_1^2+4\phi_2}}{2\phi_2},\\ \text{' + s('staționar', 'stationary') + r'}\iff |z_i|>1\iff \phi_1+\phi_2<1,\ \phi_2-\phi_1<1,\ |\phi_2|<1\end{gathered}')
                  + tbl('xr-tb') + charts('xr-chart')),
     'wx-theta': (slider('xt-th', 'θ', -3, 3, 0.05, 2) + play('xt'),
                  wx('xt-ws', W, r'\begin{gathered}X_t=\varepsilon_t+\theta\varepsilon_{t-1}:\quad \gamma(0)=\sigma^2(1+\theta^2),\quad \gamma(1)=\theta\sigma^2;\\ (\theta,\sigma^2)\ \text{' + s('și', 'and') + r'}\ (1/\theta,\ \theta^2\sigma^2)\ \text{' + s('au aceeași ACF', 'have the same ACF') + r'};\quad \varepsilon_t=\sum_{j\ge0}(-\theta)^jX_{t-j}\end{gathered}',
                     slider('xt-s2', 'σ²', 0.25, 4, 0.25, 1))
                  + tbl('xt-tb') + charts('xt-chart')),
     'wx-psi': (slider('xp-phi', 'φ', -0.95, 0.95, 0.05, 0.7) + slider('xp-th', 'θ', -0.95, 0.95, 0.05, 0.4) + play('xp'),
                wx('xp-ws', W, r'\begin{gathered}\psi_0=1,\quad \psi_j=\theta_j+\sum_{i=1}^{\min(j,p)}\phi_i\psi_{j-i};\\ \text{ARMA(1,1)}:\ \psi_j=\phi^{j-1}(\phi+\theta),\quad \sum_{j\ge0}\psi_j^2=1+\frac{(\phi+\theta)^2}{1-\phi^2}=\frac{\gamma(0)}{\sigma^2}\end{gathered}')
                + tbl('xp-tb') + charts('xp-chart')),
     'wx-arma11': (slider('xm-phi', 'φ', -0.95, 0.95, 0.05, 0.7) + slider('xm-th', 'θ', -0.95, 0.95, 0.05, 0.4) + play('xm'),
                   wx('xm-ws', W, r'\gamma(0)=\sigma^2\frac{1+2\phi\theta+\theta^2}{1-\phi^2},\quad \rho(1)=\frac{(1+\phi\theta)(\phi+\theta)}{1+2\phi\theta+\theta^2},\quad \rho(h)=\phi\,\rho(h-1)\ \ (h\ge2)')
                   + tbl('xm-tb') + charts('xm-chart')),
     'ident-table': (select('it-m', s('Modelul', 'Model'), [('AR(2)', 'AR(2), φ = (1,0; −0,6)' if ro else 'AR(2), φ = (1.0, −0.6)'), ('MA(2)', 'MA(2), θ = (0,6; 0,3)' if ro else 'MA(2), θ = (0.6, 0.3)'),
                                                           ('ARMA(1,1)', 'ARMA(1,1), φ = 0,7, θ = 0,4' if ro else 'ARMA(1,1), φ = 0.7, θ = 0.4')], 'AR(2)')
                     + slider('it-T', s('Lungimea eșantionului T', 'Sample length T'), 50, 500, 10, 500) + play('it'), tbl('it-tb1') + tbl('it-tb2')),
     'wx-yw': (slider('xy-end', end, 39, 105, 1, 105) + select('xy-p', s('Ordinul', 'Order'), [('1', 'AR(1)'), ('2', 'AR(2)')], '2') + play('xy'),
               wx('xy-ws', W, r'\hat R\hat\phi=\hat\rho:\quad \hat\phi_1=\frac{\hat\rho_1(1-\hat\rho_2)}{1-\hat\rho_1^2},\quad \hat\phi_2=\frac{\hat\rho_2-\hat\rho_1^2}{1-\hat\rho_1^2},\quad \hat\sigma^2=\hat\gamma(0)\big(1-\hat\phi_1\hat\rho_1-\hat\phi_2\hat\rho_2\big)')
               + tbl('xy-tb')),
     'ml-lik': (slider('xl-end', end, 39, 105, 1, 105) + play('xl'),
                wx('xl-ws', W, d(r'\begin{gathered}\ln L(\mu,\phi,\sigma^2)=-\tfrac12\ln\frac{2\pi\sigma^2}{1-\phi^2}-\frac{(1-\phi^2)(x_1-\mu)^2}{2\sigma^2}\\ -\sum_{t=2}^{T}\Big[\tfrac12\ln 2\pi\sigma^2+\frac{(x_t-\mu-\phi(x_{t-1}-\mu))^2}{2\sigma^2}\Big],\qquad \mathrm{se}(\hat\phi)=\Big(-\frac{d^2\ln L}{d\phi^2}\Big)^{-1/2}\end{gathered}'))
                + charts('xl-chart')),
     'wx-ic': (select('xi-m', s('Modelul', 'Model'), mods, '03') + play('xi'),
               wx('xi-ws', W, r'k=p+q+2,\qquad \mathrm{AIC}=-2\ln L+2k,\qquad \mathrm{BIC}=-2\ln L+k\ln T,\qquad \mathrm{AICc}=\mathrm{AIC}+\frac{2k(k+1)}{T-k-1}')
               + tbl('xi-tb')),
     'diag-table': (select('dt-mod', s('Modelul', 'Model'), mods, '03') + slider('dt-m', s('Numărul de laguri m', 'Number of lags m'), 1, 16, 1, 8) + play('dt'),
                    wx('dt-ws', W, r'\begin{gathered}Q^*(m)=T(T+2)\sum_{h=1}^{m}\frac{\hat\rho_{\hat\varepsilon}(h)^2}{T-h}\approx\chi^2(m-p-q),\\ \mathrm{JB}=\frac{T}{6}\Big(S^2+\frac{(K-3)^2}{4}\Big)\approx\chi^2(2)\end{gathered}')
                    + tbl('dt-tb')),
     'wx-fc': (slider('xf-c', s('Termenul liber c', 'Intercept c'), -5, 5, 0.5, 2) + slider('xf-phi', 'φ', -0.95, 0.99, 0.01, 0.8) + slider('xf-s2', 'σ²', 0.25, 4, 0.25, 1)
               + slider('xf-x', s('Ultima valoare X<sub>T</sub>', 'Last value X<sub>T</sub>'), 0, 20, 0.5, 12) + play('xf'),
               wx('xf-ws', W, d(r'\begin{gathered}\hat X_{T+h|T}=\mu+\phi^h(X_T-\mu),\quad \hat X_{T+h|T}\pm1.96\,\sigma_h,\\ \sigma_h^2=\sigma^2\sum_{j=0}^{h-1}\psi_j^2=\sigma^2\frac{1-\phi^{2h}}{1-\phi^2}\end{gathered}'))
               + tbl('xf-tb') + charts('xf-chart')),
     'gdp-grid': (select('gg-c', s('Criteriul', 'Criterion'), [('aic', 'AIC'), ('bic', 'BIC'), ('aicc', 'AICc')], 'aic')
                  + play('gg'),
                  '<div class="c2tc">' + select('gg-s', s('Ordinea rîndurilor din tabel', 'Row order of the table'), [('pq', s('după (p, q)', 'by (p, q)')), ('crit', s('după criteriu', 'by the criterion'))], 'pq') + '</div>'
                  + tbl('gg-tb') + charts('gg-chart')),
     'ma3-table': (select('mq-m', s('Modelul', 'Model'), mods, '03') + play('mq'),
                   wx('mq-ws', W, r'\begin{gathered}\hat y_t=\hat\mu+\hat\varepsilon_t+\hat\theta_1\hat\varepsilon_{t-1}+\hat\theta_2\hat\varepsilon_{t-2}+\hat\theta_3\hat\varepsilon_{t-3},\\ \text{' + s('invertibil', 'invertible') + r'}\iff \theta(z)=1+\theta_1z+\theta_2z^2+\theta_3z^3\ne0\ \ \text{' + s('pentru', 'for') + r'}\ |z|\le1\end{gathered}')
                   + tbl('mq-tb')),
     'gdp-fc-table': (slider('ft-o', s('Originea prognozei', 'Forecast origin'), 0, 48, 1, 48) + select('ft-lv', s('Nivelul intervalului', 'Interval level'), lev, '0.95') + play('ft'),
                      wx('ft-ws', W, r'\begin{gathered}\sigma_h^2=\hat\sigma^2\sum_{j=0}^{h-1}\hat\psi_j^2;\\ \text{MA(3)}:\ \hat\psi_j=\hat\theta_j,\ \ \sigma_h^2=\hat\sigma^2(1+\hat\theta_1^2+\hat\theta_2^2+\hat\theta_3^2)\ \ (h\ge4)\end{gathered}')
                      + tbl('ft-tb') + charts('ft-chart')),
     'ret-table': (slider('rx-y0', s('De la anul', 'From year'), 2000, 2023, 1, 2000) + slider('rx-y1', s('Pînă în anul', 'To year'), 2003, 2026, 1, 2026) + play('rx'),
                   wx('rx-ws', W, r'\mathrm{se}_{\text{OPG}}=\Big[\Big(\sum_t s_ts_t^\top\Big)^{-1}\Big]_{\phi\phi}^{1/2},\qquad \mathrm{se}_{\text{' + s('clasic', 'classical') + r'}}\approx\sqrt{\frac{1-\hat\phi^2}{T}},\qquad t=\frac{\hat\phi}{\mathrm{se}}')
                   + tbl('rx-tb')),
    })
    return w


def _extra_pages(lang):
    """Page and title of the frames of EXTRA, from the .tex and the beamer .nav file."""
    from interactive_common import ROOT, frames_of, latex_text
    tex = os.path.join(ROOT, TEX[lang])
    frames = frames_of(open(tex + '.tex', encoding='utf-8').read())
    pages = re.findall(r'\\beamer@framepages \{(\d+)\}\{(\d+)\}', open(tex + '.nav', encoding='utf-8').read())
    titles = [latex_text(t) for t, _ in frames]
    res = {}
    for anchor, tro, ten, _, _ in EXTRA:
        k = titles.index(tro if lang == 'ro' else ten)
        res[anchor] = (int(pages[k][0]), titles[k])
    return res


def _extra_live(anchor, lang, W):
    t = L[lang]
    ctl, ch = W[anchor]
    q, a = QA[lang][anchor]
    live = f'<div class="live"><div class="ctls">{ctl}</div>{ch}<ul class="out" id="{PFX[anchor]}-out"></ul></div>'
    tail = f'<div class="ask"><span class="ask-tag">{t["ask"]}</span> {q}</div>\n  <details><summary>{t["answer"]}</summary><p>{a}</p></details>'
    return live, tail


def widgets(lang):
    """Chart blocks for build_web_lecture.py: each host carries the web version of the EXTRA blocks of its section (moved by ch2.js)."""
    W = widgets_all(lang)
    xp = _extra_pages(lang)
    res = {}
    for a, _, _ in BLOCKS:
        ctl, ch = W[a]
        held = ''
        for x, _, _, _, host in EXTRA:
            if host == a:
                live, tail = _extra_live(x, lang, W)
                held += f'<section class="blk wlive xblk" id="{x}" data-frame="{xp[x][0]}">{live}{tail}</section>'
        res[a] = (ctl, ch + (f'<div class="xhold">{held}</div>' if held else ''))
    return res


QA = {
 'ro': {
  'ar1': ('Cu aceleași 200 de șocuri, butonul ▶ crește φ din panoul din stînga de la 0 la 0,99; φ din celelalte două panouri urmează proporțional, pînă la 0,33 și −0,77. Ce se întîmplă în panoul din stînga cu ρ̂(1) și cu varianța de selecție?',
          'Cresc amîndouă: ρ̂(1) urcă de la 0,10 la 0,95, iar varianța de selecție de la 1,04 la 14,5. Varianța teoretică, 1/(1 − φ²), crește mai repede: 5,26 la φ = 0,9 și 50,3 la φ = 0,99. '
          'La φ = 0,9, eșantionul dă 4,98, sub valoarea teoretică: o traiectorie scurtă a unui proces persistent nu acoperă toată variația lui. Cu φ = −0,7 semnul autocorelațiilor alternează: ρ̂(1) = −0,58, ρ̂(2) = 0,34; la capătul cursei, cu φ = −0,77, ρ̂(1) = −0,66 și ρ̂(2) = 0,44.'),
  'ar2': ('În exemplul C, φ₁ = 0,6 și φ₂ = 0,5 sînt amîndoi sub 1. Este procesul staționar?',
          'Nu. φ₁ + φ₂ = 1,1 > 1, deci punctul este în afara triunghiului. Rădăcinile polinomului 1 − 0,6z − 0,5z² sînt −2,14 și 0,936; a doua este în interiorul cercului unitate (rădăcina inversă are modulul 1,068). '
          'Staționaritatea se citește din rădăcini, nu din coeficienți luați separat. B are rădăcini complexe, 0,833 ± 0,986i, cu modulul 1,291: ACF este o undă cu perioada 7,2 laguri, amortizată cu 0,775 pe lag.'),
  'ma1': ('Butonul ▶ crește θ de la 0 la 2. Poate un MA(1) să aibă ρ(1) mai mare decît 0,5?',
          'Nu. ρ(1) = θ/(1 + θ²) are maximul 0,5 la θ = 1 și scade apoi: la θ = 2 este 0,4, la fel ca la θ = 0,5. La θ = ±0,8, valorile teoretice sînt ±0,488, iar cele de selecție, cu T = 300, sînt 0,56 și −0,51. '
          'ACF are o singură valoare nenulă, iar PACF descrește: φ̂₂₂ = −0,23 pentru θ = 0,8.'),
  'ma1-rho': ('Ce altă valoare a lui θ dă același ρ(1) = 0,4 ca θ = 0,5 și pe care o păstrăm?',
              'θ = 2 = 1/0,5. Cele două modele au aceeași ACF (cu σ² înmulțit cu θ²), deci datele nu le pot deosebi: o problemă de identificare. Convenția de invertibilitate, |θ| < 1, păstrează θ = 0,5.'),
  'psi': ('În ARMA(1,1) cu φ = 0,7 și θ = 0,4, de ce este primul răspuns la impuls, ψ₁ = 1,10, mai mare decît 1?',
          'ψ₁ = φ + θ = 0,7 + 0,4 = 1,1: șocul de ieri intră o dată prin termenul MA și o dată prin partea AR. De la j = 2, ψⱼ = φψⱼ₋₁: 0,77; 0,54; 0,38. '
          'Suma Σψⱼ² = 3,37 este varianța procesului în unități σ². La MA(2) răspunsul este exact zero după două perioade, iar la AR(2) cu rădăcini complexe își schimbă semnul: ψ₃ = −0,20, ψ₄ = −0,44. Butonul ▶ dă primului coeficient al celor patru modele (θ₁ la MA(2)) aceeași valoare, de la 0 la 0,95 și înapoi la 0,7: la AR răspunsul se prelungește, la MA(2) rămîne zero după j = 2.'),
  'patterns': ('Butonul ▶ crește T de la 50 la 500. Ce se întîmplă cu banda ±1,96/√T și cu valorile de selecție?',
               'Banda se îngustează de la ±0,28 la ±0,09, iar valorile de selecție se apropie de cele teoretice. La T = 500, pentru AR(2), φ̂₂₂ = −0,56 față de φ₂₂ = −0,60; pentru MA(2), ρ̂(1) = 0,51 și ρ̂(2) = 0,15 față de 0,54 și 0,21. '
               'Cu eșantioane scurte, corelogramele unor modele diferite se pot confunda.'),
  'estimators': ('Cu φ = 0,9 și T = 100, sînt estimatorii coeficientului AR(1) nedeplasați?',
                 'Nu. Mediile celor 400 de estimări sînt 0,853 (Yule–Walker), 0,864 (cele mai mici pătrate condiționate) și 0,866 (MLE), toate sub 0,9; aproximarea −(1 + 3φ)/T dă −0,037. '
                 'Abaterile standard, 0,054–0,056, depășesc valoarea asimptotică 0,044. În simularea din browser cu T = 400, mediile urcă la 0,889–0,891. Pentru MA(1) cu θ = 0,5, mediile sînt 0,504 și 0,506.'),
  'ic': ('Cu T = 1000, care criteriu găsește mai des ordinul adevărat p = 2?',
         'BIC, în 98,6% dintre eșantioane, față de 74,4% pentru AIC, care alege un ordin prea mare în 25,6% din cazuri. Cu T = 100 ordinea se inversează: AIC 54,4%, BIC 47,4%, iar BIC alege un model prea mic în 50,4% din cazuri, '
         'pentru că φ₂ = 0,25 este greu de detectat în 100 de observații.'),
  'lb-df': ('Reziduurile modelului adevărat ARMA(1,1) sînt testate cu Ljung–Box, m = 10, la 5%. Cît de des respinge testul cu 10 grade de libertate?',
            'În 1,4% dintre eșantioane, nu în 5%: reziduurile estimate sînt mai puțin corelate decît un zgomot alb adevărat la primele laguri. Cu m − p − q = 8 grade de libertate, rata este 3,4%; pentru AR(1), 3,2% cu 10 și 4,8% cu 9. '
            'La m = 3 diferența crește: pentru ARMA(1,1), 0,2% cu 3 grade de libertate și 4,8% cu 1.'),
  'forecast-mean': ('Pornind din X<sub>T</sub> = 3, ce valoare are prognoza după 5 perioade pentru AR(1) cu φ = 0,9 și pentru φ = 0,5?',
                    '1,77 pentru φ = 0,9 (eroarea standard 1,85) și 0,094 pentru φ = 0,5, practic media. Intervalul tinde spre banda necondiționată μ ± 1,96σ<sub>X</sub>, cu σ<sub>X</sub> = 2,29 pentru φ = 0,9 și 1,15 pentru φ = 0,5. '
                    'La MA(1), numai primul pas este informativ, θε<sub>T</sub> = 1,2; de la h = 2 prognoza este media și intervalul are lățimea de lungă durată. Butonul ▶ mută X<sub>T</sub> de la 3 la −4 și înapoi: prognozele AR(1) revin la medie din orice punct de plecare, iar intervalele nu se schimbă, pentru că erorile standard nu depind de X<sub>T</sub>.'),
  'gdp-ident': ('ACF a creșterii anuale a PIB se anulează după lagul 3. Rămîne tiparul dacă eșantionul se oprește în T4 2019, înainte de pandemie?',
                'Da. Cu 80 de trimestre, primele autocorelații sînt 0,62; 0,45; 0,24, apoi 0,04 la lagul 4, sub banda de ±0,22. Pe tot eșantionul, cu 106 trimestre: 0,62; 0,40; 0,21 și −0,01. '
                'Tiparul vine din construcția seriei: rata anuală este suma a patru rate trimestriale, iar ratele consecutive au trei trimestre comune. Ljung–Box Q*(8) = 64,9 pe tot eșantionul.'),
  'gdp-diag': ('Pentru MA(3), testul Ljung–Box cu m = 8 dă p = 0,375 cu 5 grade de libertate. Ce p-value ar da cu 8 grade de libertate?',
               '0,720, o concluzie prea liniștitoare: gradele de libertate trebuie reduse cu numărul parametrilor ARMA estimați. Reziduurile nu sînt Normale: Jarque–Bera 39,0, p < 0,001, asimetria −0,75, kurtosis-ul 5,61; '
               'cel mai mare reziduu, −11,4 în T2 2020, are −3,89 abateri standard. AR(1) lasă corelație în reziduuri: Q*(8) = 16,2 cu 7 grade de libertate, p = 0,023.'),
  'gdp-forecast': ('De la ce orizont devine constantă prognoza MA(3)?',
                   'De la al patrulea trimestru: −0,14; 1,81; 2,77, apoi media de 3,17%. După q = 3 pași modelul cunoaște doar media. AR(1) revine treptat: 2,99% după 8 trimestre, cu media 3,12%. '
                   'Intervalele de 95% sînt [−6,1; 5,9] pentru un trimestru și [−6,3; 12,7] de la patru trimestre. Cu originea în T4 2019, ambele modele prognozează peste 3,8% pentru 2020, iar seria scade la −9,9% în T2 2020.'),
  'inflation': ('Spre ce valoare revine prognoza AR(2): spre ținta BNR sau spre media estimată?',
                'Spre media estimată, 5,16%, care este media perioadei 2005–2026, nu spre ținta de 2,5%: de la 6,09% în august 2026 la 5,44% în septembrie și 5,13% în august 2028. '
                'φ̂₁ + φ̂₂ = 0,976, iar cea mai mare rădăcină inversă are modulul 0,963, deci revenirea este lentă. Testul Ljung–Box pe reziduuri respinge: Q*(12) = 71,6 cu 10 grade de libertate, p < 0,001.'),
  'returns': ('Pentru BET, φ̂ = 0,109 este foarte semnificativ. Explică AR(1) o parte importantă din variația randamentelor zilnice?',
              'Nu: R² este 1,2% pentru BET și 2,9% pentru EUR/RON. Rezultatul este real din punct de vedere statistic, dar mic din punct de vedere economic. '
              'Pătratele reziduurilor sînt puternic autocorelate (Ljung–Box Q*(10) = 2496 pentru BET și 2351 pentru EUR/RON), iar kurtosis-ul reziduurilor este 14,0 și 17,0: modelul descrie media, nu varianța.'),
  'sunspots': ('Ce perioadă a ciclului solar implică AR(2)-ul lui Yule estimat pe 1749–1924?',
               '10,6 ani. Coeficienții estimați, φ̂₁ = 1,336 și φ̂₂ = −0,650, dau rădăcini complexe; perioada este 2π/arccos(φ̂₁/(2√−φ̂₂)) = 10,6, iar modulul rădăcinii inverse 0,806. '
               'Pe întregul eșantion, AR(2) explică 83% din varianță cu un an înainte, iar AIC și BIC aleg amîndouă p = 9 dintre p ≤ 10.'),
 },
 'en': {
  'ar1': ('With the same 200 shocks, the ▶ button raises φ in the left panel from 0 to 0.99; φ in the other two panels follows in proportion, up to 0.33 and −0.77. What happens in the left panel to ρ̂(1) and to the sample variance?',
          'Both rise: ρ̂(1) goes from 0.10 to 0.95, and the sample variance from 1.04 to 14.5. The theoretical variance, 1/(1 − φ²), rises faster: 5.26 at φ = 0.9 and 50.3 at φ = 0.99. '
          'At φ = 0.9 the sample gives 4.98, below the theoretical value: a short path of a persistent process does not cover all its variation. With φ = −0.7 the sign of the autocorrelations alternates: ρ̂(1) = −0.58, ρ̂(2) = 0.34; at the end of the sweep, with φ = −0.77, ρ̂(1) = −0.66 and ρ̂(2) = 0.44.'),
  'ar2': ('In example C, φ₁ = 0.6 and φ₂ = 0.5 are both below 1. Is the process stationary?',
          'No. φ₁ + φ₂ = 1.1 > 1, so the point lies outside the triangle. The roots of 1 − 0.6z − 0.5z² are −2.14 and 0.936; the second lies inside the unit circle (its inverse root has modulus 1.068). '
          'Stationarity is read from the roots, not from the coefficients one at a time. B has complex roots, 0.833 ± 0.986i, with modulus 1.291: its ACF is a wave with a period of 7.2 lags, damped by 0.775 per lag.'),
  'ma1': ('The ▶ button raises θ from 0 to 2. Can an MA(1) have ρ(1) above 0.5?',
          'No. ρ(1) = θ/(1 + θ²) reaches its maximum of 0.5 at θ = 1 and then falls: at θ = 2 it is 0.4, the same as at θ = 0.5. At θ = ±0.8 the theoretical values are ±0.488, and the sample values, with T = 300, are 0.56 and −0.51. '
          'The ACF has a single non-zero value, while the PACF decays: φ̂₂₂ = −0.23 for θ = 0.8.'),
  'ma1-rho': ('Which other value of θ gives the same ρ(1) = 0.4 as θ = 0.5, and which one do we keep?',
              'θ = 2 = 1/0.5. The two models have the same ACF (with σ² multiplied by θ²), so the data cannot tell them apart: an identification problem. The invertibility convention, |θ| < 1, keeps θ = 0.5.'),
  'psi': ('In the ARMA(1,1) with φ = 0.7 and θ = 0.4, why is the first impulse response, ψ₁ = 1.10, above 1?',
          'ψ₁ = φ + θ = 0.7 + 0.4 = 1.1: yesterday’s shock enters once through the MA term and once through the AR part. From j = 2 on, ψⱼ = φψⱼ₋₁: 0.77, 0.54, 0.38. '
          'The sum Σψⱼ² = 3.37 is the variance of the process in units of σ². For the MA(2) the response is exactly zero after two periods, and for the AR(2) with complex roots it changes sign: ψ₃ = −0.20, ψ₄ = −0.44. The ▶ button gives the first coefficient of the four models (θ₁ for the MA(2)) the same value, from 0 to 0.95 and back to 0.7: the AR responses last longer, the MA(2) response stays zero after j = 2.'),
  'patterns': ('The ▶ button raises T from 50 to 500. What happens to the ±1.96/√T band and to the sample values?',
               'The band narrows from ±0.28 to ±0.09, and the sample values move towards the theoretical ones. At T = 500, for the AR(2), φ̂₂₂ = −0.56 against φ₂₂ = −0.60; for the MA(2), ρ̂(1) = 0.51 and ρ̂(2) = 0.15 against 0.54 and 0.21. '
               'With short samples, the correlograms of different models can be confused.'),
  'estimators': ('With φ = 0.9 and T = 100, are the estimators of the AR(1) coefficient unbiased?',
                 'No. The means of the 400 estimates are 0.853 (Yule–Walker), 0.864 (conditional least squares) and 0.866 (MLE), all below 0.9; the approximation −(1 + 3φ)/T gives −0.037. '
                 'The standard deviations, 0.054–0.056, exceed the asymptotic value 0.044. In the browser simulation with T = 400, the means rise to 0.889–0.891. For the MA(1) with θ = 0.5, the means are 0.504 and 0.506.'),
  'ic': ('With T = 1000, which criterion finds the true order p = 2 more often?',
         'BIC, in 98.6% of the samples, against 74.4% for AIC, which picks an order that is too large in 25.6% of the cases. With T = 100 the ranking reverses: AIC 54.4%, BIC 47.4%, and BIC picks a model that is too small in 50.4% of the cases, '
         'because φ₂ = 0.25 is hard to detect in 100 observations.'),
  'lb-df': ('The residuals of the true ARMA(1,1) are tested with Ljung–Box, m = 10, at 5%. How often does the test reject with 10 degrees of freedom?',
            'In 1.4% of the samples, not in 5%: the estimated residuals are less correlated than true white noise at the first lags. With m − p − q = 8 degrees of freedom the rate is 3.4%; for the AR(1), 3.2% with 10 and 4.8% with 9. '
            'At m = 3 the gap widens: for the ARMA(1,1), 0.2% with 3 degrees of freedom and 4.8% with 1.'),
  'forecast-mean': ('Starting from X<sub>T</sub> = 3, what is the forecast 5 periods ahead for the AR(1) with φ = 0.9 and with φ = 0.5?',
                    '1.77 for φ = 0.9 (standard error 1.85) and 0.094 for φ = 0.5, practically the mean. The interval approaches the unconditional band μ ± 1.96σ<sub>X</sub>, with σ<sub>X</sub> = 2.29 for φ = 0.9 and 1.15 for φ = 0.5. '
                    'For the MA(1) only the first step is informative, θε<sub>T</sub> = 1.2; from h = 2 on, the forecast is the mean and the interval has its long-run width. The ▶ button moves X<sub>T</sub> from 3 to −4 and back: the AR(1) forecasts return to the mean from any starting point, and the intervals do not change, because the standard errors do not depend on X<sub>T</sub>.'),
  'gdp-ident': ('The ACF of annual GDP growth cuts off after lag 3. Does the pattern remain if the sample stops in 2019Q4, before the pandemic?',
                'Yes. With 80 quarters, the first autocorrelations are 0.62, 0.45 and 0.24, then 0.04 at lag 4, inside the ±0.22 band. On the full sample of 106 quarters: 0.62, 0.40, 0.21 and −0.01. '
                'The pattern comes from the construction of the series: the annual rate is the sum of four quarterly rates, and consecutive annual rates share three quarters. Ljung–Box Q*(8) = 64.9 on the full sample.'),
  'gdp-diag': ('For the MA(3), the Ljung–Box test with m = 8 gives p = 0.375 with 5 degrees of freedom. Which p-value would it give with 8 degrees of freedom?',
               '0.720, a conclusion that is too reassuring: the degrees of freedom must be reduced by the number of estimated ARMA parameters. The residuals are not Normal: Jarque–Bera 39.0, p < 0.001, skewness −0.75, kurtosis 5.61; '
               'the largest residual, −11.4 in 2020Q2, is −3.89 standard deviations. The AR(1) leaves correlation in the residuals: Q*(8) = 16.2 with 7 degrees of freedom, p = 0.023.'),
  'gdp-forecast': ('From which horizon on is the MA(3) forecast constant?',
                   'From the fourth quarter: −0.14, 1.81, 2.77, then the mean of 3.17%. After q = 3 steps the model only knows the mean. The AR(1) returns gradually: 2.99% after 8 quarters, with a mean of 3.12%. '
                   'The 95% intervals are [−6.1, 5.9] for one quarter and [−6.3, 12.7] from four quarters on. With the origin in 2019Q4, both models forecast more than 3.8% for 2020, while the series falls to −9.9% in 2020Q2.'),
  'inflation': ('Towards which value does the AR(2) forecast return: the BNR target or the estimated mean?',
                'Towards the estimated mean, 5.16%, the average of 2005–2026, not towards the 2.5% target: from 6.09% in August 2026 to 5.44% in September and 5.13% in August 2028. '
                'φ̂₁ + φ̂₂ = 0.976 and the largest inverse root has modulus 0.963, so the return is slow. The Ljung–Box test on the residuals rejects: Q*(12) = 71.6 with 10 degrees of freedom, p < 0.001.'),
  'returns': ('For the BET, φ̂ = 0.109 is highly significant. Does the AR(1) explain a large part of the variation of daily returns?',
              'No: R² is 1.2% for the BET and 2.9% for EUR/RON. The effect is real statistically but small economically. '
              'The squared residuals are strongly autocorrelated (Ljung–Box Q*(10) = 2496 for the BET and 2351 for EUR/RON), and the kurtosis of the residuals is 14.0 and 17.0: the model describes the mean, not the variance.'),
  'sunspots': ("What period of the solar cycle does Yule's AR(2), estimated on 1749–1924, imply?",
               '10.6 years. The estimated coefficients, φ̂₁ = 1.336 and φ̂₂ = −0.650, give complex roots; the period is 2π/arccos(φ̂₁/(2√−φ̂₂)) = 10.6, and the modulus of the inverse root is 0.806. '
               'On the full sample, the AR(2) explains 83% of the variance one year ahead, and AIC and BIC both pick p = 9 among p ≤ 10.'),
 },
}

# readouts: JavaScript functions, one per block
TEXT_JS = {
 'ro': r"""window.CH2_TEXT = {
 play: 'Pornește', pause: 'Pauză', and: ' și ', lt: '< ',
 q: s => `T${s[5]} ${s.slice(0, 4)}`,
 m: d => ['ianuarie', 'februarie', 'martie', 'aprilie', 'mai', 'iunie', 'iulie', 'august', 'septembrie', 'octombrie', 'noiembrie', 'decembrie'][+d.slice(5, 7) - 1] + ' ' + d.slice(0, 4),
 a1: (p, r1, r2, v, vt) => `φ = ${p}: ρ̂(1) = ${r1}, ρ̂(2) = ${r2}; varianța de selecție ${v} (teoretic 1/(1 − φ²) = ${vt}).`,
 a2a: (a, b, z, m, st) => `φ = (${a}; ${b}): rădăcinile polinomului caracteristic ${z}; modulele rădăcinilor inverse ${m}: ${st ? 'proces staționar' : 'proces nestaționar'}.`,
 a2c: (per, d) => `Rădăcini complexe: ACF este o undă cu perioada de ${per} laguri, amortizată cu ${d} pe lag.`,
 a2r: r => `ρ(1) = φ₁/(1 − φ₂) = ${r}.`,
 m1: (th, rt, r1, p2, inv) => `θ = ${th}: ρ(1) = ${rt}, ρ̂(1) = ${r1}, φ̂₂₂ = ${p2}${inv ? '' : '; |θ| > 1: modelul nu este invertibil'}.`,
 mr: (t, r, u, inv, unit) => `θ = ${t}: ρ(1) = ${r}.` + (u === null ? ' Pentru θ = 0 seria este zgomot alb.' : unit ? ' La |θ| = 1, θ și 1/θ coincid, iar |ρ(1)| = 0,5 este maximul.' : ` Același ρ(1) pentru θ = ${u}; reprezentarea invertibilă este θ = ${inv}.`),
 ps: (lab, p, s) => `${lab}: ψ₁…ψ₅ = ${p}. ` + (s === null ? 'Partea AR nu este staționară: ponderile ψ nu se sting.' : `Σψⱼ² = ${s} (varianța procesului, în unități σ²).`),
 pt: (n, r1, t1, r2, t2, p2, q2, b) => `${n}: ρ̂(1) = ${r1} (teoretic ${t1}), ρ̂(2) = ${r2} (${t2}); φ̂₂₂ = ${p2} (${q2}); banda ±${b}.`,
 es0: l => l ? 'Cele 400 de eșantioane din curs.' : 'Simulare în browser: 400 de eșantioane noi pentru parametrii aleși.',
 esA: (yw, cls, ml, a, b) => `AR(1), media (abaterea standard): Yule–Walker ${yw}, cele mai mici pătrate condiționate ${cls}, MLE ${ml}; valoarea asimptotică ${a}; deplasarea aproximativă −(1 + 3φ)/T = ${b}.`,
 esM: (cls, ml, a) => `MA(1), media (abaterea standard): cele mai mici pătrate condiționate ${cls}, MLE ${ml}; valoarea asimptotică ${a}.`,
 ic: (t, at, bt, au, bu, ao, bo, l) => `T = ${t}${l ? '' : ' (simulare în browser)'}: p = 2 ales de AIC în ${at} și de BIC în ${bt} dintre eșantioane; ordin prea mic: AIC ${au}, BIC ${bu}; ordin prea mare: AIC ${ao}, BIC ${bo}.`,
 lb: (m, a, x1, x2, y1, y2) => `m = ${m}, nivelul ${a}%: AR(1) respinge în ${x1}% cu ${m} grade de libertate și în ${x2}% cu m − 1; ARMA(1,1) în ${y1}% cu ${m} și în ${y2}% cu m − 2.`,
 fc: (lab, f1, f5, s5, sx) => `${lab}: prognoza la h = 1 este ${f1}, la h = 5 este ${f5} (eroarea standard ${s5}); σ<sub>X</sub> = ${sx}.`,
 gi1: (a, b, n, m, bd) => `${a} – ${b}: ${n} trimestre, media ${m}%, banda ±${bd}.`,
 gi2: (r, p1, p5) => `ACF la lagurile 1–4: ${r}. PACF: ${p1} la lagul 1, ${p5} la lagul 5.`,
 gi3: (q, p) => `Ljung–Box pentru serie: Q*(8) = ${q}, ${p}.`,
 gd1: (p, q, a, b) => `ARMA(${p},${q}): AIC ${a}, BIC ${b}.`,
 gd2: (q, df, p, pw, mp) => `Ljung–Box: Q*(8) = ${q}, cu ${df} grade de libertate ${p} (cu 8 grade de libertate ar fi ${pw}); cel mai mic p-value din grafic: ${mp}.`,
 gd3: (jb, jp, sk, ku, e, d, z, ps) => `Jarque–Bera ${jb}, ${jp}; asimetria ${sk}, kurtosis-ul ${ku}. Cel mai mare reziduu: ${e} în ${d} (${z} abateri standard). Ljung–Box pentru pătratele reziduurilor, m = 8: ${ps}.`,
 gf0: (o, l) => `Originea ${o}, intervale de ${l}%.`,
 gf: (lab, f1, f4, f8, c1, c8, rm, n) => `${lab}: prognoza ${f1} la h = 1, ${f4} la h = 4, ${f8} la h = 8; intervalul ${c1} la h = 1 și ${c8} la h = 8.` + (rm === null ? '' : ` Eroarea medie pătratică (RMSE) pe cele ${n} trimestre observate: ${rm} puncte procentuale.`),
 in1: (p, b, best, mb) => `AR(${p}): BIC ${b}; cel mai mic BIC, ${mb}, îl are AR(${best}).`,
 in2: (ph, s, im, mu) => `${ph}; suma coeficienților ${s}, cea mai mare rădăcină inversă ${im}; media estimată ${mu}%.`,
 in3: (l, f1, fH, d, c) => `Ultima valoare ${l}%; prognoza ${f1}% luna următoare și ${fH}% în ${d}, cu intervalul de 95% ${c}.`,
 in4: (q, df, p) => `Ljung–Box pe reziduuri: Q*(12) = ${q} cu ${df} grade de libertate, ${p}.`,
 rt: (lab, d0, d1, n, r1, ph, r2, lb, lp, ls, full) => `${lab}, ${d0} – ${d1} (${n} zile): ρ̂(1) = ${r1}, φ̂ = ${ph}${full ? '' : ' (OLS)'}, R² = ${r2}%; Ljung–Box Q*(10) al reziduurilor ${lb} (9 grade de libertate, ${lp}), al pătratelor reziduurilor ${ls}.`,
 ss1: (p, ph, r2, a, b, ba, bb) => `AR(${p}): φ̂ = (${ph}); R² = ${r2}% cu un an înainte; AIC ${a}, BIC ${b}. AIC alege p = ${ba}, BIC p = ${bb}.`,
 ss2: (per, d) => `Rădăcini complexe: perioada ${per} ani, modulul rădăcinii inverse ${d}.`,
 ss3: (a, b, per) => `Yule, 1749–1924, prin cele mai mici pătrate: φ̂ = (${a}; ${b}), perioada ${per} ani.`
};
""",
 'en': r"""window.CH2_TEXT = {
 play: 'Play', pause: 'Pause', and: ' and ', lt: '< ',
 q: s => s,
 m: d => ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'][+d.slice(5, 7) - 1] + ' ' + d.slice(0, 4),
 a1: (p, r1, r2, v, vt) => `φ = ${p}: ρ̂(1) = ${r1}, ρ̂(2) = ${r2}; sample variance ${v} (theoretical 1/(1 − φ²) = ${vt}).`,
 a2a: (a, b, z, m, st) => `φ = (${a}, ${b}): roots of the characteristic polynomial ${z}; moduli of the inverse roots ${m}: ${st ? 'stationary process' : 'non-stationary process'}.`,
 a2c: (per, d) => `Complex roots: the ACF is a wave with a period of ${per} lags, damped by ${d} per lag.`,
 a2r: r => `ρ(1) = φ₁/(1 − φ₂) = ${r}.`,
 m1: (th, rt, r1, p2, inv) => `θ = ${th}: ρ(1) = ${rt}, ρ̂(1) = ${r1}, φ̂₂₂ = ${p2}${inv ? '' : '; |θ| > 1: the model is not invertible'}.`,
 mr: (t, r, u, inv, unit) => `θ = ${t}: ρ(1) = ${r}.` + (u === null ? ' For θ = 0 the series is white noise.' : unit ? ' At |θ| = 1, θ and 1/θ coincide, and |ρ(1)| = 0.5 is the maximum.' : ` The same ρ(1) for θ = ${u}; the invertible representation is θ = ${inv}.`),
 ps: (lab, p, s) => `${lab}: ψ₁…ψ₅ = ${p}. ` + (s === null ? 'The AR part is not stationary: the ψ weights do not die out.' : `Σψⱼ² = ${s} (the variance of the process, in units of σ²).`),
 pt: (n, r1, t1, r2, t2, p2, q2, b) => `${n}: ρ̂(1) = ${r1} (theoretical ${t1}), ρ̂(2) = ${r2} (${t2}); φ̂₂₂ = ${p2} (${q2}); band ±${b}.`,
 es0: l => l ? 'The 400 samples of the lecture.' : 'Browser simulation: 400 new samples for the chosen parameters.',
 esA: (yw, cls, ml, a, b) => `AR(1), mean (standard deviation): Yule–Walker ${yw}, conditional least squares ${cls}, MLE ${ml}; asymptotic value ${a}; approximate bias −(1 + 3φ)/T = ${b}.`,
 esM: (cls, ml, a) => `MA(1), mean (standard deviation): conditional least squares ${cls}, MLE ${ml}; asymptotic value ${a}.`,
 ic: (t, at, bt, au, bu, ao, bo, l) => `T = ${t}${l ? '' : ' (browser simulation)'}: p = 2 chosen by AIC in ${at} and by BIC in ${bt} of the samples; order too small: AIC ${au}, BIC ${bu}; order too large: AIC ${ao}, BIC ${bo}.`,
 lb: (m, a, x1, x2, y1, y2) => `m = ${m}, level ${a}%: AR(1) rejects in ${x1}% with ${m} degrees of freedom and in ${x2}% with m − 1; ARMA(1,1) in ${y1}% with ${m} and in ${y2}% with m − 2.`,
 fc: (lab, f1, f5, s5, sx) => `${lab}: the forecast at h = 1 is ${f1}, at h = 5 it is ${f5} (standard error ${s5}); σ<sub>X</sub> = ${sx}.`,
 gi1: (a, b, n, m, bd) => `${a} – ${b}: ${n} quarters, mean ${m}%, band ±${bd}.`,
 gi2: (r, p1, p5) => `ACF at lags 1–4: ${r}. PACF: ${p1} at lag 1, ${p5} at lag 5.`,
 gi3: (q, p) => `Ljung–Box for the series: Q*(8) = ${q}, ${p}.`,
 gd1: (p, q, a, b) => `ARMA(${p},${q}): AIC ${a}, BIC ${b}.`,
 gd2: (q, df, p, pw, mp) => `Ljung–Box: Q*(8) = ${q}, with ${df} degrees of freedom ${p} (with 8 degrees of freedom it would be ${pw}); smallest p-value in the chart: ${mp}.`,
 gd3: (jb, jp, sk, ku, e, d, z, ps) => `Jarque–Bera ${jb}, ${jp}; skewness ${sk}, kurtosis ${ku}. Largest residual: ${e} in ${d} (${z} standard deviations). Ljung–Box for the squared residuals, m = 8: ${ps}.`,
 gf0: (o, l) => `Origin ${o}, ${l}% intervals.`,
 gf: (lab, f1, f4, f8, c1, c8, rm, n) => `${lab}: forecast ${f1} at h = 1, ${f4} at h = 4, ${f8} at h = 8; interval ${c1} at h = 1 and ${c8} at h = 8.` + (rm === null ? '' : ` Root mean squared error (RMSE) over the ${n} observed quarters: ${rm} percentage points.`),
 in1: (p, b, best, mb) => `AR(${p}): BIC ${b}; the smallest BIC, ${mb}, belongs to AR(${best}).`,
 in2: (ph, s, im, mu) => `${ph}; sum of the coefficients ${s}, largest inverse root ${im}; estimated mean ${mu}%.`,
 in3: (l, f1, fH, d, c) => `Last value ${l}%; forecast ${f1}% next month and ${fH}% in ${d}, with the 95% interval ${c}.`,
 in4: (q, df, p) => `Ljung–Box on the residuals: Q*(12) = ${q} with ${df} degrees of freedom, ${p}.`,
 rt: (lab, d0, d1, n, r1, ph, r2, lb, lp, ls, full) => `${lab}, ${d0} – ${d1} (${n} days): ρ̂(1) = ${r1}, φ̂ = ${ph}${full ? '' : ' (OLS)'}, R² = ${r2}%; Ljung–Box Q*(10) of the residuals ${lb} (9 degrees of freedom, ${lp}), of the squared residuals ${ls}.`,
 ss1: (p, ph, r2, a, b, ba, bb) => `AR(${p}): φ̂ = (${ph}); R² = ${r2}% one year ahead; AIC ${a}, BIC ${b}. AIC picks p = ${ba}, BIC p = ${bb}.`,
 ss2: (per, d) => `Complex roots: period ${per} years, modulus of the inverse root ${d}.`,
 ss3: (a, b, per) => `Yule, 1749–1924, by least squares: φ̂ = (${a}, ${b}), period ${per} years.`
};
""",
}


TEXT_ADD = {
 'en': r"""Object.assign(window.CH2_TEXT, { x: {
 sep: ', ', yes: 'yes', no: 'no', yours: 'your model', model: 'Model', roots: 'roots z', stat: 'stationary', rec: 'recursion', share: 'share of γ(0)/σ²',
 estim: 'Estimator', yw: 'Yule–Walker', cls: 'conditional least squares (OLS)', ml: 'exact ML', penA: 'AIC penalty 2k', penB: 'BIC penalty k ln T',
 rej: 'reject', norej: 'do not reject', pWrong: 'p with df = m', at5: 'decision at 5%', lo95: '95% lower', hi95: '95% upper', width: 'width',
 quarter: 'quarter', obs: 'observed', param: 'Parameter', est: 'Estimate', ci95: '95% interval', nobs: 'Observations',
 seOPG: 'se, OPG', tOPG: 't-ratio, OPG', seCl: 'se, classical (curvature)', tCl: 't-ratio, classical',
 lbRes: 'LB Q*(10) residuals, 9 df (p)', lbSq: 'LB Q*(10) squared residuals', kurt: 'Kurtosis of residuals',
 pname: p => ({ const: 'mean μ', sigma2: 'σ²' }[p] || p.replace(/^ar\.L(\d)$/, 'φ<sub>$1</sub>').replace(/^ma\.L(\d)$/, 'θ<sub>$1</sub>')),
 xa1: a => `Stationary? |φ| = ${a} < 1: yes.`,
 xa2: (c, p, mu) => `Mean: μ = c/(1 − φ) = ${c}/(1 − ${p}) = ${mu}; the intercept is ${c}, the mean is ${mu}.`,
 xa3: (s2, p2, g0) => `Variance: γ(0) = σ²/(1 − φ²) = ${s2}/(1 − ${p2}) = ${g0}.`,
 xa4: (h, p, r) => `ACF: ρ(${h}) = φ<sup>${h}</sup> = ${p}<sup>${h}</sup> = ${r}.`,
 xa5: (a, hl, neg) => `Half-life: ln 0.5/ln ${a} = ${hl} periods: after about ${hl} periods, half of a shock is gone${neg ? ' (for φ < 0 the formula uses |φ|, and the sign of the response alternates)' : ''}.`,
 xa5z: () => 'φ = 0: white noise around μ; a shock is gone after one period.',
 xa6: (l, p2) => `ψ weights: ${l}, …: a shock of 1 today moves X<sub>t+2</sub> by ${p2}.`,
 xaCap: (mu, g0) => `ψ weights and autocovariances for μ = ${mu}, γ(0) = ${g0}; the highlighted row is the lag h of the slider.`,
 xa: (p, mu, g0, h, r, hl) => `AR(1) with φ = ${p}: μ = ${mu}, γ(0) = ${g0}, ρ(${h}) = ${r}, half-life ${hl} periods.`,
 xr1: poly => `Characteristic polynomial: φ(z) = ${poly}.`,
 xr2: (a, b, d, cx) => `Discriminant: φ₁² + 4φ₂ = ${a}² + 4 · ${b} = ${d}${cx ? ' < 0: complex roots' : ' ≥ 0: real roots'}.`,
 xr3: (z, m) => `Roots: z = (−φ₁ ± √(φ₁² + 4φ₂))/(2φ₂) = ${z}; moduli |z| = ${m}.`,
 xr3b: (a, z) => `φ₂ = 0: a single root, z = 1/φ₁ = 1/${a} = ${z}.`,
 xr4: m => `Inverse roots λ = 1/z, the form shown by software: |λ| = ${m}.`,
 xr5: (s, y1, d, y2, b, y3) => `Triangle: φ₁ + φ₂ = ${s} < 1: ${y1}; φ₂ − φ₁ = ${d} < 1: ${y2}; |φ₂| = ${b} < 1: ${y3}.`,
 xr6: st => st ? 'All roots lie outside the unit circle: the process is stationary.' : 'At least one root lies on or inside the unit circle: the process is not stationary.',
 xr: (a, b, z, m, st) => `φ = (${a}, ${b}): roots ${z}, smallest modulus ${m}: ${st ? 'stationary' : 'not stationary'}.`,
 xt1: (th, s2, g0, g1) => `X<sub>t</sub> = ε<sub>t</sub> + θε<sub>t−1</sub> with θ = ${th}, σ² = ${s2}: γ(0) = σ²(1 + θ²) = ${g0}, γ(1) = θσ² = ${g1}.`,
 xt2: (t2, v2, g0, g1) => `The other representation, θ* = 1/θ = ${t2} and σ*² = θ²σ² = ${v2}: γ(0) = σ*²(1 + θ*²) = ${g0}, γ(1) = θ*σ*² = ${g1}, the same values.`,
 xt3: r => `Same mean, same autocovariances, so ρ(1) = θ/(1 + θ²) = ${r} for both: with Gaussian shocks the two models have the same distribution, and no amount of data separates them.`,
 xt4: (inv, w1, ninv, w2) => `Only the representation with |θ| < 1 is invertible: in ε<sub>t</sub> = Σ(−θ)<sup>j</sup>X<sub>t−j</sub>, θ = ${inv} gives the weights ${w1}, …, which die out; θ = ${ninv} gives ${w2}, …, which explode.`,
 xt4u: () => '|θ| = 1: θ and 1/θ coincide; the root lies on the unit circle and the model is not invertible.',
 xt5: (t, v) => `Rule: report the invertible version, θ = ${t}, σ² = ${v}.`,
 xt0: () => 'θ = 0: the series is white noise, γ(1) = 0.',
 xtCap: 'Weights of the AR(∞) representation ε<sub>t</sub> = Σ(−θ)<sup>j</sup>X<sub>t−j</sub> for the two values of θ.',
 xt: (th, t2, r, g0, g1) => `θ = ${th} and θ = ${t2} (with σ² rescaled) give the same ρ(1) = ${r}, γ(0) = ${g0}, γ(1) = ${g1}; only the value with |θ| < 1 is invertible.`,
 xp1: (p, t, s) => `ψ₀ = 1; ψ₁ = φ + θ = ${p} + ${t} = ${s}.`,
 xp2: (p, p1, p2, p3) => `ψ₂ = φψ₁ = ${p} · ${p1} = ${p2}; ψ₃ = ${p3}; in general ψ<sub>j</sub> = φ<sup>j−1</sup>(φ + θ) for j ≥ 1.`,
 xp3: (s, p2, tot) => `Σψ<sub>j</sub>² = 1 + (φ + θ)²/(1 − φ²) = 1 + ${s}²/(1 − ${p2}) = ${tot} = γ(0)/σ².`,
 xp4: sh => `ψ₁² alone is ${sh}% of γ(0)/σ².`,
 xpCF: () => 'φ + θ = 0: the common factor cancels, ψ<sub>j</sub> = 0 for j ≥ 1, and the process is white noise.',
 xpCap: tot => `ψ weights of the ARMA(1,1) and the cumulative sum of their squares; the limit is γ(0)/σ² = ${tot}.`,
 xp: (p, t, l, tot) => `ARMA(1,1), φ = ${p}, θ = ${t}: ψ₁…ψ₄ = ${l}; Σψ<sub>j</sub>² = ${tot}.`,
 xm1: (p, t, den, d2, g0) => `γ(0) = σ²(1 + 2φθ + θ²)/(1 − φ²) = ${den}/${d2} = ${g0}, with σ² = 1.`,
 xm2: (a, s, den, r1) => `ρ(1) = (1 + φθ)(φ + θ)/(1 + 2φθ + θ²) = ${a} · ${s}/${den} = ${r1}.`,
 xm3: (p, r2, r3) => `ρ(h) = φρ(h − 1) for h ≥ 2, with φ = ${p}: ρ(2) = ${r2}, ρ(3) = ${r3}.`,
 xm4: (r1, p, sg, t) => sg > 0 ? `ρ(1) = ${r1} > φ = ${p}: with θ = ${t} > 0 the ACF starts above the AR(1) value and then decays at the rate φ, like an AR(1).`
   : sg < 0 ? `ρ(1) = ${r1} < φ = ${p}: with θ = ${t} < 0 the ACF starts below the AR(1) value and then decays at the rate φ, like an AR(1).` : `θ = 0: the model is an AR(1), and ρ(1) = φ = ${p}.`,
 xmCF: (p, t) => `φ + θ = 0 (φ = ${p}, θ = ${t}): the common factor cancels, the process is white noise and ρ(h) = 0.`,
 xmCap: 'ACF of the ARMA(1,1) and of the AR(1) with the same φ; the ratio ρ(h)/φ<sup>h</sup> is constant: the MA part only moves the starting point.',
 xm: (p, t, g0, r1, r2) => `ARMA(1,1), φ = ${p}, θ = ${t}: γ(0) = ${g0}σ², ρ(1) = ${r1}, ρ(2) = ${r2}.`,
 idRows: [['White noise', 'no significant lag', 'no significant lag'], ['AR(p)', 'decays (geometric or damped wave)', '<b>cuts off</b> after lag p'], ['MA(q)', '<b>cuts off</b> after lag q', 'decays'],
   ['ARMA(p,q)', 'decays after lag q − p', 'decays after lag p − q'], ['Unit root (Ch. 3)', 'very slow, almost linear decay', 'φ̂<sub>11</sub> ≈ 1']],
 itCap1: n => `The table of the slide; the highlighted row is the family of ${n}.`,
 itCap2: (n, T, b) => `${n}: theoretical values and one simulated path of length T = ${T}; sample values outside ±1.96/√T = ±${b} in bold.`,
 it: (n, fam, a, p, b) => `${n}, row “${fam}”: the sample ACF lies outside ±${b} at lags ${a}, the sample PACF at lags ${p}.`,
 xy1: (d0, d1, n, r1, r2, g0) => `Annual growth of real GDP, ${d0}–${d1}, T = ${n}: ρ̂(1) = ${r1}, ρ̂(2) = ${r2}, γ̂(0) = ${g0}.`,
 xy2: (r1, det, detR) => `Determinant 1 − ρ̂₁² = 1 − ${r1}² = ${det} (with ρ̂₁ rounded to two decimals: ${detR}).`,
 xy3: (r1, a, det, p1) => `φ̂₁ = ρ̂₁(1 − ρ̂₂)/(1 − ρ̂₁²) = ${r1} · ${a}/${det} = ${p1}.`,
 xy4: (r2, r11, det, p2) => `φ̂₂ = (ρ̂₂ − ρ̂₁²)/(1 − ρ̂₁²) = (${r2} − ${r11})/${det} = ${p2}.`,
 xy5: (g0, f, s2) => `σ̂² = γ̂(0)(1 − φ̂₁ρ̂₁ − φ̂₂ρ̂₂) = ${g0} · ${f} = ${s2}.`,
 xy3a: r1 => `AR(1): φ̂ = ρ̂(1) = ${r1}.`,
 xy5a: (g0, det, s2) => `σ̂² = γ̂(0)(1 − ρ̂₁²) = ${g0} · ${det} = ${s2}.`,
 xyCap: n => `AR(1) and AR(2) on the same ${n} quarters. Standard errors: asymptotic for Yule–Walker, OLS for conditional least squares, outer product of gradients (OPG, the statsmodels default) for exact ML. The highlighted rows have the order chosen above.`,
 xyNote: 'Exact ML: the fits of the GDP case study (Section 9).', xyNote2: 'Exact ML is shown for the full sample of the case study only.',
 xy: (p, d1, e, s2) => `Yule–Walker, AR(${p}), sample up to ${d1}: ${e}, σ̂² = ${s2}.`,
 xl1: () => 'X₁ ~ N(μ, σ²/(1 − φ²)), then X<sub>t</sub> | X<sub>t−1</sub> ~ N(μ + φ(X<sub>t−1</sub> − μ), σ²); ln L is the sum of the log-densities, maximised over μ, φ and σ².',
 xl2: (n, ph, mu, s2, ll) => `Exact Gaussian MLE on T = ${n} quarters: φ̂ = ${ph}, μ̂ = ${mu}, σ̂² = ${s2}, ln L = ${ll}.`,
 xl3: (n, pc, s2, ll) => `Without the first factor (conditional least squares, ${n} observations): φ̂ = ${pc}, σ̂² = ${s2}, ln L = ${ll}.`,
 xl4: (c, se, ph, sa) => `Curvature at φ̂: −d² ln L/dφ² = ${c}, so se(φ̂) = 1/√${c} = ${se}; the asymptotic formula √((1 − φ̂²)/T) gives ${sa}.`,
 xl5: (ph, se, t, lo, hi) => `t-ratio φ̂/se = ${ph}/${se} = ${t}, far above 1.96; 95% interval [${lo}; ${hi}].`,
 xl6: s => `statsmodels reports se = ${s}: by default it uses the outer product of gradients (OPG), not the curvature.`,
 xl: (d, ph, se, sa, pc) => `AR(1) up to ${d}: exact MLE φ̂ = ${ph}, se from the curvature ${se} (asymptotic ${sa}); conditional least squares φ̂ = ${pc}.`,
 xi1: (nm, p, q, k, ll) => `${nm}: k = p + q + 2 = ${p} + ${q} + 2 = ${k} (μ, σ² and the ARMA coefficients); ln L = ${ll}.`,
 xi2: (m2, k, aic) => `AIC = −2 ln L + 2k = ${m2} + 2 · ${k} = ${aic}.`,
 xi3: (m2, k, lnT, bic) => `BIC = −2 ln L + k ln T = ${m2} + ${k} · ${lnT} = ${bic}.`,
 xi4: (aic, k, n, aicc) => `AICc = AIC + 2k(k + 1)/(T − k − 1) = ${aic} + 2 · ${k} · ${k + 1}/${n - k - 1} = ${aicc}.`,
 xi5: (nm, dl, dk, da, db) => `${nm} minus this model: Δln L = ${dl}, Δk = ${dk}, ΔAIC = ${da}, ΔBIC = ${db}; a negative Δ favours ${nm}.`,
 xiCap: n => `T = ${n} quarters, exact Gaussian MLE; columns: the chosen model and the models with the smallest AIC and BIC.`,
 xi: (nm, a, b, na, ma, nb, mb) => `${nm}: AIC ${a}, BIC ${b}. Smallest AIC: ${na} (${ma}); smallest BIC: ${nb} (${mb}).`,
 dt1: (m, Q, m2, fit, df, p, pw) => `Ljung–Box: Q*(${m}) = ${Q}; degrees of freedom m − p − q = ${m2} − ${fit} = ${df}, p = ${p}; with ${m} degrees of freedom, p would be ${pw}.`,
 dt2: (s, k) => `Skewness S = ${s}, kurtosis K = ${k} (0 and 3 for the Normal distribution).`,
 dt3: (n, s2, k4, jb, rej, p) => `JB = (T/6)(S² + (K − 3)²/4) = (${n}/6)(${s2} + ${k4}) = ${jb}${rej ? ' > 5.99' : ' ≤ 5.99'}: normality ${rej ? 'rejected' : 'not rejected'} at 5% (${p}).`,
 dt4: (q, p) => `Ljung–Box on the squared residuals, m = 8: Q* = ${q}, ${p}.`,
 dtCap: (nm, n) => `Residuals of ${nm}, T = ${n}; the highlighted row is the m of the slider. “p with df = m” uses the wrong reference distribution.`,
 dt: (nm, m, Q, df, p, jb, jp) => `${nm}: Q*(${m}) = ${Q} with ${df} degrees of freedom, ${p}; Jarque–Bera ${jb}, ${jp}.`,
 xf1: (c, p, mu, x, dev) => `μ = c/(1 − φ) = ${c}/(1 − ${p}) = ${mu}; last value X<sub>T</sub> = ${x}, deviation X<sub>T</sub> − μ = ${dev}.`,
 xf2: (mu, p, dev, f1, p2, f2, f10) => `X̂<sub>T+1</sub> = ${mu} + ${p} · ${dev} = ${f1}; X̂<sub>T+2</sub> = ${mu} + ${p2} · ${dev} = ${f2}; X̂<sub>T+10</sub> = ${f10}.`,
 xf3: (s2, v1, p, v2) => `Error variances: σ₁² = σ² = ${v1}; σ₂² = σ²(1 + φ²) = ${v2}.`,
 xf4: (c1, c2) => `95% intervals X̂ ± 1.96σ<sub>h</sub>: h = 1: ${c1}; h = 2: ${c2}.`,
 xf5: (mu, sd) => `Long run: the forecast tends to μ = ${mu} and σ<sub>h</sub> to √γ(0) = ${sd}.`,
 xf: (f1, c1, f2, c2, f10, mu) => `X̂<sub>T+1</sub> = ${f1}, interval ${c1}; X̂<sub>T+2</sub> = ${f2}, interval ${c2}; X̂<sub>T+10</sub> = ${f10}; the forecasts return to μ = ${mu}.`,
 ggCap: n => `All ARMA(p, q), p, q ≤ 3, exact Gaussian MLE on the same ${n} quarters; LB = Ljung–Box Q*(8) on the residuals with 8 − p − q degrees of freedom.`,
 gg: (a, ma, b, mb, d, cr) => `AIC chooses ${a} (${ma}), BIC chooses ${b} (${mb}); the AIC of ARMA(0,3) exceeds the smallest AIC by ${d}. The heat map shows Δ${cr}.`,
 mqCap: nm => `${nm}: exact Gaussian MLE; standard errors by the outer product of gradients (OPG), the statsmodels default.`,
 mqNote: (mm, mi, am, as) => `Moduli of the MA roots: ${mm} (invertible: ${mi}); of the AR roots: ${am} (stationary: ${as}).`,
 mq1: (nm, m, se, lo, hi) => `${nm}: mean growth μ̂ = ${m}% per year with se ${se}: 95% interval [${lo}; ${hi}].`,
 mq2: (s, s2) => `σ̂ = √${s2} = ${s}.`,
 mq3: (th, mn, inv, big) => `θ̂ = (${th}); smallest modulus of the roots of θ(z): ${mn}${inv ? ' > 1: invertible' : ' ≤ 1: not invertible'}${big ? '; a coefficient |θ̂<sub>j</sub>| > 1 does not decide on its own: invertibility is read from the roots' : ''}.`,
 mq: (nm, l) => `${nm}: ${l}.`,
 ft1: (se, z, ci) => `MA(3), h = 1: σ₁ = σ̂ = ${se}; interval X̂ ± ${z}σ₁ = ${ci}.`,
 ft2: (f4, s4, s8) => `From h = 4 the MA(3) forecast is the estimated mean, ${f4}, and σ<sub>h</sub> stays at ${s4} (h = 8: ${s8}).`,
 ft3: (s2, th2, s, sh, mu, z, ci) => `σ<sub>4</sub>² = σ̂²(1 + θ̂₁² + θ̂₂² + θ̂₃²) = ${s2} · (1 + ${th2}) = ${s2} · ${s}, so σ<sub>4</sub> = ${sh}; interval ${mu} ± ${z} · ${sh} = ${ci}.`,
 ft4: (s1, s8) => `AR(1): σ<sub>h</sub> grows from ${s1} at h = 1 to ${s8} at h = 8, towards the standard deviation of the fitted process.`,
 ftCap: o => `Forecasts from ${o}; “observed” is the value recorded later.`,
 ft: (o, lv, a1, a8, b1, b8, ra, rb, n) => `Origin ${o}, ${lv}% intervals: width ${a1} at h = 1 and ${a8} at h = 8 for MA(3); ${b1} and ${b8} for AR(1).` + (n ? ` Root mean squared error (RMSE) over the ${n} observed quarters: MA(3) ${ra}, AR(1) ${rb}.` : ''),
 rx1: (lab, ph, so, to, sc, tc, n, sa) => `${lab}: φ̂ = ${ph}; OPG se ${so}, t = ${to}; classical se ${sc}, t = ${tc}; √((1 − φ̂²)/T) with T = ${n}: ${sa}.`,
 rxCap: (a, b) => `AR(1) for daily log returns (%), ${a}–${b}; OPG = outer product of gradients.`,
 rxNote: full => full ? 'Full sample: φ̂ by exact ML; the slide reports the OPG standard error (statsmodels default); the classical row is the standard error from the curvature of ln L (observed information).'
   : 'Sub-sample: φ̂ by conditional least squares; OPG from the scores of the conditional likelihood; classical = OLS standard error.',
 rx: (lab, n, ph, t, r2) => `${lab}: ${n} days, φ̂ = ${ph}, classical t = ${t}, R² = ${r2}%.`
} });
""",
 'ro': r"""Object.assign(window.CH2_TEXT, { x: {
 sep: '; ', yes: 'da', no: 'nu', yours: 'modelul ales', model: 'Modelul', roots: 'rădăcinile z', stat: 'staționar', rec: 'recurența', share: 'ponderea în γ(0)/σ²',
 estim: 'Estimatorul', yw: 'Yule–Walker', cls: 'cele mai mici pătrate condiționate (OLS)', ml: 'MLE exact', penA: 'penalizarea AIC 2k', penB: 'penalizarea BIC k ln T',
 rej: 'respinge', norej: 'nu respinge', pWrong: 'p cu df = m', at5: 'decizia la 5%', lo95: 'limita inferioară 95%', hi95: 'limita superioară 95%', width: 'lățimea',
 quarter: 'trimestrul', obs: 'observat', param: 'Parametrul', est: 'Estimația', ci95: 'intervalul de 95%', nobs: 'Observații',
 seOPG: 'se, OPG', tOPG: 'raportul t, OPG', seCl: 'se, clasic (curbura)', tCl: 'raportul t, clasic',
 lbRes: 'LB Q*(10) reziduuri, 9 g.l. (p)', lbSq: 'LB Q*(10) pătratele reziduurilor', kurt: 'Kurtosis-ul reziduurilor',
 pname: p => ({ const: 'media μ', sigma2: 'σ²' }[p] || p.replace(/^ar\.L(\d)$/, 'φ<sub>$1</sub>').replace(/^ma\.L(\d)$/, 'θ<sub>$1</sub>')),
 xa1: a => `Este staționar? |φ| = ${a} < 1: da.`,
 xa2: (c, p, mu) => `Media: μ = c/(1 − φ) = ${c}/(1 − ${p}) = ${mu}; termenul liber este ${c}, media este ${mu}.`,
 xa3: (s2, p2, g0) => `Varianța: γ(0) = σ²/(1 − φ²) = ${s2}/(1 − ${p2}) = ${g0}.`,
 xa4: (h, p, r) => `ACF: ρ(${h}) = φ<sup>${h}</sup> = ${p}<sup>${h}</sup> = ${r}.`,
 xa5: (a, hl, neg) => `Timpul de înjumătățire: ln 0,5/ln ${a} = ${hl} perioade: după aproximativ ${hl} perioade, jumătate din șoc a dispărut${neg ? ' (pentru φ < 0 formula folosește |φ|, iar semnul răspunsului alternează)' : ''}.`,
 xa5z: () => 'φ = 0: zgomot alb în jurul lui μ; un șoc dispare după o perioadă.',
 xa6: (l, p2) => `Ponderile ψ: ${l}, …: un șoc de 1 astăzi îl mută pe X<sub>t+2</sub> cu ${p2}.`,
 xaCap: (mu, g0) => `Ponderile ψ și autocovarianțele pentru μ = ${mu}, γ(0) = ${g0}; rîndul evidențiat este lagul h ales.`,
 xa: (p, mu, g0, h, r, hl) => `AR(1) cu φ = ${p}: μ = ${mu}, γ(0) = ${g0}, ρ(${h}) = ${r}, timpul de înjumătățire ${hl} perioade.`,
 xr1: poly => `Polinomul caracteristic: φ(z) = ${poly}.`,
 xr2: (a, b, d, cx) => `Discriminantul: φ₁² + 4φ₂ = ${a}² + 4 · ${b} = ${d}${cx ? ' < 0: rădăcini complexe' : ' ≥ 0: rădăcini reale'}.`,
 xr3: (z, m) => `Rădăcinile: z = (−φ₁ ± √(φ₁² + 4φ₂))/(2φ₂) = ${z}; modulele |z| = ${m}.`,
 xr3b: (a, z) => `φ₂ = 0: o singură rădăcină, z = 1/φ₁ = 1/${a} = ${z}.`,
 xr4: m => `Rădăcinile inverse λ = 1/z, forma afișată de programe: |λ| = ${m}.`,
 xr5: (s, y1, d, y2, b, y3) => `Triunghiul: φ₁ + φ₂ = ${s} < 1: ${y1}; φ₂ − φ₁ = ${d} < 1: ${y2}; |φ₂| = ${b} < 1: ${y3}.`,
 xr6: st => st ? 'Toate rădăcinile sînt în afara cercului unitate: procesul este staționar.' : 'Cel puțin o rădăcină este pe cercul unitate sau în interiorul lui: procesul nu este staționar.',
 xr: (a, b, z, m, st) => `φ = (${a}; ${b}): rădăcinile ${z}, cel mai mic modul ${m}: ${st ? 'staționar' : 'nestaționar'}.`,
 xt1: (th, s2, g0, g1) => `X<sub>t</sub> = ε<sub>t</sub> + θε<sub>t−1</sub> cu θ = ${th}, σ² = ${s2}: γ(0) = σ²(1 + θ²) = ${g0}, γ(1) = θσ² = ${g1}.`,
 xt2: (t2, v2, g0, g1) => `Cealaltă reprezentare, θ* = 1/θ = ${t2} și σ*² = θ²σ² = ${v2}: γ(0) = σ*²(1 + θ*²) = ${g0}, γ(1) = θ*σ*² = ${g1}, aceleași valori.`,
 xt3: r => `Aceeași medie, aceleași autocovarianțe, deci ρ(1) = θ/(1 + θ²) = ${r} pentru ambele: cu șocuri gaussiene cele două modele au aceeași distribuție și nicio cantitate de date nu le separă.`,
 xt4: (inv, w1, ninv, w2) => `Numai reprezentarea cu |θ| < 1 este invertibilă: în ε<sub>t</sub> = Σ(−θ)<sup>j</sup>X<sub>t−j</sub>, θ = ${inv} dă ponderile ${w1}, …, care se sting; θ = ${ninv} dă ${w2}, …, care explodează.`,
 xt4u: () => '|θ| = 1: θ și 1/θ coincid; rădăcina este pe cercul unitate și modelul nu este invertibil.',
 xt5: (t, v) => `Regula: se raportează versiunea invertibilă, θ = ${t}, σ² = ${v}.`,
 xt0: () => 'θ = 0: seria este zgomot alb, γ(1) = 0.',
 xtCap: 'Ponderile reprezentării AR(∞) ε<sub>t</sub> = Σ(−θ)<sup>j</sup>X<sub>t−j</sub> pentru cele două valori ale lui θ.',
 xt: (th, t2, r, g0, g1) => `θ = ${th} și θ = ${t2} (cu σ² rescalat) dau același ρ(1) = ${r}, γ(0) = ${g0}, γ(1) = ${g1}; numai valoarea cu |θ| < 1 este invertibilă.`,
 xp1: (p, t, s) => `ψ₀ = 1; ψ₁ = φ + θ = ${p} + ${t} = ${s}.`,
 xp2: (p, p1, p2, p3) => `ψ₂ = φψ₁ = ${p} · ${p1} = ${p2}; ψ₃ = ${p3}; în general ψ<sub>j</sub> = φ<sup>j−1</sup>(φ + θ) pentru j ≥ 1.`,
 xp3: (s, p2, tot) => `Σψ<sub>j</sub>² = 1 + (φ + θ)²/(1 − φ²) = 1 + ${s}²/(1 − ${p2}) = ${tot} = γ(0)/σ².`,
 xp4: sh => `Numai ψ₁² reprezintă ${sh}% din γ(0)/σ².`,
 xpCF: () => 'φ + θ = 0: factorul comun se simplifică, ψ<sub>j</sub> = 0 pentru j ≥ 1, iar procesul este zgomot alb.',
 xpCap: tot => `Ponderile ψ ale ARMA(1,1) și suma cumulată a pătratelor lor; limita este γ(0)/σ² = ${tot}.`,
 xp: (p, t, l, tot) => `ARMA(1,1), φ = ${p}, θ = ${t}: ψ₁…ψ₄ = ${l}; Σψ<sub>j</sub>² = ${tot}.`,
 xm1: (p, t, den, d2, g0) => `γ(0) = σ²(1 + 2φθ + θ²)/(1 − φ²) = ${den}/${d2} = ${g0}, cu σ² = 1.`,
 xm2: (a, s, den, r1) => `ρ(1) = (1 + φθ)(φ + θ)/(1 + 2φθ + θ²) = ${a} · ${s}/${den} = ${r1}.`,
 xm3: (p, r2, r3) => `ρ(h) = φρ(h − 1) pentru h ≥ 2, cu φ = ${p}: ρ(2) = ${r2}, ρ(3) = ${r3}.`,
 xm4: (r1, p, sg, t) => sg > 0 ? `ρ(1) = ${r1} > φ = ${p}: cu θ = ${t} > 0, ACF pornește peste valoarea AR(1) și apoi descrește cu rata φ, ca la un AR(1).`
   : sg < 0 ? `ρ(1) = ${r1} < φ = ${p}: cu θ = ${t} < 0, ACF pornește sub valoarea AR(1) și apoi descrește cu rata φ, ca la un AR(1).` : `θ = 0: modelul este un AR(1), iar ρ(1) = φ = ${p}.`,
 xmCF: (p, t) => `φ + θ = 0 (φ = ${p}, θ = ${t}): factorul comun se simplifică, procesul este zgomot alb și ρ(h) = 0.`,
 xmCap: 'ACF a procesului ARMA(1,1) și a AR(1) cu același φ; raportul ρ(h)/φ<sup>h</sup> este constant: partea MA mută doar punctul de pornire.',
 xm: (p, t, g0, r1, r2) => `ARMA(1,1), φ = ${p}, θ = ${t}: γ(0) = ${g0}σ², ρ(1) = ${r1}, ρ(2) = ${r2}.`,
 idRows: [['Zgomot alb', 'niciun lag semnificativ', 'niciun lag semnificativ'], ['AR(p)', 'descrește (geometric sau undă amortizată)', '<b>se anulează</b> după lagul p'], ['MA(q)', '<b>se anulează</b> după lagul q', 'descrește'],
   ['ARMA(p,q)', 'descrește după lagul q − p', 'descrește după lagul p − q'], ['Rădăcină unitară (cap. 3)', 'descreștere foarte lentă, aproape liniară', 'φ̂<sub>11</sub> ≈ 1']],
 itCap1: n => `Tabelul din slide; rîndul evidențiat este familia lui ${n}.`,
 itCap2: (n, T, b) => `${n}: valorile teoretice și o traiectorie simulată de lungime T = ${T}; valorile de selecție în afara benzii ±1,96/√T = ±${b} sînt îngroșate.`,
 it: (n, fam, a, p, b) => `${n}, rîndul „${fam}”: ACF de selecție iese din banda ±${b} la lagurile ${a}, PACF de selecție la lagurile ${p}.`,
 xy1: (d0, d1, n, r1, r2, g0) => `Creșterea anuală a PIB-ului real, ${d0}–${d1}, T = ${n}: ρ̂(1) = ${r1}, ρ̂(2) = ${r2}, γ̂(0) = ${g0}.`,
 xy2: (r1, det, detR) => `Determinantul 1 − ρ̂₁² = 1 − ${r1}² = ${det} (cu ρ̂₁ rotunjit la două zecimale: ${detR}).`,
 xy3: (r1, a, det, p1) => `φ̂₁ = ρ̂₁(1 − ρ̂₂)/(1 − ρ̂₁²) = ${r1} · ${a}/${det} = ${p1}.`,
 xy4: (r2, r11, det, p2) => `φ̂₂ = (ρ̂₂ − ρ̂₁²)/(1 − ρ̂₁²) = (${r2} − ${r11})/${det} = ${p2}.`,
 xy5: (g0, f, s2) => `σ̂² = γ̂(0)(1 − φ̂₁ρ̂₁ − φ̂₂ρ̂₂) = ${g0} · ${f} = ${s2}.`,
 xy3a: r1 => `AR(1): φ̂ = ρ̂(1) = ${r1}.`,
 xy5a: (g0, det, s2) => `σ̂² = γ̂(0)(1 − ρ̂₁²) = ${g0} · ${det} = ${s2}.`,
 xyCap: n => `AR(1) și AR(2) pe aceleași ${n} trimestre. Erorile standard: asimptotice pentru Yule–Walker, OLS pentru cele mai mici pătrate condiționate, produsul exterior al gradienților (outer product of gradients, OPG, implicit în statsmodels) pentru MLE exact. Rîndurile evidențiate au ordinul ales mai sus.`,
 xyNote: 'MLE exact: estimările din studiul de caz al PIB-ului (secțiunea 9).', xyNote2: 'MLE exact apare numai pentru eșantionul complet al studiului de caz.',
 xy: (p, d1, e, s2) => `Yule–Walker, AR(${p}), eșantionul pînă în ${d1}: ${e}, σ̂² = ${s2}.`,
 xl1: () => 'X₁ ~ N(μ, σ²/(1 − φ²)), apoi X<sub>t</sub> | X<sub>t−1</sub> ~ N(μ + φ(X<sub>t−1</sub> − μ), σ²); ln L este suma log-densităților, maximizată după μ, φ și σ².',
 xl2: (n, ph, mu, s2, ll) => `MLE gaussian exact pe T = ${n} trimestre: φ̂ = ${ph}, μ̂ = ${mu}, σ̂² = ${s2}, ln L = ${ll}.`,
 xl3: (n, pc, s2, ll) => `Fără primul factor (cele mai mici pătrate condiționate, ${n} observații): φ̂ = ${pc}, σ̂² = ${s2}, ln L = ${ll}.`,
 xl4: (c, se, ph, sa) => `Curbura în φ̂: −d² ln L/dφ² = ${c}, deci se(φ̂) = 1/√${c} = ${se}; formula asimptotică √((1 − φ̂²)/T) dă ${sa}.`,
 xl5: (ph, se, t, lo, hi) => `Raportul t φ̂/se = ${ph}/${se} = ${t}, mult peste 1,96; intervalul de 95% [${lo}; ${hi}].`,
 xl6: s => `statsmodels raportează se = ${s}: implicit folosește produsul exterior al gradienților (OPG), nu curbura.`,
 xl: (d, ph, se, sa, pc) => `AR(1) pînă în ${d}: MLE exact φ̂ = ${ph}, se din curbură ${se} (asimptotic ${sa}); cele mai mici pătrate condiționate φ̂ = ${pc}.`,
 xi1: (nm, p, q, k, ll) => `${nm}: k = p + q + 2 = ${p} + ${q} + 2 = ${k} (μ, σ² și coeficienții ARMA); ln L = ${ll}.`,
 xi2: (m2, k, aic) => `AIC = −2 ln L + 2k = ${m2} + 2 · ${k} = ${aic}.`,
 xi3: (m2, k, lnT, bic) => `BIC = −2 ln L + k ln T = ${m2} + ${k} · ${lnT} = ${bic}.`,
 xi4: (aic, k, n, aicc) => `AICc = AIC + 2k(k + 1)/(T − k − 1) = ${aic} + 2 · ${k} · ${k + 1}/${n - k - 1} = ${aicc}.`,
 xi5: (nm, dl, dk, da, db) => `${nm} minus acest model: Δln L = ${dl}, Δk = ${dk}, ΔAIC = ${da}, ΔBIC = ${db}; un Δ negativ îl favorizează pe ${nm}.`,
 xiCap: n => `T = ${n} trimestre, MLE gaussian exact; coloanele: modelul ales și modelele cu cel mai mic AIC și BIC.`,
 xi: (nm, a, b, na, ma, nb, mb) => `${nm}: AIC ${a}, BIC ${b}. Cel mai mic AIC: ${na} (${ma}); cel mai mic BIC: ${nb} (${mb}).`,
 dt1: (m, Q, m2, fit, df, p, pw) => `Ljung–Box: Q*(${m}) = ${Q}; gradele de libertate m − p − q = ${m2} − ${fit} = ${df}, p = ${p}; cu ${m} grade de libertate p ar fi ${pw}.`,
 dt2: (s, k) => `Asimetria S = ${s}, kurtosis-ul K = ${k} (0 și 3 pentru distribuția Normală).`,
 dt3: (n, s2, k4, jb, rej, p) => `JB = (T/6)(S² + (K − 3)²/4) = (${n}/6)(${s2} + ${k4}) = ${jb}${rej ? ' > 5,99' : ' ≤ 5,99'}: normalitatea ${rej ? 'este respinsă' : 'nu este respinsă'} la 5% (${p}).`,
 dt4: (q, p) => `Ljung–Box pentru pătratele reziduurilor, m = 8: Q* = ${q}, ${p}.`,
 dtCap: (nm, n) => `Reziduurile modelului ${nm}, T = ${n}; rîndul evidențiat este m ales. „p cu df = m” folosește o distribuție de referință greșită.`,
 dt: (nm, m, Q, df, p, jb, jp) => `${nm}: Q*(${m}) = ${Q} cu ${df} grade de libertate, ${p}; Jarque–Bera ${jb}, ${jp}.`,
 xf1: (c, p, mu, x, dev) => `μ = c/(1 − φ) = ${c}/(1 − ${p}) = ${mu}; ultima valoare X<sub>T</sub> = ${x}, abaterea X<sub>T</sub> − μ = ${dev}.`,
 xf2: (mu, p, dev, f1, p2, f2, f10) => `X̂<sub>T+1</sub> = ${mu} + ${p} · ${dev} = ${f1}; X̂<sub>T+2</sub> = ${mu} + ${p2} · ${dev} = ${f2}; X̂<sub>T+10</sub> = ${f10}.`,
 xf3: (s2, v1, p, v2) => `Varianțele erorilor: σ₁² = σ² = ${v1}; σ₂² = σ²(1 + φ²) = ${v2}.`,
 xf4: (c1, c2) => `Intervalele de 95% X̂ ± 1,96σ<sub>h</sub>: h = 1: ${c1}; h = 2: ${c2}.`,
 xf5: (mu, sd) => `Pe termen lung: prognoza tinde spre μ = ${mu}, iar σ<sub>h</sub> spre √γ(0) = ${sd}.`,
 xf: (f1, c1, f2, c2, f10, mu) => `X̂<sub>T+1</sub> = ${f1}, intervalul ${c1}; X̂<sub>T+2</sub> = ${f2}, intervalul ${c2}; X̂<sub>T+10</sub> = ${f10}; prognozele revin la μ = ${mu}.`,
 ggCap: n => `Toate modelele ARMA(p, q), p, q ≤ 3, MLE gaussian exact pe aceleași ${n} trimestre; LB = Ljung–Box Q*(8) pe reziduuri, cu 8 − p − q grade de libertate.`,
 gg: (a, ma, b, mb, d, cr) => `AIC alege ${a} (${ma}), BIC alege ${b} (${mb}); AIC-ul modelului ARMA(0,3) depășește cel mai mic AIC cu ${d}. Harta arată Δ${cr}.`,
 mqCap: nm => `${nm}: MLE gaussian exact; erorile standard prin produsul exterior al gradienților (OPG), varianta implicită din statsmodels.`,
 mqNote: (mm, mi, am, as) => `Modulele rădăcinilor MA: ${mm} (invertibil: ${mi}); ale rădăcinilor AR: ${am} (staționar: ${as}).`,
 mq1: (nm, m, se, lo, hi) => `${nm}: creșterea medie μ̂ = ${m}% pe an, cu se ${se}: intervalul de 95% [${lo}; ${hi}].`,
 mq2: (s, s2) => `σ̂ = √${s2} = ${s}.`,
 mq3: (th, mn, inv, big) => `θ̂ = (${th}); cel mai mic modul al rădăcinilor lui θ(z): ${mn}${inv ? ' > 1: invertibil' : ' ≤ 1: neinvertibil'}${big ? '; un coeficient |θ̂<sub>j</sub>| > 1 nu decide singur: invertibilitatea se citește din rădăcini' : ''}.`,
 mq: (nm, l) => `${nm}: ${l}.`,
 ft1: (se, z, ci) => `MA(3), h = 1: σ₁ = σ̂ = ${se}; intervalul X̂ ± ${z}σ₁ = ${ci}.`,
 ft2: (f4, s4, s8) => `De la h = 4, prognoza MA(3) este media estimată, ${f4}, iar σ<sub>h</sub> rămîne ${s4} (h = 8: ${s8}).`,
 ft3: (s2, th2, s, sh, mu, z, ci) => `σ<sub>4</sub>² = σ̂²(1 + θ̂₁² + θ̂₂² + θ̂₃²) = ${s2} · (1 + ${th2}) = ${s2} · ${s}, deci σ<sub>4</sub> = ${sh}; intervalul ${mu} ± ${z} · ${sh} = ${ci}.`,
 ft4: (s1, s8) => `AR(1): σ<sub>h</sub> crește de la ${s1} la h = 1 la ${s8} la h = 8, spre abaterea standard a procesului estimat.`,
 ftCap: o => `Prognoze din ${o}; „observat” este valoarea înregistrată ulterior.`,
 ft: (o, lv, a1, a8, b1, b8, ra, rb, n) => `Originea ${o}, intervale de ${lv}%: lățimea ${a1} la h = 1 și ${a8} la h = 8 pentru MA(3); ${b1} și ${b8} pentru AR(1).` + (n ? ` Rădăcina erorii medii pătratice (RMSE) pe cele ${n} trimestre observate: MA(3) ${ra}, AR(1) ${rb}.` : ''),
 rx1: (lab, ph, so, to, sc, tc, n, sa) => `${lab}: φ̂ = ${ph}; se OPG ${so}, t = ${to}; se clasic ${sc}, t = ${tc}; √((1 − φ̂²)/T) cu T = ${n}: ${sa}.`,
 rxCap: (a, b) => `AR(1) pentru randamentele logaritmice zilnice (%), ${a}–${b}; OPG = produsul exterior al gradienților (outer product of gradients).`,
 rxNote: full => full ? 'Eșantionul complet: φ̂ prin MLE exact; slide-ul raportează eroarea standard OPG (implicită în statsmodels); rîndul clasic este eroarea standard din curbura lui ln L (informația observată).'
   : 'Subeșantion: φ̂ prin cele mai mici pătrate condiționate; OPG din scorurile verosimilității condiționate; clasic = eroarea standard OLS.',
 rx: (lab, n, ph, t, r2) => `${lab}: ${n} zile, φ̂ = ${ph}, raportul t clasic ${t}, R² = ${r2}%.`
} });
""",
}
for _lg in TEXT_ADD:
    TEXT_JS[_lg] += TEXT_ADD[_lg]

# Questions of the EXTRA blocks
QX = {
 'ro': {
  'wx-ar1': ('În X<sub>t</sub> = 2 + 0,8X<sub>t−1</sub> + ε<sub>t</sub>, termenul liber este 2. Este 2 media procesului?',
             'Nu: μ = c/(1 − φ) = 2/0,2 = 10. Varianța este γ(0) = 1/(1 − 0,64) = 2,78, ρ(3) = 0,512, iar timpul de înjumătățire este 3,11 perioade. Cu același c și φ = 0,9, media urcă la 20, γ(0) la 5,26, iar timpul de înjumătățire la 6,58 perioade.'),
  'ar-roots': ('Exemplul B are φ₁ = 1,0. Este procesul staționar?',
               'Da. Discriminantul φ₁² + 4φ₂ = −1,4 este negativ, rădăcinile sînt complexe, 0,833 ± 0,986i, cu modulul 1,291 > 1 (rădăcinile inverse au modulul 0,775). Condițiile triunghiului sînt îndeplinite: 0,4 < 1, −1,6 < 1, 0,6 < 1. '
               'Invers, în exemplul C ambii coeficienți sînt sub 1, dar rădăcina 0,936 este în interiorul cercului unitate.'),
  'wx-theta': ('Butonul ▶ mută θ de la −3 la 3. Pentru ce valori este MA(1) invertibil și ce se întîmplă cu ponderile AR(∞) la θ = 2?',
               'Pentru |θ| < 1. La θ = 2 ponderile (−θ)<sup>j</sup> sînt 1; −2; 4; −8, … și ajung la 256 la j = 8: șocurile nu pot fi reconstituite din trecutul seriei. Reprezentarea pereche, θ = 0,5 cu σ² = 4, are ponderile 1; −0,5; 0,25, … și aceleași γ(0) = 5, γ(1) = 2.'),
  'wx-psi': ('Pentru φ = 0,7 și θ = 0,4, cîte ponderi ψ acoperă cel puțin 95% din γ(0)/σ² = 3,37?',
             'Cinci, ψ₀…ψ₄: suma cumulată a pătratelor ajunge la 3,236, adică 95,9%; cu ψ₀…ψ₃ ajunge la 91,7%. Numai ψ₁² = 1,21 reprezintă 35,9%, mai mult decît șocul curent (ψ₀² = 1, 29,7%), pentru că ψ₁ = φ + θ = 1,10 > 1.'),
  'wx-arma11': ('Pentru φ = 0,7 și θ = 0,4, ACF pornește peste φ sau sub φ? Dar pentru θ = −0,4?',
                'Peste: ρ(1) = 0,819 > 0,7. Diferența ρ(1) − φ = θ(1 − φ²)/(1 + 2φθ + θ²) are semnul lui θ, deci cu θ = −0,4 ACF pornește sub φ: ρ(1) = 0,360, ρ(2) = 0,252, iar γ(0) scade de la 3,37 la 1,18. '
                'În ambele cazuri raportul ρ(h)/φ<sup>h</sup> este constant (1,169, respectiv 0,514): după lagul 1 ACF descrește ca la un AR(1).'),
  'ident-table': ('Pentru AR(2), cu T = 500, PACF de selecție iese din bandă numai la lagurile 1 și 2. Rămîne tiparul cu T = 100?',
                  'Da, pentru PACF: banda se lărgește la ±0,196 și PACF iese tot la lagurile 1 și 2. ACF iese la lagurile 1, 3, 4 și 5: descreșterea în formă de undă amortizată se vede și în eșantionul scurt. '
                  'Pentru MA(2), cu T = 500, ACF iese la lagurile 1 și 2, iar PACF la lagurile 1–4: tabelul se citește invers.'),
  'wx-yw': ('Slide-ul scrie 1 − ρ̂₁² = 0,620, dar 1 − 0,62² = 0,616. Care valoare este corectă?',
            '0,620, calculată cu ρ̂₁ = 0,6167 nerotunjit; 0,616 vine din rotunjirea lui ρ̂₁ la 0,62. Cu valorile nerotunjite, φ̂₁ = 0,600, φ̂₂ = 0,027 și σ̂² = 11,62, ca în slide (γ̂(0) = 18,77, cu împărțire la T); cu 0,62 și 0,40 rotunjite ar ieși φ̂₁ = 0,604 și φ̂₂ = 0,025. '
            'Pe eșantionul care se oprește în T4 2019, φ̂₂ = 0,102.'),
  'ml-lik': ('Pentru AR(1) pe creșterea PIB-ului, statsmodels raportează se(φ̂) = 0,085, iar curbura log-verosimilității dă 0,076. Care valoare corespunde formulei √((1 − φ̂²)/T)?',
             'Curbura: 0,076, față de 0,075 din formula asimptotică. Valoarea 0,085 este eroarea standard OPG (produsul exterior al gradienților), implicită în statsmodels. Concluzia nu se schimbă: φ̂ = 0,630, raportul t este 8,3 cu curbura, intervalul de 95% [0,481; 0,779].'),
  'wx-ic': ('ARMA(1,3) are ln L mai mare decît MA(3) cu 1,54. De ce îl preferă BIC totuși pe MA(3)?',
            'Parametrul în plus costă 2 în AIC și ln 106 = 4,66 în BIC, iar −2 ln L scade doar cu 3,07. AIC scade deci cu 1,07 (548,97 față de 550,04), BIC crește cu 1,59 (564,95 față de 563,36). AICc, cu corecția pentru T mic, dă 549,82 și 550,64.'),
  'diag-table': ('Pentru AR(1), Ljung–Box cu m = 8 dă Q*(8) = 16,23. Respingem la 5% ipoteza că reziduurile sînt necorelate?',
                 'Da: cu 8 − 1 = 7 grade de libertate p = 0,023; și cu 8 grade de libertate (greșit) p = 0,039. La MA(3), Q*(8) = 5,35 cu 5 grade de libertate dă p = 0,375, deci nu respingem. '
                 'Normalitatea este respinsă pentru ambele: Jarque–Bera 25,9 pentru AR(1) și 39,0 pentru MA(3), peste 5,99.'),
  'wx-fc': ('Pentru X<sub>t</sub> = 2 + 0,8X<sub>t−1</sub> + ε<sub>t</sub>, spre ce lățime tinde intervalul de prognoză de 95%?',
            'Spre 2 × 1,96 × √γ(0) = 2 × 1,96 × 1,67 = 6,53. Lățimea este 3,92 la h = 1, 5,02 la h = 2 și 6,50 la h = 10: incertitudinea crește repede la primele orizonturi, apoi se stabilizează la cea necondiționată.'),
  'gdp-grid': ('Pe toată grila p, q ≤ 3, ce model are cel mai mic AICc și ce diferență are față de MA(3)?',
               'ARMA(1,3), cu AICc 549,82 față de 550,64 pentru MA(3), o diferență de 0,82. Pe AIC, ARMA(2,3) este al doilea, la 0,92 de minim, înaintea lui MA(3) (1,07), iar ARMA(3,3) este la 2,37. Pe BIC, MA(3) are 563,36, iar ARMA(1,3) îl urmează la 1,59.'),
  'ma3-table': ('În ARMA(1,3), θ̂₁ = 1,019 > 1. Este modelul invertibil?',
                'Da: rădăcinile polinomului θ(z) au modulele 1,102; 1,181; 1,181, toate peste 1. Pentru q > 1 invertibilitatea se citește din rădăcini, nu din coeficienți luați separat. '
                'Coeficientul AR, φ̂ = −0,284 cu se 0,143, are raportul t −1,98, abia peste pragul de 1,96.'),
  'gdp-fc-table': ('De ce nu se mai lărgește intervalul MA(3) după h = 4, în timp ce intervalul AR(1) continuă să crească?',
                   'La MA(3), ψ<sub>j</sub> = 0 pentru j > 3, deci σ<sub>h</sub> rămîne σ̂√(1 + θ̂₁² + θ̂₂² + θ̂₃²) = 3,07 × √2,50 = 4,85 de la h = 4. La AR(1), σ<sub>h</sub> crește de la 3,38 la 4,35 la h = 8. '
                   'Lățimea intervalului de 95% trece de la 12,0 la 19,0 pentru MA(3) și de la 13,2 la 17,0 pentru AR(1). Cu originea în T4 2019, RMSE pe cele 8 trimestre observate este 7,41 pentru MA(3) și 7,06 pentru AR(1).'),
  'ret-table': ('Slide-ul dă pentru BET raportul t = 20,9. Este acesta raportul t calculat cu eroarea standard clasică?',
                'Nu. Eroarea standard din slide, 0,0052, este cea OPG, implicită în statsmodels; eroarea standard clasică, din curbura log-verosimilității, este 0,0122, apropiată de √((1 − φ̂²)/T) = 0,0122, iar raportul t este 9,0. '
                'Pentru EUR/RON, 12,6 în loc de 30,6. Coeficienții rămîn foarte semnificativi, dar R² este 1,2%, respectiv 2,9%.'),
 },
 'en': {
  'wx-ar1': ('In X<sub>t</sub> = 2 + 0.8X<sub>t−1</sub> + ε<sub>t</sub>, the intercept is 2. Is 2 the mean of the process?',
             'No: μ = c/(1 − φ) = 2/0.2 = 10. The variance is γ(0) = 1/(1 − 0.64) = 2.78, ρ(3) = 0.512, and the half-life is 3.11 periods. With the same c and φ = 0.9, the mean rises to 20, γ(0) to 5.26 and the half-life to 6.58 periods.'),
  'ar-roots': ('Example B has φ₁ = 1.0. Is the process stationary?',
               'Yes. The discriminant φ₁² + 4φ₂ = −1.4 is negative, the roots are complex, 0.833 ± 0.986i, with modulus 1.291 > 1 (the inverse roots have modulus 0.775). The triangle conditions hold: 0.4 < 1, −1.6 < 1, 0.6 < 1. '
               'Conversely, in example C both coefficients are below 1, yet the root 0.936 lies inside the unit circle.'),
  'wx-theta': ('The ▶ button moves θ from −3 to 3. For which values is the MA(1) invertible, and what happens to the AR(∞) weights at θ = 2?',
               'For |θ| < 1. At θ = 2 the weights (−θ)<sup>j</sup> are 1, −2, 4, −8, … and reach 256 at j = 8: the shocks cannot be recovered from the past of the series. The twin representation, θ = 0.5 with σ² = 4, has the weights 1, −0.5, 0.25, … and the same γ(0) = 5, γ(1) = 2.'),
  'wx-psi': ('For φ = 0.7 and θ = 0.4, how many ψ weights cover at least 95% of γ(0)/σ² = 3.37?',
             'Five, ψ₀…ψ₄: the cumulative sum of squares reaches 3.236, that is 95.9%; with ψ₀…ψ₃ it reaches 91.7%. ψ₁² = 1.21 alone accounts for 35.9%, more than the current shock (ψ₀² = 1, 29.7%), because ψ₁ = φ + θ = 1.10 > 1.'),
  'wx-arma11': ('For φ = 0.7 and θ = 0.4, does the ACF start above or below φ? And for θ = −0.4?',
                'Above: ρ(1) = 0.819 > 0.7. The difference ρ(1) − φ = θ(1 − φ²)/(1 + 2φθ + θ²) has the sign of θ, so with θ = −0.4 the ACF starts below φ: ρ(1) = 0.360, ρ(2) = 0.252, and γ(0) falls from 3.37 to 1.18. '
                'In both cases the ratio ρ(h)/φ<sup>h</sup> is constant (1.169 and 0.514): after lag 1 the ACF decays like that of an AR(1).'),
  'ident-table': ('For the AR(2) with T = 500, the sample PACF leaves the band only at lags 1 and 2. Does the pattern survive with T = 100?',
                  'Yes for the PACF: the band widens to ±0.196 and the PACF still leaves it only at lags 1 and 2. The ACF leaves it at lags 1, 3, 4 and 5: the damped-wave decay is visible in the short sample too. '
                  'For the MA(2) with T = 500, the ACF leaves the band at lags 1 and 2 and the PACF at lags 1–4: the table is read the other way round.'),
  'wx-yw': ('The slide writes 1 − ρ̂₁² = 0.620, but 1 − 0.62² = 0.616. Which value is right?',
            '0.620, computed with the unrounded ρ̂₁ = 0.6167; 0.616 comes from rounding ρ̂₁ to 0.62. With the unrounded values, φ̂₁ = 0.600, φ̂₂ = 0.027 and σ̂² = 11.62, as on the slide (γ̂(0) = 18.77, dividing by T); with the rounded 0.62 and 0.40 one would get φ̂₁ = 0.604 and φ̂₂ = 0.025. '
            'On the sample that stops in 2019Q4, φ̂₂ = 0.102.'),
  'ml-lik': ('For the AR(1) on GDP growth, statsmodels reports se(φ̂) = 0.085, while the curvature of the log-likelihood gives 0.076. Which one matches the formula √((1 − φ̂²)/T)?',
             'The curvature: 0.076, against 0.075 from the asymptotic formula. 0.085 is the OPG (outer product of gradients) standard error, the statsmodels default. The conclusion does not change: φ̂ = 0.630, the t-ratio is 8.3 with the curvature, and the 95% interval is [0.481; 0.779].'),
  'wx-ic': ('ARMA(1,3) has a log-likelihood 1.54 higher than the MA(3). Why does BIC still prefer the MA(3)?',
            'The extra parameter costs 2 in AIC and ln 106 = 4.66 in BIC, while −2 ln L falls by only 3.07. AIC therefore falls by 1.07 (548.97 against 550.04), and BIC rises by 1.59 (564.95 against 563.36). AICc, with the small-T correction, gives 549.82 and 550.64.'),
  'diag-table': ('For the AR(1), Ljung–Box with m = 8 gives Q*(8) = 16.23. Do we reject at 5% that the residuals are uncorrelated?',
                 'Yes: with 8 − 1 = 7 degrees of freedom p = 0.023; even with the (wrong) 8 degrees of freedom p = 0.039. For the MA(3), Q*(8) = 5.35 with 5 degrees of freedom gives p = 0.375, so we do not reject. '
                 'Normality is rejected for both: Jarque–Bera 25.9 for the AR(1) and 39.0 for the MA(3), above 5.99.'),
  'wx-fc': ('For X<sub>t</sub> = 2 + 0.8X<sub>t−1</sub> + ε<sub>t</sub>, towards which width does the 95% forecast interval tend?',
            'Towards 2 × 1.96 × √γ(0) = 2 × 1.96 × 1.67 = 6.53. The width is 3.92 at h = 1, 5.02 at h = 2 and 6.50 at h = 10: uncertainty grows fast over the first horizons, then settles at the unconditional level.'),
  'gdp-grid': ('Over the whole grid p, q ≤ 3, which model has the smallest AICc, and by how much does it beat the MA(3)?',
               'ARMA(1,3), with AICc 549.82 against 550.64 for the MA(3), a difference of 0.82. On AIC, ARMA(2,3) is second, 0.92 above the minimum, ahead of the MA(3) (1.07), and ARMA(3,3) is 2.37 above. On BIC, the MA(3) has 563.36 and ARMA(1,3) follows 1.59 higher.'),
  'ma3-table': ('In the ARMA(1,3), θ̂₁ = 1.019 > 1. Is the model invertible?',
                'Yes: the roots of θ(z) have moduli 1.102, 1.181 and 1.181, all above 1. For q > 1 invertibility is read from the roots, not from the coefficients one at a time. '
                'The AR coefficient, φ̂ = −0.284 with se 0.143, has a t-ratio of −1.98, just beyond the 1.96 threshold.'),
  'gdp-fc-table': ('Why does the MA(3) interval stop widening after h = 4, while the AR(1) interval keeps growing?',
                   'For the MA(3), ψ<sub>j</sub> = 0 for j > 3, so σ<sub>h</sub> stays at σ̂√(1 + θ̂₁² + θ̂₂² + θ̂₃²) = 3.07 × √2.50 = 4.85 from h = 4 on. For the AR(1), σ<sub>h</sub> grows from 3.38 to 4.35 at h = 8. '
                   'The width of the 95% interval goes from 12.0 to 19.0 for the MA(3) and from 13.2 to 17.0 for the AR(1). With the origin in 2019Q4, the RMSE over the 8 observed quarters is 7.41 for the MA(3) and 7.06 for the AR(1).'),
  'ret-table': ('The slide gives the BET a t-ratio of 20.9. Is this the t-ratio from the classical standard error?',
                'No. The standard error of the slide, 0.0052, is the OPG one, the statsmodels default; the classical standard error, from the curvature of the log-likelihood, is 0.0122, close to √((1 − φ̂²)/T) = 0.0122, and the t-ratio is 9.0. '
                'For EUR/RON, 12.6 instead of 30.6. The coefficients remain highly significant, but R² is 1.2% and 2.9%.'),
 },
}
for _lg in QA:
    QA[_lg].update(QX[_lg])


# Predict, then check: options for the question of each block, (RO, EN) pairs; key = index of the correct option.
# The correct option agrees with the readout of the block at its initial (slide) state, or, when the question is about ▶, at its end.
# Each wrong option encodes one identifiable misconception.
PREDICT = {
 'ar1': (1, [('ρ̂(1) crește, iar varianța rămîne în jur de 1, cît varianța șocurilor.', 'ρ̂(1) rises, while the variance stays around 1, the variance of the shocks.'),
             ('Cresc amîndouă; la φ = 0,99 varianța de selecție este 14,5, mult sub valoarea teoretică de 50,3.', 'Both rise; at φ = 0.99 the sample variance is 14.5, far below the theoretical value of 50.3.'),
             ('Cresc amîndouă, iar varianța de selecție este foarte aproape de 1/(1 − φ²) pentru orice φ.', 'Both rise, and the sample variance is very close to 1/(1 − φ²) for every φ.')]),
 'ar2': (2, [('Da: un AR(2) este staționar dacă fiecare coeficient este sub 1 în valoare absolută.', 'Yes: an AR(2) is stationary if each coefficient is below 1 in absolute value.'),
             ('Da: rădăcinile sînt reale, iar rădăcinile reale nu produc explozii.', 'Yes: the roots are real, and real roots do not produce explosions.'),
             ('Nu: φ₁ + φ₂ = 1,1 > 1 și o rădăcină, 0,936, este în interiorul cercului unitate.', 'No: φ₁ + φ₂ = 1.1 > 1 and one root, 0.936, lies inside the unit circle.')]),
 'ma1': (0, [('Nu: maximul este 0,5, atins la θ = 1.', 'No: the maximum is 0.5, reached at θ = 1.'),
             ('Da: ρ(1) crește odată cu θ și se apropie de 1.', 'Yes: ρ(1) rises with θ and approaches 1.'),
             ('Da, dar numai pentru modele neinvertibile, cu |θ| > 1.', 'Yes, but only for non-invertible models, with |θ| > 1.')]),
 'ma1-rho': (1, [('θ = −0,5, iar păstrăm θ = 0,5 pentru că este pozitiv.', 'θ = −0.5, and we keep θ = 0.5 because it is positive.'),
                 ('θ = 2; păstrăm θ = 0,5, reprezentarea invertibilă.', 'θ = 2; we keep θ = 0.5, the invertible representation.'),
                 ('θ = 2; alegem după valoarea lui ρ̂(1) din date.', 'θ = 2; we choose using the value of ρ̂(1) in the data.')]),
 'psi': (0, [('Pentru că ψ₁ = φ + θ: șocul trece și prin termenul MA, și prin partea AR.', 'Because ψ₁ = φ + θ: the shock passes through both the MA term and the AR part.'),
             ('Pentru că procesul nu este staționar.', 'Because the process is not stationary.'),
             ('Pentru că ψ₁ = φ/(1 − θ).', 'Because ψ₁ = φ/(1 − θ).')]),
 'patterns': (2, [('Banda rămîne aceeași; numai valorile de selecție se schimbă.', 'The band stays the same; only the sample values change.'),
                  ('Banda se lărgește, pentru că un eșantion mai lung are mai multe valori extreme.', 'The band widens, because a longer sample has more extreme values.'),
                  ('Banda se îngustează de la ±0,28 la ±0,09, iar valorile de selecție se apropie de cele teoretice.', 'The band narrows from ±0.28 to ±0.09, and the sample values move towards the theoretical ones.')]),
 'estimators': (1, [('Da: toți trei sînt consistenți, deci nedeplasați.', 'Yes: all three are consistent, hence unbiased.'),
                    ('Nu: toți trei sînt deplasați în jos, cu medii între 0,853 și 0,866.', 'No: all three are biased downwards, with means between 0.853 and 0.866.'),
                    ('Numai Yule–Walker este deplasat; MLE este nedeplasat.', 'Only Yule–Walker is biased; MLE is unbiased.')]),
 'ic': (0, [('BIC, în 98,6% dintre eșantioane.', 'BIC, in 98.6% of the samples.'),
            ('AIC, pentru că alege ordinul adevărat mai des și cu T = 100.', 'AIC, because it also picks the true order more often with T = 100.'),
            ('Amîndouă la fel de des: cu T = 1000 criteriile coincid.', 'Both equally often: with T = 1000 the criteria agree.')]),
 'lb-df': (2, [('În 5% dintre eșantioane, nivelul nominal.', 'In 5% of the samples, the nominal level.'),
               ('Mai des decît în 5%, pentru că parametrii estimați lasă corelație în reziduuri.', 'More often than 5%, because the estimated parameters leave correlation in the residuals.'),
               ('Mult mai rar, în 1,4% dintre eșantioane.', 'Much less often, in 1.4% of the samples.')]),
 'forecast-mean': (1, [('2,7 pentru ambele: prognoza la orice orizont este φX_T.', '2.7 for both: the forecast at every horizon is φX_T.'),
                       ('1,77 pentru φ = 0,9 și 0,094 pentru φ = 0,5.', '1.77 for φ = 0.9 and 0.094 for φ = 0.5.'),
                       ('3 pentru ambele: cea mai bună prognoză este ultima valoare.', '3 for both: the best forecast is the last value.')]),
 'gdp-ident': (0, [('Da: ACF este 0,62; 0,45; 0,24, apoi 0,04 la lagul 4.', 'Yes: the ACF is 0.62, 0.45, 0.24, then 0.04 at lag 4.'),
                   ('Nu: tiparul MA(3) vine numai din căderea din 2020.', 'No: the MA(3) pattern comes only from the 2020 drop.'),
                   ('Nu: fără 2020, seria devine zgomot alb.', 'No: without 2020 the series becomes white noise.')]),
 'gdp-diag': (2, [('Tot 0,375: p-value-ul nu depinde de gradele de libertate.', 'Still 0.375: the p-value does not depend on the degrees of freedom.'),
                  ('Un p-value mai mic, deci o respingere mai ușoară.', 'A smaller p-value, so an easier rejection.'),
                  ('0,720, o concluzie prea liniștitoare.', '0.720, a conclusion that is too reassuring.')]),
 'gdp-forecast': (1, [('Niciodată: prognoza MA(3) revine treptat la medie, ca la AR(1).', 'Never: the MA(3) forecast returns gradually to the mean, like the AR(1).'),
                      ('De la al patrulea trimestru, cînd devine media de 3,17%.', 'From the fourth quarter, when it becomes the mean of 3.17%.'),
                      ('De la al doilea trimestru.', 'From the second quarter.')]),
 'inflation': (1, [('Spre ținta BNR de 2,5%.', 'Towards the BNR target of 2.5%.'), ('Spre media estimată, 5,16%.', 'Towards the estimated mean, 5.16%.'),
                   ('Spre ultima valoare, 6,09%: un AR(2) cu φ̂₁ + φ̂₂ aproape de 1 nu revine.', 'Towards the last value, 6.09%: an AR(2) with φ̂₁ + φ̂₂ close to 1 does not return.')]),
 'returns': (2, [('Da: un raport t mare înseamnă un efect mare.', 'Yes: a large t-ratio means a large effect.'),
                 ('Da: φ̂ = 0,109 înseamnă că modelul explică 10,9% din variație.', 'Yes: φ̂ = 0.109 means that the model explains 10.9% of the variation.'),
                 ('Nu: R² este 1,2% pentru BET.', 'No: R² is 1.2% for the BET.')]),
 'sunspots': (0, [('10,6 ani.', '10.6 years.'), ('2 ani, ordinul modelului.', '2 years, the order of the model.'),
                  ('9 ani, ordinul ales de AIC și BIC.', '9 years, the order chosen by AIC and BIC.')]),
}

# blocks whose initial charts show the answer: the live charts are veiled too until „Verifică” / "Check" or a change of the controls
VEIL_CHARTS = ('ar2', 'ma1-rho', 'estimators', 'ic', 'lb-df', 'forecast-mean', 'gdp-forecast', 'inflation')

# Exit ticket „Verifică-te” / "Check Yourself": the questions of the slide „Autoevaluare” / "Self-assessment" and their answers.
TICKET = [
 dict(q=('Este Xₜ = 1,2Xₜ₋₁ − 0,32Xₜ₋₂ + εₜ staționar?', 'Is Xₜ = 1.2Xₜ₋₁ − 0.32Xₜ₋₂ + εₜ stationary?'),
      key=0, opts=[
  (('Da: rădăcinile 1,25 și 2,5 sînt în afara cercului unitate.', 'Yes: the roots 1.25 and 2.5 lie outside the unit circle.'),
   ('1 − 1,2z + 0,32z² = (1 − 0,8z)(1 − 0,4z), cu rădăcinile 1,25 și 2,5.', '1 − 1.2z + 0.32z² = (1 − 0.8z)(1 − 0.4z), with roots 1.25 and 2.5.')),
  (('Nu: φ₁ = 1,2 este mai mare decît 1.', 'No: φ₁ = 1.2 is larger than 1.'),
   ('Un coeficient peste 1 nu decide singur: contează rădăcinile, iar ambele sînt în afara cercului.', 'A coefficient above 1 does not decide on its own: the roots matter, and both lie outside the circle.')),
  (('Nu: φ₁ + φ₂ = 0,88 este prea aproape de 1.', 'No: φ₁ + φ₂ = 0.88 is too close to 1.'),
   ('Condiția este φ₁ + φ₂ < 1, iar 0,88 o îndeplinește; la fel φ₂ − φ₁ < 1 și φ₂ > −1.', 'The condition is φ₁ + φ₂ < 1, and 0.88 meets it; so do φ₂ − φ₁ < 1 and φ₂ > −1.'))]),
 dict(q=('ACF a unei serii are o singură valoare semnificativă, ρ̂(1) = −0,45, iar PACF descrește. Ce model propuneți?',
         'The ACF of a series has a single significant value, ρ̂(1) = −0.45, and the PACF decays. Which model do you propose?'),
      key=1, opts=[
  (('AR(1) cu φ = −0,45.', 'AR(1) with φ = −0.45.'),
   ('Un AR(1) ar avea ACF care descrește geometric și PACF care se anulează după lagul 1: tiparul invers.', 'An AR(1) would have a geometrically decaying ACF and a PACF that cuts off after lag 1: the opposite pattern.')),
  (('MA(1) cu θ < 0, rădăcina invertibilă a ecuației θ/(1 + θ²) = −0,45.', 'MA(1) with θ < 0, the invertible root of θ/(1 + θ²) = −0.45.'),
   ('ACF care se anulează după lagul 1 și PACF care descrește indică un MA(1); dintre cele două soluții, θ și 1/θ, o păstrăm pe cea cu |θ| < 1.', 'An ACF that cuts off after lag 1 and a decaying PACF point to an MA(1); of the two solutions, θ and 1/θ, we keep the one with |θ| < 1.')),
  (('MA(1) cu θ = −0,45.', 'MA(1) with θ = −0.45.'),
   ('ρ(1) = θ/(1 + θ²), nu θ: pentru θ = −0,45, ρ(1) = −0,37.', 'ρ(1) = θ/(1 + θ²), not θ: for θ = −0.45, ρ(1) = −0.37.'))]),
 dict(q=('Un ARMA(2,1) lasă în reziduuri Q*(10) = 15,2. Respingem ipoteza de necorelare la 5%?',
         'An ARMA(2,1) leaves Q*(10) = 15.2 in the residuals. Do we reject no autocorrelation at 5%?'),
      key=2, opts=[
  (('Nu: valoarea critică pentru 10 grade de libertate este 18,31.', 'No: the critical value for 10 degrees of freedom is 18.31.'),
   ('Gradele de libertate sînt m − p − q = 10 − 3 = 7, nu 10: cu 10 am accepta greșit modelul.', 'The degrees of freedom are m − p − q = 10 − 3 = 7, not 10: with 10 we would wrongly accept the model.')),
  (('Nu: Q*(10) trebuie comparat cu 10, numărul de laguri.', 'No: Q*(10) must be compared with 10, the number of lags.'),
   ('Statistica se compară cu cuantila distribuției χ², nu cu numărul de laguri.', 'The statistic is compared with a quantile of the χ² distribution, not with the number of lags.')),
  (('Da: cu 7 grade de libertate, valoarea critică este 14,07 < 15,2.', 'Yes: with 7 degrees of freedom the critical value is 14.07 < 15.2.'),
   ('Gradele de libertate sînt 10 − 3 = 7, iar χ²₀,₉₅(7) = 14,07.', 'The degrees of freedom are 10 − 3 = 7, and χ²₀.₉₅(7) = 14.07.'))]),
]


def _predict_js():
    import json
    for i, lang in enumerate(('ro', 'en')):
        pr = {a: dict(key=k, opts=[o[i] for o in opts], **({'veil': {'out': True, 'charts': True}} if a in VEIL_CHARTS else {}))
              for a, (k, opts) in PREDICT.items()}
        tk = dict(title='Verifică-te' if lang == 'ro' else 'Check Yourself', id='check', store=f'tsa-ch2-check-{lang}',
                  qs=[dict(q=t['q'][i], key=t['key'], opts=[dict(t=o[i], why=w[i]) for o, w in t['opts']]) for t in TICKET])
        TEXT_JS[lang] += ('window.CH2_TEXT.predict = ' + json.dumps(pr, ensure_ascii=False) + ';\n'
                          + 'window.CH2_TEXT.ticket = ' + json.dumps(tk, ensure_ascii=False) + ';\n')


_predict_js()


MATHJAX = ('<script>window.MathJax = { tex: { inlineMath: [[\'\\\\(\', \'\\\\)\']], displayMath: [[\'\\\\[\', \'\\\\]\']] }, '
           'startup: { pageReady: () => MathJax.startup.defaultPageReady().then(() => document.body.setAttribute(\'data-mj\', \'done\')) } };</script>\n'
           '<script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3.2.2/es5/tex-chtml.js"></script>\n')


def _post(lang):
    """Insert the EXTRA blocks in slide order (with their section headings) and MathJax; render their slide images."""
    import pymupdf
    import interactive_common as ic
    sp = sys.modules[__name__]
    t, W = L[lang], widgets_all(lang)
    path = os.path.join(ic.OUT, t['file'])
    html = open(path, encoding='utf-8').read()
    viewer = '../pdfjs/web/viewer.html?file=../../' + TEX[lang] + '.pdf'
    chart_pages = ic.slide_pages(sp, lang)
    blocks = {m.group(1): m.group(0) for m in re.finditer(r'\n<section class="blk" id="([^"]+)">.*?</section>', html, flags=re.S)}
    items = [(chart_pages[c][0], a, k, blocks[a]) for a, c, k in BLOCKS]
    xp = _extra_pages(lang)
    doc = pymupdf.open(os.path.join(ic.ROOT, TEX[lang] + '.pdf'))
    for anchor, _, _, skey, _ in EXTRA:
        pg, title = xp[anchor]
        img = f'ch2x_{anchor}_{lang}.jpg'
        doc[pg - 1].get_pixmap(dpi=130).save(os.path.join(ic.OUT, 'slides', img), jpg_quality=82)
        live, tail = _extra_live(anchor, lang, W)
        items.append((pg, anchor, skey, f'''
<section class="blk" id="{anchor}">
  <h3>{title}</h3>
  <div class="pair">
    <figure class="slide"><a href="{viewer}#page={pg}" target="_blank" rel="noopener"><img src="slides/{img}" alt="{t["slide"]} {pg - 1}: {title}" loading="lazy"></a>
      <figcaption>{t["slide"]} {pg - 1}</figcaption></figure>
    {live}
  </div>
  {tail}
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
    import interactive_common
    page_spec = types.SimpleNamespace(**{k: globals()[k] for k in ('N', 'TEX', 'QL', 'BLOCKS', 'PFX', 'SECTIONS', 'L', 'QA', 'TEXT_JS')})
    page_spec.widgets = lambda lang: {a: v for a, v in widgets_all(lang).items() if a in {b for b, _, _ in BLOCKS}}
    interactive_common.build(page_spec)
    for lg in ('en', 'ro'):
        _post(lg)
