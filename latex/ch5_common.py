r"""
ch5_common.py -- shared helpers of the Chapter 5 generators (lecture and seminar), TSA
=====================================================================================
Numbers from Quantlets/Ch_05/ch5_numbers.json (generate_all_charts.py) and sem5_results.json (seminar5.py);
the clickable citations of Chapter 5 (DOIs checked against Crossref, 5 October 2026).
"""

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from tsa_build import ROOT, Values, n   # noqa: E402,F401
from ch1_common import T, V2, finalize, date, pv, MONTHS_EN, MONTHS_RO   # noqa: E402,F401

QL = os.path.join(ROOT, 'Quantlets', 'Ch_05')
QLURL = 'https://github.com/danpele/Time-Series-Analysis/tree/main/Quantlets/Ch_05'
ASSETS = ['sp500', 'bet', 'eurron', 'btc']
NAMES = {'sp500': 'S\\&P 500', 'bet': 'BET', 'eurron': 'EUR/RON', 'btc': 'Bitcoin'}
IG = 0.9995          # persistence at or above this value: IGARCH (alpha + beta = 1 on the boundary)


def load():
    with open(os.path.join(QL, 'ch5_numbers.json')) as f:
        return json.load(f)


def load_sem():
    with open(os.path.join(QL, 'sem5_results.json')) as f:
        return json.load(f)


def put_fit(V, key, s):
    """Parameters, robust standard errors, persistence, half-life and volatilities of one fit (summary())."""
    p, se = s['params'], s['se']
    for a, b in [('mu', 'mu'), ('c', 'Const'), ('om', 'omega'), ('a', 'alpha[1]'), ('b', 'beta[1]'), ('g', 'gamma[1]'),
                 ('nu', 'nu'), ('eta', 'eta'), ('lam', 'lambda')]:
        if b in p:
            d = 2 if a in ('nu', 'eta') else (4 if a == 'om' else 3)
            V.put(f'{key}.{a}', p[b], d)
            V.put(f'{key}.{a}.se', se[b], d)
    V.put(f'{key}.ll', s['loglik'], 1)
    V.put(f'{key}.aic', s['aic'], 1)
    V.put(f'{key}.bic', s['bic'], 1)
    V.int(f'{key}.n', s['n'])
    V.raw(f'{key}.y0', s['first'][:4])
    V.put(f'{key}.vs', s['vol_sample'], 1)
    V.put(f'{key}.ab', s['params']['alpha[1]'] + s['params']['beta[1]'], 4)
    if s['pers'] >= IG:
        V.raw(f'{key}.pers', '⁅1.000⁆')
        V.raw(f'{key}.hl', '--')
        V.raw(f'{key}.vlr', '--')
    else:
        V.put(f'{key}.pers', s['pers'], 3)
        V.put(f'{key}.hl', s['hl'], 0)
        V.put(f'{key}.hl1', s['hl'], 1)
        V.put(f'{key}.1mp', 1 - s['pers'], 4)
        if 'vol_lr' in s:
            V.put(f'{key}.vlr', s['vol_lr'], 1)
            V.put(f'{key}.uv', s['uv'], 2)


def values(N):
    """All lecture numbers as @{key} values."""
    V = Values()
    V.raw('end', date(N['end']))
    S = N['sty']
    for k, d in S.items():
        V.int(f'sty.{k}.n', d['n'])
        V.raw(f'sty.{k}.y0', d['first'][:4])
        V.put(f'sty.{k}.mean', d['mean'], 3)
        V.put(f'sty.{k}.sd', d['sd'], 2)
        V.put(f'sty.{k}.vol', d['vol'], 1)
        V.put(f'sty.{k}.skew', d['skew'], 2)
        V.put(f'sty.{k}.kurt', d['kurt'], 1)
        V.put(f'sty.{k}.min', d['min'], 1)
        V.put(f'sty.{k}.max', d['max'], 1)
        V.raw(f'sty.{k}.dmin', date(d['date_min']))
        V.raw(f'sty.{k}.dmax', date(d['date_max']))
        V.put(f'sty.{k}.rr', d['rho_r'], 2)
        V.put(f'sty.{k}.rr2', d['rho_r2'], 2)
        V.put(f'sty.{k}.rabs', d['rho_abs'], 2)
        V.put(f'sty.{k}.q', d['lb_r'][0], 0)
        V.put(f'sty.{k}.q2', d['lb_r2'][0], 0)
        V.put(f'sty.{k}.lm', d['arch']['lm'], 0)
        V.put(f'sty.{k}.lmr2', d['arch']['r2'], 3)
        V.int(f'sty.{k}.lmn', d['arch']['n'])
    for k, d in N['returns'].items():
        for c, v in d.items():
            V.put(f'ret.{k}.{c}', v, 2)
    V.put('ret.sp500.r08', N['returns']['sp500']['sd2008'] / N['returns']['sp500']['sd2017'], 1)
    V.put('ret.sp500.r20', N['returns']['sp500']['sd2020'] / N['returns']['sp500']['sd2017'], 1)
    for k, d in N['acf_sq'].items():
        V.put(f'aq.{k}.r1', d['r2_1'], 2)
        V.put(f'aq.{k}.r10', d['r2_10'], 2)
        V.put(f'aq.{k}.r50', d['r2_50'], 2)
        V.put(f'aq.{k}.band', d['band'], 3)
        V.raw(f'aq.{k}.out', str(d['out_r2']))
        V.raw(f'aq.{k}.outr', str(d['out_r']))
    A = N['arma']
    V.raw('ar.p', str(A['p_bic']))
    V.put('ar.phi', A['phi'], 3)
    V.put('ar.se', A['phi_se'], 3)
    V.put('ar.ser', A['phi_se_robust'], 3)
    V.put('ar.c', A['c'], 3)
    V.put('ar.qe', A['lb_e'][0], 1)
    V.raw('ar.qep', pv(A['lb_e'][1]))
    V.put('ar.qe2', A['lb_e2'][0], 0)
    V.put('ar.lm', A['archlm_e']['lm'], 0)
    V.put('ar.lmr2', A['archlm_e']['r2'], 3)
    V.int('ar.lmn', A['archlm_e']['n'])
    V.put('ar.lmcrit', A['archlm_e']['crit'], 2)
    for i, b in enumerate(A['archlm_e']['b']):
        V.put(f'ar.b{i}', b, 3)
    V.put('ar.qz', A['lb_z'][0], 1)
    V.raw('ar.qzp', pv(A['lb_z'][1]))
    V.put('ar.qz2', A['lb_z2'][0], 1)
    V.raw('ar.qz2p', pv(A['lb_z2'][1]))
    V.put('ar.lmz', A['archlm_z']['lm'], 1)
    V.raw('ar.lmzp', pv(A['archlm_z']['p']))
    V.put('ar.e21', A['acf_e2_1'], 2)
    V.put('ar.z21', A['acf_z2_1'], 3)
    Sm = N['sim']
    for k, lab in [('iid', 'i.i.d. Normal(0, 1)'), ('arch', 'ARCH(1)'), ('garch', 'GARCH(1,1)')]:
        V.put(f'sim.{k}.k', Sm[lab]['kurt'], 2)
        V.put(f'sim.{k}.max', Sm[lab]['max_abs'], 2)
        V.put(f'sim.{k}.a1', Sm[lab]['acf2_1'], 2)
    V.put('sim.garch.kth', 3 * (1 - 0.98 ** 2) / (1 - 0.98 ** 2 - 2 * 0.1 ** 2), 2)
    for n_ in ['100', '1000']:
        V.put(f'la.{n_}', N['lik_arch1'][n_]['alpha_hat'], 3)
        V.put(f'la.{n_}.se', N['lik_arch1'][n_]['se'], 3)
    for n_ in ['500', '2000']:
        V.put(f'lg.{n_}.a', N['lik_garch'][n_]['a_hat'], 2)
        V.put(f'lg.{n_}.b', N['lik_garch'][n_]['b_hat'], 2)
    for lab, k in [('ARCH(1)', 'a1'), ('ARCH(5)', 'a5'), ('ARCH(10)', 'a10'), ('GARCH(1,1)', 'g11')]:
        d = N['archq'][lab]
        V.put(f'aqt.{k}.ll', d['loglik'], 1)
        V.put(f'aqt.{k}.aic', d['aic'], 1)
        V.put(f'aqt.{k}.bic', d['bic'], 1)
        V.put(f'aqt.{k}.pers', d['pers'], 3)
        V.raw(f'aqt.{k}.k', str(d['k']))
    E = N['est']
    st_ = E['step']
    for a, b in [('mu', 'mu'), ('om', 'omega'), ('a', 'alpha[1]'), ('b', 'beta[1]')]:
        V.put(f'step.{a}', st_['params'][b], 4)
        V.put(f'step.{a}.se', st_['se'][b], 4)
        V.put(f'arch.{a}', E['normal']['params'][b], 4)
        V.put(f'arch.{a}.sec', E['normal']['se_classic'][b], 4)
        V.put(f'arch.{a}.ser', E['normal']['se'][b], 4)
    V.put('step.ll', st_['loglik'], 1)
    put_fit(V, 'en', E['normal'])
    put_fit(V, 'et', E['t'])
    put_fit(V, 'es', E['skewt'])
    V.put('kurt.r', E['kurt_r'], 1)
    V.put('kurt.z', E['kurt_z'], 1)
    V.put('kurt.zt', E['kurt_zt'], 1)
    V.put('lr.tn', 2 * (E['t']['loglik'] - E['normal']['loglik']), 1)
    V.put('lr.st', 2 * (E['skewt']['loglik'] - E['t']['loglik']), 1)
    V.put('se.ratio', sum(E['normal']['se'][c] / E['normal']['se_classic'][c] for c in ('alpha[1]', 'beta[1]')) / 2, 1)
    for k, s in N['markets'].items():
        put_fit(V, f'm.{k}', s)
    for grp in ['vol1', 'vol2']:
        for k, d in N[grp].items():
            V.put(f'v.{k}.last', d['last'], 1)
            V.put(f'v.{k}.med', d['median'], 1)
            V.put(f'v.{k}.max', d['max'], 0)
            V.raw(f'v.{k}.dmax', date(d['date_max']))
            for e in ['2008', '2020']:
                if f'peak{e}' in d:
                    V.put(f'v.{k}.p{e}', d[f'peak{e}'], 0)
                    V.raw(f'v.{k}.d{e}', date(d[f'date{e}']))
    Q = N['qq']
    V.put('qq.nu', Q['nu'], 1)
    V.put('qq.n001', Q['q']['GARCH(1,1)-N']['q001'], 2)
    V.put('qq.n001th', Q['q']['GARCH(1,1)-N']['th001'], 2)
    V.put('qq.t001', Q['q']['GARCH(1,1)-t']['q001'], 2)
    V.put('qq.t001th', Q['q']['GARCH(1,1)-t']['th001'], 2)
    W = N['ewma']
    for c in ['g', 'e', 'w']:
        V.put(f'ew.{c}.peak', W[f'{c}_peak'], 0)
        V.raw(f'ew.{c}.dpeak', date(W[f'{c}_dpeak']))
        V.put(f'ew.{c}.jun', W[f'{c}_jun'], 0)
    V.put('ew.hl', W['hl_ewma'], 1)
    for k, d in N['ag'].items():
        V.raw(f'ag.{k}.p', str(d['p_bic']))
        V.put(f'ag.{k}.phio', d['phi_ols'], 3)
        V.put(f'ag.{k}.seo', d['se_ols'], 3)
        V.put(f'ag.{k}.phig', d['phi_g'], 3)
        V.put(f'ag.{k}.seg', d['se_g'], 3)
        V.put(f'ag.{k}.tg', d['t_g'], 1)
        V.put(f'ag.{k}.qc', d['lbz_c'][0], 1)
        V.raw(f'ag.{k}.qcp', pv(d['lbz_c'][1]))
        V.put(f'ag.{k}.qa', d['lbz_a'][0], 1)
        V.raw(f'ag.{k}.qap', pv(d['lbz_a'][1]))
        V.put(f'ag.{k}.dbic', d['bic_a'] - d['bic_c'], 1)
    B = N['bands']
    V.put('bd.sd', B['sd_const'], 2)
    V.put('bd.wc', B['w_c'], 1)
    V.put('bd.wgmin', B['w_g_min'], 1)
    V.put('bd.wgmax', B['w_g_max'], 1)
    for c in ['out_c', 'out_g', 'out_c_calm2017', 'out_g_calm2017', 'out_c_covid', 'out_g_covid']:
        V.put(f'bd.{c}', 100 * B[c], 1)
    for k, d in N['nic'].items():
        V.put(f'nic.{k}.ratio', d['ratio_gjr'], 2)
    for k, d in N['asym'].items():
        V.put(f'as.{k}.g', d['gjr_gamma'], 3)
        V.put(f'as.{k}.gt', d['gjr_t'], 1)
        V.put(f'as.{k}.a', d['gjr_alpha'], 3)
        V.put(f'as.{k}.lr', d['lr'], 1)
        V.raw(f'as.{k}.lrp', pv(d['lr_p']))
        V.put(f'as.{k}.eg', d['eg_gamma'], 3)
        V.put(f'as.{k}.egt', d['eg_t'], 1)
        V.put(f'as.{k}.dbic', d['bic_j'] - d['bic_g'], 1)
        V.put(f'as.{k}.sb', d['sb']['joint'], 1)
        V.raw(f'as.{k}.sbp', pv(d['sb']['joint_p']))
        V.put(f'as.{k}.sbt', d['sb']['sign_t'], 2)
        V.put(f'as.{k}.pers', d['gjr_pers'], 3)
    D = N['diag']
    V.put('dg.r', D['r'][0], 1)
    V.raw('dg.r.p', pv(D['r'][1]))
    V.put('dg.r2', D['r2'][0], 0)
    V.raw('dg.r2.p', pv(D['r2'][1]))
    V.put('dg.lm.r', D['archlm_r']['lm'], 0)
    V.raw('dg.lm.r.p', pv(D['archlm_r']['p']))
    for d in ['normal', 't']:
        for c in ['z', 'z2']:
            V.put(f'dg.{d}.{c}', D[d][c][0], 1)
            V.raw(f'dg.{d}.{c}.p', pv(D[d][c][1]))
        V.put(f'dg.{d}.lm', D[d]['archlm']['lm'], 1)
        V.raw(f'dg.{d}.lm.p', pv(D[d]['archlm']['p']))
    for k, d in N['diag_m'].items():
        V.put(f'dm.{k}.r2', d['r2'][0], 0)
        V.put(f'dm.{k}.z2', d['z2'][0], 1 if d['z2'][0] > 0.05 else 3)
        V.raw(f'dm.{k}.z2p', pv(d['z2'][1]))
        V.put(f'dm.{k}.z', d['z'][0], 1)
        V.raw(f'dm.{k}.zp', pv(d['z'][1]))
    A = N['acf_diag']
    V.put('acf.r1', A['a1_1'], 2)
    V.put('acf.r50', A['a1_50'], 3)
    V.put('acf.z1', A['a2_1'], 3)
    V.put('acf.zmax', A['max_a2'], 3)
    V.put('acf.band', A['band'], 3)
    M = N['managed']
    V.raw('mg.date', date(M['date']))
    V.put('mg.z', M['z'], 0)
    V.put('mg.r', M['r'], 2)
    V.put('mg.rn', M['r_next'], 2)
    V.put('mg.sig', M['sig_before'], 4)
    V.put('mg.abs', M['mean_abs_month'], 4)
    V.put('mg.z2nd', M['z_second'], 1)
    V.put('mg.q', M['lb_z2'][0], 3)
    V.put('mg.qwo', M['lb_z2_wo'][0], 1)
    V.raw('mg.qwop', pv(M['lb_z2_wo'][1]))
    for r in N['ms']:
        key = f"ms.{r['model']}.{r['dist'].replace(' ', '')}"
        V.put(key + '.ll', r['loglik'], 1)
        V.put(key + '.dbic', r['d_bic'], 1)
        V.raw(key + '.k', str(r['k']))
    MSd = {(r['model'], r['dist']): r['bic'] for r in N['ms']}
    V.put('ms.gain.asym', MSd[('GARCH', 'Normal')] - MSd[('EGARCH', 'Normal')], 0)
    V.put('ms.gain.dist', MSd[('GARCH', 'Normal')] - MSd[('GARCH', 'skewed t')], 0)
    V.put('ms.gain.both', MSd[('GARCH', 'Normal')] - min(MSd.values()), 0)
    best = min(N['ms'], key=lambda r: r['bic'])
    lab = f"{best['model']}-{best['dist']}"
    V.raw('ms.best', V2(lab, lab.replace('-skewed t', ' cu inovații t asimetrice').replace('-t', '-t').replace('-Normal', ' cu inovații Normale')))
    TS = N['term']
    V.put('ts.lr', TS['lr'], 1)
    for d, lab in zip(sorted(k for k in TS if k[:2] == '20'), ['calm', 'covid', 'last']):
        V.raw(f'ts.{lab}.d', date(d))
        for c in ['h1', 'h10', 'h22', 'h250', 'avg10', 'avg22', 'avg250']:
            V.put(f'ts.{lab}.{c}', TS[d][c], 1)
    V.put('ts.s2bar', TS['s2bar'], 2)
    for k, d in N['fe'].items():
        V.int(f'fe.{k}.n', d['n'])
        for m in ['garch', 'gjr', 'ewma']:
            V.put(f'fe.{k}.q.{m}', d['qlike'][m], 3)
        V.put(f'fe.{k}.dm', d['dm_garch_ewma']['t'], 2)
        V.raw(f'fe.{k}.dmp', pv(d['dm_garch_ewma']['p']))
        V.put(f'fe.{k}.dmj', d['dm_gjr_garch']['t'], 2)
        V.raw(f'fe.{k}.dmjp', pv(d['dm_gjr_garch']['p']))
        V.put(f'fe.{k}.eg', 100 * d['exc_garch'], 2)
        V.put(f'fe.{k}.ee', 100 * d['exc_ewma'], 2)
        V.raw(f'fe.{k}.neg', str(d['nexc_garch']))
        V.raw(f'fe.{k}.nee', str(d['nexc_ewma']))
        V.raw(f'fe.{k}.nexp', str(round(0.01 * d['n'])))
    for k, d in N['fe_fig'].items():
        V.raw(f'ff.{k}.day', date(d['max_day']))
        V.put(f'ff.{k}.share', 100 * d['share_max'], 0)
    VF = N['var_fig']
    V.raw('vf.n', str(VF['n']))
    V.raw('vf.eg', str(VF['exc_garch']))
    V.raw('vf.ee', str(VF['exc_ewma']))
    V.raw('vf.exp', str(round(0.01 * VF['n'])))
    return V


# -----------------------------------------------------------------------------
# Clickable citations (DOIs verified via Crossref, 5 October 2026)
# -----------------------------------------------------------------------------
def ref(key, url, en, ro=None):
    return f'\\newcommand{{\\ref{key}}}{{\\href{{{url}}}{{{T(en, ro or en)}}}}}\n'


D_ = 'https://doi.org/'
REFS = (ref('HP', D_ + '10.1007/978-3-031-13584-2', 'Huang and Petukhina (2022)', 'Huang și Petukhina (2022)')
        + ref('FPP', 'https://otexts.com/fpp3/', 'Hyndman and Athanasopoulos (2021)', 'Hyndman și Athanasopoulos (2021)')
        + ref('Tsay', D_ + '10.1002/9780470644560', 'Tsay (2010)')
        + ref('FHH', D_ + '10.1007/978-3-030-13751-9', 'Franke, Härdle and Hafner (2019)', 'Franke, Härdle și Hafner (2019)')
        + ref('Hamilton', D_ + '10.2307/j.ctv14jx6sm', 'Hamilton (1994)')
        + ref('Engle', D_ + '10.2307/1912773', 'Engle (1982)')
        + ref('EngleN', D_ + '10.1257/0002828041464597', 'Engle (2004)')
        + ref('Boll', D_ + '10.1016/0304-4076(86)90063-1', 'Bollerslev (1986)')
        + ref('BollT', D_ + '10.2307/1925546', 'Bollerslev (1987)')
        + ref('EB', D_ + '10.1080/07474938608800095', 'Engle and Bollerslev (1986)', 'Engle și Bollerslev (1986)')
        + ref('Nelson', D_ + '10.2307/2938260', 'Nelson (1991)')
        + ref('GJR', D_ + '10.1111/j.1540-6261.1993.tb05128.x', 'Glosten, Jagannathan and Runkle (1993)', 'Glosten, Jagannathan și Runkle (1993)')
        + ref('EN', D_ + '10.1111/j.1540-6261.1993.tb05127.x', 'Engle and Ng (1993)', 'Engle și Ng (1993)')
        + ref('BW', D_ + '10.1080/07474939208800229', 'Bollerslev and Wooldridge (1992)', 'Bollerslev și Wooldridge (1992)')
        + ref('Hansen', D_ + '10.2307/2527081', 'Hansen (1994)')
        + ref('Patton', D_ + '10.1016/j.jeconom.2010.03.034', 'Patton (2011)')
        + ref('HL', D_ + '10.1002/jae.800', 'Hansen and Lunde (2005)', 'Hansen și Lunde (2005)')
        + ref('AB', D_ + '10.2307/2527343', 'Andersen and Bollerslev (1998)', 'Andersen și Bollerslev (1998)')
        + ref('Christie', D_ + '10.1016/0304-405X(82)90018-6', 'Christie (1982)')
        + ref('DM', D_ + '10.1080/07350015.1995.10524599', 'Diebold and Mariano (1995)', 'Diebold și Mariano (1995)')
        + ref('LB', D_ + '10.1093/biomet/65.2.297', 'Ljung and Box (1978)', 'Ljung și Box (1978)')
        + ref('ML', D_ + '10.1111/j.1467-9892.1983.tb00373.x', 'McLeod and Li (1983)', 'McLeod și Li (1983)')
        + ref('Mandelbrot', D_ + '10.1086/294632', 'Mandelbrot (1963)')
        + ref('RM', 'https://www.msci.com/documents/10199/5915b101-4206-4ba0-aee2-3449d5c7e95a', 'J.P. Morgan/Reuters (1996)')
        + ref('Akaike', D_ + '10.1109/TAC.1974.1100705', 'Akaike (1974)')
        + ref('Schwarz', D_ + '10.1214/aos/1176344136', 'Schwarz (1978)')
        + ref('Kupiec', D_ + '10.3905/jod.1995.407942', 'Kupiec (1995)'))

BIB = {
    'Akaike': r'Akaike, H. (1974). A new look at the statistical model identification. \textit{IEEE Transactions on Automatic Control}, 19(6), 716--723. \href{https://doi.org/10.1109/TAC.1974.1100705}{doi:10.1109/TAC.1974.1100705}',
    'AB': r'Andersen, T. G., \& Bollerslev, T. (1998). Answering the skeptics: Yes, standard volatility models do provide accurate forecasts. \textit{International Economic Review}, 39(4), 885--905. \href{https://doi.org/10.2307/2527343}{doi:10.2307/2527343}',
    'Boll': r'Bollerslev, T. (1986). Generalized autoregressive conditional heteroskedasticity. \textit{Journal of Econometrics}, 31(3), 307--327. \href{https://doi.org/10.1016/0304-4076(86)90063-1}{doi:10.1016/0304-4076(86)90063-1}',
    'BollT': r'Bollerslev, T. (1987). A conditionally heteroskedastic time series model for speculative prices and rates of return. \textit{Review of Economics and Statistics}, 69(3), 542--547. \href{https://doi.org/10.2307/1925546}{doi:10.2307/1925546}',
    'BW': r'Bollerslev, T., \& Wooldridge, J. M. (1992). Quasi-maximum likelihood estimation and inference in dynamic models with time-varying covariances. \textit{Econometric Reviews}, 11(2), 143--172. \href{https://doi.org/10.1080/07474939208800229}{doi:10.1080/07474939208800229}',
    'Christie': r'Christie, A. A. (1982). The stochastic behavior of common stock variances: Value, leverage and interest rate effects. \textit{Journal of Financial Economics}, 10(4), 407--432. \href{https://doi.org/10.1016/0304-405X(82)90018-6}{doi:10.1016/0304-405X(82)90018-6}',
    'DM': r'Diebold, F. X., \& Mariano, R. S. (1995). Comparing predictive accuracy. \textit{Journal of Business \& Economic Statistics}, 13(3), 253--263. \href{https://doi.org/10.1080/07350015.1995.10524599}{doi:10.1080/07350015.1995.10524599}',
    'Engle': r'Engle, R. F. (1982). Autoregressive conditional heteroscedasticity with estimates of the variance of United Kingdom inflation. \textit{Econometrica}, 50(4), 987--1007. \href{https://doi.org/10.2307/1912773}{doi:10.2307/1912773}',
    'EngleN': r'Engle, R. F. (2004). Risk and volatility: Econometric models and financial practice. \textit{American Economic Review}, 94(3), 405--420. \href{https://doi.org/10.1257/0002828041464597}{doi:10.1257/0002828041464597}',
    'EB': r'Engle, R. F., \& Bollerslev, T. (1986). Modelling the persistence of conditional variances. \textit{Econometric Reviews}, 5(1), 1--50. \href{https://doi.org/10.1080/07474938608800095}{doi:10.1080/07474938608800095}',
    'EN': r'Engle, R. F., \& Ng, V. K. (1993). Measuring and testing the impact of news on volatility. \textit{Journal of Finance}, 48(5), 1749--1778. \href{https://doi.org/10.1111/j.1540-6261.1993.tb05127.x}{doi:10.1111/j.1540-6261.1993.tb05127.x}',
    'FHH': r'Franke, J., Härdle, W. K., \& Hafner, C. M. (2019). \textit{Statistics of Financial Markets: An Introduction} (5th ed.). Springer. \href{https://doi.org/10.1007/978-3-030-13751-9}{doi:10.1007/978-3-030-13751-9}',
    'GJR': r'Glosten, L. R., Jagannathan, R., \& Runkle, D. E. (1993). On the relation between the expected value and the volatility of the nominal excess return on stocks. \textit{Journal of Finance}, 48(5), 1779--1801. \href{https://doi.org/10.1111/j.1540-6261.1993.tb05128.x}{doi:10.1111/j.1540-6261.1993.tb05128.x}',
    'Hamilton': r'Hamilton, J. D. (1994). \textit{Time Series Analysis}. Princeton University Press. \href{https://doi.org/10.2307/j.ctv14jx6sm}{doi:10.2307/j.ctv14jx6sm}',
    'Hansen': r'Hansen, B. E. (1994). Autoregressive conditional density estimation. \textit{International Economic Review}, 35(3), 705--730. \href{https://doi.org/10.2307/2527081}{doi:10.2307/2527081}',
    'HL': r'Hansen, P. R., \& Lunde, A. (2005). A forecast comparison of volatility models: Does anything beat a GARCH(1,1)? \textit{Journal of Applied Econometrics}, 20(7), 873--889. \href{https://doi.org/10.1002/jae.800}{doi:10.1002/jae.800}',
    'HP': r'Huang, C., \& Petukhina, A. (2022). \textit{Applied Time Series Analysis and Forecasting with Python}. Springer. \href{https://doi.org/10.1007/978-3-031-13584-2}{doi:10.1007/978-3-031-13584-2}',
    'FPP': r'Hyndman, R. J., \& Athanasopoulos, G. (2021). \textit{Forecasting: Principles and Practice} (3rd ed.). OTexts. \href{https://otexts.com/fpp3/}{otexts.com/fpp3}',
    'RM': r'J.P. Morgan/Reuters (1996). \textit{RiskMetrics -- Technical Document} (4th ed.). Morgan Guaranty Trust Company. \href{https://www.msci.com/documents/10199/5915b101-4206-4ba0-aee2-3449d5c7e95a}{msci.com}',
    'Kupiec': r'Kupiec, P. H. (1995). Techniques for verifying the accuracy of risk measurement models. \textit{Journal of Derivatives}, 3(2), 73--84. \href{https://doi.org/10.3905/jod.1995.407942}{doi:10.3905/jod.1995.407942}',
    'LB': r'Ljung, G. M., \& Box, G. E. P. (1978). On a measure of lack of fit in time series models. \textit{Biometrika}, 65(2), 297--303. \href{https://doi.org/10.1093/biomet/65.2.297}{doi:10.1093/biomet/65.2.297}',
    'Mandelbrot': r'Mandelbrot, B. (1963). The variation of certain speculative prices. \textit{Journal of Business}, 36(4), 394--419. \href{https://doi.org/10.1086/294632}{doi:10.1086/294632}',
    'ML': r'McLeod, A. I., \& Li, W. K. (1983). Diagnostic checking ARMA time series models using squared-residual autocorrelations. \textit{Journal of Time Series Analysis}, 4(4), 269--273. \href{https://doi.org/10.1111/j.1467-9892.1983.tb00373.x}{doi:10.1111/j.1467-9892.1983.tb00373.x}',
    'Nelson': r'Nelson, D. B. (1991). Conditional heteroskedasticity in asset returns: A new approach. \textit{Econometrica}, 59(2), 347--370. \href{https://doi.org/10.2307/2938260}{doi:10.2307/2938260}',
    'Patton': r'Patton, A. J. (2011). Volatility forecast comparison using imperfect volatility proxies. \textit{Journal of Econometrics}, 160(1), 246--256. \href{https://doi.org/10.1016/j.jeconom.2010.03.034}{doi:10.1016/j.jeconom.2010.03.034}',
    'Schwarz': r'Schwarz, G. (1978). Estimating the dimension of a model. \textit{Annals of Statistics}, 6(2), 461--464. \href{https://doi.org/10.1214/aos/1176344136}{doi:10.1214/aos/1176344136}',
    'Tsay': r'Tsay, R. S. (2010). \textit{Analysis of Financial Time Series} (3rd ed.). Wiley. \href{https://doi.org/10.1002/9780470644560}{doi:10.1002/9780470644560}',
}
ORDER = ['Akaike', 'AB', 'Boll', 'BollT', 'BW', 'Christie', 'DM', 'Engle', 'EngleN', 'EB', 'EN', 'FHH', 'GJR', 'Hamilton',
         'Hansen', 'HL', 'HP', 'FPP', 'RM', 'Kupiec', 'LB', 'Mandelbrot', 'ML', 'Nelson', 'Patton', 'Schwarz', 'Tsay']


def bib(keys=None):
    return [BIB[k] for k in ORDER if keys is None or k in keys]
