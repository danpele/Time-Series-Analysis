<!--
Course text of the web version, chapter 14 (EN). Rendered into the page by latex/build_web_lecture.py.
Format: # section title (introduction), ## frame title {h=hash} (text), @after (text after the live block), ## @end (synthesis).
Numbered equations: $$ ... $$ {#eq:label}, reference @eq:label. Citations: @Key -> Author (year), [@Key] -> (Author, year).
Current hashes: python3 build_web_lecture.py 14 --hashes
-->

## @intro

@text
[Chapter 5](../EN/Courses/chapter5_conditional_volatility_garch.pdf) modelled the volatility of one return series at a time with generalised autoregressive conditional heteroskedasticity (GARCH) models. Most questions of risk management, however, concern several assets at once. The risk of a portfolio depends on how its components move together, a hedge works only as well as the link between the asset and the hedging instrument, and the spread of a crisis from one market to another is a statement about co-movement. All three questions are answered by covariances and correlations, and these change over time.

This chapter extends GARCH to $N$ series. The object of interest is the conditional covariance matrix $\mathbf{H}_t$, which holds the variances of all assets on its diagonal and their covariances off the diagonal. The chapter first sets out the conditions that any model for $\mathbf{H}_t$ must respect, then presents the general VEC and BEKK models and the reason they become unusable beyond a few assets. The constant conditional correlation (CCC) and dynamic conditional correlation (DCC) models solve this by separating volatilities from correlations and estimating them in two steps.

The applications use daily data on the S&P 500, the German DAX index, the Romanian BET index, the EUR/RON exchange rate and Bitcoin. The correlation between the S&P 500 and the DAX is not constant: the likelihood ratio test against constant correlation gives 131.8, far above the 5% critical value of 5.99. Correlations rise in crises, most of all for the assets that are weakly linked to the large markets in calm times. The same conditional covariances give a portfolio Value at Risk (VaR) at 1% that adapts to the state of the market and dynamic hedge ratios that remove more risk out of sample than a static ratio.

@chart tsa_ch14_dcc_sp_dax
The one-step-ahead DCC correlation of daily S&P 500 and DAX returns, the 250-day rolling correlation and the constant CCC value, 4 January 2000 – 18 September 2026.

@goals
- **Explain** why portfolio risk, hedging and contagion need a model for time-varying covariances.
- **Write** the VEC, BEKK, CCC and DCC models and count their parameters.
- **Compute** one step of the DCC recursion by hand and interpret $a$, $b$ and $a+b$.
- **Estimate** a DCC model in two steps and test it against constant correlations.
- **Compute** a portfolio VaR 1% from $\mathbf{H}_t$ and backtest it with the Kupiec and Christoffersen tests.
- **Compute** dynamic minimum-variance hedge ratios and their hedging effectiveness.

@plan
- The role of multivariate volatility | Portfolio variance, hedging and contagion; rolling correlations and asynchronous trading.
- The multivariate GARCH framework | The conditional covariance matrix, symmetry, positive definiteness and the likelihood.
- VEC and BEKK models | The general models, volatility spillovers and the number of parameters.
- Constant and dynamic conditional correlations | The decomposition into volatilities and correlations, the DCC recursion and two-step estimation.
- DCC for the S&P 500 and the DAX | Two GARCH models, the estimated DCC and correlations in the crises of 2008 and 2020.
- Five markets: S&P 500, DAX, BET, EUR/RON, Bitcoin | A five-asset DCC and correlations in calm times and in the 2020 crash.
- Portfolio VaR 1% with DCC | From $\mathbf{H}_t$ to the VaR of a portfolio and its out-of-sample backtest.
- Dynamic hedge ratios | The minimum-variance hedge ratio and the hedging of the BET with the DAX.
- Links with other chapters and limits | GARCH, VAR and cointegration; the limits of DCC and its extensions.
- Possible contribution of AI | What an assistant can write and what must be checked.
- Summary | Key ideas, formulas and self-assessment questions.

## Motivation for the chapter {h=08ac1655}

The question of the chapter is how the risks of several assets move together. In [Chapter 5](../EN/Courses/chapter5_conditional_volatility_garch.pdf) the conditional variance of a single return series was enough to describe its risk. For a portfolio of two or more assets this is no longer enough: two portfolios with the same individual volatilities can have very different risk, depending on whether their components tend to fall on the same days or on different days. The quantity that captures this is the covariance, or its standardised version, the correlation, and the chapter treats both as variables that change from day to day.

The route goes from the general to the practical. It starts from the conditional covariance matrix $\mathbf{H}_t$ and the two conditions it must satisfy, continues with the VEC and BEKK models, which describe all elements of $\mathbf{H}_t$ at once, and arrives at the CCC and DCC models, which estimate volatilities and correlations separately. The last sections use these models for three applications: correlations in crises, the VaR of a portfolio and hedge ratios.

The chapter builds directly on three earlier ones. The univariate GARCH model of [Chapter 5](../EN/Courses/chapter5_conditional_volatility_garch.pdf) is the building block of every multivariate model; the vector autoregression (VAR) of [Chapter 6](../EN/Courses/chapter6_var_models_granger_causality.pdf) supplies the multivariate notation and the idea of spillovers between series; cointegration, from [Chapter 7](../EN/Courses/chapter7_cointegration_vecm.pdf), is the long-run counterpart of the short-run links studied here.

## Learning outcomes {h=3950ba46}

The first outcomes are conceptual. The reader should be able to explain why the variance of a portfolio, a hedge ratio and the question of contagion all depend on covariances that change over time, and to write the main multivariate GARCH models in matrix form. Counting the parameters of each model is part of understanding it: the count shows immediately which models can be estimated for five assets and which cannot.

The other outcomes are computational and empirical. One step of the DCC recursion can be computed by hand, and its parameters $a$, $b$ and $a+b$ have a direct reading as the reaction to news, the persistence and the speed of return to the long-run correlation. A DCC model is estimated in two steps and compared with constant correlations by a likelihood ratio test. Finally, $\mathbf{H}_t$ is turned into two practical numbers, the VaR 1% of a portfolio, judged by backtesting, and the minimum-variance hedge ratio, judged by its hedging effectiveness out of sample.

## Reading and tools {h=df6cac14}

@HuangPetukhina2022 give the background on GARCH models and their implementation in Python. Two surveys cover the multivariate models in more depth: @BauwensLaurentRombouts2006 classify them and discuss estimation and testing, and @SilvennoinenTerasvirta2009 give a compact overview with the conditions for positive definiteness. Each model of the chapter also has its original paper: @BollerslevEngleWooldridge1988 for VEC, @EngleKroner1995 for BEKK, @Bollerslev1990 for CCC and @Engle2002 for DCC.

The chapter's code uses the arch package for the univariate GARCH models of step 1. The package has no DCC model, so step 2 is written out in about 40 lines of numpy: the recursion for $\mathbf{Q}_t$, the rescaling to correlations and the correlation part of the likelihood. Writing the recursion explicitly has a teaching advantage, since every element of the model is visible in the code. The lecture notebook reproduces the computations and the charts of the chapter.

# The role of multivariate volatility

The section shows why a model for a single volatility is not enough. Portfolio risk, hedging and contagion all depend on correlations, and the correlations observed in real markets move a great deal over time. The section ends with a practical trap of daily data: markets that close at different hours.

## Three applications that require covariances {h=3bb38524}

The first question is the risk of a portfolio. If the vector $\mathbf{w}$ holds the weights of the assets and $\mathbf{H}$ is the covariance matrix of their returns, the variance of the portfolio return is a quadratic form, as in @Markowitz1952:

$$ \begin{gathered} \sigma_p^2 = \mathbf{w}^\top\mathbf{H}\mathbf{w}, \\ \text{for two assets: } \sigma_p^2 = w_1^2\sigma_1^2 + w_2^2\sigma_2^2 + 2w_1w_2\rho\,\sigma_1\sigma_2. \end{gathered} $$ {#eq:pvar}

The first two terms come from the individual volatilities; the third is the only place where the joint behaviour enters. With positive weights, a lower correlation $\rho$ means a lower portfolio variance, and this is the whole mechanism of diversification.

The second question is hedging: how many units of one asset to sell in order to offset the risk of another. Section 7 shows that the answer is the ratio of a covariance to a variance, so a hedge ratio moves whenever the covariance moves. The third question is contagion: do markets move more closely together in a crisis? @LonginSolnik2001 found that correlations between international equity markets are higher in bear markets. If correlations rise exactly when volatility rises, diversification weakens at the moment it is needed most.

A univariate GARCH model gives the volatilities $\sigma_{1,t}$ and $\sigma_{2,t}$ day by day. Formula @eq:pvar also needs the correlation, and the aim of this chapter is to add a correlation $\rho_t$ that changes over time in the same way.

## Worked example: correlation and portfolio risk {h=8fccf5ea}

Take two assets with an annual volatility of 20% each and a portfolio split 50/50. With volatilities in percent, $\sigma_1^2 = \sigma_2^2 = 400$ and $w_1^2 = w_2^2 = 0.25$, so the first two terms of @eq:pvar add up to 200 whatever the correlation. With $\rho = 0.2$ the covariance term is $2 \cdot 0.25 \cdot 0.2 \cdot 400 = 40$, the variance is 240 and the volatility $\sqrt{240} = 15.5\%$. With $\rho = 0.8$ the covariance term is 160, the variance 360 and the volatility 19.0%.

Nothing changed except the correlation: the same assets, the same weights, the same individual volatilities. Yet the risk of the portfolio rises by more than a fifth. The practical consequence is that a risk model which keeps $\rho$ at the value it had in a calm period understates the risk when correlations rise, and the data below show that they do rise in crises.

## Portfolio volatility against the correlation {h=45b17b3e}

The chart plots the annual volatility of the 50/50 portfolio, in percent, on the vertical axis against the correlation $\rho$ on the horizontal axis, from $-1$ to $1$, with both assets at 20% volatility. The two marked points are the cases of the worked example: 15.5% at $\rho = 0.2$ and 19.0% at $\rho = 0.8$.

The curve is the square root of a linear function of $\rho$: for this portfolio $\sigma_p^2 = 200 + 200\rho$. It starts at zero for $\rho = -1$, rises steeply at first and then more slowly, and reaches 20% at $\rho = 1$. Reading the chart from right to left gives the benefit of diversification, the vertical distance between the curve and the 20% level at each correlation.

## Interpreting the diversification curve {h=c98bc673}

Three points of the curve are worth remembering. At $\rho = 1$ the two assets move as one, there is no diversification and the portfolio has the 20% volatility of each asset. At $\rho = 0$ the covariance term disappears and the volatility is $20/\sqrt{2} \approx 14.1\%$. At $\rho = -1$ the movements of the two assets cancel exactly and the risk disappears, a case that does not occur with real equity indices.

Because the curve is concave, the same increase of the correlation adds more volatility at low correlations than at high ones. For portfolios that combine weakly correlated assets, the very portfolios built for diversification, a rise in correlation is therefore most costly. This leads to the empirical question of the chapter: does $\rho$ stay put in real markets, or does it move with the state of the market?

## Time-varying correlations {h=ba949959}

A first answer comes from rolling correlations (*rolling windows*): the sample correlation of daily log returns computed over the last 250 trading days, about one year, and dated at the end of the window. The chart plots this correlation on the vertical axis against time on the horizontal axis for three pairs: the S&P 500 and the DAX, from 2001; the BET and the DAX, from 2001; the S&P 500 and Bitcoin, from 2015. The shaded bands mark September 2008 – March 2009 and February – April 2020, and a dashed line marks zero.

Each correlation uses only the days on which both markets trade, so that each pair of returns refers to the same calendar day. A rolling correlation is a descriptive tool, not a model: it gives every day in the window the same weight and forgets everything older than 250 days. It is useful here because it shows, without any assumption about dynamics, whether the correlation is stable.

## Interpreting the rolling correlations {h=edbebb12}

None of the three correlations is stable. The S&P 500 and the DAX range between 0.21 in April 2026 and 0.80 in October 2012, around a full-sample value of 0.60. The highest values do not come in 2008 but during the euro-area debt crisis of 2010–2012, which tied the two markets even more closely.

The BET and the DAX show a slower change. Before 2007 the correlation was close to zero, with a minimum of $-0.08$; after Romania joined the European Union (EU) it rose in steps, up to 0.70 in August 2020. The Romanian market became integrated with European markets, and its value for diversifying a European portfolio fell accordingly. The S&P 500 and Bitcoin went from $-0.16$ in December 2019 to 0.57 in February 2023: after 2020 Bitcoin behaved more like a risk asset correlated with equities.

The conclusion is that correlations wander far from any constant value, over years through integration and within months through crises. A model must let them move.

## Asynchronous trading and the bias of daily correlations {h=652f48d0}

Daily returns of different markets do not cover the same hours. Bucharest closes at 16:45 Central European Time (CET), 17:45 local time, Frankfurt at 17:30 CET and New York at 22:00 CET, while Bitcoin trades all day, every day. News that arrives after the European close enters European prices only on the next trading day. This is asynchronous trading (*non-synchronous trading*), and it splits the reaction to one event across two daily returns.

The tariff announcement of 9 April 2025 shows the effect. The pause of the United States (US) tariffs was announced in the afternoon in New York, after the European close: on 9 April the S&P 500 rose by 9.1% and the DAX fell by 3.1%; on 10 April the S&P 500 fell by 3.5% and the DAX rose by 4.4%. The same news produced returns of opposite sign on the same calendar day, and this pulls the daily correlation towards zero.

Weekly returns are much less affected, because a one-day delay matters little over five days. Since 2015 the correlation of daily against weekly returns is 0.32 against 0.46 for the S&P 500 and the BET, 0.56 against 0.68 for the S&P 500 and the DAX, and 0.42 against 0.47 for the BET and the DAX. The gap is small for the last pair, whose markets close almost together. The remedies are weekly returns or returns measured between synchronous times; the daily models of this chapter keep the bias in mind when reading transatlantic correlations.

## Recap: The role of multivariate volatility {h=ada4bf84}

The portfolio variance $\mathbf{w}^\top\mathbf{H}\mathbf{w}$, the hedge ratio and the question of contagion all depend on covariances. Rolling correlations of real markets move a great deal, through market integration, through crises and through the arrival of new asset classes such as Bitcoin. Before computing any daily correlation, the trading hours must be checked, because asynchronous closes bias daily correlations towards zero.

## @end

The section has motivated a model in which correlations change over time; rolling windows show the change but do not explain it and depend on an arbitrary window length. The next section sets up the general framework, a conditional covariance matrix that is updated every day from past shocks, and the conditions it must satisfy.

# The multivariate GARCH framework

The section defines the multivariate GARCH model in general form: the vector of returns, the conditional covariance matrix $\mathbf{H}_t$, the conditions of symmetry and positive definiteness, and estimation by maximum likelihood.

## The model for N return series {h=1a1911e2}

The model for $N$ returns has the same structure as the univariate GARCH model, written with vectors and matrices:

$$ \begin{gathered} \mathbf{r}_t = \boldsymbol{\mu} + \boldsymbol{\varepsilon}_t, \qquad \boldsymbol{\varepsilon}_t = \mathbf{H}_t^{1/2}\boldsymbol{\eta}_t, \\ \mathbf{H}_t = \Var(\mathbf{r}_t \mid \mathcal{F}_{t-1}). \end{gathered} $$ {#eq:mgarch}

Here $\mathbf{r}_t$ is the vector of the $N$ returns on day $t$, $\boldsymbol{\mu}$ the vector of means (or the forecast of a VAR model, as in [Chapter 6](../EN/Courses/chapter6_var_models_granger_causality.pdf)), and $\boldsymbol{\eta}_t$ a vector of independent and identically distributed (i.i.d.) innovations with mean zero and covariance matrix $\mathbf{I}_N$. The matrix $\mathbf{H}_t^{1/2}$ is a square root of $\mathbf{H}_t$, for instance the Cholesky factor $\mathbf{L}_t$ with $\mathbf{L}_t\mathbf{L}_t^\top = \mathbf{H}_t$; any such root gives the shocks $\boldsymbol{\varepsilon}_t$ the conditional covariance matrix $\mathbf{H}_t$.

The conditional covariance matrix (*conditional covariance matrix*) $\mathbf{H}_t$ is computed from the information $\mathcal{F}_{t-1}$ available up to day $t-1$. Its diagonal holds the conditional variances $h_{ii,t} = \sigma_{i,t}^2$, exactly the quantities of [Chapter 5](../EN/Courses/chapter5_conditional_volatility_garch.pdf). Off the diagonal are the conditional covariances $h_{ij,t}$, and the conditional correlation follows from them:

$$ \rho_{ij,t} = \frac{h_{ij,t}}{\sqrt{h_{ii,t}\,h_{jj,t}}}. $$ {#eq:rho}

A multivariate GARCH (MGARCH) model is a rule that computes $\mathbf{H}_t$ from the past shocks $\boldsymbol{\varepsilon}_{t-1}$ and from $\mathbf{H}_{t-1}$. The models of the following sections differ only in this rule.

## Two conditions on Hₜ {h=a0bc9d5c}

The first condition is symmetry: the covariance of asset $i$ with asset $j$ is the covariance of $j$ with $i$, so $h_{ij,t} = h_{ji,t}$ and only the $N(N+1)/2$ elements on and below the diagonal are distinct. The operator $\mathrm{vech}$ stacks these elements column by column in a vector; for two assets, $\mathrm{vech}(\mathbf{H}) = (h_{11}, h_{12}, h_{22})^\top$. The number of distinct elements grows quickly: 3 for $N = 2$, 55 for $N = 10$ and 1 275 for $N = 50$.

The second condition is positive definiteness (*positive definiteness*): $\mathbf{w}^\top\mathbf{H}_t\mathbf{w} > 0$ for every non-zero vector $\mathbf{w}$. By @eq:pvar the left side is the variance of the portfolio with weights $\mathbf{w}$, so the condition says that every portfolio has a positive variance. For two assets it reduces to $h_{11,t} > 0$ and $h_{11,t}h_{22,t} - h_{12,t}^2 > 0$, which is the same as $|\rho_{12,t}| < 1$.

The example on the slide shows what goes wrong without this condition. With $h_{11} = 4$, $h_{22} = 1$ and $h_{12} = 2.5$ the determinant is $4 - 6.25 < 0$, and @eq:rho gives a "correlation" of 1.25, which no pair of random variables can have. A model that computes each element of $\mathbf{H}_t$ separately can produce such matrices. A good MGARCH model guarantees both conditions for every $t$, whatever the shocks, and does so with few parameters.

## Estimation by maximum likelihood {h=ae6a4cb9}

If the innovations $\boldsymbol{\eta}_t$ follow the multivariate Normal distribution, the shocks $\boldsymbol{\varepsilon}_t$ are conditionally Normal with covariance matrix $\mathbf{H}_t$, and the log-likelihood of the sample is

$$ \ell(\theta) = -\frac12\sum_{t=1}^{T}\left(N\log 2\pi + \log|\mathbf{H}_t| + \boldsymbol{\varepsilon}_t^\top\mathbf{H}_t^{-1}\boldsymbol{\varepsilon}_t\right), $$ {#eq:loglik}

where $|\mathbf{H}_t|$ is the determinant and $\theta$ collects all parameters. For $N = 1$ the determinant is $\sigma_t^2$, the quadratic form is $\varepsilon_t^2/\sigma_t^2$, and @eq:loglik becomes the GARCH likelihood of [Chapter 5](../EN/Courses/chapter5_conditional_volatility_garch.pdf). The two terms play the same roles as there: the determinant penalises large variances, the quadratic form penalises shocks that are large relative to the predicted covariance.

Daily returns have fat tails, so the Normal assumption is not literally true. The Normal likelihood is then used as a quasi-maximum likelihood (QML): the estimates of the parameters of $\mathbf{H}_t$ remain consistent as long as $\mathbf{H}_t$ is correctly specified, but the usual standard errors are wrong and need a robust (sandwich) correction.

The practical difficulty is computational. Each evaluation of $\ell(\theta)$ requires the inverse and the determinant of $\mathbf{H}_t$ for every day of the sample, and a numerical optimiser evaluates $\ell(\theta)$ many times. The cost grows quickly with $N$ and, above all, with the number of parameters, which is the subject of the next section.

## Recap: The multivariate GARCH framework {h=8b2829d6}

The conditional covariance matrix $\mathbf{H}_t$ has the conditional variances on its diagonal and the covariances off it, and the conditional correlation follows from @eq:rho. A valid model must keep $\mathbf{H}_t$ symmetric, with $N(N+1)/2$ distinct elements, and positive definite on every day. Estimation uses the multivariate Normal likelihood @eq:loglik, as a quasi-likelihood when returns have fat tails.

## @end

The framework leaves open the rule that updates $\mathbf{H}_t$. The next section presents the two classic answers, VEC and BEKK, which model all elements of the matrix at once, and shows the price they pay in parameters.

# VEC and BEKK models

The section presents the most general linear MGARCH model, VEC, and the BEKK model, which guarantees positive definiteness by construction. Counting their parameters shows why neither can be used for more than a handful of assets.

## The VEC model {h=36a94139}

@BollerslevEngleWooldridge1988 wrote the most general linear extension of GARCH(1,1): every distinct element of $\mathbf{H}_t$ depends on every past squared shock and cross-product and on every element of $\mathbf{H}_{t-1}$. With the $\mathrm{vech}$ operator, the VEC(1,1) model is

$$ \begin{aligned} \mathrm{vech}(\mathbf{H}_t) = \mathbf{c} &+ \mathbf{A}\,\mathrm{vech}(\boldsymbol{\varepsilon}_{t-1}\boldsymbol{\varepsilon}_{t-1}^\top) \\ &+ \mathbf{B}\,\mathrm{vech}(\mathbf{H}_{t-1}). \end{aligned} $$ {#eq:vec}

The matrix $\boldsymbol{\varepsilon}_{t-1}\boldsymbol{\varepsilon}_{t-1}^\top$ holds yesterday's squared shocks on its diagonal and their cross-products off it. The vector $\mathbf{c}$ holds constants, $\mathbf{A}$ measures the reaction to shocks and $\mathbf{B}$ the persistence, as $\omega$, $\alpha$ and $\beta$ do in a univariate GARCH.

The generality is expensive. With $k = N(N+1)/2$ distinct elements, $\mathbf{c}$ has $k$ entries and $\mathbf{A}$ and $\mathbf{B}$ are $k \times k$, so the model has $k + 2k^2$ parameters: 21 for two assets and 6 105 for ten. Worse, nothing in @eq:vec keeps $\mathbf{H}_t$ positive definite; the restrictions that would do so are complicated and hard to impose during estimation.

The diagonal VEC keeps only the diagonal of $\mathbf{A}$ and $\mathbf{B}$, so that each element follows its own GARCH(1,1):

$$ h_{ij,t} = c_{ij} + a_{ij}\varepsilon_{i,t-1}\varepsilon_{j,t-1} + b_{ij}h_{ij,t-1}. $$

This leaves $3k$ parameters, 9 for $N = 2$, but positive definiteness is still not guaranteed, because each element is updated without regard to the others.

## The BEKK model {h=a76deaa8}

@EngleKroner1995 proposed a form named after Baba, Engle, Kraft and Kroner (BEKK) that is positive definite by construction:

$$ \begin{aligned} \mathbf{H}_t = \mathbf{C}\mathbf{C}^\top &+ \mathbf{A}^\top\boldsymbol{\varepsilon}_{t-1}\boldsymbol{\varepsilon}_{t-1}^\top\mathbf{A} \\ &+ \mathbf{B}^\top\mathbf{H}_{t-1}\mathbf{B}, \end{aligned} $$ {#eq:bekk}

with $\mathbf{C}$ lower triangular and $\mathbf{A}$, $\mathbf{B}$ full $N \times N$ matrices. $\mathbf{A}$ describes the reaction to yesterday's shocks and $\mathbf{B}$ the persistence of $\mathbf{H}_{t-1}$; the elements of the constant matrix $\mathbf{C}\mathbf{C}^\top$ are written $c_{ij}^*$.

The guarantee comes from the form of each term, a "square" $\mathbf{X}^\top\mathbf{M}\mathbf{X}$ of a positive semidefinite matrix $\mathbf{M}$: the identity in the first term (with $\mathbf{X} = \mathbf{C}^\top$), the outer product $\boldsymbol{\varepsilon}_{t-1}\boldsymbol{\varepsilon}_{t-1}^\top$ in the second and $\mathbf{H}_{t-1}$ in the third. For any vector $\mathbf{w}$,

$$ \mathbf{w}^\top\mathbf{X}^\top\mathbf{M}\mathbf{X}\mathbf{w} = (\mathbf{X}\mathbf{w})^\top\mathbf{M}(\mathbf{X}\mathbf{w}) \ge 0, $$

and a constant term $\mathbf{C}\mathbf{C}^\top$ of full rank makes the sum strictly positive. The number of parameters is $N(N+1)/2 + 2N^2$: 11 for two assets, 255 for ten.

The off-diagonal elements of a full $\mathbf{A}$ have an economic meaning: they measure volatility spillovers (*volatility spillovers*), the effect of a shock to asset 1 on the variance of asset 2. Restricting $\mathbf{A}$ and $\mathbf{B}$ to be diagonal gives the diagonal BEKK, in which the covariance follows

$$ \begin{aligned} h_{12,t} = c_{12}^* &+ a_{11}a_{22}\,\varepsilon_{1,t-1}\varepsilon_{2,t-1} \\ &+ b_{11}b_{22}\,h_{12,t-1}. \end{aligned} $$

The scalar BEKK goes further, with $\mathbf{A} = a\mathbf{I}$ and $\mathbf{B} = b\mathbf{I}$, so that all elements share the same dynamics. Both restrictions save parameters and remove the spillovers.

## Worked example: one step of a diagonal BEKK {h=f9799888}

The covariance equation of a diagonal BEKK can be computed by hand. The given values are $c_{12}^* = 0.02$, $a_{11} = 0.30$, $a_{22} = 0.25$, $b_{11} = 0.94$, $b_{22} = 0.95$, yesterday's covariance $h_{12,t-1} = 0.5$ and yesterday's shocks $\varepsilon_{1,t-1} = -2$ and $\varepsilon_{2,t-1} = -1.5$. The coefficients of the covariance equation are products: $a_{11}a_{22} = 0.075$ for the reaction to the cross-product of shocks and $b_{11}b_{22} = 0.893$ for persistence.

The cross-product of the shocks is $(-2)(-1.5) = 3.000$, positive because both assets fell. Today's covariance is

$$ \begin{aligned} h_{12,t} &= 0.020 + 0.075 \cdot 3.000 + 0.893 \cdot 0.5 \\ &= 0.020 + 0.225 + 0.4465 = 0.6915. \end{aligned} $$

A joint fall raises the covariance from 0.5 to 0.6915; a shock with opposite signs would have given a negative cross-product and lowered it.

The example also shows what the diagonal model cannot do. Because $a_{12} = a_{21} = 0$, a shock to asset 1 affects the variance of asset 2 only through the covariance, never directly: the diagonal BEKK has no volatility spillovers.

## The curse of dimensionality {h=02dca81d}

The table counts the parameters of the variance equation of each model for 2, 5, 10 and 50 assets; the means $\boldsymbol{\mu}$ are not counted. For CCC and DCC, introduced in the next section, the count is 3 GARCH parameters per asset plus the $N(N-1)/2$ correlations, and DCC adds 2 more.

The rows differ by orders of magnitude. The full VEC needs 21 parameters for two assets, 465 for five, 6 105 for ten and 3 252 525 for fifty: for fifty assets it has more parameters than any sample of daily data has observations. The full BEKK is much more economical but still reaches 6 275 parameters for fifty assets. The diagonal BEKK, CCC and DCC stay close to each other, with 1 375 to 1 377 parameters for fifty assets, and the scalar BEKK is the cheapest, with 1 277. This growth of the number of parameters with the dimension is called the curse of dimensionality (*curse of dimensionality*).

## Parameters against the number of assets {h=cbee4879}

The chart plots the number of parameters of the variance equation, on a logarithmic vertical axis, against the number of assets $N$, from 2 to 50, for six models: VEC(1,1), BEKK(1,1), diagonal VEC, diagonal BEKK, DCC and CCC. The scalar BEKK, from the table, is not drawn.

On a logarithmic scale a constant vertical distance means a constant ratio. The VEC curve lies far above all others and keeps moving away, passing $10^6$ at $N = 50$. The full BEKK and the diagonal VEC lie in between, and the last three curves (diagonal BEKK, DCC and CCC) almost coincide, because their counts differ by at most two parameters.

## Interpreting the parameter counts {h=01764b4c}

The leading terms of the formulas give the growth rates. For VEC, $2k^2$ with $k \approx N^2/2$ grows like $N^4/2$; for the full BEKK the count grows like $2N^2$. Both become impossible to estimate beyond a handful of assets, because a numerical optimiser cannot search reliably over hundreds or thousands of parameters with a likelihood like @eq:loglik.

CCC and DCC grow like $N^2/2$, but the count hides an important difference: the correlations are not found by numerical optimisation. As the next section shows, they are sample correlations of standardised residuals, computed directly. The GARCH parameters are estimated one series at a time, with three parameters each. The only joint optimisation in DCC concerns two parameters, $a$ and $b$, whatever the number of assets. This is why DCC can be used where VEC and BEKK cannot.

## Recap: VEC and BEKK {h=1f7d8365}

VEC is the most general linear model, with $k + 2k^2$ parameters and no guarantee of positive definiteness. BEKK is positive definite by construction, and the off-diagonal elements $a_{ij}$ of its matrix $\mathbf{A}$ measure volatility spillovers. The diagonal and scalar versions cut the number of parameters but remove the spillovers.

## @end

VEC and BEKK model the whole matrix $\mathbf{H}_t$ in one equation, and this is the source of their cost. The next section takes a different route: it splits $\mathbf{H}_t$ into volatilities, which univariate GARCH models already handle, and a correlation matrix with very few dynamic parameters.

# Constant and dynamic conditional correlations

The section decomposes the covariance matrix into volatilities and correlations, presents the constant conditional correlation model and Engle's dynamic conditional correlation model, explains their estimation in two steps and the test of DCC against CCC.

## Splitting volatilities and correlations {h=b8d1efa0}

Any covariance matrix can be written as volatilities times correlations:

$$ \begin{gathered} \mathbf{H}_t = \mathbf{D}_t\mathbf{R}_t\mathbf{D}_t, \\ \mathbf{D}_t = \mathrm{diag}(\sigma_{1,t}, \dots, \sigma_{N,t}), \end{gathered} $$ {#eq:drd}

where $\mathbf{R}_t$ is the conditional correlation matrix, with ones on the diagonal. Element by element, $h_{ij,t} = \rho_{ij,t}\,\sigma_{i,t}\sigma_{j,t}$, which is @eq:rho read the other way. Each volatility follows a univariate GARCH(1,1) model [@Bollerslev1986]:

$$ \sigma_{i,t}^2 = \omega_i + \alpha_i\varepsilon_{i,t-1}^2 + \beta_i\sigma_{i,t-1}^2. $$

Dividing each shock by its own volatility gives the standardised residuals (*standardised residuals*) $z_{i,t} = \varepsilon_{i,t}/\sigma_{i,t}$. Each has conditional variance one, and the conditional covariance of $z_{i,t}$ and $z_{j,t}$ is $h_{ij,t}/(\sigma_{i,t}\sigma_{j,t}) = \rho_{ij,t}$. The conditional covariance matrix of the vector $\mathbf{z}_t$ is therefore exactly $\mathbf{R}_t$: once the volatilities are removed, only the correlations are left to model.

The simplest choice is the constant conditional correlation (CCC) model of @Bollerslev1990, with $\mathbf{R}_t = \mathbf{R}$ for all $t$. $\mathbf{H}_t$ is then positive definite whenever $\mathbf{R}$ is and the GARCH variances are positive, and $\mathbf{R}$ is estimated by the sample correlation matrix of the $z_{i,t}$. In CCC the covariances still move, but only through the volatilities. The rolling correlations of the first section contradict this assumption.

## Robert Engle and the DCC model {h=d1f62457}

Robert F. Engle, born in 1942, received the Nobel Prize in economics in 2003 for methods of analysing economic time series with time-varying volatility. @Engle2002 proposed letting the correlations follow a recursion of the same form as GARCH. An auxiliary matrix $\mathbf{Q}_t$, with elements $q_{ij,t}$, is updated each day from yesterday's vector of standardised residuals:

$$ \begin{aligned} \mathbf{Q}_t &= (1 - a - b)\,\bar{\mathbf{Q}} + a\,\mathbf{z}_{t-1}\mathbf{z}_{t-1}^\top \\ &\quad + b\,\mathbf{Q}_{t-1}, \\ \rho_{ij,t} &= \frac{q_{ij,t}}{\sqrt{q_{ii,t}\,q_{jj,t}}}. \end{aligned} $$ {#eq:dcc}

The rescaling in the second part is needed because $\mathbf{Q}_t$ is a weighted average of matrices whose diagonals are not exactly one; dividing by the square roots of the diagonal turns it into a proper correlation matrix with ones on the diagonal.

$\bar{\mathbf{Q}}$ is the unconditional correlation matrix of $\mathbf{z}_t$, and it is set to its sample value before the optimisation. This is correlation targeting (*correlation targeting*): the recursion is anchored at the observed long-run correlation, and only $a$ and $b$ remain to be estimated. The parameter $a \ge 0$ is the reaction to yesterday's joint shock, $b \ge 0$ the persistence, and $a + b < 1$ ensures that $\mathbf{Q}_t$ reverts to $\bar{\mathbf{Q}}$.

With $a = b = 0$ the recursion gives $\mathbf{Q}_t = \bar{\mathbf{Q}}$ on every day, which is CCC. DCC therefore contains CCC as a special case, and this nesting is what makes a likelihood ratio test between the two possible.

## Two-step estimation {h=c4950e91}

Substituting @eq:drd into the log-likelihood @eq:loglik splits it into two parts, $\ell = \ell_V(\theta_1) + \ell_C(\theta_1, a, b)$. The volatility part $\ell_V$ depends only on $\theta_1$, the GARCH parameters of all series, and is the sum of the univariate GARCH likelihoods. The correlation part $\ell_C$ depends on $\theta_1$ through the standardised residuals and on $a$ and $b$.

The split suggests estimating in two steps. In step 1 a GARCH(1,1) is fitted to each series separately, as in [Chapter 5](../EN/Courses/chapter5_conditional_volatility_garch.pdf), and the fitted volatilities $\hat\sigma_{i,t}$ and standardised residuals $\hat z_{i,t} = \hat\varepsilon_{i,t}/\hat\sigma_{i,t}$ are kept. In step 2, $\bar{\mathbf{Q}}$ is set to the sample correlation of $\hat{\mathbf{z}}_t$ and the correlation log-likelihood is maximised over $(a, b)$ only:

$$ \ell_C(a, b) = -\frac12\sum_{t=1}^{T}\left(\log|\mathbf{R}_t| + \hat{\mathbf{z}}_t^\top\mathbf{R}_t^{-1}\hat{\mathbf{z}}_t - \hat{\mathbf{z}}_t^\top\hat{\mathbf{z}}_t\right). $$ {#eq:lc}

The estimator is consistent, fast and works for large $N$. Its weak point is inference: the standard errors of step 2 treat the step-1 estimates as known and ignore their estimation error, so they are too small.

The test against CCC compares the maximised correlation log-likelihoods: $LR = 2(\ell_C^{DCC} - \ell_C^{CCC})$, compared with the $\chi^2(2)$ distribution, whose 5% critical value is 5.99. A large value rejects constant correlation. Because the null hypothesis $a = b = 0$ lies on the boundary of the parameter space, the $\chi^2(2)$ critical value is conservative: a rejection with it is a safe rejection. @Tse2000 proposed another test of constant correlations, based on the Lagrange multiplier principle.

## Worked example: one step of the DCC recursion {h=adb4843d}

The recursion @eq:dcc can be followed by hand for two assets. The given values are $a = 0.05$, $b = 0.93$, a long-run correlation $\bar q_{12} = 0.5$ (with $\bar q_{11} = \bar q_{22} = 1$), yesterday's matrix $\mathbf{Q}_{t-1}$ with ones on the diagonal and 0.5 off it, and yesterday's standardised residuals $z_{1,t-1} = -2$ and $z_{2,t-1} = -2.5$, a large joint fall. The weight of the long-run matrix is $1 - a - b = 0.020$.

The off-diagonal element and, because the squared residuals enter them, the diagonal elements are

$$ \begin{aligned} q_{12,t} &= 0.020 \cdot 0.5 + 0.05 \cdot (-2)(-2.5) \\ &\quad + 0.93 \cdot 0.5 \\ &= 0.010 + 0.250 + 0.465 = 0.725, \\ q_{11,t} &= 0.020 + 0.05 \cdot 4 + 0.93 = 1.150, \\ q_{22,t} &= 0.020 + 0.05 \cdot 6.25 + 0.93 = 1.2625. \end{aligned} $$

The correlation is $\rho_{12,t} = 0.725/\sqrt{1.150 \cdot 1.2625} = 0.602$.

The large joint fall raises tomorrow's correlation from 0.50 to 0.602. The rescaling matters: without it, the value 0.725 of $q_{12,t}$ would be read as the correlation and overstate the reaction. Large shocks raise the diagonal of $\mathbf{Q}_t$ as well, and the division by $\sqrt{q_{11,t}q_{22,t}}$ takes this into account.

## Interpreting a and b {h=c65ef033}

The parameter $a$ is the news coefficient: it measures how much one day's joint shock moves the correlation. Typical daily estimates lie between 0.01 and 0.05, so a single day moves the correlation only a little, and a large change requires a run of joint shocks.

The sum $a + b$ is the persistence. Taking the long-run value away from both sides of @eq:dcc and replacing $\mathbf{z}_{t-1}\mathbf{z}_{t-1}^\top$ by its conditional mean, approximately $\mathbf{Q}_{t-1}$, shows that a deviation of $q_{ij,t}$ from $\bar q_{ij}$ is expected to shrink by the factor $a + b$ each day. After $n$ days it is multiplied by $(a+b)^n$, and it is halved when $(a+b)^n = 0.5$, which gives the half-life (*half-life*) $\ln 0.5/\ln(a + b)$ days. With $a + b = 0.98$ the half-life is 34 days; with $0.99$ it is 69 days. Near one, small changes in $a + b$ change the half-life a great deal.

The reading is the same as for $\alpha$ and $\alpha + \beta$ in a GARCH(1,1) model in [Chapter 5](../EN/Courses/chapter5_conditional_volatility_garch.pdf). Two variants address limits of the basic model: the asymmetric DCC of @CappielloEngleSheppard2006 lets joint falls move correlations more than joint rises, and the corrected DCC of @Aielli2013 modifies the recursion so that the two-step estimator remains consistent.

## Recap: CCC and DCC {h=b77ecd26}

The decomposition @eq:drd writes $\mathbf{H}_t$ as GARCH volatilities times a correlation matrix. In CCC the correlation matrix is constant; in DCC the auxiliary matrix $\mathbf{Q}_t$ follows a GARCH-like recursion with two parameters, $a$ and $b$. Estimation proceeds in two steps, univariate GARCH models first and $(a, b)$ with correlation targeting second, and a likelihood ratio test compares DCC with CCC.

## @end

The section has built the model; the next applies it to the pair of the S&P 500 and the DAX, following both steps of the estimation and asking whether the correlation is constant and how it behaves in the crises of 2008 and 2020.

# DCC for the S&P 500 and the DAX

The section estimates a DCC model for daily returns of the S&P 500 and the DAX from January 2000 to September 2026, step by step, and uses it to study correlations in the crises of 2008 and 2020.

## Step 1: two univariate GARCH models {h=59c6a5fe}

The data are daily log returns in percent on the 6 605 days on which both markets traded, from 4 January 2000 to 18 September 2026, from the data provider EOD Historical Data (EODHD). Each series receives a GARCH(1,1) with a constant mean, estimated by Gaussian QML. The table gives the mean $\mu$, the three GARCH parameters and the persistence $\alpha + \beta$, together with the excess kurtosis of the raw returns and of the standardised residuals.

The GARCH estimates have the usual shape for equity indices. For the S&P 500, $\alpha = 0.122$ and $\beta = 0.857$, with persistence 0.980; for the DAX, $\alpha = 0.098$ and $\beta = 0.886$, with persistence 0.984. The reaction to news is moderate and volatility shocks fade slowly.

The kurtosis columns show what step 1 achieves. The excess kurtosis of the S&P 500 falls from 10.5 for the returns to 1.8 for the standardised residuals, and that of the DAX from 6.2 to 1.4: most of the fat tails come from changing volatility. What remains is still above zero, so the tails of $z_t$ are fatter than those of the Normal distribution. The correlation of the returns is 0.599 and that of the residuals is $\bar q_{12} = 0.575$; the latter is the CCC estimate and the anchor of the DCC recursion.

## Step 1: conditional volatilities {h=b8cfe6d6}

The chart plots the one-step-ahead GARCH(1,1) volatility of each index, annualised as $\hat\sigma_{i,t}\sqrt{252}$ and expressed in percent, on the vertical axis against time from 2000 to 2026. The two lines are the S&P 500 and the DAX, and the shaded bands mark the crises of 2008 and 2020.

In calm years the two volatilities stay low, with the DAX usually somewhat above the S&P 500. The largest peaks are in the autumn of 2008 and in March 2020, and smaller peaks appear in 2001–2003, 2011, 2022 and April 2025. The two lines rise and fall together.

## Interpreting the volatilities {h=39dcf4f3}

Both volatilities peak in the same weeks: the S&P 500 reaches 109% on 17 March 2020 and the DAX 80% on 25 March 2020. Volatility clustering happens jointly. This alone makes the covariance $h_{12,t} = \rho_{12,t}\sigma_{1,t}\sigma_{2,t}$ rise sharply in a crisis, even if the correlation did not change at all.

The persistence of 0.980 and 0.984 means that volatility shocks fade slowly, as in [Chapter 5](../EN/Courses/chapter5_conditional_volatility_garch.pdf). Step 2 asks a different question from step 1: once the volatilities have been divided out, is the correlation of what remains constant? This separation is the point of DCC: it distinguishes a rise in covariance caused by higher volatility from a rise caused by a closer link.

## Step 2: the estimated DCC {h=5810c008}

Step 2 estimates the two DCC parameters on the standardised residuals of step 1. The news coefficient is $a = 0.0152$ with a standard error (SE) of 0.0035, and the persistence parameter is $b = 0.9766$ with an SE of 0.0065. The sum $a + b = 0.9918$ implies a half-life of a correlation shock of 85 days.

The estimate of $a$ is small, at the low end of the typical range, but more than four times its standard error, so the correlation does react to news. The sum close to one means that a deviation of the correlation from its long-run value lasts for months. The SEs come from the Hessian of the step-2 likelihood and ignore the estimation error of step 1, so they are too small; even allowing for this, $a$ is clearly separated from zero.

The likelihood ratio statistic against CCC is 131.8, far above the 5% critical value of 5.99. Constant correlation is rejected: the correlation between the S&P 500 and the DAX changes over time even after the changes in volatility have been removed.

## DCC correlation of the S&P 500 and the DAX {h=be2d194f}

The chart plots three estimates of the correlation on the vertical axis against time, daily from 4 January 2000 to 18 September 2026: the one-step-ahead DCC correlation $\hat\rho_{12,t}$, the 250-day rolling correlation and the constant CCC value $\bar q_{12} = 0.575$ as a dashed horizontal line. The shaded bands mark the crises of 2008 and 2020.

The rolling correlation starts only in 2001, after its first window is complete, whereas the DCC correlation is available from the start of the sample. The two dynamic estimates follow the same long swings: high values in 2002–2003 and 2010–2012, lower values in 2005–2006 and 2018–2019.

## Interpreting the DCC correlation {h=e6edbc17}

The DCC correlation ranges from 0.29 in March 2000 to 0.75 in January 2010, around a mean of 0.57. It follows the same swings as the rolling correlation, but it reacts within days instead of a year, and it needs no choice of window. Its weights decline geometrically with the age of the observation, at rate $b$, rather than being equal inside a window and zero outside it.

The rolling estimate is more erratic: its standard deviation over the sample is 0.122, against 0.083 for DCC. It jumps when an extreme day enters the window and jumps again, in the opposite direction, exactly 250 days later when the day leaves it. The fall of the rolling correlation in 2025–2026 is an example. The asynchronous days of April 2025, described in the first section, enter the window and pull it down to about 0.21 for exactly 250 days, after which it jumps back. The DCC correlation registers the same days but forgets them gradually.

## Correlations in crises {h=44ea31c7}

The table compares three periods: a calm period, 2017–2019, and the two crises, September 2008 – March 2009 (the months that followed the bankruptcy of Lehman Brothers, shown in the photograph) and February – April 2020. For each period it gives the average DCC correlation between the S&P 500 and the DAX and the average annualised GARCH volatility of the S&P 500. In the calm period the correlation averages 0.56 and the volatility 13%; in 2008–2009, 0.63 and 51%; in 2020, 0.70 and 58%.

Correlations are higher in crises, but the change is moderate compared with the change in volatility, which roughly quadruples. Most of the rise in covariance during a crisis therefore comes from volatility, a smaller part from the correlation.

The comparison needs care. @ForbesRigobon2002 showed that the sample correlation of returns rises mechanically when volatility rises, even if the underlying link between markets is unchanged: a larger common shock dominates the idiosyncratic noise. A higher correlation of raw returns in a crisis is therefore not by itself evidence of contagion. DCC reduces this effect because it measures correlation on the standardised residuals $z_t$, not on the returns $r_t$, so the rise in volatility is removed before the correlation is computed.

## DCC correlation in the crises of 2008 and 2020 {h=6f291dc7}

The chart has two panels, one for each crisis: June 2008 – June 2009 on the left and December 2019 – August 2020 on the right. Each panel plots the DCC correlation of the S&P 500 and the DAX on the left axis, from 0 to 1, and the annualised GARCH volatility of the S&P 500, in percent, on the right axis, against time.

The two axes make it possible to compare the timing of the two quantities, not their levels. In both panels the volatility line rises and falls steeply, while the correlation line moves in a much narrower range.

## Interpreting the two crises {h=35d1672a}

The two crises unfolded differently. In 2008 the volatility of the S&P 500 peaked at 90% on 16 October 2008, while the correlation rose slowly, from 0.57 to its maximum of 0.68, reached only on 23 June 2009, months after the volatility peak. The crisis started in the US financial system and spread to Europe gradually.

In 2020 the order was reversed. The correlation jumped to 0.73 on 28 February 2020, before the volatility peak of 109% on 17 March 2020. A global shock, the COronaVIrus Disease 2019 (COVID-19) pandemic, hit both markets on the same days, and the run of joint falls drove the DCC recursion upwards quickly.

Correlations and volatilities need not peak together. A model that ties them, such as a single rolling covariance, would mix the two stories; modelling them separately, as DCC does, shows the difference between a crisis that spreads and a shock that strikes everywhere at once.

## Recap: S&P 500 and DAX {h=cbec3c12}

Step 1 gives persistent GARCH volatilities, with persistence 0.980 and 0.984, which peak together. Step 2 gives $a + b = 0.9918$ and a likelihood ratio of 131.8 against CCC: the correlation is not constant. Correlations are higher in 2008 and 2020, but the largest change during a crisis is in volatility.

## @end

Two assets show the method; a portfolio usually has more. The next section estimates a single DCC model for five markets of very different kinds, including the Romanian equity market, the leu and Bitcoin, and compares their correlations in calm times and in the 2020 crash.

# Five markets: S&P 500, DAX, BET, EUR/RON, Bitcoin

The section fits one DCC model to five assets, two large equity indices, the Romanian BET, the EUR/RON exchange rate and Bitcoin, and compares the average correlations in a calm period and in the COVID-19 crash.

## A five-asset DCC {h=61e76d52}

The data are daily log returns on the 2 815 days on which all five markets trade, from 6 January 2015. The BET is the index of the Bucharest Stock Exchange, shown in the photograph. For EUR/RON the reference rate of the National Bank of Romania (BNR), fixed at 13:00 Bucharest time, is used, and for Bitcoin the close of the day in Coordinated Universal Time (UTC). The closing times differ even more than in the first section, so daily correlations of these series carry the asynchronous-trading bias discussed there.

Step 1 gives five GARCH(1,1) models with persistence 0.961 for the S&P 500, 0.965 for the DAX, 0.941 for the BET, 0.985 for EUR/RON and 0.958 for Bitcoin. Step 2 gives $\hat a = 0.0088$ (SE 0.0019) and $\hat b = 0.9782$ (SE 0.0062), so the half-life of a correlation shock is 53 days. The likelihood ratio against CCC is 88.5, again far above the 5% critical value of 5.99.

The model has 10 correlation pairs but only one pair of dynamics parameters $(a, b)$ for all of them. The 10 long-run correlations come from $\bar{\mathbf{Q}}$, the sample correlation of the standardised residuals, so the optimiser of step 2 still works on two parameters only.

## Dynamic correlations of four pairs {h=76776761}

The chart plots the one-step-ahead DCC correlations of four of the ten pairs on the vertical axis against time, from 2015 to 2026: the S&P 500 and the DAX, the BET and the DAX, the S&P 500 and Bitcoin, and the BET and EUR/RON. A dashed line marks zero, and the shaded band marks the COVID-19 crash, from mid-February to June 2020.

The four lines are clearly separated in level: the S&P 500 and the DAX at the top, the BET and the DAX and the S&P 500 and Bitcoin in the middle, and the BET and EUR/RON around or below zero. Their movements over time are the subject of the next frame.

## Interpreting the four pairs {h=19ada51d}

The S&P 500 and the DAX have the strongest link, with a mean correlation of 0.54 over 2015–2026. The BET and the DAX have a mean of 0.32 and move between 0.15 and 0.53, the maximum reached in April 2022. The jump of this pair at the start of the 2020 crash is the largest of the four.

The S&P 500 and Bitcoin range from $-0.05$ to 0.42, the maximum in May 2022, with a mean of 0.21. Before 2020 the correlation stayed low; after 2020 it stayed clearly higher. The diversification benefit of Bitcoin for an equity portfolio has shrunk.

The BET and EUR/RON have a correlation close to zero, with a mean of $-0.06$. The exchange rate is quoted as lei per euro, so a rise of EUR/RON is a weaker leu, and the small negative correlation means that a weaker leu tends to come with a weaker BET. One feature of the chart is a consequence of the model rather than of the data: all four lines share the same $(a, b)$, so pairs that might adjust at different speeds are forced to the same rhythm. This is the price of the parsimony of DCC.

## Average correlations: calm against crisis {h=c4799eca}

The chart shows two heatmaps of the average one-step-ahead DCC correlation matrix of the five assets. The left one averages over the calm period 2017–2019 and the right one over the COVID-19 crash, from mid-February to June 2020, 89 common trading days. Rows and columns are the five assets, each cell holds the average correlation, and the colour scale runs from $-1$ (blue) to $1$ (dark red).

The diagonals are ones by construction. The cells for EUR/RON are close to white in both matrices, while the cells linking the BET and Bitcoin with the large equity markets become clearly darker in the crisis panel.

## Interpreting the two matrices {h=be41040d}

The average of the ten pairwise correlations rises from 0.12 in the calm period to 0.19 in the crash. The rise is uneven. For the BET and the DAX the average correlation goes from 0.28 to 0.48, for the S&P 500 and the BET from 0.22 to 0.38, and for the DAX and Bitcoin from 0.10 to 0.27. The S&P 500 and the DAX, already strongly linked, go from 0.55 to 0.64.

The largest rises concern the BET and Bitcoin, the assets weakly linked to the large markets in calm times, which are exactly the assets that were expected to diversify an equity portfolio. When a global shock arrives, these weak links become stronger, and the benefit of diversification shrinks when it matters most.

EUR/RON stays close to zero or slightly negative with all equity markets in both periods, from $-0.03$ to $-0.10$. In this set of five assets it is the only one that kept its diversifying role in the crash.

## Recap: Five markets {h=8b6685c3}

One DCC model describes the ten correlations of five assets with only two dynamics parameters, and the likelihood ratio of 88.5 rejects constant correlation. The correlations of the Romanian market with Europe and of Bitcoin with equities move a great deal. In the 2020 crash the weak links strengthened most, so diversification weakened in the crisis.

## @end

Correlations and volatilities are inputs; risk managers need numbers that can be checked against outcomes. The next section turns the conditional covariance matrix into the VaR of a portfolio and tests it on data that were not used for estimation.

# Portfolio VaR 1% with DCC

The section computes the Value at Risk of a portfolio from $\mathbf{H}_t$, presents the Kupiec and Christoffersen backtests and compares DCC, CCC and a static covariance matrix on an S&P 500 and DAX portfolio from 2015 to 2026.

## From Hₜ to the VaR of a portfolio {h=abf9f101}

A portfolio with weights $\mathbf{w}$ has the return $r_{p,t} = \mathbf{w}^\top\mathbf{r}_t$, with conditional mean $\mu_p = \mathbf{w}^\top\boldsymbol{\mu}$ and conditional volatility $\sigma_{p,t} = \sqrt{\mathbf{w}^\top\mathbf{H}_t\mathbf{w}}$, by @eq:pvar applied to the conditional covariance matrix. The VaR 1% is the loss that is exceeded with probability 1%, minus the 1% quantile of the portfolio return:

$$ \mathrm{VaR}_{0.01,t} = -q_{0.01}(r_{p,t}) = -\left(\mu_p + q\,\sigma_{p,t}\right). $$ {#eq:var}

The value of $q$ depends on the assumed distribution of the standardised portfolio return. Under the Normal distribution $q = z_{0.01} = -2.326$. Filtered historical simulation (FHS) uses instead the empirical 1% quantile of the past standardised portfolio returns $(r_{p,t} - \mu_p)/\sigma_{p,t}$, which takes the fat tails of the data into account while keeping the volatility dynamics of the model.

Three choices of $\mathbf{H}_t$ are compared: DCC; CCC, which has GARCH volatilities but a constant correlation; and a static matrix, a single sample covariance matrix used for all days. The static VaR is the same every day, the CCC VaR moves with the volatilities only, and the DCC VaR moves with both volatilities and correlation.

## Worked example: VaR 1% of two assets {h=a6fcd8db}

Today's forecasts are daily volatilities $\sigma_1 = 1.5\%$ and $\sigma_2 = 1.2\%$, the weights are 50% and 50% and the mean is zero. In a calm state with $\rho_t = 0.3$, @eq:pvar gives

$$ \begin{aligned} \sigma_p^2 &= 0.25 \cdot 2.25 + 0.25 \cdot 1.44 \\ &\quad + 2 \cdot 0.25 \cdot 0.3 \cdot 1.8, \end{aligned} $$

where $1.8 = 1.5 \cdot 1.2$ is the product of the volatilities. The portfolio volatility is $\sigma_p = 1.092\%$ and the Normal VaR 1% is $2.326 \cdot 1.092 = 2.54\%$.

In a crisis state with $\rho_t = 0.9$ and the same volatilities, $\sigma_p = 1.316\%$ and the VaR 1% is 3.06%. The jump of the correlation alone adds about a fifth to the VaR. In a real crisis the volatilities rise as well, so the actual increase is larger.

For a portfolio of 1 million lei, the VaR 1% is about 25.4 thousand lei in the calm case and 30.6 thousand lei in the crisis case. A model with a fixed correlation would report the first number on both days.

## Backtesting a VaR (1/2): frequency {h=840b70d1}

A VaR model is judged by backtesting: comparing its forecasts with the returns that actually followed. A violation (*violation*) occurs on day $t$ when $r_{p,t} < -\mathrm{VaR}_{0.01,t}$, that is when the loss exceeds the VaR. A correct VaR 1% has violations on 1% of the days, and these days occur at random, not in clusters. The two properties are tested separately.

The test of @Kupiec1995 checks the frequency. With $x$ violations in $n$ days and observed rate $\hat p = x/n$, the statistic

$$ LR_{uc} = -2\log\frac{(1-0.01)^{n-x}\,0.01^{x}}{(1-\hat p)^{n-x}\,\hat p^{x}} $$ {#eq:kupiec}

compares the likelihood of the observed violations under the nominal rate of 1% with their likelihood under the observed rate. The subscript $uc$ stands for unconditional coverage. Under a correct model $LR_{uc}$ follows approximately a $\chi^2(1)$ distribution, and the model is rejected at 5% if $LR_{uc} > 3.84$. The test rejects both too many violations, a VaR that is too low, and too few, a VaR that is too high.

## Backtesting a VaR (2/2): independence {h=acf958e7}

The test of @Christoffersen1998 checks independence: is a violation more likely on the day after a violation? It compares $\pi_{11}$, the probability of a violation given a violation yesterday, with $\pi_{01}$, the probability of a violation given no violation yesterday. For a correct model the two are equal, because a VaR that uses all available information leaves nothing predictable in the violations.

The statistic is built from counts: $n_{ij}$ is the number of days in state $j$ that follow a day in state $i$, where state 1 is a violation and 0 is none. The estimates are $\hat\pi_{01} = n_{01}/(n_{00}+n_{01})$ and $\hat\pi_{11} = n_{11}/(n_{10}+n_{11})$, and $\hat\pi$ is the overall violation rate. The statistic

$$ LR_{ind} = -2\log\frac{(1-\hat\pi)^{n_{00}+n_{10}}\,\hat\pi^{\,n_{01}+n_{11}}}{(1-\hat\pi_{01})^{n_{00}}\,\hat\pi_{01}^{\,n_{01}}\,(1-\hat\pi_{11})^{n_{10}}\,\hat\pi_{11}^{\,n_{11}}} $$

has a numerator that assumes $\pi_{01} = \pi_{11}$, no clustering, and a denominator that lets the two probabilities differ. Under independence it follows approximately $\chi^2(1)$; a small p-value rejects independence.

A backtest is meaningful only out of sample. All parameters are estimated on 2000–2014, 3 712 days, and the filters are then run forward on 2015–2026, 2 893 days, without re-estimating. Each day's VaR uses only information available on the previous day, as it would in practice.

## VaR 1% of an S&P 500 and DAX portfolio, 2015–2026 {h=b3009f37}

The chart plots, against time from 2015 to 2026, the daily return in percent of a portfolio with 50% in the S&P 500 and 50% in the DAX, together with minus the VaR 1% of two models: DCC with filtered historical simulation, a line that moves, and the static covariance matrix, a flat line. Red triangles mark the 48 violations of the DCC-FHS model and green triangles the 26 violations of the static model.

The DCC line follows the volatility of the returns: it stays close to the returns in calm years and drops far below them in March 2020, when the returns themselves swing widely. The static line stays at the same level throughout. The green triangles of the static model are concentrated in a few episodes, most visibly in early 2020; the red triangles of DCC are spread across the whole period.

## Interpreting the backtest {h=df962b77}

With 2 893 out-of-sample days, a correct VaR 1% should have about 29 violations. The table gives, for each model, the number and rate of violations, the Kupiec statistic $LR_{uc}$ and its p-value, the p-value of the Christoffersen independence test and the number of violations in February – April 2020. DCC with Normal quantiles has 65 violations, a rate of 2.25%, $LR_{uc} = 33.6$ with a p-value below 0.001 and an independence p-value of 0.003. DCC with FHS has 48 violations, 1.66%, $LR_{uc} = 10.6$, p-value 0.001, independence p-value 0.008. CCC with Normal quantiles has 57 violations, 1.97%, $LR_{uc} = 21.4$, p-value below 0.001, independence p-value 0.030. The static model has 26 violations, 0.90%, $LR_{uc} = 0.3$, p-value 0.577, independence p-value 0.001.

The dynamic models with Normal quantiles have too many violations, because the standardised residuals have fatter tails than the Normal distribution, as the kurtosis of step 1 showed and as [Chapter 5](../EN/Courses/chapter5_conditional_volatility_garch.pdf) found for single series. FHS replaces $-2.326$ by the empirical quantile $q = -2.59$ and brings the rate down to 1.66%, still above 1% but much closer. The remaining violations are also less clustered than for the static model: in February – April 2020 every dynamic model has 4 violations.

The static VaR passes the frequency test, and this is the instructive case. Its 26 violations are close to the expected 29, but 10 of them fall in February – April 2020, and after a violation the next day is also a violation with probability 11.5%, so independence is rejected. The static VaR is 2.95% every day: too high in calm years and far too low in a crash. Its correct average frequency comes from errors in both directions that offset each other. A dynamic VaR adapts in both directions; its remaining problem is the quantile, not the covariance.

## Recap: Portfolio VaR {h=cbcd6d6f}

The portfolio volatility is $\sigma_{p,t} = \sqrt{\mathbf{w}^\top\mathbf{H}_t\mathbf{w}}$ and the VaR 1% is $-(\mu_p + q\,\sigma_{p,t})$. A VaR is backtested out of sample with the Kupiec test, for the frequency of violations, and the Christoffersen test, for their independence. Dynamic covariances remove most of the clustering of violations; fat tails require a better quantile than the Normal one.

## @end

The VaR uses the whole matrix $\mathbf{H}_t$ through one quadratic form. The next application uses its elements separately: the ratio of a covariance to a variance gives the number of units of a hedging asset to sell, and DCC lets this number change every day.

# Dynamic hedge ratios

The section derives the minimum-variance hedge ratio and its hedging effectiveness, makes the ratio dynamic with DCC, and compares static, rolling and DCC hedges of the BET with the DAX out of sample.

## The minimum-variance hedge ratio {h=803ec2cc}

An investor holds one unit of an asset $s$, for example Romanian shares represented by the BET, and sells $h$ units of a hedging asset $f$, for example a futures contract on the DAX. The hedged return is $r_{s,t} - h\,r_{f,t}$, and its variance is $\sigma_s^2 - 2h\,\sigma_{sf} + h^2\sigma_f^2$, where $\sigma_s^2$ and $\sigma_f^2$ are the variances of the two returns and $\sigma_{sf}$ their covariance. Setting the derivative with respect to $h$ to zero gives the minimum-variance hedge ratio (*hedge ratio*) of @Ederington1979:

$$ h^* = \frac{\sigma_{sf}}{\sigma_f^2} = \rho\,\frac{\sigma_s}{\sigma_f}. $$ {#eq:hedge}

The static version of $h^*$ is the slope of the ordinary least squares (OLS) regression of $r_s$ on $r_f$, because the OLS slope is exactly a covariance divided by a variance. The dynamic version, proposed by @KronerSultan1993, uses the elements of $\mathbf{H}_t$ from an MGARCH model, $h_t^* = h_{sf,t}/h_{ff,t}$, and therefore changes with the forecast covariance and variance every day.

The hedging effectiveness (*hedging effectiveness*) measures the share of variance removed: $HE = 1 - \Var(r_s - h\,r_f)/\Var(r_s)$. Substituting the constant optimal ratio into the hedged variance gives $\sigma_s^2(1 - \rho^2)$, so $HE = \rho^2$. A hedge is only as good as the correlation between the asset and the hedging instrument.

## Worked example: a hedge ratio {h=e15ab5ea}

Take daily volatilities $\sigma_s = 1.2\%$ and $\sigma_f = 1.0\%$ and a correlation $\rho = 0.8$. The covariance is $\sigma_{sf} = \rho\,\sigma_s\sigma_f = 0.8 \cdot 1.2 \cdot 1.0 = 0.96$, and by @eq:hedge the hedge ratio is $h^* = 0.96/1 = 0.96$: the investor sells 0.96 units of $f$ for each unit of $s$.

The hedging effectiveness is $\rho^2 = 0.64$, so the hedge removes 64% of the variance. The volatility of the hedged position is $\sigma_s\sqrt{1 - \rho^2} = 0.72\%$ instead of 1.2%. Volatility falls less than variance, by 40% instead of 64%, because it is the square root.

If tomorrow the DCC model forecasts $\rho_t = 0.5$ and $\sigma_{f,t} = 2\%$, with $\sigma_s$ unchanged, the ratio becomes $0.5 \cdot 1.2/2 = 0.3$. Both the lower correlation and the higher volatility of the hedging asset reduce the number of units to sell, and the position must be adjusted.

## Hedging the BET with the DAX, 2015–2026 {h=71e21079}

The chart plots three hedge ratios, in units of DAX sold per unit of BET, against time from 2015 to 2026: the DCC ratio $h_t = \hat\rho_t\hat\sigma_{BET,t}/\hat\sigma_{DAX,t}$, the rolling OLS slope over the previous 250 days and the static OLS ratio estimated on 2005–2014, a dashed horizontal line. The data are daily returns on common trading days of the two markets since 2005, and all parameters of the DCC model are estimated on 2005–2014.

The DCC ratio is the most volatile line, with short spikes, the highest at the end of 2018. The rolling OLS ratio moves in steps, and its large jump in early 2020 lasts about a year, the length of its window. The static line lies above both dynamic ratios for most of the period.

## Interpreting the hedge ratios {h=82f26349}

The static ratio of 0.51 reflects the period 2005–2014, which includes the 2008 crisis, when the BET and the DAX were closely linked. After 2015 the DCC ratio averages 0.29 and ranges from 0.07 to 1.00, the maximum reached on 20 December 2018. The static ratio thus over-hedges in most of the later period.

The hedging effectiveness out of sample, over 2 893 days, is 13.6% for the static ratio, 18.0% for the rolling OLS ratio and 19.1% for the DCC ratio. The dynamic ratios remove more risk, and DCC does so without the choice of a window.

All three values are modest. The DCC correlation between the BET and the DAX averages 0.32 since 2015, so $\rho^2$ is about a tenth, and no ratio can remove much of the variance of the BET with the DAX. A daily DCC ratio also implies daily rebalancing and trading costs, which are not counted in these figures; after costs, the advantage of DCC over the static ratio would be smaller.

## Recap: Hedging {h=dd9b0339}

The dynamic minimum-variance hedge ratio is $h_t^* = h_{sf,t}/h_{ff,t} = \rho_t\sigma_{s,t}/\sigma_{f,t}$, and the hedging effectiveness is the share of variance removed. For the BET hedged with the DAX, DCC removes 19.1% of the variance out of sample, against 13.6% for the static ratio. A hedge must be judged out of sample and after trading costs.

## @end

The applications have shown what the conditional covariance matrix adds in practice. The next section places the chapter among the earlier ones and lists what the DCC model cannot do.

# Links with other chapters and limits

The section links multivariate GARCH models with univariate GARCH, VAR models and cointegration, and lists the main limits of DCC and the extensions that address them.

## Links with Chapters 5, 6 and 7 {h=f29bfee7}

Every MGARCH model contains univariate GARCH models: the diagonal of $\mathbf{H}_t$ consists of conditional variances, as in [Chapter 5](../EN/Courses/chapter5_conditional_volatility_garch.pdf). In DCC the link is literal, since the univariate GARCH models are step 1 of the estimation.

A VAR model of [Chapter 6](../EN/Courses/chapter6_var_models_granger_causality.pdf) describes spillovers in the mean, for example whether yesterday's DAX return helps to forecast today's BET return. A BEKK model with a full matrix $\mathbf{A}$ describes spillovers in the variance, whether yesterday's shock to the DAX raises today's variance of the BET. The two combine naturally: a VAR for the mean $\boldsymbol{\mu}_t$ and an MGARCH model for $\mathbf{H}_t$. @DieboldYilmaz2009 built spillover indices from the variance decompositions of a VAR.

Cointegration, from [Chapter 7](../EN/Courses/chapter7_cointegration_vecm.pdf), is a long-run link between price levels; correlation is a short-run link between returns. The two are distinct. Two prices can be highly correlated from day to day and still drift apart for ever, or be cointegrated with a low daily correlation. A vector error correction model (VECM) can carry DCC errors, with equilibrium in the mean and dynamic correlation in the shocks.

## Limits and extensions {h=a6835ce1}

DCC buys its parsimony with restrictions. A single pair $(a, b)$ drives all correlations, so pairs that adjust at different speeds are forced to the same rhythm, as the five-market model showed. The volatilities come from separate univariate GARCH models, so DCC has no spillovers between volatilities. And the Normal quasi-likelihood says nothing about the tails: for risk measures, Student $t$ innovations or filtered historical simulation are needed, as the backtest showed.

Correlations also react asymmetrically: they rise more after joint falls than after joint rises, which the asymmetric DCC of @CappielloEngleSheppard2006 captures. For hundreds of assets even DCC becomes hard to handle, because the sample matrix $\bar{\mathbf{Q}}$ is noisy and the step-2 likelihood requires inverting a large matrix every day; factor models, composite likelihood and shrinkage of $\bar{\mathbf{Q}}$ are the usual answers.

Finally, correlation measures linear dependence only. Two assets can have a moderate correlation and still crash together much more often than a Normal model implies. Joint extremes are described by copulas and by tail dependence; @LonginSolnik2001 studied the correlation of international equity markets in the tails directly.

## @end

DCC is the workhorse of multivariate volatility because it splits a hard problem into univariate GARCH models and a correlation recursion with two parameters. Its limits, a common speed for all pairs, no volatility spillovers and Normal tails, are known and have standard extensions.

# Possible contribution of AI

The section describes what an artificial intelligence (AI) assistant can do in multivariate volatility modelling and what must be checked every time.

## Possible contribution of AI {h=71f39877}

An AI assistant can quickly write a first version of the code the chapter needs: the DCC correlation likelihood, a BEKK filter or the Kupiec test. It can also give a second explanation, phrased differently, of the ideas that are harder on first reading, such as the decomposition $\mathbf{D}_t\mathbf{R}_t\mathbf{D}_t$, correlation targeting or the formula of the hedge ratio.

The greatest benefit comes in exploration: DCC models for many pairs at once, such as the markets of Central and Eastern Europe (CEE), sectors or currencies, and a first look at how their correlations behaved in crises. The prompt on the slide is an example of a precise request: it names the data, the alignment on common trading days, the two steps of estimation, correlation targeting and the out-of-sample comparison with static OLS. The more of these choices the request fixes, the easier the result is to check.

## Necessary checks {h=26be2e86}

The first check is the library. The arch package has no DCC model, so an answer that calls arch_model(..., vol='DCC') has invented a function. The second is timing: $\mathbf{Q}_t$ must use $\mathbf{z}_{t-1}$, not $\mathbf{z}_t$; otherwise the VaR and the hedge ratio use tomorrow's information, and the backtest looks better than any real forecast could. The third is the rescaling @eq:dcc: every $\mathbf{R}_t$ must have ones on the diagonal and correlations below one in absolute value.

The other checks concern data and evaluation. Prices must be aligned on common trading days before returns are computed, and asynchronous closes must be kept in mind. Models must be estimated on one period and tested on another. A simple test of any DCC code is a simulation: generate data from a DCC with known $(a, b)$ and check that the code recovers them. Every reference suggested by an assistant must exist, and its digital object identifier (DOI) must be checked.

## @end

An assistant shortens the writing of code but does not replace the checks specific to multivariate models: the timing of the recursion, the validity of each correlation matrix, the alignment of calendars and an honest out-of-sample evaluation.

# Summary

The summary gathers the chapter's main ideas, the formulas to remember and the self-assessment questions.

## Key takeaways {h=d8b0f79f}

Portfolio risk, hedging and contagion depend on the conditional covariance matrix $\mathbf{H}_t$, which changes over time. VEC is the most general linear model but has far too many parameters; BEKK is positive definite by construction and measures volatility spillovers. CCC and DCC split $\mathbf{H}_t = \mathbf{D}_t\mathbf{R}_t\mathbf{D}_t$ and are estimated in two steps, and step 2 of DCC needs only $(a, b)$ whatever the number of assets.

The data show that real correlations move: through the integration of the Romanian market with Europe, through crises and through the arrival of Bitcoin as a risk asset. Constant correlation is rejected for the S&P 500 and the DAX and for the five-market model. The applications turn $\mathbf{H}_t$ into a portfolio VaR 1%, judged by backtesting, and into dynamic hedge ratios, judged out of sample.

## Key formulas {h=f30d2ca0}

The table gathers the chapter's formulas: the conditional correlation @eq:rho; the BEKK model @eq:bekk; the decomposition @eq:drd used by CCC and DCC; the DCC recursion with its rescaling @eq:dcc; the half-life $\ln 0.5/\ln(a + b)$; the portfolio VaR 1% @eq:var; and the hedge ratio @eq:hedge with the hedging effectiveness.

The formulas form a chain. The decomposition separates what univariate GARCH models already handle from the correlations; the DCC recursion makes the correlations dynamic; the half-life summarises their persistence; and the VaR and the hedge ratio turn the resulting $\mathbf{H}_t$ into decisions.

## Self-assessment (1/2) {h=3c7dc14b}

The first question applies the parameter count of the diagonal BEKK: for $N = 3$ the constant matrix has $N(N+1)/2 = 6$ distinct elements and the diagonals of $\mathbf{A}$ and $\mathbf{B}$ have $2N = 6$, so 12 in total. The second applies @eq:rho and the positive-definiteness condition: with $h_{11} = 1$, $h_{22} = 4$ and $h_{12} = 1.2$ the correlation is $1.2/\sqrt{4} = 0.6$, and the determinant $1 \cdot 4 - 1.44$ is positive, so the matrix is positive definite.

The third question is one step of @eq:dcc with $a = 0.02$, $b = 0.97$, $\bar q_{12} = 0.4$ and $z_{t-1} = (1, -1)$. The weight of the long-run term is 0.01 and the cross-product of the residuals is $-1$, so

$$ \begin{aligned} q_{12,t} &= 0.01 \cdot 0.4 + 0.02 \cdot (-1) + 0.97 \cdot 0.4 \\ &= 0.372: \end{aligned} $$

moves in opposite directions lower the correlation. The fourth computes the half-life for $a + b = 0.95$: $\ln 0.5/\ln 0.95 \approx 13.5$ days.

## Self-assessment (2/2) {h=f9646e22}

The first question applies @eq:hedge: with $\sigma_s = 2\%$, $\sigma_f = 1\%$ and $\rho = 0.5$ the hedge ratio is $0.5 \cdot 2/1 = 1$ and the hedging effectiveness $\rho^2 = 0.25$. The second concerns backtesting: 26 violations in 2 500 days are close to the expected 25, so the frequency test is passed, but 10 violations in a single month are a cluster that the independence test rejects, and they show a VaR that does not adapt to crises, like the static VaR of the S&P 500 and DAX portfolio.

The third question returns to asynchronous trading: news from New York reaches Bucharest only on the next trading day, so the daily returns of the same calendar day do not match and the daily correlation is lower than the weekly one. The fourth concerns contagion: a higher correlation of returns in a crisis does not prove contagion by itself, because higher volatility raises the sample correlation mechanically [@ForbesRigobon2002]; the comparison must use correlations of standardised residuals. [Chapter 15](../EN/Courses/chapter15_review.pdf) reviews the whole course.

## @end

The chapter extended volatility modelling from one series to several. The central object is the conditional covariance matrix, and the central idea of DCC is to model volatilities and correlations separately, with univariate GARCH models for the first and a two-parameter recursion for the second. On real data the correlations move, rise in crises most for the weakest links, and change the risk of a portfolio and the size of a hedge.
