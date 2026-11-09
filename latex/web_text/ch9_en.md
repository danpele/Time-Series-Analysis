<!--
Course text of the web version, Chapter 9 (EN). Inserted into the page by latex/build_web_lecture.py.
Format: # section title (introduction), ## slide title {h=hash} (text), @after (text after the live block), ## @end (summary).
Numbered equations: $$ ... $$ {#eq:label}, reference @eq:label. Citations: @Key -> Author (year), [@Key] -> (Author, year).
Current hashes: python3 build_web_lecture.py 9 --hashes
-->

## @intro

@epigraph MakridakisSpiliotisAssimakopoulos2018a | Statistical and Machine Learning forecasting methods: Concerns and ways forward
> After comparing the post-sample accuracy of popular ML methods with that of eight traditional statistical ones, we found that the former are dominated across both accuracy measures used and for all forecasting horizons examined.

@text
Machine learning (ML) methods such as random forests, gradient boosting and neural networks learn a prediction rule from examples, without a model written down in advance. They dominate image recognition, translation and many prediction tasks on large tabular data sets. The question of this chapter is whether they also forecast time series better than the statistical models of the previous chapters: ARIMA, exponential smoothing (ETS) and dynamic harmonic regression, and even the naive forecast.

The answer depends on the data and on the way the comparison is made. A time series has to be turned into a table of examples before any ML method can use it, and the examples are ordered and dependent. A careless table or a careless validation scheme lets information from the future into the training data, and the resulting scores can look excellent while the model has no forecasting ability. A large part of the chapter is therefore about building the table and validating the model honestly.

The chapter then applies ridge and lasso regressions, regression trees and their ensembles, small neural networks and the LSTM to four problems: the daily electricity load of Romania, Romanian inflation, the volatility of stock indices and the sign of daily returns. The ML forecasts enter the same comparison as the statistical models of [Chapter 4](../EN/Courses/chapter4_seasonality_forecasting.pdf), with the same forecast origins, the same accuracy measures and the Diebold–Mariano test. On the electricity load, a ridge regression with calendar features is as accurate as the best statistical model, while the LSTM is clearly less accurate. The landmark M4 and M5 competitions and the study of @GuKellyXiu2020 show where ML methods do win: with many related series and rich features.

@chart tsa_ch9_load_results
Daily electricity load of Romania, 26 forecast origins from 30 June 2025 to 15 June 2026, horizons of 1 to 14 days. Left: the mean absolute scaled error (MASE) of five statistical models (blue) and six ML models (red); below 1 means more accurate than the in-sample seasonal naive forecast. Right: the mean absolute error (MAE) in GW by horizon for six models.

@goals
- **Build** a supervised-learning table of lags, rolling statistics and calendar features from a time series, without future information.
- **Produce** multi-step forecasts with the recursive, direct and multi-output strategies.
- **Validate** a model walk-forward and recognise leakage and look-ahead bias.
- **Explain** ridge, lasso, regression trees, random forests, gradient boosting, the multilayer perceptron and the LSTM, with their tuning parameters.
- **Construct** prediction intervals by quantile regression and split conformal prediction and check their coverage.
- **Compare** ML forecasts with statistical benchmarks on the same test period, with MASE and the Diebold–Mariano test.

@plan
- Forecasting as supervised learning | Lag, rolling-window and calendar features; recursive, direct and multi-output forecasts.
- Validation without leakage | Training, validation and test samples, walk-forward validation, leakage and the bias–variance trade-off.
- Regularised linear models: ridge and lasso | Penalised least squares on many correlated lags.
- Trees, random forest and gradient boosting | Regression trees, their ensembles, extrapolation and permutation importance.
- Neural networks | The multilayer perceptron, its training, the recurrent network and the LSTM.
- Application: electricity load against Chapter 4 | Eleven models on 26 origins and 14 horizons, and the Easter window.
- Prediction intervals | Quantile regression, the pinball loss and split conformal prediction.
- Local and global models: Romanian inflation | One model per series against one model for 27 countries.
- Applications in finance: volatility and the sign of returns | HAR against ML for realised volatility; the sign of tomorrow's return.
- Landmark case studies: M4, M5 and Gu, Kelly and Xiu | Forecasting competitions and machine learning for stock returns.
- Possible contribution of AI | Foundation models, what an assistant can draft and what must be checked.
- Summary | Key ideas, formulas and self-assessment questions.

## Today's question and route {h=c6c2ed06}

The chapter asks whether flexible learning algorithms, such as random forests, gradient boosting and neural networks, forecast a time series more accurately than ARIMA, ETS and the naive forecast, and how such a claim is checked. Machine learning here means a family of algorithms that learn a prediction rule from examples. For a time series the examples are not given: they have to be built from the past of the series itself, and this construction is the first topic of the chapter.

The route has three stages. The first turns forecasting into supervised learning: lag features, rolling windows, calendar features and the three strategies for multi-step forecasts. The second presents the methods and the way to validate them: walk-forward validation and leakage, ridge and lasso, trees and their ensembles, small neural networks and the LSTM. The third covers prediction intervals, the difference between local and global models, four applications and the M4 and M5 competitions. The chapter builds on the accuracy measures and benchmarks of [Chapter 0](../EN/Courses/chapter0_introduction.pdf) and on the time-series cross-validation and the Diebold–Mariano test of [Chapter 4](../EN/Courses/chapter4_seasonality_forecasting.pdf).

## Learning outcomes {h=98b57111}

The first outcomes concern the data: the reader should be able to turn a series into a supervised-learning table that uses only information available at the forecast origin, and to produce forecasts several steps ahead with the recursive, direct and multi-output strategies. The third outcome is methodological: to validate a model walk-forward and to recognise leakage and look-ahead bias, the two errors that make ML forecasts look better than they are.

The remaining outcomes concern the methods and their evaluation. The reader should be able to explain ridge, lasso, regression trees, random forests, gradient boosting, the multilayer perceptron and the LSTM, together with the parameter that controls the complexity of each; to build prediction intervals by quantile regression and conformal prediction; and to compare ML forecasts with statistical benchmarks on the same test period, using the MASE and the Diebold–Mariano test.

## Reading and tools {h=c6b25a2c}

The course textbook is @HuangPetukhina2022, whose Chapter 10 covers machine learning for time series. The free online textbook of @HyndmanAthanasopoulos2021, Section 12.4, presents neural network autoregressions, and @HastieTibshiraniFriedman2009 give the theory of the methods: linear models and their penalties (Chapter 3), trees (Chapter 9), boosting (Chapter 10), neural networks (Chapter 11) and random forests (Chapter 15). Three reviews complete the reading: @HewamalageBergmeirBandara2021 on recurrent networks for forecasting, @LimZohren2021 on deep learning for time series and @MonteroMansoHyndman2021 on global models.

The computations use Python. The *scikit-learn* library provides every model of the chapter except the LSTM: *RidgeCV* and *LassoCV* for penalised regressions, *RandomForestRegressor*, *HistGradientBoostingRegressor*, *MLPRegressor* and *TimeSeriesSplit* for walk-forward validation. The LSTM is a small network written in *PyTorch*. The models are kept small, with fixed random seeds, so that every notebook runs on an ordinary processor in Google Colab in a few minutes.

# Forecasting as supervised learning

Supervised learning is the part of machine learning that learns to predict a target from inputs, using examples where both are known. A forecast fits this frame once the series is rewritten as a table: each row holds what was known at a forecast origin and the value that was observed later. The section builds this table, shows it on the daily electricity load of Romania and presents the three ways of producing forecasts more than one step ahead.

## Supervised learning: data, loss function and training {h=8402bd0d}

Supervised learning starts from $n$ pairs $(\mathbf{x}_i, y_i)$. The vector $\mathbf{x}_i$ holds the features, the inputs of the prediction, and $y_i$ is the target, the quantity to be predicted. The goal is a function $\hat f$ that predicts well on new data, that is, whose expected loss $E[L(y, \hat f(\mathbf{x}))]$ is small for pairs not used in training. The loss function decides what "predicts well" means. The squared error $(y - \hat y)^2$ is minimised by the conditional mean of $y$, so it leads to forecasts of the mean; the absolute error $|y - \hat y|$ is minimised by the conditional median.

Training chooses $\hat f$ within a class of functions (linear functions, trees, neural networks) by minimising the average loss on the training data. Each class also has hyperparameters, settings fixed before training: the depth of a tree, the number of trees, the size of a penalty. They cannot be estimated from the training loss, because a more complex model always fits the training data better; they are chosen by validation on data the model has not seen.

Standard machine learning treats the pairs as independent draws from one distribution. The rows of a time-series table are ordered in time and dependent: neighbouring rows share most of their information. The table must therefore be built so that each row uses only the past of its origin, and the validation must keep the order of time. These two requirements shape the rest of the chapter.

## From a series to a table {h=2f9ec6b9}

The simplest features are lags. At the forecast origin $t$ the values $y_t, y_{t-1}, \dots, y_{t-p+1}$ are known, and the target is $y_{t+h}$, the value $h$ steps later. An AR($p$) model of [Chapter 2](../EN/Courses/chapter2_arma_models.pdf) is exactly a linear regression of $y_{t+1}$ on these lags. Machine learning keeps the table and replaces the linear function by a more flexible one, such as a tree ensemble or a neural network.

Rolling-window features summarise the recent past: means, standard deviations or minima over the last $w$ values. The 7-day mean, for example, is $\bar y_t^{(7)} = \frac{1}{7}\sum_{j=0}^{6} y_{t-j}$, the average of the seven values up to and including the origin. The window must end at $t$. A window that contains $y_{t+1}$ puts part of the target into the features, an error called leakage that the next section examines in detail.

Calendar features describe the target day: the day of the week, the month, the public holidays and Fourier terms such as $\sin(2\pi k\,d/365.25)$, where $d$ is the day of the year and $k = 1, 2, \dots$ is the harmonic ($k = 1$ gives one cycle per year). They are known in advance, so they are legitimate inputs, as the Fourier terms and the Orthodox Easter dummy of [Chapter 4](../EN/Courses/chapter4_seasonality_forecasting.pdf) were. Exogenous variables such as temperature are allowed only in the form available at the origin: the weather forecast, not the temperature measured later.

## Worked example: a lag table for the electricity load {h=c9324bb2}

The table shows six rows built from the daily mean load of Romania, in gigawatts (GW), for the week that ends just before 30 June 2025. The first row has its origin on Monday 23 June: the load that day was 5.80 GW, on Sunday 4.73 GW and on Saturday 5.21 GW, and the target is Tuesday's load, 6.03 GW. Each row is a snapshot of what was known at the origin, paired with the value that the forecast had to predict.

The construction loses a few observations at both ends. With $p$ lags, the first $p - 1$ days have no complete row, because some of their lags fall before the start of the series. For horizon $h$, the last $h$ days have no observed target. A series of length $n$ therefore gives $n - p - h + 1$ complete rows.

The weekday column shows why calendar features matter. The Sunday load is about 1 GW below the working-day load, and the target of the last row, Sunday 29 June, is 5.09 GW after 5.78 GW on Saturday. A model that sees only the lags cannot know that the next day is a Sunday; it needs the weekday of the target day as a feature.

## Electricity load: the series and three features {h=8219fd34}

The top panel shows the daily mean load of Romania from the European Network of Transmission System Operators for Electricity (ENTSO-E), from 1 January 2022 to 30 June 2026, 1,642 days. The red dots mark public holidays. The series combines a weekly cycle, visible as the dense oscillation, an annual cycle with peaks in winter, and sharp drops on holidays, especially around Christmas and Easter.

The three bottom panels plot tomorrow's load $y_{t+1}$ against three candidate features: today's load $y_t$, the load on the same weekday of the previous week $y_{t-6}$ and the 7-day mean at $t$. Each dot is one row of the table; the orange dots are rows whose target falls on a Sunday and the blue ones targets from Monday to Saturday. A good feature makes the cloud narrow and close to a line.

## Interpreting the load features {h=0c9a2cfa}

The mean load over the sample is 6.18 GW. The minimum, 3.79 GW, occurred on 20 April 2025, the Orthodox Easter Sunday, and the maximum, 8.15 GW, on 13 January 2022, during a winter week. The weekly cycle, the annual cycle of heating and cooling and the holiday drops together form the multiple seasonality studied in [Chapter 4](../EN/Courses/chapter4_seasonality_forecasting.pdf).

The correlations measure how much each feature says about tomorrow's load: 0.77 for today's load, 0.82 for the same weekday of the previous week and 0.74 for the 7-day mean. The same-weekday lag is the best single feature because it carries the weekly cycle with it. With today's load alone, the Sunday targets form a separate cloud below the others, so a single linear function of $y_t$ cannot represent the weekly pattern; a model needs either the weekday or the right lag.

The table used in the rest of the chapter has 1,614 rows and 30 features: 14 lags, two rolling means, the same-weekday value and calendar features (weekday dummies, holiday flags and Fourier terms). The first rows are lost to the 14 lags, which explains why there are fewer rows than days.

## Multi-step forecasts: three strategies {h=b8c46af3}

A forecast for several days ahead, $h = 1, \dots, H$, can be produced in three ways. The recursive, or iterated, strategy trains one model for $h = 1$. Its forecast $\hat y_{T+1}$ replaces the unknown $y_{T+1}$ in the lags, and the same model is applied again to obtain $\hat y_{T+2}$, and so on. ARIMA models forecast in this way ([Chapter 3](../EN/Courses/chapter3_unit_roots_arima_models.pdf)). The strategy needs a single model, but if the one-step model is wrong, its errors are fed back as inputs and accumulate with the horizon.

The direct strategy trains a separate model $\hat f_h$ for each horizon, with $y_{t+h}$ as the target. No forecast is fed back, so errors do not accumulate. The cost is $H$ models instead of one, fewer complete rows for long horizons, and forecasts for neighbouring horizons that come from different models and need not form a coherent path.

The multi-output strategy (MIMO, multiple-input multiple-output) trains one model that returns the whole vector $(\hat y_{T+1}, \dots, \hat y_{T+H})$ at once. It suits neural networks, whose output layer can have $H$ units. @BenTaiebEtAl2012 review the three strategies and compare them on the NN5 forecasting competition.

## Recursive, direct and multi-output forecasts {h=d839a630}

The diagram shows the three strategies for three steps ahead. Blue boxes are inputs known at the origin $T$, green boxes are fitted models and red boxes are forecasts. In the recursive strategy on the left, the same model $f$ is applied three times, and the orange arrows show each forecast fed back as an input for the next step.

In the direct strategy in the middle, the same inputs go to three different models, $f_1$, $f_2$ and $f_3$, each trained for its own horizon. In the multi-output strategy on the right, one model $f$ produces the three forecasts together. The diagram makes the trade-off visible: the recursive strategy reuses one model and propagates its errors, the direct strategy multiplies the models, and MIMO keeps one model with several outputs.

## Worked example: recursive and direct two-step forecasts {h=15e71da2}

Suppose that the one-step model estimated on the table is $\hat y_{t+1} = 1.2 + 0.8\,y_t$ and that the last observation is $y_T = 5$. The recursive forecast for tomorrow is $\hat y_{T+1} = 1.2 + 0.8 \cdot 5 = 5.20$. For the day after, the unknown $y_{T+1}$ is replaced by 5.20, which gives $\hat y_{T+2} = 1.2 + 0.8 \cdot 5.20 = 5.36$. Substituting the first equation into the second shows the implied two-step rule: $\hat y_{T+2} = 1.2 + 0.8(1.2 + 0.8\,y_T) = 2.16 + 0.64\,y_T$.

The direct strategy estimates a separate regression with $y_{t+2}$ as the target. Suppose it gives $\hat y_{t+2} = 2.3 + 0.6\,y_t$, so the direct forecast is $\hat y_{T+2} = 2.3 + 0.6 \cdot 5 = 5.30$. If the series really were an AR(1), the direct regression would estimate the same coefficients as the implied rule, 2.16 and 0.64, up to sampling error.

Different coefficients signal that the one-step model is misspecified at two steps, for example because the series has a weekly pattern that one lag cannot carry. The direct model adapts to the horizon it is trained for, while the recursive model extrapolates a one-step rule. Neither strategy is always more accurate, so the choice is made on validation data.

## Recap: Forecasting as supervised learning {h=6240892b}

A forecast is a prediction of $y_{t+h}$ from features known at the origin $t$: lags, rolling statistics and calendar variables of the target day. An AR($p$) is a linear regression on lags, so machine learning changes the function, not the table. Forecasts several steps ahead are recursive (one model, iterated), direct (one model per horizon) or multi-output (one model with $H$ outputs).

## @end

Turning a series into a table is what makes machine learning applicable to forecasting, and it is also where the first errors can enter. Every feature must be computable at the origin, the calendar of the target day must be included explicitly, and the strategy for several horizons is a modelling choice to be validated like any other.

# Validation without leakage

A forecasting method is only as credible as the way it is evaluated. The section defines the training, validation and test samples for time series, compares four validation schemes, describes leakage, the most common source of false results, and shows it on a random walk and on the S&P 500. It ends with the bias–variance decomposition, which explains why the complexity of a model must be chosen on data it has not seen.

## Training, validation and test samples {h=c09feeea}

The data have three roles. The training sample estimates the parameters of a model, the validation sample chooses its hyperparameters, and the test sample measures the final accuracy. The test sample is used once, at the end: if it is used to choose among models, it becomes a second validation sample and its error is no longer an honest estimate. For a time series the three samples are consecutive blocks in this order, so that every model is evaluated on a period that comes after the data it learned from.

Walk-forward validation, the time-series cross-validation of [Chapter 4](../EN/Courses/chapter4_seasonality_forecasting.pdf), repeats this split. The model is trained on the data up to an origin, forecasts the next block, and the origin moves forward. With an expanding window the training sample contains all past data; with a rolling window it contains only the last $W$ observations, which adapts faster to changes in the series but uses less data. The errors of all origins are pooled into the MAE, the root mean squared error (RMSE) and the MASE of [Chapter 0](../EN/Courses/chapter0_introduction.pdf), and two methods are compared with the Diebold–Mariano test [@DieboldMariano1995].

Random $K$-fold cross-validation splits the rows at random into $K$ groups and uses each group once as test data. @BergmeirHyndmanKoo2018 show that it remains valid for purely autoregressive models whose errors are white noise, because the rows are then conditionally uncorrelated. When the errors are correlated, or the targets overlap, random folds give optimistic errors.

## Four validation schemes {h=2e4a19f5}

Each panel shows four splits of the same series along the time axis: blue cells are training data, red cells test data, golden cells a gap and empty cells data that are not used. In random $K$-fold the test cells are scattered through the sample, so the model is often trained on observations that come after the test day: it learns from the future.

The three walk-forward schemes keep the order of time. With an expanding window the training block grows from one split to the next; with a rolling window it has a fixed length and slides forward. In walk-forward validation with a gap, the training rows just before each test block are dropped. These rows have targets that extend into the test period, for example a 21-day return that starts a few days before the test block, and the gap prevents them from sharing information with the test rows.

## Look-ahead bias and leakage {h=c16f5614}

Leakage is the use, in training or in the features, of information that would not have been available at the forecast origin [@KaufmanEtAl2012]. Its effect is always the same: validation scores that look excellent and disappear when the model is used for real forecasts. In finance the same error is called look-ahead bias, because the backtest looks ahead at data that the investor could not have known.

The typical sources are easy to miss. A rolling mean or a difference that includes $y_{t+1}$ contains part of the target; centred moving averages and two-sided filters, such as those used to estimate a trend in [Chapter 0](../EN/Courses/chapter0_introduction.pdf), use future values by construction. Scaling, imputation of missing values or feature selection estimated on the whole sample before the split pass information from the test period into training. Overlapping targets, such as sums over the next $h$ days, leak through random folds or through walk-forward splits without a gap. Revised macroeconomic data are a subtler case: the GDP figure published today for a past quarter was not known at the time, and the actual weather is not the weather forecast that was available.

The rule that prevents leakage is to write every feature as a function of the data up to $t$ only, and to refit every preprocessing step, such as the mean and standard deviation used for scaling, inside each training window.

## Leakage in practice: random folds on overlapping targets {h=e16dded8}

The experiment shows how strong leakage can be. A random forest predicts the sum of the next 21 daily returns, roughly one month ahead, from past sums over 5, 21 and 63 days and from past volatilities. The left pair of bars uses a simulated random walk, for which the future sum is unpredictable by construction; the right pair uses 6,631 daily returns of the S&P 500 from 2000 to 2026.

The vertical axis is the out-of-sample $R^2$, $1 - \sum(y - \hat y)^2/\sum(y - \bar y)^2$, in percent. It is positive when the model forecasts better than the historical mean of the target and negative when it forecasts worse. Red bars come from random 5-fold cross-validation, blue bars from walk-forward validation and green bars from walk-forward validation with a gap of 21 days.

## Interpreting the leakage experiment {h=79a1ec90}

Random 5-fold cross-validation reports an $R^2$ of 34.5% for the random walk, where nothing is predictable, and 36.8% for the S&P 500. Both numbers are artefacts. The target of day $t$ and the target of day $t+1$ share 20 of their 21 daily returns, so almost every test row has near-copies among the training rows of the other folds, and the forest simply recognises them.

Walk-forward validation gives $-26.5$% for the random walk and $-38.6$% for the S&P 500; with a gap of 21 days the values are $-26.8$% and $-38.9$%. A negative out-of-sample $R^2$ means that the forest forecasts worse than the historical mean, which is the expected result for future returns ([Chapter 1](../EN/Courses/chapter1_stochastic_processes_stationarity.pdf)). The gap changes little here because the walk-forward blocks are long, but it removes the remaining overlap at their borders.

The lesson is general: a model is only as good as its validation, and a spectacular score on a problem known to be hard is a reason to look for a leak before believing it.

## Overfitting and the bias–variance trade-off {h=443723c6}

Suppose the data follow $y = f(x) + \varepsilon$, where $f$ is the true function and $\varepsilon$ is noise with variance $\sigma^2$, and that $\hat f$ is estimated on a random training sample. For a new observation $y_0$ at a point $x_0$, the expected squared error, taken over training samples and noise, splits into three parts:

$$
\begin{aligned}
E[(y_0 - \hat f(x_0))^2] &= \bigl(E[\hat f(x_0)] - f(x_0)\bigr)^2 \\
&\quad + \Var(\hat f(x_0)) + \sigma^2 .
\end{aligned}
$$ {#eq:bv}

The first term is the squared bias, the systematic error of the method averaged over training samples. The second is the variance, which measures how much the fitted function changes from one training sample to another. The third, $\sigma^2$, is the noise that no model can remove.

A rigid model, such as a very shallow tree, has a large bias: it misses part of the signal, which is called underfitting. A very flexible model has a small bias but a large variance: it follows the noise of its particular training sample, which is called overfitting, and shows a small training error with a large test error. Every ML method has a parameter that moves it along this trade-off, such as the tree depth, the number of trees, the penalty or the number of hidden units, and validation sets this parameter.

## Bias and variance of regression trees {h=4f4890f7}

The simulation makes @eq:bv visible. The true function is $f(x) = \sin(2\pi x)$, the noise has standard deviation $\sigma = 0.3$, so $\sigma^2 = 0.09$, and each training sample has 80 points. Regression trees of depth 1 to 10 are fitted to 300 independent training samples, which allows the bias and the variance to be computed exactly as averages over samples.

The left panel shows one training sample (dots), the true function (black) and three fitted trees. The tree of depth 1 is a single step and misses the shape of the sine. The tree of depth 3 follows the shape with a few steps. The tree of depth 10 chases individual points and jumps around the true curve. The right panel plots, against the depth, the squared bias, the variance, their sum plus the noise (the expected test error, red), the training error (dashed) and the noise variance (dotted).

## Interpreting the U-shaped test error {h=77d356eb}

At depth 1 the squared bias is 0.070 and the variance 0.031; at depth 10 the squared bias is practically 0 and the variance 0.090. As the depth grows, the bias falls and the variance rises, so the expected test error first decreases and then increases, a U shape. Its minimum, 0.132, is at depth 3; it lies above the noise variance of 0.09, which is the floor that no model can go below.

The training error behaves differently: it falls steadily and reaches 0.002 at depth 10, far below the noise variance, because the deep tree has memorised its sample. The training error therefore cannot choose the complexity; only data that the model has not seen can. The ensembles of the next sections (random forests and boosting) and the penalties of ridge and lasso are ways of reducing the variance without giving up much of the bias.

## Recap: Validation without leakage {h=71899550}

Walk-forward validation trains on the past, tests on the next block and moves forward; random folds are acceptable only for pure autoregressions with white-noise errors. Leakage enters through future values in features, preprocessing fitted on the full sample, overlapping targets and revised data. The test error is the sum of the squared bias, the variance and the noise, and validation sets the complexity that balances the first two.

## @end

Honest validation keeps the order of time, refits every step of the analysis inside each training window and separates the rows whose targets overlap. On overlapping 21-day returns, random folds turn a negative out-of-sample $R^2$ into an apparent 35%, even on a random walk. The bias–variance trade-off explains why complexity must be tuned on unseen data, and why the methods that follow all come with a complexity parameter.

# Regularised linear models: ridge and lasso

Before nonlinear methods, the section returns to linear regression on lags, the AR model of [Chapter 2](../EN/Courses/chapter2_arma_models.pdf) with many lags and calendar features. With 30 correlated features, ordinary least squares becomes unstable; ridge and lasso add a penalty on the size of the coefficients, which trades a little bias for a large reduction in variance. The section derives both estimators, works through a one-feature example and shows their coefficient paths on the electricity load.

## Many lags, unstable least squares {h=f71d08a8}

The linear model on $p$ features predicts $\hat y = \beta_0 + \sum_j \beta_j x_j$, and ordinary least squares (OLS) chooses the coefficients that minimise the sum of squared errors $\sum_t (y_t - \hat y_t)^2$. When the features are 14 lags of a daily series, neighbouring lags are strongly correlated: yesterday's load and the load of the day before carry almost the same information. OLS can then trade a large positive coefficient on one lag against a large negative one on its neighbour with little change in the fit, so the coefficients become large, of opposite signs, and change from one training window to the next.

Penalised regressions measure the size of the coefficients, and this size depends on the units of the features. The features are therefore standardised first, $x_j \to (x_j - \bar x_j)/s_j$, where $\bar x_j$ and $s_j$ are the mean and standard deviation in the training window only; computing them on the full sample would be leakage. Without standardisation, the penalty would treat a lag measured in GW and a 0/1 weekday dummy unequally.

The idea of regularisation follows from the bias–variance trade-off @eq:bv: shrinking the coefficients towards zero introduces a little bias, but it reduces the variance of the estimates much more, and the test error falls.

## Ridge and lasso {h=c6fa28aa}

Ridge regression [@HoerlKennard1970] adds the sum of squared coefficients to the least-squares criterion:

$$ \min_{\beta}\ \sum_t (y_t - \beta_0 - \mathbf{x}_t'\beta)^2 + \lambda\sum_j \beta_j^2 , $$ {#eq:ridge}

where $\lambda \ge 0$ is the penalty; $\lambda = 0$ gives OLS. With $\mathbf{X}$ the matrix of standardised features and $\mathbf{y}$ the vector of targets (both centred, so that $\beta_0$ drops out), the solution has the closed form $\hat\beta = (\mathbf{X}'\mathbf{X} + \lambda\mathbf{I})^{-1}\mathbf{X}'\mathbf{y}$, with $\mathbf{I}$ the identity matrix. Adding $\lambda$ to the diagonal makes the matrix well conditioned even when the lags are nearly collinear. All coefficients shrink towards 0, but none becomes exactly 0.

The lasso of @Tibshirani1996 (least absolute shrinkage and selection operator) uses the penalty $\lambda\sum_j |\beta_j|$ instead. The absolute value has a corner at zero, and as a result the lasso sets some coefficients exactly to 0: it shrinks and selects features at the same time. With a single standardised feature, $S_{xy} = \sum_t x_ty_t$ and $S_{xx} = \sum_t x_t^2$, the two estimators have explicit forms:

$$
\begin{aligned}
\hat\beta_{\text{ridge}} &= \frac{S_{xy}}{S_{xx} + \lambda}, \\
\hat\beta_{\text{lasso}} &= \frac{\mathrm{sign}(S_{xy})\max(|S_{xy}| - \lambda/2,\ 0)}{S_{xx}} .
\end{aligned}
$$ {#eq:lasso1d}

The lasso formula is called soft thresholding. With $S_{xy} = 90$ and $S_{xx} = 100$, OLS gives $90/100 = 0.90$; ridge with $\lambda = 50$ gives $90/150 = 0.60$; the lasso with $\lambda = 40$ gives $(90 - 20)/100 = 0.70$; and for every $\lambda \ge 180$ the lasso coefficient is exactly 0. The penalty $\lambda$ is a hyperparameter, chosen by walk-forward validation (*TimeSeriesSplit*), never by random folds.

## Ridge and lasso paths on the load features {h=6d8982bc}

A coefficient path shows every coefficient as a function of the penalty. The two panels use the same problem: tomorrow's load as the target, the 30 standardised features of the load table and the data up to 30 June 2025, 1,249 rows. The penalty decreases from left to right on a logarithmic scale, so each panel starts from a heavily penalised model, where all coefficients are 0, and ends close to OLS. The vertical axis is the coefficient in GW per standard deviation of the feature.

Seven features are highlighted in colour: today's load (*lag0*), the same weekday of the previous week (*same_dow*), the 7-day mean (*mean7*), the Saturday dummy and the Easter, Christmas and holiday flags; the thin lines are the other features. In the ridge panel all lines leave zero together and grow smoothly. In the lasso panel the lines leave zero one after another and some stay at zero over a wide range of penalties. The dashed red line marks the lasso penalty chosen by walk-forward cross-validation.

## Interpreting the coefficient paths {h=dcefa219}

With a large penalty all coefficients are close to 0. As $\lambda$ falls, the lasso first admits the lag of the same weekday last week, then today's load and the lag of 13 days; today's load soon dominates and ends with a coefficient of 0.55 GW per standard deviation. At the chosen penalty the next largest coefficients are those of the weekday dummies, the largest of them the Monday dummy with 0.38 GW: after the persistence of the series, the calendar carries most of the remaining signal.

At the penalty chosen by validation, the lasso keeps 22 of the 30 features. Many lags become redundant once today's load, the weekday and the holidays are in the model, which is the multicollinearity of the previous slide resolved by selection. Ridge keeps all 30 features and shrinks them smoothly. On the 26 test origins of [Section 6](#sec-6), ridge turns out to be the most accurate ML model, as accurate as the best statistical model of [Chapter 4](../EN/Courses/chapter4_seasonality_forecasting.pdf).

## Recap: Ridge and lasso {h=9b512bd3}

Ridge and lasso are penalised least squares: the ridge penalty $\lambda\sum\beta_j^2$ shrinks all coefficients, the lasso penalty $\lambda\sum|\beta_j|$ shrinks them and sets some to zero. The features are standardised inside each training window, and $\lambda$ is chosen walk-forward. A regularised autoregression with calendar features is a strong baseline that every nonlinear method has to beat.

## @end

Penalties make a linear model with many correlated lags stable. Ridge shrinks every coefficient, the lasso also selects; on the electricity load the lasso keeps 22 of 30 features, led by today's load and the calendar. Because the linear structure is often close to right for seasonal series, a well-specified ridge regression is the benchmark for the nonlinear methods of the next sections.

# Trees, random forest and gradient boosting

Regression trees capture nonlinearities and interactions without being told where they are, but a single tree is unstable. Random forests average many trees to reduce their variance, and gradient boosting adds small trees one at a time to reduce their bias. The section explains the three methods, compares them on the electricity load, shows the main limitation of all tree methods, their inability to extrapolate, and measures which features a fitted model uses.

## Regression trees {h=d05a89b1}

A regression tree partitions the feature space with a sequence of yes/no questions of the form "$x_j \le c$?". Each question splits a group of training rows into two, and each final group, called a leaf, predicts the mean target of its training rows. The splits are chosen greedily, one at a time: at each node the algorithm tries every feature $j$ and every threshold $c$ and keeps the pair that gives the smallest sum of squared errors in the two child groups.

Trees have two strengths for forecasting. They capture nonlinear effects and interactions, such as a weekday effect that differs between winter and summer, without the analyst specifying them, and they need no scaling of the features, since a split depends only on the order of the values. Their weaknesses are equally clear: a single tree is unstable, because a small change in the data can change the first split and with it the whole tree, and its prediction is a step function.

The most important limitation for time series is that a tree cannot extrapolate. Its forecast is always the mean of some training targets, so it can never exceed the largest target or fall below the smallest one seen in training. For trending series the remedy is to model differences, growth rates or deviations from a trend, never levels.

## A depth-2 tree for tomorrow's load {h=ee9a32dd}

The tree is fitted to the load data up to 30 June 2025, 1,249 days, with four features: today's load (*lag0*), the load on the same weekday of the previous week (*same_dow*), and the weekend and holiday flags. Its depth is limited to 2, so it asks at most two questions, and each leaf must contain at least 20 days.

Each box shows the splitting rule, the number of training days that reach the node (*samples*) and the mean load of those days in GW (*value*). The root contains all 1,249 days, with mean 6.19 GW. Days that satisfy the rule go to the left branch (*True*), the others to the right (*False*). The left branch splits again on the same-weekday load at 5.5 GW, the right branch on today's load at 6.94 GW.

## Interpreting the tree {h=4ee982a7}

The first question is whether the load on the same weekday of the previous week was below 6.08 GW. Weekends and holidays have low load, and the same day a week earlier was usually also a weekend day, so this question separates them from working days without any weekday dummy. The left branch (548 days, mean 5.67 GW) then separates the lowest days from the rest; the right branch (701 days, mean 6.59 GW) separates ordinary working days from the days of highest load.

The four leaves predict 5.29, 5.89, 6.39 and 7.17 GW and contain 195, 353, 523 and 178 days. These four numbers already explain 64% of the variance of tomorrow's load in the training data ($R^2 = 0.64$). Deeper trees add more detail, but also more variance, as the bias–variance experiment showed. Whatever the inputs, the forecast of this tree for any day is one of the four leaf means.

## Random forest {h=d7e739e1}

Bagging, short for bootstrap aggregation, reduces the variance of trees by averaging. It draws $B$ bootstrap samples of the training rows (samples of the same size, drawn with replacement), grows a deep tree on each, and averages their forecasts. If each tree has variance $\sigma^2$ and two trees have correlation $\rho$, the average of $B$ trees has variance

$$ \Var\Bigl(\frac{1}{B}\sum_{b=1}^{B} T_b(\mathbf{x})\Bigr) = \rho\sigma^2 + \frac{1-\rho}{B}\,\sigma^2 , $$

where $T_b$ is the $b$-th tree. The second term vanishes as $B$ grows, but the first remains: averaging cannot remove the variance that the trees share.

The random forest of @Breiman2001 attacks the first term. At each split, only a random subset of the features is tried (the hyperparameter *max_features*), so the trees differ more from one another, $\rho$ falls, and the variance of the average falls with it. The main hyperparameters are the number of trees, where more trees never make the forest worse, only slower, the size of the feature subset and the minimum leaf size.

The bootstrap draws rows without regard to their order in time. This does not violate the rules of the previous section, because the bootstrap only builds the model inside the training window, and the evaluation remains walk-forward on later data.

## Gradient boosting {h=7b784965}

Boosting [@Friedman2001] builds the model in the opposite way. Instead of averaging deep, independent trees, it adds small trees one after another, each fitted to the residuals of the current model. For the squared error, the residuals are the negative gradient of the loss with respect to the current forecast, which gives the method its name. After $m$ trees the model is

$$ F_m(\mathbf{x}) = F_{m-1}(\mathbf{x}) + \nu\,g_m(\mathbf{x}) , $$ {#eq:boost}

where $g_m$ is the $m$-th small tree and $\nu$, the learning rate, usually between 0.01 and 0.3, scales its contribution. A small learning rate makes each step cautious, so the model needs more trees but overfits less.

A worked example shows one step. The targets are 5, 6 and 8. The model starts from their mean, $F_0 = 6.33$, so the residuals are $-1.33$, $-0.33$ and $+1.67$. A tree with one split (a stump) that isolates the third point predicts the residual 1.67 for it. With $\nu = 0.1$, the forecast for that point becomes $F_1 = 6.33 + 0.1 \cdot 1.67 = 6.50$: it moves one tenth of the way towards the target, and the next trees continue from the new residuals.

Fast implementations bin each feature into a histogram, which makes the search for splits much faster: XGBoost [@ChenGuestrin2016], LightGBM [@KeEtAl2017] and *HistGradientBoostingRegressor* in scikit-learn. In the rest of the chapter GB denotes gradient boosting; the winners of the M5 competition ([Section 10](#sec-10)) used LightGBM. The number of trees is chosen by early stopping, which stops adding trees when the error on validation data no longer falls; the other hyperparameters are the tree size, the learning rate and the minimum leaf size.

## One tree, a forest and boosting on the load {h=bf048181}

The chart compares the three methods on tomorrow's load. They are trained on the data up to 31 December 2024 (1,069 days) and evaluated on the validation period from January to June 2025 (180 days). The vertical axis is the validation mean squared error (MSE) in GW². The dotted line is the seasonal naive forecast, the load of the same weekday last week.

The left panel varies the maximum depth of a single tree and of the trees of a random forest of 200 trees. The right panel shows boosting as the number of trees grows, on a logarithmic scale, for two learning rates, 0.3 and 0.05. Each curve shows how validation error responds to the complexity knob of its method.

## Interpreting the ensembles {h=c6f635c0}

The best single tree reaches a validation MSE of 0.183 at depth 9, and its curve is flat and noisy beyond depth 3. The random forest reaches 0.105 at depth 12 and keeps improving with depth, because averaging removes most of the variance that deep trees add. Both are far below the seasonal naive forecast, whose MSE is 0.303.

Boosting with $\nu = 0.3$ reaches its minimum, 0.110, after 72 trees; with $\nu = 0.05$ it needs 256 trees and reaches 0.098, the lowest value in the comparison. This is the usual pattern: a small learning rate needs more trees and ends slightly lower. Early stopping finds the number of trees automatically, at the point where the validation error stops falling.

## Trees cannot extrapolate: the Romanian price level {h=fc23679a}

The chart shows the harmonised index of consumer prices (HICP) of Romania, published by Eurostat with 2015 = 100, together with one-month-ahead forecasts from two random forests trained on the data up to December 2020 (the vertical dotted line). The first forest uses lagged levels of the index as features and forecasts the level. The second uses monthly log changes of the index and forecasts the next change, which is then added to the last observed level.

The horizontal dotted line marks the largest level seen in training. After 2020 the price level rises steeply, and the chart shows what each forest does with values it has never seen.

## Interpreting the extrapolation failure {h=eb0e0399}

The highest index value in training was 111.3; by August 2026 the index had reached 173.4. The forest on levels stays near 110.8 for more than five years, because each of its forecasts is a mean of training levels and none of them is higher than 111.3. Its RMSE over the test period is 36.9 index points.

The same forest applied to monthly changes, with the forecasts cumulated onto the last observed level, follows the index closely, with an RMSE of 0.71 index points. Monthly inflation rates after 2021 were higher than before, but they stayed within a range that the forest could learn, while the level moved far outside it. The lesson repeats [Chapter 3](../EN/Courses/chapter3_unit_roots_arima_models.pdf): a non-stationary series is differenced or transformed before a tree sees it.

## The features a model uses {h=0b62175c}

A random forest or a boosted model has no coefficients to read, so its use of the features has to be measured. Permutation importance shuffles the values of one feature in the test data, which destroys its link with the target while keeping its distribution, and measures how much the forecast error grows. A feature that the model relies on produces a large increase; an unused feature produces none. The measure works for any model and is computed out of sample.

Correlated features are a weakness of the method. If two features carry the same information, shuffling one of them leaves the other in place, so each appears less important than the pair really is. Trees also offer an impurity-based importance (mean decrease impurity, MDI), the total reduction of the squared error obtained by splits on each feature. It is computed on the training data and favours continuous features with many possible split points, so permutation importance is preferred.

Importance describes the model, not the electricity system. A feature can be important because it is correlated with the true cause, and importance says nothing about what would happen if the feature changed.

## Permutation importance of gradient boosting {h=eff88bd6}

The chart shows permutation importances for two direct gradient boosting models, one for the load one day ahead and one for the load 14 days ahead. Both are trained on the data up to 31 December 2024 and evaluated on January to June 2025, and each feature is shuffled 20 times. The bars give the average increase in MAE, in megawatts (MW), for the eight most important features of each model.

The names follow the feature table: *lag0* is the load at the origin, *lagk* the load $k$ days earlier, *mean7* and *mean28* rolling means, *mon* and *sat* weekday dummies, *holiday* the holiday flag, and *cos1*, *cos2* the first two cosine Fourier terms of the annual cycle.

## Interpreting the importances {h=5a0f20f5}

One day ahead, two features dominate. Shuffling *lag0* increases the MAE by 371 MW and shuffling *lag6* by 117 MW. For the target $y_{t+1}$, *lag0* is yesterday's load and *lag6* is $y_{t-6}$, the load on the same weekday one week earlier: the persistence of the series and the weekly cycle.

Fourteen days ahead the order changes: *cos1* increases the MAE by 126 MW and *lag0* by 89 MW. The recent values say less about a day two weeks away, and the annual cycle, carried by the Fourier term, gains weight. *lag0* stays important because for $h = 14$ the origin falls on the same weekday as the target, and *lag7* is again a same-weekday value.

## Recap: Trees and ensembles {h=95084114}

A tree is a piecewise-constant function found by greedy splits, and it cannot extrapolate beyond the range of its training targets. A random forest averages decorrelated deep trees and lowers the variance; boosting adds small trees fitted to residuals and lowers the bias. Gradient boosting (LightGBM, XGBoost, *HistGradientBoosting*) is tuned through the learning rate, the number of trees and early stopping. Permutation importance measures, out of sample, which features a model uses.

## @end

On the electricity load, ensembles of trees reduce the validation MSE of a single tree from 0.183 to about 0.10, against 0.303 for the seasonal naive forecast. Tree methods need stationary targets: on the price level, a forest trained on levels stays flat at the highest training value, while the same forest trained on monthly changes follows the index.

# Neural networks

Neural networks are flexible function approximators built from layers of simple units. The section presents the multilayer perceptron, the way it is trained, and two networks designed for sequences: the recurrent network and the long short-term memory (LSTM) network. The evidence on when such networks help in forecasting closes the section.

## The multilayer perceptron {h=e45e40ae}

The multilayer perceptron (MLP) with one hidden layer of $q$ neurons computes, for an input vector $\mathbf{x}$,

$$
\begin{aligned}
h_j &= g(b_j + \mathbf{w}_j'\mathbf{x}),\quad j = 1, \dots, q, \\
\hat y &= c + \sum_{j=1}^{q} v_j h_j .
\end{aligned}
$$ {#eq:mlp}

Each hidden neuron $j$ forms a weighted sum of the inputs with weights $\mathbf{w}_j$ and bias $b_j$, and passes it through an activation function $g$. The output is a linear combination of the hidden outputs $h_j$, with weights $v_j$ and constant $c$. Common activations are ReLU, $\max(0, z)$, the hyperbolic tangent and the logistic function. The activation is what makes the network nonlinear: with $g(z) = z$ the whole network collapses to a linear regression.

@HornikStinchcombeWhite1989 proved that such a network can approximate any continuous function on a bounded set as closely as desired, if it has enough hidden neurons. This universal approximation property explains the flexibility of networks; it does not say how many neurons are needed or how much data is required to estimate them. The number of parameters grows quickly: with 3 inputs and 5 hidden neurons there are $3 \cdot 5 = 15$ input weights, 5 hidden biases, 5 output weights and 1 output constant, 26 parameters in all.

The image shows the Mark I Perceptron, built at Cornell around 1960, the first trainable neural network, implemented in hardware.

## A small network and its activation functions {h=23b2bc53}

The left panel draws the network of the example for tomorrow's load: three inputs (today's load $y_t$, the load on the same weekday of the previous week $y_{t-6}$ and a holiday flag), a hidden layer of five neurons and one output, $\hat y_{t+1}$. Every line is a weight; each hidden neuron receives all three inputs, and the output receives all five hidden values.

The right panel plots the three activation functions. The logistic function rises from 0 to 1 and the hyperbolic tangent from $-1$ to 1, both smoothly and both flat in the tails; ReLU is 0 for negative arguments and the identity for positive ones. The flat tails of the first two make their derivatives small for large inputs, which slows training in deep networks; ReLU avoids this for positive arguments, which is one reason for its popularity.

## Training a network {h=e7401bae}

A network is trained by minimising the training loss with gradient descent: the weights move in small steps against the gradient of the loss. The gradient with respect to every weight is computed by back-propagation [@RumelhartHintonWilliams1986], an application of the chain rule that passes the error from the output back through the layers. Stochastic gradient descent (SGD) and its variant Adam compute each step on a small random batch of rows instead of the whole sample; one pass through all the data is called an epoch.

The loss of a network, as a function of its weights, has many local minima, so the result depends on the random starting weights. Two practices follow: fix the random seed so that the result can be reproduced, and average a few networks trained from different seeds, which forms an ensemble with a lower variance.

Networks overfit easily and are regularised in three main ways: an L2 penalty on the weights, as in ridge regression; early stopping on validation data, as in boosting; and dropout, which switches off random neurons during training. Inputs and targets must be standardised, with the statistics of the training window. With a few thousand rows, as in the load data, small networks are appropriate; deep networks need many series or long high-frequency data.

## Recurrent networks and the LSTM (1/2): the recurrent network {h=8bf0c6e5}

A recurrent neural network (RNN) reads a sequence one step at a time and carries a hidden state that summarises the past. At each step the state is updated from the previous state and the new input:

$$ h_t = \tanh(W h_{t-1} + U x_t + b) , $$

where $x_t$ is the input at time $t$, $h_t$ the hidden state, and $W$, $U$ and $b$ are weights and a bias shared by all steps. The hyperbolic tangent keeps each component of the state between $-1$ and 1. The forecast is read from the last state, for example $\hat y_{t+1} = v'h_t + c$. Sharing the weights across steps is what lets the same network process sequences of any length.

Training an RNN means differentiating through many steps, which multiplies many derivatives of the form $\partial h_t/\partial h_{t-1}$. If they are smaller than 1, the product shrinks towards zero (vanishing gradients); if they are larger than 1, it explodes. In practice a plain RNN cannot learn dependencies that reach more than a few steps back.

The review of @HewamalageBergmeirBandara2021 summarises the evidence for forecasting. On single, short series, recurrent networks are rarely more accurate than ETS or ARIMA. They become competitive with many related series and enough data, as global models such as DeepAR [@SalinasEtAl2020], which train one network on thousands of series.

## Recurrent networks and the LSTM (2/2): the LSTM cell {h=126db729}

The long short-term memory network of @HochreiterSchmidhuber1997 adds a cell state $c_t$ that is updated additively and controlled by three gates. Each gate is a vector of numbers between 0 and 1, computed by a logistic layer from the previous output and the new input:

$$
\begin{aligned}
f_t &= \sigma(W_f[h_{t-1}, x_t] + b_f), \\
i_t &= \sigma(W_i[h_{t-1}, x_t] + b_i), \\
o_t &= \sigma(W_o[h_{t-1}, x_t] + b_o), \\
\tilde c_t &= \tanh(W_c[h_{t-1}, x_t] + b_c).
\end{aligned}
$$

Here $\sigma(z) = 1/(1 + e^{-z})$ is the logistic function, $[h_{t-1}, x_t]$ stacks the two vectors, and $W_f, W_i, W_o, W_c$ and $b_f, b_i, b_o, b_c$ are weights and biases. The forget gate $f_t$, the input gate $i_t$ and the output gate $o_t$ decide what is kept, what is added and what is passed on; $\tilde c_t$ is the candidate new content.

The cell and the output are then updated as

$$
\begin{aligned}
c_t &= f_t \odot c_{t-1} + i_t \odot \tilde c_t, \\
h_t &= o_t \odot \tanh(c_t) ,
\end{aligned}
$$ {#eq:lstm}

where $\odot$ is the element-wise product. When the forget gate is close to 1, the old memory passes through many steps almost unchanged, and so does its gradient, which solves the vanishing-gradient problem of the plain RNN. The price is the number of weights: each gate is a full layer, and an LSTM with 32 units and 8 inputs has about 5,400 weights, four times as many as a plain RNN of the same size.

## A recurrent network and the LSTM cell {h=b98731f6}

The left diagram unrolls an RNN over three steps. Each green box is the hidden state at one time, each blue box an input, and the red arrows carry the state forward with the same weights at every step; the forecast $\hat y_{t+1}$ is read from the last state.

The right diagram shows the LSTM cell. The red line is the cell state, which flows from $c_{t-1}$ to $c_t$ as a long memory; the three gates act on it from below. The forget gate scales the old memory, the input gate admits new content, and the output gate selects what part of the memory becomes the output $h_t$. The equations under the diagram are those of @eq:lstm.

## Sepp Hochreiter and the LSTM {h=7e73e984}

In his 1991 diploma thesis in Munich, Sepp Hochreiter analysed why the gradients of recurrent networks vanish over long sequences. Together with Jürgen Schmidhuber he then proposed the LSTM [@HochreiterSchmidhuber1997], whose additive cell state lets the gradient pass through many steps.

The paper became one of the most cited in neural computing. LSTMs powered speech recognition and machine translation systems for many years, until transformers, the architecture behind the foundation models of [Chapter 11](../EN/Courses/chapter11_foundation_models_time_series.pdf), replaced them in most of these tasks.

## Recap: Neural networks {h=85e0eaf2}

An MLP stacks layers of units $g(b + \mathbf{w}'\mathbf{x})$ and is trained by gradient descent; the seed, the scaling of the data, the L2 penalty and early stopping decide whether training succeeds. RNNs and LSTMs process sequences, and the LSTM cell state avoids vanishing gradients. Small data favour small models; neural networks become competitive with many series or long data.

## @end

Neural networks can approximate any continuous function, but this flexibility has to be paid for with data and careful training. For a single daily series with a few thousand observations, a small MLP is a reasonable candidate; recurrent networks and the LSTM are designed for long sequences and many related series, and the next section shows what happens when they are applied to one series of moderate length.

# Application: electricity load against Chapter 4

This section puts the ML methods into the forecasting exercise of [Chapter 4](../EN/Courses/chapter4_seasonality_forecasting.pdf). The statistical models of that chapter and six ML models forecast the daily electricity load of Romania from the same 26 origins, for the same 14 horizons, and are compared with the same accuracy measures and the Diebold–Mariano test.

## The forecasting exercise {h=d1201b43}

The evaluation design is that of [Chapter 4](../EN/Courses/chapter4_seasonality_forecasting.pdf). There are 26 forecast origins, one every 14 days from 30 June 2025 to 15 June 2026, and from each origin every model forecasts the next 1 to 14 days. The training window is expanding: at each origin it contains all data from 1 January 2022. The statistical models are those of Chapter 4: the seasonal naive forecast, ETS, seasonal ARIMA (SARIMA), dynamic harmonic regression (DHR) and the combination of these forecasts.

The ML models are refitted at every origin. Ridge, a random forest of 200 trees and gradient boosting with 300 trees and learning rate $\nu = 0.05$ use the direct strategy, with one model per horizon. A second gradient boosting model uses the recursive strategy on the one-step table. Two neural networks use the multi-output strategy: an MLP, averaged over 3 networks with 32 hidden neurons, and an LSTM that reads the last 56 days and outputs the next 14.

The MASE divides the MAE of the forecasts by a fixed scale, the in-sample MAE of the weekly seasonal naive method, 309 MW. A MASE below 1 means that the model is more accurate out of sample than the seasonal naive forecast is in sample.

## Results: 26 origins, 14 horizons {h=69ac729f}

The table gives, for each model, the MAE and the RMSE in GW over all 364 forecasts (26 origins times 14 horizons), the MASE, and the p-value of the Diebold–Mariano test against DHR, the most accurate model. The test is applied to the 26 origin-average absolute errors, with the small-sample correction of @HarveyLeybourneNewbold1997. A small p-value means that the difference in accuracy from DHR is unlikely to be due to chance.

The table splits into three groups. DHR has the lowest MAE, 0.276 GW, and ridge is practically equal, 0.277 GW, with a slightly lower RMSE, 0.337 against 0.350 GW. The MLP (MASE 0.95), direct gradient boosting (0.97), the combination (0.99), recursive gradient boosting (0.99) and the random forest (1.00) form a middle group whose differences from DHR are not significant at 5%. The seasonal naive forecast, ETS, SARIMA and the LSTM are significantly less accurate than DHR, with p-values from below 0.001 to 0.016.

## MASE and accuracy by horizon {h=4eeedcc9}

The left panel ranks the eleven models by MASE: blue bars are the models of [Chapter 4](../EN/Courses/chapter4_seasonality_forecasting.pdf), red bars the ML models, and the dotted line marks 1. DHR, ridge and the MLP are furthest below the line; direct and recursive gradient boosting and the combination are just below it, the random forest is at 1.00, and the LSTM, SARIMA, ETS and the seasonal naive forecast are above it.

The right panel shows the MAE by horizon for six models. Errors grow with the horizon for every model, but not smoothly: most models have their largest errors at horizons 10, 11 and 14. DHR is the most accurate model at short horizons, with an MAE of 0.18 GW one day ahead. Direct gradient boosting stays close to DHR from horizon 3 onwards, while the seasonal naive forecast and the LSTM show the largest peaks.

## Interpreting the load results {h=7344dcea}

DHR, with a MASE of 0.89, and ridge, with 0.90, are the most accurate models, and the difference between them is not significant (Diebold–Mariano p-value 0.942). Both are linear regressions with the right calendar features: Fourier terms, weekday effects and holiday dummies. The trees and the networks do not improve on them.

Direct gradient boosting (MASE 0.97), the MLP (0.95) and the random forest (1.00) are clearly more accurate than the seasonal naive forecast, with Diebold–Mariano p-values of 0.0015, 0.0074 and 0.0015, and they have a lower MASE than SARIMA and ETS, a difference not tested with Diebold–Mariano. The direct and recursive versions of gradient boosting are close, 0.97 and 0.99, so the strategy matters less than the features here. The LSTM, with a MASE of 1.16, is less accurate than every other model except the seasonal naive forecast, SARIMA and ETS.

The lesson of the exercise is that the features matter more than the algorithm. With four years of daily data on a single series, a well-specified linear model that knows the calendar is hard to beat, and the flexibility of trees and networks has little to add.

## Forecasts around Orthodox Easter 2026 {h=dadbb31a}

The chart zooms in on one origin, 6 April 2026, a week before Orthodox Easter on 12 April. The black line is the actual load, and the coloured lines are the 14-day forecasts of five models from the vertical dotted line onwards. The actual load falls to about 4 GW on Easter Sunday and stays below its usual level during the following week.

The MAE in this window measures how each model handled the holiday: 344 MW for DHR, 536 MW for direct gradient boosting, 503 MW for recursive gradient boosting and 661 MW for the LSTM. DHR forecasts the deepest dip on Easter Sunday; the tree models forecast a shallower one, and the LSTM and the seasonal naive forecast repeat the ordinary weekly pattern, with only the usual Sunday dip.

## Interpreting the Easter window {h=31fa051d}

All models except the LSTM see the Easter flag among their features, but the training data contain only four Orthodox Easters, from 2022 to 2025. A tree can isolate the Easter days only by splitting on the flag, and with four episodes such splits are rare and their leaves are mixed with ordinary days, so the forecast dip is too shallow. DHR estimates one Easter coefficient from the same four episodes; the regression structure pools them efficiently and gives a dip of the right depth.

The LSTM receives no holiday flag. It repeats the weekly pattern of the recent past and misses the holiday entirely. Rare events therefore need either structure, such as a regression coefficient for the event, or many series that share the event, which a global model can learn from.

## Recap: Electricity load {h=0c1343b5}

With the same origins, the same horizons and the same error measures, the ML models enter the comparison of [Chapter 4](../EN/Courses/chapter4_seasonality_forecasting.pdf) on equal terms. Ridge and DHR do not differ significantly (p = 0.942); gradient boosting, the random forest and the MLP have a lower MASE than SARIMA and ETS; the LSTM is the weakest of the ML models. Calendar features and holidays decide the ranking.

## @end

On the daily electricity load of Romania, the most accurate forecasts come from linear models with calendar features: DHR (MASE 0.89) and ridge (0.90), whose difference is not significant. Tree ensembles and the MLP are more accurate than the seasonal naive forecast, SARIMA and ETS, but not than DHR, and the LSTM, without a holiday flag, ranks near the bottom. Around Easter, a regression coefficient estimated from four episodes beats trees that see the same flag.

# Prediction intervals

A point forecast says nothing about its uncertainty, yet many decisions depend on it. The section presents two ways of building prediction intervals for ML models: quantile regression with the pinball loss, which estimates the quantiles directly, and split conformal prediction, which widens a point forecast by the empirical quantile of out-of-sample errors. Both are applied to the load forecasts of the previous section.

## Quantile regression and the pinball loss {h=09c3b33c}

A grid operator needs more than the expected load: it needs the load that will not be exceeded with 95% probability, to plan reserves. The $\tau$-quantile $q_\tau$ of $y_{t+h}$ is the value with $P(y_{t+h} \le q_\tau) = \tau$, and a 90% prediction interval is the pair $[q_{0.05}, q_{0.95}]$. A model that forecasts quantiles therefore gives intervals directly.

Quantiles are estimated by minimising the pinball loss of @KoenkerBassett1978:

$$ L_\tau(y, q) = \begin{cases} \tau\,(y - q), & y \ge q, \\ (1 - \tau)(q - y), & y < q. \end{cases} $$ {#eq:pinball}

The loss is asymmetric. For $\tau = 0.95$, an outcome above the forecast costs 0.95 per unit and an outcome below costs 0.05 per unit, so the forecast is pushed up until only 5% of the outcomes exceed it. Its expected value is minimised exactly by the true $\tau$-quantile, and for $\tau = 0.5$ it equals half the absolute error, whose minimiser is the median.

An example with $\tau = 0.95$ and a forecast $q = 7.0$ GW: an outcome of 7.4 GW costs $0.95 \cdot 0.4 = 0.380$, while an outcome of 6.5 GW costs only $0.05 \cdot 0.5 = 0.025$. Any learner that can minimise a custom loss can forecast quantiles: linear quantile regression, gradient boosting with *loss='quantile'*, or a neural network.

## Split conformal prediction {h=29213051}

Conformal prediction, in the split form of @LeiEtAl2018, builds an interval around any point forecast from the errors of the model on data it has not seen. The procedure has three steps. First, the model is fitted on the training data without the last $n$ points, which form the calibration set. Second, the absolute errors $|e_i|$ of the model on the calibration set are computed and sorted. Third, the interval for a new forecast $\hat y$ is

$$ \hat y \pm \hat q, \qquad \hat q = |e|_{(\lceil (n+1)(1-\alpha) \rceil)} , $$ {#eq:conformal}

where $|e|_{(k)}$ is the $k$-th smallest absolute calibration error, $\alpha$ the target miscoverage ($\alpha = 0.1$ for a 90% interval) and $\lceil\cdot\rceil$ rounding up.

The guarantee is that the interval covers the outcome with probability at least $1 - \alpha$ if the errors are exchangeable, that is, if their joint distribution does not depend on their order. Time-series errors are not exchangeable: their size changes with the season and they are correlated. The guarantee is therefore only approximate, and the remedy is to calibrate on recent data and to check the coverage walk-forward.

In the load application the calibration set has $n = 182$ days and $\alpha = 0.1$, so $\lceil 183 \cdot 0.9 \rceil = 165$ and $\hat q$ is the 165-th smallest of the 182 absolute errors. A separate calibration set is used for each horizon and each origin, so the width of the interval grows with the horizon.

## 90% intervals for the load {h=d7e8f1ab}

The chart evaluates the two methods on the 364 direct gradient boosting forecasts of [Section 6](#sec-6), all origins and all horizons. The left panel shows the outcomes (black dots), the 90% intervals of quantile gradient boosting (shaded band) and those of split conformal prediction (dashed lines). The right panel shows the empirical coverage, the share of outcomes inside the interval, by horizon; the dotted line marks the nominal 90%.

The shaded band is visibly narrower than the conformal interval, and many outcomes, especially the low ones on Sundays and holidays, fall outside it. The conformal coverage stays close to 90% at most horizons, while the coverage of quantile boosting stays between about 55% and 75%.

## Interpreting the intervals {h=a95c95fd}

Quantile gradient boosting covers 66.5% of the outcomes, with a mean width of 884 MW; split conformal prediction covers 90.4%, with a mean width of 1,310 MW. For quantile boosting, 19.0% of the outcomes fall below the interval and 14.6% above it, against a nominal 5% on each side. Its intervals are too narrow on both sides, and more so on the lower side, where the holiday drops are.

Quantiles estimated by trees are noisy in the tails, where each leaf contains few extreme observations, and they reflect the training fit, which is tighter than the out-of-sample errors. Conformal calibration uses the real out-of-sample errors of the point forecast to set the width, so it corrects this optimism. An interval is only informative together with its empirical coverage, which should always be reported.

## Recap: Prediction intervals {h=5d719188}

Quantile forecasts minimise the pinball loss, and an interval is a pair of quantiles. Split conformal prediction builds the interval $\hat y \pm \hat q$ from an empirical quantile of calibration errors. In both cases the coverage is checked walk-forward.

## @end

On the load forecasts, nominal 90% intervals from quantile gradient boosting cover only 66.5% of the outcomes, while split conformal intervals, about 50% wider, cover 90.4%. The coverage of an interval has to be measured out of sample, and calibration on real forecast errors is a simple way to obtain the coverage that was promised.

# Local and global models: Romanian inflation

All models so far were local: one model per series, estimated on that series alone. A global model is estimated on the pooled data of many related series. The section explains the distinction and tests it on Romanian inflation, with gradient boosting trained either on Romania alone or on the 27 countries of the European Union (EU).

## Local and global models {h=47b0c4be}

A local model is estimated on one series only, as ARIMA, ETS and every model of the previous sections. The number of observations of that series limits the complexity the model can afford: a monthly series of 20 years has 240 observations, too few for a large ensemble of trees or a neural network.

A global model is estimated on the pooled rows of many related series; @MonteroMansoHyndman2021 and @JanuschowskiEtAl2020 discuss this class of models. Every series contributes rows with the same features, such as scaled lags and calendar variables, and an identifier of the series can be added. The model has many more rows per parameter, so complex learners such as gradient boosting and neural networks become usable. The winners of the M4 and M5 competitions were global models.

The risk is the reverse of the benefit. The pooled series must share their dynamics; if they do not, a global model averages away what is special about each one.

## The inflation exercise {h=b0d01a0e}

The target is the 12-month inflation rate of Romania measured by the HICP, $\pi_t = 100(P_t/P_{t-12} - 1)$, where $P_t$ is the index in month $t$, forecast at horizons of 1, 3, 6 and 12 months. The strategy is direct, and every model forecasts the change $\pi_{t+h} - \pi_t$ rather than the level, so that the trees never have to extrapolate.

The features at the origin $t$ are the inflation rates $\pi_t, \pi_{t-1}, \pi_{t-2}, \pi_{t-3}, \pi_{t-6}, \pi_{t-12}$, the last three monthly log changes of the index with their 3- and 6-month sums, and the calendar month. Six models are compared: the random walk (RW), which forecasts no change, $\hat\pi_{t+h} = \pi_t$; an AR model estimated by OLS on the same features; the lasso; a random forest; local gradient boosting, trained on Romania only; and global gradient boosting, trained on the pooled data of the 27 EU countries.

The validation is walk-forward with yearly refits: every January, each model is re-estimated on all rows from 2006 whose target month is already observed, so that no training target lies after the origin. The test origins run from January 2016 to July 2026. @MedeirosEtAl2021 found that random forests forecast US inflation better than standard benchmarks when they use more than 100 predictors; the exercise asks whether a similar gain appears with the few predictors available for Romania.

## Romanian inflation: forecasts and relative RMSE {h=3a584e6e}

The left panel shows the 12-month inflation rate of Romania (black) and the 12-month-ahead forecasts of four models, each plotted at its target month, so that a forecast and its outcome are vertically aligned. Inflation rose from below zero in 2016 to a peak in late 2022 and then declined; the forecasts made a year earlier follow it with a lag and miss the 2022 surge.

The right panel gives, for each model and horizon, the RMSE divided by the RMSE of the random walk. A value below the dotted line at 1 means that the model is more accurate than forecasting no change. The AR model lies clearly below 1 up to 6 months, while most ML models lie at or above it.

## Inflation results {h=292bae1e}

The table gives the RMSE in percentage points (pp) for each model and horizon, with the ratio to the random walk in parentheses. The random walk has an RMSE of 0.63 pp at 1 month, 1.41 pp at 3 months, 2.25 pp at 6 months and 3.90 pp at 12 months: errors grow quickly with the horizon. The AR model has ratios of 0.90 at 1, 3 and 6 months and 1.10 at 12 months. The lasso and the random forest have ratios between 1.01 and 1.09, and gradient boosting is worse at most horizons, with ratios of up to 1.21 for the local model at 12 months.

The Diebold–Mariano test, based on squared errors and with $h - 1$ autocovariances to account for the overlap of $h$-step errors, gives p-values for AR against the random walk of 0.019 at 1 month, 0.186 at 3 months and 0.388 at 6 months. For global against local gradient boosting the p-values are 0.117, 0.113, 0.999 and 0.828 at the four horizons. The number of test origins falls from 127 months at $h = 1$ to 116 months at $h = 12$.

## Interpreting the inflation results {h=06c747b0}

The linear AR model is the most accurate up to 6 months, with an RMSE about 10% below the random walk, but its advantage is significant only at 1 month (p = 0.019). No ML model beats the random walk significantly at any horizon. At 12 months no model beats the random walk at all: the 2022 peak, 14.6% in November 2022, came from energy prices, an external shock that the past of inflation did not contain.

The global gradient boosting model is more accurate than the local one at short horizons, with an RMSE of 1.37 against 1.64 pp at 3 months, but the difference is not significant (p = 0.113). Pooling 27 countries gives the trees enough rows to learn from; the local model, with 120 to 240 rows at each refit, overfits.

The sample size limits what can be concluded. About 120 monthly test errors, strongly affected by one episode, cannot separate models whose RMSEs differ by 5%. The honest conclusion for most comparisons is that the data give no evidence of a difference, which is not the same as evidence that the models are equally accurate.

## Recap: Local and global models {h=b5efb25b}

A global model pools many related series and lets flexible learners use more data. For Romanian inflation, the AR model is the most accurate, global gradient boosting is more accurate than local gradient boosting, and no model beats the random walk at 12 months. With small test samples, Diebold–Mariano p-values must be reported next to RMSE rankings.

## @end

Monthly inflation offers too few observations per country for flexible learners: a linear AR is the most accurate model up to 6 months, and gradient boosting gains from pooling 27 countries without becoming better than the random walk. The comparison also shows the limits of inference with about 120 monthly test errors.

# Applications in finance: volatility and the sign of returns

[Chapters 1](../EN/Courses/chapter1_stochastic_processes_stationarity.pdf) and [5](../EN/Courses/chapter5_conditional_volatility_garch.pdf) showed that the size of daily returns is predictable and their direction is not. The section tests whether machine learning changes this picture. The first application forecasts realised volatility against the HAR model; the second predicts the sign of tomorrow's return against the correct baseline.

## Realised volatility: HAR against ML (1/2) {h=eb8034ca}

The target is the logarithm of the mean daily variance over the next 5 trading days. The daily variance is not observed, so it is estimated from the open, high, low and close prices of each day, following @GarmanKlass1980:

$$
\begin{aligned}
\hat\sigma_t^2 &= [\ln(O_t/C_{t-1})]^2 + \tfrac12[\ln(H_t/L_t)]^2 \\
&\quad - (2\ln 2 - 1)[\ln(C_t/O_t)]^2 ,
\end{aligned}
$$

where $O_t$, $H_t$, $L_t$ and $C_t$ are the open, high, low and close of day $t$. The first term is the squared overnight return; the other two form the Garman–Klass estimator of the variance during the trading day, which uses the daily range and is much more precise than a squared close-to-close return. Volatility is persistent and predictable ([Chapter 5](../EN/Courses/chapter5_conditional_volatility_garch.pdf)), so this is a problem where ML could find structure.

The benchmark is the heterogeneous autoregressive (HAR) model of @Corsi2009, an OLS regression on the log variance over the last day, week and month:

$$
\begin{aligned}
\ln \bar\sigma^2_{t+1:t+5} &= c + \beta_d \ln\hat\sigma_t^2 + \beta_w \ln\bar\sigma^2_{t-4:t} \\
&\quad + \beta_m \ln\bar\sigma^2_{t-21:t} + u_t ,
\end{aligned}
$$ {#eq:har}

where $\bar\sigma^2_{a:b}$ is the mean of $\hat\sigma^2$ from day $a$ to day $b$ and $u_t$ is the error. The three averages represent traders with daily, weekly and monthly horizons, and together they reproduce the slow decay of volatility. The ML models receive the three HAR features and 8 more: further lags of the daily variance, the quarterly mean, the daily return, its negative part and its absolute value.

## Realised volatility: HAR against ML (2/2) {h=aa3d15cf}

The models are evaluated walk-forward from 2013, refitted every year, with a gap of 5 days between the training and the test data, because the 5-day targets overlap. Accuracy is measured relative to HAR by the out-of-sample $R^2$:

$$ R^2_{\text{HAR}} = 1 - \frac{\sum_t (y_t - \hat y_t)^2}{\sum_t (y_t - \hat y_t^{\text{HAR}})^2} , $$

where $y_t$ is the realised log variance and $\hat y_t$, $\hat y_t^{\text{HAR}}$ are the forecasts of the model and of HAR. A positive value means that the model has a smaller squared error than HAR.

The second measure is the QLIKE loss of @Patton2011, the quasi-likelihood loss for variance forecasts used in [Chapter 5](../EN/Courses/chapter5_conditional_volatility_garch.pdf):

$$ L(\sigma^2, h) = \frac{\sigma^2}{h} + \ln h , $$ {#eq:qlike}

where $\sigma^2$ is the realised variance and $h$ the variance forecast. For a given $\sigma^2$ the loss is smallest at $h = \sigma^2$, and it penalises forecasts that are too low more than forecasts that are too high. @Patton2011 showed that it is robust to the noise in the variance proxy: it ranks forecasts in the same order as the true, unobserved variance would. Lower values are better, and the Diebold–Mariano test of [Chapter 4](../EN/Courses/chapter4_seasonality_forecasting.pdf) is applied to the daily differences in QLIKE.

## Results: S&P 500 and DAX, 2013–2026 {h=300e43b9}

The table reports, for the S&P 500 and the German DAX index, the out-of-sample $R^2$ relative to HAR in percent, the mean QLIKE and the Diebold–Mariano statistic $t$ against HAR, over 3,444 and 3,473 test days. A statistic below $-1.96$ means that the model has a significantly lower QLIKE than HAR at the 5% level.

The lasso and the MLP improve on HAR on both indices. On the S&P 500 the lasso has an $R^2$ relative to HAR of 7.1% and a QLIKE of 0.400 against 0.414 for HAR (DM $t = -5.12$), and the MLP 8.3% and 0.398 ($t = -3.96$). On the DAX the gains in $R^2$ are similar, 7.9% and 8.6%, but only the lasso lowers QLIKE significantly ($t = -2.42$; for the MLP $t = -1.69$). The random forest and gradient boosting do not beat HAR: their $R^2$ relative to HAR is between $-0.5$% and 3.5% and their QLIKE is not lower.

The gain comes from the extra features, above all the negative part of the return, which captures the leverage effect of [Chapter 5](../EN/Courses/chapter5_conditional_volatility_garch.pdf): falls in prices raise future volatility more than rises. The lasso and the MLP use these features through smooth functions, while trees cut a smooth relation into steps, which costs them the gain.

## Volatility forecasts in 2020 and 2025 {h=433ea3e3}

The chart shows the annualised realised volatility of the S&P 500 over the next 5 days (black) and the forecasts of HAR, the random forest and gradient boosting in 2020 and 2025. In March 2020, during the COVID-19 crash, realised volatility reached about 100% per year; in early April 2025 it reached about 70%.

In both episodes all forecasts rise only after realised volatility has risen, and none of them reaches the peak. A forecast built from past variances reacts to a shock after it has happened, so the first days of a storm are always underestimated. After the peaks, the forecasts follow the decay of volatility closely. The three models are close to one another throughout, which agrees with the small differences in the table.

## Forecasting the sign of the next-day return: choice of the baseline {h=792ae2ee}

The second application is a classification problem. The target is $1\{r_{t+1} > 0\}$, equal to 1 if tomorrow's return is positive and 0 otherwise. The features are the last five daily returns, the sums of returns over 5, 21 and 63 days and the volatility over 21 and 63 days. Three classifiers are compared, logistic regression (logit), a random forest and gradient boosting, evaluated walk-forward with yearly refits from 2008.

The right baseline is not 50%. Stock indices rise on more days than they fall, so the rule "always predict up", the majority class of the training data, is right on 54.4% of the S&P 500 days and 53.8% of the BET days in the test period. A classifier is useful only if it beats this rule. The accuracies are 53.8% for logit, 53.1% for the random forest and 52.5% for gradient boosting on the S&P 500, and 53.7%, 53.6% and 52.2% on the BET.

Accuracy depends on a threshold; the area under the ROC curve (AUC) does not. It is the probability that a randomly chosen up day receives a higher model score than a randomly chosen down day: 0.5 means no ability to rank days and 1 a perfect ranking. The models have AUC values between 0.505 and 0.533, barely above 0.5, so their ability to rank days by the probability of a rise is at most very weak.

## Accuracy against the baseline {h=aba765da}

The chart compares the out-of-sample accuracy of the three classifiers with the baseline, from 2008 to 2026, on 4,705 S&P 500 days and 4,686 BET days. The golden bars are the baseline, the share of up days.

On both indices every classifier is below or at the baseline. On the BET, logit and the random forest are within 0.2 percentage points of it; on the S&P 500 all three are below it by 0.6 to 1.9 points. Gradient boosting, the most flexible model, has the lowest accuracy on both indices.

## Interpreting the sign experiment {h=0b3f1fd6}

A model with "53% accuracy" sounds skilful, because it is right more often than a coin. Against the 54.4% of the rule "always up", it is worse than predicting the same answer every day. Gradient boosting on the S&P 500 is significantly worse than the baseline (p = 0.008): the flexible model fits patterns of the training years that are noise, and pays for them out of sample.

The result is the weak-form efficiency of [Chapter 1](../EN/Courses/chapter1_stochastic_processes_stationarity.pdf) seen through classifiers: past returns contain almost no information about the sign of the next return. Together with the volatility results, it confirms what the statistical models of [Chapters 1](../EN/Courses/chapter1_stochastic_processes_stationarity.pdf) and [5](../EN/Courses/chapter5_conditional_volatility_garch.pdf) showed: volatility is predictable, the direction is not.

## Recap: Volatility and the sign of returns {h=3f3fbe47}

For realised volatility, the lasso and the MLP with extra features are more accurate than HAR, and the trees are not. For the sign of returns, the comparison is with the majority class, not with 50%, and no model beats it.

## @end

Machine learning adds value in finance where there is structure to find: the lasso and the MLP gain 7 to 9% of $R^2$ over HAR in volatility forecasting, thanks to the asymmetric response of volatility to returns. Where there is no structure, as in the direction of daily returns, flexible models only fit noise, and the majority-class baseline exposes it.

# Landmark case studies: M4, M5 and Gu, Kelly and Xiu

The applications of this chapter use one series or a few. The three case studies below show the evidence at a much larger scale: two open forecasting competitions on tens of thousands of series, and a study of machine learning for the returns of US stocks.

## The M competitions {h=9781be40}

Since 1982 Spyros Makridakis has organised open forecasting competitions, the M competitions. Every participating method forecasts the same series, and the organisers evaluate the forecasts on data that were hidden from the participants. This is the honest version of the walk-forward test: no participant can tune a method on the test data, and the results compare methods on equal terms.

Before M4, @MakridakisSpiliotisAssimakopoulos2018a compared popular ML methods with eight statistical methods on 1,045 monthly series of the M3 competition. The ML methods were less accurate at every horizon and needed much more computation. The M4 competition of 2018 [@MakridakisSpiliotisAssimakopoulos2020] then used 100,000 series of six frequencies, from yearly to hourly, and ranked 61 methods by the overall weighted average (OWA):

$$ \text{OWA} = \frac12\Bigl(\frac{\text{sMAPE}}{\text{sMAPE}_{N2}} + \frac{\text{MASE}}{\text{MASE}_{N2}}\Bigr) . $$

Here $N2$ is Naive2, the naive forecast of the seasonally adjusted series, and an OWA below 1 means more accurate than Naive2. The symmetric mean absolute percentage error is $\text{sMAPE} = \frac{200}{H}\sum_{h=1}^{H} |y_{T+h} - \hat y_{T+h}| / (|y_{T+h}| + |\hat y_{T+h}|)$, in percent. The M5 competition of 2020 [@MakridakisSpiliotisAssimakopoulos2022] used 42,840 hierarchical daily series of Walmart unit sales, with a horizon of 28 days.

## M4: the accuracy of all methods {h=31d8bc36}

The chart plots the OWA of the 59 ranked methods of M4, submissions and benchmarks, against their rank, from the official evaluation file of the competition. The colours give the type of method: statistical, combinations of statistical methods, combinations of statistical and ML methods, combinations of ML methods, pure ML, hybrid and other. The dotted line is Naive2 (OWA = 1) and the dashed line the Comb benchmark; two methods with an OWA above 1.7 are outside the chart.

The ranking is dominated by combinations in the first places, with the single hybrid method in first place. The pure ML methods (red) are scattered: one ranks 23rd, the others rank 46th or lower, at the right end of the chart.

## Interpreting M4 {h=28a6f270}

The winner was the hybrid ES-RNN of @Smyl2020: exponential smoothing equations whose parameters are learned by an LSTM trained on all series together, which makes it a global model. It reached an OWA of 0.821 and a sMAPE of 11.37%. The Comb benchmark, the simple average of simple exponential smoothing (SES), Holt's linear trend and the damped trend method, reached an OWA of 0.898 and a sMAPE of 12.55%; Naive2 had a sMAPE of 13.56%.

The six pure ML methods did poorly: the best of them ranked 23rd, with an OWA of 0.915; none was more accurate than Comb and only one was more accurate than Naive2, as @MakridakisSpiliotisAssimakopoulos2018b report. Twelve of the 17 most accurate methods were combinations. On short and heterogeneous series, the lesson is to combine statistical models or to let ML learn across series; pure ML applied series by series loses to simple benchmarks.

## M5: accuracy of gradient boosting on retail data {h=db3bb484}

M5 attracted 5,507 teams from 101 countries. The data were the daily unit sales of 3,049 products in 10 Walmart stores, organised in a hierarchy of products, departments, categories, stores and states. The error measure was the weighted root mean squared scaled error (WRMSSE). For each series, the RMSSE divides the RMSE of the forecasts by the in-sample RMSE of the one-step naive forecast; the WRMSSE averages the RMSSE over all series with weights proportional to their sales value.

The winning method reached a WRMSSE of 0.520, 22.4% more accurate than the best benchmark, bottom-up exponential smoothing, with 0.671. Most of the top methods used LightGBM [@KeEtAl2017]. The winner averaged LightGBM models pooled by store, by category and by department, with both recursive and direct strategies.

M5 had exactly the conditions under which ML wins: many related series, global models trained across them, and rich calendar and price features such as holidays, promotions and price changes. These conditions differ sharply from those of the single-series applications of this chapter.

## Gu, Kelly and Xiu (2020): machine learning for stock returns {h=b2142d59}

@GuKellyXiu2020 applied machine learning to the monthly returns of about 30,000 US stocks from 1957 to 2016, with 920 predictors built from stock characteristics, their interactions with macroeconomic variables and industry dummies. They compared 12 methods, from OLS to neural networks with five hidden layers, in a walk-forward design: 18 years of training, 12 years of validation and 30 years of test, with yearly refits.

The out-of-sample monthly $R^2$ was at most 0.40%, reached by a network with three hidden layers. OLS with all predictors had an $R^2$ of $-3.46$%: with so many predictors, unpenalised least squares overfits badly. Small as it is, the predictability has economic value. A long-short portfolio, which buys the decile of stocks with the highest forecasts and sells the decile with the lowest, reaches an annual Sharpe ratio of 1.35 with the four-layer network, against 0.61 for OLS-3, an OLS regression on three classic predictors. The Sharpe ratio is the mean annual return divided by its annual standard deviation.

## Gu, Kelly and Xiu (2020): out-of-sample R² and Sharpe ratios {h=4c17e785}

The chart reproduces two published tables of the paper. The left panel gives the monthly out-of-sample $R^2$ for all stocks (their Table 1), the right panel the annualised Sharpe ratio of the value-weighted long-short decile portfolio (their Table 7), for each method; the colours separate linear models, trees and neural networks. The suffix +H marks models estimated with the Huber loss, which reduces the influence of extreme returns.

The methods are OLS-3, an OLS regression on size, book-to-market and momentum; PLS and PCR, partial least squares and principal component regression; ENet, the elastic net, which combines the ridge and lasso penalties; GLM, a generalised linear model with splines; RF, the random forest; GBRT, gradient-boosted regression trees; and NN1 to NN5, networks with one to five hidden layers. Both panels show the same order: the nonlinear models, and the networks above all, have higher $R^2$ and higher Sharpe ratios than the linear ones. An $R^2$ below half a percent per month is small for a single forecast, but applied to thousands of stocks every month it is enough for a profitable portfolio.

## Recap: Case studies {h=98612a6f}

In M4, combinations and a hybrid global model were the most accurate, while pure ML applied series by series lost to simple benchmarks. In M5, global LightGBM models with calendar and price features won clearly. @GuKellyXiu2020 found a small but real predictability in the cross-section of stock returns, where nonlinear models add economic value.

## @end

The large-scale evidence agrees with the applications of the chapter. Machine learning wins where many related series, global models and rich features are available, as in M5 and in the cross-section of stocks; on short, heterogeneous series analysed one at a time, combinations of statistical methods remain hard to beat.

# Possible contribution of AI

The section points to foundation models, the extreme case of global models, and lists what an AI assistant can contribute to an ML forecasting project and what the analyst has to check, because the typical errors of such projects, leakage above all, produce no error message.

## Foundation models: a pointer to Chapter 11 {h=1497be5c}

A foundation model for time series is a large neural network, usually a transformer, pre-trained on millions of series and then applied to a new series without retraining, which is called zero-shot forecasting. Chronos, TimesFM, Moirai and Lag-Llama are examples. They are global models taken to the extreme: one model for nearly every series. [Chapter 11](../EN/Courses/chapter11_foundation_models_time_series.pdf), a self-study chapter, presents them.

The rules of this chapter apply to them without change. The test data must be data the model has not seen in pre-training, which is hard to guarantee for public series; the evaluation is walk-forward; and the comparison uses the same benchmarks, the MASE and the Diebold–Mariano test.

## Possible contribution of AI {h=5f40da58}

An assistant can help in three ways. It can write a first draft of the code: the feature table, the walk-forward loop and the evaluation table. It can give a second explanation of a method, such as boosting, the LSTM gates or conformal prediction. And it can explore many feature sets and hyperparameters at once, provided the exploration uses validation data only and the test data stay untouched.

The example prompt specifies every choice that matters for a leakage-free result: the features (lags, rolling means, calendar), the model (*HistGradientBoostingRegressor*), the strategy (direct, horizons 1 to 14), the evaluation (walk-forward against the seasonal naive forecast, with the MASE and the Diebold–Mariano test) and the rule that every preprocessing step is refitted inside each training window. A precise prompt produces code whose results can be checked against those of this chapter.

## Checks you must run {h=0df18780}

The checks follow the errors described in this chapter. Every feature must use data up to the origin only; the simplest check is to print a few rows of the table and verify the shifts of the lags and rolling windows by hand. The code must contain no random shuffling of the rows in cross-validation, such as *KFold(shuffle=True)*, and no scaler or feature selection fitted on the full sample.

The benchmarks must be present: the naive or seasonal naive forecast and ETS or ARIMA, evaluated on the same origins. A comparison of forecasts needs a Diebold–Mariano test, and a classifier needs the majority-class baseline. The random seeds must be fixed, and the conclusion must survive a change of seed. Finally, every reference that an assistant cites must exist, which is checked through its DOI.

## @end

An assistant can draft the feature table, the validation loop and the evaluation, and explain the methods, but its output is accepted only after the checks for leakage, benchmarks, significance tests, seeds and references.

# Summary

The summary collects the main ideas of the chapter, the formulas to remember and three self-assessment questions.

## Key takeaways {h=38d1c5e3}

Forecasting with machine learning is supervised learning on a table of lags, rolling statistics and calendar features known at the origin. Only walk-forward validation is honest for time series; leakage produces spectacular and false scores. Ridge and lasso, random forests, gradient boosting, the MLP and the LSTM each have a complexity parameter that validation sets, and trees cannot extrapolate, so they model changes, not levels.

On the data of this chapter, linear models with good features match or beat ML on the electricity load and on inflation; ML helps for volatility, where the lasso and the MLP beat HAR; and no model predicts the sign of daily returns. ML wins with many related series, as in global models, the M4 hybrid and the LightGBM models of M5. Whatever the method, it is compared with benchmarks, with the MASE and the Diebold–Mariano test.

## Key formulas {h=249d2075}

The table collects the formulas of the chapter in the order in which they were introduced: the direct forecast from lags and calendar features of the target day; the ridge @eq:ridge and lasso penalties; the bias–variance decomposition @eq:bv; the boosting update @eq:boost; the MLP @eq:mlp; the LSTM cell and output @eq:lstm; the pinball loss @eq:pinball; the split conformal interval @eq:conformal; and the MASE, the MAE scaled by the in-sample MAE of the seasonal naive forecast with period $m$.

## Self-assessment {h=359fc028}

The first question concerns leakage. A 7-day moving average centred on $t$ averages $y_{t-3}, \dots, y_{t+3}$, so as a feature for $y_{t+1}$ it contains the target itself and the two following values; it is not allowed. The trailing mean of $y_{t-6}, \dots, y_t$ is the correct feature.

The second question concerns extrapolation. A random forest trained on GDP levels up to 2019 forecasts each level as a mean of training levels, so its forecast for 2024 cannot exceed the largest level seen before 2020, while GDP has grown since. The forest should be trained on growth rates. The third question concerns the baseline: a sign classifier with 53% accuracy is useful only if it beats, significantly, the share of up days in the test period, about 54% for the S&P 500. The next chapter, [Chapter 10](../EN/Courses/chapter10_state_space_kalman_markov_switching.pdf), presents state space models, the Kalman filter and Markov switching.

## @end

The chapter has placed machine learning inside the forecasting framework of the course: the same table of past information, the same walk-forward evaluation and the same benchmarks. Under these rules, flexible methods do not automatically win. They win when the data offer what they need, many related series, rich features and stable structure, and a well-specified linear model remains the benchmark to beat.
