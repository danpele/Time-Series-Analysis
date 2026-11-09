<!--
Course text of the web version, chapter 11 (EN). Rendered into the page by latex/build_web_lecture.py.
Format: # section title (introduction), ## frame title {h=hash} (text), @after (text after the live block), ## @end (synthesis).
Numbered equations: $$ ... $$ {#eq:label}, reference @eq:label. Citations: @Key -> Author (year), [@Key] -> (Author, year).
Current hashes: python3 build_web_lecture.py 11 --hashes
-->

## @intro

@epigraph BommasaniEtAl2021 | On the opportunities and risks of foundation models
> AI is undergoing a paradigm shift with the rise of models (e.g., BERT, DALL-E, GPT-3) that are trained on broad data at scale and are adaptable to a wide range of downstream tasks. We call these models foundation models to underscore their critically central yet incomplete character.

@text
Up to this chapter, every forecast in the course started from a model built for a single series. ETS, ARIMA, GARCH and VAR models are estimated on the data of the series, and their parameters describe that series and nothing else. The machine learning models of [Chapter 9](../EN/Courses/chapter9_machine_learning_time_series.pdf) took a first step further: one model, trained on all the series of a data set. Foundation models for time series take the idea to its end. A large model is trained once, by its developer, on a very broad corpus of series from many domains, and then forecasts series it has never seen, with no estimation on them at all.

The quotation above, from the report of @BommasaniEtAl2021, gave this class of models its name in the field of language and images. The question of the chapter is whether the idea also works for time series: can a pretrained model forecast Romania's electricity load, its industrial production or the EUR/RON exchange rate as well as a model estimated on these very series? The answer needs three ingredients: an understanding of how a series of numbers becomes the input of a neural network, scores suited to probabilistic forecasts, and an honest test, on the same origins, against the classical benchmarks.

The answer obtained on the seven series of the chapter is nuanced. On hourly electricity load, the best foundation model reduces the weighted quantile loss by 42% relative to the seasonal naive forecast; on quarterly GDP, an ARIMA model remains better; on the EUR/RON rate and the unemployment rate, no model moves far from the naive forecast. The chapter shows why each of these results arises and what must be checked before believing a published score.

@chart tsa_ch11_series
The seven series of the test: Romania's hourly electricity load (ENTSO-E, June 2026), Romania's industrial production, non-seasonally adjusted quarterly real GDP, monthly HICP inflation and unemployment rate (Eurostat), the NBR EUR/RON reference rate and US retail sales (FRED).

@goals
- **Define** a foundation model, zero-shot forecasting and fine-tuning, and distinguish them from local and global models.
- **Explain** how a series becomes the input of a Transformer: attention, mean scaling, quantisation and patching.
- **Describe** the Chronos, TimesFM, Lag-Llama, Moirai and TimeGPT families and say how they differ.
- **Evaluate** a quantile forecast with the pinball loss, the CRPS, the WQL, the MASE and the coverage of the 80% interval.
- **Compare** a zero-shot model honestly with the seasonal naive forecast, ETS and ARIMA, and recognise leakage and contamination.

@plan
- The foundation-model idea | Local, global and pretrained models; zero-shot and fine-tuning; pretraining at scale.
- Transformers in brief | The encoder–decoder architecture, self-attention and a numerical example of attention weights.
- From numbers to tokens | Mean scaling, the quantisation of Chronos and patching, on Romanian data.
- The main model families | Chronos, Chronos-Bolt, Chronos-2, TimesFM, Lag-Llama, Moirai and TimeGPT.
- Probabilistic forecasts and their scores | Quantile forecasts, the pinball loss, the CRPS, the WQL, the MASE and coverage.
- Zero-shot forecasts on real data | Electricity load, industrial production and the EUR/RON rate.
- Rolling-origin evaluation on seven series | A rolling-origin test on seven series; coverage, horizon, context and model size.
- Leakage and contamination | Mistakes in your own code and test series present in the pretraining corpus.
- Possible contribution of AI | What an assistant can draft and what must be checked.
- Summary | Key ideas, formulas and self-assessment questions.

## Today's question and route {h=1811a05d}

The chapter asks a single question: can one large model, pretrained on many series, forecast a series it has not seen as well as a model built for that series? In [Chapters 0](../EN/Courses/chapter0_introduction.pdf)–[10](../EN/Courses/chapter10_state_space_kalman_markov_switching.pdf), the implicit answer was that the question does not arise: every model, whether ETS, ARIMA, GARCH or VAR, is estimated on the series it forecasts. [Chapter 9](../EN/Courses/chapter9_machine_learning_time_series.pdf) introduced global models, trained on all the series of a data set, but still on the data of the problem.

The route has four stages. The first defines the foundation-model idea and presents, briefly, the network on which almost all these models rest, the Transformer. The second shows how a series of numbers becomes a sequence of inputs for this network and describes the main model families. The third defines the scores for probabilistic forecasts and applies the models to Romanian and US series in a rolling-origin test. The fourth discusses leakage and contamination, the reasons why a score can look better than it is, and the situations in which these models help.

## Learning outcomes {h=309070d7}

The first group of outcomes is conceptual. The reader should be able to define a foundation model, zero-shot forecasting and fine-tuning, and explain how a series becomes the input of a Transformer, through mean scaling, quantisation and patching. The second group concerns the actual models: Chronos, TimesFM, Moirai, Lag-Llama and TimeGPT differ in architecture, in the form of their input, in the form of their output and in their licence, and these differences have practical consequences.

The third group is empirical. A quantile forecast is evaluated with the pinball loss, the CRPS, the MASE of the median and the coverage of the 80% interval. An open model runs on a laptop processor, and its comparison with the seasonal naive forecast, ETS and ARIMA must use the same origins, the same context and the same scores. Finally, the reader should recognise the situations in which a good score may come from contamination rather than forecasting ability.

## Prerequisites for Today {h=c9a1423f}

The chapter uses the forecasting benchmarks of [Chapter 0](../EN/Courses/chapter0_introduction.pdf) and [Chapter 4](../EN/Courses/chapter4_seasonality_forecasting.pdf). The most important is the seasonal naive forecast, $\hat y_{T+h} = y_{T+h-m}$: the forecast for period $T+h$ is the value observed in the same season of the previous cycle. Here $T$ is the last observed period, $h$ the horizon and $m$ the season length, for example 12 for monthly data or 168 for hourly data with a weekly cycle. For series without seasonality, $m = 1$, and the seasonal naive forecast becomes the naive forecast, which repeats the last value.

The same chapters supply the ETS and ARIMA models, rolling-origin cross-validation and the error measures MAE, RMSE and MASE, and [Chapter 9](../EN/Courses/chapter9_machine_learning_time_series.pdf) supplies the split of the data into training, validation and test samples and the idea of a neural network as a flexible function with many parameters, estimated from data.

The last notion is the quantile. The quantile of level $\tau$, written $q_\tau$, is the value below which the variable lies with probability $\tau$, where $\tau \in (0, 1)$. The median is $q_{0.5}$, and the quantiles $q_{0.1}$ and $q_{0.9}$ bound a central interval that contains the variable with probability 80%. Foundation models report their forecasts in exactly this form, as a set of quantiles.

# The foundation-model idea

This section defines three ways of forecasting many series, which differ in the data the model is trained on, and the basic vocabulary of the chapter: the context, zero-shot forecasting and fine-tuning. It then describes pretraining, the expensive stage that the model's developer carries out once.

## Three ways to forecast many series {h=e2fe1fef}

A local model is estimated separately for each series, on its own data only. ETS and ARIMA are local models: they have few parameters, each with an interpretation, for example an autoregressive coefficient or a smoothing constant. The price is that 10,000 series mean 10,000 estimations, and a short series has too few observations to learn a complicated pattern.

A global model, as in [Chapter 9](../EN/Courses/chapter9_machine_learning_time_series.pdf), is a single model trained on all the series of a data set. The classic example is DeepAR of @SalinasEtAl2020, a recurrent neural network trained on the sales history of all the products of a retailer. The series borrow information from one another: a new product benefits from the patterns learnt on old products. The model remains, however, tied to the data set it was trained on.

A foundation model goes one step further: it is pretrained once on a very broad corpus of series from many domains and is then used on new series and new data sets, without re-estimation. The term comes from @BommasaniEtAl2021, who defined it for models trained on broad data at scale and adaptable to many tasks. The three classes therefore differ in their training data: the series itself, the data set of the problem, or a general corpus.

## Zero-shot, fine-tuning, in-context {h=d13d734f}

The context is the recent history $y_{T-C+1}, \dots, y_T$ passed to the model at forecast time, where $C$ is the context length. The context plays the role of the estimation data of a classical model, with one essential difference: the model does not change its parameters because of it, but reads it, as a language model reads a text and continues it.

A zero-shot forecast keeps the pretrained weights fixed and uses only the context of the new series. No parameter is estimated on the target series, so the forecast takes a fraction of a second. Fine-tuning is the alternative: the pretrained weights are trained further on the target data. It can help for unusual series that resemble little in the corpus, but it needs a validation sample, to know when to stop training, and much more computation. The tests in this chapter use zero-shot forecasts only.

The output of these models is usually not a single number but a set of quantiles $\hat q_{0.1}, \dots, \hat q_{0.9}$ for each future step. Such a forecast is called probabilistic: it says not only where the series will probably be but also how uncertain that estimate is. For this reason the chapter needs scores that evaluate whole distributions, not only points.

## Pretraining at scale {h=55889787}

Pretraining corpora mix public series from many domains: energy, transport, retail, weather, web traffic, finance and economics. The variety is deliberate. A model that has seen daily cycles in electricity load, weekly cycles in web traffic and annual cycles in sales can recognise a similar pattern in a new series. For Chronos, @AnsariEtAl2024 also added synthetic series: random mixtures of real series and series generated from Gaussian processes, which cover shapes that public data do not contain often enough.

The scale is far above that of any data set in the previous chapters. TimesFM, for example, was pretrained on about $10^{11}$ data points [@DasEtAl2024]. Pretraining is done once, by the developer, on many graphics processing units (GPUs), and the photograph of a Google data centre is a reminder of the infrastructure it requires.

The user does not repeat this cost. They only run the model, and the small versions run on the central processing unit (CPU) of an ordinary computer. This division of cost is central to the practical value of the models: an analyst with a laptop can obtain in a fraction of a second a probabilistic forecast that would otherwise have required estimating a separate model.

## Recap: The foundation-model idea {h=1bf66685}

Local, global and foundation models differ in the data they are trained on: the series itself, the series of one data set, or a broad corpus from many domains. A zero-shot forecast uses fixed weights and only the context of the new series, and its result is a set of quantiles, that is, a probabilistic forecast.

## @end

A foundation model moves estimation from forecast time to pretraining time. The developer pays the cost once, on a very broad corpus, and the user obtains a probabilistic forecast from a context, with no estimation. What remains to be seen is which network makes this possible and how a series of numbers is turned into an input for it.

# Transformers in brief

Almost all foundation models for time series are built on the Transformer, the network introduced by @VaswaniEtAl2017 for machine translation. This section presents the architecture, its central mechanism, self-attention, and a numerical example, without going into the details of training neural networks.

## The Transformer (Vaswani et al., 2017) {h=092a6056}

@VaswaniEtAl2017 proposed a network for sequences built entirely on attention, without recurrence. The recurrent networks of [Chapter 9](../EN/Courses/chapter9_machine_learning_time_series.pdf) read the sequence step by step and carry their memory in an internal state; the Transformer compares each position directly with all the others. The network was designed for translation and today underlies most large language models (LLMs).

The input is a sequence of tokens, that is, discrete units: in a text, words or pieces of words. Each token is turned into a vector of numbers, called an embedding, which the network learns in training. The original diagram, reproduced on the slide, has two parts. The encoder (left) reads the whole input and turns it into an internal representation. The decoder (right) produces the output step by step, using this representation and what it has already produced. Decoder-only models, such as GPT, TimesFM and Lag-Llama, drop the encoder and forecast the next token from all the previous ones, which fits naturally with forecasting a series.

One technical detail has consequences: attention, as defined below, ignores the order of its inputs. If the positions are permuted, the output is permuted in the same way. For a time series, however, order is essential, so information about its position in the sequence is added to each vector (positional encoding).

## Self-attention (1/2): the formula {h=30ccdda4}

Self-attention produces, for each position of the sequence, a weighted average of the information at all positions. Each input vector $x_i$ produces three vectors through three linear maps: a query $q_i = W_Q x_i$, which describes what position $i$ looks for; a key $k_i = W_K x_i$, which describes what position $i$ offers; and a value $v_i = W_V x_i$, which is the information passed on by position $i$. The matrices $W_Q$, $W_K$ and $W_V$ are the same for every position and are learnt in training.

The output at position $i$ is

$$ \begin{aligned} \text{out}_i &= \sum_j w_{ij} v_j, \\ w_{ij} &= \frac{\exp(q_i^\top k_j/\sqrt{d})}{\sum_l \exp(q_i^\top k_l/\sqrt{d})}, \end{aligned} $$ {#eq:attn}

where the indices $j$ and $l$ run over all input positions and $w_{ij}$ is the weight that position $i$ gives to position $j$. The dot product $q_i^\top k_j$ is a similarity score: it is large when the query and the key point in roughly the same direction. $d$ is the length of the key vectors; without the division by $\sqrt{d}$, the dot products would grow with the dimension of the vectors, and the weights would concentrate almost entirely on a single position.

The function that turns scores into weights is the softmax. The exponential makes every weight positive, and the denominator makes the weights of position $i$ sum to 1. The output is therefore a weighted average of the values, with larger weights for the positions whose keys match the query. An AR($p$) model also takes a weighted average of the past, but with the same weights at every time; in attention, the weights depend on the content of the sequence.

## Self-attention (2/2): matrix form and intuition {h=30c623ca}

The computation in @eq:attn is written compactly in matrix form: $\text{softmax}(QK^\top/\sqrt{d})\,V$. The rows of the matrices $Q$, $K$ and $V$ are the vectors $q_i$, $k_i$ and $v_i$, and the product $QK^\top$ contains all the scores $q_i^\top k_j$ at once. The softmax is applied to each row and gives the weights $w_{ij}$, and multiplication by $V$ forms the weighted averages. The matrix form is the one in which the computation runs efficiently on graphics processors.

For a time series, the intuition is the following. Today's step can "look at" the same hour last week and give it a large weight, if its query matches the key of that hour. Unlike the lags of an AR($p$), fixed in advance by the analyst, these links are learnt from data and can differ for each position and for each series.

The price is cost. Each of the $n$ inputs is compared with all the others, so attention needs $n^2$ comparisons. A context of 2048 values gives $2048^2$, about 4 million comparisons in every attention layer. This quadratic growth is the reason why modern models group values into patches, as the next section shows.

## Worked example: attention weights {h=0a4a7f4d}

The example follows @eq:attn for one query and three keys. The scaled scores $q^\top k_j/\sqrt{d}$ are 2, 1 and 0, and the values attached to the keys are $v_1 = 10$, $v_2 = 20$ and $v_3 = 30$. The first step applies the exponential: $e^2 = 7.39$, $e^1 = 2.72$ and $e^0 = 1$, with sum 11.11.

The second step divides each term by the sum and gives the weights 0.67, 0.24 and 0.09, which sum to 1. The third step takes the weighted average of the values: $0.665 \times 10 + 0.245 \times 20$ $+\, 0.090 \times 30 = 14.2$. The first key, with the largest score, receives two thirds of the weight, and the output is close to its value, 10. The other two values still contribute and pull the output up. Attention is therefore a "soft" selection: it favours the matching positions without ignoring the others.

A foundation model stacks many such layers. Each layer has several attention heads, each with its own matrices $W_Q$, $W_K$ and $W_V$, so that one head can follow the daily cycle and another the recent level.

## Recap: Transformers {h=327eadc5}

Attention makes each step a learnt weighted average of all steps, with weights given by the match between queries and keys. Chronos uses an encoder–decoder architecture, while TimesFM and Lag-Llama are decoder-only. The cost of attention grows with the square of the number of inputs.

## @end

The Transformer replaces the fixed lags of an AR model with weights that depend on the content of the sequence. The numerical example shows that the result is a weighted average dominated by the best-matching positions. The network was, however, built for discrete tokens, and its cost grows with the square of the input length, two problems that a long time series of real numbers makes immediately visible.

# From numbers to tokens

A language model reads tokens from a fixed vocabulary, while a time series contains real numbers in very different units. This section presents the three solutions used in practice and applies them to two Romanian series: the mean scaling and quantisation of Chronos, on industrial production, and patching, on hourly electricity load.

## Representing numerical series as model inputs {h=c6ec5264}

Series differ in units and levels. Romania's electricity load is measured in gigawatts (GW), with hourly values between about 3 and 8; an industrial production index moves around 100; the EUR/RON rate is around 5. A single model used for all these series needs a common scale. Otherwise the same shape, for example a 10% rise, would look completely different in each series.

Practice has produced three solutions. The first, from Chronos, combines scaling with quantisation: each value becomes a token from a fixed vocabulary, exactly like a word. The second, patching, groups consecutive values into one input vector; TimesFM, Moirai and Chronos-Bolt use it. The third writes the digits as text and gives them to a general language model, which reads them as character strings; @GruverEtAl2023 showed that an LLM can produce zero-shot forecasts in this way.

In all three cases the model works on a unit-free representation, and the forecasts are transformed back to the original units at the end. For the user the transformation is invisible; for understanding the results it matters, because it explains what the model can and cannot "see".

## Mean scaling and quantisation (Chronos) (1/2) {h=8bc61745}

The first step of Chronos is mean scaling. Each value of the context is divided by the mean of the absolute values of the context:

$$ \tilde x_t = \frac{x_t}{s}, \qquad s = \frac{1}{C}\sum_{t=1}^{C} |x_t|, $$ {#eq:scale}

where $x_t$ is the $t$-th value of the context, $t = 1, \dots, C$, $C$ is the context length, $s$ is the scale and $\tilde x_t$ the scaled value. After scaling, the context has values of order one, whatever the original units: a series in GW and one in lei look the same if they have the same shape. Unlike ordinary standardisation, mean scaling does not subtract the mean, so it keeps the sign of the values and the ratios between them.

The second step is quantisation: each scaled value is replaced by the number of a bin. Chronos divides the interval $[-15, 15]$ into 4093 equal bins of width about 0.0073, and the token of $\tilde x_t$ is the number of the nearest bin centre. The vocabulary also contains a few special tokens, for example for padding a sequence that is too short and for the end of the sequence. After these two steps, a series is a sequence of integers, exactly like a tokenised text.

## Mean scaling and quantisation (Chronos) (2/2) {h=9511f05d}

On the token sequence, Chronos trains an ordinary language model, T5, to forecast the next token. The loss function is the cross-entropy $-\log \hat p(z_{t+1})$, where $z_{t+1}$ is the actual next token and $\hat p(z_{t+1})$ the probability the model gives it. The loss is 0 when the model gives probability 1 to the correct token and grows as that probability falls. The model thus learns a probability distribution over the 4093 bins, not a single number.

The forecast is made in three steps. First, many future token paths are sampled from the model, each token drawn from the forecast distribution given everything before it. Then each token is replaced by the centre of its bin and multiplied by the scale $s$, which brings the values back to the original units. Finally, at each future step, the quantiles of the sampled values are read off: the median, the 10% quantile, the 90% quantile and so on.

The construction has a clear limit: a scaled value outside the interval $[-15, 15]$ cannot be produced. A series that rose to 20 times the mean level of its context would exceed the model's vocabulary. For ordinary economic series the limit is rarely reached, but it shows that the model forecasts shapes relative to the context, not absolute levels.

## Worked example: scaling and tokens {h=6bcef27d}

The context is Romania's unemployment rate, in percent, seasonally adjusted (Eurostat), for the six months from March to August 2026: 6.5, 6.8, 6.7, 6.8, 6.4, 6.4. The first step computes the scale of @eq:scale: all values are positive, so $s$ is their mean, $39.6/6 = 6.60$.

The second step divides each value by 6.60 and gives 0.985, 1.030, 1.015, 1.030, 0.970, 0.970. The third step finds the nearest bin centre. The centres are numbered from 0 to 4092 and are 0.0073 apart, starting at $-15$; the value 1 therefore corresponds roughly to centre 2182, and the six values give the tokens 2180, 2187, 2184, 2187, 2178 and 2178.

Two observations follow from the example. Equal values give equal tokens, and a change of 0.1 percentage points in the unemployment rate, that is 1.5% of the scale, moves the token by two to three bins: the grid is fine enough for the typical moves of the series. More importantly, the model never sees the 6.6% level; it sees only the shape of the series relative to its scale. For the model, an unemployment rate of 6.6% and one of 13.2% with the same relative shape are identical.

## Scaling and quantisation on real data (1/2) {h=f1e4b842}

The chart applies the two steps to Romania's industrial production, an index with base 2021 = 100, for the last 96 months of the series, from August 2018 to July 2026. The left panel shows the series in its own units, and the dashed red line marks the scale $s$, the mean absolute value of the context, 97.8. One can see the monthly seasonality, with regular drops, and the sudden fall of April 2020, when the lockdown stopped a large part of industry.

The right panel shows the scaled context, $x_t/s$ (blue line), and its tokens (red step line), on a coarse grid of only 21 bins, chosen so that the steps are visible. The centres of the bins in use are marked by dotted horizontal lines at 0.6, 0.9 and 1.2. Each scaled value is replaced by the nearest centre, which is why the red line jumps between the three levels.

## Scaling and quantisation on real data (2/2) {h=78c12879}

After scaling, all 96 values lie between 0.63 and 1.17. They therefore occupy only a few percent of the interval $[-15, 15]$ covered by the Chronos vocabulary; the rest of the interval is reserved for series with large jumps relative to their usual level. On the coarse grid of 21 bins, the 96 months use only three tokens, and the April 2020 fall is the only month that receives a separate token.

The coarse grid loses almost all the seasonal shape: ordinary months are either "0.9" or "1.2". With the 4093 bins of Chronos, the distance between centres is about 40 times smaller than that between 0.9 and 1.2, and the grid is fine enough to keep the seasonal shape. In practice, the information lost through quantisation is much smaller than the forecast error.

The example also shows a weakness of mean scaling. The scale is the mean of all the values of the context, so an extreme value changes it. A very large spike raises $s$ and shrinks all the other scaled values; a fall like the one in 2020 lowers it slightly. Outliers in the context therefore matter for the whole forecast, not only for the period in which they occur.

## Patching {h=717428c7}

A patch is a block of $P$ consecutive values, turned by a small network into a single input vector for the Transformer. The idea was popularised by PatchTST of @NieEtAl2023, whose title, "a time series is worth 64 words", sums up the analogy: a patch plays the role of a word, and the series becomes a short sentence.

Patching has two advantages. The first is length: 2048 values with $P = 16$ give 128 patches, so attention needs $128^2 = 16{,}384$ comparisons instead of $2048^2$, that is, 256 times fewer. The second is content: each input carries a local shape, for example a slope or a peak, rather than a single noisy number. One hourly value on its own says little; 16 consecutive values show whether the series is climbing towards the evening peak or falling towards the night minimum.

TimesFM uses input patches of 32 values, Moirai chooses the patch size according to the frequency of the series, and Chronos-Bolt and Chronos-2 use patches of 16. The output can also be a whole patch: the model forecasts a block of future values at once, which is much faster than generating them one at a time, token by token, as in Chronos.

## Patching the Romanian electricity load {h=1b54ecd7}

The chart shows 336 hourly values of Romania's electricity load, in GW, for the first two weeks of June 2026, with data from ENTSO-E, the European network of transmission system operators. The 336 hours are split into $336/16 = 21$ patches of 16 values each; the patches alternate in colour, blue and red, on grey and pink backgrounds, so that their boundaries are visible.

The series has a clear daily pattern: a night minimum around 4 GW and an evening peak between 6 and 7 GW. Two of the minima, around 7 and 14 June, are deeper and longer: they are the Sundays, when industrial demand falls. Each patch contains a piece of this pattern: some cover only the morning rise, others the evening peak and decline.

## Interpreting patching {h=7e951f65}

A day has 24 hours, so a patch of 16 hours covers two thirds of a day, and the patches do not align with the daily cycle. The same patch starts once at midnight, another time at 16:00 and another at 08:00. The model therefore does not receive "the day" as a unit but fragments of days, shifted from one day to the next.

Attention solves this problem. It links the patches by content, not by fixed position: Monday's morning rise can be connected with last Monday's, even if the two rises fall in different positions of their patches. This is the same intuition as in the example of the same hour last week, applied to blocks.

Attention can only link what is in the context. A weekly cycle has 168 hours, that is $168/16 = 10.5$, so 11 patches. The model can "see" the difference between weekdays and the weekend only if the context covers at least one full week; the section on evaluation confirms this rule empirically.

## Recap: From numbers to tokens {h=08f6712e}

Mean scaling removes the units, and quantisation turns the scaled values into tokens from a fixed vocabulary, as in Chronos. Patching groups consecutive values, which shortens the sequence and gives each input a local shape. The forecast is transformed back to the original units at the end.

## @end

A series becomes the input of a Transformer through a unit-free representation: tokens of scaled values or patches of consecutive values. The representation explains two properties that the evaluation will confirm: the model forecasts the shape relative to the scale, not the level, and it can use only the cycles that the context contains in full.

# The main model families

Between 2023 and 2025 several families of foundation models for time series appeared, built by large technology companies and academic groups. This section describes the most cited ones through four characteristics that matter in practice: architecture, form of the input, form of the output and licence.

## Chronos, Chronos-Bolt and Chronos-2 (Amazon) {h=b451c4fc}

Chronos, published by @AnsariEtAl2024 at Amazon, is the model described in the previous section: a T5 encoder–decoder trained on scaled and quantised values, which samples token paths. Its advantage is conceptual simplicity, since it reuses a language model without changes; its drawback is speed, since every path is generated token by token.

Chronos-Bolt, released in November 2024, replaces the tokens with patches of 16 values and outputs the quantiles of level 0.1, 0.2, …, 0.9 directly, without sampling. The chapter uses two sizes: tiny, with 9 million parameters, and small, with 48 million; both accept a context of up to 2048 values. Chronos-2, described by @AnsariEtAl2025 and released in October 2025, has 119 million parameters, outputs 21 quantile levels and can use covariates, that is, explanatory variables, as well as groups of related series.

All three have open weights under the Apache-2.0 licence, which also allows commercial use, and are installed with the Python package *chronos-forecasting*. All run on a CPU, which makes them suitable for a course: every result in this chapter can be reproduced on a laptop.

## TimesFM (Google) and Lag-Llama {h=b276cb4f}

TimesFM, published by @DasEtAl2024 at Google, is a decoder-only Transformer. The input consists of patches of 32 values and the output of patches of 128 values: the model forecasts a long block at once, which reduces the number of steps needed for a long horizon. It was pretrained on real series, for example Google Trends and Wikipedia page views, and on synthetic series, and it has about 200 million parameters. The weights are open (Apache-2.0); later versions add quantile outputs.

Lag-Llama, by @RasulEtAl2023, is also a decoder-only model, built on the Llama architecture of language models. Its input is different: instead of patches, each step receives the values of the series at lags chosen by frequency, for example 7 days for daily data, 12 months for monthly data or 24 hours for hourly data, plus calendar variables. The construction resembles a seasonal autoregressive model in which the seasonal lags are given explicitly.

The output of Lag-Llama is parametric: the model forecasts the parameters of a Student-$t$ distribution for the next step, and repeated sampling gives paths and, from them, quantiles. The model is small and open (Apache-2.0) and was designed mainly as a starting point for fine-tuning.

## Moirai (Salesforce) and TimeGPT {h=ccba75fa}

Moirai, by @WooEtAl2024 at Salesforce, aims to be a "universal" forecasting model. It can take any number of variables in a single sequence, so it treats a univariate and a multivariate series in the same way, and it chooses the patch size according to the frequency of the data. The output is a mixture distribution, which can represent varied shapes, for example skewed or heavy-tailed distributions. It was pretrained on the LOTSA corpus, with about 27 billion observations. The weights are open but under a non-commercial licence (CC BY-NC 4.0).

TimeGPT, by @GarzaEtAl2023, is a closed model, available only through a paid application programming interface (API). The weights and training data cannot be inspected, the model version can change without notice, and the user's data leave their computer for the provider's servers.

For a course, these three features are decisive. A result obtained today through an API may not be reproducible a year from now, and the question whether the test series was in the training data has no answer. For these reasons TimeGPT is not used in the tests of this chapter.

## The families side by side {h=71c993b9}

The table summarises the four characteristics. In the architecture column, the models split into encoder–decoder (Chronos, Chronos-Bolt), encoder-only (Chronos-2, Moirai) and decoder-only (TimesFM, Lag-Llama). In the input column, Chronos is the only one that uses value tokens; the rest use patches, except Lag-Llama, which uses values at fixed lags. In the output column the forms are more varied: sampled paths, a fixed number of quantiles, patches of values, a mixture of distributions or a Student-$t$ distribution. For TimeGPT, the paper describes the architecture only in broad terms, and the output consists of point forecasts and intervals.

The last column, access, decides what can be checked. Weights under the Apache-2.0 licence can be downloaded, run and inspected by anyone; Moirai's CC BY-NC 4.0 licence forbids commercial use; TimeGPT can be used only through the paid API.

Another direction uses general LLMs that read the digits of the series as text [@GruverEtAl2023]. @TanEtAl2024 tested such methods systematically and showed that when the language-model part is removed or replaced by a simple attention layer, accuracy does not fall: the LLM component adds computational cost, but not accuracy.

## Recap: The model families {h=cd4e33d5}

Chronos works with tokens, while Chronos-Bolt and Chronos-2 work with patches and output quantiles directly. TimesFM and Lag-Llama are decoder-only models, and Moirai accepts any number of variables. Open weights make results reproducible; a paid API does not.

## @end

The model families differ less in idea than in detail: how values are scaled and grouped, what form the forecast takes and who has access to the weights. Two things matter for the tests of the chapter: all the models used have open weights and run on a CPU, and all produce quantiles, which calls for scores suited to probabilistic forecasts.

# Probabilistic forecasts and their scores

A point forecast is evaluated by the distance between the forecast and the observed value. A probabilistic forecast is a distribution, while the observed value is a single number, so scores are needed that compare these two objects. This section defines the pinball loss for one quantile, the CRPS for a whole distribution, the WQL as a practical version of the CRPS, the MASE for the median and coverage for the calibration of intervals.

## Quantile forecasts {h=53e161f5}

A probabilistic forecast gives the distribution of $y_{T+h}$, conditional on the data available up to time $T$. Foundation models report this distribution through quantiles $\hat q_\tau$ at the levels $\tau = 0.1, 0.2, \dots, 0.9$. From them one reads directly the point forecast, the median $\hat q_{0.5}$, and the central 80% interval, $[\hat q_{0.1}, \hat q_{0.9}]$. A fan chart draws several such bands, for example the 50% and 80% bands, one over the other.

A good probabilistic forecast has two qualities, following @GneitingRaftery2007. Calibration requires the stated probabilities to match the observed frequencies: an 80% interval should contain about 80% of the observed values. Sharpness requires the distribution to be concentrated: of two equally well calibrated forecasts, the narrower is more informative. The principle of @GneitingRaftery2007 is to maximise sharpness subject to calibration.

The classical benchmarks do not produce quantiles directly, but we obtain them from the Normal distribution. For ETS and ARIMA, $\hat q_\tau = \hat y_{T+h} + z_\tau \hat\sigma_h$, where $\hat y_{T+h}$ is the point forecast, $z_\tau$ the quantile of level $\tau$ of the standard Normal distribution (for example $z_{0.9} = 1.28$), and $\hat\sigma_h$ the estimated standard deviation of the forecast error at horizon $h$. The comparison between foundation models and benchmarks is thus made on the same kind of output.

## The pinball loss {h=b604dde3}

The pinball loss, also called the quantile loss, evaluates a single forecast quantile $\hat q_\tau$ against the value $y$ observed later:

$$ \begin{aligned} &L_\tau(y, \hat q_\tau) = \\ &\quad \begin{cases} \tau\,(y - \hat q_\tau), & y \ge \hat q_\tau, \\ (1 - \tau)(\hat q_\tau - y), & y < \hat q_\tau. \end{cases} \end{aligned} $$ {#eq:pinball}

The loss is 0 when the quantile coincides with the observed value and grows linearly with the distance between them. The asymmetry is essential, however: an observed value above the quantile receives weight $\tau$, and one below the quantile receives weight $1 - \tau$. For $\tau = 0.9$, missing on the upside costs 0.9 per unit of distance, and missing on the downside only 0.1.

The property that justifies this function is that its expected value is minimised exactly at the true quantile of level $\tau$. A forecaster who knows the true distribution has no reason to report anything other than its quantile: any shift up or down raises the mean loss. A score with this property is called strictly proper, and it prevents "gaming" through deliberately wide or narrow intervals.

## The CRPS {h=18828a7e}

The continuous ranked probability score (CRPS), introduced by @MathesonWinkler1976, evaluates a whole forecast distribution against the observed value $y$:

$$ \begin{aligned} &\text{CRPS}(F, y) \\ &\quad = \int \big(F(z) - \mathbf{1}\{z \ge y\}\big)^2\,dz \\ &\quad = 2\int_0^1 L_\tau\big(y, F^{-1}(\tau)\big)\,d\tau. \end{aligned} $$ {#eq:crps}

Here $F$ is the forecast distribution function and $z$ the integration variable, which runs over all possible values. The indicator $\mathbf{1}\{z \ge y\}$ is 1 if $z \ge y$ and 0 otherwise; it is the distribution function of a forecast that puts all its mass on the observed value $y$. The first form therefore measures the squared distance between the forecast distribution function and the step at $y$. The second form, in which $F^{-1}(\tau)$ is the quantile of level $\tau$ of $F$, shows that the CRPS is twice the average pinball loss over all levels.

The CRPS is greater than or equal to 0 and is expressed in the units of $y$. It is 0 only for a forecast that puts all its mass exactly on the observed value, and a smaller value means a better forecast. For a point forecast, $F$ is itself a step at $\hat y$, and the CRPS becomes $|y - \hat y|$: the CRPS generalises the absolute error and can be compared directly with the MAE of a point forecast.

## The weighted quantile loss (WQL) {h=85758c0c}

The integral in @eq:crps requires the whole distribution, but foundation models report only a few quantiles. The weighted quantile loss (WQL) approximates the CRPS with the 9 levels 0.1, …, 0.9 and divides the result by the size of the series:

$$ \text{WQL} = \frac{1}{9}\sum_{\tau}\frac{2\sum_t L_\tau(y_t, \hat q_{\tau,t})}{\sum_t |y_t|}. $$ {#eq:wql}

The index $\tau$ runs over the 9 levels, and $t$ runs over all forecast steps from all origins of the test. $\hat q_{\tau,t}$ is the quantile of level $\tau$ forecast for step $t$, and $y_t$ the value observed at that step. The numerator is, for each level, twice the sum of the pinball losses, as in the second form of the CRPS.

The denominator $\sum_t |y_t|$ removes the units: the WQL is a loss relative to the level of the series, so series with very different levels can be compared, for example load in GW and the exchange rate in lei. The WQL is 0 for a perfect forecast, and a smaller value means a better forecast. It is the score reported by Chronos and by the GIFT-Eval benchmark, which is why it is also the main score of the test in this chapter.

## The pinball loss and the CRPS {h=e89eae2a}

The left panel plots the pinball loss @eq:pinball against the difference $y - \hat q$ between the observed value and the forecast quantile, for three levels. The dark blue line, for $\tau = 0.5$, is symmetric: it has slope 0.5 on both sides. The red line, for $\tau = 0.9$, rises steeply to the right, with slope 0.9, and very slowly to the left, with slope 0.1; the turquoise line, for $\tau = 0.1$, is its mirror image. The three lines meet at 0, where the quantile is exact.

The right panel illustrates the first form of @eq:crps. The forecast is the Normal distribution with mean 0 and variance 1, written N(0, 1), and its distribution function is the blue curve. The observed value is $y = 1$, and the red step is the distribution function of a perfect forecast. The shaded area is the gap between the two curves; the CRPS is the integral of the square of this gap and equals 0.602 here. The notation N($\mu$, $\sigma^2$) denotes in general the Normal distribution with mean $\mu$ and variance $\sigma^2$.

## Interpreting the scores {h=2598f3fd}

The asymmetry of the pinball loss explains why the score demands the correct quantile. At $\tau = 0.9$, an observed value above the quantile costs, per unit of distance, 9 times more than one below it ($0.9/0.1 = 9$). A forecaster minimises the mean loss only if exceedances occur in 10% of cases, exactly the definition of the 0.9 quantile. At $\tau = 0.5$, the two weights are equal and the loss is half the absolute error: the median is evaluated as a point forecast.

Three values of the CRPS show what the score rewards. For the N(0, 1) forecast, the CRPS is 0.234 if the observed value is $y = 0$, the centre of the distribution, and 0.602 if $y = 1$, one standard deviation away. The N(0, 4) forecast, with standard deviation 2, gives a CRPS of 0.663 at $y = 1$, larger than 0.602, even though its band covers the observed value more comfortably.

The comparison of the last two numbers shows that a wider band is penalised even when it covers the observed value. The CRPS therefore rewards calibration and sharpness at the same time: a good score cannot be obtained either with very wide intervals, which cover everything, or with very narrow ones, which often miss.

## Worked example: pinball losses {h=59be8433}

Consider a forecast of next month's unemployment rate with three quantiles: $\hat q_{0.1} = 6.5$, $\hat q_{0.5} = 7.0$ and $\hat q_{0.9} = 7.6$, in percent. The value observed later is $y = 7.2$. Each loss is computed with @eq:pinball, according to the position of $y$ relative to the quantile.

At $\tau = 0.1$, the observed value is above the quantile, so the loss is $0.1 \times (7.2 - 6.5) = 0.07$. At $\tau = 0.5$, the observed value is also above, and the loss is $0.5 \times (7.2 - 7.0) = 0.10$. At $\tau = 0.9$, the observed value is below, so the weight $1 - \tau$ applies: $(1 - 0.9) \times (7.6 - 7.2) = 0.04$. The largest loss falls on the median, which is the furthest from the observed value relative to its weight.

The CRPS approximated with these three levels is twice the mean loss, $\frac{2}{3}(0.07 + 0.10 + 0.04) = 0.140$ percentage points. The observed value, 7.2, lies in the 80% interval [6.5, 7.6], so in the coverage score defined two slides later this forecast counts as covered.

## Point accuracy: the MASE {h=89bd06ed}

The mean absolute scaled error (MASE), proposed by @HyndmanKoehler2006, evaluates the point forecast, here the median. It divides the mean absolute error (MAE) of the forecast by the in-sample MAE of the seasonal naive forecast:

$$ \text{MASE} = \frac{\frac1h\sum_{j=1}^{h}|y_{T+j} - \hat y_{T+j}|}{\frac{1}{T-m}\sum_{t=m+1}^{T}|y_t - y_{t-m}|}. $$ {#eq:mase}

The numerator is the mean absolute error of the forecasts $\hat y_{T+j}$ over the $h$ steps after the origin $T$. The denominator is the mean absolute error of the seasonal naive forecast $y_{t-m}$ inside the sample, for $t = m+1, \dots, T$, where $m$ is the season length. The denominator therefore measures how hard the series is to forecast with the simplest method, computed on the data available at the origin.

The MASE has no units. A value below 1 means the forecast is better than the in-sample seasonal naive forecast, and a value above 1 that it is worse. For example, a forecast with MAE 0.30 for a series on which the seasonal naive forecast has an in-sample MAE of 0.25 has $\text{MASE} = 0.30/0.25 = 1.2$. Because the denominator is computed in sample, with a lag of one season, and the numerator out of sample, over $h$ steps, even the seasonal naive forecast can have a MASE different from 1 in the test.

## Coverage and relative scores {h=f9f5ae34}

Coverage checks the calibration of the intervals directly. It is the share of observed values that fall in the 80% interval:

$$ \text{Cov} = \frac1n\sum_{i=1}^{n} \mathbf{1}\big\{\hat q_{0.1}^{(i)} \le y_i \le \hat q_{0.9}^{(i)}\big\}, $$ {#eq:cov}

where $n$ is the number of forecasts, $\hat q_{\tau}^{(i)}$ the quantile of forecast $i$, and the indicator is 1 if the observed value is in the interval and 0 otherwise. The target is 80%. If 7 of 10 observed values fall in the interval, coverage is 70%, and the intervals are slightly too narrow; a coverage of 95% would indicate intervals that are too wide.

Relative scores make series comparable with one another. The MASE or WQL of a model is divided by the score of the seasonal naive forecast computed on the same origins; a value below 1 means the model is better than the seasonal naive forecast. The ratios of the series are averaged with the geometric mean $(r_1 r_2 \cdots r_S)^{1/S}$, where $r_k$ is the ratio of series $k$ and $S$ the number of series. The geometric mean treats relative gains and losses symmetrically: a ratio of 2 on one series and 0.5 on another cancel exactly and give 1, whereas the arithmetic mean would give 1.25.

## Recap: Scores {h=342d2401}

The pinball loss evaluates each quantile, the CRPS the whole distribution, and the WQL is the approximation of the CRPS on nine quantiles, scaled by the level of the series. The MASE evaluates the median, and coverage checks the calibration of the intervals. Scores are reported relative to the seasonal naive forecast and averaged geometrically across series.

## @end

The scores of this section form a coherent system. The pinball loss is strictly proper for one quantile, the CRPS integrates it over all levels and rewards calibration and sharpness together, and the WQL makes it comparable across series. The MASE and coverage complete the picture: the first says how good the central forecast is, the second whether the intervals tell the truth about uncertainty.

# Zero-shot forecasts on real data

This section applies a foundation model, Chronos-Bolt small, to series it was not trained on, with no estimation. It first presents the seven series of the test, then three forecasts from a single origin each, chosen to illustrate three different situations: a series with strong seasonality, a monthly series with shocks and a series close to a random walk.

## Seven series for the test (1/3) {h=3796a6ce}

The chart, split over two slides, shows the series of the test. The first panel is Romania's hourly electricity load in June 2026: a regular daily cycle, with the minimum at night and the peak in the evening, and a weekly cycle, with visibly lower Sundays. The second panel is Romania's industrial production since 2005: a rise from about 55 to about 100 by 2017, then a plateau, strong monthly seasonality and the fall of April 2020.

The third panel is Romania's quarterly real GDP, not seasonally adjusted, with a "saw-tooth" pattern: the value rises from the first to the fourth quarter and drops sharply in the first quarter of the next year, while the level grows over time. The fourth panel is monthly HICP inflation, in percent on the previous month: a noisy series around a small mean, with a few extreme values and an episode of high values in 2022.

## Seven series for the test (2/3) {h=ac47103e}

The second part of the chart shows the other three series. Romania's unemployment rate moves slowly, in waves of several years: it rises after the 2008 crisis, falls until 2019, rises slightly in 2020 and then stays around 5–7%. The NBR EUR/RON reference rate rises in steps from less than 3.5 lei in 2007 to above 5.2 lei in 2026, with long periods of stability.

The last series, US retail sales, has an upward trend, a pronounced annual seasonality with the peak in December, and two breaks: the fall in spring 2020 and the jump of 2021. It is the only series from outside Romania and a very well-known public series, probably present in many pretraining corpora, which will matter in the section on contamination.

## Seven series for the test (3/3) {h=185d56b5}

The sources of the series are public. Hourly load comes from ENTSO-E; industrial production, non-seasonally adjusted real GDP, HICP (harmonised index of consumer prices) inflation and the unemployment rate come from Eurostat; EUR/RON is the reference rate of the National Bank of Romania (NBR); US retail sales are the series RSXFSN from the FRED database of the Federal Reserve Bank of St. Louis.

The series were chosen to cover different behaviours. Load, industrial production, US retail sales and GDP have strong seasonality: daily and weekly cycles for load, annual for production and sales, quarterly for GDP. EUR/RON and the unemployment rate are close to a random walk, so the best forecast is close to the last value. Monthly inflation is noisy and has short memory. Several series have breaks: the 2020 pandemic for production, GDP and sales, the inflation surge of 2022.

The series also differ greatly in length: from 126 quarters for GDP to 39,408 hours for load. This means very different amounts of context. A model with a context of 2048 values sees 85 days of the load series but the entire history of GDP, which is nonetheless very short for learning a pattern.

## Romanian electricity load, 48 hours ahead (1/2) {h=2ff43c9b}

The chart shows a zero-shot forecast of electricity load for 48 hours from the origin 22 June 2026, 00:00. The solid black line is the last week of the context, and the dashed black line is the actual load over the two forecast days. The blue line is the Chronos-Bolt small median, and the blue bands are the central 50% (darker) and 80% (lighter) intervals. The brown line is the seasonal naive forecast, which repeats the load of the same hours in the previous week.

The model follows the daily shape: the night minimum, the morning rise and the evening peak, at the right hours. Its median is close to the seasonal naive forecast, but slightly higher. The actual load, the dashed line, is above both forecasts almost all the time, however, especially on the second day, when it rises above 7 GW.

## Romanian electricity load, 48 hours ahead (2/2) {h=5a3a4ec7}

The forecast used a context of 2048 hours, that is 85 days, and no estimation on this series. That the model reproduces the daily shape, with the night minimum and the evening peak, shows that it recognised the pattern in the context: exactly the ability that pretraining on energy and traffic series was meant to provide. The MASE of the median is 1.31, against 1.61 for the seasonal naive forecast, so the model is better than simply repeating last week.

The intervals tell another story. The actual load stays above the forecast all day, and only 29% of the 48 hours fall in the 80% band, instead of about 80%. The explanation is the information the model lacks. It receives only the past values of the load, with no temperature or calendar, and a change in the weather, for example a heat wave that raises air-conditioning demand, is not announced by the last 85 days.

A single origin proves little, however. A good or poor result on one date can be a coincidence, so the next section repeats the forecast at 52 weekly origins and compares it with several benchmarks.

## Worked example: scoring one hour {h=33763fce}

The example scores hour 24 of the forecast, that is 22 June 2026, 23:00. The quantiles forecast by Chronos-Bolt small are $\hat q_{0.1} = 5.1837$, $\hat q_{0.5} = 5.4763$ and $\hat q_{0.9} = 5.7758$ GW, and the observed load is 5.8033 GW. All three quantiles are below the observed value, so each loss in @eq:pinball has the form $\tau\,(y - \hat q_\tau)$.

The losses are $0.1 \times (5.8033 - 5.1837) = 0.062$ GW for $\tau = 0.1$, $0.5 \times (5.8033 - 5.4763) = 0.163$ GW for $\tau = 0.5$ and $0.9 \times (5.8033 - 5.7758) = 0.025$ GW for $\tau = 0.9$. The quantiles are given to four decimals because the last difference is small: rounded to two decimals, it would become $0.9 \times 0.02 = 0.018$. The largest loss again falls on the median. The 0.9 quantile has a small loss because the observed value exceeds it by very little.

The observed value is nevertheless above $\hat q_{0.9}$, so the coverage indicator of @eq:cov is 0 for this hour. The example shows how a model can have small pinball losses and still poor coverage: the observed value is close to the edge of the band, but outside it. Averaged over many hours and origins, these losses give the WQL of the next section.

## Romanian industrial production, 12 months ahead (1/2) {h=4487a73f}

The second example is a monthly series. The chart shows the last four years of the context of industrial production (black line), then the 12-month forecasts from the origin of August 2025 and the actual values (dashed black line with dots). Two models are compared: Chronos-Bolt small, zero-shot, with a blue median and a blue 80% band, and ETS, estimated on the series, with a green median and a green 80% band.

Both medians follow the seasonal pattern of the series. The actual values are, however, below both forecasts in most months: the first forecast month, August 2025, and January 2026 are far below the medians, and the last months of 2026 remain 5–6 points below them.

## Romanian industrial production, 12 months ahead (2/2) {h=3e473629}

Both models copy the seasonal pattern, with the drops in August and December, and keep the recent level of the series. That a zero-shot model and a model estimated on the series produce such similar forecasts shows that both extract the same information from the past: the seasonality and the level.

The scores at this origin, however, favour the simplest method. The MASE is 1.04 for Chronos-Bolt, 1.20 for ETS and only 0.67 for the seasonal naive forecast. The weak production of 2026 is not announced by the context, and the seasonal naive forecast, which repeats the lower values of the previous year, happened to land closer. The 80% bands cover 83% of the months for both models, that is 10 of 12, a coverage close to the target.

The difference between the models lies in the uncertainty. The Chronos-Bolt bands are narrower than the ETS bands at long horizons: the ETS band widens steadily with the horizon, while the pretrained model estimates a more stable uncertainty. With the same coverage, the narrower band is more informative, so the CRPS will favour it.

## EUR/RON, 20 days ahead (1/2) {h=0e8732d3}

The third example is the NBR EUR/RON reference rate. The chart shows the last 120 days of the context, with a sudden rise of the rate in early May 2026 from about 5.10 to above 5.20 lei, followed by a period of stability. The forecasts for the 20 business days after the origin of 24 August 2026 are drawn on the right.

Chronos-Bolt small (blue median, blue 80% band) is compared with the random walk, whose forecast is the last value, with Normal bands (brown median, brown 80% band). The two medians are almost flat and almost on top of each other, and the bands have almost the same shape, widening with the horizon like a funnel.

## EUR/RON, 20 days ahead (2/2) {h=d953686a}

The last value before the origin is 5.2581 lei, and the Chronos-Bolt median after 20 days is 5.2604 lei, practically the same. The model was not trained explicitly to treat exchange rates as a random walk; it learnt this behaviour from the context, because the series contains no exploitable regularity. The width of the 80% band at 20 days is 0.064 lei for Chronos-Bolt and 0.063 lei for the random walk: the two models assign almost the same uncertainty.

Point accuracy is also almost equal. The MAE over the 20 days is 0.0032 lei for Chronos-Bolt and 0.0035 lei for the random walk, a difference too small to mean anything at a single origin. The result confirms the conclusion of [Chapter 1](../EN/Courses/chapter1_stochastic_processes_stationarity.pdf): exchange-rate changes are almost unpredictable from the past, and a model, however large, cannot extract from the context information that the context does not contain.

## Recap: Zero-shot forecasts {h=ec0a66bc}

With no estimation, the model copies the cycles and levels in the context. It cannot anticipate what the context does not show, for example a heat wave or a recession. For a series close to a random walk, its forecast practically coincides with that of the random walk.

## @end

The three examples show what a zero-shot model does: it recognises the patterns in the context and extends them, with plausible uncertainty. Electricity load shows the strength of this approach for a series with regular cycles, industrial production shows its limit in the face of an unannounced shock, and EUR/RON shows that when the past contains no information, the model correctly reduces to the random walk. A single origin each, however, does not support general conclusions.

# Rolling-origin evaluation on seven series

A comparison between models is fair only if all of them receive the same information and are evaluated on the same forecasts. This section describes the test protocol, presents the results on the seven series, analyses coverage, the error by horizon, the context length needed and the relation between model size and accuracy, and then places the results in the context of published benchmarks.

## The test protocol {h=7429ade7}

The test uses the rolling-origin cross-validation of [Chapter 4](../EN/Courses/chapter4_seasonality_forecasting.pdf). At each origin, each model forecasts $h$ steps using only the data available up to that origin; then the origin moves forward and the procedure is repeated. For load, the horizon is 48 hours, with weekly origins from July 2025 to June 2026. For the monthly series, the horizon is 12 months, with monthly origins from 2016. For GDP, the horizon is 4 quarters, with quarterly origins from 2012, and for EUR/RON 20 days, with an origin every 20 days from 2016. In total there are 708 origins. The context is the whole available past, but at most 2048 observations.

The benchmarks are re-estimated at each origin. The first is the seasonal naive forecast; for unemployment and EUR/RON, which have no seasonality, it is the naive forecast. The second is an ETS model with a damped trend and additive seasonality. The third is an ARIMA whose orders were chosen once, by the Akaike information criterion (AIC), before the first origin, and then kept; for load, the model is an ARIMA(2,0,2) on the weekly differences of the series. Choosing the orders before the first origin prevents any use, even indirect, of the test data.

The foundation models are Chronos-Bolt tiny, Chronos-Bolt small and Chronos-2, all zero-shot. All models receive the same origins, the same context and the same horizon, and are evaluated with the same scores: the MASE, the WQL and the coverage of the 80% interval. Only under these conditions can a difference in score be attributed to the model rather than to the protocol.

## Results: WQL relative to the seasonal naive {h=43fe878c}

The table shows, for each series and each model, the ratio of the model's WQL to the WQL of the seasonal naive forecast on the same origins. A value below 1 means the model is better than the seasonal naive forecast; for example, 0.58 means a 42% smaller loss. The "Origins" column gives the number of origins of each series, from 52 for load to 134 for EUR/RON. The last row is the geometric mean of the ratios over the seven series.

Hourly load shows the widest spread: ETS has a ratio of 1.75, so it is much worse than the seasonal naive forecast, ARIMA has 0.96, and the three foundation models have between 0.58 and 0.69. For real GDP the ranking reverses: ARIMA has 0.72 and ETS 0.82, while Chronos-Bolt tiny has 1.48 and Bolt small 1.22. For US retail sales, Chronos-2 has 0.60, but ARIMA (0.66) and ETS (0.69) are better than the two Bolt models (0.84 and 0.90). For unemployment, all ratios lie between 0.96 and 1.03.

On the geometric mean, Chronos-2 has 0.78, a loss 22% smaller than the seasonal naive forecast, followed by ARIMA with 0.87, Bolt small with 0.90, Bolt tiny with 0.95 and ETS with 0.97.

## MASE and WQL by series and model {h=0a0970a9}

The chart shows the same ratios for the MASE (top) and the WQL (bottom), with one group of bars for each series and one bar for each model. The vertical axis is logarithmic, so a ratio of 2 and one of 0.5 are the same distance from the dotted line, which marks the value 1, the level of the seasonal naive forecast. Bars below the line indicate models better than the benchmark.

The two panels look alike, because a good median also contributes to a good distribution. The clearest difference appears for EUR/RON. On the MASE, all three foundation models are above 1, with ratios of 1.05 for Chronos-2, 1.07 for Bolt small and 1.17 for Bolt tiny: their median is slightly worse than the naive forecast. On the WQL, the same models are below 1, with 0.87 for Chronos-2. The gain does not come from the central forecast but from the shape of the distribution: the bands of the naive forecast cover 94% of the values, far above the 80% target, so they are too wide, and the narrower bands of the pretrained models are rewarded by the WQL.

## Interpreting the benchmark {h=8d73abc7}

Hourly load is the series on which foundation models win clearly: the WQL ratio is 0.58 for Chronos-2 and 0.66 for Bolt small, against 0.96 for ARIMA. ETS fails, with 1.75, because its specification has a single season, the daily one, and cannot represent the difference between weekdays and the weekend. The series has exactly the properties that pretraining favours: regular cycles, many observations and a long context.

Romanian GDP is the opposite. The series is quarterly and short, and ARIMA is the best model, with 0.72; Chronos-Bolt small, with 1.22, is worse even than the seasonal naive forecast. For unemployment, all models are close to the naive forecast. For EUR/RON, the ratios are also close to 1, except Chronos-2, with a WQL ratio of 0.87; its MASE ratio, 1.05, shows that this gain comes, as the previous chart shows, from narrower bands, not from a better median. On these series nothing beats the naive forecast by much on the central forecast.

Overall, the geometric mean of the WQL ranks the models as follows: Chronos-2 0.78, ARIMA 0.87, Bolt small 0.90, Bolt tiny 0.95 and ETS 0.97. Chronos-2 is best on 5 of the 7 series. A well-specified ARIMA, however, is as good as the small foundation models, slightly better on average, and there is no single winner across all series. The practical conclusion is to compare models on our own series rather than choose one by reputation.

## Coverage of the 80% intervals {h=6770dff8}

The chart shows the coverage of the interval $[\hat q_{0.1}, \hat q_{0.9}]$, computed with @eq:cov over all origins and all forecast steps, for each series and each model. The dotted line marks the target of 0.8. A bar above the line indicates intervals that are too wide, and one below the line intervals that are too narrow.

Several patterns are immediately visible. For load, ETS covers 97% of the values, so its intervals are much too wide. For EUR/RON, the classical benchmarks cover about 94%, and Chronos-2 less than 80%. For US retail sales, all models are well below the target. For industrial production and GDP, most bars are close to 0.8.

## Interpreting the coverage {h=5b02edc2}

The average coverage over the seven series is 83% for the seasonal naive forecast, 84% for ETS, 84% for ARIMA, 80% for Bolt small and 76% for Chronos-2. Chronos-2 has the best WQL but intervals that are too narrow: it is sharp, but not fully calibrated. The WQL favours it because its narrow bands bring large gains in sharpness, while the penalty for the additional exceedances is smaller. For a user who needs reliable intervals, for example to size a capacity reserve, this lack of calibration matters.

For US retail sales, every model covers well below 80%: from 64.7% for the seasonal naive forecast and 65.0% for Chronos-2 to 75.2% for Bolt tiny. The 2020 shock and the 2021 jump surprise all of them; no model that learns only from the past of the series could have anticipated the shutdown of the economy. Poor coverage common to all models points to a test period with exceptional events, not to a flaw of one particular model.

The ETS and ARIMA bands assume Normal errors and a constant variance. They are therefore too wide in calm periods, as for EUR/RON, where the estimated variance includes the turbulent episodes of the past, and too narrow in crises, when errors suddenly become larger. This is the same problem that the GARCH models of [Chapter 5](../EN/Courses/chapter5_conditional_volatility_garch.pdf) solve for returns, through a conditional variance.

## Romanian load: error by horizon {h=0af5f771}

The chart shows, for Romanian load, the mean absolute error of the median forecast, in GW, for each of the 48 hours of the horizon, averaged over the 52 weekly origins. The horizontal axis is the number of hours after the origin; each line is a model.

All curves have two humps, around hours 13 and 37. The origin is at midnight, so the humps correspond to midday on each of the two days, when load is high and variable, and the valleys correspond to the nights, when load is low and predictable. The ETS curve (green) is highest almost everywhere, the curves of the seasonal naive forecast (brown) and ARIMA (purple) overlap after the first hours, and the three foundation models form a lower group, with Chronos-2 (red) the lowest.

## Interpreting the error by horizon {h=64f09b4a}

In the first hour, ARIMA and Chronos-2 have an error of 0.05 GW, well below the seasonal naive forecast, at 0.17 GW. At very short horizons the last observation contains the most information, and both models use it; the seasonal naive forecast ignores it, because it repeats last week.

After one day, ARIMA returns to the level of the naive forecast: 0.24 GW against 0.25 GW at hour 24. The ARIMA model on weekly differences forecasts, in essence, last week plus a correction given by the recent deviation, and this correction dies out quickly. Chronos-2 keeps its advantage over the whole horizon, with 0.14 GW at hour 48, against 0.20 GW for the naive forecast and for ARIMA: it combines the recent information with the shape of the daily and weekly cycles in the context.

ETS is good for a few hours, while the recent level dominates, and then becomes the worst model. A daily season cannot represent the weekend, so on a Sunday ETS forecasts the load of a working day. The chart thus explains the WQL ratio of 1.75 in the results table.

## The required context length (1/2) {h=88a3f006}

The chart shows the MASE for Romanian load as a function of the context length given to the model: 96, 192, 336, 672, 1344 and 2048 hours, that is from 4 to 85 days. The test uses the last 26 weekly origins, that is the last half-year. Each line with dots is a foundation model, and the dashed horizontal line is the seasonal naive forecast, which does not depend on the context.

All three curves fall as the context grows, steeply at first and slowly at the end. Chronos-2 starts highest and ends lowest.

## The required context length (2/2) {h=a1305d52}

With a context of 96 hours, that is 4 days, every model is worse than the seasonal naive forecast, which has a MASE of 1.16: Chronos-Bolt small has 1.33 and Chronos-2 has 2.04. Four days do not contain a full week, so the model cannot know what the weekend looks like; the seasonal naive forecast, which repeats last week from the full history of the series, has this information.

From 192 hours, that is 8 days, the weekly cycle enters the context in full, and the error falls below that of the naive forecast for Bolt small, with a MASE of 0.98. This is the empirical confirmation of the rule from the section on patching: attention can only link what is in the context. Chronos-2 keeps improving up to 1344 hours, where it reaches a MASE of 0.63, while Chronos-Bolt levels off from 672 hours, at about 0.78.

The practical rule is to give the model several complete seasonal cycles. A zero-shot model repeats the patterns it sees in the context; if a cycle is missing from the context, it is missing from the forecast. For a monthly series with annual seasonality, this means several years of data, not a few months.

## Size and accuracy {h=cd329f96}

The chart puts on the horizontal axis the number of parameters of each foundation model, in millions, on a logarithmic scale, and on the vertical axis the geometric mean over the seven series of the WQL ratio to the seasonal naive forecast. The three points are Chronos-Bolt tiny, Chronos-Bolt small and Chronos-2. The horizontal lines mark the benchmarks, which have only a few parameters per series: the seasonal naive forecast at 1, ETS at 0.97 and ARIMA at 0.87.

The points fall from left to right: 0.95 for Bolt tiny, 0.90 for Bolt small and 0.78 for Chronos-2. Only Chronos-2 lies below the ARIMA line.

## Interpreting size and cost {h=e748909d}

The three models have 9 million (Bolt tiny), 48 million (Bolt small) and 119 million parameters (Chronos-2); ETS and ARIMA have a few parameters per series. The larger models score better here, but ARIMA, at 0.87, sits between Bolt small and Chronos-2. Three points do not establish a law, and size alone is not the cause: Chronos-2 differs from Chronos-Bolt not only in size but also in architecture and training data, so part of its gain may come from these differences.

The cost of use is small. The time for one load forecast on a laptop processor is 19 milliseconds for Bolt small, 43 ms for Chronos-2, 114 ms for ARIMA and 411 ms for ETS. The pretrained models are faster than the classical benchmarks because they estimate nothing: a single pass through the network replaces the optimisation of the likelihood.

Zero-shot inference is cheap because the real cost was paid once, at pretraining, by the developer. For a user with thousands of series this changes the calculation: a strong benchmark is obtained in seconds for all the series, before any dedicated model.

## Reference benchmarks {h=0123e7e7}

The test of this chapter has only seven series. General conclusions about forecasting methods rest on large benchmarks. The M3 competition, described by @MakridakisHibon2000, used 3003 series and showed that sophisticated methods do not, on average, beat simple ones. The M4 competition [@MakridakisEtAl2020] used 100,000 series; it was won by forecast combinations and a hybrid of exponential smoothing and a neural network. The Monash archive [@GodahewaEtAl2021] collects public data sets with fixed horizons and reference results.

For foundation models, the reference benchmark is GIFT-Eval, by @AksuEtAl2024. It covers many domains and frequencies, reports the MASE and the CRPS and publishes a leaderboard of models. Importantly for the next section, GIFT-Eval separates its test data from a pretraining corpus released separately, to limit contamination.

The literature also has critical studies. @ZengEtAl2023 showed that very simple linear models beat the first Transformer-based forecasting models on the long-term forecasting benchmarks used at the time. @TanEtAl2024 showed that, in methods that adapt LLMs to time series, the language-model component adds no accuracy. Both studies reinforce the lesson of this chapter's test: a complex model must be compared with simple benchmarks before it is preferred.

## Advantages and limitations of foundation models {h=193832d8}

Foundation models help first of all when there are many series and little time for each. They work best on series with strong, regular seasonality and long contexts, as with electricity load, sales or traffic. They are a fast and strong benchmark before a dedicated model and give a probabilistic forecast at no extra cost.

They do not help on series close to a random walk, such as exchange rates or returns: there is nothing to learn from the past of these series. Nor do they help on short series of a few dozen quarters, where a carefully specified ARIMA is better, as Romanian GDP shows. They are also limited when explanatory factors such as temperature, prices or policies matter and are not in the context; models such as Chronos-2, which accept covariates, can reduce this limitation. Finally, they are not suitable when the model must be explained parameter by parameter or when a causal question is asked, for example the effect of a tax change.

The rule that follows from the whole chapter is one of method: always compare a foundation model with the seasonal naive forecast, ETS and ARIMA on the same origins, and test the differences with the test of @DieboldMariano1995 from [Chapter 4](../EN/Courses/chapter4_seasonality_forecasting.pdf), to separate a real gain from noise.

## Recap: The evaluation {h=61379432}

Chronos-2 is the best on average, especially for hourly load, but ARIMA is close and wins on GDP. On the series close to a random walk, no model beats the naive forecast on the central forecast. The intervals of foundation models tend to be too narrow, and the context must cover complete seasonal cycles.

## @end

The test on 708 origins gives a more nuanced picture than any single example. The largest foundation model reduces the average WQL by 22% relative to the seasonal naive forecast, but a well-specified ARIMA achieves 13%, and on short series or series close to a random walk the advantage disappears. The results depend on the context, and the intervals of the pretrained models are narrower than they should be. One question remains: are these scores really forecasts, or does part of them come from data the model has already seen?

# Leakage and contamination

A test score is credible only if the forecast did not use information from the test period. This section distinguishes two sources of the problem. Leakage arises in the analyst's code and can be controlled. Contamination arises in the model's pretraining corpus and can only be checked, indirectly, by comparing results before and after the model's release.

## Leakage in the forecasting pipeline {h=64f2bd49}

Leakage means that information from the test period reaches the forecasts, directly or indirectly [@HewamalageEtAl2023]. Its effect is always optimistic: the model looks better than it will be when it forecasts an unknown future. The typical causes are mundane, which is why they are easy to overlook.

Four mistakes are common. The first is scaling or standardising the series with the mean and standard deviation of the whole sample, test period included. The second is choosing the ARIMA orders, the context length or the model version after seeing the test errors: the choice itself uses the test data. The third is using revised data, for example GDP from the latest release, as if they had been known at the origin, although the first estimate published then was different. The fourth, discussed in [Chapter 9](../EN/Courses/chapter9_machine_learning_time_series.pdf), is a random split of a time series into training and test sets, which puts future observations in the training sample.

The test in this chapter avoids the first two mistakes: each context ends at the origin, the ARIMA orders were fixed before the first origin, and Chronos computes its scale from the context only. The third remains: GDP and industrial production from Eurostat are the latest revised vintage, not the data available in real time. The effect, however, applies to all models in the same way.

## Contamination of benchmarks {h=80d22480}

Contamination means that the test series, or series strongly correlated with it, were in the model's pretraining corpus. The score then partly measures the model's memory, not its forecasting ability. Like leakage, contamination produces an optimistic bias, but it is not under the user's control, since the user did not choose the training data.

Serious developers acknowledge the problem. @AnsariEtAl2024 report two separate benchmarks for Chronos: an in-domain one, on data sets also used in training, with held-out test periods, and a zero-shot one, on data sets not used in training at all. For public series the boundary is hard to draw, however: the M4 series or US macroeconomic data are probably present in many corpora, and FRED and Eurostat are public.

There are three remedies. The first is to test only on data published after the release date of the weights, which could not have been in the corpus. The second is live benchmarks, in which forecasts are submitted before the outcome exists. The third is transparency: every study should report the training corpus and the release date of each model it uses.

## A contamination check: before and after the release {h=be529dce}

The chart applies the first remedy. For each series, it compares the WQL ratio to the seasonal naive forecast computed on the origins before the release of the weights (hatched bars) with the one computed on the origins after the release (solid bars). The left panel concerns Chronos-Bolt small, whose weights were released on 25 November 2024, on six series. The right panel concerns Chronos-2, released on 30 October 2025, on the two series that have origins after that date: load and EUR/RON.

ETS, estimated on the same origins, serves as a point of comparison. ETS has no pretraining corpus, so it cannot be contaminated; a change in its score between the two periods shows how different the post-release period is in itself. The reasoning is that of a difference in differences: the contamination signal is a deterioration of the foundation model larger than that of ETS.

## Interpreting the contamination check {h=63f9ab3b}

Chronos-2 on EUR/RON has a WQL ratio of 0.86 on the 124 origins before the release and 1.02 on the 10 origins after it: the advantage disappears. On load the result is the reverse: 0.68 before and 0.54 after, so the gain is real and holds on data the model could not have seen. Chronos-Bolt small on US retail sales moves from 0.82 before to 1.08 after, while ETS stays stable, at 0.69 and 0.68. US retail sales, a very well-known public series, is exactly the kind of series that may be in the corpus.

The conclusions must be stated with care. Between 3 and 35 origins after the release are few, and the post-release period, 2025–2026, differs from the earlier one; for GDP, for example, ETS also deteriorates after the release of Chronos-Bolt. A drop in performance after the release is a warning sign, not proof of contamination.

The check has a consequence for the results table. Chronos-2 was released in October 2025, and the last origins of the monthly and quarterly series are from July, August and September 2025, so all its results on these series use only origins before the release. They may therefore be partly contaminated and should be read as an optimistic bound on its performance.

## Recap: Leakage and contamination {h=7d46c96e}

Leakage is under the analyst's control: at each origin, only the data available then are used. Contamination is not, because the pretraining corpus may contain the very series being tested. Results are checked on data published after the model's release.

## @end

A good score of a foundation model has three possible explanations: real forecasting ability, leakage in the evaluation code and contamination of the pretraining corpus. The first can be separated from the other two only through discipline in the code and tests on data published after the model's release. On the data of this chapter, the Chronos-2 gain on load survives this test, and the gain on EUR/RON does not.

# Possible contribution of AI

An AI assistant can speed up the routine part of working with foundation models: the evaluation code, reading the papers and exploring many series. This section lists these uses and the checks that remain the analyst's responsibility, because the typical errors in this field, leakage and contamination, produce no error messages.

## Possible contribution of AI {h=78da55af}

Three uses are realistic. The first is code: an assistant can write a first version of a rolling-origin backtest, of functions for the pinball loss and the WQL, or of a fan chart. The second is reading: a summary of the paper of a new model, with its architecture, training corpus, licence and release date, exactly the information an honest evaluation needs. The third is exploration: running several open models on many Romanian series from Eurostat or from the National Institute of Statistics (INS), a task that is tedious to do by hand.

The example prompt on the slide shows how to phrase a verifiable request. It specifies the data (Romania's monthly industrial production, the Eurostat data set *sts_inpr_m*, not seasonally adjusted), the protocol (12-month forecasts from monthly origins since 2016), the models (Chronos-Bolt small from the package *chronos-forecasting*, on a CPU, ETS and the seasonal naive forecast) and the scores (the MASE, the WQL and the coverage of the 80% interval). A prompt this precise produces results that can be compared with those of this chapter.

## Necessary checks {h=3edaac65}

The first check concerns names: the package and the models must exist, for example the class *BaseChronosPipeline* and the model *amazon/chronos-bolt-small*. An invented class or model name fails at once, which is in fact the lucky case. The dangerous errors are those that do not stop the code: scaling or calibration done with data from after the origin, which produces leakage, or a quantile requested outside the model's levels. Chronos-Bolt outputs only the 0.1–0.9 quantiles; a "1% quantile" obtained from it is an extrapolation, not a forecast of the model.

The benchmarks must be present: the seasonal naive forecast, ETS and ARIMA on the same origins, and the scores must be reported relative to the naive forecast. The release date of the weights must be compared with the test period, for the reasons discussed in the section on contamination, and claims of the type "the best model" need a source, for example a published leaderboard. Finally, every paper cited by the assistant must exist: the arXiv number or the DOI must lead to the paper with the stated title.

## @end

An assistant can write the evaluation code, summarise papers and explore many series, but its output is accepted only after checking the package and model names, the absence of leakage, the quantile levels, the benchmarks, the release date against the test period and the references.

# Summary

The summary collects the main ideas of the chapter, the formulas to remember and the self-assessment questions.

## Key takeaways {h=245a9983}

A foundation model is pretrained once on many series and forecasts new series zero-shot, from the context only. Series become inputs of a Transformer through scaling and quantisation, as in Chronos, or through patching, as in TimesFM, Moirai and Chronos-Bolt. The outputs are quantiles, evaluated with the pinball loss, the CRPS or the WQL, and their calibration is checked through coverage.

On the seven series of the chapter, the gains are large for hourly load, absent on the central forecast of the series close to a random walk and negative for the Bolt models on the short quarterly GDP, while ARIMA remains a strong benchmark. Contamination can make published scores look better than they are, so results must be checked on data published after the model's release.

## Key formulas {h=c4246052}

The table collects the formulas of the chapter in the order in which they were introduced: attention @eq:attn, in the matrix form $\text{softmax}(QK^\top/\sqrt{d})\,V$; mean scaling @eq:scale; the pinball loss @eq:pinball, written compactly in the table as the maximum of $\tau(y - \hat q_\tau)$ and $(\tau - 1)(y - \hat q_\tau)$; the CRPS @eq:crps; the WQL @eq:wql; the MASE @eq:mase and coverage @eq:cov, with target 0.8. The compact form of the pinball loss is equivalent to the case form: the first term is the positive one when the observed value exceeds the quantile, the second when it is below it.

## Self-assessment (1/2) {h=a7e90990}

The first four questions check the basic definitions and computations. The difference between zero-shot forecasting and fine-tuning is whether the weights change: zero-shot keeps them fixed and uses only the context, fine-tuning trains them further on the target data. For the context 10, 20, 30, the scale of @eq:scale is $s = 60/3 = 20$, and the scaled values are 0.5, 1 and 1.5.

For $\hat q_{0.9} = 5$ and $y = 6$, the observed value is above the quantile, so the pinball loss is $0.9 \times (6 - 5) = 0.9$; exceeding the 0.9 quantile is costly, as the asymmetry of @eq:pinball shows. A context of 512 values split into patches of 16 gives $512/16 = 32$ patches, so attention compares 32 inputs instead of 512.

## Self-assessment (2/2) {h=54e9a007}

An 80% interval that covers 95% of the observed values is not a good result: the intervals are too wide, so they lack sharpness, and the CRPS penalises this. A foundation model cannot beat the random walk on EUR/RON because changes in the rate are almost unpredictable from the past, and no model can extract information that the context does not contain.

A model released in 2025 that scores very well on test data from 2018–2023 raises the question of contamination: those series may have been part of its pretraining corpus. The check consists in repeating the test on data published after the release. The project idea of the chapter applies exactly this check: comparing Chronos-Bolt, Chronos-2 and ARIMA on all the Romanian monthly series of a Eurostat data set, before and after the release dates. The next chapter, [Chapter 12](../EN/Courses/chapter12_spectral_analysis.pdf), turns to spectral analysis.

## @end

The chapter has shown that a pretrained model can forecast zero-shot series it has not seen, with large gains on series with regular cycles and long contexts and no gains on short series or series close to a random walk. The tools for judging these results are the same as in the rest of the course, extended to probabilistic forecasts: simple benchmarks, rolling origins, strictly proper scores and the discipline of using only the information available at each origin.
