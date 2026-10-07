# Acronime specifice Capitolului 9 TSA (Învățare automată pentru serii de timp)
# format: acronim -> (forma de origine, limba de origine, traducere RO, traducere EN)
# OVERRIDE_CH (optional): acronim -> tuplu, sens diferit doar in acest capitol.
# GB = gradient boosting; acronimul GBM nu se folosește (în dicționarul comun GBM = mișcarea browniană geometrică).
EXTRA = {
    'RW': ('Random Walk (forecast)', 'en', 'prognoza de tip mers aleator', None),
    'MIMO': ('Multiple-Input Multiple-Output (forecasting strategy)', 'en', 'strategia cu mai multe ieșiri: un model dă toate orizonturile deodată', None),
    'OWA': ('Overall Weighted Average (of the relative sMAPE and MASE, M4 competition)', 'en', 'media ponderată a sMAPE și MASE relative (competiția M4)', None),
    'MAPE': ('Mean Absolute Percentage Error', 'en', 'eroarea procentuală absolută medie', None),
    'ES-RNN': ('Exponential Smoothing – Recurrent Neural Network (hybrid model)', 'en', 'model hibrid de netezire exponențială și rețea neuronală recurentă', None),
    'GBRT': ('Gradient Boosted Regression Trees', 'en', 'arbori de regresie cu gradient boosting', None),
    'NN': ('Neural Network', 'en', 'rețea neuronală', None),
    'GLM': ('Generalised Linear Model', 'en', 'model liniar generalizat', None),
    'ENet': ('Elastic Net', 'en', 'regresie penalizată elastic net', None),
    'ReLU': ('Rectified Linear Unit', 'en', 'funcția de activare liniară rectificată', None),
    'CPU': ('Central Processing Unit', 'en', 'procesorul central', None),
    'GPU': ('Graphics Processing Unit', 'en', 'procesorul grafic', None),
    'SGD': ('Stochastic Gradient Descent', 'en', 'coborîrea stochastică pe gradient', None),
    'RMSSE': ('Root Mean Squared Scaled Error', 'en', 'rădăcina erorii pătratice medii scalate', None),
    'WRMSSE': ('Weighted Root Mean Squared Scaled Error (M5 competition)', 'en', 'rădăcina erorii pătratice medii scalate, ponderată (competiția M5)', None),
    'OLS-3': ('OLS with three predictors: size, book-to-market and momentum (Gu, Kelly and Xiu, 2020)', 'en', 'OLS cu trei predictori: capitalizarea, raportul valoare contabilă/valoare de piață și momentum', None),
    'SSE': ('Sum of Squared Errors', 'en', 'suma pătratelor erorilor', None),
    'MSE': ('Mean Squared Error', 'en', 'eroarea pătratică medie', None),
    'HICP': ('Harmonised Index of Consumer Prices', 'en', 'indicele armonizat al prețurilor de consum', None),
    'IAPC': ('Indicele armonizat al prețurilor de consum', 'ro', None, 'the harmonised index of consumer prices'),
    'ENTSO-E': ('European Network of Transmission System Operators for Electricity', 'en', 'rețeaua europeană a operatorilor de transport și de sistem pentru energie electrică', None),
    'EU': ('European Union', 'en', 'Uniunea Europeană', None),
    'UE': ('Uniunea Europeană', 'ro', None, 'the European Union'),
    'DHR': ('Dynamic Harmonic Regression', 'en', 'regresie armonică dinamică', None),
    'HLN': ('Harvey–Leybourne–Newbold (small-sample correction of the DM test)', 'en', 'corecția Harvey–Leybourne–Newbold a testului DM pentru eșantioane mici', None),
}
OVERRIDE_CH = {
    'RF': ('Random Forest', 'en', 'pădure aleatoare', None),
    'AR': ('AutoRegressive (model)', 'en', 'model autoregresiv', None),
    'GB': ('Gradient Boosting (ensemble of regression trees)', 'en', 'gradient boosting (ansamblu de arbori de regresie)', None),
}
