##############################################################################
# Forecasting Models with Python                                             #
# Auto Regressive Integrated Moving Average Models                           #
# (c) Diego Fernandez Garcia 2015-2018                                       #
# www.exfinsis.com                                                           #
##############################################################################

# 1. Packages Importing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.tsa.holtwinters as ets
import statsmodels.tsa.stattools as ts
import statsmodels.tsa.statespace.sarimax as sarima
import statsmodels.stats.diagnostic as st
import statsmodels.tools.eval_measures as fa

##########################################

# 2. Forecasting Models Data

# 2.1. Data Reading
spy = pd.read_csv('Data//Forecasting-Models-Data.txt', index_col='Date', parse_dates=True)
spy = spy.asfreq('B')
spy = spy.fillna(method='ffill')
print('')
print('== Data Ranges Length ==')
print('')
print('Full Range Days: ', len(spy))
print('Full Range Months: ', np.round(len(spy)/22, 4))
print('')

# 2.2. Training Range Delimiting
spyt = spy[:'2013-12-31']
print('Training Range Days: ', len(spyt))
print('Training Range Months: ', np.round(len(spyt)/22, 4))
print('')

# 2.3. Testing Range Delimiting
spyf = spy['2014-01-02':]
print('Testing Range Days: ', len(spyf))
print('Testing Range Months: ', np.round(len(spyf)/22, 4))
print('')

# 2.4. Training and Testing Ranges Chart
fig1, ax = plt.subplots()
ax.plot(spyt, label='spyt')
ax.plot(spyf, label='spyf')
plt.legend(loc='upper left')
plt.title('SPY 2007-2015')
plt.ylabel('Price')
plt.xlabel('Date')
plt.show()

##########################################

# 4.4. Exponential Trend Method ETS(A,M,N)

# 4.4.1. Multi-Steps Forecast
expt1 = ets.ExponentialSmoothing(spyt.iloc[:, 0].values, trend='multiplicative', damped=False,
                                 seasonal=None).fit()
print('')
print('== Exponential Trend ETS(A,M,N) Parameters ==')
print('')
print('Smoothing Level: ', np.round(expt1.params['smoothing_level'], 4))
print('Smoothing Slope: ', np.round(expt1.params['smoothing_slope'], 4))
print('Initial Level: ', np.round(expt1.params['initial_level'], 4))
print('Initial Slope: ', np.round(expt1.params['initial_slope'], 4))
print('')
expf1 = expt1.forecast(steps=len(spyf))
expf1 = pd.DataFrame(expf1).set_index(spyf.index)

fig2, ax = plt.subplots()
ax.plot(spyt, label='spyt')
ax.plot(spyf, label='spyf')
ax.plot(expf1, label='expf1')
plt.legend(loc='upper left')
plt.title('Exponential Trend ETS(A,M,N) Method 1')
plt.ylabel('Price')
plt.xlabel('Date')
plt.show()

##########################################

# 5. Auto Regressive Integrated Moving Average Models

# 5.1. First Order Stationary Trend

# 5.1.1. Unit Root Tests
adfspyt = ts.adfuller(spyt.iloc[:, 0].values)
print('')
print('== Unit Root Test (spyt) ==')
print('')
print('Augmented Dickey-Fuller ADF Test P-Value: ', adfspyt[1])
print('')

# 5.1.2. Time Series Differencing
dspyt = spyt-spyt.shift(1)
dspyt = dspyt.fillna(method='bfill')

fig3, ax = plt.subplots()
ax.plot(spyt, label='spyt')
plt.legend(loc='upper left')
plt.title('SPY 2007-2013')
plt.ylabel('Price')
plt.xlabel('Date')
plt.show()

fig4, ax = plt.subplots()
ax.plot(dspyt, label='dspyt')
plt.legend(loc='upper left')
plt.title('SPY 2007-2013')
plt.ylabel('Price Differences')
plt.xlabel('Date')
plt.show()

adfdspyt = ts.adfuller(dspyt.iloc[:, 0].values)
print('')
print('== Unit Root Test (dspyt) ==')
print('')
print('Augmented Dickey-Fuller Test ADF P-Value: ', adfdspyt[1])
print('')

# 5.2. ARIMA Models Specification

# 5.2.1. Auto-correlation Function ACF
dspytacf = ts.acf(dspyt.iloc[:, 0].values)

plt.title('Auto-correlation Function ACF (dspyt)')
plt.bar(range(len(dspytacf)), dspytacf, width=1/2)
plt.axhline(y=0, color='black', linewidth=1/4)
plt.axhline(y=-1.96/np.sqrt(len(dspyt)), linestyle=':', color='gray')
plt.axhline(y=1.96/np.sqrt(len(dspyt)), linestyle=':', color='gray')
plt.show()

# 5.2.2. Partial Auto-correlation Function PACF
dspytpacf = ts.pacf(dspyt.iloc[:, 0].values)

plt.title('Partial Auto-correlation Function PACF (dspyt)')
plt.bar(range(len(dspytpacf)), dspytpacf, width=1/2)
plt.axhline(y=0, color='black', linewidth=1/4)
plt.axhline(y=-1.96/np.sqrt(len(dspyt)), linestyle=':', color='gray')
plt.axhline(y=1.96/np.sqrt(len(dspyt)), linestyle=':', color='gray')
plt.show()

##########################################

# 5.3. Random Walk with Drift Model ARIMA(0,1,0) With Constant

# 5.3.1. Multi-Steps Forecast
arwdt1 = sarima.SARIMAX(spyt, order=(0, 1, 0), trend='c').fit(disp=-1)
print('')
print('== Random Walk with Drift Model ARIMA(0,1,0) With Constant Parameters ==')
print('')
print(arwdt1.params)
print('')
arwdf1 = arwdt1.forecast(steps=len(spyf))
arwdf1 = pd.DataFrame(arwdf1).set_index(spyf.index)

fig5, ax = plt.subplots()
ax.plot(spyt, label='spyt')
ax.plot(spyf, label='spyf')
ax.plot(arwdf1, label='arwdf1')
plt.legend(loc='upper left')
plt.title('Random Walk with Drift ARIMA(0,1,0) With Constant Model 1')
plt.ylabel('Price')
plt.xlabel('Date')
plt.show()

# 5.3.2. One-Step Forecast without Re-Estimation
arwdf2 = sarima.SARIMAX(spy.tail(len(spyf)+1), order=(0, 1, 0), trend='c').smooth(params=arwdt1.params)
arwdf2 = arwdf2.fittedvalues.tail(len(spyf))
arwdf2 = pd.DataFrame(arwdf2).set_index(spyf.index)

fig6, ax = plt.subplots()
ax.plot(spyt, label='spyt')
ax.plot(spyf, label='spyf')
ax.plot(arwdf2, label='arwdf2', linestyle=':')
plt.legend(loc='upper left')
plt.title('Random Walk with Drift ARIMA(0,1,0) With Constant Model 2')
plt.ylabel('Price')
plt.xlabel('Date')
plt.show()

##########################################

# 5.4. Differentiated First Order Autoregressive ARIMA(1,1,0) With Constant

# 5.4.1. Multi-Steps Forecast
dart1 = sarima.SARIMAX(spyt, order=(1, 1, 0), trend='c').fit(disp=-1)
print('')
print('== Differentiated First Order Autoregressive ARIMA(1,1,0) With Constant Parameters ==')
print('')
print(dart1.params)
print('')
darf1 = dart1.forecast(steps=len(spyf))
darf1 = pd.DataFrame(darf1).set_index(spyf.index)

fig7, ax = plt.subplots()
ax.plot(spyt, label='spyt')
ax.plot(spyf, label='spyf')
ax.plot(darf1, label='darf1')
plt.legend(loc='upper left')
plt.title('Differentiated First Order Autoregressive ARIMA(1,1,0) With Constant Model 1')
plt.ylabel('Price')
plt.xlabel('Date')
plt.show()

# 5.4.2. One-Step Forecast without Re-Estimation
darf2 = sarima.SARIMAX(spy.tail(len(spyf)+1), order=(1, 1, 0), trend='c').smooth(params=dart1.params)
darf2 = darf2.fittedvalues.tail(len(spyf))
darf2 = pd.DataFrame(darf2).set_index(spyf.index)

fig8, ax = plt.subplots()
ax.plot(spyt, label='spyt')
ax.plot(spyf, label='spyf')
ax.plot(darf2, label='darf2', linestyle=':')
plt.legend(loc='upper left')
plt.title('Differentiated First Order Autoregressive ARIMA(1,1,0) With Constant Model 2')
plt.ylabel('Price')
plt.xlabel('Date')
plt.show()

##########################################

# 5.5. Brown Simple Exponential Smoothing ARIMA(0,1,1) Without Constant

# 5.5.1. Multi-Steps Forecast
abrownt1 = sarima.SARIMAX(spyt, order=(0, 1, 1), trend=None).fit(disp=-1)
print('')
print('== Brown Simple Exponential Smoothing ARIMA(0,1,1) Without Constant Parameters ==')
print('')
print(abrownt1.params)
print('')
abrownf1 = abrownt1.forecast(steps=len(spyf))
abrownf1 = pd.DataFrame(abrownf1).set_index(spyf.index)

fig9, ax = plt.subplots()
ax.plot(spyt, label='spyt')
ax.plot(spyf, label='spyf')
ax.plot(abrownf1, label='abrownf1')
plt.legend(loc='upper left')
plt.title('Brown Simple Exponential Smoothing ARIMA(0,1,1) Without Constant Model 1')
plt.ylabel('Price')
plt.xlabel('Date')
plt.show()

# 5.5.2. One-Step Forecast without Re-Estimation
abrownf2 = sarima.SARIMAX(spy.tail(len(spyf)+1), order=(0, 1, 1),
                          trend=None).smooth(params=abrownt1.params)
abrownf2 = abrownf2.fittedvalues.tail(len(spyf))
abrownf2 = pd.DataFrame(abrownf2).set_index(spyf.index)

fig10, ax = plt.subplots()
ax.plot(spyt, label='spyt')
ax.plot(spyf, label='spyf')
ax.plot(abrownf2, label='abrownf2', linestyle=':')
plt.legend(loc='upper left')
plt.title('Brown Simple Exponential Smoothing ARIMA(0,1,1) Without Constant Model 2')
plt.ylabel('Price')
plt.xlabel('Date')
plt.show()

##########################################

# 5.6. Holt Linear Trend ARIMA(0,2,2) Without Constant

# 5.6.1. Multi-Steps Forecast
aholtt1 = sarima.SARIMAX(spyt, order=(0, 2, 2), trend=None).fit(disp=-1)
print('')
print('== Holt Linear Trend ARIMA(0,2,2) Without Constant Parameters ==')
print('')
print(aholtt1.params)
print('')
aholtf1 = aholtt1.forecast(steps=len(spyf))
aholtf1 = pd.DataFrame(aholtf1).set_index(spyf.index)

fig11, ax = plt.subplots()
ax.plot(spyt, label='spyt')
ax.plot(spyf, label='spyf')
ax.plot(aholtf1, label='aholtf1')
plt.legend(loc='upper left')
plt.title('Holt Linear Trend ARIMA(0,2,2) Without Constant Model 1')
plt.ylabel('Price')
plt.xlabel('Date')
plt.show()

# 5.6.2. One-Step Forecast without Re-Estimation
aholtf2 = sarima.SARIMAX(spy.tail(len(spyf)+2), order=(0, 2, 2),
                         trend=None).smooth(params=aholtt1.params)
aholtf2 = aholtf2.fittedvalues.tail(len(spyf))
aholtf2 = pd.DataFrame(aholtf2).set_index(spyf.index)

fig12, ax = plt.subplots()
ax.plot(spyt, label='spyt')
ax.plot(spyf, label='spyf')
ax.plot(aholtf2, label='aholtf2', linestyle=':')
plt.legend(loc='upper left')
plt.title('Holt Linear Trend ARIMA(0,2,2) Without Constant Model 2')
plt.ylabel('Price')
plt.xlabel('Date')
plt.show()

##########################################

# 5.7. Gardner Additive Damped Trend ARIMA(1,1,2) Without Constant

# 5.7.1. Multi-Steps Forecast
agardnert1 = sarima.SARIMAX(spyt, order=(1, 1, 2), trend=None).fit(disp=-1)
print('')
print('== Gardner Additive Damped Trend ARIMA(1,1,2) Without Constant Parameters ==')
print('')
print(agardnert1.params)
print('')
agardnerf1 = agardnert1.forecast(steps=len(spyf))
agardnerf1 = pd.DataFrame(agardnerf1).set_index(spyf.index)

fig13, ax = plt.subplots()
ax.plot(spyt, label='spyt')
ax.plot(spyf, label='spyf')
ax.plot(agardnerf1, label='agardnerf1')
plt.legend(loc='upper left')
plt.title('Gardner Additive Damped Trend ARIMA(1,1,2) Without Constant Model 1')
plt.ylabel('Price')
plt.xlabel('Date')
plt.show()

# 5.7.2. One-Step Forecast without Re-Estimation
agardnerf2 = sarima.SARIMAX(spy.tail(len(spyf)+1), order=(1, 1, 2),
                            trend=None).smooth(params=agardnert1.params)
agardnerf2 = agardnerf2.fittedvalues.tail(len(spyf))
agardnerf2 = pd.DataFrame(agardnerf2).set_index(spyf.index)

fig14, ax = plt.subplots()
ax.plot(spyt, label='spyt')
ax.plot(spyf, label='spyf')
ax.plot(agardnerf2, label='agardnerf2', linestyle=':')
plt.legend(loc='upper left')
plt.title('Gardner Additive Damped Trend ARIMA(1,1,2) Without Constant Model 2')
plt.ylabel('Price')
plt.xlabel('Date')
plt.show()

##########################################

# 5.8. Seasonal Random Walk with Drift SARIMA(0,0,0)x(0,1,0)[22] With Constant

# 5.8.1. Multi-Steps Forecast
srwdt1 = sarima.SARIMAX(spyt, order=(0, 0, 0), seasonal_order=(0, 1, 0, 22), trend='c').fit(disp=-1)
print('')
print('== Seasonal Random Walk with Drift SARIMA(0,0,0)x(0,1,0)[22] With Constant Parameters ==')
print('')
print(srwdt1.params)
print('')
srwdf1 = srwdt1.forecast(steps=len(spyf))
srwdf1 = pd.DataFrame(srwdf1).set_index(spyf.index)

fig15, ax = plt.subplots()
ax.plot(spyt, label='spyt')
ax.plot(spyf, label='spyf')
ax.plot(srwdf1, label='srwdf1')
plt.legend(loc='upper left')
plt.title('Seasonal Random Walk with Drift SARIMA(0,0,0)x(0,1,0)[22] With Constant Model 1')
plt.ylabel('Price')
plt.xlabel('Date')
plt.show()

# 5.8.2. One-Step Forecast without Re-Estimation
srwdf2 = sarima.SARIMAX(spy.tail(len(spyf)+22), order=(0, 0, 0), seasonal_order=(0, 1, 0, 22),
                        trend='c').smooth(params=srwdt1.params)
srwdf2 = srwdf2.fittedvalues.tail(len(spyf))
srwdf2 = pd.DataFrame(srwdf2).set_index(spyf.index)

fig16, ax = plt.subplots()
ax.plot(spyt, label='spyt')
ax.plot(spyf, label='spyf')
ax.plot(srwdf2, label='srwdf2', linestyle=':')
plt.legend(loc='upper left')
plt.title('Seasonal Random Walk with Drift SARIMA(0,0,0)x(0,1,0)[22] With Constant Model 2')
plt.ylabel('Price')
plt.xlabel('Date')
plt.show()

##########################################

# 5.9. Seasonally Differentiated First Order Autoregressive SARIMA(1,0,0)x(0,1,0)[22] With Constant

# 5.9.1. Multi-Steps Forecast
sdart1 = sarima.SARIMAX(spyt, order=(1, 0, 0), seasonal_order=(0, 1, 0, 22), trend='c').fit(disp=-1)
print('')
print('== Seasonally Differentiated First Order Autoregressive SARIMA(1,0,0)x(0,1,0)[22] With Constant Parameters ==')
print('')
print(sdart1.params)
print('')
sdarf1 = sdart1.forecast(steps=len(spyf))
sdarf1 = pd.DataFrame(sdarf1).set_index(spyf.index)

fig17, ax = plt.subplots()
ax.plot(spyt, label='spyt')
ax.plot(spyf, label='spyf')
ax.plot(sdarf1, label='sdarf1')
plt.legend(loc='upper left')
plt.title('Seasonally Differentiated First Order Autoregressive SARIMA(1,0,0)x(0,1,0)[22] With Constant Model 1')
plt.ylabel('Price')
plt.xlabel('Date')
plt.show()

# 5.9.2. One-Step Forecast without Re-Estimation
sdarf2 = sarima.SARIMAX(spy.tail(len(spyf)+22), order=(1, 0, 0), seasonal_order=(0, 1, 0, 22),
                        trend='c').smooth(sdart1.params)
sdarf2 = sdarf2.fittedvalues.tail(len(spyf))
sdarf2 = pd.DataFrame(sdarf2).set_index(spyf.index)

fig18, ax = plt.subplots()
ax.plot(spyt, label='spyt')
ax.plot(spyf, label='spyf')
ax.plot(sdarf2, label='sdarf2', linestyle=':')
plt.legend(loc='upper left')
plt.title('Seasonally Differentiated First Order Autoregressive SARIMA(1,0,0)x(0,1,0)[22] With Constant Model 2')
plt.ylabel('Price')
plt.xlabel('Date')
plt.show()

##########################################

# 5.10. Holt-Winters Additive Seasonality SARIMA(0,1,22+1)x(0,1,0)[22] Without Constant

# 5.10.1. Multi-Steps Forecast
# ahwat1 = sarima.SARIMAX(spyt, order=(0, 1, 23), seasonal_order=(0, 1, 0, 22), trend=None).fit(disp=-1)
# print('')
# print('== Holt-Winters Additive Seasonality SARIMA(0,1,22+1)x(0,1,0)[22] Without Constant Parameters ==')
# print('')
# print(ahwat1.params)
# print('')
# ahwaf1 = ahwat1.forecast(steps=len(spyf))
# ahwaf1 = pd.DataFrame(ahwaf1).set_index(spyf.index)

# fig19, ax = plt.subplots()
# ax.plot(spyt, label='spyt')
# ax.plot(spyf, label='spyf')
# ax.plot(ahwaf1, label='ahwaf1')
# plt.legend(loc='upper left')
# plt.title('Holt-Winters Additive Seasonality SARIMA (0,1,22+1)x(0,1,0)[22] Without Constant Model 1')
# plt.ylabel('Price')
# plt.xlabel('Date')
# plt.show()

# 5.10.2. One-Step Forecast without Re-Estimation
# ahwaf2 = sarima.SARIMAX(spy.tail(len(spyf)+23), order=(0, 1, 23), seasonal_order=(0, 1, 0, 22),
#                         trend=None).smooth(ahwat1.params)
# ahwaf2 = ahwaf2.fittedvalues.tail(len(spyf))
# ahwaf2 = pd.DataFrame(ahwaf2).set_index(spyf.index)

# fig20, ax = plt.subplots()
# ax.plot(spyt, label='spyt')
# ax.plot(spyf, label='spyf')
# ax.plot(ahwaf2, label='ahwaf2', linestyle=':')
# plt.legend(loc='upper left')
# plt.title('Holt-Winters Additive Seasonality SARIMA (0,1,22+1)x(0,1,0)[22] Without Constant Model 2')
# plt.ylabel('Price')
# plt.xlabel('Date')
# plt.show()

#########################################

# 5.11. ARIMA Model Selection

arwdaict1 = arwdt1.aic
arwdbict1 = arwdt1.bic
daraict1 = dart1.aic
darbict1 = dart1.bic
abrownaict1 = abrownt1.aic
abrownbict1 = abrownt1.bic
aholtaict1 = aholtt1.aic
aholtbict1 = aholtt1.bic
agardneraict1 = agardnert1.aic
agardnerbict1 = agardnert1.bic
srwdaict1 = srwdt1.aic
srwdbict1 = srwdt1.bic
sdaraict1 = sdart1.aic
sdarbict1 = sdart1.bic
# ahwaaict1 = ahwat1.aic
# ahwabict1 = ahwat1.bic

msdata = [{'0': '', '1': 'AIC', '2': 'BIC'},
        {'0': 'Random Walk with Drift ARIMA(0,1,0) Model 1', '1': np.round(arwdaict1, 4),
         '2': np.round(arwdbict1, 4)},
        {'0': 'Differentiated First Order ARIMA(1,1,0) Model 1', '1': np.round(daraict1, 4),
         '2': np.round(darbict1, 4)},
        {'0': 'Brown Simple Exponential Smoothing ARIMA(0,1,1) Model 1', '1': np.round(abrownaict1, 4),
         '2': np.round(abrownbict1, 4)},
        {'0': 'Holt Linear Trend ARIMA(0,2,2) Model 1', '1': np.round(aholtaict1, 4),
         '2': np.round(aholtbict1, 4)},
        {'0': 'Gardner Additive Damped Trend ARIMA(1,1,2) Model 1', '1': np.round(agardneraict1, 4),
         '2': np.round(agardnerbict1, 4)},
        {'0': 'Seasonal Random Walk with Drift SARIMA(0,0,0)x(0,1,0)[22] Model 1',
         '1': np.round(srwdaict1, 4), '2': np.round(srwdbict1, 4)},
        {'0': 'Seasonally Differentiated First Order Autoregressive SARIMA(1,0,0)x(0,1,0)[22] Model 1',
         '1': np.round(sdaraict1, 4), '2': np.round(sdarbict1, 4)},
        # {'0': 'Holt-Winters Additive SARIMA(0,1,22+1)x(0,1,0)[22] Model 1',
        #  '1': np.round(ahwaaict1, 4), '2': np.round(ahwabict1, 4)},
          ]
mstable = pd.DataFrame(msdata)
print('')
print('== ARIMA Model Selection ==')
print('')
print(mstable)
print('')

##########################################

# 5.12. Models Forecasting Accuracy

# 5.12.1. Multi-Steps Forecast
expmae1 = fa.meanabs(expf1, spyf)
exprmse1 = fa.rmse(expf1, spyf)
arwdmae1 = fa.meanabs(arwdf1, spyf)
arwdrmse1 = fa.rmse(arwdf1, spyf)
darmae1 = fa.meanabs(darf1, spyf)
darrmse1 = fa.rmse(darf1, spyf)
abrownmae1 = fa.meanabs(abrownf1, spyf)
abrownrmse1 = fa.rmse(abrownf1, spyf)
aholtmae1 = fa.meanabs(aholtf1, spyf)
aholtrmse1 = fa.rmse(aholtf1, spyf)
agardnermae1 = fa.meanabs(agardnerf1, spyf)
agardnerrmse1 = fa.rmse(agardnerf1, spyf)
srwdmae1 = fa.meanabs(srwdf1, spyf)
srwdrmse1 = fa.rmse(srwdf1, spyf)
sdarmae1 = fa.meanabs(sdarf1, spyf)
sdarrmse1 = fa.rmse(sdarf1, spyf)
# ahwamae1 = fa.meanabs(ahwaf1, spyf)
# ahwarmse1 = fa.rmse(ahwaf1, spyf)

fadata1 = [{'0': '', '1': 'MAE', '2': 'RMSE'},
        {'0': 'Exponential Trend ETS(A,M,N) Method 1', '1': np.round(expmae1, 4),
         '2': np.round(exprmse1, 4)},
        {'0': 'Random Walk with Drift ARIMA(0,1,0) Model 1', '1': np.round(arwdmae1, 4),
         '2': np.round(arwdrmse1, 4)},
        {'0': 'Differentiated First Order ARIMA(1,1,0) Model 1', '1': np.round(darmae1, 4),
         '2': np.round(darrmse1, 4)},
        {'0': 'Brown Simple Exponential Smoothing ARIMA(0,1,1) Model 1', '1': np.round(abrownmae1, 4),
         '2': np.round(abrownrmse1, 4)},
        {'0': 'Holt Linear Trend ARIMA(0,2,2) Model 1', '1': np.round(aholtmae1, 4),
         '2': np.round(aholtrmse1, 4)},
        {'0': 'Gardner Additive Damped Trend ARIMA(1,1,2) Model 1', '1': np.round(agardnermae1, 4),
         '2': np.round(agardnerrmse1, 4)},
        {'0': 'Seasonal Random Walk with Drift SARIMA(0,0,0)x(0,1,0)[22] Model 1',
         '1': np.round(srwdmae1, 4), '2': np.round(srwdrmse1, 4)},
        {'0': 'Seasonally Differentiated First Order Autoregressive SARIMA(1,0,0)x(0,1,0)[22] Model 1',
         '1': np.round(sdarmae1, 4), '2': np.round(sdarrmse1, 4)},
        # {'0': 'Holt-Winters Additive SARIMA(0,1,22+1)x(0,1,0)[22] Model 1',
        #  '1': np.round(ahwamae1, 4), '2': np.round(ahwarmse1, 4)},
           ]
fatable1 = pd.DataFrame(fadata1)
print('')
print('== Multi-Steps Forecasting Accuracy ==')
print('')
print(fatable1)
print('')

# 5.12.2. One-Step Forecast without Re-Estimation
arwdmae2 = fa.meanabs(arwdf2, spyf)
arwdrmse2 = fa.rmse(arwdf2, spyf)
darmae2 = fa.meanabs(darf2, spyf)
darrmse2 = fa.rmse(darf2, spyf)
abrownmae2 = fa.meanabs(abrownf2, spyf)
abrownrmse2 = fa.rmse(abrownf2, spyf)
aholtmae2 = fa.meanabs(aholtf2, spyf)
aholtrmse2 = fa.rmse(aholtf2, spyf)
agardnermae2 = fa.meanabs(agardnerf2, spyf)
agardnerrmse2 = fa.rmse(agardnerf2, spyf)
srwdmae2 = fa.meanabs(srwdf2, spyf)
srwdrmse2 = fa.rmse(srwdf2, spyf)
sdarmae2 = fa.meanabs(sdarf2, spyf)
sdarrmse2 = fa.rmse(sdarf2, spyf)
# ahwamae2 = fa.meanabs(ahwaf2, spyf)
# ahwarmse2 = fa.rmse(ahwaf2, spyf)

fadata2 = [{'0': '', '1': 'MAE', '2': 'RMSE'},
        {'0': 'Random Walk with Drift ARIMA(0,1,0) Model 2', '1': np.round(arwdmae2, 4),
         '2': np.round(arwdrmse2, 4)},
        {'0': 'Differentiated First Order ARIMA(1,1,0) Model 2', '1': np.round(darmae2, 4),
         '2': np.round(darrmse2, 4)},
        {'0': 'Brown Simple Exponential Smoothing ARIMA(0,1,1) Model 2', '1': np.round(abrownmae2, 4),
         '2': np.round(abrownrmse2, 4)},
        {'0': 'Holt Linear Trend ARIMA(0,2,2) Model 2', '1': np.round(aholtmae2, 4),
         '2': np.round(aholtrmse2, 4)},
        {'0': 'Gardner Additive Damped Trend ARIMA(1,1,2) Model 2', '1': np.round(agardnermae2, 4),
         '2': np.round(agardnerrmse2, 4)},
        {'0': 'Seasonal Random Walk with Drift SARIMA(0,0,0)x(0,1,0)[22] Model 2',
         '1': np.round(srwdmae2, 4), '2': np.round(srwdrmse2, 4)},
        {'0': 'Seasonally Differentiated First Order Autoregressive SARIMA(1,0,0)x(0,1,0)[22] Model 2',
         '1': np.round(sdarmae2, 4), '2': np.round(sdarrmse2, 4)},
        # {'0': 'Holt-Winters Additive SARIMA(0,1,22+1)x(0,1,0)[22] Model 2',
        #  '1': np.round(ahwamae2, 4), '2': np.round(ahwarmse2, 4)},
           ]
fatable2 = pd.DataFrame(fadata2)
print('')
print('== One-Step without Re-Estimation Forecasting Accuracy ==')
print('')
print(fatable2)
print('')

##########################################

# 5.13. Residuals White Noise

# 5.13.1. Auto-correlation Function
arwdt1res = arwdt1.resid.tail(len(spyt)-1)
arwdt1sres = (arwdt1res-arwdt1res.mean())/arwdt1res.std()
arwdt1acf = ts.acf(arwdt1sres)

plt.title('Auto-correlation Function ACF (arwdt1sres)')
plt.bar(range(len(arwdt1acf)), arwdt1acf, width=1/2)
plt.axhline(y=0, color='black', linewidth=1/4)
plt.axhline(y=-1.96/np.sqrt(len(spyt)), linestyle=':', color='gray')
plt.axhline(y=1.96/np.sqrt(len(spyt)), linestyle=':', color='gray')
plt.show()

# 5.13.2. Partial Auto-correlation Function PACF
arwdt1pacf = ts.pacf(arwdt1sres)

plt.title('Partial Auto-correlation Function PACF (arwdt1sres)')
plt.bar(range(len(arwdt1pacf)), arwdt1pacf, width=1/2)
plt.axhline(y=0, color='black', linewidth=1/4)
plt.axhline(y=-1.96/np.sqrt(len(spyt)), linestyle=':', color='gray')
plt.axhline(y=1.96/np.sqrt(len(spyt)), linestyle=':', color='gray')
plt.show()

# 5.13.3. Ljung-Box Autocorrelation Test
print('')
print('== Random Walk with Drift ARIMA(0,1,0) With Constant Model Residuals White Noise ==')
print('')
print('Ljung-Box Autocorrelation Test P-Value (arwdt1sres): ',
      np.round(st.acorr_ljungbox(arwdt1sres, lags=22)[1][21], 4))