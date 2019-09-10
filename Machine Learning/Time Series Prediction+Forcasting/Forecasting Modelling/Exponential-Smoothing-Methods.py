##############################################################################
# Forecasting Models with Python                                             #
# Exponential Smoothing Methods                                              #
# (c) Diego Fernandez Garcia 2015-2018                                       #
# www.exfinsis.com                                                           #
##############################################################################

# 1. Packages Importing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.tsa.holtwinters as ets
import statsmodels.tools.eval_measures as fa
import os

##########################################

# 2. Forecasting Models Data

# 2.1. Data Reading
os.chdir("/Users/sulbha/Documents/git_repo/Machine Learning/Time Series Prediction+Forcasting/forecasting Modelling")

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

# 3.4. Random Walk with Drift Method

# 3.4.1. Multi-Steps Forecast

# Random Walk Calculation
rwdf1 = pd.concat([spyt.tail(1)]*len(spyf))
rwdf1 = pd.DataFrame(rwdf1).set_index(spyf.index)

# Drift Calculation
dspy = spy-spy.shift(1)
dmeanf1 = pd.concat([dspy[:'2013-12-31'].mean()]*len(spyf))
dmeanf1 = pd.DataFrame(dmeanf1).set_index(spyf.index)
driftf1 = dmeanf1.cumsum()

# Random Walk with Drift Calculation
rwdf1['SPY.Drift'] = driftf1
rwdf1 = rwdf1.sum(axis=1)
rwdf1 = pd.DataFrame(rwdf1).set_index(spyf.index)

fig2, ax = plt.subplots()
ax.plot(spyt, label='spyt')
ax.plot(spyf, label='spyf')
ax.plot(rwdf1, label='rwdf1')
plt.legend(loc='upper left')
plt.title('Random Walk with Drift Method 1')
plt.ylabel('Price')
plt.xlabel('Date')
plt.show()

# 3.4.2. One-Step Forecast without Re-Estimation
rwdf2 = spy.shift(1)['2014-01-02':]
rwdf2['SPY.DiffMean'] = dmeanf1
rwdf2 = rwdf2.sum(axis=1)
rwdf2 = pd.DataFrame(rwdf2).set_index(spyf.index)

fig3, ax = plt.subplots()
ax.plot(spyt, label='spyt')
ax.plot(spyf, label='spyf')
ax.plot(rwdf2, label='rwdf2', linestyle=':')
plt.legend(loc='upper left')
plt.title('Random Walk with Drift Method 2')
plt.ylabel('Price')
plt.xlabel('Date')
plt.show()

##########################################

# 4. Exponential Smoothing Methods

# 4.1. Simple Moving Average SMA

# 4.1.1. Multi-Steps Forecast
smaf1 = pd.concat([spyt.tail(22).mean()]*len(spyf))
smaf1 = pd.DataFrame(smaf1).set_index(spyf.index)

fig4, ax = plt.subplots()
ax.plot(spyt, label='spyt')
ax.plot(spyf, label='spyf')
ax.plot(smaf1, label='smaf1')
plt.legend(loc='upper left')
plt.title('Simple Moving Average SMA Method 1')
plt.ylabel('Price')
plt.xlabel('Date')
plt.show()

# 4.1.2. One-Step Forecast without Re-Estimation
smaf2 = spy.shift(1).rolling(window=22).mean()
smaf2 = smaf2['2014-01-02':]

fig5, ax = plt.subplots()
ax.plot(spyt, label='spyt')
ax.plot(spyf, label='spyf')
ax.plot(smaf2, label='smaf2', linestyle=':')
plt.legend(loc='upper left')
plt.title('Simple Moving Average SMA Method 2')
plt.ylabel('Price')
plt.xlabel('Date')
plt.show()

##########################################

# 4.2. Brown Simple Exponential Smoothing ETS(A,N,N)

# 4.2.1. Multi-Steps Forecast
brownt1 = ets.ExponentialSmoothing(spyt, trend=None, damped=False, seasonal=None).fit()
print('')
print('== Brown Simple Exponential Smoothing ETS(A,N,N) Parameters ==')
print('')
print('Smoothing Level: ', np.round(brownt1.params['smoothing_level'], 4))
print('Initial Level: ', np.round(brownt1.params['initial_level'], 4))
print('')
brownf1 = brownt1.forecast(steps=len(spyf))
brownf1 = pd.DataFrame(brownf1).set_index(spyf.index)

fig6, ax = plt.subplots()
ax.plot(spyt, label='spyt')
ax.plot(spyf, label='spyf')
ax.plot(brownf1, label='brownf1')
plt.legend(loc='upper left')
plt.title('Brown Simple Exponential Smoothing ETS(A,N,N) Method 1')
plt.ylabel('Price')
plt.xlabel('Date')
plt.show()

# 4.2.2. One-Step Forecast without Re-Estimation
spyf1 = spy.shift(1)['2014-01-02':]
brownf2 = ets.ExponentialSmoothing(spyf1, trend=None, damped=False, seasonal=None).fit(
    smoothing_level=brownt1.params['smoothing_level'])
brownf2 = brownf2.predict(start=1, end=len(spyf))
brownf2 = pd.DataFrame(brownf2).set_index(spyf.index)

fig7, ax = plt.subplots()
ax.plot(spyt, label='spyt')
ax.plot(spyf, label='spyf')
ax.plot(brownf2, label='brownf2', linestyle=':')
plt.legend(loc='upper left')
plt.title('Brown Simple Exponential Smoothing ETS(A,N,N) Method 2')
plt.ylabel('Price')
plt.xlabel('Date')
plt.show()

##########################################

# 4.3. Holt Linear Trend Method ETS(A,A,N)

# 4.3.1. Multi-Steps Forecast
holtt1 = ets.ExponentialSmoothing(spyt, trend='additive', damped=False, seasonal=None).fit()
print('')
print('== Holt Linear Trend ETS(A,A,N) Parameters ==')
print('')
print('Smoothing Level: ', np.round(holtt1.params['smoothing_level'], 4))
print('Smoothing Slope: ', np.round(holtt1.params['smoothing_slope'], 4))
print('Initial Level: ', np.round(holtt1.params['initial_level'], 4))
print('Initial Slope: ', np.round(holtt1.params['initial_slope'], 4))
print('')
holtf1 = holtt1.forecast(steps=len(spyf))
holtf1 = pd.DataFrame(holtf1).set_index(spyf.index)

fig8, ax = plt.subplots()
ax.plot(spyt, label='spyt')
ax.plot(spyf, label='spyf')
ax.plot(holtf1, label='holtf1')
plt.legend(loc='upper left')
plt.title('Holt Linear Trend ETS(A,A,N) Method 1')
plt.ylabel('Price')
plt.xlabel('Date')
plt.show()

# 4.3.2. One-Step Forecast without Re-Estimation
holtf2 = ets.ExponentialSmoothing(spyf1, trend='additive', damped=False, seasonal=None).fit(
    smoothing_level=holtt1.params['smoothing_level'], smoothing_slope=holtt1.params['smoothing_slope'])
holtf2 = holtf2.predict(start=1, end=len(spyf))
holtf2 = pd.DataFrame(holtf2).set_index(spyf.index)

fig9, ax = plt.subplots()
ax.plot(spyt, label='spyt')
ax.plot(spyf, label='spyf')
ax.plot(holtf2, label='holtf2', linestyle=':')
plt.legend(loc='upper left')
plt.title('Holt Linear Trend ETS(A,A,N) Method 2')
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

fig10, ax = plt.subplots()
ax.plot(spyt, label='spyt')
ax.plot(spyf, label='spyf')
ax.plot(expf1, label='expf1')
plt.legend(loc='upper left')
plt.title('Exponential Trend ETS(A,M,N) Method 1')
plt.ylabel('Price')
plt.xlabel('Date')
plt.show()

# 4.4.2. One-Step Forecast without Re-Estimation
expf2 = ets.ExponentialSmoothing(spyf1.iloc[:, 0].values, trend='multiplicative', damped=False,
                                 seasonal=None).fit(smoothing_level=expt1.params['smoothing_level'],
                                                    smoothing_slope=expt1.params['smoothing_slope'])
expf2 = expf2.predict(start=1, end=len(spyf))
expf2 = pd.DataFrame(expf2).set_index(spyf.index)

fig11, ax = plt.subplots()
ax.plot(spyt, label='spyt')
ax.plot(spyf, label='spyf')
ax.plot(expf2, label='expf2', linestyle=':')
plt.legend(loc='upper left')
plt.title('Exponential Trend ETS(A,M,N) Method 2')
plt.ylabel('Price')
plt.xlabel('Date')
plt.show()

##########################################

# 4.5. Gardner Additive Damped Trend Method ETS(A,Ad,N)

# 4.5.1. Multi-Steps Forecast
gardnert1 = ets.ExponentialSmoothing(spyt, trend='additive', damped=True, 
                                  seasonal=None).fit()
print('')
print('== Gardner Additive Damped Trend ETS(A,Ad,N) Parameters ==')
print('')
print('Smoothing Level: ', np.round(gardnert1.params['smoothing_level'], 4))
print('Smoothing Slope: ', np.round(gardnert1.params['smoothing_slope'], 4))
print('Damping Slope: ', np.round(gardnert1.params['damping_slope'], 4))
print('Initial Level: ', np.round(gardnert1.params['initial_level'], 4))
print('Initial Slope: ', np.round(gardnert1.params['initial_slope'], 4))
print('')
gardnerf1 = gardnert1.forecast(steps=len(spyf))
gardnerf1 = pd.DataFrame(gardnerf1).set_index(spyf.index)

fig12, ax = plt.subplots()
ax.plot(spyt, label='spyt')
ax.plot(spyf, label='spyf')
ax.plot(gardnerf1, label='gardnerf1')
plt.legend(loc='upper left')
plt.title('Gardner Additive Damped Trend ETS(A,Ad,N) Method 1')
plt.ylabel('Price')
plt.xlabel('Date')
plt.show()

# 4.5.2. One-Step Forecast without Re-Estimation
gardnerf2 = ets.ExponentialSmoothing(spyf1, trend='additive', damped=True, seasonal=None).fit(
     smoothing_level=gardnert1.params['smoothing_level'], smoothing_slope=gardnert1.params['smoothing_slope'],
     damping_slope=gardnert1.params['damping_slope'])
gardnerf2 = gardnerf2.predict(start=1, end=len(spyf))
gardnerf2 = pd.DataFrame(gardnerf2).set_index(spyf.index)

fig13, ax = plt.subplots()
ax.plot(spyt, label='spyt')
ax.plot(spyf, label='spyf')
ax.plot(gardnerf2, label='gardnerf2', linestyle=':')
plt.legend(loc='upper left')
plt.title('Gardner Additive Damped Trend ETS(A,Ad,N) Method 2')
plt.ylabel('Price')
plt.xlabel('Date')
plt.show()

##########################################

# 4.6. Taylor Multiplicative Damped Trend Method ETS(A,Md,N)

# 4.6.1. Multi-Steps Forecast
taylort1 = ets.ExponentialSmoothing(spyt.iloc[:, 0].values, 
                                    trend='multiplicative',
                                    damped=True, 
                                 seasonal=None).fit()
print('')
print('== Taylor Multiplicative Damped Trend ETS(A,Md,N) Parameters ==')
print('')
print('Smoothing Level: ', np.round(taylort1.params['smoothing_level'], 4))
print('Smoothing Slope: ', np.round(taylort1.params['smoothing_slope'], 4))
print('Damping Slope: ', np.round(taylort1.params['damping_slope'], 4))
print('Initial Level: ', np.round(taylort1.params['initial_level'], 4))
print('Initial Slope: ', np.round(taylort1.params['initial_slope'], 4))
print('')
taylorf1 = taylort1.forecast(steps=len(spyf))
taylorf1 = pd.DataFrame(taylorf1).set_index(spyf.index)

fig14, ax = plt.subplots()
ax.plot(spyt, label='spyt')
ax.plot(spyf, label='spyf')
ax.plot(taylorf1, label='taylorf1')
plt.legend(loc='upper left')
plt.title('Taylor Multiplicative Damped Trend ETS(A,Md,N) Method 1')
plt.ylabel('Price')
plt.xlabel('Date')
plt.show()

# 4.6.2. One-Step Forecast without Re-Estimation
taylorf2 = ets.ExponentialSmoothing(spyf1.iloc[:, 0].values, trend='multiplicative', damped=True,
                                     seasonal=None).fit(smoothing_level=taylort1.params['smoothing_level'],
                                                        smoothing_slope=taylort1.params['smoothing_slope'],
                                                        damping_slope=taylort1.params['damping_slope'])
taylorf2 = taylorf2.predict(start=1, end=len(spyf))
taylorf2 = pd.DataFrame(taylorf2).set_index(spyf.index)

fig15, ax = plt.subplots()
ax.plot(spyt, label='spyt')
ax.plot(spyf, label='spyf')
ax.plot(taylorf2, label='taylorf2', linestyle=':')
plt.legend(loc='upper left')
plt.title('Taylor Multiplicative Damped Trend ETS(A,Md,N) Method 2')
plt.ylabel('Price')
plt.xlabel('Date')
plt.show()

##########################################

# 4.7. Holt-Winters Additive Method ETS(A,A,A)

# 4.7.1. Multi-Steps Forecast
 hwat1 = ets.ExponentialSmoothing(spyt, trend='additive', damped=False, seasonal='additive',
                                  seasonal_periods=22).fit()
 print('')
 print('== Holt-Winters Additive ETS(A,A,A) Parameters ==')
 print('')
 print('Smoothing Level: ', np.round(hwat1.params['smoothing_level'], 4))
 print('Smoothing Slope: ', np.round(hwat1.params['smoothing_slope'], 4))
 print('Smoothing Seasonal: ', np.round(hwat1.params['smoothing_seasonal'], 4))
 print('Initial Level: ', np.round(hwat1.params['initial_level'], 4))
 print('Initial Slope: ', np.round(hwat1.params['initial_slope'], 4))
 print('Initial Seasons: ', np.round(hwat1.params['initial_seasons'], 4))
 print('')
 hwaf1 = hwat1.forecast(steps=len(spyf))
 hwaf1 = pd.DataFrame(hwaf1).set_index(spyf.index)

 fig16, ax = plt.subplots()
 ax.plot(spyt, label='spyt')
 ax.plot(spyf, label='spyf')
 ax.plot(hwaf1, label='hwaf1')
 plt.legend(loc='upper left')
 plt.title('Holt-Winters Additive ETS(A,A,A) Method 1')
 plt.ylabel('Price')
 plt.xlabel('Date')
 plt.show()

# 4.7.2. One-Step Forecast without Re-Estimation
# hwaf2 = ets.ExponentialSmoothing(spyf1, trend='additive', damped=False, seasonal='additive',
#                                  seasonal_periods=22).fit(smoothing_level=hwat1.params['smoothing_level'],
#                                                           smoothing_slope=hwat1.params['smoothing_slope'],
#                                                           smoothing_seasonal=hwat1.params['smoothing_seasonal'])
# hwaf2 = hwaf2.predict(start=1, end=len(spyf))
# hwaf2 = pd.DataFrame(hwaf2).set_index(spyf.index)

# fig17, ax = plt.subplots()
# ax.plot(spyt, label='spyt')
# ax.plot(spyf, label='spyf')
# ax.plot(hwaf2, label='hwaf2', linestyle=':')
# plt.legend(loc='upper left')
# plt.title('Holt-Winters Additive ETS(A,A,A) Method 2')
# plt.ylabel('Price')
# plt.xlabel('Date')
# plt.show()

##########################################

# 4.8. Holt-Winters Multiplicative Method ETS(A,A,M)

# 4.8.1. Multi-Steps Forecast
# hwmt1 = ets.ExponentialSmoothing(spyt.iloc[:, 0].values, trend='additive', damped=False,
#                                  seasonal='multiplicative', seasonal_periods=22).fit()
# print('')
# print('== Holt-Winters Multiplicative ETS(A,A,M) Parameters ==')
# print('')
# print('Smoothing Level: ', np.round(hwmt1.params['smoothing_level'], 4))
# print('Smoothing Slope: ', np.round(hwmt1.params['smoothing_slope'], 4))
# print('Smoothing Seasonal: ', np.round(hwmt1.params['smoothing_seasonal'], 4))
# print('Initial Level: ', np.round(hwmt1.params['initial_level'], 4))
# print('Initial Slope: ', np.round(hwmt1.params['initial_slope'], 4))
# print('Initial Seasons: ', np.round(hwmt1.params['initial_seasons'], 4))
# print('')
# hwmf1 = hwmt1.forecast(steps=len(spyf))
# hwmf1 = pd.DataFrame(hwmf1).set_index(spyf.index)

# fig18, ax = plt.subplots()
# ax.plot(spyt, label='spyt')
# ax.plot(spyf, label='spyf')
# ax.plot(hwmf1, label='hwmf1')
# plt.legend(loc='upper left')
# plt.title('Holt-Winters Multiplicative ETS(A,A,M) Method 1')
# plt.ylabel('Price')
# plt.xlabel('Date')
# plt.show()

# 4.8.2. One-Step Forecast without Re-Estimation
# hwmf2 = ets.ExponentialSmoothing(spyf1.iloc[:, 0].values, trend='additive', damped=False,
#                                  seasonal='multiplicative',
#                                  seasonal_periods=22).fit(smoothing_level=hwmt1.params['smoothing_level'],
#                                                           smoothing_slope=hwmt1.params['smoothing_slope'],
#                                                           smoothing_seasonal=hwmt1.params['smoothing_seasonal'])
# hwmf2 = hwmf2.predict(start=1, end=len(spyf))
# hwmf2 = pd.DataFrame(hwmf2).set_index(spyf.index)

# fig19, ax = plt.subplots()
# ax.plot(spyt, label='spyt')
# ax.plot(spyf, label='spyf')
# ax.plot(hwmf2, label='hwmf2', linestyle=':')
# plt.legend(loc='upper left')
# plt.title('Holt-Winters Multiplicative ETS(A,A,M) Method 2')
# plt.ylabel('Price')
# plt.xlabel('Date')
# plt.show()

##########################################

# 4.9. Exponential Smoothing Method Selection

brownaict1 = brownt1.aic
brownbict1 = brownt1.bic
holtaict1 = holtt1.aic
holtbict1 = holtt1.bic
expaict1 = expt1.aic
expbict1 = expt1.bic
# gardneraict1 = gardnert1.aic
# gardnerbict1 = gardnert1.bic
# tayloraict1 = taylort1.aic
# taylorbict1 = taylort1.bic
# hwaaict1 = hwat1.aic
# hwabict1 = hwat1.bic
# hwmaict1 = hwmt1.aic
# hwmbict1 = hwmt1.bic

msdata1 = [{'0': '', '1': 'AIC', '2': 'BIC'},
        {'0': 'Brown Simple Exponential Smoothing ETS(A,N,N) Method 1', '1': np.round(brownaict1, 4),
         '2': np.round(brownbict1, 4)},
        {'0': 'Holt Linear Trend ETS(A,A,N) Method 1', '1': np.round(holtaict1, 4),
         '2': np.round(holtbict1, 4)},
        {'0': 'Exponential Trend ETS(A,M,N) Method 1', '1': np.round(expaict1, 4),
         '2': np.round(expbict1, 4)},
        # {'0': 'Gardner Additive Damped Trend ETS(A,Ad,N) Method 1', '1': np.round(gardneraict1, 4),
        #  '2': np.round(gardnerbict1, 4)},
        # {'0': 'Taylor Multiplicative Damped Trend ETS(A,Md,N) Method 1', '1': np.round(tayloraict1, 4),
        #  '2': np.round(taylorbict1, 4)},
        # {'0': 'Holt-Winters Additive ETS(A,A,A) Method 1', '1': np.round(hwaaict1, 4),
        #  '2': np.round(hwabict1, 4)},
        # {'0': 'Holt-Winters Multiplicative ETS(A,A,M) Method 1', '1': np.round(hwmaict1, 4),
        #  '2': np.round(hwmbict1, 4)},
           ]
mstable1 = pd.DataFrame(msdata1)
print('')
print('== Exponential Smoothing Method Selection ==')
print('')
print(mstable1)
print('')

##########################################

# 4.10. Methods Forecasting Accuracy

# 4.10.1. Multi-Steps Forecast
rwdmae1 = fa.meanabs(rwdf1, spyf)
rwdrmse1 = fa.rmse(rwdf1, spyf)
smamae1 = fa.meanabs(smaf1, spyf)
smarmse1 = fa.rmse(smaf1, spyf)
brownmae1 = fa.meanabs(brownf1, spyf)
brownrmse1 = fa.rmse(brownf1, spyf)
holtmae1 = fa.meanabs(holtf1, spyf)
holtrmse1 = fa.rmse(holtf1, spyf)
expmae1 = fa.meanabs(expf1, spyf)
exprmse1 = fa.rmse(expf1, spyf)
# gardnermae1 = fa.meanabs(gardnerf1, spyf)
# gardnerrmse1 = fa.rmse(gardnerf1, spyf)
# taylormae1 = fa.meanabs(taylorf1, spyf)
# taylorrmse1 = fa.rmse(taylorf1, spyf)
# hwamae1 = fa.meanabs(hwaf1, spyf)
# hwarmse1 = fa.rmse(hwaf1, spyf)
# hwmmae1 = fa.meanabs(hwmf1, spyf)
# hwmrmse1 = fa.rmse(hwmf1, spyf)

fadata1 = [{'0': '', '1': 'MAE', '2': 'RMSE'},
        {'0': 'Random Walk with Drift Method 1', '1': np.round(rwdmae1, 4),
         '2': np.round(rwdrmse1, 4)},
        {'0': 'Simple Moving Average SMA Method 1', '1': np.round(smamae1, 4),
         '2': np.round(smarmse1, 4)},
        {'0': 'Brown Simple Exponential Smoothing ETS(A,N,N) Method 1', '1': np.round(brownmae1, 4),
         '2': np.round(brownrmse1, 4)},
        {'0': 'Holt Linear Trend ETS(A,A,N) Method 1', '1': np.round(holtmae1, 4),
         '2': np.round(holtrmse1, 4)},
        {'0': 'Exponential Trend ETS(A,M,N) Method 1', '1': np.round(expmae1, 4),
         '2': np.round(exprmse1, 4)},
        # {'0': 'Gardner Additive Damped Trend ETS(A,Ad,N) Method 1', '1': np.round(gardnermae1, 4),
        #  '2': np.round(gardnerrmse1, 4)},
        # {'0': 'Taylor Multiplicative Damped Trend ETS(A,Md,N) Method 1', '1': np.round(taylormae1, 4),
        #  '2': np.round(taylorrmse1, 4)},
        # {'0': 'Holt-Winters Additive ETS(A,A,A) Method 1', '1': np.round(hwamae1, 4),
        #  '2': np.round(hwarmse1, 4)},
        # {'0': 'Holt-Winters Multiplicative ETS(A,A,M) Method 1', '1': np.round(hwmmae1, 4),
        #  '2': np.round(hwmrmse1, 4)},
           ]
fatable1 = pd.DataFrame(fadata1)
print('')
print('== Multi-Steps Forecasting Accuracy ==')
print('')
print(fatable1)
print('')

# 4.7.2. One-Step Forecast without Re-Estimation
rwdmae2 = fa.meanabs(rwdf2, spyf)
rwdrmse2 = fa.rmse(rwdf2, spyf)
smamae2 = fa.meanabs(smaf2, spyf)
smarmse2 = fa.rmse(smaf2, spyf)
brownmae2 = fa.meanabs(brownf2, spyf)
brownrmse2 = fa.rmse(brownf2, spyf)
holtmae2 = fa.meanabs(holtf2, spyf)
holtrmse2 = fa.rmse(holtf2, spyf)
expmae2 = fa.meanabs(expf2, spyf)
exprmse2 = fa.rmse(expf2, spyf)
# gardnermae2 = fa.meanabs(gardnerf2, spyf)
# gardnerrmse2 = fa.rmse(gardnerf2, spyf)
# taylormae2 = fa.meanabs(taylorf2, spyf)
# taylorrmse2 = fa.rmse(taylorf2, spyf)
# hwamae2 = fa.meanabs(hwaf2, spyf)
# hwarmse2 = fa.rmse(hwaf2, spyf)
# hwmmae2 = fa.meanabs(hwmf2, spyf)
# hwmrmse2 = fa.rmse(hwmf2, spyf)

fadata2 = [{'0': '', '1': 'MAE', '2': 'RMSE'},
        {'0': 'Random Walk with Drift Method 2', '1': np.round(rwdmae2, 4),
         '2': np.round(rwdrmse2, 4)},
        {'0': 'Simple Moving Average SMA Method 2', '1': np.round(smamae2, 4),
         '2': np.round(smarmse2, 4)},
        {'0': 'Brown Simple Exponential Smoothing ETS(A,N,N) Method 2', '1': np.round(brownmae2, 4),
         '2': np.round(brownrmse2, 4)},
        {'0': 'Holt Linear Trend ETS(A,A,N) Method 2', '1': np.round(holtmae2, 4),
         '2': np.round(holtrmse2, 4)},
        {'0': 'Exponential Trend ETS(A,M,N) Method 2', '1': np.round(expmae2, 4),
         '2': np.round(exprmse2, 4)},
        # {'0': 'Gardner Additive Damped Trend ETS(A,Ad,N) Method 2', '1': np.round(gardnermae2, 4),
        #  '2': np.round(gardnerrmse2, 4)},
        # {'0': 'Taylor Multiplicative Damped Trend ETS(A,Md,N) Method 2', '1': np.round(taylormae2, 4),
        #  '2': np.round(taylorrmse2, 4)},
        # {'0': 'Holt-Winters Additive ETS(A,A,A) Method 2', '1': np.round(hwamae2, 4),
        #  '2': np.round(hwarmse2, 4)},
        # {'0': 'Holt-Winters Multiplicative ETS(A,A,M) Method 2', '1': np.round(hwmmae2, 4),
        #  '2': np.round(hwmrmse2, 4)},
           ]
fatable2 = pd.DataFrame(fadata2)
print('')
print('== One-Step without Re-Estimation Forecasting Accuracy ==')
print('')
print(fatable2)
print('')