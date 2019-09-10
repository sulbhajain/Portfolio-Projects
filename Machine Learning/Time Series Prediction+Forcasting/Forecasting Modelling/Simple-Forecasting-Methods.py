##############################################################################
# Forecasting Models with Python                                             #
# Simple Forecasting Methods                                                 #
# (c) Diego Fernandez Garcia 2015-2018                                       #
# www.exfinsis.com                                                           #
##############################################################################

# 1. Packages Importing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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

# 3. Simple Forecasting Methods

# 3.1. Arithmetic Mean Method

# 3.1.1. Multi-Steps Forecast
meanf1 = pd.concat([spyt.mean()]*len(spyf))
meanf1 = pd.DataFrame(meanf1).set_index(spyf.index)

fig2, ax = plt.subplots()
ax.plot(spyt, label='spyt')
ax.plot(spyf, label='spyf')
ax.plot(meanf1, label='meanf1')
plt.legend(loc='upper left')
plt.title('Arithmetic Mean Method 1')
plt.ylabel('Price')
plt.xlabel('Date')
plt.show()

##########################################

# 3.2. Random Walk Method

# 3.2.1. Multi-Steps Forecast
rwf1 = pd.concat([spyt.tail(1)]*len(spyf))
rwf1 = pd.DataFrame(rwf1).set_index(spyf.index)

fig4, ax = plt.subplots()
ax.plot(spyt, label='spyt')
ax.plot(spyf, label='spyf')
ax.plot(rwf1, label='rwf1')
plt.legend(loc='upper left')
plt.title('Random Walk Method 1')
plt.ylabel('Price')
plt.xlabel('Date')
plt.show()

# 3.2.2. One-Step Forecast without Re-Estimation
rwf2 = spy.shift(1)['2014-01-02':]

fig5, ax = plt.subplots()
ax.plot(spyt, label='spyt')
ax.plot(spyf, label='spyf')
ax.plot(rwf2, label='rwf2', linestyle=':')
plt.legend(loc='upper left')
plt.title('Random Walk Method 2')
plt.ylabel('Price')
plt.xlabel('Date')
plt.show()

##########################################

# 3.3. Seasonal Random Walk Method

# 3.3.1. Multi-Steps Forecast
srwf1 = pd.concat([spyt.tail(22)]*(np.round((len(spyf)/22), 0).astype(int)))
srwf1 = srwf1.head(len(spyf))
srwf1 = pd.DataFrame(srwf1).set_index(spyf.index)

fig6, ax = plt.subplots()
ax.plot(spyt, label='spyt')
ax.plot(spyf, label='spyf')
ax.plot(srwf1, label='srwf1')
plt.legend(loc='upper left')
plt.title('Seasonal Random Walk Method 1')
plt.ylabel('Price')
plt.xlabel('Date')
plt.show()

# 3.3.2. One-Step Forecast without Re-Estimation
srwf2 = spy.shift(22)['2014-01-02':]

fig7, ax = plt.subplots()
ax.plot(spyt, label='spyt')
ax.plot(spyf, label='spyf')
ax.plot(srwf2, label='srwf2', linestyle=':')
plt.legend(loc='upper left')
plt.title('Seasonal Random Walk Method 2')
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

fig8, ax = plt.subplots()
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

fig9, ax = plt.subplots()
ax.plot(spyt, label='spyt')
ax.plot(spyf, label='spyf')
ax.plot(rwdf2, label='rwdf2', linestyle=':')
plt.legend(loc='upper left')
plt.title('Random Walk with Drift Method 2')
plt.ylabel('Price')
plt.xlabel('Date')
plt.show()

##########################################

# 3.5. Methods Forecasting Accuracy

# 3.5.1. Multi-Steps Forecast
meanmae1 = fa.meanabs(meanf1, spyf)
meanrmse1 = fa.rmse(meanf1, spyf)
rwmae1 = fa.meanabs(rwf1, spyf)
rwrmse1 = fa.rmse(rwf1, spyf)
srwmae1 = fa.meanabs(srwf1, spyf)
srwrmse1 = fa.rmse(srwf1, spyf)
rwdmae1 = fa.meanabs(rwdf1, spyf)
rwdrmse1 = fa.rmse(rwdf1, spyf)

fadata1 = [{'0': '', '1': 'MAE', '2': 'RMSE'},
        {'0': 'Arithmetic Mean Method 1', '1': np.round(meanmae1, 4), '2': np.round(meanrmse1, 4)},
        {'0': 'Random Walk Method 1', '1': np.round(rwmae1, 4), '2': np.round(rwrmse1, 4)},
        {'0': 'Seasonal Random Walk Method 1', '1': np.round(srwmae1, 4), '2': np.round(srwrmse1, 4)},
        {'0': 'Random Walk with Drift Method 1', '1': np.round(rwdmae1, 4), '2': np.round(rwdrmse1, 4)}]
fatable1 = pd.DataFrame(fadata1)
print('')
print('== Multi-Steps Forecasting Accuracy ==')
print('')
print(fatable1)
print('')

# 3.5.2. One-Step Forecast without Re-Estimation
rwmae2 = fa.meanabs(rwf2, spyf)
rwrmse2 = fa.rmse(rwf2, spyf)
srwmae2 = fa.meanabs(srwf2, spyf)
srwrmse2 = fa.rmse(srwf2, spyf)
rwdmae2 = fa.meanabs(rwdf2, spyf)
rwdrmse2 = fa.rmse(rwdf2, spyf)

fadata2 = [{'0': '', '1': 'MAE', '2': 'RMSE'},
        {'0': 'Random Walk Method 2', '1': np.round(rwmae2, 4), '2': np.round(rwrmse2, 4)},
        {'0': 'Seasonal Random Walk Method 2', '1': np.round(srwmae2, 4), '2': np.round(srwrmse2, 2)},
        {'0': 'Random Walk with Drift Method 2', '1': np.round(rwdmae2, 4), '2': np.round(rwdrmse2, 4)}]
fatable2 = pd.DataFrame(fadata2)
print('')
print('== One-Step without Re-Estimation Forecasting Accuracy ==')
print('')
print(fatable2)
print('')