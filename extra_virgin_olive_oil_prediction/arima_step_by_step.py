# importing libraries
from sklearn import neighbors
from sklearn.model_selection import GridSearchCV
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.stattools import adfuller
scaler = MinMaxScaler(feature_range=(0, 1))
import matplotlib.pyplot as plt
from pmdarima.arima import auto_arima
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
from pandas.plotting import autocorrelation_plot
from statsmodels.tsa.arima_model import ARIMA

from statsmodels.tsa.stattools import acf, pacf
df = pd.read_csv("food_dataset.csv")
df.reset_index(drop=True, inplace=True)
dfevoo = df[df['product'].str.contains('χοιρ')]


dfevoo = dfevoo[dfevoo['country'] == 'greece']
dfevoo['priceStringDate'] = pd.to_datetime(dfevoo['priceStringDate'])
dfevoo = dfevoo.drop(columns=['price_id', 'product', 'priceDate', 'url', 'country', 'dataSource']).sort_values(
    by='priceStringDate')
dfevoo = pd.DataFrame(dfevoo)
dfevoo = dfevoo.groupby('priceStringDate').mean().reset_index()
train_perc = 0.9
dfevoo = dfevoo.set_index('priceStringDate')

data=dfevoo
print(data)
print(data.info())
plt.plot(data)
plt.show()

train = dfevoo[:int(len(dfevoo) * train_perc)]  # .sample(frac=0.8, random_state=0)
test = dfevoo.drop(train.index)
# data=dfevoo['price']




def test_stationarity(timeseries):
    #Determing rolling statistics
    rolmean = timeseries.rolling(window=4).mean()
    rolstd = timeseries.rolling(window=4).std()

    #Plot rolling statistics:
    orig = plt.plot(timeseries, color='blue',label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)
    # # Perform Dickey-Fuller test:
    # print ('Results of Dickey-Fuller Test:')
    # dftest = adfuller(timeseries, autolag='AIC')
    # dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    # for key, value in dftest[4].items():
    #     dfoutput['Critical Value (%s)' % key] = value
    # print(dfoutput)

test_stationarity(data)

def difference(dataset, interval=1):
    index = list(dataset.index)
    diff = list()
    for i in range(interval, len(dataset)):
        value = dataset["price"][i] - dataset["price"][i - interval]
        diff.append(value)
    return (diff)

diff = difference(data)
plt.plot(diff)
plt.show()

ts_log = np.log(data)
plt.title('Log of the data')
plt.plot(ts_log)
plt.show()

moving_avg = ts_log.rolling(4).mean()
plt.plot(ts_log)
plt.title('4 years of Moving average')
plt.plot(moving_avg, color='red')
plt.show()

ts_log_moving_avg_diff = ts_log - moving_avg
ts_log_moving_avg_diff.dropna(inplace=True)
test_stationarity(ts_log_moving_avg_diff)

expwighted_avg = ts_log.ewm(halflife=3).mean()
#parameter halflife is used to define the amount of exponential decay
plt.plot(ts_log)
plt.plot(expwighted_avg, color='red')
plt.show()

ts_log_ewma_diff = ts_log - expwighted_avg
test_stationarity(ts_log_ewma_diff)

ts_log_diff = ts_log - ts_log.shift()
plt.plot(ts_log_diff)
plt.show()
ts_log_diff.dropna(inplace=True)
test_stationarity(ts_log_diff)

decompfreq=360

decomposition = seasonal_decompose(ts_log, freq=decompfreq)

trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

plt.subplot(411)
plt.plot(ts_log, label='Original')
plt.legend(loc='best')
plt.subplot(412)
plt.plot(trend, label='Trend')
plt.legend(loc='best')
plt.subplot(413)
plt.plot(seasonal,label='Seasonality')
plt.legend(loc='best')
plt.subplot(414)
plt.plot(residual, label='Residuals')
plt.legend(loc='best')
plt.show()
plt.tight_layout()

ts_log_decompose = residual
ts_log_decompose.dropna(inplace=True)
test_stationarity(ts_log_decompose)

lag_acf = acf(ts_log_diff, nlags=20)
lag_pacf = pacf(ts_log_diff, nlags=20, method='ols')

plt.subplot(121)
plt.plot(lag_acf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
plt.show()
plt.title('Autocorrelation Function')

plt.subplot(122)
plt.plot(lag_pacf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
plt.title('Partial Autocorrelation Function')
plt.show()
plt.tight_layout()

model = ARIMA(ts_log, order=(1, 1, 0), freq=ts_log.index.inferred_freq)
results_AR = model.fit(disp=-1)
plt.plot(ts_log_diff)
plt.plot(results_AR.fittedvalues, color='red')
plt.show()

model = ARIMA(ts_log, order=(1, 1, 0), freq=ts_log.index.inferred_freq)
results_MA = model.fit(disp=-1)
plt.plot(ts_log_diff)
plt.plot(results_MA.fittedvalues, color='red')
plt.show()



model = ARIMA(ts_log, order=(1, 1, 1), freq=ts_log.index.inferred_freq)
results_ARIMA = model.fit(disp=-1)
plt.plot(ts_log_diff)
plt.plot(results_ARIMA.fittedvalues, color='red')
plt.show()








