from pandas import read_csv
from pandas import datetime
from pandas import DataFrame
from matplotlib import pyplot
from sklearn.metrics import mean_squared_error
from matplotlib.pylab import rcParams
from statsmodels.tsa.ar_model import AR

rcParams['figure.figsize'] = 15, 6


def parser(dates):
    return datetime.strptime(dates, '%Y-%m-%d %H')


# read the data in
feature_SEA = "/Users/ramanathan/Google Drive/Arizona State University/Spring 2017/Statistical Machine Learning/Project/code/WeatherForecastingSML/dataset/feature_CIN.csv"
series = read_csv(feature_SEA, header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)

ts = series['12']

X = ts
number_of_days = 365
number_of_hours = 365 * 24

train, test = X[1:len(X) - number_of_hours], X[len(X) - number_of_hours:]

# train autoregression
model = AR(train)
model_fit = model.fit()

window = model_fit.k_ar
coefficient = model_fit.params
print('Lag: %s' % model_fit.k_ar)
print('Coefficients: %s' % model_fit.params)

# walk forward over time steps in test
history = train[len(train) - window:]
history = [history[i] for i in range(len(history))]
predictions = list()
for t in range(len(test)):
    length = len(history)
    lag = [history[i] for i in range(length - window, length)]
    yhat = coefficient[0]
    for d in range(window):
        yhat += coefficient[d + 1] * lag[window - d - 1]
    obs = test[t]
    predictions.append(yhat)
    history.append(obs)
    print('predicted=%f, expected=%f' % (yhat, obs))
error = mean_squared_error(test, predictions)
print('Test MSE: %.3f' % error)

# plot
pyplot.plot(test)
title = 'AR Model Predictions for Cincinnati'
pyplot.title(title)
pyplot.xlabel('Hourly Test Data Points (2016)')
pyplot.ylabel('Temperature (F)')
pyplot.plot(predictions, color='#ef9058')
pyplot.show()

# Reference : http://machinelearningmastery.com/autoregression-models-time-series-forecasting-python/
