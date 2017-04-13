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
feature_SEA = "/Users/ramanathan/Google Drive/Arizona State University/Spring 2017/Statistical Machine Learning/Project/code/WeatherForecastingSML/dataset/feature_SEA.csv"
series = read_csv(feature_SEA, header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)

ts = series['12']

X = ts

train, test = X[1:len(X) - 7], X[len(X) - 7:]
# train autoregression
model = AR(train)
model_fit = model.fit()
window = model_fit.k_ar
coef = model_fit.params
# walk forward over time steps in test
history = train[len(train) - window:]
history = [history[i] for i in range(len(history))]
predictions = list()
for t in range(len(test)):
    length = len(history)
    lag = [history[i] for i in range(length - window, length)]
    yhat = coef[0]
    for d in range(window):
        yhat += coef[d + 1] * lag[window - d - 1]
    obs = test[t]
    predictions.append(yhat)
    history.append(obs)
    print('predicted=%f, expected=%f' % (yhat, obs))
error = mean_squared_error(test, predictions)
print('Test MSE: %.3f' % error)
# plot
pyplot.plot(test)
pyplot.plot(predictions, color='red')
pyplot.show()

# Reference : http://machinelearningmastery.com/autoregression-models-time-series-forecasting-python/
