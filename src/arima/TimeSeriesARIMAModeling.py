from pandas import read_csv
from pandas import datetime
from pandas import DataFrame
from matplotlib import pyplot
from sklearn.metrics import mean_squared_error
from matplotlib.pylab import rcParams
from statsmodels.tsa.arima_model import ARMA

rcParams['figure.figsize'] = 15, 6

def parser(dates):
	return datetime.strptime(dates, '%Y-%m-%d %H')

# read the data in
feature_SEA = "/Users/ramanathan/Google Drive/Arizona State University/Spring 2017/Statistical Machine Learning/Project/code/WeatherForecastingSML/dataset/feature_SEA.csv"
series = read_csv(feature_SEA, header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)


ts = series['12']

X = DataFrame(ts.values)
"""dataframe = concat([values.shift(1), values], axis=1)
dataframe.columns = ['t-1', 't+1']
print(dataframe.head(5))


# split into train and test sets
X = dataframe.values
train_size = int(len(X) * 0.66)
train, test = X[1:train_size], X[train_size:]
train_X, train_y = train[:,0], train[:,1]
test_X, test_y = test[:,0], test[:,1]
"""


size = int(len(X) * 0.66)
train, test = X[0:size], X[size:len(X)]
history = [x for x in train]
predictions = list()
for t in range(len(test)):
    model = ARMA(history, order=(10,0,0))
    model_fit = model.fit(disp=0)
    output = model_fit.forecast()
    yhat = output[0]
    predictions.append(yhat)
    obs = test[t]
    history.append(obs)
    print('predicted=%f, expected=%f' % (yhat, obs))

error = mean_squared_error(test, predictions)
print('Test MSE: %.3f' % error)
# plot
pyplot.plot(test)
pyplot.plot(predictions, color='red')
pyplot.show()

# Reference : http://machinelearningmastery.com/arima-for-time-series-forecasting-with-python/