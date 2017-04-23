from pandas import read_csv, Series
from pandas import datetime
from pandas import DataFrame
from matplotlib import pyplot
from matplotlib.pylab import rcParams
from pandas.tools.plotting import autocorrelation_plot, lag_plot
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

rcParams['figure.figsize'] = 15, 6


def parser(dates):
    return datetime.strptime(dates, '%Y-%m-%d %H')


# read the data in
feature_SEA = "/Users/ramanathan/Google Drive/Arizona State University/Spring 2017/Statistical Machine Learning/Project/code/WeatherForecastingSML/dataset/feature_CIN.csv"
series = read_csv(feature_SEA, header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)

ts = series['12']


X = DataFrame(ts.values)
print X.head()
X.plot()
pyplot.show()

lag_plot(series)
title = 'Auto Correlation for Cincinnati'
pyplot.title(title)
pyplot.show()

#plot_acf(X, lags=24)
#pyplot.show()

#plot_pacf(X, lags=24)
#pyplot.show()


