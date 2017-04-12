from pandas import read_csv
from pandas import datetime
from pandas import DataFrame
from matplotlib import pyplot
from matplotlib.pylab import rcParams
from pandas.tools.plotting import autocorrelation_plot
import itertools

rcParams['figure.figsize'] = 15, 6

def parser(dates):
	return datetime.strptime(dates, '%Y-%m-%d %H')

# read the data in
feature_SEA = "/Users/ramanathan/Google Drive/Arizona State University/Spring 2017/Statistical Machine Learning/Project/code/WeatherForecastingSML/dataset/feature_SEA.csv"
series = read_csv(feature_SEA, header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)


ts = series['12']

X = DataFrame(ts.values)
print X.head()
X.plot()
pyplot.show()


# ========================== X represents the temperature as a time series


# Define the p, d and q parameters to take any value between 0 and 2
p = d = q = range(0, 2)

# Generate all different combinations of p, q and q triplets
pdq = list(itertools.product(p, d, q))

# Generate all different combinations of seasonal p, q and q triplets
seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]
print seasonal_pdq


autocorrelation_plot(X)
pyplot.show()

# Reference : http://machinelearningmastery.com/arima-for-time-series-forecasting-with-python/