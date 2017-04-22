import numpy as np
import matplotlib.pyplot as plt
from pandas import read_csv
from pandas import datetime
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back), 0]
		dataX.append(a)
		dataY.append(dataset[i + look_back, 0])
	return np.array(dataX), np.array(dataY)

np.random.seed(7)
datasets=['SEA','CIN','DAL']
for input in datasets:
	url='../dataset/preprocessed_'+input+'.csv'
	dataframe=read_csv(url,header=0)
	print (dataframe.shape)
	dataframe=dataframe.fillna(dataframe.mean())
	array=dataframe.values
	#removing the output class label column
	X=np.concatenate((array[:,1:5],array[:,6:]),axis=1)
	#filtering only those features got from the feature selection
	X=np.concatenate((X[:,2:4],X[:,5:6],X[:,7:14],X[:,16:]),axis=1)
	result=None

	for i in xrange(0,11):
		dataset=X[:,i:i+1]
		dataset = dataset.astype('float32')
		# normalize the dataset
		scaler = MinMaxScaler(feature_range=(0, 1))
		dataset = scaler.fit_transform(dataset)
		# split into train and test sets
		train_size = int(len(dataset) * 0.67)
		test_size = len(dataset) - train_size
		train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
		# reshape into X=t and Y=t+1
		look_back = 1
		trainX, trainY = create_dataset(train, look_back)
		testX, testY = create_dataset(test, look_back)
		# reshape input to be [samples, time steps, features]
		trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
		testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
		# create and fit the LSTM network
		model = Sequential()
		model.add(LSTM(4, input_shape=(1, look_back)))
		model.add(Dense(1))
		model.compile(loss='mean_squared_error', optimizer='adam')
		model.fit(trainX, trainY, epochs=12, batch_size=1, verbose=2)
		testPredict = model.predict(testX)
		testPredict = scaler.inverse_transform(testPredict)
		testY = scaler.inverse_transform([testY])
		print type(testPredict)
		print testPredict.shape
		if result is None:
			result=testPredict
		else:
			result=np.concatenate((result[:,:],testPredict[:,:]),axis=1)
		print result.shape
	writeUrl='../dataset/test'+input+'.csv'
	np.savetxt(writeUrl,result,delimiter=',')


