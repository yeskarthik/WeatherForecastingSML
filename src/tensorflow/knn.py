
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import requests
import pandas as pd
from tensorflow.python.framework import ops
ops.reset_default_graph()



sess = tf.Session()

datasets=['SEA','DAL','CIN']
for input in datasets:

	url="../../dataset/featureselection/"+input+".csv"
	dataframe=pd.read_csv(url,header=0)
	print (dataframe.shape)
	dataframe=dataframe.fillna(dataframe.mean())
	array=dataframe.values
	Y=dataframe[['HOURLYDRYBULBTEMPF']].values
	X=np.concatenate((array[:,1:5],array[:,5:12]),axis=1)
	
	testseturl="../../dataset/test"+input+".csv"
	testdataframe=pd.read_csv(testseturl)
	testrows=len(testdataframe.index)

	train_indices = np.array(range(0, len(dataframe.index)-testrows))
	test_indices = np.array(range(len(dataframe.index)-testrows, len(X)))
	x_vals_train = X[train_indices]
	x_vals_test = testdataframe.values
	y_vals_train = Y[train_indices]
	y_vals_test = Y[test_indices]
	num_features = 11

	k = 4
	batch_size=len(x_vals_test)

	# Placeholders
	x_data_train = tf.placeholder(shape=[None, num_features], dtype=tf.float32)
	x_data_test = tf.placeholder(shape=[None, num_features], dtype=tf.float32)
	y_target_train = tf.placeholder(shape=[None, 1], dtype=tf.float32)
	y_target_test = tf.placeholder(shape=[None, 1], dtype=tf.float32)



	distance = tf.reduce_sum(tf.abs(tf.subtract(x_data_train, tf.expand_dims(x_data_test,1))), axis=2)


	top_k_xvals, top_k_indices = tf.nn.top_k(tf.negative(distance), k=k)
	x_sums = tf.expand_dims(tf.reduce_sum(top_k_xvals, 1),1)
	x_sums_repeated = tf.matmul(x_sums,tf.ones([1, k], tf.float32))
	x_val_weights = tf.expand_dims(tf.div(top_k_xvals,x_sums_repeated), 1)

	top_k_yvals = tf.gather(y_target_train, top_k_indices)
	#prediction = tf.squeeze(tf.matmul(x_val_weights,top_k_yvals), axis=[1])
	prediction = tf.reduce_mean(top_k_yvals, 1)

	# Calculate MSE
	mse = tf.div(tf.reduce_sum(tf.square(tf.subtract(prediction, y_target_test))), batch_size)

	# Calculate how many loops over training data
	num_loops = int(np.ceil(len(x_vals_test)/batch_size))

	for i in range(num_loops):
	    min_index = i*batch_size
	    max_index = min((i+1)*batch_size,len(x_vals_train))
	    x_batch = x_vals_test[min_index:max_index]
	    y_batch = y_vals_test[min_index:max_index]
	    predictions = sess.run(prediction, feed_dict={x_data_train: x_vals_train, x_data_test: x_batch,
	                                         y_target_train: y_vals_train, y_target_test: y_batch})
	    batch_mse = sess.run(mse, feed_dict={x_data_train: x_vals_train, x_data_test: x_batch,
	                                         y_target_train: y_vals_train, y_target_test: y_batch})

	    print('Batch #' + str(i+1) + ' MSE: ' + str(np.round(batch_mse,3)))


	# In[70]:

	get_ipython().magic('matplotlib inline')
	# Plot prediction and actual distribution
	bins = np.linspace(10, 110, 200)

	print (predictions)
	print (y_batch)

	print ([predictions[i]-y_batch[i] for i in xrange(len(y_batch))])
	plt.hist(predictions, bins, alpha=0.5, label='Prediction')
	plt.hist(y_batch, bins, alpha=0.5, label='Actual')
	plt.title('Histogram of Predicted and Actual Values')
	plt.xlabel('Med Home Value in $1,000s')
	plt.ylabel('Frequency')
	plt.legend(loc='upper right')
	plt.show()




