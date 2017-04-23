import numpy as np
from pandas import read_csv
import csv
datasets=['SEA','CIN','DAL']
for input in datasets:
	url='../dataset/preprocessed_'+input+'.csv'
	dataframe=read_csv(url,header=0)
	headers=dataframe.columns.values.tolist()
	
	outputY=headers[5]
	outputDate=headers[0]
	del headers[5]
	del headers[0]
	removeArray=[False,False,True,True,False,True,False,True,True,True,True,True,True,True,False,False,True]
	removeArray=removeArray[::-1]
	for index in xrange(len(removeArray)):
		element=removeArray[index]
		displacement=len(removeArray)-1-index
		if element==False:
			del headers[displacement]
	headers.insert(0,outputDate)
	headers.append(outputY)
	dataframe=dataframe.fillna(dataframe.mean())
	array=dataframe.values
	datecolumn=dataframe[[outputDate]].values
	Y=dataframe[[outputY]].values
	#removing the output class label column
	X=np.concatenate((array[:,1:5],array[:,6:]),axis=1)
	#filtering only those features got from the feature selection
	X=np.concatenate((X[:,2:4],X[:,5:6],X[:,7:14],X[:,16:]),axis=1)
	csvoutput=np.concatenate((datecolumn[:,:],X[:,:],Y[:,:]),axis=1)
	csvoutput=csvoutput.tolist()
	csvoutput.insert(0,headers)
	writeUrl='../dataset/featureselection/'+input+'.csv'
	with open(writeUrl,"wb") as f:
		writer=csv.writer(f)
		writer.writerows(csvoutput)



