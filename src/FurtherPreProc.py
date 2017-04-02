import pandas as pd
import numpy as np
import math
weathertypecolname='HOURLYPRESENTWEATHERTYPE'
skyconditionscolname='HOURLYSKYCONDITIONS'

def makeTransformation(value,typelis,colname):
	if pd.isnull(value)==True:
		return -1
	#missing values assign -1
	maximum=-1
	countmax=0
	for index in xrange(len(typelis)):
		string=typelis[index]
		count=value.count(string)
		if count>countmax:
			countmax=count
			maximum=index
	return maximum

def changeSkyCondition(value):
	global skyconditionscolname
	typelis=['CLR','FEW','SCT','BKN','OVC']
	return makeTransformation(value,typelis,skyconditionscolname)

def changeWeatherType(value):
	
	typelis=['RA:','BR:','SQ:','TS:','DZ:','FG:','SN:','BC:','FU:','HZ:','GR:','TS','HAIL:','FZ','SH:','GS:','PL:','MI:','UP:']
	global weathertypecolname
	return makeTransformation(value,typelis,weathertypecolname)
	
url="../dataset/processed_SEA.csv"
df=pd.read_csv(url,header=0)
print df.shape
df[weathertypecolname]=df[weathertypecolname].apply(changeWeatherType)
df[skyconditionscolname]=df[skyconditionscolname].apply(changeSkyCondition)

print df.shape
df.to_csv('../dataset/preprocessed_SEA.csv',header=True,index=False)
