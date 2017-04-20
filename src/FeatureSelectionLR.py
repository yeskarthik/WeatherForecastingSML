import pandas as pd
import numpy as np
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn import preprocessing
url="../dataset/preprocessed_SEA.csv"
dataframe=pd.read_csv(url,header=0)
'''
As of now, filling each missing value with the mean of that column
'''
dataframe=dataframe.fillna(dataframe.mean())
array=dataframe.values
# X is the input to PCA, I'm removing the time stamp from this data, that is still a String and important. We know that it can't be discarded
X=np.concatenate((array[:,1:5],array[:,6:]),axis=1)
lab_enc = preprocessing.LabelEncoder()

#Note: Scikit learn requires numpy array and not pandas dataframe
Y=dataframe[['HOURLYDRYBULBTEMPF']].values
Y=Y.ravel()
encoded = lab_enc.fit_transform(Y)
print encoded
#encoded = encoded.ravel()
model=LogisticRegression()
rfe=RFE(model,11)
rfe=rfe.fit(X,encoded)
print(rfe.support_)
print(rfe.ranking_)

'''df=pd.DataFrame(X2)
df.to_csv('../dataset/feature_SEA.csv',header=True,index=False)'''
