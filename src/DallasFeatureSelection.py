import pandas as pd
import numpy as np
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
url="../dataset/preprocessed_DAL.csv"
dataframe=pd.read_csv(url,header=0)
'''
As of now, filling each missing value with the mean of that column
'''
dataframe=dataframe.fillna(dataframe.mean())
array=dataframe.values
# X is the input to PCA, I'm removing the time stamp from this data, that is still a String and important. We know that it can't be discarded
X=np.concatenate((array[:,1:5],array[:,6:]),axis=1)
#Note: Scikit learn requires numpy array and not pandas dataframe
Y=dataframe[['HOURLYDRYBULBTEMPF']].values
pca = PCA()
fit =pca.fit(X)
print fit.explained_variance_
'''
[  3.32350338e+04   1.00729458e+04   5.36675729e+02   2.78253257e+02
   8.08248068e+01   1.78692997e+01   4.39899659e+00   3.43776104e+00
   2.01547591e+00   1.91792445e+00   1.45561406e+00   5.53005136e-01
   1.03327979e-01   2.67802658e-04   2.46111672e-04   6.74019287e-05
   1.18655102e-05]
this is the variance of the fit . Note that they are in the sorted order . The value after e is positive for 11 terms., so in the next step
i'll be selecting the number of components to be 11

   '''
pca.n_components = 11
# Applies pca and also does dimensionality reduction so the original values will be modified right now 
X_reduced=pca.fit_transform(X)

'''
Now I have to concatenate date column, dry bulb temperature to the reduced and modified X
'''
X1=np.concatenate((array[:,:1],X_reduced),axis=1)

X2=np.concatenate((X1,Y),axis=1)
df=pd.DataFrame(X2)
df.to_csv('../dataset/feature_DAL.csv',header=True,index=False)
