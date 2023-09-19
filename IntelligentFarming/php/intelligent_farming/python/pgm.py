import os
import pickle
import pandas as pd
from sklearn.ensemble import IsolationForest

path='dataset'
for i in os.listdir(path):
	#print(i)
	data=pd.read_csv(path+'/'+i)
	#print(data.head(5))
	data.dropna(inplace=True)
	X=data[['season','pH','soil_type','elevation','temperature']]
	#print(X.head(5))
	clf = IsolationForest(n_estimators=100, contamination=0.0,random_state=42)
	clf.fit(X)
	name=i.split('.')
	pickle.dump(clf, open('models/model_'+name[0]+'.sav', 'wb'))

