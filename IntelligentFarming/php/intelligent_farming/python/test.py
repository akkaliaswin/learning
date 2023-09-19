import os
import pickle
import pandas as pd

data=pd.read_csv('test.csv')
#print(data.head(4))

out=''
path='models'
for i in os.listdir(path):
	model=pickle.load(open(path+'/'+i, 'rb'))
	pred=model.predict(data)
	label=i.split('_')
	label=label[1].split('.')
	for p in pred:
		if p == -1:
			#print('not suitable for {}'.format(label[0]))
			pass
		else:
			#print(label[0])
			out = out+label[0]+'#'
print(out)			
	#print('##################################################################')










