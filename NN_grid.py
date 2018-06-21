import numpy as np
from sklearn.neural_network import MLPClassifier
import pandas as pd
from time import time
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split



mnist = pd.read_csv('2015EE10466.csv')



mnist = np.array(mnist)

train_x,test_x,train_y,test_y = train_test_split(mnist[0:3000,0:784],mnist[0:3000,784:],train_size=0.9,random_state=42)


print(test_x.shape, test_y.shape)
print(train_x.shape, train_y.shape)


#uncomment this for gridsearch
#kf = StratifiedKFold(n_splits=4)
#hidden_layer_sizes_range:[10,10],[10,10,10],[10,10,10,10],[50,50],,[1500,1500,1500],[1000,1000,1000]
# learning_rate_init_range = [0.0001,0.001,0.01]

hidden_layer_sizes_range = [(35,30)]
activation_range = ['relu']
alpha_range=[0.01,0.1,1,10,50,500]
learning_rate_init_range = [0.0000001,0.000001,0.00001,0.0001,0.001,0.001,0.1,1]
param_grid = dict(hidden_layer_sizes=hidden_layer_sizes_range,activation = activation_range,alpha=alpha_range,
                  learning_rate_init=learning_rate_init_range,max_iter=[100],solver=['adam'])
clf = GridSearchCV(MLPClassifier(),param_grid=param_grid,cv=StratifiedKFold(n_splits=4))
clf.fit(train_x,train_y.ravel())
pred=clf.predict(test_x)
print('train accuracy',clf.score(train_x,train_y.ravel()))
print('test accuracy', clf.score(test_x,test_y.ravel()))
print(clf.best_estimator_)



#BEST FIT
#hidden layer = [250,200,150]
#clf=MLPClassifier(hidden_layer_sizes=(250,200,150),activation='relu',alpha = 50, solver='adam',learning_rate='constant',
#learning_rate_init=1e-05,max_iter=2500)=0.9333



