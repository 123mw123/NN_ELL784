import numpy as np
from sklearn.neural_network import MLPClassifier
import pandas as pd
from time import time
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt



mnist = pd.read_csv('2015EE10466.csv')



mnist = np.array(mnist)

train_x,test_x,train_y,test_y = train_test_split(mnist[0:3000,0:784],mnist[0:3000,784:],train_size=0.9,random_state=42)

print(test_x.shape, test_y.shape)
print(train_x.shape, train_y.shape)
print(test_y[0])



#BEST FIT
'''
clf=MLPClassifier(hidden_layer_sizes=(250,200,150),activation='relu',alpha = 50, solver='adam',learning_rate='constant',
	learning_rate_init=1e-05,max_iter=2500)
	'''
'''
clf=MLPClassifier(hidden_layer_sizes=(160,80),activation='relu',alpha = 100, solver='adam',learning_rate='constant',
	learning_rate_init=0.0001,max_iter=1000)
	'''
'''
#hidden layer = [250,200,150]
clf=MLPClassifier(hidden_layer_sizes=(80,50),activation='relu',alpha = 50, solver='adam',learning_rate='constant',
	learning_rate_init=0.0001,max_iter=200)
'''
clf=MLPClassifier(hidden_layer_sizes=(250,200,150),activation='relu',alpha = 50, solver='adam',learning_rate='constant',
	learning_rate_init=1e-05,max_iter=2500,random_state=1)


t0 =time()
clf.fit(train_x,train_y.ravel())
print(0.001,'learning_rate_init')
print('fiiting time',time()-t0)
t0 = time()
pred=clf.predict(test_x)

correct_x = []
correct_y =[]
wrong_x = []
wrong_y = []

print(test_x)
for i in range(300):
    if(pred[i] != test_y.ravel()[i]):
        wrong_x = wrong_x + [test_x[i]]
        wrong_y = wrong_y + [[test_y[i],pred[i]]]
    else:
        correct_x = correct_x + [test_x[i]]
        correct_y = correct_y + [test_y[i]]

print('predicting time',time()-t0)
print('train accuracy',clf.score(train_x,train_y.ravel()))
print('test accuracy', clf.score(test_x,test_y.ravel()))

print(clf.n_layers_)
print(clf.n_outputs_)
print("coefficients")
weight_matrix = clf.coefs_
print(len(weight_matrix))
print('intercepts')
print(clf.intercepts_)
print(len(weight_matrix))

print(weight_matrix[0].shape)


w1= MinMaxScaler(feature_range=(0,255)).fit_transform(weight_matrix[0])
w1 = np.reshape(w1,(250,784))
plt.imshow(w1,cmap='gray')
plt.show()

w2= MinMaxScaler(feature_range=(0,255)).fit_transform(weight_matrix[1])
w2 = np.reshape(w2,(200,250))
plt.imshow(w2,cmap='gray')
plt.show()

w3 = MinMaxScaler(feature_range=(0,255)).fit_transform(weight_matrix[2])
w3 = np.reshape(w3,(150,200))
plt.imshow(w3,cmap='gray')
plt.show()

w4 = MinMaxScaler(feature_range=(0,255)).fit_transform(weight_matrix[3])
w4 = np.reshape(w4,(10,150))
plt.imshow(w4,cmap='gray')
plt.show()



fig, axes = plt.subplots(4, 4)
# use global min / max to ensure all weights are shown on the same scale
vmin, vmax = clf.coefs_[0].min(), clf.coefs_[0].max()
for coef, ax in zip(clf.coefs_[0].T, axes.ravel()):
    #,  vmin=.5 * vmin vmax=.5 * vmax
    ax.matshow(coef.reshape(28, 28),cmap=plt.get_cmap('gray'))
    ax.set_xticks(())
    ax.set_yticks(())
print(0)
plt.show()

fig, axes = plt.subplots(4, 4)
for coef, ax in zip(clf.coefs_[1].T, axes.ravel()):
    #,  vmin=.5 * vmin vmax=.5 * vmax
    ax.matshow(coef.reshape(25, 10),cmap=plt.get_cmap('gray'))
    ax.set_xticks(())
    ax.set_yticks(())
print(1)
plt.show()

fig, axes = plt.subplots(4, 4)
for coef, ax in zip(clf.coefs_[2].T, axes.ravel()):
    #,  vmin=.5 * vmin vmax=.5 * vmax
    ax.matshow(coef.reshape(20, 10),cmap=plt.get_cmap('gray'))
    ax.set_xticks(())
    ax.set_yticks(())
print(12)
plt.show()


fig, axes = plt.subplots(2, 5)
for coef, ax in zip(clf.coefs_[3].T, axes.ravel()):
    #,  vmin=.5 * vmin vmax=.5 * vmax
    ax.matshow(coef.reshape(15, 10),cmap=plt.get_cmap('gray'))
    ax.set_xticks(())
    ax.set_yticks(())
print(123)
plt.show()

#correct classification visualization
img = np.reshape(correct_x[0],(28,28))
plt.imshow(img,cmap='gray')
plt.title(correct_y[0])
plt.show()

img = np.reshape(img,(784,1))
img = np.dot(w1,img)
img1 = np.reshape(img,(25,10))
plt.imshow(img1,cmap='gray')
plt.title(correct_y[0])
plt.show()

img = np.dot(w2,img)
img2 = np.reshape(img,(20,10))
plt.imshow(img2,cmap='gray')
plt.title(correct_y[0])
plt.show()

img = np.dot(w3,img)
img3 = np.reshape(img,(15,10))
plt.imshow(img3,cmap='gray')
plt.title(correct_y[0])
plt.show()

img = np.dot(w4,img)
img4 = np.reshape(img,(10,1))
plt.imshow(img4,cmap='gray')
plt.title(correct_y[0])
plt.show()


#wrong classification visualization
img = np.reshape(wrong_x[0],(28,28))
plt.imshow(img,cmap='gray')
plt.title(wrong_y[0])
plt.show()

img = np.reshape(img,(784,1))
img = np.dot(w1,img)
img1 = np.reshape(img,(25,10))
plt.imshow(img1,cmap='gray')
plt.title(wrong_y[0])
plt.show()

img = np.dot(w2,img)
img2 = np.reshape(img,(20,10))
plt.imshow(img2,cmap='gray')
plt.title(wrong_y[0])
plt.show()

img = np.dot(w3,img)
img3 = np.reshape(img,(15,10))
plt.imshow(img3,cmap='gray')
plt.title(wrong_y[0])
plt.show()

img = np.dot(w4,img)
img4 = np.reshape(img,(10,1))
plt.imshow(img4,cmap='gray')
plt.title(wrong_y[0])
plt.show()


for i in range(10):

	img = np.reshape(wrong_x[i],(28,28))
	plt.imshow(img,cmap='gray')
	plt.title(wrong_y[i])
	plt.show()