import tensorflow as tf
import pandas as pd
import numpy as np
from random import shuffle

a = np.array([0,1,2,3,4,5,6,7,8,9])
one_hot_labels = np.zeros((10, 10))
one_hot_labels[np.arange(10), a] = 1
mnist = pd.read_csv('2015EE10466.csv')
#mnist = mnist.reindex(np.random.permutation(mnist.index))
mnist = np.array(mnist)
data = []
for i in range(0,mnist.shape[0]):
   data.append([mnist[i][0:784],one_hot_labels[mnist[i][784:][0]]])

#print(data[0][0],data[0][1],len(data))
i=0
train_x =[]
train_y = []
test_x = []
test_y = []
while(i<3000):
    sample = data[i:i+300]
    shuffle(sample)
    train_x = train_x + list(j[0] for j in sample[i:i+240])

    train_y=train_y+list(j[1] for j in sample[i:i+240])
    test_x = test_x+list(j[0] for j in sample[i+240:])
    test_y = test_y+list(j[1] for j in sample[i+240:])
    i= i+300

print(len(train_x),len(train_y),len(test_x),len(test_y))
'''
train_x = list(i[0] for i in data[0:2400])
train_y = list(i[1] for i in data[0:2400])
test_x = list(i[0] for i in data[2400:])
test_y = list(i[1] for i in data[2400:])'''

print(train_x[0],train_y[0])


n_nodes_hl1 = 10
n_nodes_hl2 = 10
n_nodes_hl3 = 10

n_classes = 10
batch_size = 100
hm_epochs = 50

x = tf.placeholder('float')
y = tf.placeholder('float')

hidden_1_layer = {
                  'weight': tf.Variable(tf.random_normal([len(train_x[0]), n_nodes_hl1])),
                  'bias': tf.Variable(tf.random_normal([n_nodes_hl1]))}

hidden_2_layer = {
                  'weight': tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                  'bias': tf.Variable(tf.random_normal([n_nodes_hl2]))}

hidden_3_layer = {
                  'weight': tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
                  'bias': tf.Variable(tf.random_normal([n_nodes_hl3]))}

output_layer = {
                'weight': tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
                'bias': tf.Variable(tf.random_normal([n_classes])), }


# Nothing changes
def neural_network_model(data):
    l1 = tf.add(tf.matmul(data, hidden_1_layer['weight']), hidden_1_layer['bias'])
    l1 = tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1, hidden_2_layer['weight']), hidden_2_layer['bias'])
    l2 = tf.nn.relu(l2)

    l3 = tf.add(tf.matmul(l2, hidden_3_layer['weight']), hidden_3_layer['bias'])
    l3 = tf.nn.relu(l3)

    output = tf.matmul(l3, output_layer['weight']) + output_layer['bias']

    return output


def train_neural_network(x):
    prediction = neural_network_model(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=y))
    #reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    #reg_constant = 0.1  # Choose an appropriate one.
    #cost = cost + reg_constant * sum(reg_losses)
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())

        for epoch in range(hm_epochs):
            epoch_loss = 0
            i = 0
            while i < len(train_x):
                start = i
                end = i + batch_size
                batch_x = np.array(train_x[start:end])
                batch_y = np.array(train_y[start:end])

                _, c = sess.run([optimizer, cost], feed_dict={x: batch_x,
                                                              y: batch_y})
                epoch_loss += c
                i += batch_size

            print('Epoch', epoch + 1, 'completed out of', hm_epochs, 'loss:', epoch_loss)

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

        print('Accuracy:', accuracy.eval({x: test_x, y: test_y}))
        print('Accuracy:', accuracy.eval({x: train_x, y: train_y}))


train_neural_network(x)

