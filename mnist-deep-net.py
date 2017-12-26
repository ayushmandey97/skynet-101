import tensorflow as tf
'''
-	Input -> weights -> HiddenLayer1 -> Activation fn -> weights -> HiddenLayer2 -> Activation fn -> weights -> Output layer (Feed - Forward)
-	Compare output to intended output -> Cost/Loss fn (Cross entropy - how far from right answer are we)
-	Optimization fn (Optimizer) -> Minimize cost (AdamOptimizer, Stochasic gradient descent, AdaGrad etc) (Back-propagation)
-	Feed-forward + Back-propagation = epoch (1 cycle). Multple epochs take place to minimize cost.
'''

from tensorflow.examples.tutorials.mnist import input_data # MNIST dataset
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)


'''
ONE HOT = TRUE, and 10 classes, 0-9

0 = [1,0,0,0,0,0,0,0,0]
1 = [0,1,0,0,0,0,0,0,0]
2 = [0,0,1,0,0,0,0,0,0]
3 = [0,0,0,1,0,0,0,0,0]
...

'''

n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500

n_classes = 10

batch_size = 100

#height x width(28x28)
x = tf.placeholder('float', [None, 784]) #input data
y = tf.placeholder('float', [None, 10])

def neural_network_model(data):
	# input data X weights + bias
	hidden_1_layer = {'weights':tf.Variable(tf.random_normal([784, n_nodes_hl1])), 'biases':tf.Variable(tf.random_normal(n_nodes_hl1))}
	hidden_2_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])), 'biases':tf.Variable(tf.random_normal(n_nodes_hl2))}
	hidden_3_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])), 'biases':tf.Variable(tf.random_normal(n_nodes_hl3))}
	output_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])), 'biases':tf.Variable(tf.random_normal(n_classes))}

	#Rectified linear -> Activation function (like sigmoid)

	l1 = tf.add(tf.matmul(data, hidden_1_layer['weights']),hidden_1_layer['biases'])
	l1 = tf.nn.relu(l1)

	l2 = tf.add(tf.matmul(l1, hidden_2_layer['weights']),hidden_2_layer['biases'])
	l2 = tf.nn.relu(l1)

	l3 = tf.add(tf.matmul(l2, hidden_3_layer['weights']),hidden_3_layer['biases'])
	l3 = tf.nn.relu(l1)

	output = tf.matmul(l3, output_layer['weights']) + output_layer['biases']

	return output



