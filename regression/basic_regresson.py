import numpy as np
import tensorflow as tf
from mlxtend.data import boston_housing_data
from mlxtend.preprocessing import shuffle_arrays_unison
from sklearn.metrics import mean_squared_error



X, y = boston_housing_data()
X, y = shuffle_arrays_unison([X, y], random_seed=1)
X_test, y_test = X[:50], y[:50]
X_train, y_train = X[50:], y[50:]

mu1, mu2 = X_train.mean(axis=0), y_train.mean()
sigma1, sigma2 = X_train.std(axis=0), y_train.std()
X_train = (X_train - mu1) / sigma1
X_test = (X_test - mu1) / sigma1
y_train = (y_train - mu2) / sigma2
y_test = (y_test - mu2) / sigma2


  num_label = 1
  graph = tf.Graph()
  with graph.as_default():
    # Input data.
    # Load the training, validation and test data into constants that are
    # attached to the graph.
    tf_train_dataset = tf.constant(X_train, dtype=tf.float64)
    tf_train_labels  = tf.constant(y_train[:, np.newaxis], dtype=tf.float64)
    tf_test_dataset  = tf.constant(X_test, dtype=tf.float64)

    # Variables.
    weights = tf.Variable( tf.truncated_normal([X_train.shape[1],num_label], seed = np.random.seed(1), dtype=tf.float64),dtype=tf.float64)
    biases = tf.Variable(tf.zeros([num_label], dtype=tf.float64),dtype=tf.float64)


    # Training computation.
    logits = tf.add(tf.matmul(tf_train_dataset, weights),biases)
    loss = tf.reduce_mean(tf.square(logits - tf_train_labels))


    # Optimizer.
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(loss)
    train_prediction = logits
    test_prediction = tf.add( tf.matmul( tf_test_dataset, weights ), biases)


with tf.Session(graph=graph) as session:
    tf.initialize_all_variables().run()
    print('Initialized')
    for step in range(200):
        _, l, predictions = session.run([optimizer, loss, train_prediction])
        print('Loss at step %d: %f' % (step, l) )
        print('MSE test: %.3f' % ( mean_squared_error(y_train, predictions)))
        print('MSE Test: %.3f' % ( mean_squared_error(y_test,test_prediction.eval())) )
