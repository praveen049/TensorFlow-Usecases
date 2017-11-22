import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt
import warnings

from sklearn.metrics import mean_squared_error

from lstm_predictor import lstm_model, generate_cpu_data

warnings.filterwarnings("ignore")

def load_data():                             
    # Load the CPU data from the file provided
    if FLAGS.source == 'zabbix':
        server = load_server('http://10.91.60.53/zabbix', 'Admin', 'zabbix', None)
        data = fetch_data(server, hostname='sprintlab474vm4', metric='cpu.util[ ]', duration= 60 * 24 * 8)
        print ('The size of data downloaded from Zabbix is {!r}'.format(len(data)))

        X, y = generate_timestep_data(data, TIMESTEPS)

    elif FLAGS.source == 'txt': # The source is locally as txt. Mainly for testing purposes.
        cpu=np.fromfile(FLAGS.file, dtype=float, sep=' ', count=-1)
        # Lets convert this into int as faster processing
        cpu = cpu.astype(int, copy=False)
        X, y = generate_cpu_data(cpu, TIMESTEPS, seperate=False)
    elif FLAGS.source == 'sin':
        data = np.arange(-100,100,.05)
        data = np.sin(data)
        print (data.shape)
        X, y = generate_cpu_data(data, TIMESTEPS)
    else:
        return None
    
    print('-----------------------------------------')
    print('train y shape',y['train'].shape)
    return X, y

import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--source',
        type=str,
        default='zabbix',
        help='The source of the metric data'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='../config/zabbix.config',
        help='The configuration to be used for accessing the source'
    )
    parser.add_argument(
        '--file',
        type=str,
        default='../data/cpuload.txt',
        help='The source of the metric data'
    )

FLAGS, unparsed = parser.parse_known_args()

TIMESTEPS = 20
num_hidden=200

Xd, yd = load_data()

train_input = Xd['train']
train_input = train_input.reshape(-1,20,1)
train_output = yd['train']

test_input = Xd['test']
test_output = yd['test']

X = tf.placeholder(tf.float32, [None, 20, 1])
y = tf.placeholder(tf.float32, [None, 1])

cell = tf.nn.rnn_cell.LSTMCell(num_hidden, state_is_tuple=True)

val, state = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)
#val = tf.Print(val, [tf.argmax(val,1)], 'argmax(val)=' , summarize=20, first_n=7)
# The value is [batch_size, output vector, timesteps?] and we need the output vector so transpose
val = tf.transpose(val, [1, 0, 2])
#val = tf.Print(val, [tf.argmax(val,1)], 'argmax(val2)=' , summarize=20, first_n=7)

# Take only the last output after 20 time steps
last = tf.gather(val, int(val.get_shape()[0]) - 1)
#last = tf.Print(last, [tf.argmax(last,1)], 'argmax(val3)=' , summarize=20, first_n=7)

# define variables for weights and bias
weight = tf.Variable(tf.truncated_normal([num_hidden, int(y.get_shape()[1])]))
bias = tf.Variable(tf.constant(0.1, shape=[y.get_shape()[1]]))

# Prediction is matmul of last value + wieght + bais
prediction = tf.matmul(last, weight) + bias

"""
 Tried different cost functions for optimization and for this problem finally mean square was the best way to
 go.
"""
# y is the true distrubution and prediction is the predicted
#cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(prediction), reduction_indices=[1]))
#cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
cost = 0.5*tf.square(y - prediction)
cost = tf.reduce_mean(cost)

optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
minimize = optimizer.minimize(cost)

from tensorflow.python import debug as tf_debug
inita = tf.initialize_all_variables()
sess = tf.Session()
sess.run(inita)

batch_size = 100
no_of_batches = int(len(train_input)/batch_size)
epoch = 10
test_size = 100
for i in range(epoch):
    for start, end in zip(range(0, len(train_input), batch_size), range(batch_size, len(train_input)+1, batch_size)):
        sess.run(minimize, feed_dict={X: train_input[start:end], y: train_output[start:end]})

    # Get a mini batch of 100 sample and test it    
    test_indices = np.arange(len(test_input))  # Get A Test Batch
    np.random.shuffle(test_indices)
    test_indices = test_indices[0:test_size]
    print (i, mean_squared_error(np.argmax(test_output[test_indices], axis=1), sess.run(prediction, feed_dict={X: test_input[test_indices]})))

print ("predictions", prediction.eval(feed_dict={X: train_input}, session=sess))
y_pred = prediction.eval(feed_dict={X: test_input}, session=sess)
sess.close()
train_size = train_output.shape[0]
test_size = test_output.shape[0]
tr_x = np.arange(0, train_size,1)
te_x = np.arange(train_size, train_size+test_size, 1)
plot_train, plot_test, plot_pred, = plt.plot(tr_x, train_output, 'g', te_x, test_output, 'r', te_x, y_pred, 'b')
plt.legend([plot_train, plot_test, plot_pred], ['train','test', 'prediction'], loc=1)
plt.show()
