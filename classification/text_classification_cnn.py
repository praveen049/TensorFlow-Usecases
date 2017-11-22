import pandas as pd
import numpy as np
import tensorflow as tf

from sklearn.cross_validation import train_test_split

MAX_DOCUMENT_LENGTH = 500
EMBEDDING_SIZE = 20
N_FILTER = 10
WINDOW_SIZE = 20
FILTER_SHAPE1 = [WINDOW_SIZE, EMBEDDING_SIZE]
FILTER_SHAPE2 = [WINDOW_SIZE, N_FILTER]
POOLING_WINDOW = 4
POOLING_STRIDE = 2
n_words = 0
MAX_LABEL=150
WORDS_FEATURE = 'words'

def cnn_model(features, labels, mode):
    word_vectors = tf.contrib.layers.embed_sequence(
        features[WORDS_FEATURE], vocab_size = n_words, embed_dim=EMBEDDING_SIZE)

    word_vectors = tf.expand_dims(word_vectors, 3)
    with tf.variable_scope('CNN_layer1'):
        conv1= tf.layers.conv2d(
            word_vectors,
            filters=N_FILTER,
            kernel_size=FILTER_SHAPE1,
            padding='VALID',
            activation=tf.nn.relu)

        pool1 = tf.layers.max_pooling2d(
            conv1,
            pool_size=POOLING_WINDOW,
            strides=POOLING_STRIDE,
            padding='SAME')

        pool1 = tf.transpose(pool1, [0,1,3,2])

    with tf.variable_scope('CNN_Layer2'):
        conv2=tf.layers.conv2d(
            pool1,
            filters=N_FILTER,
            kernel_size=FILTER_SHAPE2,
            padding='VALID')
        pool2 = tf.squeeze(tf.reduce_max(conv2,1), squeeze_dims=[1])

    logits = tf.layers.dense(pool2, MAX_LABEL, activation = None)

    predicted_classes = tf.argmax(logits, 1)
    if mode == tf.estimator.ModeKeys.PREDICT:
       return tf.estimator.EstimatorSpec(
           mode=mode,
           predictions={
               'class': predicted_classes,
               'prob': tf.nn.softmax(logits)
           })

    onehot_labels = tf.one_hot(labels, MAX_LABEL, 1, 0)
    loss = tf.losses.softmax_cross_entropy(
          onehot_labels=onehot_labels, logits=logits)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
        train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

    eval_metric_ops = {
        'accuracy': tf.metrics.accuracy(
              labels=labels, predictions=predicted_classes)
    }
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

def load_file (samplefile, rows):
    """
    """
    #Read the sample excel file for 
    sample = pd.read_json(samplefile, encoding='utf-8')
    sample = sample.sample(frac=1).reset_index(drop=True)
    sample = sample.iloc[0:rows]

    # For this demo keep only the title of the pronto and the group in charge name which is under the columns Group In Charge
    sample_test = sample[['title', 'description','groupInCharge','revisionHistory', 'reasonForTransfer']]
#    sample_test.dropna(how='any', inplace=True)

    if FLAGS.counter == 'revision':
        """
        The below piece of code, convert the revision history which is a list of revision to a dataframe with 
        multiple columns and then combines it back into a single column of string.
        """
        tmp = sample_test['revisionHistory'].apply(pd.Series)
        sample_test['revisionHistory'] = tmp[tmp.columns[1:]].apply(lambda x: u','.join(x.astype(unicode, str)), axis=1)
        sample_test['transfercount'] = sample_test.revisionHistory.str.count('group in charge changed')
    else :
        sample_test['transfercount'] = sample_test['reasonForTransfer'].apply(lambda x: len(x) if type(x) is list else (1 if x else 0))

    # Rename the group in charge to something more recognizable - GIC
    sample_test.columns = ['Title', 'Description', 'GIC', 'revisionHistory', 'reasonForTransfer', 'transferCount']

    # Create a new column combining title and description
    sample_test['TitleNDescrip'] = sample_test['Title'] + ' ' + sample_test['Description']

    return sample_test

def prepare_dfs (sample_test):
    """
    This method takes the read DF and creates the X and y for the 
    """
    # Use only the Title for this demo.
    X = sample_test['TitleNDescrip']

    # The label is the GIC information already provided
    y = sample_test['GIC']
    from sklearn import preprocessing
    le = preprocessing.LabelEncoder()
    y = le.fit_transform(y)

    # Split the entire data set into train set and testing set. 
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    print ('The split between trianing and testing data is train:%0.2f  test:%0.2f\n' %(len(X_train) *100/len(X),len(X_test) * 100 / len(X)))

    return X_train, X_test, y_train, y_test

def main(unused_argv):
    global n_words

    #dbpedia = tf.contrib.learn.datasets.load_dataset('dbpedia')
    # Read all the rows and then sample so that we can get a radmon order
    '''
    x_tr = pd.read_csv('./dbpedia_data/dbpedia_csv/train.csv', header=None, names=['label','title', 'content'])
    x_te = pd.read_csv('./dbpedia_data/dbpedia_csv/test.csv', header=None, names=['label','title','content'])
    x_tr = x_tr.sample(frac=0.002, replace=True)
    x_te = x_te.sample(frac=0.002, replace=True)
    x_train = x_tr['content']
    y_train = x_tr['label']
    x_test = x_te['content']
    y_test = x_te['label']
    print (x_train.shape)
    '''
    '''
    dbpedia = tf.contrib.learn.datasets.load_dataset(
        'dbpedia')
    x_train = pd.DataFrame(dbpedia.train.data)[1]
    y_train = pd.Series(dbpedia.train.target)
    
    x_test = pd.DataFrame(dbpedia.test.data)[1]
    y_test = pd.Series(dbpedia.test.target)
    print (type(x_train))
    '''
    sample_test = load_file(FLAGS.path, FLAGS.rows)

    x_train, x_test, y_train, y_test = prepare_dfs(sample_test)
    
    vocab_proc = tf.contrib.learn.preprocessing.VocabularyProcessor(MAX_DOCUMENT_LENGTH)
    x_train = np.array(list(vocab_proc.fit_transform(x_train)))
    x_test = np.array(list(vocab_proc.transform(x_test)))
    n_words = len(vocab_proc.vocabulary_)
    print('Total words are: {!r}\n'.format(n_words))

      # Build model
    classifier = tf.estimator.Estimator(model_fn=cnn_model)

    # Train.
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={WORDS_FEATURE: x_train},
        y=y_train,
        batch_size=len(x_train),
        num_epochs=None,
        shuffle=True)
    classifier.train(input_fn=train_input_fn, steps=100)

    # Evaluate.
    test_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={WORDS_FEATURE: x_test},
        y=y_test,
        num_epochs=1,
        shuffle=False)

    scores = classifier.evaluate(input_fn=test_input_fn)
    print('Accuracy: {0:f}'.format(scores['accuracy']))

    
import argparse
import sys
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--path',
        type=str,
        default='',
        help='The path of the data file'
    )
    parser.add_argument(
        '--rows',
        type=int,
        default=-1,
        help='The number of records to read'
    )
    parser.add_argument(
        '--counter',
        type=str,
        default='transfer',
        help='This parameter controls is transferReason(transfer) or RevisionHistory(revision) are used for counting'
    )
    FLAGS, unparsed = parser.parse_known_args()

    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
