# -*- coding: utf-8 -*-
"""
Created on Tue Aug 30 09:19:56 2016

@author: nn31
"""

from __future__ import print_function
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
from six.moves import range

import os
os.chdir('/Users/nn31/Dropbox/40-githubRrepos/tensorflow_pipelines/udacity_deep_learn/data')
pickle_file = 'notMNIST.pickle'

with open(pickle_file, 'rb') as f:
  save = pickle.load(f)
  train_dataset = save['train_dataset']
  train_labels = save['train_labels']
  valid_dataset = save['valid_dataset']
  valid_labels = save['valid_labels']
  test_dataset = save['test_dataset']
  test_labels = save['test_labels']
  del save  # hint to help gc free up memory
  print('Training set', train_dataset.shape, train_labels.shape)
  print('Validation set', valid_dataset.shape, valid_labels.shape)
  print('Test set', test_dataset.shape, test_labels.shape)
 
import matplotlib.pyplot as plt
#plt.imshow(train_dataset[0,:])
#train_labels[0]

#We'll need to reshape the data to align more with the modeling frameworks we'll use
image_size = 28
num_labels = 10

def reformat(dataset, labels):
  dataset = dataset.reshape((-1, image_size * image_size)).astype(np.float32)
  # Map 0 to [1.0, 0.0, 0.0 ...], 1 to [0.0, 1.0, 0.0 ...]
  labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
  return dataset, labels
train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)
print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)


#look at one of the files
#import matplotlib.pyplot as plt
#plt.imshow(train_dataset2.reshape((-1,28,28))[0,:,:])


# With gradient descent training, even this much data is prohibitive.
# Subset the training data for faster turnaround.
train_subset = 200

def simple_logistic(train_subset):
    train_dataset_temp = train_dataset[:train_subset, :]
    train_labels_temp  = train_labels[:train_subset]
    batch_size = 128 #this is the number of "images" to run through sgc
    graph = tf.Graph()
    with graph.as_default():
    
      # Input data. For the training data, we use a placeholder that will be fed
      # at run time with a training minibatch.
      tf_train_dataset = tf.placeholder(tf.float32,shape=(batch_size, image_size * image_size))
      tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
      tf_valid_dataset = tf.constant(valid_dataset)
      tf_test_dataset = tf.constant(test_dataset)
      
      # Variables.
      # These are the parameters that we are going to be training. The weight
      # matrix will be initialized using random valued following a (truncated)
      # normal distribution. The biases get initialized to zero.
      weights = tf.Variable(tf.truncated_normal([image_size * image_size, num_labels]))
      biases = tf.Variable(tf.zeros([num_labels]))
      
      # Training computation.
      # We multiply the inputs with the weight matrix, and add biases. We compute
      # the softmax and cross-entropy (it's one operation in TensorFlow, because
      # it's very common, and it can be optimized). We take the average of this
      # cross-entropy across all training examples: that's our loss.
      logits = tf.matmul(tf_train_dataset, weights) + biases
      loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels))
      
      # Optimizer.
      # We are going to find the minimum of this loss using gradient descent.
      optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
      
      # Predictions for the training, validation, and test data.
      # These are not part of training, but merely here so that we can report
      # accuracy figures as we train.
      train_prediction = tf.nn.softmax(logits)
      valid_prediction = tf.nn.softmax(tf.matmul(tf_valid_dataset, weights) + biases)
      test_prediction = tf.nn.softmax(tf.matmul(tf_test_dataset, weights) + biases)
      
    num_steps = 3001
    
    def accuracy(predictions, labels):
      return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0])
    output = dict()
    with tf.Session(graph=graph) as session:
      # This is a one-time operation which ensures the parameters get initialized as
      # we described in the graph: random weights for the matrix, zeros for the
      # biases. 
      tf.initialize_all_variables().run()
      print('Initialized')
      for step in range(num_steps):
        offset = (step * batch_size) % (train_labels_temp.shape[0] - batch_size)
        # Generate a minibatch.
        batch_data = train_dataset_temp[offset:(offset + batch_size), :]
        batch_labels = train_labels_temp[offset:(offset + batch_size), :]
        # Prepare a dictionary telling the session where to feed the minibatch.
        # The key of the dictionary is the placeholder node of the graph to be fed,
        # and the value is the numpy array to feed to it.
        feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
        _, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)
        if (step == num_steps-1):
            a = session.run(weights)
            b = session.run(biases)
            output['sample_size'] = train_subset
            output['loss'] = l
            output['train'] = accuracy(predictions, batch_labels)
            output['validation'] = accuracy(valid_prediction.eval(), valid_labels)
            output['test'] = accuracy(test_prediction.eval(), test_labels)
    return(output,a,b)

samples = [200,500,1000,4000,8000,10000]  
results = [simple_logistic(num) for num in samples]
    
#figure out some nice visualizations to demonstrate:
#https://github.com/napsternxg/Udacity-Deep-Learning/blob/master/udacity/1_notmnist.ipynb 
#http://sebastianraschka.com/faq/docs/closed-form-vs-gd.html
  
colors = ['b', 'c', 'y', 'm', 'r']
plt.title('Multinomial Logistic Classifier - Stochastic Gradient Descent (128)')
plt.xlabel('Sample Size')
plt.ylabel('Accuracy')
plt.ylim(0, 110)
plt.xlim(0,10100)  
test = plt.scatter([x[0].get('sample_size') for x in results],[x[0].get('test') for x in results],marker='o',color = colors[0]) 
vali = plt.scatter([x[0].get('sample_size') for x in results],[x[0].get('validation') for x in results],marker='o',color = colors[1])  
trai = plt.scatter([x[0].get('sample_size') for x in results],[x[0].get('train') for x in results],marker='o',color = colors[4])  
    
plt.legend((test,vali,trai),
           ('Test Set', 'Validation Set', 'Training Set'),
           scatterpoints=1,
           loc='upper right',
           ncol=3,
           fontsize=8)
plt.savefig('/Users/nn31/Dropbox/40-githubRrepos/lungmap-scratch/queries/viable_sample_size/mlc_sgd.png')



  
  
  