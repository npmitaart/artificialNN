#import libraries
import tensorflow as tf
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split

data = np.array(
    [[1, 1, -1],
    [2, 1, 1],
    [-1, 2, -3],
    [1, 2, 3],
    [1, 1, 3]]
)


class SOM:
    def __init__(self, width, height, input_dimension):
        self.width = width
        self.height = height
        self.input_dimension = input_dimension

        self.weight = tf.Variable(tf.random_normal([width * height, input_dimension]))
        self.input = tf.placeholder(tf.float32, [input_dimension])

        self.location = [tf.to_float([y,x]) for y in range(height) for x in range(width)]

        self.bmu = self.getBMU()

        self.update_weight = self.update_neigbours()

    def getBMU(self):
        #Best Matching Unit

        #Eucledian distance
        square_distance = tf.square(self.input - self.weight)
        distance = tf.sqrt(tf.reduce_sum(square_distance, axis=1))

        #Get BMU index
        bmu_index = tf.argmin(distance)
        #Get the position
        bmu_position = tf.to_float([tf.div(bmu_index,self.width), tf.mod(bmu_index, self.width)])
        return bmu_position

    def update_neigbours(self):

        learning_rate = 0.5

        #Formula calculate sigma / radius
        sigma = tf.to_float(tf.maximum(self.width, self.height) / 2)

        #Eucledian Distance between BMU and location
        square_difference = tf.square(self.bmu - self.location)
        distance = tf.sqrt(tf.reduce_sum(square_difference,axis=1))

        #Calculate Neighbour Strength based on formula
        # NS = tf.exp((- distance ** 2) /  (2 * sigma ** 2))
        NS = tf.exp(tf.div(tf.negative(tf.square(distance)), 2 * tf.square(sigma)))

        #Calculate rate before reshape
        rate = NS * learning_rate

        #Reshape to [width * height, input_dimension]
        rate_stacked = tf.stack([tf.tile(tf.slice(rate,[i],[1]), [self.input_dimension]) 
            for i in range(self.width * self.height)])

        #Calculate New Weight
        new_weight = self.weight + rate_stacked * (self.input - self.weight)

        return tf.assign(self.weight, new_weight)

    def train(self, dataset, epoch):

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            #training
            for i in range(epoch+1):
                for data in dataset:
                    dictionary = {
                        self.input : data
                    }

                    sess.run(self.update_weight,feed_dict=dictionary)

            #assign clusters
            location = sess.run(self.location)
            weight = sess.run(self.weight)

            clusters = [[] for i in range(self.height)]

            for i, loc in enumerate(location):
                clusters[int(loc[0])].append(weight[i])

            self.clusters = clusters
            
colors_dataset = [
    [0,0,0],
    [0,0,1],
    [0,1,0],
    [1,0,0],
    [1,1,0],
    [1,0,1],
    [0,1,1],
    [1,1,1]
]

input_dimension = len(colors_dataset[0])
epoch = 1000
print(input_dimension)
som = SOM(3,5,input_dimension)

som.train(colors_dataset,epoch)
plt.imshow(som.clusters)
plt.show()