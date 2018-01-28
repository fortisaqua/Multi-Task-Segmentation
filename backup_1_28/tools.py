import numpy as np
import os
import re
from random import shuffle
import tensorflow as tf
import matplotlib.pyplot as plt
import random
from util import read_dicoms
import SimpleITK as ST
import gc

class Data_block:
    # single input data block
    def __init__(self,ranger,data_array):
        self.ranger=ranger
        self.data_array=data_array

    def get_range(self):
        return self.ranger

    def load_data(self):
        return self.data_array

class Test_data():
    # load data and translate to original array
    def __init__(self,data,block_shape,type):
        if type == 'dicom_data':
            self.img = read_dicoms(data)
        elif type == 'vtk_data':
            self.img = data
        self.space = self.img.GetSpacing()
        self.image_array = ST.GetArrayFromImage(self.img)
        self.image_array = np.transpose(self.image_array,[2,1,0])
        self.image_shape = np.shape(self.image_array)
        self.block_shape=block_shape
        self.blocks=dict()
        self.results=dict()

    # do the simple threshold function
    def threshold(self,low,high):
        mask_array=np.float32(np.float32(self.image_array<=high)*np.float32(self.image_array>=low))
        return np.float32(np.float32(self.image_array<=high)*np.float32(self.image_array>=low))

    def organize_blocks(self):
        block_num=0
        original_shape=np.shape(self.image_array)
        img_array = self.image_array*np.float32(self.image_array>=300)
        # img_array = self.image_array
        img_unseged = ST.GetImageFromArray(np.transpose(img_array,[2,1,0]))
        ST.WriteImage(img_unseged,'./test_result/img_unseged.vtk')
        print 'data shape: ', original_shape
        for i in range(0,original_shape[0],48):
            for j in range(0,original_shape[1],48):
                for k in range(0,original_shape[2],self.block_shape[2]/2):
                    if i<original_shape[0] and j<original_shape[1] and k<original_shape[2]:
                        block_array = img_array[i:i+self.block_shape[0],j:j+self.block_shape[1],k:k+self.block_shape[2]]
                        block_shape = np.shape(block_array)
                        ranger=[i,i+block_shape[0],j,j+block_shape[1],k,k+block_shape[2]]
                        this_block=Data_block(ranger,img_array[i:i+self.block_shape[0],j:j+self.block_shape[1],k:k+self.block_shape[2]])
                        self.blocks[block_num]=this_block
                        block_num+=1

    def upload_result(self,block_num,result_array):
        ranger = self.blocks[block_num].get_range()
        partial_result = result_array
        this_result = Data_block(ranger,partial_result)
        self.results[block_num]=this_result
        del self.blocks[block_num]
        gc.collect()

    def get_result(self):
        ret=np.zeros(self.image_shape,np.float32)
        for number in self.results.keys():
            try:
                ranger=self.results[number].get_range()
                xmin=ranger[0]
                xmax=ranger[1]
                ymin=ranger[2]
                ymax=ranger[3]
                zmin=ranger[4]
                zmax=ranger[5]
                temp_result = self.results[number].load_data()[:,:,:]
                # temp_shape = np.shape(temp_result)
                ret[xmin:xmax,ymin:ymax,zmin:zmax]+=temp_result[:xmax-xmin,:ymax-ymin,:zmax-zmin]
            except Exception,e:
                # print e
                print np.shape(self.results[number].load_data()[:,:,:]),self.results[number].get_range()
        return np.float32(ret)

class Ops:

    @staticmethod
    def lrelu(x, leak=0.2):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)

    @staticmethod
    def relu(x):
        return tf.nn.relu(x)

    @staticmethod
    def xxlu(x,name='relu'):
        if name =='relu':
            return  Ops.relu(x)
        if name =='lrelu':
            return  Ops.lrelu(x,leak=0.2)

    @staticmethod
    def variable_sum(var, name):
        with tf.name_scope(name):
            try:
                mean = tf.reduce_mean(var)
                tf.summary.scalar('mean', mean)
                stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
                tf.summary.scalar('stddev', stddev)
                tf.summary.scalar('max', tf.reduce_max(var))
                tf.summary.scalar('min', tf.reduce_min(var))
                tf.summary.histogram('histogram', var)
            except Exception,e:
                print e

    @staticmethod
    def variable_count():
        total_para = 0
        for variable in tf.trainable_variables():
            shape = variable.get_shape()
            variable_para = 1
            for dim in shape:
                variable_para *= dim.value
            total_para += variable_para
        return total_para

    @staticmethod
    def fc(x, out_d, name):
        xavier_init = tf.contrib.layers.xavier_initializer()
        zero_init = tf.zeros_initializer()
        in_d = x.get_shape()[1]
        w = tf.get_variable(name + '_w', [in_d, out_d], initializer=xavier_init)
        b = tf.get_variable(name + '_b', [out_d], initializer=zero_init)
        y = tf.nn.bias_add(tf.matmul(x, w), b)
        Ops.variable_sum(w, name)
        return y

    @staticmethod
    def maxpool3d(x,k,s,pad='SAME'):
        ker =[1,k,k,k,1]
        str =[1,s,s,s,1]
        y = tf.nn.max_pool3d(x,ksize=ker,strides=str,padding=pad)
        return y

    @staticmethod
    def conv3d(x, k, out_c, str, name,pad='SAME'):
        xavier_init = tf.contrib.layers.xavier_initializer()
        zero_init = tf.zeros_initializer()
        in_c = x.get_shape()[4]
        w = tf.get_variable(name + '_w', [k, k, k, in_c, out_c], initializer=xavier_init)
        b = tf.get_variable(name + '_b', [out_c], initializer=zero_init)

        stride = [1, str, str, str, 1]
        y = tf.nn.bias_add(tf.nn.conv3d(x, w, stride, pad), b)
        Ops.variable_sum(w, name)
        return y

    @staticmethod
    def deconv3d(x, k, out_c, str, name,pad='SAME'):
        xavier_init = tf.contrib.layers.xavier_initializer()
        zero_init = tf.zeros_initializer()
        bat, in_d1, in_d2, in_d3, in_c = [int(d) for d in x.get_shape()]
        w = tf.get_variable(name + '_w', [k, k, k, out_c, in_c], initializer=xavier_init)
        b = tf.get_variable(name + '_b', [out_c], initializer=zero_init)
        out_shape = [bat, in_d1 * str, in_d2 * str, in_d3 * str, out_c]
        stride = [1, str, str, str, 1]
        y = tf.nn.conv3d_transpose(x, w, output_shape=out_shape, strides=stride, padding=pad)
        y = tf.nn.bias_add(y, b)
        Ops.variable_sum(w, name)
        return y

    @staticmethod
    def batch_norm(x, name_scope, training, epsilon=1e-3, decay=0.999):
        '''Assume 2d [batch, values] tensor'''

        with tf.variable_scope(name_scope):
            size = x.get_shape().as_list()[-1]
            x_shape = x.get_shape()
            axis = list(range(len(x_shape) - 1))
            scale = tf.get_variable('scale', [size], initializer=tf.constant_initializer(0.1))
            offset = tf.get_variable('offset', [size])

            pop_mean = tf.get_variable('pop_mean', [size], initializer=tf.zeros_initializer, trainable=False)
            pop_var = tf.get_variable('pop_var', [size], initializer=tf.ones_initializer, trainable=False)
            batch_mean, batch_var = tf.nn.moments(x, axis)

            train_mean_op = tf.assign(pop_mean, pop_mean * decay + batch_mean * (1 - decay))
            train_var_op = tf.assign(pop_var, pop_var * decay + batch_var * (1 - decay))

            def batch_statistics():
                with tf.control_dependencies([train_mean_op, train_var_op]):
                    return tf.nn.batch_normalization(x, batch_mean, batch_var, offset, scale, epsilon)

            def population_statistics():
                return tf.nn.batch_normalization(x, pop_mean, pop_var, offset, scale, epsilon)

            return tf.cond(training, batch_statistics, population_statistics)

