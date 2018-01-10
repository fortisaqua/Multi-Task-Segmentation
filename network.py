import os
import tensorflow as tf
from tensorflow.python.lib.io.tf_record import TFRecordCompressionType
import glob
import numpy as np
import util as ut
from data import TF_Records
import tools
import config
import gc


FLAGS = tf.app.flags.FLAGS

class Network():
    def __init__(self):
        self.FLAGS = FLAGS
        self.record_dir = FLAGS.record_dir
        self.block_shape = [FLAGS.block_shape_1,FLAGS.block_shape_2,FLAGS.block_shape_3]
        self.batch_size_train = FLAGS.batch_size_train
        self.batch_size_test = FLAGS.batch_size_test
        local_dirs=[]
        self.train_models_dir = FLAGS.train_models_dir
        local_dirs.append(self.train_models_dir)
        self.train_sum_dir = FLAGS.summary_dir_train
        local_dirs.append(self.train_sum_dir)
        self.test_result = FLAGS.test_result
        local_dirs.append(self.test_result)
        for dir in local_dirs:
            if not os.path.exists(dir):
                os.makedirs(dir)

    def dense_block(self,X,growth,depth,block_name,training,scope):
        layers = []
        layers.append(X)
        for i in range(depth):
            layer = tools.Ops.batch_norm(layers[-1], name_scope=scope+'_bn_'+str(i), training=training)
            # layer = tools.Ops.xxlu(layer, name=block_name + 'relu' + str(i))
            layer = tools.Ops.conv3d(layer, k=3, out_c=growth, str=1, name=block_name + '_layer_' + str(i))
            next_input = tf.concat([layer, layers[-1]], axis=4)
            layers.append(next_input)
        return tools.Ops.xxlu(tools.Ops.batch_norm(layers[-1], name_scope=scope+'_bn_output', training=training))

    def Segment_part(self,inputs,input_1,down_1,down_2,name,training,threshold):
        original_down = 16
        growth_down = 15
        depth_down = 5

        growth_up = 6
        depth_up = 3

        with tf.variable_scope(name+'_segment'):
            # up sample input 1
            up_sample_input_1 = tf.concat([inputs,down_2],axis=4,name='up_sample_input_1')
            # up sample 1
            up_sample_1 = tools.Ops.deconv3d(up_sample_input_1,k=2,
                                             out_c=original_down+3*(growth_down*depth_down),str=2,name='up_sample_1')

            # dense block 4
            dense_block_output_4 = self.dense_block(up_sample_1,growth_up,depth_up,'dense_block_4',training,scope='dense_block_4')

            # up sample input 2
            up_sample_input_2 = tf.concat([dense_block_output_4,down_1],axis=4,name='up_sample_input_2')
            # up sample 2
            up_sample_2 = tools.Ops.deconv3d(up_sample_input_2,k=2,
                                             out_c=original_down+4*(growth_down*depth_down)+1*(growth_up*depth_up),str=2,name='up_sample_2')

            # dense block 5
            dense_block_output_5 = self.dense_block(up_sample_2,growth_up,depth_up,'dense_block_5',training,scope='dense_block_5')

            # segment input
            segment_input = tf.concat([dense_block_output_5,input_1],axis=4,name='segment_input')
            # segment conv 1
            segment_conv_1 = tools.Ops.conv3d(segment_input,k=3,
                                              out_c=original_down+5*(growth_down*depth_down)+2*(growth_up*depth_up),str=1,name='segment_conv_1')
            # segment conv 2
            segment_conv_2 = tools.Ops.conv3d(segment_conv_1,k=3,out_c=64,str=1,name='segment_conv_2')
            # segment input
            segment_input = tools.Ops.batch_norm(segment_conv_2,name_scope='bn_segment_input',training=training)
            # segment predict
            segment_predict = tools.Ops.conv3d(segment_input,k=1,out_c=1,str=1,name='segment_predict')

            #sigmoid predict
            segment_sigmoid = tf.sigmoid(segment_predict,name='segmoid_predict')
            #modify predict
            modified_segment = tf.maximum(segment_sigmoid,threshold,name='modified_segment')
            return modified_segment


    def Dense_Net(self,inputs,training,batch_size,threshold):
        original_down = 16
        growth_down = 12
        depth_down = 5
        X = tf.reshape(inputs,[batch_size,self.block_shape[0],self.block_shape[1],self.block_shape[2],1],name='input')

        with tf.variable_scope('feature_extract'):
            # dense block 1
            dense_block_input_1 = tools.Ops.conv3d(X,k=3,out_c=original_down,str=1,name='dense_block_input_1')
            dense_block_output_1 = self.dense_block(dense_block_input_1,growth_down,depth_down,'dense_block_1',training,scope='dense_block_1')

            # down sample 1
            down_sample_1 = tools.Ops.conv3d(dense_block_output_1,k=2,out_c=original_down+1*(growth_down*depth_down),str=2,name='down_sample_1')

            # dense block 2
            dense_block_output_2 = self.dense_block(down_sample_1,growth_down,depth_down,'dense_block_2',training,scope='dense_block_2')

            # down sample 2
            down_sample_2 = tools.Ops.conv3d(dense_block_output_2,k=2,out_c=original_down+2*(growth_down*depth_down),str=2,name='down_sample_2')

            # dense block 3
            dense_block_output_3 = self.dense_block(down_sample_2,growth_down,depth_down,'dense_block_3',training,scope='dense_block_3')

            lung_predict = self.Segment_part(dense_block_output_3,dense_block_input_1,down_sample_1,down_sample_2,
                                             name='lung',training=training,threshold=threshold)
            airway_predict = self.Segment_part(dense_block_output_3, dense_block_input_1, down_sample_1, down_sample_2,
                                             name='airway', training=training, threshold=threshold)
            artery_predict = self.Segment_part(dense_block_output_3, dense_block_input_1, down_sample_1, down_sample_2,
                                             name='artery', training=training, threshold=threshold)

            return lung_predict,airway_predict,artery_predict

    # check if the network is correct
    def check_net(self):
        try:
            net = Network()
            block_shape = net.block_shape
            inputs = tf.placeholder(dtype=tf.float32,shape=[net.batch_size_train,block_shape[0],block_shape[1],block_shape[2]])
            training = tf.placeholder(tf.bool)
            lung,airway,artery = net.Dense_Net(inputs,training,net.batch_size_train,FLAGS.accept_threshold)
            # a way to get shape from a tensor and convert it into a list
            print [int(d) for d in lung.get_shape()]
            print [int(d) for d in airway.get_shape()]
            print [int(d) for d in artery.get_shape()]
            print "net is good!"
        except Exception,e:
            print e
        exit(1)