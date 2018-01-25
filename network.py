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
import SimpleITK as ST
import time
import sys

FLAGS = tf.app.flags.FLAGS
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

class Network():
    def __init__(self):
        self.FLAGS = FLAGS
        self.record_dir = FLAGS.record_dir
        self.record_dir_test = FLAGS.record_test_dir
        self.block_shape = [FLAGS.block_shape_1,FLAGS.block_shape_2,FLAGS.block_shape_3]
        self.batch_size_train = FLAGS.batch_size_train
        self.batch_size_test = FLAGS.batch_size_test
        self.test_step = FLAGS.test_step
        local_dirs=[]
        self.train_models_dir = FLAGS.train_models_dir
        local_dirs.append(self.train_models_dir)
        self.train_sum_dir = FLAGS.summary_dir_train
        local_dirs.append(self.train_sum_dir)
        self.test_sum_dir = FLAGS.summary_dir_test
        local_dirs.append(self.test_sum_dir)
        self.test_result = FLAGS.test_result
        local_dirs.append(self.test_result)
        for dir in local_dirs:
            if not os.path.exists(dir):
                os.makedirs(dir)

    def dense_block(self,X,growth,depth,block_name,training,scope):
        layers = []
        layers.append(X)
        if depth==0:
            return X
        for i in range(depth):
            layer = tools.Ops.batch_norm(layers[-1], name_scope=scope+'_bn_'+str(i), training=training)
            # layer = tools.Ops.xxlu(layer, name=block_name + 'relu' + str(i))
            layer = tools.Ops.conv3d(layer, k=3, out_c=growth, str=1, name=block_name + '_layer_' + str(i))
            temp_shape = layers[-1].get_shape()
            next_input = tf.concat([layer, tools.Ops.conv3d(layers[-1],k=1,out_c=int(temp_shape[-1]),str=1,name=scope+"_down_sprading_"+str(i))], axis=4)
            layers.append(next_input)
        return tools.Ops.xxlu(tools.Ops.batch_norm(layers[-1], name_scope=scope+'_bn_output', training=training))

    def Segment_part(self,inputs,input_1,down_1,down_2,name,training,threshold):
        original_down = 16
        growth_down = 12
        depth_down = 6

        if 'lung' in name:
            growth_up = 6
            depth_up = 0
        if 'airway' in name:
            growth_up = 6
            depth_up = 1
        else:
            growth_up = 12
            depth_up = 1

        with tf.variable_scope(name+'_segment'):
            # up sample input 1
            up_sample_input_1 = tf.concat([inputs,down_2],axis=4,name='up_sample_input_1')
            # up sample 1
            up_sample_1 = tools.Ops.deconv3d(up_sample_input_1,k=2,
                                             out_c=original_down+1*(growth_down*depth_down),str=2,name='up_sample_1')

            # dense block 4
            dense_block_output_4 = self.dense_block(up_sample_1,growth_up,depth_up,'dense_block_4',training,scope='dense_block_4')

            # up sample input 2
            up_sample_input_2 = tf.concat([dense_block_output_4,down_1],axis=4,name='up_sample_input_2')
            # up sample 2
            up_sample_2 = tools.Ops.deconv3d(up_sample_input_2,k=2,
                                             out_c=original_down+1*(growth_down*depth_down)+1*(growth_up*depth_up),str=2,name='up_sample_2')

            # dense block 5
            dense_block_output_5 = self.dense_block(up_sample_2,growth_up,depth_up,'dense_block_5',training,scope='dense_block_5')

            # segment input
            segment_input = tf.concat([dense_block_output_5,input_1],axis=4,name='segment_input')
            # segment conv 1
            segment_conv_1 = tools.Ops.conv3d(segment_input,k=3,
                                              out_c=original_down+1*(growth_down*depth_down)+2*(growth_up*depth_up),str=1,name='segment_conv_1')
            # segment conv 2
            segment_conv_2 = tools.Ops.conv3d(segment_conv_1,k=3,out_c=64,str=1,name='segment_conv_2')
            # segment input
            segment_input = tools.Ops.batch_norm(segment_conv_2,name_scope='bn_segment_input',training=training)
            # segment predict
            segment_predict = tools.Ops.conv3d(segment_input,k=1,out_c=1,str=1,name='segment_predict')

            #sigmoid predict
            segment_sigmoid = tf.sigmoid(segment_predict,name='segmoid_predict')
            #modify predict
            modified_segment = tf.maximum(segment_sigmoid-threshold,0.01,name='modified_segment')
            return modified_segment,segment_sigmoid

    def Dense_Net(self,inputs,training,batch_size,threshold):
        original_down = 16
        growth_down = 12
        depth_down = 6
        casted_inputs = tf.cast(inputs,tf.float32)
        X = tf.reshape(casted_inputs,[batch_size,self.block_shape[0],self.block_shape[1],self.block_shape[2],1],name='input')

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

        lung_predict,lung_sigmoid = self.Segment_part(dense_block_output_3,dense_block_input_1,down_sample_1,down_sample_2,
                                         name='lung',training=training,threshold=threshold)
        # airway_predict,airway_sigmoid = self.Segment_part(dense_block_output_3, dense_block_input_1, down_sample_1, down_sample_2,
        #                                  name='airway', training=training, threshold=threshold)
        artery_predict,artery_sigmoid = self.Segment_part(dense_block_output_3, dense_block_input_1, down_sample_1, down_sample_2,
                                         name='artery', training=training, threshold=threshold)

        return lung_predict,lung_sigmoid,artery_predict,artery_sigmoid

    # artery branch only network
    def Dense_Net_Test(self,inputs,training,batch_size,threshold):
        original_down = 16
        growth_down = 12
        depth_down = 6
        casted_inputs = tf.cast(inputs,tf.float32)
        X = tf.reshape(casted_inputs,[batch_size,self.block_shape[0],self.block_shape[1],self.block_shape[2],1],name='input')

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

        artery_predict,artery_sigmoid = self.Segment_part(dense_block_output_3, dense_block_input_1, down_sample_1, down_sample_2,
                                         name='artery', training=training, threshold=threshold)

        # lung_predict,lung_sigmoid = self.Segment_part(dense_block_output_3,dense_block_input_1,down_sample_1,down_sample_2,name='lung',training=training,threshold=threshold)

        return artery_predict,artery_sigmoid

    def Dis(self, X, Y,training,batch_size):
        casted_inputs_X = tf.cast(X, tf.float32)
        casted_inputs_Y = tf.cast(Y, tf.float32)
        X_input = tf.reshape(casted_inputs_X, [batch_size, self.block_shape[0], self.block_shape[1], self.block_shape[2], 1],
                       name='input_dis_X')
        Y_input = tf.reshape(casted_inputs_Y, [batch_size, self.block_shape[0], self.block_shape[1], self.block_shape[2], 1],
                       name='input_dis_Y')
        layer = tf.concat([X_input,Y_input],axis=4)
        c_d = [1,2,32,63,128,192]
        s_d = [0,2,2,2,2,2]
        layers_d =[]
        layers_d.append(layer)
        for i in range(1,6,1):
            layer = tools.Ops.conv3d(layers_d[-1],k=4,out_c=c_d[i],str=s_d[i],name='d_1'+str(i))
            if i!=5:
                layer = tools.Ops.xxlu(layer, name='lrelu')
                # batch normal layer
                layer = tools.Ops.batch_norm(layer, 'bn_up' + str(i), training=training)
            layers_d.append(layer)
        y = tf.reshape(layers_d[-1],[batch_size,-1])
        # for j in range(len(layers_d)-1):
        #     y = tf.concat([y,tf.reshape(layers_d[j],[batch_size,-1])],axis=1)
        return tf.nn.sigmoid(y)

    # check if the network is correct
    def check_net(self):
        try:
            net = Network()
            block_shape = net.block_shape
            X = tf.placeholder(dtype=tf.float32,shape=[net.batch_size_train,block_shape[0],block_shape[1],block_shape[2]])
            training = tf.placeholder(tf.bool)
            with tf.variable_scope('generator'):
                lung,lung_sig,artery,artery_sig = net.Dense_Net(X,training,net.batch_size_train,FLAGS.accept_threshold)
            lung_pred_mask = tf.cast((lung > 0.1), tf.float32)
            with tf.variable_scope('discriminator'):
                XY_fake_pair = self.Dis(X, artery_sig, training, self.batch_size_train)
            # airway_pred_mask = tf.cast((airway > 0.1), tf.float32)
            artery_pred_mask = tf.cast((artery > 0.1), tf.float32)
            fake_mean = tf.reduce_mean(lung+artery)
            tf.summary.scalar('fake_mean',fake_mean)
            merge_summary_op = tf.summary.merge_all()
            # a way to get shape from a tensor and convert it into a list
            print [int(d) for d in lung.get_shape()]
            # print [int(d) for d in airway.get_shape()]
            print [int(d) for d in artery.get_shape()]
            print "net has ",tools.Ops.variable_count()," variables"
            sess = tf.Session()
            summary_writer = tf.summary.FileWriter(self.train_sum_dir, sess.graph)
            init_op = tf.group(tf.global_variables_initializer(),
                               tf.local_variables_initializer())
            sess.run(init_op)
            lung_np,artey_np,merge_summary\
                = sess.run([lung,artery,merge_summary_op],
                           feed_dict={X:np.int16(np.random.rand(net.batch_size_train,block_shape[0],block_shape[1],block_shape[2])*100),training:False})
            summary_writer.add_summary(merge_summary, global_step=0)
            print np.max(lung_np),"  ",np.min(lung_np)
            # print np.max(airway_np),"  ",np.min(airway_np)
            print np.max(artey_np),"  ",np.min(artey_np)
            print "net is good!"
        except Exception,e:
            print e
        exit(1)

    # training the network
    def train(self):
        flags = self.FLAGS
        block_shape = self.block_shape
        record_dir = self.record_dir
        record_dir_test = self.record_dir_test
        batch_size_train = self.batch_size_train
        batch_size_test = self.batch_size_test
        test_step = self.test_step
        LEARNING_RATE_BASE = flags.training_rate_base
        LEARNING_RATE_DECAY = flags.training_rate_decay
        X = tf.placeholder(dtype=tf.float32,shape=[batch_size_train,block_shape[0],block_shape[1],block_shape[2]])
        training = tf.placeholder(tf.bool)
        with tf.variable_scope('generator'):
            lung_pred,lung_sig,artery_pred,artery_sig = self.Dense_Net(X,training,flags.batch_size_train,flags.accept_threshold)

        lung_pred = tf.reshape(lung_pred,[batch_size_train,block_shape[0],block_shape[1],block_shape[2]])
        # airway_pred = tf.reshape(airway_pred,[batch_size_train,block_shape[0],block_shape[1],block_shape[2]])
        artery_pred = tf.reshape(artery_pred,[batch_size_train,block_shape[0],block_shape[1],block_shape[2]])

        # binary predict mask
        lung_pred_mask = tf.cast((lung_pred>0.01),tf.float32)
        # airway_pred_mask = tf.cast((airway_pred>0.01),tf.float32)
        artery_pred_mask = tf.cast((artery_pred>0.01),tf.float32)

        # labels
        lung_lable = tf.placeholder(dtype=tf.float32,shape=[batch_size_train,block_shape[0],block_shape[1],block_shape[2]])
        # airway_lable = tf.placeholder(dtype=tf.float32,shape=[batch_size_train,block_shape[0],block_shape[1],block_shape[2]])
        artery_lable = tf.placeholder(dtype=tf.float32,shape=[batch_size_train,block_shape[0],block_shape[1],block_shape[2]])

        # discriminator output
        with tf.variable_scope('discriminator'):
            XY_real_pair = self.Dis(X, artery_lable, training,flags.batch_size_train)
        with tf.variable_scope('discriminator',reuse=True):
            XY_fake_pair = self.Dis(X,artery_sig,training,flags.batch_size_train)

        # accuracy
        lung_acc = 2*tf.reduce_sum(lung_lable*lung_pred_mask)/(tf.reduce_sum(lung_lable+lung_pred_mask)+1e-6)
        tf.summary.scalar('lung_acc', lung_acc)
        # airway_acc = 2*tf.reduce_sum(airway_lable*airway_pred_mask)/(tf.reduce_sum(airway_lable+airway_pred_mask)+1e-6)
        # tf.summary.scalar('airway_acc', airway_acc)
        artery_acc = 2*tf.reduce_sum(artery_lable*artery_pred_mask)/(tf.reduce_sum(artery_lable+artery_pred_mask)+1e-6)
        tf.summary.scalar('artery_acc', artery_acc)

        # generator cross entropy loss
        w_fore_lung = flags.lung_fore_weight
        # w_fore_airway = flags.airway_fore_weight
        w_fore_artery = flags.artery_fore_weight
        # lung loss

        lung_lable_ = tf.reshape(lung_lable,shape=[batch_size_train,-1])
        artery_lable_ = tf.reshape(artery_lable,shape=[batch_size_train,-1])
        lung_pred_ = tf.reshape(lung_pred,shape=[batch_size_train,-1])
        artery_pred_ = tf.reshape(artery_pred,shape=[batch_size_train,-1])
        lung_loss = flags.lung_weight*tf.reduce_mean( -tf.reduce_mean(w_fore_lung*lung_lable_*tf.log(lung_pred_ + 1e-8),reduction_indices=[1]) -
                                                       tf.reduce_mean((1-w_fore_lung)*(1-lung_lable_)*tf.log(1-lung_pred_ + 1e-8),reduction_indices=[1]))
        tf.summary.scalar('lung_loss_cross_entropy', lung_loss)
        # predict_mean_lung = tf.reduce_mean(tf.log(1-lung_pred + 1e-8),reduction_indices=[1])
        # mask_mean_lung = tf.reduce_mean((1-lung_lable))
        # artery loss
        artery_loss = flags.artery_weight*tf.reduce_mean( -tf.reduce_mean(w_fore_artery*artery_lable_*tf.log(artery_pred_ + 1e-8),reduction_indices=[1]) -
                                                       tf.reduce_mean((1-w_fore_artery)*(1-artery_lable_)*tf.log(1-artery_pred_ + 1e-8),reduction_indices=[1]))
        tf.summary.scalar('artery_loss_cross_entropy', artery_loss)
        # predict_mean_artery = tf.reduce_mean(tf.log(1 - artery_pred + 1e-8), reduction_indices=[1])
        # mask_mean_artery = tf.reduce_mean((1 - artery_lable))

        # generator cross entropy loss
        ge_loss = lung_loss+artery_loss
        tf.summary.scalar('generator_cross_entropy_loss',ge_loss)

        # discriminator and gan loss
        gan_g_loss = -tf.reduce_mean(XY_fake_pair)
        gan_d_loss = tf.reduce_mean(XY_fake_pair)-tf.reduce_mean(XY_real_pair)
        tf.summary.scalar('dis_g_loss',gan_g_loss)
        tf.summary.scalar('dis_d_loss',gan_d_loss)
        alpha = tf.random_uniform(shape=[batch_size_train,block_shape[0]*block_shape[1]*block_shape[2]],minval=0.0,maxval=1.0)
        artery_sig_ = tf.reshape(artery_sig,shape=[batch_size_train,-1])
        diffenences_ = artery_sig_ - artery_lable_
        interpolates = artery_lable_ + alpha*diffenences_
        with tf.variable_scope('discriminator',reuse=True):
            XY_fake_intep = self.Dis(X,interpolates,training,batch_size_train)
        gradients = tf.gradients(XY_fake_intep,[interpolates])[0]
        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
        gradient_penalty = tf.reduce_mean((slopes - 1.0) ** 2)
        gan_d_loss += 10 * gradient_penalty

        # total generator loss
        gan_g_w = 5
        ge_w = 100 - gan_g_w
        total_g_loss = ge_w*ge_loss + gan_g_w*gan_g_loss
        tf.summary.scalar('total_g_loss', total_g_loss)

        # set training step and learning rate into tensors to save
        global_step = tf.Variable(0, trainable=False)
        learning_rate = tf.maximum(tf.train.exponential_decay(LEARNING_RATE_BASE, global_step, 15000/flags.batch_size_train,LEARNING_RATE_DECAY, staircase=True), 1e-9)

        # merge operation for tensorboard summary
        merge_summary_op = tf.summary.merge_all()
        # trainer
        ge_var = [var for var in tf.trainable_variables() if var.name.startswith('generator')]
        dis_var = [var for var in tf.trainable_variables() if var.name.startswith('discriminator')]
        ge_train_op = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8).minimize(total_g_loss,global_step,ge_var)
        dis_train_op = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8).minimize(gan_d_loss,global_step,dis_var)

        # data part
        records = ut.get_records(record_dir)
        records_processor = TF_Records(records, block_shape)
        single_blocks = records_processor.read_records()
        queue = tf.RandomShuffleQueue(capacity=8, min_after_dequeue=4,
                                      dtypes=(
                                          single_blocks['airway'].dtype,
                                          single_blocks['artery'].dtype,
                                          single_blocks['lung'].dtype,
                                          single_blocks['original'].dtype,
                                      ))
        enqueue_op = queue.enqueue((single_blocks['airway'],
                                    single_blocks['artery'],
                                    single_blocks['lung'],
                                    single_blocks['original'],
                                    ))
        (airway_block, artery_block, lung_block, original_block) = queue.dequeue()
        qr = tf.train.QueueRunner(queue, [enqueue_op] * 2)

        # test data part
        records_test = ut.get_records(record_dir_test)
        records_processor_test = TF_Records(records_test,block_shape)
        single_blocks_test = records_processor_test.read_records()
        queue_test = tf.RandomShuffleQueue(capacity=8,min_after_dequeue=4,dtypes=(
            single_blocks_test['airway'].dtype,
            single_blocks_test['artery'].dtype,
            single_blocks_test['lung'].dtype,
            single_blocks_test['original'].dtype,
        ))
        enqueue_op_test = queue_test.enqueue((
            single_blocks_test['airway'],
            single_blocks_test['artery'],
            single_blocks_test['lung'],
            single_blocks_test['original'],
        ))
        (airway_block_test, artery_block_test, lung_block_test, original_block_test) = queue_test.dequeue()
        qr_test = tf.train.QueueRunner(queue,[enqueue_op_test]*2)

        saver = tf.train.Saver(max_to_keep=1)
        config = tf.ConfigProto(allow_soft_placement=True)

        with tf.Session(config=config) as sess:

            # load variables if saved before
            if len(os.listdir(self.train_models_dir)) > 0:
                print "load saved model"
                sess.run(tf.group(tf.global_variables_initializer(),
                                  tf.local_variables_initializer()))
                saver.restore(sess, self.train_models_dir+"train_models.ckpt")
            else:
                sess.run(tf.group(tf.global_variables_initializer(),
                                  tf.local_variables_initializer()))

            # coord for the reading threads
            coord = tf.train.Coordinator()
            enqueue_threads = qr.create_threads(sess, coord=coord, start=True)
            enqueue_threads_test = qr_test.create_threads(sess,coord=coord,start=True)
            tf.train.start_queue_runners(sess=sess)

            summary_writer_test = tf.summary.FileWriter(self.test_sum_dir,sess.graph)
            summary_writer_train = tf.summary.FileWriter(self.train_sum_dir,sess.graph)

            # main train loop
            # for i in range(flags.max_iteration_num):
            try:
                for i in range(flags.max_iteration_num):
                    # organize a batch of data for training
                    # airway_np = np.zeros([batch_size_train,block_shape[0],block_shape[1],block_shape[2]], np.int16)
                    artery_np = np.zeros([batch_size_train,block_shape[0],block_shape[1],block_shape[2]], np.int16)
                    lung_np = np.zeros([batch_size_train,block_shape[0],block_shape[1],block_shape[2]], np.int16)
                    original_np = np.zeros([batch_size_train,block_shape[0],block_shape[1],block_shape[2]], np.int16)

                    # store values into data block
                    for m in range(flags.batch_size_train):
                        artery_data, lung_data, original_data = \
                            sess.run([artery_block, lung_block, original_block])
                        # airway_np[m, :, :, :] += airway_data
                        artery_np[m, :, :, :] += artery_data
                        lung_np[m, :, :, :] += lung_data
                        original_np[m, :, :, :] += original_data

                    train_ge_, train_dis_,step_num = sess.run([ge_train_op,dis_train_op, global_step],
                                                 feed_dict={X: original_np, lung_lable: lung_np,
                                                            artery_lable: artery_np, training: True})

                    if i%10==0:
                        sum_train, accuracy_artery, accuracy_lung, \
                        artery_l_val, lung_l_val, total_l_val,total_l_dis \
                            = sess.run([merge_summary_op, artery_acc, lung_acc,
                                        artery_loss, lung_loss, total_g_loss,gan_d_loss],
                                       feed_dict={X: original_np, lung_lable: lung_np,
                                                                artery_lable: artery_np, training: False})
                        summary_writer_train.add_summary(sum_train,global_step=int(step_num))
                        print "train :\nstep %d , lung loss = %f artery loss = %f total generator loss = %f total discriminator = %f \n\t\t\tlung accuracy = %f , artery accuracy = %f\n =====================" \
                              % (int(step_num), lung_l_val, artery_l_val, total_l_val,total_l_dis
                                 , accuracy_lung, accuracy_artery)

                    if i%test_step ==0 and i>0:
                        # block testing part
                        # airway_np_test = np.zeros([batch_size_train, block_shape[0], block_shape[1], block_shape[2]],
                        #                      np.int16)
                        artery_np_test = np.zeros([batch_size_train, block_shape[0], block_shape[1], block_shape[2]],
                                             np.int16)
                        lung_np_test = np.zeros([batch_size_train, block_shape[0], block_shape[1], block_shape[2]], np.int16)
                        original_np_test = np.zeros([batch_size_train, block_shape[0], block_shape[1], block_shape[2]],
                                               np.int16)

                        # store values into data block
                        for m in range(flags.batch_size_train):
                            artery_data_test, lung_data_test, original_data_test = \
                                sess.run([artery_block_test, lung_block_test, original_block_test])
                            # airway_np_test[m, :, :, :] += airway_data_test
                            artery_np_test[m, :, :, :] += artery_data_test
                            lung_np_test[m, :, :, :] += lung_data_test
                            original_np_test[m, :, :, :] += original_data_test

                        sum_test, accuracy_artery, accuracy_lung, \
                        artery_l_val, lung_l_val, total_l_val, \
                        artery_np_pred,artery_np_sig\
                            = sess.run([merge_summary_op, artery_acc, lung_acc,
                                        artery_loss, lung_loss, total_g_loss,
                                        artery_pred,artery_sig],
                                       feed_dict={X: original_np_test, lung_lable: lung_np_test,
                                                  artery_lable: artery_np_test, training: False})

                        summary_writer_test.add_summary(sum_test, global_step=int(step_num))
                        print "\ntest :\nstep %d , lung loss = %f artery loss = %f total loss = %f \n\t\tlung accuracy = %f , artery accuracy = %f\n=====================" \
                              % (int(step_num), lung_l_val, artery_l_val, total_l_val
                                 , accuracy_lung, accuracy_artery)
                        # print "airway percentage : ",str(np.float32(np.sum(np.float32(airway_np_test))/(flags.batch_size_train*block_shape[0]*block_shape[1]*block_shape[2])))
                        print "artery percentage : ",str(np.float32(np.sum(np.float32(artery_np_test))/(flags.batch_size_train*block_shape[0]*block_shape[1]*block_shape[2])))
                        # print "prediction of airway : maximum = ",np.max(airway_np_sig)," minimum = ",np.min(airway_np_sig)
                        print "prediction of artery : maximum = ",np.max(artery_np_sig)," minimum = ",np.min(artery_np_sig),'\n'
                        # print 'airway_log_mean = ',airway_log_mean,' airway_mask_mean = ',airway_mask_mean
                        # print 'lung_log_mean = ',lung_log_mean,' lung_mask_mean = ',lung_mask_mean
                        # print 'artery_log_mean = ',artery_log_mean,' artery_mask_mean = ',artery_mask_mean
                    if i%100 ==0:
                        saver.save(sess,self.train_models_dir+"train_models.ckpt")
                        print "regular model saved!"
            except Exception,e:
                print e
                # exit(2)
                coord.request_stop(e)
            coord.request_stop()
            coord.join(enqueue_threads)
            coord.join(enqueue_threads_test)

    def test(self,dicom_dir):
        flags = self.FLAGS
        block_shape = self.block_shape
        batch_size_test = self.batch_size_test
        data_type = "dicom_data"
        X = tf.placeholder(dtype=tf.float32, shape=[batch_size_test, block_shape[0], block_shape[1], block_shape[2]])
        training = tf.placeholder(tf.bool)
        with tf.variable_scope('generator'):
            _,__,artery_pred, artery_sig = self.Dense_Net(X, training,
                                                          flags.batch_size_test,
                                                          flags.accept_threshold+0.1)
        artery_pred = tf.reshape(artery_pred, [batch_size_test, block_shape[0], block_shape[1], block_shape[2]])

        # binary predict mask
        artery_pred_mask = tf.cast((artery_pred > 0.01), tf.float32)

        saver = tf.train.Saver(max_to_keep=1)
        config = tf.ConfigProto(allow_soft_placement=True)
        with tf.Session(config=config) as sess:
            # load variables if saved before
            if len(os.listdir(self.train_models_dir)) > 0:
                print "load saved model"
                sess.run(tf.group(tf.global_variables_initializer(),
                                  tf.local_variables_initializer()))
                saver.restore(sess, self.train_models_dir + "train_models.ckpt")
            else:
                print "no model detected from %s"%(self.train_models_dir)
                exit(1)

            test_data = tools.Test_data(dicom_dir, block_shape,data_type)
            test_data.organize_blocks()
            block_numbers = test_data.blocks.keys()
            blocks_num = len(block_numbers)
            print "block count: ",blocks_num
            time1 = time.time()
            sys.stdout.write("\r>>>deep learning calculating : %f" % (0.0)+"%")
            sys.stdout.flush()
            for i in range(0, blocks_num, batch_size_test):
                batch_numbers = []
                if i + batch_size_test < blocks_num:
                    temp_batch_size = batch_size_test
                else:
                    temp_batch_size = blocks_num - i
                temp_input = np.zeros(
                    [batch_size_test, block_shape[0], block_shape[1], block_shape[2]])
                for j in range(temp_batch_size):
                    temp_num = block_numbers[i + j]
                    temp_block = test_data.blocks[temp_num]
                    batch_numbers.append(temp_num)
                    block_array = temp_block.load_data()
                    data_block_shape = np.shape(block_array)
                    temp_input[j, 0:data_block_shape[0], 0:data_block_shape[1], 0:data_block_shape[2]] += block_array
                artery_predict = sess.run(artery_pred_mask,feed_dict={X: temp_input,training: False})
                for j in range(temp_batch_size):
                    test_data.upload_result(batch_numbers[j], artery_predict[j, :, :, :])
                if (i)%(batch_size_test*10)==0:
                    sys.stdout.write("\r>>>deep learning calculating : %f"%((1.0*i)*100/blocks_num)+"%")
                    sys.stdout.flush()

            sys.stdout.write("\r>>>deep learning calculating : %f" % (100.0)+"%")
            sys.stdout.flush()
            time2 = time.time()
            print "\ndeep learning time consume : ",str(time2-time1)
            time3 = time.time()
            test_result_array = test_data.get_result()
            print "result shape: ", np.shape(test_result_array)
            r_s = np.shape(test_result_array)  # result shape
            e_t = 10  # edge thickness
            to_be_transformed = np.zeros(r_s, np.float32)
            to_be_transformed[e_t:r_s[0] - e_t, e_t:r_s[1] - e_t, 0:r_s[2] - e_t] += test_result_array[
                                                                                     e_t:r_s[0] - e_t,
                                                                                     e_t:r_s[1] - e_t,
                                                                                     0:r_s[2] - e_t]
            print "maximum value in mask: ", np.max(to_be_transformed)
            print "minimum value in mask: ", np.min(to_be_transformed)
            final_img = ST.GetImageFromArray(np.transpose(to_be_transformed, [2, 1, 0]))
            final_img.SetSpacing(test_data.space)
            time4 = time.time()
            print "post processing time consume : ",str(time4-time3)
            print "writing final testing result"
            print './test_result/test_result_final.vtk'
            ST.WriteImage(final_img, './test_result/test_result_final.vtk')
            return final_img

if __name__ == "__main__":
    test_dicom_dir = '/opt/Multi-Task-data-process/multi_task_data_test/ZHANG_YU_KUN/original1'
    net = Network()
    # net.check_net()
    # net.train()
    time1 = time.time()
    net.test(test_dicom_dir)
    time2 = time.time()
    print "total time consume ",str(time2-time1)
