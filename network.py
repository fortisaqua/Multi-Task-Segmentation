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
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

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
        depth_down = 4

        if 'lung' in name:
            growth_up = 6
            depth_up = 0
        if 'airway' in name:
            growth_up = 6
            depth_up = 2
        else:
            growth_up = 6
            depth_up = 4

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
            return modified_segment

    def Dense_Net(self,inputs,training,batch_size,threshold):
        original_down = 16
        growth_down = 12
        depth_down = 4
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
            inputs = tf.placeholder(dtype=tf.float32,shape=[net.batch_size_test,block_shape[0],block_shape[1],block_shape[2]])
            training = tf.placeholder(tf.bool)
            lung,airway,artery = net.Dense_Net(inputs,training,net.batch_size_test,FLAGS.accept_threshold)
            lung_pred_mask = tf.cast((lung > 0.1), tf.float32)
            airway_pred_mask = tf.cast((airway > 0.1), tf.float32)
            artery_pred_mask = tf.cast((artery > 0.1), tf.float32)
            fake_mean = tf.reduce_mean(lung+airway+artery)
            tf.summary.scalar('fake_mean',fake_mean)
            merge_summary_op = tf.summary.merge_all()
            # a way to get shape from a tensor and convert it into a list
            print [int(d) for d in lung.get_shape()]
            print [int(d) for d in airway.get_shape()]
            print [int(d) for d in artery.get_shape()]
            print "net has ",tools.Ops.variable_count()," variables"
            sess = tf.Session()
            summary_writer = tf.summary.FileWriter(self.train_sum_dir, sess.graph)
            init_op = tf.group(tf.global_variables_initializer(),
                               tf.local_variables_initializer())
            sess.run(init_op)
            lung_np,airway_np,artey_np,merge_summary\
                = sess.run([lung,airway,artery,merge_summary_op],
                           feed_dict={inputs:np.int16(np.random.rand(net.batch_size_test,block_shape[0],block_shape[1],block_shape[2])*100),training:False})
            summary_writer.add_summary(merge_summary, global_step=0)
            print np.max(lung_np),"  ",np.min(lung_np)
            print np.max(airway_np),"  ",np.min(airway_np)
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
        model_dir = self.train_models_dir
        test_step = self.test_step
        LEARNING_RATE_BASE = flags.training_rate_base
        LEARNING_RATE_DECAY = flags.training_rate_decay
        X = tf.placeholder(dtype=tf.float32,shape=[batch_size_train,block_shape[0],block_shape[1],block_shape[2]])
        training = tf.placeholder(tf.bool)
        lung_pred,airway_pred,artery_pred = self.Dense_Net(X,training,flags.batch_size_train,flags.accept_threshold)
        lung_pred = tf.reshape(lung_pred,[batch_size_train,block_shape[0],block_shape[1],block_shape[2]])
        airway_pred = tf.reshape(airway_pred,[batch_size_train,block_shape[0],block_shape[1],block_shape[2]])
        artery_pred = tf.reshape(artery_pred,[batch_size_train,block_shape[0],block_shape[1],block_shape[2]])

        # binary predict mask
        lung_pred_mask = tf.cast((lung_pred>0.01),tf.float32)
        airway_pred_mask = tf.cast((airway_pred>0.01),tf.float32)
        artery_pred_mask = tf.cast((artery_pred>0.01),tf.float32)

        # labels
        lung_lable = tf.placeholder(dtype=tf.float32,shape=[batch_size_train,block_shape[0],block_shape[1],block_shape[2]])
        airway_lable = tf.placeholder(dtype=tf.float32,shape=[batch_size_train,block_shape[0],block_shape[1],block_shape[2]])
        artery_lable = tf.placeholder(dtype=tf.float32,shape=[batch_size_train,block_shape[0],block_shape[1],block_shape[2]])

        # accuracy
        lung_acc = 2*tf.reduce_sum(lung_lable*lung_pred_mask)/(tf.reduce_sum(lung_lable+lung_pred_mask)+1e-6)
        tf.summary.scalar('lung_acc', lung_acc)
        airway_acc = 2*tf.reduce_sum(airway_lable*airway_pred_mask)/(tf.reduce_sum(airway_lable+airway_pred_mask)+1e-6)
        tf.summary.scalar('airway_acc', airway_acc)
        artery_acc = 2*tf.reduce_sum(artery_lable*artery_pred_mask)/(tf.reduce_sum(artery_lable+artery_pred_mask)+1e-6)
        tf.summary.scalar('artery_acc', artery_acc)

        # loss function
        w_fore_lung = flags.lung_fore_weight
        w_fore_airway = flags.airway_fore_weight
        w_fore_artery = flags.artery_fore_weight

        # lung loss
        lung_loss = flags.lung_weight*tf.reduce_mean( -tf.reduce_mean(w_fore_lung*lung_lable*tf.log(lung_pred + 1e-8),reduction_indices=[1]) -
                                                       tf.reduce_mean((1-w_fore_lung)*(1-lung_lable)*tf.log(1-lung_pred + 1e-8),reduction_indices=[1]))
        tf.summary.scalar('lung_loss', lung_loss)
        # predict_mean_lung = tf.reduce_mean(tf.log(1-lung_pred + 1e-8),reduction_indices=[1])
        # mask_mean_lung = tf.reduce_mean((1-lung_lable))

        #  airway loss
        airway_loss = flags.airway_weight*tf.reduce_mean( -tf.reduce_mean(w_fore_airway*airway_lable*tf.log(airway_pred + 1e-8),reduction_indices=[1]) -
                                                       tf.reduce_mean((1-w_fore_airway)*(1-airway_lable)*tf.log(1-airway_pred + 1e-8),reduction_indices=[1]))
        tf.summary.scalar('airway_loss', airway_loss)
        # predict_mean_airway = tf.reduce_mean(tf.log(1-airway_pred + 1e-8),reduction_indices=[1])
        # mask_mean_airway = tf.reduce_mean((1 - airway_lable))

        # artery loss
        artery_loss = flags.artery_weight*tf.reduce_mean( -tf.reduce_mean(w_fore_artery*artery_lable*tf.log(artery_pred + 1e-8),reduction_indices=[1]) -
                                                       tf.reduce_mean((1-w_fore_artery)*(1-artery_lable)*tf.log(1-artery_pred + 1e-8),reduction_indices=[1]))
        tf.summary.scalar('artery_loss', artery_loss)
        # predict_mean_arte/ry = tf.reduce_mean(tf.log(1 - artery_pred + 1e-8), reduction_indices=[1])
        # mask_mean_artery = tf.reduce_mean((1 - artery_lable))

        # total loss
        total_loss = lung_loss+airway_loss+artery_loss
        tf.summary.scalar('total_loss',total_loss)

        # set training step and learning rate into tensors to save
        global_step = tf.Variable(0, trainable=False)
        learning_rate = tf.maximum(tf.train.exponential_decay(LEARNING_RATE_BASE, global_step, 33522/flags.batch_size_train,LEARNING_RATE_DECAY, staircase=True), 1e-9)

        # merge operation for tensorboard summary
        merge_summary_op = tf.summary.merge_all()
        # trainer
        train_op = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8).minimize(total_loss, global_step)

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
        qr = tf.train.QueueRunner(queue, [enqueue_op] * 4)

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
                saver.restore(sess, self.train_models_dir)
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
                    airway_np = np.zeros([batch_size_train,block_shape[0],block_shape[1],block_shape[2]], np.int16)
                    artery_np = np.zeros([batch_size_train,block_shape[0],block_shape[1],block_shape[2]], np.int16)
                    lung_np = np.zeros([batch_size_train,block_shape[0],block_shape[1],block_shape[2]], np.int16)
                    original_np = np.zeros([batch_size_train,block_shape[0],block_shape[1],block_shape[2]], np.int16)

                    # store values into data block
                    for m in range(flags.batch_size_train):
                        airway_data, artery_data, lung_data, original_data = \
                            sess.run([airway_block, artery_block, lung_block, original_block])
                        airway_np[m, :, :, :] += airway_data
                        artery_np[m, :, :, :] += artery_data
                        lung_np[m, :, :, :] += lung_data
                        original_np[m, :, :, :] += original_data

                    train__, step_num = sess.run([train_op, global_step],
                                                 feed_dict={X: original_np, lung_lable: lung_np, airway_lable: airway_np,
                                                            artery_lable: artery_np, training: True})

                    if i%5==0:
                        sum_train, accuracy_airway, accuracy_artery, accuracy_lung, \
                        airway_l_val, artery_l_val, lung_l_val, total_l_val \
                            = sess.run([merge_summary_op, airway_acc, artery_acc, lung_acc,
                                        airway_loss, artery_loss, lung_loss, total_loss],
                                       feed_dict={X: original_np, lung_lable: lung_np, airway_lable: airway_np,
                                                                artery_lable: artery_np, training: False})
                        summary_writer_train.add_summary(sum_train,global_step=int(step_num))
                        print "train :\nstep %d , total loss = %f lung loss = %f airway loss = %f artery loss = %f \nairway accuracy = %f , artery accuracy = %f , lung accuracy = %f\n=============\n" \
                              % (int(step_num), total_l_val, lung_l_val, airway_l_val, artery_l_val
                                 , accuracy_airway, accuracy_artery, accuracy_lung)

                    if i%test_step ==0 and i>0:
                        # block testing part
                        airway_np_test = np.zeros([batch_size_test, block_shape[0], block_shape[1], block_shape[2]],
                                             np.int16)
                        artery_np_test = np.zeros([batch_size_test, block_shape[0], block_shape[1], block_shape[2]],
                                             np.int16)
                        lung_np_test = np.zeros([batch_size_test, block_shape[0], block_shape[1], block_shape[2]], np.int16)
                        original_np_test = np.zeros([batch_size_test, block_shape[0], block_shape[1], block_shape[2]],
                                               np.int16)

                        # store values into data block
                        for m in range(flags.batch_size_train):
                            airway_data_test, artery_data_test, lung_data_test, original_data_test = \
                                sess.run([airway_block_test, artery_block_test, lung_block_test, original_block_test])
                            airway_np_test[m, :, :, :] += airway_data_test
                            artery_np_test[m, :, :, :] += artery_data_test
                            lung_np_test[m, :, :, :] += lung_data_test
                            original_np_test[m, :, :, :] += original_data_test

                        sum_test, accuracy_airway, accuracy_artery, accuracy_lung, \
                        airway_l_val, artery_l_val, lung_l_val, total_l_val \
                            = sess.run([merge_summary_op, airway_acc, artery_acc, lung_acc,
                                        airway_loss, artery_loss, lung_loss, total_loss],
                                       feed_dict={X: original_np_test, lung_lable: lung_np_test, airway_lable: airway_np_test,
                                                  artery_lable: artery_np_test, training: False})

                        summary_writer_test.add_summary(sum_test, global_step=int(step_num/test_step))
                        print "test :\nstep %d , total loss = %f lung loss = %f airway loss = %f artery loss = %f \nairway accuracy = %f , artery accuracy = %f , lung accuracy = %f\n=============\n" \
                              % (int(step_num/test_step), total_l_val, lung_l_val, airway_l_val, artery_l_val
                                 , accuracy_airway, accuracy_artery, accuracy_lung)
                        # print 'airway_log_mean = ',airway_log_mean,' airway_mask_mean = ',airway_mask_mean
                        # print 'lung_log_mean = ',lung_log_mean,' lung_mask_mean = ',lung_mask_mean
                        # print 'artery_log_mean = ',artery_log_mean,' artery_mask_mean = ',artery_mask_mean
                    if i%50 ==0:
                        saver.save(sess,model_dir)
            except Exception,e:
                print e
                # exit(2)
                coord.request_stop(e)
            coord.request_stop()
            coord.join(enqueue_threads)
            coord.join(enqueue_threads_test)

net = Network()
# net.check_net()
net.train()
