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
import lung_seg

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
            layer = tools.Ops.xxlu(layer, name='lrelu')
            layer = tools.Ops.conv3d(layer, k=3, out_c=growth, str=1, name=block_name + '_layer_' + str(i))
            temp_shape = layers[-1].get_shape()
            next_input = tf.concat([layer, layers[-1]], axis=4)
            layers.append(next_input)
        return tools.Ops.xxlu(tools.Ops.batch_norm(layers[-1], name_scope=scope+'_bn_output', training=training))

    def Segment_part(self,inputs,input_1,down_1,down_2,name,training,threshold):
        original_down = 16
        growth_down = 6
        depth_down = 10

        with tf.variable_scope(name+'_segment'):
            # up sample input 1
            up_sample_input_1 = tf.concat([inputs,down_2],axis=4,name='up_sample_input_1')
            # up sample 1
            up_sample_1 = tools.Ops.deconv3d(up_sample_input_1,k=2,
                                             out_c=original_down+2*(growth_down*depth_down),str=2,name='up_sample_1')

            up_sample_1_conv = tools.Ops.conv3d(up_sample_1, k=3, out_c=original_down+2*(growth_down*depth_down), str=1, name='up_sample_1_conv')
            up_bn_1 = tools.Ops.batch_norm(up_sample_1_conv,name_scope='up_bn_1',training=training)
            # dense block 4
            # dense_block_output_4 = self.dense_block(up_sample_1,growth_up,depth_up,'dense_block_4',training,scope='dense_block_4')

            # up sample input 2
            up_sample_input_2 = tf.concat([up_bn_1,down_1],axis=4,name='up_sample_input_2')
            # up sample 2
            up_sample_2 = tools.Ops.deconv3d(up_sample_input_2,k=2,
                                             out_c=original_down+2*(growth_down*depth_down),str=2,name='up_sample_2')
            up_sample_2_conv = tools.Ops.conv3d(up_sample_2, k=3, out_c=original_down+2*(growth_down*depth_down), str=1, name='up_sample_2_conv')
            # dense block 5
            # dense_block_output_5 = self.dense_block(up_sample_2,growth_up,depth_up,'dense_block_5',training,scope='dense_block_5')

            # segment input
            up_bn_2 = tools.Ops.batch_norm(up_sample_2_conv,name_scope='up_bn_2',training=training)
            segment_input = tf.concat([up_bn_2,input_1],axis=4,name='segment_input')
            # segment conv 1
            segment_conv_1 = tools.Ops.conv3d(segment_input,k=3,
                                              out_c=original_down+1*(growth_down*depth_down),str=1,name='segment_conv_1')
            # segment conv 2
            segment_bn_1 = tools.Ops.batch_norm(segment_conv_1,name_scope='bn_segment_1',training=training)
            relu_1 = tools.Ops.xxlu(segment_bn_1,name='lrelu')
            segment_conv_2 = tools.Ops.conv3d(relu_1,k=3,out_c=32,str=1,name='segment_conv_2')
            # segment input
            segment_input = tools.Ops.xxlu(tools.Ops.batch_norm(segment_conv_2,name_scope='bn_segment_2',training=training),name='lrelu')
            # segment predict
            segment_predict = tools.Ops.conv3d(segment_input,k=3,out_c=3,str=1,name='segment_predict')

            return segment_predict

    def Dense_Net(self,inputs,training,batch_size,threshold):
        original_down = 16
        growth_down = 6
        depth_down = 10
        casted_inputs = tf.cast(inputs,tf.float32)
        X = tf.reshape(casted_inputs,[batch_size,self.block_shape[0],self.block_shape[1],self.block_shape[2],1],name='input')

        with tf.variable_scope('feature_extract'):
            # dense block 1
            dense_block_input_1 = tools.Ops.conv3d(X,k=3,out_c=original_down,str=1,name='dense_block_input_1')
            # dense_block_output_1 = self.dense_block(dense_block_input_1,growth_down,depth_down,'dense_block_1',training,scope='dense_block_1')

            # down sample 1
            down_sample_1 = tools.Ops.conv3d(dense_block_input_1,k=2,out_c=original_down+1*(growth_down*depth_down),str=2,name='down_sample_1')

            # dense block 2
            dense_block_output_2 = self.dense_block(down_sample_1,growth_down,depth_down,'dense_block_2',training,scope='dense_block_2')

            # down sample 2
            down_sample_2 = tools.Ops.conv3d(dense_block_output_2,k=2,out_c=original_down+2*(growth_down*depth_down),str=2,name='down_sample_2')

            # dense block 3
            dense_block_output_3 = self.dense_block(down_sample_2,growth_down,depth_down,'dense_block_3',training,scope='dense_block_3')

        artery_predict = self.Segment_part(dense_block_output_3, dense_block_input_1, down_sample_1, down_sample_2,
                                         name='artery', training=training, threshold=threshold)

        return artery_predict

    # check if the network is correct
    def check_net(self):
        try:
            net = Network()
            block_shape = net.block_shape
            X = tf.placeholder(dtype=tf.float32,shape=[net.batch_size_train,block_shape[0],block_shape[1],block_shape[2]])
            training = tf.placeholder(tf.bool)
            with tf.variable_scope('generator'):
                artery_pred = net.Dense_Net(X,training,net.batch_size_train,FLAGS.accept_threshold)
            # airway_pred_mask = tf.cast((airway > 0.1), tf.float32)
            artery_pred_mask = tf.cast((artery_pred[:,:,:,:,0] > 0.1), tf.float32)
            fake_mean = tf.reduce_mean(artery_pred_mask)
            tf.summary.scalar('fake_mean',fake_mean)
            merge_summary_op = tf.summary.merge_all()
            # a way to get shape from a tensor and convert it into a list
            # print [int(d) for d in lung.get_shape()]
            # print [int(d) for d in airway.get_shape()]
            print [int(d) for d in artery_pred.get_shape()]
            print "net has ",tools.Ops.variable_count()," variables"
            sess = tf.Session()
            summary_writer = tf.summary.FileWriter(self.train_sum_dir, sess.graph)
            init_op = tf.group(tf.global_variables_initializer(),
                               tf.local_variables_initializer())
            sess.run(init_op)
            artey_np,merge_summary\
                = sess.run([artery_pred,merge_summary_op],
                           feed_dict={X:np.int16(np.random.rand(net.batch_size_train,block_shape[0],block_shape[1],block_shape[2])*100),training:False})
            summary_writer.add_summary(merge_summary, global_step=0)
            # print np.max(lung_np),"  ",np.min(lung_np)
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
        threashold = flags.accept_threshold
        LEARNING_RATE_BASE = flags.training_rate_base
        LEARNING_RATE_DECAY = flags.training_rate_decay
        weight_vec = tf.constant([flags.airway_weight,flags.artery_weight,flags.back_ground_weight],tf.float32)
        X = tf.placeholder(dtype=tf.float32,shape=[batch_size_train,block_shape[0],block_shape[1],block_shape[2]])
        training = tf.placeholder(tf.bool)
        with tf.variable_scope('network'):
            seg_pred = self.Dense_Net(X,training,flags.batch_size_train,flags.accept_threshold)

        # lost function
        '''
        lable vector: [airway,artery,background]
        '''
        lables = tf.placeholder(dtype=tf.float32,shape=[batch_size_train,block_shape[0],block_shape[1],block_shape[2],3])
        weight_map = tf.reduce_sum(tf.multiply(lables,weight_vec),4)
        loss_origin = tf.nn.softmax_cross_entropy_with_logits(logits=seg_pred,labels=lables)
        loss_weighted = weight_map*loss_origin
        loss = tf.reduce_mean(loss_weighted)
        tf.summary.scalar('loss', loss)

        # accuracy
        predict_softmax = tf.nn.softmax(seg_pred)
        pred_map = tf.argmax(predict_softmax,axis=-1)
        pred_map_bool = tf.equal(pred_map,1)
        artery_pred_mask = tf.cast(pred_map_bool,tf.float32)
        artery_lable = tf.cast(lables[:, :, :, :, 1],tf.float32)
        artery_acc = 2 * tf.reduce_sum(artery_lable * artery_pred_mask) / (
                tf.reduce_sum(artery_lable + artery_pred_mask))
        tf.summary.scalar('artery_block_acc', artery_acc)

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

        global_step = tf.Variable(0, trainable=False)
        learning_rate = tf.maximum(
            tf.train.exponential_decay(LEARNING_RATE_BASE, global_step, 13500 / flags.batch_size_train,
                                       LEARNING_RATE_DECAY, staircase=True), 1e-9)
        train_op = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8).minimize(loss,global_step)
        # merge operation for tensorboard summary
        merge_summary_op = tf.summary.merge_all()

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
            for i in range(flags.max_iteration_num):
                # organize a batch of data for training
                lable_np = np.zeros([batch_size_train,block_shape[0],block_shape[1],block_shape[2],3], np.int16)
                original_np = np.zeros([batch_size_train,block_shape[0],block_shape[1],block_shape[2]], np.int16)

                # store values into data block
                for m in range(flags.batch_size_train):
                    '''
                    lable vector: [airway,artery,background]
                    '''
                    artery_data, airway_data, original_data = \
                        sess.run([artery_block, airway_block, original_block])
                    airway_array = airway_data
                    artery_array = artery_data
                    back_ground_array = np.int16((airway_array+artery_array)==0)
                    check_array = airway_array+artery_array+back_ground_array
                    while not np.max(check_array) == np.min(check_array) ==1:
                        artery_data, airway_data, original_data = \
                            sess.run([artery_block, airway_block, original_block])
                        airway_array = airway_data
                        artery_array = artery_data
                        back_ground_array = np.int16((airway_array + artery_array) == 0)
                        check_array = airway_array + artery_array + back_ground_array
                    lable_np[m, :, :, :, 0] += airway_array
                    lable_np[m, :, :, :, 1] += artery_array
                    lable_np[m, :, :, :, 2] += back_ground_array
                    original_np[m, :, :, :] += original_data
                train_,step_num = sess.run([train_op, global_step],
                                                           feed_dict={X: original_np,
                                                                      lables: lable_np, training: True})
                if step_num%flags.full_test_step==0:
                #     full testing
                    print "****************************full testing******************************"
                    data_type = "dicom_data"
                    test_dicom_dir = '/opt/Multi-Task-data-process/multi_task_data_test/FU_LI_JUN/original1'
                    test_mask_dir = '/opt/Multi-Task-data-process/multi_task_data_test/FU_LI_JUN/artery'
                    test_mask = ut.read_dicoms(test_mask_dir)
                    test_mask_array = np.transpose(ST.GetArrayFromImage(test_mask),[2,1,0])
                    test_data = tools.Test_data(test_dicom_dir, block_shape, data_type)
                    test_data.organize_blocks()
                    block_numbers = test_data.blocks.keys()
                    blocks_num = len(block_numbers)
                    print "block count: ", blocks_num
                    time1 = time.time()
                    sys.stdout.write("\r>>>deep learning calculating : %f" % (0.0) + "%")
                    sys.stdout.flush()
                    for m in range(0, blocks_num, batch_size_train):
                        batch_numbers = []
                        if m + batch_size_train < blocks_num:
                            temp_batch_size = batch_size_train
                        else:
                            temp_batch_size = blocks_num - m
                        temp_input = np.zeros(
                            [batch_size_train, block_shape[0], block_shape[1], block_shape[2]])
                        for j in range(temp_batch_size):
                            temp_num = block_numbers[m + j]
                            temp_block = test_data.blocks[temp_num]
                            batch_numbers.append(temp_num)
                            block_array = temp_block.load_data()
                            data_block_shape = np.shape(block_array)
                            temp_input[j, 0:data_block_shape[0], 0:data_block_shape[1],
                            0:data_block_shape[2]] += block_array
                        artery_predict = sess.run(artery_pred_mask, feed_dict={X: temp_input, training: False})
                        for j in range(temp_batch_size):
                            test_data.upload_result(batch_numbers[j], artery_predict[j, :, :, :])
                        if (m) % (batch_size_train * 10) == 0:
                            sys.stdout.write(
                                "\r>>>deep learning calculating : %f" % ((1.0 * m) * 100 / blocks_num) + "%")
                            sys.stdout.flush()

                    sys.stdout.write("\r>>>deep learning calculating : %f" % (100.0) + "%")
                    sys.stdout.flush()
                    time2 = time.time()
                    print "\ndeep learning time consume : ", str(time2 - time1)
                    time3 = time.time()
                    test_result_array = test_data.get_result()
                    test_result_array = np.float32(test_result_array>=2)
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
                    print "post processing time consume : ", str(time4 - time3)
                    print "writing final testing result"
                    if not os.path.exists('./test_result'):
                        os.makedirs('./test_result')
                    print './test_result/test_result_'+str(step_num)+'.vtk'
                    ST.WriteImage(final_img, './test_result/test_result_'+str(step_num)+'.vtk')
                    total_accuracy = 2*np.sum(1.0*test_mask_array*to_be_transformed)/np.sum(1.0*(test_mask_array+to_be_transformed))
                    print "total IOU accuracy : ",total_accuracy
                    if i==0:
                        mask_img = ST.GetImageFromArray(np.transpose(test_mask_array,[2,1,0]))
                        mask_img.SetSpacing(test_data.space)
                        ST.WriteImage(mask_img,'./test_result/mask_img.vtk')
                    print "***********************full testing end*******************************"
                if i%10==0:
                    sum_train,\
                    l_val \
                        = sess.run([merge_summary_op,
                                    loss],
                                   feed_dict={X: original_np,
                                              lables: lable_np, training: False})
                    summary_writer_train.add_summary(sum_train, global_step=int(step_num))
                    print "train :\nstep %d , loss = %f\n =====================" \
                          % (int(step_num), l_val)
                if i%test_step ==0 and i>0:
                    lable_np_test = np.zeros([batch_size_train, block_shape[0], block_shape[1], block_shape[2], 3],
                                        np.int16)
                    original_np_test = np.zeros([batch_size_train, block_shape[0], block_shape[1], block_shape[2]],
                                           np.int16)
                    for m in range(flags.batch_size_train):
                        '''
                        lable vector: [airway,artery,background]
                        '''
                        artery_data, airway_data, original_data = \
                            sess.run([artery_block_test, airway_block_test, original_block_test])
                        airway_array = airway_data
                        artery_array = artery_data
                        back_ground_array = np.int16((airway_array + artery_array) == 0)
                        check_array = airway_array + artery_array + back_ground_array
                        while not np.max(check_array) == np.min(check_array) == 1:
                            artery_data, airway_data, original_data = \
                                sess.run([artery_block, airway_block, original_block])
                            airway_array = airway_data
                            artery_array = artery_data
                            back_ground_array = np.int16((airway_array + artery_array) == 0)
                            check_array = airway_array + artery_array + back_ground_array
                        lable_np_test[m, :, :, :, 0] += airway_array
                        lable_np_test[m, :, :, :, 1] += artery_array
                        lable_np_test[m, :, :, :, 2] += back_ground_array
                        original_np_test[m, :, :, :] += original_data
                    sum_test, accuracy_artery, l_val, predict_array = \
                        sess.run([merge_summary_op,artery_acc,loss,pred_map],
                            feed_dict={X: original_np_test,lables: lable_np_test, training: False})
                    summary_writer_test.add_summary(sum_test, global_step=int(step_num))
                    print "\ntest :\nstep %d , artery loss = %f \n\t artery block accuracy = %f\n=====================" \
                          % (int(step_num), l_val, accuracy_artery)
                    print "artery percentage : ", str(np.float32(np.sum(np.float32(lable_np_test[:, :, :, :, 1])) / (flags.batch_size_train * block_shape[0] * block_shape[1] * block_shape[2])))
                    # print "prediction of airway : maximum = ",np.max(airway_np_sig)," minimum = ",np.min(airway_np_sig)
                    print "prediction : maximum = ", np.max(predict_array), " minimum = ", np.min(predict_array)
                if i%100 ==0:
                    saver.save(sess,self.train_models_dir+"train_models.ckpt")
                    print "regular model saved! step count : ",step_num

            coord.request_stop()
            coord.join(enqueue_threads)
            coord.join(enqueue_threads_test)

    def test(self,dicom_dir):
        flags = self.FLAGS
        block_shape = self.block_shape
        batch_size_test = self.batch_size_test
        data_type = "vtk_data"
        X = tf.placeholder(dtype=tf.float32, shape=[batch_size_test, block_shape[0], block_shape[1], block_shape[2]])
        training = tf.placeholder(tf.bool)
        with tf.variable_scope('generator'):
            artery_pred = self.Dense_Net(X, training,
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
            lung_part = lung_seg.Lung_Seg(dicom_dir)
            test_data = tools.Test_data(lung_part, block_shape,data_type)
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
    net.train()
    time1 = time.time()
    net.test(test_dicom_dir)
    time2 = time.time()
    print "total time consume ",str(time2-time1)
