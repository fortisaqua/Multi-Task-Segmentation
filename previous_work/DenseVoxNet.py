import os
import shutil
import tensorflow as tf
import scipy.io
import tools
import numpy as np
import time
import test
import SimpleITK as ST
from dicom_read import read_dicoms
import gc

resolution = 64
batch_size = 4
lr_down = [0.001,0.0002,0.0001]
ori_lr = 0.001
power = 0.9
GPU0 = '0'
input_shape = [64,64,128]
output_shape = [64,64,128]
type_num = 0
already_trained=545
epoch_walked=545
upper_threshold = 0.9

###############################################################
config={}
config['train_names'] = ['chair']
for name in config['train_names']:
    config['X_train_'+name] = './Data/'+name+'/train_25d/voxel_grids_64/'
    config['Y_train_'+name] = './Data/'+name+'/train_3d/voxel_grids_64/'

config['test_names']=['chair']
for name in config['test_names']:
    config['X_test_'+name] = './Data/'+name+'/test_25d/voxel_grids_64/'
    config['Y_test_'+name] = './Data/'+name+'/test_3d/voxel_grids_64/'

config['resolution'] = resolution
config['batch_size'] = batch_size
config['meta_path'] = '/opt/analyse_airway/data_meta.pkl'
config['data_size'] = input_shape

################################################################

class Network:
    def __init__(self):
        self.train_models_dir = './train_models/'
        self.train_sum_dir = './train_sum/'
        self.test_results_dir = './test_results/'
        self.test_sum_dir = './test_sum/'

        if os.path.exists(self.test_results_dir):
            shutil.rmtree(self.test_results_dir)
            print 'test_results_dir: deleted and then created!\n'
        os.makedirs(self.test_results_dir)

        if os.path.exists(self.train_models_dir):
            # shutil.rmtree(self.train_models_dir)
            print 'train_models_dir: existed! will be loaded! \n'
        # os.makedirs(self.train_models_dir)

        if os.path.exists(self.train_sum_dir):
            # shutil.rmtree(self.train_sum_dir)
            print 'train_sum_dir: existed! \n'
        # os.makedirs(self.train_sum_dir)

        if os.path.exists(self.test_sum_dir):
            shutil.rmtree(self.test_sum_dir)
            print 'test_sum_dir: deleted and then created!\n'
        os.makedirs(self.test_sum_dir)

    def ae_u(self,X,training,batch_size,threshold):
        original=16
        growth=10
        dense_layer_num=12
        # input layer
        X=tf.reshape(X,[batch_size,input_shape[0],input_shape[1],input_shape[2],1])
        # image reduce layer
        # conv_input_1=tools.Ops.conv3d(X,k=3,out_c=2,str=2,name='conv_input_down')
        # conv_input_normed=tools.Ops.batch_norm(conv_input_1, 'bn_dense_0_0', training=training)
        # network start
        conv_input=tools.Ops.conv3d(X,k=3,out_c=original,str=2,name='conv_input')
        with tf.device('/gpu:'+GPU0):
            ##### dense block 1
            c_e = []
            s_e = []
            layers_e=[]
            layers_e.append(conv_input)
            for i in range(dense_layer_num):
                c_e.append(original+growth*(i+1))
                s_e.append(1)
            for j in range(dense_layer_num):
                layer = tools.Ops.batch_norm(layers_e[-1], 'bn_dense_1_' + str(j), training=training)
                layer = tools.Ops.xxlu(layer, name='relu')
                layer = tools.Ops.conv3d(layer,k=3,out_c=growth,str=s_e[j],name='dense_1_'+str(j))
                next_input = tf.concat([layer,layers_e[-1]],axis=4)
                layers_e.append(next_input)

        # middle down sample
            mid_layer = tools.Ops.batch_norm(layers_e[-1], 'bn_mid', training=training)
            mid_layer = tools.Ops.xxlu(mid_layer,name='relu')
            mid_layer = tools.Ops.conv3d(mid_layer,k=1,out_c=original+growth*dense_layer_num,str=1,name='mid_conv')
            mid_layer_down = tools.Ops.maxpool3d(mid_layer,k=2,s=2,pad='SAME')

        ##### dense block
        with tf.device('/gpu:'+GPU0):
            c_d = []
            s_d = []
            layers_d = []
            layers_d.append(mid_layer_down)
            for i in range(dense_layer_num):
                c_d.append(original+growth*(dense_layer_num+i+1))
                s_d.append(1)
            for j in range(dense_layer_num):
                layer = tools.Ops.batch_norm(layers_d[-1],'bn_dense_2_'+str(j),training=training)
                layer = tools.Ops.xxlu(layer, name='relu')
                layer = tools.Ops.conv3d(layer,k=3,out_c=growth,str=s_d[j],name='dense_2_'+str(j))
                next_input = tf.concat([layer,layers_d[-1]],axis=4)
                layers_d.append(next_input)

            ##### final up-sampling
            bn_1 = tools.Ops.batch_norm(layers_d[-1],'bn_after_dense',training=training)
            relu_1 = tools.Ops.xxlu(bn_1 ,name='relu')
            conv_27 = tools.Ops.conv3d(relu_1,k=1,out_c=original+growth*dense_layer_num*2,str=1,name='conv_up_sample_1')
            deconv_1 = tools.Ops.deconv3d(conv_27,k=2,out_c=128,str=2,name='deconv_up_sample_1')
            concat_up = tf.concat([deconv_1,mid_layer],axis=4)
            deconv_2 = tools.Ops.deconv3d(concat_up,k=2,out_c=64,str=2,name='deconv_up_sample_2')

            predict_map = tools.Ops.conv3d(deconv_2,k=1,out_c=1,str=1,name='predict_map')

            # zoom in layer
            # predict_map_normed = tools.Ops.batch_norm(predict_map,'bn_after_dense_1',training=training)
            # predict_map_zoomed = tools.Ops.deconv3d(predict_map_normed,k=2,out_c=1,str=2,name='deconv_zoom_3')

            vox_no_sig = predict_map
            # vox_no_sig = tools.Ops.xxlu(vox_no_sig,name='relu')
            vox_sig = tf.sigmoid(predict_map)
            vox_sig_modified = tf.maximum(vox_sig-threshold,0.01)
        return vox_sig, vox_sig_modified,vox_no_sig

    def dis(self, X, Y,training):
        with tf.device('/gpu:'+GPU0):
            X = tf.reshape(X,[batch_size,input_shape[0],input_shape[1],input_shape[2],1])
            Y = tf.reshape(Y,[batch_size,output_shape[0],output_shape[1],output_shape[2],1])
            layer = tf.concat([X,Y],axis=4)
            c_d = [1,2,64,128,256,512]
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

    def train(self,configure):
        data = tools.Data(configure,epoch_walked)
        best_acc = 0
        # X = tf.placeholder(shape=[batch_size, input_shape[0], input_shape[1], input_shape[2]], dtype=tf.float32)
        X = tf.placeholder(shape=[batch_size, input_shape[0], input_shape[1], input_shape[2]], dtype=tf.float32)
        # Y = tf.placeholder(shape=[batch_size, output_shape[0], output_shape[1], output_shape[2]], dtype=tf.float32)
        Y = tf.placeholder(shape=[batch_size, output_shape[0], output_shape[1], output_shape[2]], dtype=tf.float32)
        print X.get_shape()
        lr = tf.placeholder(tf.float32)
        training = tf.placeholder(tf.bool)
        threshold = tf.placeholder(tf.float32)
        with tf.variable_scope('ae'):
            Y_pred, Y_pred_modi,Y_pred_nosig = self.ae_u(X,training,batch_size,threshold)

        with tf.variable_scope('dis'):
            XY_real_pair = self.dis(X, Y,training)
        with tf.variable_scope('dis',reuse=True):
            XY_fake_pair = self.dis(X, Y_pred,training)

        with tf.device('/gpu:'+GPU0):
            ################################ ae loss
            Y_ = tf.reshape(Y,shape=[batch_size,-1])
            Y_pred_modi_ = tf.reshape(Y_pred_modi,shape=[batch_size,-1])
            w = tf.placeholder(tf.float32) # power of foreground against background
            ae_loss = tf.reduce_mean( -tf.reduce_mean(w*Y_*tf.log(Y_pred_modi_ + 1e-8),reduction_indices=[1]) -
                                      tf.reduce_mean((1-w)*(1-Y_)*tf.log(1-Y_pred_modi_ + 1e-8), reduction_indices=[1]) )
            sum_ae_loss = tf.summary.scalar('ae_loss', ae_loss)

            ################################ wgan loss
            gan_g_loss = -tf.reduce_mean(XY_fake_pair)
            gan_d_loss = tf.reduce_mean(XY_fake_pair) - tf.reduce_mean(XY_real_pair)
            sum_gan_g_loss = tf.summary.scalar('gan_g_loss',gan_g_loss)
            sum_gan_d_loss = tf.summary.scalar('gan_d_loss',gan_d_loss)
            alpha = tf.random_uniform(shape=[batch_size,input_shape[0]*input_shape[1]*input_shape[2]],minval=0.0,maxval=1.0)

            Y_pred_ = tf.reshape(Y_pred,shape=[batch_size,-1])
            differences_ = Y_pred_ -Y_
            interpolates = Y_ + alpha*differences_
            with tf.variable_scope('dis',reuse=True):
                XY_fake_intep = self.dis(X, interpolates,training)
            gradients = tf.gradients(XY_fake_intep,[interpolates])[0]
            slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients),reduction_indices=[1]))
            gradient_penalty = tf.reduce_mean((slopes-1.0)**2)
            gan_d_loss +=10*gradient_penalty

            #################################  ae + gan loss
            gan_g_w = 5
            ae_w = 100-gan_g_w
            ae_gan_g_loss = ae_w * ae_loss + gan_g_w * gan_g_loss

        with tf.device('/gpu:' + GPU0):
            ae_var = [var for var in tf.trainable_variables() if var.name.startswith('ae')]
            dis_var = [var for var in tf.trainable_variables() if var.name.startswith('dis')]
            ae_g_optim = tf.train.AdamOptimizer(learning_rate=lr, beta1=0.9, beta2=0.999, epsilon=1e-8).minimize(ae_gan_g_loss, var_list=ae_var)
            dis_optim = tf.train.AdamOptimizer(learning_rate=lr,beta1=0.9,beta2=0.999,epsilon=1e-8).minimize(gan_d_loss,var_list=dis_var)

        print tools.Ops.variable_count()
        sum_merged = tf.summary.merge_all()

        saver = tf.train.Saver(max_to_keep=1)
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.visible_device_list = GPU0
        with tf.Session(config=config) as sess:
            # if os.path.exists(self.train_models_dir):
            #     try:
            #         saver.restore(sess,self.train_models_dir+'model.cptk')
            #     except Exception,e:
            #         saver.restore(sess,'./regular/'+'model.cptk')
            sum_writer_train = tf.summary.FileWriter(self.train_sum_dir, sess.graph)
            sum_write_test = tf.summary.FileWriter(self.test_sum_dir)

            if os.path.isfile(self.train_models_dir + 'model.cptk.data-00000-of-00001'):
                print "restoring saved model"
                saver.restore(sess, self.train_models_dir + 'model.cptk')
            else:
                sess.run(tf.global_variables_initializer())

            learning_rate_g = ori_lr*pow(power,(epoch_walked/4))
            for epoch in range(epoch_walked,15000):
                # data.shuffle_X_Y_files(label='train')
                #### select data randomly each 10 epochs
                if epoch % 2== 0 and epoch > 0:
                    del data
                    gc.collect()
                    data = tools.Data(configure, epoch)
                #### full testing
                # ...
                train_amount = len(data.train_numbers)
                test_amount = len(data.test_numbers)
                if train_amount>=test_amount and train_amount>0 and test_amount>0 and data.total_train_batch_num>0 and data.total_test_seq_batch>0:
                    weight_for = 0.35*(1-epoch*1.0/15000)+0.5
                    if epoch % 4 == 0:
                        print '********************** FULL TESTING ********************************'
                        time_begin = time.time()
                        lung_img = ST.ReadImage('./WANG_REN/lung_img.vtk')
                        mask_dir = "./WANG_REN/airway"
                        test_batch_size = batch_size
                        # test_data = tools.Test_data(dicom_dir,input_shape)
                        test_data = tools.Test_data(lung_img, input_shape, 'vtk_data')
                        test_data.organize_blocks()
                        test_mask = read_dicoms(mask_dir)
                        array_mask = ST.GetArrayFromImage(test_mask)
                        array_mask = np.transpose(array_mask, (2, 1, 0))
                        print "mask shape: ",np.shape(array_mask)
                        time1 = time.time()
                        block_numbers = test_data.blocks.keys()
                        for i in range(0,len(block_numbers),test_batch_size):
                            batch_numbers = []
                            if i+test_batch_size<len(block_numbers):
                                temp_input = np.zeros(
                                                [test_batch_size, input_shape[0], input_shape[1], input_shape[2]])
                                for j in range(test_batch_size):
                                    temp_num = block_numbers[i+j]
                                    temp_block = test_data.blocks[temp_num]
                                    batch_numbers.append(temp_num)
                                    block_array = temp_block.load_data()
                                    block_shape = np.shape(block_array)
                                    temp_input[j,0:block_shape[0],0:block_shape[1],0:block_shape[2]] += block_array
                                Y_temp_pred, Y_temp_modi, Y_temp_pred_nosig = sess.run([Y_pred, Y_pred_modi, Y_pred_nosig],
                                                                                           feed_dict={X: temp_input,
                                                                                                      training: False,
                                                                                                      w:weight_for,
                                                                                                      threshold:upper_threshold})
                                for j in range(test_batch_size):
                                    test_data.upload_result(batch_numbers[j],Y_temp_modi[j,:,:,:])
                            else:
                                temp_batch_size = len(block_numbers) - i
                                temp_input = np.zeros(
                                    [temp_batch_size, input_shape[0], input_shape[1], input_shape[2]])
                                for j in range(temp_batch_size):
                                    temp_num = block_numbers[i + j]
                                    temp_block = test_data.blocks[temp_num]
                                    batch_numbers.append(temp_num)
                                    block_array = temp_block.load_data()
                                    block_shape = np.shape(block_array)
                                    temp_input[j,0:block_shape[0],0:block_shape[1],0:block_shape[2]] += block_array
                                X_temp = tf.placeholder(
                                    shape=[temp_batch_size, input_shape[0], input_shape[1], input_shape[2]],
                                    dtype=tf.float32)
                                with tf.variable_scope('ae', reuse=True):
                                    Y_pred_temp, Y_pred_modi_temp, Y_pred_nosig_temp = self.ae_u(X_temp, training,
                                                                                                 temp_batch_size,threshold)
                                Y_temp_pred, Y_temp_modi, Y_temp_pred_nosig = sess.run([Y_pred_temp, Y_pred_modi_temp, Y_pred_nosig_temp],
                                                                                            feed_dict={X_temp: temp_input,
                                                                                                       training: False,
                                                                                                       w : weight_for,
                                                                                                       threshold : upper_threshold})
                                for j in range(temp_batch_size):
                                    test_data.upload_result(batch_numbers[j], Y_temp_modi[j, :, :, :])
                        test_result_array = test_data.get_result()
                        print "result shape: ",np.shape(test_result_array)
                        r_s = np.shape(test_result_array) # result shape
                        e_t = 10                          # edge thickness
                        to_be_transformed = np.zeros(r_s,np.float32)
                        to_be_transformed[e_t:r_s[0]-e_t,e_t:r_s[1]-e_t,e_t:r_s[2]-e_t]+=test_result_array[e_t:r_s[0]-e_t,e_t:r_s[1]-e_t,e_t:r_s[2]-e_t]
                        print np.max(to_be_transformed)
                        print np.min(to_be_transformed)
                        final_img = ST.GetImageFromArray(np.transpose(to_be_transformed, [2, 1, 0]))
                        final_img.SetSpacing(test_data.space)
                        print "writing full testing result"
                        print '/usr/analyse_airway/test_result/test_result' + str(epoch) + '.vtk'
                        ST.WriteImage(final_img, '/usr/analyse_airway/test_result/test_result' + str(epoch) + '.vtk')
                        if epoch==0:
                            mask_img = ST.GetImageFromArray(np.transpose(array_mask, [2, 1, 0]))
                            mask_img.SetSpacing(test_data.space)
                            ST.WriteImage(mask_img, '/usr/analyse_airway/test_result/test_mask.vtk')
                        test_IOU = 2*np.sum(to_be_transformed*array_mask)/(np.sum(to_be_transformed)+np.sum(array_mask))
                        print "IOU accuracy: ",test_IOU
                        time_end = time.time()
                        print '******************** time of full testing: '+str(time_end-time_begin)+'s ********************'
                    data.shuffle_X_Y_pairs()
                    total_train_batch_num = data.total_train_batch_num
                    # train_files=data.X_train_files
                    # test_files=data.X_test_files
                    # total_train_batch_num = 500
                    print "total_train_batch_num:", total_train_batch_num
                    for i in range(total_train_batch_num):

                        #### training
                        X_train_batch, Y_train_batch = data.load_X_Y_voxel_train_next_batch()
                        # X_train_batch, Y_train_batch = data.load_X_Y_voxel_grids_train_next_batch()
                        # Y_train_batch=np.reshape(Y_train_batch,[batch_size, output_shape[0], output_shape[1], output_shape[2], 1])
                        gan_d_loss_c, = sess.run([gan_d_loss],feed_dict={X: X_train_batch, Y: Y_train_batch,training:False, w: weight_for, threshold:upper_threshold})
                        ae_loss_c, gan_g_loss_c, sum_train = sess.run([ae_loss, gan_g_loss, sum_merged],feed_dict={X: X_train_batch, Y: Y_train_batch,training:False, w: weight_for, threshold:upper_threshold})
                        if epoch%4 == 0 and epoch>0 and i == 0:
                            learning_rate_g = learning_rate_g*power
                        sess.run([ae_g_optim],feed_dict={X: X_train_batch, threshold:upper_threshold, Y: Y_train_batch, lr: learning_rate_g, training: True, w: weight_for})
                        if epoch<=5:
                            sess.run([dis_optim], feed_dict={X: X_train_batch, threshold:upper_threshold, Y: Y_train_batch, lr: learning_rate_g,training:True, w: weight_for})
                        elif epoch<=20:
                            sess.run([dis_optim], feed_dict={X: X_train_batch, threshold:upper_threshold, Y: Y_train_batch, lr: learning_rate_g,training:True, w: weight_for})
                        else:
                            sess.run([dis_optim], feed_dict={X: X_train_batch, threshold:upper_threshold, Y: Y_train_batch, lr: learning_rate_g,training:True, w: weight_for})

                        sum_writer_train.add_summary(sum_train, epoch * total_train_batch_num + i)
                        if i%2==0:
                            print "epoch:", epoch, " i:", i, " train ae loss:", ae_loss_c," gan g loss:",gan_g_loss_c," gan d loss:",gan_d_loss_c," learning rate: ",learning_rate_g
                        #### testing
                        if i  %20== 0 and epoch %1 ==0 :
                            try:
                                X_test_batch, Y_test_batch = data.load_X_Y_voxel_test_next_batch(fix_sample=False)
                                # Y_test_batch = np.reshape(Y_test_batch,[batch_size, output_shape[0], output_shape[1], output_shape[2], 1])
                                ae_loss_t,gan_g_loss_t,gan_d_loss_t, Y_test_pred,Y_test_modi, Y_test_pred_nosig= \
                                    sess.run([ae_loss, gan_g_loss,gan_d_loss, Y_pred,Y_pred_modi,Y_pred_nosig],feed_dict={X: X_test_batch, threshold:upper_threshold, Y: Y_test_batch,training:False, w: weight_for})
                                predict_result = np.float32(Y_test_modi>0.01)
                                predict_result = np.reshape(predict_result,[batch_size,input_shape[0], input_shape[1], input_shape[2]])
                                # Foreground
                                # if np.sum(Y_test_batch)>0:
                                #     accuracy_for = np.sum(predict_result*Y_test_batch)/np.sum(Y_test_batch)
                                # Background
                                # accuracy_bac = np.sum((1-predict_result)*(1-Y_test_batch))/(np.sum(1-Y_test_batch))
                                # IOU
                                predict_probablity = np.float32((Y_test_modi-0.01)>0)
                                predict_probablity = np.reshape(predict_probablity,[batch_size,input_shape[0], input_shape[1], input_shape[2]])
                                accuracy = 2*np.sum(np.abs(predict_probablity*Y_test_batch))/np.sum(np.abs(predict_result)+np.abs(Y_test_batch))
                                # if epoch%30==0 and epoch>0:
                                #     to_save = {'X_test': X_test_batch, 'Y_test_pred': Y_test_pred,'Y_test_true': Y_test_batch}
                                #     scipy.io.savemat(self.test_results_dir + 'X_Y_pred_' + str(epoch).zfill(2) + '_' + str(i).zfill(4) + '.mat', to_save, do_compression=True)
                                print "epoch:", epoch, " i:","\nIOU accuracy: ",accuracy,"\ntest ae loss:", ae_loss_t, " gan g loss:",gan_g_loss_t," gan d loss:", gan_d_loss_t
                                if accuracy>best_acc:
                                    saver.save(sess, save_path=self.train_models_dir + 'model.cptk')
                                    print "epoch:", epoch, " i:", i, "best model saved!"
                                    best_acc = accuracy
                            except Exception,e:
                                print e
                        #### model saving
                        if i %30 == 0 and epoch%1==0:
                           # regular_train_dir = "./regular/"
                           # if not os.path.exists(regular_train_dir):
                           #     os.makedirs(regular_train_dir)
                           saver.save(sess, save_path=self.train_models_dir +'model.cptk')
                           print "epoch:", epoch, " i:", i, "regular model saved!"
                else:
                    print "bad data , next epoch",epoch

    def test(self,dicom_dir):
        # X = tf.placeholder(shape=[batch_size, input_shape[0], input_shape[1], input_shape[2]], dtype=tf.float32)
        test_input_shape = input_shape
        test_batch_size = batch_size*2
        threshold = tf.placeholder(tf.float32)
        training = tf.placeholder(tf.bool)
        X = tf.placeholder(shape=[test_batch_size, test_input_shape[0], test_input_shape[1], test_input_shape[2]],
                           dtype=tf.float32)
        with tf.variable_scope('ae'):
            Y_pred, Y_pred_modi, Y_pred_nosig = self.ae_u(X, training, test_batch_size, threshold)

        print tools.Ops.variable_count()
        sum_merged = tf.summary.merge_all()
        saver = tf.train.Saver(max_to_keep=1)
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.visible_device_list = GPU0
        with tf.Session(config=config) as sess:
            # if os.path.exists(self.train_models_dir):
            #     saver.restore(sess, self.train_models_dir + 'model.cptk')
            sum_writer_train = tf.summary.FileWriter(self.train_sum_dir, sess.graph)
            sum_write_test = tf.summary.FileWriter(self.test_sum_dir)

            if os.path.exists(self.train_models_dir) and os.path.isfile(self.train_models_dir + 'model.cptk.data-00000-of-00001'):
                print "restoring saved model"
                saver.restore(sess, self.train_models_dir + 'model.cptk')
            else:
                sess.run(tf.global_variables_initializer())
            test_data = tools.Test_data(dicom_dir, input_shape)
            test_data.organize_blocks()
            block_numbers = test_data.blocks.keys()
            for i in range(0, len(block_numbers), test_batch_size):
                batch_numbers = []
                if i + test_batch_size < len(block_numbers):
                    temp_input = np.zeros(
                        [test_batch_size, input_shape[0], input_shape[1], input_shape[2]])
                    for j in range(test_batch_size):
                        temp_num = block_numbers[i + j]
                        temp_block = test_data.blocks[temp_num]
                        batch_numbers.append(temp_num)
                        block_array = temp_block.load_data()
                        block_shape = np.shape(block_array)
                        temp_input[j, 0:block_shape[0], 0:block_shape[1], 0:block_shape[2]] += block_array
                    Y_temp_pred, Y_temp_modi, Y_temp_pred_nosig = sess.run([Y_pred, Y_pred_modi, Y_pred_nosig],
                                                                           feed_dict={X: temp_input,
                                                                                      training: False,
                                                                                      threshold: upper_threshold})
                    for j in range(test_batch_size):
                        test_data.upload_result(batch_numbers[j], Y_temp_modi[j, :, :, :])
                else:
                    temp_batch_size = len(block_numbers) - i
                    temp_input = np.zeros(
                        [temp_batch_size, input_shape[0], input_shape[1], input_shape[2]])
                    for j in range(temp_batch_size):
                        temp_num = block_numbers[i + j]
                        temp_block = test_data.blocks[temp_num]
                        batch_numbers.append(temp_num)
                        block_array = temp_block.load_data()
                        block_shape = np.shape(block_array)
                        temp_input[j, 0:block_shape[0], 0:block_shape[1], 0:block_shape[2]] += block_array
                    X_temp = tf.placeholder(
                        shape=[temp_batch_size, input_shape[0], input_shape[1], input_shape[2]],
                        dtype=tf.float32)
                    with tf.variable_scope('ae', reuse=True):
                        Y_pred_temp, Y_pred_modi_temp, Y_pred_nosig_temp = self.ae_u(X_temp, training,
                                                                                     temp_batch_size, threshold)
                    Y_temp_pred, Y_temp_modi, Y_temp_pred_nosig = sess.run(
                        [Y_pred_temp, Y_pred_modi_temp, Y_pred_nosig_temp],
                        feed_dict={X_temp: temp_input,
                                   training: False,
                                   threshold: upper_threshold})
                    for j in range(temp_batch_size):
                        test_data.upload_result(batch_numbers[j], Y_temp_modi[j, :, :, :])
            test_result_array = test_data.get_result()
            print "result shape: ", np.shape(test_result_array)
            r_s = np.shape(test_result_array)  # result shape
            e_t = 10  # edge thickness
            to_be_transformed = np.zeros(r_s, np.float32)
            to_be_transformed[e_t:r_s[0] - e_t, e_t:r_s[1] - e_t, 0:r_s[2] - e_t] += test_result_array[
                                                                                       e_t:r_s[0] - e_t,
                                                                                       e_t:r_s[1] - e_t,
                                                                                       0:r_s[2] - e_t]
            print np.max(to_be_transformed)
            print np.min(to_be_transformed)
            final_img = ST.GetImageFromArray(np.transpose(to_be_transformed, [2, 1, 0]))
            final_img.SetSpacing(test_data.space)
            print "writing final testing result"
            print './test_result/test_result_final.vtk'
            ST.WriteImage(final_img, './test_result/test_result_final.vtk')
            return final_img

if __name__ == "__main__":
    dicom_dir = "./WANG_REN/original1"
    net = Network()
    net.train(config)
    final_img = net.test(dicom_dir)
    # ST.WriteImage(final_img,'./final_result.vtk')
