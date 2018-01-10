import tensorflow as tf
from tensorflow.python.lib.io.tf_record import TFRecordCompressionType
import numpy as np
import os
import SimpleITK as ST
import gc
from util import read_dicoms

def _int64_feature(values):
  if not isinstance(values, (tuple, list)):
    values = [values]
  return tf.train.Feature(int64_list=tf.train.Int64List(value=values))

def _bytes_feature(values):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))


class Data_dict():
    def __init__(self,meta):
        data_dict = {}
        for name in meta.keys():
            if not "name" in name:
                data_dict[name] = self.process_dicom(meta[name])
                # temp_img = ST.GetImageFromArray(np.transpose(data_dict[name],[2,1,0]))
                # ST.WriteImage(temp_img,meta[name]+'/'+name+'.vtk')
                # if "lung" in name:
                #     lung_array = self.process_dicom(meta[name])
                    # data_dict["back_ground"] = np.uint8(lung_array==0)
                    # back_ground_img = ST.GetImageFromArray(np.transpose(data_dict["back_ground"]),[2,1,0])
                    # ST.WriteImage(back_ground_img,meta[name]+'/back_ground.vtk')
            # else:
            #     data_dict[name] = meta[name]
        self.data_dict = data_dict

    def process_dicom(self,dicom_path):
        img = read_dicoms(dicom_path)
        array = np.transpose(ST.GetArrayFromImage(img),[2,1,0])
        return array

class Data():
    def __init__(self,data_meta,record_dir):
        self.data_meta = data_meta
        self.record_dir = record_dir
        self.saved_number = 0
        if not os.path.exists(self.record_dir):
            os.makedirs(self.record_dir)

    def get_single_meta(self,meta):
        try:
            del self.single_data
            gc.collect()
            self.single_data = Data_dict(meta)
        except Exception,e:
            self.single_data = Data_dict(meta)

    def process_data_single(self,block_shape):
        count = self.saved_number
        for meta in self.data_meta:
            self.get_single_meta(meta)

            open_switch = False
            checked = [True,True,True]
            last_shape = [0,0,0]
            for name,array in self.single_data.data_dict.items():
                if open_switch:
                    array_shape = np.shape(array)
                    for i in range(len(last_shape)):
                        checked[i] = (last_shape[i] == array_shape[i])
                else:
                    # first to be compared , record the shape and turn on the switch
                    array_shape = np.shape(array)
                    for i in range(len(last_shape)):
                        last_shape[i] = array_shape[i]
                    open_switch = True
            for i in range(len(last_shape)):
                if not checked[i]:
                    print "shapes from %s do not match"%(meta["project_name"])
                    continue
            print "shapes checked!!"

            # if np.sum(self.single_data.data_dict['lung']+self.single_data.data_dict['back_ground']) != \
            #         array_shape[0]*array_shape[1]*array_shape[2]:
            #     print "background mask not match!"
            #     continue
            print "background mask checked!!"
            print "convert data %s into record!"%(meta["project_name"])
            self.convert_to_record(block_shape,meta,count)
            count+=1

    def to_tfrecord(self,data_group):
        example = tf.train.Example(features=tf.train.Features(feature={
            'airway' : _bytes_feature(data_group['airway'].tostring()),
            'artery' : _bytes_feature(data_group['artery'].tostring()),
            'lung' : _bytes_feature(data_group['lung'].tostring()),
            'original' : _bytes_feature(data_group['original'].tostring()),
            # 'back_ground' : _bytes_feature(data_group['back_ground'].tostring())
        }))
        return example

    def convert_to_record(self,block_shape,meta,count):
        project_name = meta["project_name"]
        arrays={}
        for name in self.single_data.data_dict.keys():
            if not "name" in name:
                arrays[name] = self.single_data.data_dict[name]
        data_shape = np.shape(self.single_data.data_dict['original'])
        with tf.Graph().as_default(), tf.device('/gpu:0'):
            with tf.Session('') as sess:
                record_file_name = self.record_dir+'data_set_%04d_%s.tfrecord'%(count,project_name)
                options = tf.python_io.TFRecordOptions(TFRecordCompressionType.ZLIB)
                with tf.python_io.TFRecordWriter(record_file_name,options=options) as tfrecord_writer:
                    counter = 0
                    for i in range(0,data_shape[0],block_shape[0]/2):
                        for j in range(0,data_shape[1],block_shape[1]/2):
                            for k in range(0,data_shape[2],block_shape[2]/2):
                                if i+block_shape[0]/2<data_shape[0] and \
                                        j+block_shape[1]/2<data_shape[1] and k+block_shape[2]/2<data_shape[2]:
                                    data_group = {}

                                    # make sure that blocks extracted will be in the fixed shape by padding zeros
                                    tops = [i + block_shape[0], j + block_shape[1], k + block_shape[2]]
                                    for n in range(len(tops)):
                                        if tops[n] > data_shape[n]:
                                            tops[n] = data_shape[n]

                                    # extract original image first to see if this block is necessary
                                    temp_block = np.zeros(block_shape, np.int16)
                                    temp_block[:tops[0] - i, :tops[1] - j, :tops[2] - k] += \
                                        arrays['original'][i:tops[0], j:tops[1], k:tops[2]]
                                    original_block = temp_block
                                    data_group['original'] = temp_block

                                    # extract the rest masks if this block is necessary
                                    if not np.max(original_block) == np.min(original_block) == 0:
                                        for name in arrays.keys():
                                            if not 'origin' in name:
                                                temp_block = np.zeros(block_shape,np.int16)
                                                temp_block[:tops[0] - i, :tops[1] - j, :tops[2] - k]+= \
                                                    arrays[name][i:tops[0], j:tops[1], k:tops[2]]
                                                data_group[name] = temp_block
                                                if np.max(temp_block)>1 or np.min(temp_block)<0:
                                                    print "error occured at %s"%(str([i,j,k]))
                                        example = self.to_tfrecord(data_group)
                                        tfrecord_writer.write(example.SerializeToString())
                                        counter += 1
                print "data set number %04d has %d examples"%(count,counter)

class TF_Records():
    def __init__(self,records,block_shape):
        self.records = records
        self.block_shape = block_shape

    def read_records(self):
        if not isinstance(self.records,list):
            tfrecords_filename = [self.records]
        tfrecords_filename = self.records
        filename_queue = tf.train.string_input_producer(tfrecords_filename,num_epochs=10)

        ret = {}
        options = tf.python_io.TFRecordOptions(TFRecordCompressionType.ZLIB)
        reader = tf.TFRecordReader(options=options)
        _,serialized_example = reader.read(filename_queue)
        features = tf.parse_single_example(
            serialized_example,
            features={
                'airway' : tf.FixedLenFeature([],tf.string),
                'artery' : tf.FixedLenFeature([],tf.string),
                'lung' : tf.FixedLenFeature([],tf.string),
                'original' : tf.FixedLenFeature([],tf.string),
                # 'back_ground' : tf.FixedLenFeature([],tf.string)
            })
        ret['airway'] = tf.reshape(tf.decode_raw(features['airway'],tf.int16),self.block_shape)
        ret['artery'] = tf.reshape(tf.decode_raw(features['artery'],tf.int16),self.block_shape)
        ret['lung'] = tf.reshape(tf.decode_raw(features['lung'],tf.int16),self.block_shape)
        ret['original'] = tf.reshape(tf.decode_raw(features['original'],tf.int16),self.block_shape)
        # ret['back_ground'] =tf.reshape(tf.decode_raw(features['back_ground'],tf.uint8),self.block_shape)

        return ret