import tensorflow as tf

tf.app.flags.DEFINE_string(
    'dicom_root',"/opt/multi_task_data/",
    'Directory where original dicom files stored'
)

tf.app.flags.DEFINE_string(
    'record_dir',"/opt/Multi-Task-data-process/records",
    'Directory where tfrecord files will be stored'
)

tf.app.flags.DEFINE_integer(
    'block_shape_1',64,
    'shape of single data block'
)
tf.app.flags.DEFINE_integer(
    'block_shape_2',64,
    'shape of single data block'
)
tf.app.flags.DEFINE_integer(
    'block_shape_3',64,
    'shape of single data block'
)

tf.app.flags.DEFINE_integer(
    'batch_size_train',2,
    'batch size for training'
)

tf.app.flags.DEFINE_integer(
    'batch_size_test',4,
    'batch size for testing'
)

tf.app.flags.DEFINE_string(
    'train_models_dir','./train_models',
    'location to save trained models'
)

tf.app.flags.DEFINE_string(
    'summary_dir_train','./train_sum',
    'location to save tensorboard datas from training'
)

tf.app.flags.DEFINE_string(
    'test_result','./test_result',
    'location to save periodical results while training'
)

tf.app.flags.DEFINE_float(
    'airway_weight',0.5,
    'weight of airway segmentation branch while doing the training'
)

tf.app.flags.DEFINE_float(
    'artery_weight',0.5,
    'weight of artery segmentation branch while doing the training'
)

tf.app.flags.DEFINE_float(
    'lung_weight',0.5,
    'weight of lung segmentation branch while doing the training'
)

tf.app.flags.DEFINE_float(
    'accept_threshold',0.8,
    'threshold for judging if the corresponding pixel can be classfied as foreground'
)

tf.app.flags.DEFINE_float(
    'traing_rate_decay',0.99,
    'decay rate for training rate'
)