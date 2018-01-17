import tensorflow as tf

tf.app.flags.DEFINE_string(
    'dicom_root',"/opt/multi_task_data/",
    'Directory where original dicom files stored'
)

tf.app.flags.DEFINE_string(
    'record_dir',"/opt/Multi-Task-data-process/records/",
    'Directory where tfrecord files will be stored'
)

tf.app.flags.DEFINE_string(
    'record_test_dir','/opt/Multi-Task-data-process/records_test/',
    'Directory where test data stored'
)

tf.app.flags.DEFINE_integer(
    'block_shape_1',48,
    'shape of single data block'
)
tf.app.flags.DEFINE_integer(
    'block_shape_2',48,
    'shape of single data block'
)
tf.app.flags.DEFINE_integer(
    'block_shape_3',48,
    'shape of single data block'
)

tf.app.flags.DEFINE_integer(
    'batch_size_train',2,
    'batch size for training'
)

tf.app.flags.DEFINE_integer(
    'batch_size_test',2,
    'batch size for testing'
)

tf.app.flags.DEFINE_integer(
    'max_iteration_num',5000000,
    'maximum training step'
)

tf.app.flags.DEFINE_integer(
    'test_step',20,
    'steps period that will do testing'
)

tf.app.flags.DEFINE_string(
    'train_models_dir','./train_models/',
    'location to save trained models'
)

tf.app.flags.DEFINE_string(
    'summary_dir_train','./train_sum/',
    'location to save tensorboard datas from training'
)

tf.app.flags.DEFINE_string(
    'summary_dir_test','./test_sum/',
    'location to save tensorboard datas from testing'
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
    'airway_fore_weight',0.9,
    'foreground weight for lost function of airway part'
)

tf.app.flags.DEFINE_float(
    'artery_weight',4,
    'weight of artery segmentation branch while doing the training'
)

tf.app.flags.DEFINE_float(
    'artery_fore_weight',0.9,
    'foreground weight for lost function of artery part'
)

tf.app.flags.DEFINE_float(
    'lung_weight',0.5,
    'weight of lung segmentation branch while doing the training'
)

tf.app.flags.DEFINE_float(
    'lung_fore_weight',0.7,
    'foreground weight for lost function of lung part'
)

tf.app.flags.DEFINE_float(
    'accept_threshold',0.8,
    'threshold for judging if the corresponding pixel can be classfied as foreground'
)

tf.app.flags.DEFINE_float(
    'training_rate_base',0.0001,
    'original learning rate'
)

tf.app.flags.DEFINE_float(
    'training_rate_decay',0.9,
    'decay rate for training rate'
)
