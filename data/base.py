import tensorflow as tf

def _int64_feature(x):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=x))

def _byte_feature(x):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=x))

def _float_feature(x):
    return tf.train.Feature(float_list=tf.train.FloatList(value=x))

def parse_features(x):
    features_schema = {
            "input_ids": tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
            "token_type_ids": tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
            "attention_mask": tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
            "label_ids": tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
    }
    sample = tf.io.parse_single_example(x, features_schema)
