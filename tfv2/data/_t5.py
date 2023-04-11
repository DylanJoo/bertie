from _base import _int64_feature, _float_feature, _byte_feature
from _base import src_preprocessor, tgt_preprocessor

def createTFRecord(filename, dataset, **kwargs):

    # tokenize texts
    def preprocess_fn(ex):
        features = tokenizer(
                [src_preprocessor(text) for text in ex['source']], 
        )
        features['label_ids'] = tokenizer(
                [tgt_preprocessor(text) for text in ex['target']], 
        )
        return features

    # initalize a writer
    writer = tf.io.TFRecordWriter(filename)
    def to_tfrecord(ft):
        ft = {k: _int64_feature(v) for (k, v) in ft.items())
        fts = tf.train.Features(feature=ft)
        example = tf.train.Example(features=fts)
        writer.write(example.SerializeToString())
        return ft

    dataset = dataset.map(preprocess_fn)
    dataset.map(to_tfrecord)
    writer.close()

    return 0

def loadTFRecord(tfrecord_path, params):

    batch_size = params["batch_size"]
    output_buffer_size = batch_size * 1000

    dataset = tf.data.TFRecordDataset(tfrecord_path)
    dataset = dataset.map(parse_features, num_parallel_calls=4)
    dataset = dataset.prefetch(output_buffer_size)

    # [TODO] the `ds` indicates the tf.data.Dataset 
    # with tokenized inputs (and outputs

    if is_training:
        dataset = dataset.repeat()
        dataset = dataset.shuffle(buffer_size=1000)
    else:
        dataset = dataset.repeat()

    dataset = dataset.batch(
            batch_size=batch_size, 
            drop_remainder=drop_remainder
    )

    return dataset


def parse_features(x):
    features_schema = {
            "input_ids": tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
            "token_type_ids": tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
            "attention_mask": tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
            "label_ids": tf.io.FixedLenSequenceFeature([], tf.int32, allow_missing=True),
    }
    sample = tf.io.parse_single_example(x, features_schema)
    return sample
