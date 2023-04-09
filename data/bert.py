from _base import _int64_feature, _float_feature, _byte_feature
from _base import parse_features
from _base import src_preprocessor, tgt_preprocessor

def createDataRecord(filename, dataset, **kwargs):

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
        example = tf.train.Example(features=ft)
        writer.write(example.SerializeToString())
        return ft

    dataset = dataset.map(preprocess_fn)
    dataset.map(to_tfrecord)
    writer.close()

    return 0

def input_fn_builder(tfrecord_path):
        # , seq_length, is_training, drop_remainder):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""

    def input_fn(params):
    """The actual input function."""
        batch_size = params["batch_size"]
        output_buffer_size = batch_size * 1000
        num_examples = len(features)

        dataset = tf.io.TFRecordWriter(tfrecord_path)
        dataset = dataset.map(parse_features, num_parallel_calls=4)
        dataset = dataset.prefetch(output_buffer_size)

        # [TODO] the `ds` indicates the tf.data.Dataset 
        # with tokenized inputs (and outputs

        if is_training:
            dataset = dataset.repeat()
            dataset = dataset.shuffle(buffer_size=1000)
            dataset = dataset.batch(batch_size=batch_size, drop_remainder=drop_remainder)
        else:
            pass

        return dataset

    return input_fn
