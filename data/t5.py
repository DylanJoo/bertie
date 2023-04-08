from base import _int64_feature, _float_feature, _byte_feature
from base import parse_features


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
    dataset = tf.io.TFRecordWriter(tfrecord_path)
    dataset = dataset.map(
            extract_fn, num_parallel_calls=4
    ).prefetch(output_buffer_size)

  def input_fn(params):
    """The actual input function."""
    batch_size = params["batch_size"]

    num_examples = len(features)

    # [TODO] the `ds` indicates the tf.data.Dataset 
    # with tokenized inputs (and outputs
    # not use Dataset.from_generator() because that uses tf.py_func which is
    # not TPU compatible. The right way to load data is with TFRecordReader.
    d = tf.data.Dataset.from_tensor_slices({
        "input_ids":
            tf.constant(
                all_input_ids, shape=[num_examples, seq_length],
                dtype=tf.int32),
        "input_mask":
            tf.constant(
                all_input_mask,
                shape=[num_examples, seq_length],
                dtype=tf.int32),
        "segment_ids":
            tf.constant(
                all_segment_ids,
                shape=[num_examples, seq_length],
                dtype=tf.int32),
        "label_ids":
            tf.constant(all_label_ids, shape=[num_examples], dtype=tf.int32),
    })

    if is_training:
      d = d.repeat()
      d = d.shuffle(buffer_size=100)

    d = d.batch(batch_size=batch_size, drop_remainder=drop_remainder)
    return d

  return input_fn
