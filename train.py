import datetime
import json
import os
import pprint
import random
import string
import sys
import tensorflow as tf

# model_fn = run_classifier_with_tfhub.model_fn_builder(
#   num_labels=len(label_list),
#   learning_rate=LEARNING_RATE,
#   num_train_steps=num_train_steps,
#   num_warmup_steps=num_warmup_steps,
#   use_tpu=True,
#   bert_hub_module_handle=BERT_MODEL_HUB
# )
#
# estimator_from_tfhub = tf.contrib.tpu.TPUEstimator(
#   use_tpu=True,
#   model_fn=model_fn,
#   config=get_run_config(OUTPUT_DIR),
#   train_batch_size=TRAIN_BATCH_SIZE,
#   eval_batch_size=EVAL_BATCH_SIZE,
#   predict_batch_size=PREDICT_BATCH_SIZE,
# )

def training(estimator):
    """
    The training stage.
    
    Args:
        train_batch_size (int)
        eval_batch_size (int)
        max_seq_length (int)

    TODO: 
    1. Few training setup should be done within a config object.
    2. Adopt to huggingface API with multiprocessing.
    """
    train_features = 0
    print('***** Started loading dataset at {} *****'.format(task_dir))
    ## [DEMO] the demo data
    N = 1000
    sentA = [f'this is a {i} query for bertie testing.' for i in range(N)]
    sentB = [f'this is a {i} passage for bertie testing. In addition, this sentence is longer than the queries.' for i in range(N)]
    labels = random.choices([0, 1], k=N)
    data = {'query': sentA, 'document': sentB, 'label': labels}
    dataset = Dataset.from_dict(data)

    print('***** Started training at {} *****'.format(datetime.datetime.now()))

    print('  Num examples = {}'.format(len(dataset)))
    print('  Batch size = {}'.format(TRAIN_BATCH_SIZE))
    tf.logging.info("  num steps = %d", num_train_steps)

    if not os.path.exists(train_file):
        createDataRecord(train_file, dataset)
        tf.logging.info("  .tfrecord dataset saved at {}.".format(train_file))
    train_input_fn = data.t5.input_fn_builder(train_file)

    # features=train_features,
    # seq_length=max_seq_length,
    # is_training=true,
    # drop_remainder=true
    estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)

    print('***** finished training at {} *****'.format(datetime.datetime.now()))

# Setup task specific model and TPU running config.
BERT_MODEL = 'uncased_L-12_H-768_A-12' 
BERT_MODEL_HUB = 'https://tfhub.dev/google/bert_' + BERT_MODEL + '/4'
BERT_PRETRAINED_DIR = 'gs://cloud-tpu-checkpoints/bert/' + BERT_MODEL 
print('***** BERT pretrained directory: {} *****'.format(BERT_PRETRAINED_DIR))

CONFIG_FILE = os.path.join(BERT_PRETRAINED_DIR, 'bert_config.json')
INIT_CHECKPOINT = os.path.join(BERT_PRETRAINED_DIR, 'bert_model.ckpt')

model_fn = run_classifier.model_fn_builder(
    bert_config=modeling.BertConfig.from_json_file(CONFIG_FILE),
    num_labels=len(label_list),
    init_checkpoint=INIT_CHECKPOINT,
    learning_rate=LEARNING_RATE,
    num_train_steps=num_train_steps,
    num_warmup_steps=num_warmup_steps,
    use_tpu=True,
    use_one_hot_embeddings=True
)

OUTPUT_DIR = OUTPUT_DIR.replace('bert-tfhub', 'bert-checkpoints')
tf.gfile.MakeDirs(OUTPUT_DIR)

estimator_from_checkpoints = tf.contrib.tpu.TPUEstimator(
  use_tpu=True,
  model_fn=model_fn,
  config=get_run_config(OUTPUT_DIR),
  train_batch_size=TRAIN_BATCH_SIZE,
  eval_batch_size=EVAL_BATCH_SIZE,
  predict_batch_size=PREDICT_BATCH_SIZE,
)


model_train(estimator_from_checkpoints)

# model_eval(estimator_from_checkpoints)
# model_predict(estimator_from_checkpoints)
# def model_eval(estimator):
#   # Eval the model.
#   eval_examples = processor.get_dev_examples(TASK_DATA_DIR)
#   eval_features = run_classifier.convert_examples_to_features(
#       eval_examples, label_list, MAX_SEQ_LENGTH, tokenizer)
#   print('***** Started evaluation at {} *****'.format(datetime.datetime.now()))
#   print('  Num examples = {}'.format(len(eval_examples)))
#   print('  Batch size = {}'.format(EVAL_BATCH_SIZE))
#
#   # Eval will be slightly WRONG on the TPU because it will truncate
#   # the last batch.
#   eval_steps = int(len(eval_examples) / EVAL_BATCH_SIZE)
#   eval_input_fn = run_classifier.input_fn_builder(
#       features=eval_features,
#       seq_length=MAX_SEQ_LENGTH,
#       is_training=False,
#       drop_remainder=True)
#   result = estimator.evaluate(input_fn=eval_input_fn, steps=eval_steps)
#   print('***** Finished evaluation at {} *****'.format(datetime.datetime.now()))
#   output_eval_file = os.path.join(OUTPUT_DIR, "eval_results.txt")
#   with tf.gfile.GFile(output_eval_file, "w") as writer:
#     print("***** Eval results *****")
#     for key in sorted(result.keys()):
#       print('  {} = {}'.format(key, str(result[key])))
#       writer.write("%s = %s\n" % (key, str(result[key])))
#
# model_eval(estimator_from_tfhub)
#
# def model_predict(estimator):
#   # Make predictions on a subset of eval examples
#   prediction_examples = processor.get_dev_examples(TASK_DATA_DIR)[:PREDICT_BATCH_SIZE]
#   input_features = run_classifier.convert_examples_to_features(prediction_examples, label_list, MAX_SEQ_LENGTH, tokenizer)
#   predict_input_fn = run_classifier.input_fn_builder(features=input_features, seq_length=MAX_SEQ_LENGTH, is_training=False, drop_remainder=True)
#   predictions = estimator.predict(predict_input_fn)
#
#   for example, prediction in zip(prediction_examples, predictions):
#     print('text_a: %s\ntext_b: %s\nlabel:%s\nprediction:%s\n' % (example.text_a, example.text_b, str(example.label), prediction['probabilities']))
#
# model_predict(estimator_from_tfhub)

