import datetime
import json
import os
import pprint
import random
import string
import sys
import tensorflow as tf
import data

def get_demo_dataset(N=1000):
    """ A demo dataset with 1000 examples. """
    sentA = [f'this is a {i} query for bertie testing.' for i in range(N)]
    sentB = [f'this is a {i} passage for bertie testing. In addition, this sentence is longer than the queries.' for i in range(N)]
    labels = random.choices([0, 1], k=N)
    data = {'query': sentA, 'document': sentB, 'label': labels}
    dataset = Dataset.from_dict(data)
    return dataset

from _base import _int64_feature, _float_feature, _byte_feature

def main(estimator):
    # Prepare TPU running config
	is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
	run_config = tf.contrib.tpu.RunConfig(
				cluster=tpu_cluster_resolver,
				model_dir=OUTPUT_DIR,
				save_checkpoints_steps=SAVE_CHECKPOINTS_STEPS,
				tpu_config=tf.contrib.tpu.TPUConfig(
					iterations_per_loop=ITERATIONS_PER_LOOP,
					num_shards=NUM_TPU_CORES,
					per_host_input_for_training=is_per_host)
				)
    tpu_cluster_resolver = None
    if FLAGS.use_tpu and FLAGS.tpu_name:
        tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
                FLAGS.tpu_name, 
                zone=FLAGS.tpu_zone, 
                project=FLAGS.gcp_project
        )
	is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2

    ## [NOTE] The following codes may need to move to 'model'
    BERT_MODEL = 'uncased_L-12_H-768_A-12' 
    BERT_MODEL_HUB = 'https://tfhub.dev/google/bert_' + BERT_MODEL + '/4'
    BERT_PRETRAINED_DIR = 'gs://cloud-tpu-checkpoints/bert/' + BERT_MODEL 
    print('***** BERT pretrained directory: {} *****'.format(BERT_PRETRAINED_DIR))

	# Setup configuration of data/model/train
	## data

	## training
	run_config = tf.contrib.tpu.RunConfig(
				cluster=tpu_cluster_resolver,
				model_dir=OUTPUT_DIR,
				save_checkpoints_steps=SAVE_CHECKPOINTS_STEPS,
				tpu_config=tf.contrib.tpu.TPUConfig(
					iterations_per_loop=ITERATIONS_PER_LOOP,
					num_shards=NUM_TPU_CORES,
					per_host_input_for_training=is_per_host)
				)

	## model
    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)
    INIT_CHECKPOINT = os.path.join(BERT_PRETRAINED_DIR, 'bert_model.ckpt')
    model_fn = model_fn_builder(
            bert_config=bert_config,
            init_checkpoint=FLAGS.init_checkpoint,
            learning_rate=FLAGS.learning_rate,
            num_train_steps=num_train_steps,
            num_warmup_steps=num_warmup_steps,
            use_tpu=FLAGS.use_tpu,
            use_one_hot_embeddings=FLAGS.use_tpu
    )
    OUTPUT_DIR = OUTPUT_DIR.replace('bert-tfhub', 'bert-checkpoints')
    tf.gfile.MakeDirs(OUTPUT_DIR)

    ## model -- initalize
    estimator = tf.contrib.tpu.TPUEstimator(
      use_tpu=FLAGS.use_tpu,
      model_fn=model_fn,
      config=run_config,
      train_batch_size=TRAIN_BATCH_SIZE,
      eval_batch_size=EVAL_BATCH_SIZE,
      predict_batch_size=PREDICT_BATCH_SIZE,
    )

    """ the training codes.
    TODO: 
    -----
    0. Customized dataset schema for hf dataset.
    1. Few training setup should be done within a config object.
    2. Adopt to huggingface API with multiprocessing.
    """


    print('***** Started loading dataset *****')
    if train_file.endswith('.tfrecord') is False:
        train_file = f"{train_file.rsplit('.', 1)[0]}.tfrecord"
        dataset = get_demo_dataset()

        print('  convert hf dataset into tfrecord, and will save at {}'.format(train_file))
        createDataRecord(train_file, dataset)
        print('  .tfrecord is saved.')

    print('***** Started training at {} *****'.format(datetime.datetime.now()))
    print('  Num examples = {}'.format(len(dataset)))
    print('  Batch size = {}'.format(TRAIN_BATCH_SIZE))
    tf.logging.info("  num steps = %d", num_train_steps)

    train_input_fn = data.t5.input_fn_builder(train_file)
    estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)

    print('***** finished training at {} *****'.format(datetime.datetime.now()))

    if DO_OUTPUT:
        tf.logging.info("***** Output prediction at {} *****".format(EMBEDDING_FILE))
        tf.logging.info("  Batch size = %d", EVAL_BATCH_SIZE)

        max_eval_examples = None
        max_eval_examples = 100 # debugging

        if eval_file.endswith('.tfrecord') is False:
            eval_file = f"{eval_file.rsplit('.', 1)[0]}.tfrecord"
            dataset = get_demo_dataset()

            print('***** Started converting dataset into tfrecord at {} *****'.format(train_file))
            createDataRecord(train_file, dataset)
            print("  .tfrecord is saved at {}.".format(train_file))

        print('***** Started prediction *****'.format(datetime.datetime.now()))
        eval_input_fn = data.t5.input_fn_builder(
                eval_file, is_train=False
        ) # the difference of output and eval.

        writer = tf.io.TFRecordWriter(f"{eval_file.rsplit('.', 1)[0]}.embeddings.tf")

        tf.logging.set_verbosity(tf.logging.WARN)

		outputs = estimator.predict(
		        input_fn=eval_input_fn,
		        yield_single_examples=True, 
		        checkpoint_path=EVAL_CHECKPOINT
		)

		start_time = time.time()
		# collecting outputs
		for output in tqdm(outputs):
		    # pooled embeddings (specified during train)
		    pooled_emb = output['pooling_embeddings'].astype('float16')
            pooled_emb = pooled_emb.reshape(-1).tostring()

		    fts = tf.train.Features(feature={
		        "doc_id": _int64_feature(output['docid']),
		        "doc_emb": _byte_feature([pooled_emb]),
            })
            example = tf.train.Example(features=fts)
            writer.write(example.SerializeToString())

        tf.logging.warm(" Latency = %.4f", (time.time() - start_time))
        writer.close()
		    

if __name__ == "__main__":
    flags.mark_flag_as_required("data_dir")
    flags.mark_flag_as_required("task_name")
    flags.mark_flag_as_required("vocab_file")
    flags.mark_flag_as_required("bert_config_file")
    flags.mark_flag_as_required("output_dir")
    tf.app.run()
