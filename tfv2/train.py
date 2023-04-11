import os
import tensorflow as tf
import tensorflow_hub as hub
# import tensorflow_addons as tfa (call when using metrics)
from official.nlp import optimization
import numpy as np
tf.get_logger().setLevel('ERROR')

from modeling import dualencoder_builder
from data import createTFRecord_bert, loadTFRecord_bert

def main():

    # get tpu config
    import os
    if os.environ['COLAB_TPU_ADDR']:
        cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='')
        tf.config.experimental_connect_to_cluster(cluster_resolver)
        tf.tpu.experimental.initialize_tpu_system(cluster_resolver)
        print('Using TPU')
        print("All devices: ", tf.config.list_logical_devices('TPU'))
    elif tf.config.list_physical_devices('GPU'):
        strategy = tf.distribute.MirroredStrategy()
        print('Using GPU')
    else:
        raise ValueError('Running on CPU is not recommended.')

    # prebuild dataset (if it's not a tfrecord)
    if train_file.endswith('.tfrecord') is False:
        dataset = load_dataset(train_file)
        train_file = f"{train_file.rsplit('.', 1)[0]}.tfrecord"
        print('  convert hf dataset into tfrecord, and will save at {}'.format(train_file))
        createTFRecord(train_file, dataset)
        print('  .tfrecord is saved.')

    print('  load and parse tfrecord dataset...')
    dataset = loadTFRecord(train_file, dataset)

    # get 
    strategy = tf.distribute.TPUStrategy(cluster_resolver)
    with strategy.scope():
        # model = tf.keras.models.Squential([tf.keras.layers.Dense(1)])
        model = dualencoder_builder()

        # optimizer = tf.keras.optimizers.Adagrad(learning_rate=0.05)
        optimizer = optimization.create_optimizer(
                # [TODO] add arguments: init_lr
                # [TODO] add arguments: num_train_steps
                # [TODO] add arguments: num_warmup_steps
        )

        model.compile(
                optimizer=optimizer, 
                loss='mse',  # [TODO] this should be task-dependant
                steps_per_execution=10 
        )

        model.fit(
                x=dataset, 
                validation_data=validation_dataset,
                epochs=5, 
                steps_per_epoch=10,
                validation_steps=validation_steps
        )
