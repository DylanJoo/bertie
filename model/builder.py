import modeling

def model_fn_builder(
        bert_config, 
        num_labels, 
        init_checkpoint, 
        learning_rate, 
        num_train_steps, 
        num_warmup_steps, 
        use_tpu, 
        use_one_hot_embeddings
    ):
    """Returns `model_fn` closure for TPUEstimator."""

    def model_fn(features, labels, mode, params):
    """The `model_fn` for TPUEstimator. """

        # Collect the input features
        tf.logging.info("*** Features ***")
        for name in sorted(features.keys()):
            tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))
        is_train = (mode == tf.estimator.ModeKeys.TRAIN)

        tf.logging.info("*** Start creating model ***")
        if is_train:
            # [TODO] adjust the order of keyword args.
            model_outputs = create_model(
                    bert_config, is_training, **features, 
                    num_labels, use_one_hot_embeddings
            )
            total_loss, per_example_loss, logits, probabilities = model_outputs

        tf.logging.info("*** Creating model done ***")
        tvars = tf.trainable_variables()
        # initialized_variabale_names = {}

        scaffold_fn = None

        if init_checkpoint:
            assignment_map_outputs = modeling.get_assignment_map_from_checkpoint(
                    tvars, init_checkpoint
            )
            (assignment_map, initialized_variable_names) = assignment_map_outputs
            # (assignment_map1, initialized_variable_names1) = modeling.get_assignment_map_from_checkpoint(
            #         tvars, init_checkpoint, 'Student/', 'query_reformulator/'
            # )
            # assignment_maps = [assignment_map, assignment_map1]
            # initialized_variable_names.update(initialized_variable_names)
            tf.logging.info("*** Assignment Map ***")

            if use_tpu:
                def tpu_scaffold():
                    tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
                    return tf.train.Scaffold()
                scaffold_fn = tpu_scaffold
            else:
                tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
        tf.logging.info("*** Trainable Variables ***")
        for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
            tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape, init_string)

        output_spec = None
        if mode == tf.estimator.ModeKeys.TRAIN:

            # [TODO] maybe keyword arguments
            train_op = optimization.create_optimizer(
                    total_loss, 
                    learning_rate, 
                    num_train_steps, 
                    num_warmup_steps, 
                    use_tpu
            )
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                    mode=mode,
                    loss=total_loss,
                    train_op=train_op,
                    scaffold_fn=scaffold_fn
            )
        elif mode == tf.estimator.ModeKeys.PREDICT:
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                    mode=mode,
                    predictions={"probabilities": probabilities},
                    scaffold_fn=scaffold_fn
            )
        else:
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                    mode=mode,
                    loss=total_loss,
                    eval_metrics=eval_metrics,
                    scaffold_fn=scaffold_fn
            )
        return output_spec

    return model_fn


# def model_fn_builder(
#         bert_config, 
#         init_checkpoint, 
#         learning_rate,
#         num_train_steps, 
#         num_warmup_steps, 
#         use_tpu,
#         use_one_hot_embeddings
#     ):
# 	"""Returns `model_fn` closure for TPUEstimator."""
#     # colbert_dim, dotbert_dim, max_q_len, max_p_len, doc_type,
#     # loss, kd_source, train_model, eval_model, is_eval, is_output
#
#     def model_fn(features, labels, mode, params):
# 		"""The `model_fn` for TPUEstimator."""
# 		tf.logging.info("*** Features ***")
# 		for name in sorted(features.keys()):
# 		    tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))
#
# 		is_train = (mode == tf.estimator.ModeKeys.TRAIN)
#
#         # load inputs
#         input_ids = features["input_ids"]
#         token_type_ids = features["token_type_ids"]
#         attention_mask = features["attention_mask"]
#
#         if is_train:
#             label_ids = features["label_ids"]
