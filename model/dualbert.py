import modeling

def create_model(
        bert_config, 
        is_training, 
        input_ids, 
        input_mask, 
        segment_ids, 
        label,
        use_one_hot_embeddings,
        is_eval, 
        is_output
    ):
    with tf.variable_scope("query_encoder"):
    with tf.variable_scope("document_encoder"):

    # with tf.variable_scope("loss"):
    #
    # if is_training:
    # output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)
    #
    # logits = tf.matmul(output_layer, output_weights, transpose_b=True)
    # logits = tf.nn.bias_add(logits, output_bias)
    # probabilities = tf.nn.softmax(logits, axis=-1)
    # log_probs = tf.nn.log_softmax(logits, axis=-1)
    #
    # one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)
    #
    # per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
    # loss = tf.reduce_mean(per_example_loss)
    #
    # return (loss, per_example_loss, logits, probabilities)
