def create_model(
        bert_config, 
        is_training, 
        input_ids, 
        input_mask, 
        segment_ids,
        labels, 
        num_labels, 
        use_one_hot_embeddings,
        as_an_encoder=False
        encoder_pooling='cls'
    ):
    """Creates a classification model."""
    model = modeling.BertModel(
            config=bert_config,
            is_training=is_training,
            input_ids=input_ids,
            input_mask=input_mask,
            token_type_ids=segment_ids,
            use_one_hot_embeddings=use_one_hot_embeddings
    )

    # Output 1: the last hidden states (ctx reprs includes sent/token-level)
    # [NOTE] with layers: output_all_embedding = model.get_all_encoder_layers()
    output_embedding = model.get_all_encoder_layers()[-1] # contextualized embeds
    batch_size = output_embedding.shape[0].value
    seq_length = output_embedding.shape[1].value
    hidden_size = output_embedding.shape[2].value

    # Output 2: the pooled output (for classification)
    # output_layer = model.get_pooled_output()

    # Output postprocessing
    # Pooling 1: cls pooling
    cls_embedding = output_embedding[:, :1, :] # [bs, 1, h]

    # Pooling 2: average pooling #[TODO]

    input_length = tf.reduce_sum(effective_mask, axis=1)

    # It's better initialize an output object like hf
    return {"last_hidden_states": output_embedding, 
            "cls_embedding": cls_embedding,
            "total_length": input_length}

    # output_weights = tf.get_variable(
    #         "output_weights", [num_labels, hidden_size],
    #         initializer=tf.truncated_normal_initializer(stddev=0.02)
    # )
    # output_bias = tf.get_variable(
    #         "output_bias", [num_labels], 
    #         initializer=tf.zeros_initializer()
    # )
