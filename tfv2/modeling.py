def dualencoder_builder():

    class DualEncoder(tf.keras.Model):
        def __init__(self):
            self.encoder0 = hub.KerasLayer(tfhub_handle_encoder, trainable=True)
            self.decoder1 = hub.KerasLayer(tfhub_handle_encoder, trainable=True)

        def call(self, preprocessed_inputs):

            ## parse the preprocessed inputs
            encoder_inputs0 = preprocessed_inputs[0]
            encoder_inputs1 = preprocessed_inputs[1]

            # [NOTE] the output keys contain 
            # ['pooled_output', 'sequence_output', 'encoder_outputs', 'default']
            outputs0 = self.encoder0(preprocessed_inputs0)
            outputs1 = self.encoder1(preprocessed_inputs1)

            # pooling 
            ## [CLS] outputs of each batch (B, 1, H)
            cls_embedding0 = outputs0['sequence_output'][:, 0:1, :]
            cls_embedding1 = outputs1['sequence_output'][:, 0:1, :]

            # loss calculation
            ## In-batch negative 
            logits_matrix = encoder0_outputs * encoder0_outputs
            return logits_matrix

    return DualEncoder
