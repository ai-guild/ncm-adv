import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer as xinit
from recurrence import *

import data_utils


class AttentiveSeq2seq():

    def __init__(self, L, vocab_size, enc_hdim=150, dec_hdim=150):

        tf.reset_default_graph()

        # placeholders
        self.inputs = tf.placeholder(tf.int32, shape=[None,L], name='inputs')
        self.targets = tf.placeholder(tf.int32, shape=[None,L], name='targets')
        go_token = tf.reshape(tf.fill(tf.shape(self.inputs[:,0]), w2i['GO']), [-1,1])
        decoder_inputs = tf.concat( values=[go_token, self.targets[:, 1:]], axis=1)
        #self.training = tf.placeholder(tf.bool, name='is_training')
        batch_size = tf.shape(self.inputs)[0] # infer batch size
        
        # encoder sequence length
        # enc_seq_len = seq_len(inputs)

        # padding mask
        padding_mask = tf.cast(self.targets > 0, tf.float32 )

        # embedding
        emb_mat = tf.get_variable('emb', shape=[vocab_size, enc_hdim], dtype=tf.float32, 
                                 initializer=xinit())
        emb_enc_inputs = tf.nn.embedding_lookup(emb_mat, self.inputs)
        emb_dec_inputs = tf.nn.embedding_lookup(emb_mat, decoder_inputs)

        # encoder 
        with tf.variable_scope('encoder'):
            (estates_f, estates_b), _ = bi_net(gru_n(num_layers=3, num_units=enc_hdim),
                                               gru_n(num_layers=3, num_units=enc_hdim),
                                               emb_enc_inputs,
                                               batch_size=batch_size,
                                               timesteps=L,
                                               num_layers=3
                                              )
        # encoder states
        estates = tf.concat([estates_f, estates_b], axis=-1)
        # reshape encoder states
        Ws = tf.get_variable('Ws', shape=[2*enc_hdim, enc_hdim], dtype=tf.float32)
        estates = tf.reshape(tf.matmul(tf.reshape(estates, [-1, 2*enc_hdim]), Ws), [-1, L, enc_hdim])

        # convert decoder inputs to time_major format
        emb_dec_inputs = tf.transpose(emb_dec_inputs, [1,0,2], name='time_major')
        # decoder
        with tf.variable_scope('decoder') as scope:
            decoder_outputs, _ = attentive_decoder(estates, batch_size, dec_hdim, L,
                                                 inputs=emb_dec_inputs, reuse=False)
            
            tf.get_variable_scope().reuse_variables()
            
            decoder_outputs_inf, _ = attentive_decoder(estates, batch_size, dec_hdim, L,
                                                     inputs=emb_dec_inputs,
                                                     reuse=True,
                                                     feed_previous=True)
            
        # calculate logits and probabilities
        Wo = tf.get_variable('Wo', shape=[dec_hdim, vocab_size], dtype=tf.float32, 
                                 initializer=xinit())
        bo = tf.get_variable('bo', shape=[vocab_size], dtype=tf.float32, 
                                 initializer=xinit())
        proj_outputs = tf.matmul(tf.reshape(decoder_outputs, [-1, dec_hdim]), Wo) + bo
        proj_outputs_inf = tf.matmul(tf.reshape(decoder_outputs_inf, [-1, dec_hdim]), Wo) + bo

        # get logits (train/inference)
        logits = tf.cond(tf.random_normal(shape=()) > 0.,
                    lambda : tf.reshape(proj_outputs, [batch_size, L, vocab_size]),
                    lambda : tf.reshape(proj_outputs_inf, [batch_size, L, vocab_size]))

        # probabilities
        self.probs = tf.nn.softmax(tf.reshape(proj_outputs_inf, [batch_size, L, vocab_size]))

        # calculate loss
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
                        logits = logits,
                        labels = self.targets)
        # apply mask
        masked_cross_entropy = cross_entropy * padding_mask
        # average across sequence, batch
        self.loss = tf.reduce_mean(masked_cross_entropy)

        # optimization
        optimizer = tf.train.AdagradOptimizer(learning_rate=0.01)
        self.train_op = optimizer.minimize(self.loss)

        # inference
        self.prediction = tf.argmax(self.probs, axis=-1)

        # attach session to object
        self.sess = tf.Session()


    #  Get sequence lengths
    def seq_len(self, t):
        return tf.reduce_sum(tf.cast(t>0, tf.int32), axis=1)

    def train(self, batch_size, epochs, trainset):

        idx_q, idx_a = trainset
        B = batch_size

        self.sess.run(tf.global_variables_initializer())

        for i in range(epochs):
            avg_loss = 0.
            for j in range(len(idx_q)//B):
                _, l = self.sess.run([self.train_op, self.loss], feed_dict = {
                    self.inputs : idx_q[j*B:(j+1)*B],
                    self.targets : idx_a[j*B:(j+1)*B]
                })
                avg_loss += l
                if j and j%60==0:
                    print('{}.{} : {}'.format(i,j,avg_loss/60))
                    avg_loss = 0.


if __name__ == '__main__':

    # fetch data
    metadata, idx_q, idx_a = data_utils.load_data('../data/')
    # add special symbol
    i2w = metadata['idx2w']
    w2i = metadata['w2idx']

    # infer params from data
    L = len(idx_q[0])
    vocab_size = len(i2w)

    # create model
    model = AttentiveSeq2seq(L, vocab_size)

    # begin training
    model.train(batch_size=32, epochs=20, trainset=(idx_q, idx_a))
