import tensorflow as tf


def gru(num_units):
    return tf.contrib.rnn.GRUCell(num_units)

def gru_n(num_units, num_layers):
    
    # https://github.com/tensorflow/tensorflow/issues/8191
    return tf.contrib.rnn.MultiRNNCell(
                    [gru(num_units) for _ in range(num_layers)],
                    #state_is_tuple=True
                    )


'''
    Uni-directional RNN

    [usage]
    cell_ = gru_n(hdim, 3)
    outputs, states = uni_net(cell = cell_,
                             inputs= inputs_emb,
                             init_state= cell_.zero_state(batch_size, tf.float32),
                             timesteps = L)
'''
def uni_net(cell, inputs, init_state, timesteps, time_major=False, scope='uni_net_0'):
    # convert to time major format
    if not time_major:
        inputs_tm = tf.transpose(inputs, [1, 0, -1])
    # collection of states and outputs
    states, outputs = [init_state], []

    with tf.variable_scope(scope):

        for i in range(timesteps):
            if i > 0:
                tf.get_variable_scope().reuse_variables()
            output, state = cell(inputs_tm[i], states[-1])
            outputs.append(output)
            states.append(state)

    return tf.stack(outputs), tf.stack(states[1:])


'''
    Bi-directional RNN

    [usage]
    (states_f, states_b), outputs = bi_net(cell_f= gru_n(hdim,3),
                                        cell_b= gru_n(hdim,3),
                                        inputs= inputs_emb,
                                        batch_size= batch_size,
                                        timesteps=L,
                                        scope='bi_net_5',
                                        num_layers=3,
                                        project_outputs=True)
'''
def bi_net(cell_f, cell_b, inputs, batch_size, timesteps, 
           scope= 'bi_net',
           project_outputs=False,
           num_layers=1):

    # forward
    _, states_f = uni_net(cell_f, 
                          inputs,
                          cell_f.zero_state(batch_size, tf.float32),
                          timesteps,
                          scope=scope + '_f')
    # backward
    _, states_b = uni_net(cell_b, 
                          tf.reverse(inputs, axis=[1]),
                          cell_b.zero_state(batch_size, tf.float32),
                          timesteps,
                          scope=scope + '_b')
    
    outputs = None
    # outputs
    if project_outputs:
        states = tf.concat([states_f, states_b], axis=-1)
        
        if len(states.shape) == 4 and num_layers:
            states = tf.reshape(tf.transpose(states, [-2, 0, 1, -1]), [-1, hdim*2*num_layers])
            Wo = tf.get_variable(scope+'/Wo', dtype=tf.float32, shape=[num_layers*2*hdim, hdim])
        elif len(states.shape) == 3:
            states = tf.reshape(tf.transpose(states, [-2, 0, -1]), [-1, hdim*2])
            Wo = tf.get_variable(scope+'/Wo', dtype=tf.float32, shape=[2*hdim, hdim])
        else:
            print('>> ERR : Unable to handle state reshape')
            return None
        
        outputs = tf.reshape(tf.matmul(states, Wo), [batch_size, timesteps, hdim])

    return (states_f, states_b), outputs


'''
    Attention Mechanism

    [usage]
    ci = attention(enc_states, dec_state, params= {
        'Wa' : Wa, # [d,d]
        'Ua' : Ua, # [d,d]
        'Va' : Va  # [d,1]
        })
    shape(enc_states) : [B, L, d]
    shape(dec_state)  : [B, d]
    shape(ci)         : [B,d]

'''
def attention(enc_states, dec_state, params):
    # based on "Neural Machine Translation by Jointly Learning to Align and Translate"
    #  https://arxiv.org/abs/1409.0473
    Wa, Ua = params['Wa'], params['Ua']
    # s_ij -> [B,L,d]
    a = tf.tanh(tf.expand_dims(tf.matmul(dec_state, Wa), axis=1) + 
            tf.reshape(tf.matmul(tf.reshape(enc_states,[-1, d]), Ua), [-1, L, d]))
    Va = params['Va'] # [d, 1]
    # e_ij -> softmax(aV_a) : [B, L]
    scores = tf.nn.softmax(tf.reshape(tf.matmul(tf.reshape(a, [-1, d]), Va), [-1, L]))
    # c_i -> weighted sum of encoder states
    return tf.reduce_sum(enc_states*tf.expand_dims(scores, axis=-1), axis=1) # [B, d]    
