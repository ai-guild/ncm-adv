import tensorflow as tf


def gru(num_units):
    return tf.contrib.rnn.GRUCell(num_units)

def gru_n(num_units, num_layers):
    
    # https://github.com/tensorflow/tensorflow/issues/8191
    return tf.contrib.rnn.MultiRNNCell(
                    [gru(num_units) for _ in range(num_layers)],
                    #state_is_tuple=True
                    )


def get_variables(n, shape, name='W'):
    return (tf.get_variable(name+str(i), dtype=tf.float32, shape=shape)
               for i in range(n))


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

    d = cell.state_size[0]
    num_layers = len(cell.state_size)

    with tf.variable_scope(scope):

        for i in range(timesteps):
            if i > 0:
                tf.get_variable_scope().reuse_variables()
            output, state = cell(inputs_tm[i], states[-1])
            outputs.append(output)
            states.append(state)

    states = tf.reshape(tf.transpose(tf.stack(states[1:]), [2, 0, 1, 3]), [-1, d*num_layers])
    Ws = tf.get_variable('Ws', dtype=tf.float32, shape=[num_layers*d, d])
    states = tf.reshape(tf.matmul(states, Ws), [-1, timesteps, d])

    return tf.stack(outputs), states


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

    with tf.variable_scope('fwd'):
        # forward
        _, states_f = uni_net(cell_f, 
                              inputs,
                              cell_f.zero_state(batch_size, tf.float32),
                              timesteps)
    with tf.variable_scope('bwd'):
        # backward
        _, states_b = uni_net(cell_b, 
                              tf.reverse(inputs, axis=[1]),
                              cell_b.zero_state(batch_size, tf.float32),
                              timesteps)
    
    outputs = None
    # outputs
    #  TODO : fix dimensions
    if project_outputs:
        states = tf.concat([states_f, states_b], axis=-1)
        
        if len(states.shape) == 4 and num_layers:
            states = tf.reshape(tf.transpose(states, [-2, 0, 1, -1]), [-1, hdim*2*num_layers])
            Wo = tf.get_variable('/Wo', dtype=tf.float32, shape=[num_layers*2*hdim, hdim])
        elif len(states.shape) == 3:
            states = tf.reshape(tf.transpose(states, [-2, 0, -1]), [-1, hdim*2])
            Wo = tf.get_variable('/Wo', dtype=tf.float32, shape=[2*hdim, hdim])
        else:
            print('>> ERR : Unable to handle state reshape')
            return None
        
        outputs = tf.reshape(tf.matmul(states, Wo), [batch_size, timesteps, hdim])

    return (states_f, states_b), outputs


'''
    Attention Mechanism

    based on "Neural Machine Translation by Jointly Learning to Align and Translate"
        https://arxiv.org/abs/1409.0473

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
def attention(enc_states, dec_state, params, d, timesteps):

    Wa, Ua = params['Wa'], params['Ua']
    # s_ij -> [B,L,d]
    a = tf.tanh(tf.expand_dims(tf.matmul(dec_state, Wa), axis=1) + 
            tf.reshape(tf.matmul(tf.reshape(enc_states,[-1, d]), Ua), [-1, timesteps, d]))
    Va = params['Va'] # [d, 1]
    # e_ij -> softmax(aV_a) : [B, L]
    scores = tf.nn.softmax(tf.reshape(tf.matmul(tf.reshape(a, [-1, d]), Va), [-1, timesteps]))
    # c_i -> weighted sum of encoder states
    return tf.reduce_sum(enc_states*tf.expand_dims(scores, axis=-1), axis=1) # [B, d]    


'''
    Attentive Decoder

    [usage]
    dec_outputs, dec_states = attentive_decoder(enc_states,
                                    tf.zeros(dtype=tf.float32, shape=[B,d]),
                                    batch_size=B,timesteps=L,feed_previous=True,
                                    inputs = inputs)
    shape(enc_states) : [B, L, d]
    shape(inputs) : [[B, d]] if feed_previous else [L, B, d]


'''
def attentive_decoder(enc_states, batch_size, 
                      d, timesteps,
                      inputs = [],
                      scope='attentive_decoder_0',
                      reuse = False,
                      num_layers=1,
                      feed_previous=False):

    with tf.variable_scope('decoder', reuse=reuse):
        # get parameters
        U,W,C,Ur,Wr,Cr,Uz,Wz,Cz,Uo,Vo,Co = get_variables(12, [d,d], name='decoder_param')
        Wa, Ua = get_variables(2, [d,d], 'att')
        Va = tf.get_variable('Va', shape=[d, 1], dtype=tf.float32)
        att_params = {
            'Wa' : Wa, 'Ua' : Ua, 'Va' : Va
        }
        
        
    def step(input_, state, ci):
        z = tf.nn.sigmoid(tf.matmul(input_, Wz)+tf.matmul(state, Uz)+tf.matmul(ci, Cz))
        r = tf.nn.sigmoid(tf.matmul(input_, Wr)+tf.matmul(state, Ur)+tf.matmul(ci, Cr))
        si = tf.nn.tanh(tf.matmul(input_, W)+tf.matmul(ci, C)+tf.matmul(r*state, U))
        
        state = (1-z)*state + z*si
        output = tf.matmul(state, Uo) + tf.matmul(input_, Vo) + tf.matmul(ci, Co)
        
        return output, state
    
    outputs = [inputs[0]] # include GO token as init input
    states = [ tf.zeros(dtype=tf.float32, shape=[batch_size,d]) ]

    # inputs : to time_major
    #inputs = tf.transpose(inputs, [1,0,-1]) # [L, B, d]

    for i in range(timesteps):
        input_ = outputs[-1] if feed_previous else inputs[i]
        ci = attention(enc_states, states[-1], att_params, d, timesteps)

        #print(i, inputs, input_, states[-1], ci)

        output, state = step(input_, states[-1], ci)
    
        outputs.append(output)
        states.append(state)

    
    # time major -> batch major
    states_bm = tf.transpose(tf.stack(states[1:]), [1, 0, 2])
    outputs_bm = tf.transpose(tf.stack(outputs[1:]), [1, 0, 2])

    return outputs_bm, states_bm
