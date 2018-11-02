import tensorflow as tf

def ZeroPadEmbeddingWeight(weights, name="", trainable=True):
    vs, dims = weights.shape
    weight_init = tf.constant_initializer(weights[1:, :])
    embedding_weights = tf.get_variable(
        name=f'{name + "_" if name else ""}embedding_weights', shape=(vs-1, dims),
        initializer=weight_init,
        trainable=trainable)
    zeropad = tf.zeros((1,dims), dtype=tf.float32)
    return tf.concat((zeropad, embedding_weights), 0)
    
def embedded(weights, name="", trainable=True, mask_padding=True):
    if mask_padding:
        embedding_weights = ZeroPadEmbeddingWeight(weights, name=name, trainable=trainable)
    else:
        weight_init = tf.constant_initializer(weights)
        embedding_weights = tf.get_variable(
            name = f'{name + "_" if name else ""}embedding_weights',
            shape = weights.shape,
            initializer = weight_init,
            trainable = trainable)

    def lookup(x):
        nonlocal embedding_weights
        return tf.nn.embedding_lookup(embedding_weights, x)

    return lookup


def char_conv(inp,
              filter_size=5,
              channel_out=100,
              strides=[1, 1, 1, 1],
              padding="SAME",
              dilations=[1, 1, 1, 1]):
    inc = inp.get_shape().as_list()[-1]
    filts = tf.get_variable("char_filter", shape=(1, filter_size, inc, channel_out), dtype=tf.float32)
    bias = tf.get_variable("char_bias", shape=(channel_out,), dtype=tf.float32)
    conv = tf.nn.conv2d(inp, filts,
                        strides=strides,
                        padding=padding,
                        dilations=dilations) + bias
    out = tf.reduce_max(tf.nn.relu(conv), 2)
    return out

def mask(x, x_mask=None):
    if x_mask is None:
        return x

    dim = x.get_shape().as_list()[-1]
    mask = tf.tile(tf.expand_dims(x_mask, -1), [1, 1, dim])
    return x * mask


def normalize(inputs,
              epsilon=1e-8,
              scope="ln",
              reuse=None):
    '''Applies layer normalization.

    Args:
      inputs: A tensor with 2 or more dimensions, where the first dimension has
        `batch_size`.
      epsilon: A floating number. A very small number for preventing ZeroDivision Error.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.

    Returns:
      A tensor with the same shape and data dtype as `inputs`.
    '''
    with tf.variable_scope(scope, reuse=reuse):
        inputs_shape = inputs.get_shape()
        params_shape = inputs_shape[-1:]

        mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
        beta = tf.Variable(tf.zeros(params_shape))
        gamma = tf.Variable(tf.ones(params_shape))
        normalized = (inputs - mean) / ((variance + epsilon) ** (.5))
        outputs = gamma * normalized + beta

    return outputs


def highway(x, activation):
    size = x.get_shape().as_list()[-1]
    W = tf.get_variable("W", (size, size),
                        initializer=tf.random_normal_initializer())
    b = tf.get_variable("b", size,
                        initializer=tf.constant_initializer(0.0))

    Wt = tf.get_variable("Wt", (size, size),
                         initializer=tf.random_normal_initializer())
    bt = tf.get_variable("bt", size,
                         initializer=tf.constant_initializer(0.0))

    T = tf.sigmoid(tf.tensordot(x, Wt, 1) + bt, name="transform_gate")
    H = activation(tf.tensordot(x, W, 1) + b, name="activation")
    C = tf.subtract(1.0, T, name="carry_gate")

    y = tf.add(tf.multiply(H, T), tf.multiply(x, C), "y")
    return y


def highway_network(x, num, activation, name, reuse=None):
    for i, a in zip(range(num), activation):
        with tf.variable_scope(f"{name}_highway{i+1}", reuse=reuse):
            x = highway(x, a)
    return x


def attention(q, k): #q: [B, ql, d], k: [B, kl, d]
    ql = tf.shape(q)[1]
    kl = tf.shape(k)[1]
    
    qs = tf.tile(tf.expand_dims(q, 2), [1, 1, kl, 1]) #[B, ql, kl, d]
    ks = tf.tile(tf.expand_dims(k, 1), [1, ql, 1, 1]) #[B, ql, kl, d]
    qk = qs * ks
    
    h = tf.concat([qs, ks, qk], -1)
    e = tf.squeeze(tf.layers.dense(h, 1), -1) # [B, ql, kl]
    a = tf.nn.softmax(e, -1) 
    v = tf.matmul(a, k) #[B, ql, d]
    return v

def ffn(x, d, act=None):
    a = tf.layers.dense(x, d, activation=act)
    return tf.layers.dense(a, d)

def pwffn(x, ff=2048):
    dim = x.get_shape().as_list()[-1]
    a = tf.layers.dense(x, ff, activation=tf.nn.relu)
    return tf.layers.dense(a, dim)

def multihead_attention(q, k, v,
                        dk=64,
                        h = 8,
                        ff=2048,
                        scope="multihead_attention",
                        reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        dim = q.get_shape().as_list()[-1]
        b = tf.shape(q)[0]
        ql = tf.shape(q)[1]
        kl = tf.shape(k)[1]
        vl = kl
        Q = tf.layers.dense(q, h*dk)
        K = tf.layers.dense(k, h*dk)
        V = tf.layers.dense(v, h*dk)
        hq = tf.reshape(Q, (b*h, ql, dk))
        hk = tf.reshape(K, (b*h, kl, dk))
        hv = tf.reshape(V, (b*h, vl, dk)) #[b*h, vl, dk]
        
        qkt = tf.matmul(hq, tf.transpose(hk, (0, 2, 1))) / (dk ** 0.5)#[b*h, ql, kl=vl]
        a = tf.nn.softmax(qkt, -1)
        
        hatten = tf.matmul(a , hv) #[b*h, ql, dk]
        hatten = tf.reshape(hatten, (b, ql, h*dk))

        atten = tf.layers.dense(hatten, dim, use_bias=False)
        aoutput = normalize(atten + q)
        output = pwffn(aoutput, ff)
        output = normalize(aoutput + output)
        
        return output
