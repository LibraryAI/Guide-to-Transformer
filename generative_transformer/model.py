import numpy as np
import tensorflow as tf

N_VOCAB #토큰 수 ex. 40000
N_CTX #한 인풋의 토큰 허용 수
embedding_dim #토큰 임베딩 차원 수
FFN_DIM #FFN weight 차원 수
EMBED_PDROP #임베딩 dropout percentage
RESID_PDROP 
ATTN_PDROP 
N_HEAD #multi-headed 에서의 head의 수


TEXT_ENCODER = TextEncoder(encoder_path, bpe_path)

train_tokens = []
for datas in data['data']:
    for paragraph in datas["paragraphs"]:
        train_tokens.append(TEXT_ENCODER.encode(paragraph['context']))

def transform_texts(list_of_tokens, positional=True):
	'''
	return xmb: tokenized matrix with unique token values mapped, size (n_batch × N_CTX × 2)
	       mmb: masking matrix with input text token assigend 1 and empty as 0, size  (n_batch × N_CTX)
	'''
	n_batch = len(list_of_tokens)
	xmb = np.zeros((n_batch, N_CTX, 2), dtype=np.int32)
	mmb = np.zeros((n_batch, N_CTX), dtype=np.float32)
	for i, token in enumerate(list_of_tokens):
		token1 = token[:N_CTX]
		l1 = len(token1)
		xmb[i, :l1, 0] = token1
		mmb[i, :l1] = 1
	if not positional:
		xmb[:, :, 1] = np.arange(N_VOCAB, N_VOCAB+N_CTX)
	return xmb, mmb

class PositionEncoding:
	def __init__(self, max_len=512, embedding_dim=256):
		self.dim = embedding_dim
		self.position = [[pos / 10000**(2*(t//2)/embedding_dim) for t in range(embedding_dim)] if pos != 0 else [0 for t in range(embedding_dim)] for pos in range(max_len)]
		self.pe = np.zeros((max_len, embedding_dim))
		for i in range(max_len):
			for j in range(embedding_dim):
				if j % 2 ==0:
					self.pe[i, j] = np.sin(self.position[i][j])
				else:
					self.pe[i, j] = np.cos(self.position[i][j])

		# self.pe[:, 0::2] = np.sin(self.position)
		# self.pe[:, 1::2] = np.cos(self.position)

	def positional_encoding(embed, mask_mat):
		'''
		return positional encoding matrix with (n_batch × N_CTX × embedding_dim)
		'''
		n_batch = embed.shape[0]
		pos_mat = np.zeros((n_batch, N_CTX, embedding_dim), dtype=np.float32)
		for i in range(n_batch):
			l1 = int(np.reduce_sum(mask_mat[i,:]))
			pos_mat[i,:l1,:] = self.pe[:l1,:]
		return pos_mat

def embed(X, embed_mat, mask_mat=None, positional=False):
	'''
	return embeded mat of (n_batch × n_ctx × embedding_dim)
	'''
	if positional and mask_mat:
		emb = tf.gather(embed_mat, X[:,:,0])
		pos = PE.positional_encoding(embed_mat, mask_mat)
		pos = tf.add(emb, pos)
	else:
		emb = tf.gather(embed_mat, X)
		pos = tf.reduce_sum(emb, 2)
	return pos

def dropout(embed, pdrop=0.1, train=False):
	'''
	return dropped out matrix with the same size as input. if train is false just return input
	'''
	if train and pdrop > 0:
		embed = tf.nn.dropout(embed, 1-pdrop)
	return embed

def ffn(x, scope, weight_dim, train=False):
	with tf.variable_scope(scope):
		nx = x.shape[-1]
		c = conv1d(x, 'conv1', weight_dim, 1, train=train)
		r = tf.nn.relu(c)
		c2 = conv1d(r, 'conv2', nx, 1, train=train)
		c2 = dropout(c2, RESID_PDROP, train)
	return c2

def conv1d(x, scope, nf, rf, w_init=tf.random_normal_initializer(stddev=0.02), b_init=tf.constant_initializer(0), pad='VALID', train=False):
	'''
	nx: embedding dim
	nf: output dim

	(n_batch × n_ctx, embedding_dim) × (embedding_dim, output_dim) -> (n_batch × n_ctx, output_dim) -> (n_batch, n_ctx, output_dim)

	return 1d convolution output of shape (n_batch, n_ctx, output_dim) 
	'''
	with tf.variable_scope(scope):
		nx = shape_list(x)[-1]
        w = tf.get_variable("w", [rf, nx, nf], initializer=w_init)
        b = tf.get_variable("b", [nf], initializer=b_init)
        if rf == 1:  # faster 1x1 conv
            c = tf.reshape(tf.matmul(tf.reshape(
                x, [-1, nx]), tf.reshape(w, [-1, nf]))+b, shape_list(x)[:-1]+[nf])
        else:  # was used to train LM
            c = tf.nn.conv1d(x, w, stride=1, padding=pad)+b
        return c

def split_heads(x, n_head):
	'''
	input x: (n_batch, n_ctx, attn_dim)
	new_shape: (n_batch, n_ctx, n_head, attn_dim//n_head)
	'''
	x_shape = tf.shape(x)
	new_shape = x_shape[:-1] + [n_head, x_shape[-1]//n_head]
	new_mat = tf.reshape(x, new_shape)
	return new_mat

def merge_heads(x):
	'''
	input: (n_batch, n_head, n_ctx, attn_dim/n_head)
	output: (n_batch, n_ctx, attn_dim)
	'''
	x = tf.transpose(x, [0, 2, 1, 3])
	a_dim = [x.shape[-2] * x.shape[-1]]
	x = tf.reshape(x, x.shape[:-2]+a_dim)
	return x

def masked_attn(query, key, value, train=train, scale=scale):
	'''
	input: (n_batch, n_ctx, n_head, attn_dim//n_head)

	q: (n_batch, n_head, n_ctx, attn/n_head)
	k: (n_batch, n_head, aatt/, n_ctx)
	score: (n_batch, n_head, n_ctx, n_ctx)
	attn: (n_batch, n_head, n_ctx, attn_dim/n_head)
	'''
	q = tf.transpose(query, [0, 2, 1, 3])
	k = tf.transpose(key, [0, 2, 3, 1])
	v = tf.transpose(value, [0, 2, 1, 3])

	score = tf.matmul(q, k)

	if scale:
		d = tf.sqrt(q.shape[-1])
		score = tf.div(score, d)

	n = score.shape[-1]
	b = tf.matrix_band_part(tf.ones([n,n]), -1, 0)
		# b_diag = tf.matrix_band_part(tf.ones([n,n]), 0, 0)
		# b = b + b_diag
	b = tf.reshape(b, [1, 1, n, n])
	score = score*b + -1e9*(1-b)
	score = tf.nn.softmax(score)
	score = dropout(score, ATTN_DROP, train)
	attn = tf.matmul(score, v)
	return attn

def masked_multi_attn(embed, scope, n_state, attention_dim=None, train=False, scale=False):
	'''
	input: (n_batch, n_ctx, embedding_dim)
	n_state: embedding_dim
	'''
	assert n_state % N_HEAD == 0
	with tf.variable_scope(scope):
		c = conv1d(embed, 'conv_attn', n_state*3, 1, train=train)
		k, v, q = tf.split(c, 3, 2)
		k = split_heads(k, N_HEAD)
		v = split_heads(v, N_HEAD)
		q = split_heads(q, N_HEAD)
		a = masked_attn(q, k, v, train=train, scale=scale)
        a = merge_heads(a)
        a = conv1d(a, 'c_proj', n_state, 1, train=train)
        a = dropout(a, RESID_PDROP, train)
        return a

def norm(x, scope, e=1e-5, axis=[1]):
	with tf.variable_scope(scope):
		m = tf.reduce_mean(x, axis=axis, keepdims=True)
		stdev = tf.sqrt(tf.reduce_mean(tf.square(x-m), axis=axis, keepdims=True) + e)
		x = (x-m)/stdev
	return x

def block(x, scope, train=False, scale=True):
	'''
	input: (n_batch, N_CTX, embedding_dim)
	output: (n_batch, N_CTX, embedding_dim)
	'''
	with tf.variable_scope(scope):
		nx = x.shape[-1]
		a = masked_multi_attn(x, 'attn', nx, train=train, scale=scale)
		n = norm(x+a, 'norm1')
		f = ffn(n, 'ffn', FFN_DIM, train=train)
		n2 = norm(f+n, 'norm2')
	return n2

def transformer_decoder(X, M, train=False, reuse=False, positional=False):
	
	with tf.variable_scope('transformer_decoder', reuse=reuse):		
		X = tf.reshape(X, [-1, N_CTX, 2])
		M = tf.reshape(M, [-1, N_CTX])
		if positional:
			we = tf.get_variable('we', [N_VOCAB, embedding_dim], initializer=tf.random_normal_initializer(stddev=0.02))
			t = embed(X, we, M, positional=True)
		else:
			we = tf.get_variable('we', [N_VOCAB+N_CTX, embedding_dim], initializer=tf.random_normal_initializer(stddev=0.02))
			t = embed(X, we)
		for layer in range(N_LAYER):
			t = block(t, 't%d' % layer, train=train, scale=True)

		lm_t = tf.reshape(t, [-1, embedding_dim])
        lm_logits = tf.reshape(
            tf.matmul(lm_t, we[:N_VOCAB, :], transpose_b=True),
            [-1, N_CTX, N_VOCAB]
        )
        lm_logits_truncated = tf.reshape(
            lm_logits[:, :-1],
            [-1, N_VOCAB]
        )
        lm_losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=lm_logits_truncated, labels=tf.reshape(X[:, 1:, 0], [-1]))
        lm_losses = tf.reshape(
            lm_losses, [X.shape[0], X.shape[1]-1])
        lm_losses = tf.reduce_sum(
            lm_losses*M[:, 1:], 1)/tf.reduce_sum(M[:, 1:], 1)
        return lm_logits, lm_losses

def build_graph(sess, positional = False):
	X = tf.placeholder(tf.int32, [None, N_CTX, 2])
	M = tf.placeholder(tf.float32, [None, N_CTX])
	if position:
		PE = PositionEncoding()
	lm_logits, lm_losses = transformer_decoder(X, M, train=True, resue=False, positional=positional)
	params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, ".*{}.*".format(key))
	sess.run(tf.global_variables_initializer())
    

    shapes = json.load(open('model/params_shapes.json'))
    offsets = np.cumsum([np.prod(shape) for shape in shapes])
    init_params = [np.load('model/params_{}.npy'.format(n)) for n in range(10)]
    init_params = np.split(np.concatenate(init_params, 0), offsets)[:-1]
    init_params = [param.reshape(shape)
                   for param, shape in zip(init_params, shapes)]
    init_params[0] = init_params[0][:N_CTX]
    init_params[0] = np.concatenate([init_params[1], init_params[0]], 0)
    del init_params[1]
    n_transfer = 1 + N_TRANSFER * 12
    sess.run([p.assign(ip)
              for p, ip in zip(
        params[:n_transfer],
        init_params[:n_transfer])])
    return X, M, lm_logits, lm_losse


def model(X, M, Y, train=False, reuse=False):
    with tf.variable_scope('model', reuse=reuse):
        we = tf.get_variable("we", [n_vocab+n_special+n_ctx, n_embd],
                             initializer=tf.random_normal_initializer(stddev=0.02))
        we = dropout(we, embd_pdrop, train)

        X = tf.reshape(X, [-1, n_ctx, 2])
        M = tf.reshape(M, [-1, n_ctx])

        h = embed(X, we)
        for layer in range(n_layer):
            h = block(h, 'h%d' % layer, train=train, scale=True)

        lm_h = tf.reshape(h[:, :-1], [-1, n_embd])
        lm_logits = tf.matmul(lm_h, we, transpose_b=True)
        lm_losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=lm_logits, labels=tf.reshape(X[:, 1:, 0], [-1]))
        lm_losses = tf.reshape(
            lm_losses, [shape_list(X)[0], shape_list(X)[1]-1])
        lm_losses = tf.reduce_sum(
            lm_losses*M[:, 1:], 1)/tf.reduce_sum(M[:, 1:], 1)

        clf_h = tf.reshape(h, [-1, n_embd])
        pool_idx = tf.cast(
            tf.argmax(tf.cast(tf.equal(X[:, :, 0], clf_token), tf.float32), 1), tf.int32)
        clf_h = tf.gather(clf_h, tf.range(
            shape_list(X)[0], dtype=tf.int32)*n_ctx+pool_idx)

        clf_h = tf.reshape(clf_h, [-1, 2, n_embd])
        if train and clf_pdrop > 0:
            shape = shape_list(clf_h)
            shape[1] = 1
            clf_h = tf.nn.dropout(clf_h, 1-clf_pdrop, shape)
        clf_h = tf.reshape(clf_h, [-1, n_embd])
        clf_logits = clf(clf_h, 1, train=train)
        clf_logits = tf.reshape(clf_logits, [-1, 2])

        clf_losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=clf_logits, labels=Y)
        return clf_logits, clf_losses, lm_losses
