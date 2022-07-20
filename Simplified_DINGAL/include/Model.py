import math
from .Init import *
from include.Test import *
import scipy.spatial as sp
import json
from include.Config import Config


def get_mat(e, KG):
    du = [1] * e
    for tri in KG:
        if tri[0] != tri[2]:
            du[tri[0]] += 1
            du[tri[2]] += 1
    M = {}
    for tri in KG:
        if tri[0] == tri[2]:
            continue
        if (tri[0], tri[2]) not in M:
            M[(tri[0], tri[2])] = 1
        else:
            pass
        if (tri[2], tri[0]) not in M:
            M[(tri[2], tri[0])] = 1
        else:
            pass

    for i in range(e):
        M[(i, i)] = 1
    return M, du

def get_sparse_tensor(e, KG):
    print('getting a sparse tensor...')
    M, du = get_mat(e, KG)
    ind = []
    val = []
    for fir, sec in M:
        ind.append((fir, sec))
        val.append(M[(fir, sec)]/du[fir])
    M = tf.SparseTensor(indices=ind, values=val, dense_shape=[e, e])
    
    return M


def add_mask_and_gcn_layer(inlayer, dimension, M, act_func, name, dropout=0.0, init=ones):
    """
    We adopt the identity matrix as the GCN's weight matrix to avoid overfitting on these three datasets.
    The idea still holds true for other datasets and applications.
    """
    inlayer = tf.nn.dropout(inlayer, 1 - dropout)
    w0 = init([1, dimension], name)
    mask_gate = tf.nn.relu(w0)
    tosum = tf.sparse_tensor_dense_matmul(M, tf.multiply(inlayer, mask_gate))
    if act_func is None:
        return tosum
    else:
        return act_func(tosum)



def highway(layer1, layer2, dimension, name):
    kernel_gate = glorot([dimension,dimension],name + "kernel_gate")
    bias_gate = zeros([dimension], name + "bias_gate")
    transform_gate = tf.matmul(layer1, kernel_gate) + bias_gate
    transform_gate = tf.nn.sigmoid(transform_gate)
    carry_gate = 1.0 - transform_gate
    return transform_gate * layer2 + carry_gate * layer1

def get_input_layer(e, dimension, lang):
    print('adding the primal input layer...')
    with open(file='data/' + lang + '_en/' + lang + '_vectorList.json', mode='r', encoding='utf-8') as f:
        embedding_list = json.load(f)
        print(len(embedding_list), 'rows,', len(embedding_list[0]), 'columns.')
    input_embeddings = tf.convert_to_tensor(embedding_list)
    return tf.nn.l2_normalize(input_embeddings, 1)


def get_loss(outlayer, ILL, gamma, k, neg_left, neg_right, neg2_left, neg2_right):
    print('getting loss...')
    left = ILL[:, 0]
    right = ILL[:, 1]
    t = len(ILL)
    left_x = tf.nn.embedding_lookup(outlayer, left)
    right_x = tf.nn.embedding_lookup(outlayer, right)
    A = tf.reduce_sum(tf.abs(left_x - right_x), 1)
    neg_l_x = tf.nn.embedding_lookup(outlayer, neg_left)
    neg_r_x = tf.nn.embedding_lookup(outlayer, neg_right)
    B = tf.reduce_sum(tf.abs(neg_l_x - neg_r_x), 1)
    C = - tf.reshape(B, [t, k])
    D = A + gamma
    L1 = tf.nn.relu(tf.add(C, tf.reshape(D, [t, 1])))
    neg_l_x = tf.nn.embedding_lookup(outlayer, neg2_left)
    neg_r_x = tf.nn.embedding_lookup(outlayer, neg2_right)
    B = tf.reduce_sum(tf.abs(neg_l_x - neg_r_x), 1)
    C = - tf.reshape(B, [t, k])
    L2 = tf.nn.relu(tf.add(C, tf.reshape(D, [t, 1])))
    return (tf.reduce_sum(L1) + tf.reduce_sum(L2)) / (2.0 * k * t)

def build(dimension, act_func, gamma, k, e, ILL, KG, feat_matrix):
    tf.reset_default_graph()
    input_layer = tf.nn.l2_normalize(tf.convert_to_tensor(feat_matrix,dtype=tf.float32), 1)
    M = get_sparse_tensor(e, KG)
    gcn_layer_1 = add_mask_and_gcn_layer(input_layer, dimension, M, act_func, "gcn1", dropout=0.0)
    gcn_layer_1 = highway(input_layer,gcn_layer_1,dimension, "highway1")
    gcn_layer_2 = add_mask_and_gcn_layer(gcn_layer_1, dimension, M, act_func, "gcn2", dropout=0.0)
    output_prel_e = highway(gcn_layer_1,gcn_layer_2,dimension, "highway2")
    t = len(ILL)
    neg_left = tf.placeholder(tf.int32, [t * k], "neg_left")
    neg_right = tf.placeholder(tf.int32, [t * k], "neg_right")
    neg2_left = tf.placeholder(tf.int32, [t * k], "neg2_left")
    neg2_right = tf.placeholder(tf.int32, [t * k], "neg2_right")
    loss_1 = get_loss(output_prel_e, ILL, gamma, k, neg_left, neg_right, neg2_left, neg2_right)
   
    return output_prel_e, loss_1


# get negative samples
def get_neg(ILL, output_layer, k):
    neg = []
    t = len(ILL)
    ILL_vec = np.array([output_layer[e1] for e1 in ILL])
    KG_vec = np.array(output_layer)
    sim = sp.distance.cdist(ILL_vec, KG_vec, metric='cityblock')
    for i in range(t):
        rank = sim[i, :].argsort()
        neg.append(rank[0:k])

    neg = np.array(neg)
    neg = neg.reshape((t * k,))
    return neg


def training(output_prel_e,  loss_1, learning_rate, epochs, ILL, k, s, test):
    train_step_1 = tf.train.AdamOptimizer(learning_rate).minimize(loss_1)
    print('initializing...')
    init = tf.global_variables_initializer()
    sess = tf.Session()
    saver = tf.train.Saver()
    sess.run(init)
    print('running...')
    J = []
    t = len(ILL)
    ILL = np.array(ILL)
    L = np.ones((t, k)) * (ILL[:, 0].reshape((t, 1)))
    neg_left = L.reshape((t * k,))
    L = np.ones((t, k)) * (ILL[:, 1].reshape((t, 1)))
    neg2_right = L.reshape((t * k,))
    result = [0, 0, 0, 0]
    for i in range(epochs):
        if i<10000: 
            if i % 50 == 0:
                out = sess.run(output_prel_e)
                neg2_left = get_neg(ILL[:, 1], out, k)
                neg_right = get_neg(ILL[:, 0], out, k)
                feeddict = {"neg_left:0": neg_left,
                            "neg_right:0": neg_right,
                            "neg2_left:0": neg2_left,
                            "neg2_right:0": neg2_right}

            sess.run(train_step_1, feed_dict=feeddict)
            if i % 25 == 0:
                th, outvec_e = sess.run([loss_1, output_prel_e],
                                                feed_dict=feeddict)
                J.append(th)
                tmp_result = get_hits(outvec_e, test)
                for j in range(len(tmp_result)):
                    if result[j] < tmp_result[j]:
                        result[j] = tmp_result[j]
                print(result)
        print('%d/%d' % (i + 1, epochs), 'epochs...')
    print(result)
    saver.save(sess, './model/'+Config.language[0:2])
    sess.close()
    return J

def res_mask_and_GCN_identity_layer(inlayer, M, act_func, name, w):
    print(' reset a mask and GCN identity weight layer')
    w0 = tf.Variable(tf.convert_to_tensor(w, tf.float32),name=name)
    tosum = tf.sparse_tensor_dense_matmul(M, tf.multiply(inlayer, w0))
    if act_func is None:
        return tosum
    else:
        return act_func(tosum)

def res_highway(layer1, layer2, name, k_gate, b_gate):
    print("reset a highway gate weight")
    kernel_gate = tf.Variable(tf.convert_to_tensor(k_gate, tf.float32),name=name + "kernel_gate")
    bias_gate = tf.Variable(tf.convert_to_tensor(b_gate, tf.float32),name=name + "bias_gate")
    transform_gate = tf.matmul(layer1, kernel_gate) + bias_gate
    transform_gate = tf.nn.sigmoid(transform_gate)
    carry_gate = 1.0 - transform_gate
    return transform_gate * layer2 + carry_gate * layer1

def get_weight(model_dir_1, model_file_1):
    result_list = []
    with tf.Session() as sess:
        new_saver = tf.train.import_meta_graph(model_dir_1+model_file_1)
        new_saver.restore(sess, tf.train.latest_checkpoint(model_dir_1))
        gcn1 = tf.get_default_graph().get_tensor_by_name("gcn1:0").eval()
        gcn2 = tf.get_default_graph().get_tensor_by_name("gcn2:0").eval()
        highway1_k = tf.get_default_graph().get_tensor_by_name("highway1kernel_gate:0").eval()
        highway2_k = tf.get_default_graph().get_tensor_by_name("highway2kernel_gate:0").eval()
        highway1_b = tf.get_default_graph().get_tensor_by_name("highway1bias_gate:0").eval()
        highway2_b = tf.get_default_graph().get_tensor_by_name("highway2bias_gate:0").eval()
        result_list =[gcn1, highway1_k, highway1_b, gcn2, highway2_k, highway2_b]
    return result_list

def re_build(dimension, act_func, e, lang, KG):
    tf.reset_default_graph()
    inlayer = get_input_layer(e, dimension, lang)
    M = get_sparse_tensor(e, KG)
    weight_list = get_weight('./model/', lang[0:2] + ".meta")
    mask_gcn1 = res_mask_and_GCN_identity_layer(inlayer, M, act_func, "gcn1", weight_list[0])
    highway1 = res_highway(inlayer, mask_gcn1, "highway1", weight_list[1], weight_list[2])
    mask_gcn2 = res_mask_and_GCN_identity_layer(highway1, M, act_func, "gcn2", weight_list[3])
    highway2 = res_highway(highway1, mask_gcn2, "highway2", weight_list[4], weight_list[5])
    return highway2

def re_test(highway2, test1, test2):
    print('initializing...')
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    print('running...')
    out_e = sess.run(highway2)
    print("the testing result for all test datapoints in DINGAL-O:")
    whole_res1 = get_hits(out_e, test1)
    print("the testing result for the new test datapoints in DINGAL-O")
    new_res1 = get_hits(out_e, test2)

   
