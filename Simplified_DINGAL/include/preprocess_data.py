import numpy as np
import json
import pickle
def get(KG):
    kg_adj_dic = {}
    for tri in KG:
        if tri[0] not in kg_adj_dic.keys():
            kg_adj_dic[tri[0]] = []
        if tri[2] not in kg_adj_dic[tri[0]]:
            kg_adj_dic[tri[0]].append(tri[2])
        if tri[2] not in kg_adj_dic.keys():
            kg_adj_dic[tri[2]] = []
        if tri[2] not in kg_adj_dic[tri[0]]:
            kg_adj_dic[tri[2]].append(tri[0])
    return kg_adj_dic

def get_input_feat(dimension, lang, new2old):
    #print('adding the primal input layer...')
    with open(file='data/' + lang + '_en/' + lang + '_vectorList.json', mode='r', encoding='utf-8') as f:
        embedding_list = json.load(f)
        #print(len(embedding_list), 'rows,', len(embedding_list[0]), 'columns.')
    node_num  = len(new2old.keys())
    print(embedding_list[0][1])
    print("node number: " + str(node_num) + '\n')
    new_embedding = np.zeros([node_num, len(embedding_list[0])])
    print("shape: " + str(new_embedding.shape))
    for i in range(node_num):
        new_embedding[i] = embedding_list[new2old[i]]
    #print(new_embedding[39000])
    return new_embedding
    #input_embeddings = tf.convert_to_tensor(embedding_list)
    #ent_embeddings = tf.Variable(input_embeddings)
    #return tf.nn.l2_normalize(input_embeddings, 1)

def split_data(iLL, KG1, KG2, dim, lang, ratio):
    old2new = dict()
    new2old = dict()
    reserved_entity_pair = iLL[:int(len(iLL)*ratio)]
    print("The reserved entity pair number:", len(reserved_entity_pair))
    deleted_entity_pair = iLL[int(len(iLL)*ratio):]
    deleted_nodes = []
    for pair in deleted_entity_pair:
        deleted_nodes.append(pair[0])
        deleted_nodes.append(pair[1])
    print("The deleted node number:", len(deleted_nodes))
    KG1 = get(KG1)
    KG2 = get(KG2)
    i = 0
    for node in KG1.keys():
        if node not in deleted_nodes:
            old2new[node] = i
            new2old[i] = node
            i += 1
    for node in KG2.keys():
        if node not in deleted_nodes:
            old2new[node] = i
            new2old[i] = node
            i += 1
    print("total number of reserved nodes: " + str(i) + '\n')
    new_KG1_dic = dict()
    new_KG2_dic = dict()
    for node in KG1.keys():
        if node in old2new.keys():
            if old2new[node] not in new_KG1_dic.keys():
                new_KG1_dic[old2new[node]] = []
            for neighbour in KG1[node]:
                if neighbour in old2new.keys() and neighbour not in new_KG1_dic[old2new[node]]:
                    new_KG1_dic[old2new[node]].append(old2new[neighbour])
    for node in KG2.keys():
        if node in old2new.keys():
            if old2new[node] not in new_KG2_dic.keys():
                new_KG2_dic[old2new[node]] = []
            for neighbour in KG2[node]:
                if neighbour in old2new.keys() and neighbour not in new_KG2_dic[old2new[node]]:
                    new_KG2_dic[old2new[node]].append(old2new[neighbour])
    ori_iLL = []
    e = len(new_KG1_dic.keys()) + len(new_KG2_dic.keys())
    print("e: " + str(e) + '\n')
    for pair in reserved_entity_pair:
        ori_iLL.append((old2new[pair[0]], old2new[pair[1]]))
    new_KG1 = []
    new_KG2 = []
    for node in new_KG1_dic.keys():
        for neighbour in new_KG1_dic[node]:
            new_KG1.append((node, 1, neighbour))
    for node in new_KG2_dic.keys():
        for neighbour in new_KG2_dic[node]:
            new_KG2.append((node, 1, neighbour))
    feat_matrix = get_input_feat(dim, lang, new2old)
    return ori_iLL, e, new_KG1, new_KG2, feat_matrix, deleted_entity_pair, new2old

def recover_new2old(test, new2old):
    old_test = []
    for test_pair in test:
        old_test.append((new2old[test_pair[0]], new2old[test_pair[1]]))
    return old_test

