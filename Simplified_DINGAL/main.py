import sys
sys.path.remove('/usr/local/lib/python3.7/site-packages')
from include.Config import Config
from include.Model import build, training, get_weight, re_build, re_test
from include.Test import *
from include.Load import *
import os
import sys
import tensorflow as tf
from include.preprocess_data import *

'''
Follow the code style of RDGCN:
'''
if __name__ == '__main__':
    e = len(set(loadfile(Config.e1, 1)) | set(loadfile(Config.e2, 1)))
    print(e)
    ILL = loadfile(Config.ill, 2)
    illL = len(ILL) 
    print(illL)
    np.random.shuffle(ILL)
    KG1 = loadfile(Config.kg1, 3)
    KG2 = loadfile(Config.kg2, 3)
    """
    The last parameter is the ratio of reserved entities, if it is 0.9 it means we only have 0.1 in the dynamic step.
    """
    new_iLL, e, new_KG1, new_KG2, feat_matrix, deleted_iLL, new2old = split_data(ILL, KG1, KG2, Config.dim, Config.language[0:2], 0.9)

    illL = len(new_iLL)
    train = np.array(new_iLL[:illL // 10 * Config.seed])
    test = new_iLL[illL // 10 * Config.seed:]
    output_prel_e, loss_1 = build(Config.dim, Config.act_func,
                                                                                  Config.gamma, Config.k, e, train, new_KG1 + new_KG2, feat_matrix)
    J = training(output_prel_e, loss_1, 0.001, Config.epochs, train,
                 Config.k, Config.s, test)
    e = len(set(loadfile(Config.e1, 1)) | set(loadfile(Config.e2, 1)))
    print(e)
    KG1 = loadfile(Config.kg1, 3)
    KG2 = loadfile(Config.kg2, 3)
    update_embeddings = re_build(Config.dim, Config.act_func, e, Config.language[0:2], KG1 + KG2)
    re_test(update_embeddings, deleted_iLL + recover_new2old(test, new2old), deleted_iLL)
