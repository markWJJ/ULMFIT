from data_preprocess import Intent_Slot_Data
from model import LanguageModel
import tensorflow as tf
import numpy as np
import os
gpu_id=2
os.environ["CUDA_VISIBLE_DEVICES"] = "%s"%gpu_id
tf.app.flags.DEFINE_float("mask_lambda1", 0.02, "l2学习率")
tf.app.flags.DEFINE_float("mask_learning_rate", 0.001, "学习率")
tf.app.flags.DEFINE_float("mask_keep_dropout", 0.6, "dropout")
tf.app.flags.DEFINE_integer("mask_batch_size", 64, "批处理的样本数量")
tf.app.flags.DEFINE_integer("mask_max_len", 30, "句子长度")
tf.app.flags.DEFINE_integer("mask_embedding_dim", 400, "词嵌入维度.")
tf.app.flags.DEFINE_integer("mask_hidden_dim", 1150, "中间节点维度.")
tf.app.flags.DEFINE_integer("mask_epoch", 50, "epoch次数")
tf.app.flags.DEFINE_string("mask_summary_write_dir", './', "训练数据过程可视化文件保存地址")
tf.app.flags.DEFINE_string("mask_model_dir", './save_model', "模型保存路径")
tf.app.flags.DEFINE_boolean('mask_use_Encoder2Decoder',False,'123')
tf.app.flags.DEFINE_string("mask_mod", "infer_dev", "默认为训练")  # true for prediction
FLAGS = tf.app.flags.FLAGS

dd = Intent_Slot_Data(train_path="./data/train_out_char.txt",
                                  test_path="./data/dev_out_char.txt",
                                  dev_path="./data/dev_out_char.txt", batch_size=FLAGS.mask_batch_size,
                                  max_length=FLAGS.mask_max_len, flag="train",
                                  use_auto_bucket=False)


mode='domain_train'

if mode=='pre_train':
    vocab_num=dd.vocab_num
    print(vocab_num)
    print(dd.num_batch)
    with tf.device('/device:GPU:%s' % gpu_id):
        LM = LanguageModel(seq_len=FLAGS.mask_max_len, eln=3, hidden_dim=FLAGS.mask_hidden_dim, emb_size=FLAGS.mask_embedding_dim,
                       vocab_size=vocab_num, scope='lm')
        LM.train(dd,10)

elif mode=='domain_train':
    with tf.device('/device:GPU:%s' % gpu_id):

        saver = tf.train.import_meta_graph("./save_model/lm.ckpt.meta")

        # We can now access the default graph where all our metadata has been loaded
        graph = tf.get_default_graph()
        #
        for ele in graph.get_operations():
            print(ele.name)
        with tf.variable_scope(name_or_scope='domain_train'):
            input_sent=graph.get_tensor_by_name('lm/input_sent:0')
            input_len=graph.get_tensor_by_name('lm/input_len:0')
            droput = graph.get_tensor_by_name('lm/dropout:0')

            # encoder=graph.get_tensor_by_name('lstm_encoder2/rnn/transpose_1:0')
            # decoder=graph.get_tensor_by_name('Linear_decoder/dense/BiasAdd:0')

            loss=graph.get_tensor_by_name('sparse_softmax_cross_entropy_loss/value:0')

            opt=tf.train.AdamOptimizer(0.001).minimize(loss)

        saver_restore = tf.train.import_meta_graph("./save_model/lm.ckpt.meta")
        config = tf.ConfigProto(allow_soft_placement=True)
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            saver_restore.restore(sess,"./save_model/lm.ckpt")

            train_sent, train_slot, train_intent, train_rel_len, _ = dd.get_train()

            for _ in range(10):
                loss_,_ = sess.run(
                    [loss,opt],
                    feed_dict={input_sent: train_sent,
                               input_len: train_rel_len,
                               droput: np.array([0.6, 0.7, 0.8], dtype=np.float32)
                               })
                print(loss_)


elif mode=='classify':
    saver_restore = tf.train.import_meta_graph("./save_model/lm.ckpt.meta")

    # We can now access the default graph where all our metadata has been loaded
    graph = tf.get_default_graph()
    #


    input_sent = graph.get_tensor_by_name('lm/input_sent:0')
    input_len = graph.get_tensor_by_name('lm/input_len:0')
    droput = graph.get_tensor_by_name('lm/dropout:0')
    # intent_y=graph.get_tensor_by_name('lm/class_y:0')
    intent_y=tf.placeholder(shape=(None,dd.intent_num),dtype=tf.int32)


    # encoder = graph.get_tensor_by_name('lstm_encoder2/rnn/transpose_1:0')
    decoder = graph.get_tensor_by_name('Linear_decoder/dense/BiasAdd:0')
    decoder1 = tf.expand_dims(decoder, -1)

    with tf.variable_scope(name_or_scope='classify'):
        out=tf.nn.max_pool(decoder1,ksize=(1,input_sent.get_shape()[1],1,1),padding='VALID',strides=(1,2,1,1))
        out=tf.squeeze(out,-1)
        out=tf.squeeze(out,1)
        out=tf.layers.dense(out,dd.intent_num)

        logit=tf.nn.softmax(out,name='logit')

        # loss1=tf.losses.sparse_softmax_cross_entropy(labels=intent_y,logits=out)
        loss1=tf.losses.softmax_cross_entropy(onehot_labels=intent_y,logits=out)


        opt1=tf.train.AdamOptimizer(0.001).minimize(loss1)

    saver=tf.train.Saver()
    config = tf.ConfigProto(allow_soft_placement=True)
    with tf.Session(config=config) as sess:
        saver.restore(sess, './save_model/class.ckpt')

        # sess.run(tf.global_variables_initializer())
        # saver_restore.restore(sess, "./save_model/lm.ckpt")

        train_sent, train_slot, train_intent, train_rel_len, _ = dd.get_dev()
        print(train_intent.shape)
        for _ in range(20):
            loss_, _ = sess.run(
                [loss1, opt1],
                feed_dict={input_sent: train_sent,
                           input_len: train_rel_len,
                           droput: np.array([0.6, 0.7, 0.8], dtype=np.float32),
                           intent_y:train_intent
                           })
            print(loss_)
            saver.save(sess,'./save_model/class.ckpt')

elif mode=='infer':
    saver_restore = tf.train.import_meta_graph("./save_model/class.ckpt.meta")


    # We can now access the default graph where all our metadata has been loaded
    graph = tf.get_default_graph()

    input_sent = graph.get_tensor_by_name('lm/input_sent:0')
    input_len = graph.get_tensor_by_name('lm/input_len:0')
    droput = graph.get_tensor_by_name('lm/dropout:0')
    logit=graph.get_tensor_by_name('classify/logit:0')


    saver=tf.train.Saver()
    config = tf.ConfigProto(allow_soft_placement=True)
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        saver_restore.restore(sess, "./save_model/class.ckpt")

        train_sent, train_slot, train_intent, train_rel_len, _ = dd.get_train()
        logit_ = sess.run(
            logit,
            feed_dict={input_sent: train_sent,
                       input_len: train_rel_len,
                       droput: np.array([0.6, 0.7, 0.8], dtype=np.float32),
                       })
        arg_logit=np.argmax(logit_,1)

        for sent,label in zip(train_sent,arg_logit):
            ss=''.join([dd.id2sent[e] for e in sent if e in dd.id2sent and e not in [0]])
            intent=dd.id2intent[label]
            print(ss,'\t\t',intent)
        # print(arg_logit)
