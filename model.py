import tensorflow as tf
import numpy as np
import os
gpu_id=2
os.environ["CUDA_VISIBLE_DEVICES"] = "%s"%gpu_id



class LanguageModel():
    def __init__(self,seq_len,eln,emb_size,hidden_dim,vocab_size,scope,mod='pre_train',class_num=None,**kwargs):

        self.seq_len=seq_len
        self.eln=eln #enocder layer num
        self.mod=mod
        self.hidden_dim=hidden_dim
        self.class_num=class_num
        self.vocab_size=vocab_size
        self.emb_size=emb_size

        with tf.variable_scope(name_or_scope=scope):

            self.input=tf.placeholder(shape=(None,seq_len),dtype=tf.int32,name='input_sent')
            self.input_len=tf.placeholder(shape=(None,),dtype=tf.int32,name='input_len')
            self.dropout=tf.placeholder(shape=(self.eln,),dtype=tf.float32,name='dropout')
            self.class_y=tf.placeholder(shape=(None,),dtype=tf.int32,name='class_y')


            with tf.variable_scope(name_or_scope='emb'):
                emb=tf.Variable(tf.random_normal(shape=(self.vocab_size,self.emb_size),dtype=tf.float32))

                self.input_emb=tf.nn.embedding_lookup(emb,self.input,name='input_emb')



    def Encoder(self,scope='lstm_encoder',**kwargs):
        '''

        :param input:
        :param seq_len:
        :param kwargs:
        :return:
        '''
        with tf.device('/device:GPU:%s'%gpu_id):

            lstm_input=self.input_emb
            for i in range(self.eln):
                with tf.variable_scope(name_or_scope=scope+'%s'%i):
                    cell=tf.contrib.rnn.BasicLSTMCell(self.hidden_dim)
                    cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=tf.to_float(self.dropout[i]))
                    encoder, _ = tf.nn.dynamic_rnn(
                        cell,
                        lstm_input,
                        dtype=tf.float32,
                        sequence_length=self.input_len,
                    )
                    lstm_input=encoder

            return lstm_input


    def Decoder(self,encoder_input,decoder_dim,scope='Linear_decoder'):
        '''
        线性解码
        :param scope:
        :return:
        '''
        with tf.device('/device:GPU:%s'%gpu_id):

            with tf.variable_scope(name_or_scope=scope):

                out=tf.layers.dense(encoder_input,decoder_dim)
            return out

    def Loss(self,decoder_out,loss_fn=None):
        '''

        :param decoder_out:
        :param loss_fn:
        :return:
        '''
        with tf.device('/device:GPU:%s'%gpu_id):

            self.loss=tf.losses.sparse_softmax_cross_entropy(labels=self.input,logits=decoder_out)

            return self.loss

    def Opt(self,loss,opt_fn=None):
        '''

        :param loss:
        :param opt_fn:
        :return:
        '''
        with tf.device('/device:GPU:%s'%gpu_id):

            return tf.train.AdamOptimizer(0.001).minimize(loss)


    def train(self,dd,epoch):
        '''

        :param dd:
        :return:
        '''

        encoder_out = self.Encoder()
        decoder_out = self.Decoder(encoder_out, dd.vocab_num)
        self.loss=self.Loss(decoder_out)
        self.opt=self.Opt(self.loss)

        config = tf.ConfigProto(allow_soft_placement=True)
        saver=tf.train.Saver()

        with tf.Session(config=config) as sess:
            num_batch = dd.num_batch
            #
            # saver = tf.train.Saver()
            # if os.path.exists('%s.meta' % FLAGS.mask_model_dir):
            #     saver.restore(sess, FLAGS.mask_model_dir)
            # else:
            sess.run(tf.global_variables_initializer())
            # dev_sent, dev_slot, dev_intent, dev_rel_len, _ = dd.get_dev()
            # train_sent, train_slot, train_intent, train_rel_len, _ = dd.get_train()
            for i in range(epoch):
                all_loss=[]
                for j in range(num_batch):
                    sent, slot, intent_label, rel_len = dd.next_batch()

                    loss_,_ = sess.run(
                        [self.loss, self.opt,],
                        feed_dict={self.input: sent,
                                   self.input_len: rel_len,
                                   self.dropout: np.array([0.6,0.7,0.8])
                                   })
                    all_loss.append(loss_)

                print('%s:%s'%(i,np.mean(loss_)))

                # loss= sess.run(
                #     self.loss,
                #     feed_dict={self.input: train_sent,
                #                self.input_len: train_rel_len,
                #                self.dropout: np.array([0.6, 0.7, 0.8],dtype=np.float32)
                #                })
                # print(i,'\t\t',loss)
                saver.save(sess,'./save_model/lm.ckpt')
if __name__ == '__main__':
    LM=LanguageModel(seq_len=20,eln=3,hidden_dim=1150,emb_size=400,vocab_size=2000,scope='lm')
    encoder_out=LM.Encoder()
    decoder_out=LM.Decoder(encoder_out,400)
    print(decoder_out)



