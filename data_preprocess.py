import numpy as np
import pickle
import os
PATH=os.path.split(os.path.split(os.path.split(os.path.split(os.path.realpath(__file__))[0])[0])[0])[0]
PATH='.'
from tqdm import tqdm
import logging
import threadpool
from threadpool import ThreadPool
import random
import re
from multiprocessing import Pool

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S',
                    filemode='w')
_logger=logging.getLogger("data_preprocess")


class Intent_Slot_Data(object):
    '''
    intent_slot 数据处理模块
    '''
    def __init__(self, train_path, dev_path, test_path, batch_size, flag,max_length,use_auto_bucket):

        #分隔符
        self.split_token='\t'
        self.train_path = train_path  # 训练文件路径
        self.dev_path = dev_path  # 验证文件路径
        self.test_path = test_path  # 测试文件路径
        self.batch_size = batch_size  # batch大小
        self.max_length = max_length
        self.use_auto_bucket=use_auto_bucket
        self.bucket_len=[0,5,15,30]
        self.index=0
        self.padd_index=0
        if flag == "train_new":
            self.vocab, self.slot_vocab, self.intent_vocab,self.intent_num_vocab,self.sent_arr,self.slot_arr,self.intent_arr,self.sent_len = self.get_vocab()
            self.num_batch = int(len(self.sent_arr) / self.batch_size)

            pickle.dump(self.vocab, open(PATH+"/save_model/vocab_dn.p", 'wb'))  # 词典
            pickle.dump(self.slot_vocab, open( PATH+"/save_model/slot_vocab_dn.p", 'wb'))  # 词典
            pickle.dump(self.intent_vocab, open(PATH+"/save_model/intent_vocab_dn.p", 'wb'))  # 词典
        elif flag == "test" or flag == "train":
            self.vocab = pickle.load(open(PATH+"/save_model/vocab_dn.p", 'rb'))  # 词典
            self.slot_vocab = pickle.load(open(PATH+"/save_model/slot_vocab_dn.p", 'rb'))  # 词典
            self.intent_vocab = pickle.load(open(PATH+"/save_model/intent_vocab_dn.p", 'rb'))  # 词典
        else:
            pass

        self.vocab_num=len(self.vocab)
        self.slot_num=len(self.slot_vocab)
        self.intent_num=len(self.intent_vocab)
        self.id2intent={}
        for k,v in self.intent_vocab.items():
            self.id2intent[v]=k

        self.id2sent={}
        for k,v in self.vocab.items():
            self.id2sent[v]=k

        self.id2slot={}
        for k,v in self.slot_vocab.items():
            self.id2slot[v]=k

    def shuffle(self, data_list):
        '''

        :param data_list:
        :return:
        '''
        index = [i for i in range(len(data_list))]
        random.shuffle(index)
        new_data_list = [data_list[e] for e in index]
        return new_data_list

    def get_vocab(self):
        '''
        构造字典 dict{NONE:0,word1:1,word2:2...wordn:n} NONE为未登录词
        :return:
        '''
        train_file = open(self.train_path, 'r')
        test_file = open(self.dev_path, 'r')
        dev_file = open(self.test_path, 'r')
        vocab = {"NONE": 0,'bos':1,'eos':2}
        intent_vocab = {}
        slot_vocab={}
        self.index = len(vocab)
        self.intent_label_index = len(intent_vocab)
        self.slot_label_index=len(slot_vocab)
        intent_num_vocab={}
        _logger.info('构建词典')

        sent_arr,slot_arr,intent_arr,sent_len=[],[],[],[]
        def _vocab(file):
            for ele in tqdm(list(file)):

                ele = ele.replace("\n", "")
                eles=ele.split(self.split_token)

                words=eles[1].split(' ')
                slots=eles[2].split(' ')

                slots=[e for e in slots if e]
                intents=[e for e in eles[3].split(' ')]

                for slots_label in slots:
                    slots_label=str(slots_label.lower().replace('[','')).replace("'",'').replace("'",'').replace(',','')
                    if slots_label not in slot_vocab :
                        slot_vocab[slots_label] = self.slot_label_index
                        self.slot_label_index += 1


                for word in words:
                    word=word.lower()
                    if word not in vocab and word!='':
                        vocab[word]=self.index
                        self.index+=1
                for intent in intents:
                    intent = intent.lower()
                    if intent not in intent_vocab and intent not in [' ','']:
                        intent_vocab[intent]=self.intent_label_index
                        self.intent_label_index+=1

                    if intent not in intent_num_vocab and intent not in ['',' ']:
                        intent_num_vocab[intent]=1
                    elif intent in intent_num_vocab and intent not in ['',' ']:
                        num=intent_num_vocab[intent]
                        num+=1
                        intent_num_vocab[intent]=num


                ss=[vocab[e] for e in words if e in vocab ]
                ss.extend(['0']*self.max_length)
                sent_arr.append(ss[:self.max_length])

                ss=[slot_vocab[e] for e in slots]
                ss.extend(['0']*self.max_length)

                slot_arr.append([slot_vocab[e] for e in slots])
                intent_arr.append([int(intent_vocab[e]) for e in intents][0])
                if len(words)>self.max_length:
                    sent_len.append(self.max_length)
                else:
                    sent_len.append(len(words))

        _vocab(train_file)
        _vocab(dev_file)
        _vocab(test_file)

        return vocab,slot_vocab,intent_vocab,intent_num_vocab,\
               np.array(sent_arr),np.array(slot_arr),np.array(intent_arr),np.array(sent_len)


    def padd_sentences_infer(self, sent_list):
        '''
        find the max length from sent_list , and standardation
        infer的动态数据构建
        :param sent_list:
        :return:
        '''
        self.bucket_len.append(0)
        bucket_len=list(set(self.bucket_len))
        words, slot_labels, intent_labels, loss_weights = [], [], [], []
        ss=[[ele,len(ele.split(self.split_token)[0].split(' '))] for ele in sent_list]
        ss.sort(key=lambda x:x[1])
        data_list=[]
        for min_l,max_l in zip(bucket_len[:-1],bucket_len[1:]):
            dd=[]
            for ele in ss:
                if ele[1]>min_l and ele[1]<=max_l:
                    dd.append(ele[0])
            if dd:
                data_list.append([dd,max_l])


        word_arr_list = []
        slot_arr_list = []
        intent_arr_list = []
        real_len_arr_list = []
        for index,(sent_list,max_len) in enumerate(data_list):
            print(index)
            word_arr = []
            slot_arr = []
            intent_arr = []
            real_len_arr = []
            real_len_arr1,word_arr1,slot_arr1,intent_arr1=[],[],[],[]

            for sent in sent_list:
                sent = sent.replace('\n', '')
                sents = sent.split(self.split_token)
                words,slot_labels,intent_labels=[],[],[]
                words.append(sents[0].split(' '))
                slot_labels.append(sents[1].split(' '))
                intent_labels.append(sents[2].split(' '))

                for sent, slot,intent in zip(words, slot_labels,intent_labels):
                    real_len=0
                    sent_list = []
                    real_len = len(sent)
                    for word in sent:
                        word = word.lower()
                        if word in self.vocab and word!='':
                            sent_list.append(self.vocab[word])
                        else:
                            sent_list.append(0)


                    slot_list = []
                    for ll in slot:
                        ll=str(ll.lower().replace('[','')).replace("'",'').replace("'",'').replace(',','')
                        if ll in self.slot_vocab:
                            slot_list.append(self.slot_vocab[ll])
                        else:
                            slot_list.append(self.slot_vocab['O'.lower()])

                    intent_list=[]
                    for ll in intent:
                        ll=ll.lower()
                        if ll in self.intent_vocab:
                            intent_list.append(self.intent_vocab[ll])
                        else:
                            intent_list.append(0)

                    intent_normal = [0] * self.intent_num
                    for e in intent_list:
                        try:
                            intent_normal[e] = 1
                        except:
                            print(e)

                    if len(sent_list) > max_len:
                        new_sent_list = sent_list[0:max_len]
                    else:
                        new_sent_list = sent_list
                        ss = [0] * (max_len - len(sent_list))
                        new_sent_list.extend(ss)

                    if len(slot_list) > max_len:
                        new_slot_list = slot_list[0:max_len]
                    else:
                        new_slot_list = slot_list
                        ss_l = [0] * (max_len - len(slot_list))
                        new_slot_list.extend(ss_l)

                    if real_len >= max_len:
                        real_len = max_len

                    real_len_arr.append(real_len)
                    word_arr.append(new_sent_list)
                    slot_arr.append(new_slot_list)
                    intent_arr.append(intent_normal)
                    del real_len
                    del new_sent_list
                    del new_slot_list
                    del intent_normal

            real_len_arr1 = np.array(real_len_arr)
            word_arr1 = np.array(word_arr)
            slot_arr1=np.array(slot_arr)
            intent_arr1 = np.array(intent_arr)

            word_arr_list.append(word_arr1)
            slot_arr_list.append(slot_arr1)
            intent_arr_list.append(intent_arr1)
            real_len_arr_list.append(real_len_arr1)
            del real_len_arr, word_arr, slot_arr, intent_arr,real_len_arr1, word_arr1, slot_arr1, intent_arr1
            import gc
            gc.collect()


        return word_arr_list, slot_arr_list, intent_arr_list, real_len_arr_list

    def padd_sentences_no_buckets(self, sent_list):
        '''
        find the max length from sent_list , and standardation
        :param sent_list:
        :return:
        '''
        # slot_labels = [' '.join(str(sent).replace('\n', '').split('\t')[1].split(' ')[:-1]) for sent in sent_list]
        # intent_labels=[str(sent).replace('\n', '').split('\t')[1].split(' ')[-1] for sent in sent_list]
        _logger.info(self.padd_index)
        self.padd_index+=1
        words,slot_labels,intent_labels,loss_weights,w_indexs=[],[],[],[],[]
        for sent in sent_list:
            sent=sent.replace('\n','')
            sents=sent.split(self.split_token)
            w_indexs.append(sents[0])
            words.append(sents[1].split(' '))
            slot_labels.append(sents[2].split(' '))
            intent_labels.append(sents[3].split(' '))
        # slot_labels = [' '.join([sent.replace('\n', '').split('\t')[1].split(' ')[:-1]]) for sent in sent_list]
        # intent_labels = [' '.join([sent.replace('\n', '').split('\t')[1].split(' ')[-1]]) for sent in sent_list]
        # print(slot_labels)
        max_len=self.max_length
        word_arr = []
        slot_arr = []
        intent_arr = []
        real_len_arr = []
        for sent, slot,intent in zip(words, slot_labels,intent_labels):
            sent_list = []
            real_len = len(sent)
            for word in sent:
                word = word.lower()

                if word in self.vocab:
                    sent_list.append(self.vocab[word])
                else:
                    sent_list.append(0)

            slot_list = []
            slots = slot
            for ll in slots:
                ll=str(ll.lower().replace('[','')).replace("'",'').replace("'",'').replace(',','')
                if ll in self.slot_vocab:
                    slot_list.append(self.slot_vocab[ll])
                else:
                    slot_list.append(self.slot_vocab['O'.lower()])

            intent_list=[]
            for ll in intent:
                ll=ll.lower()
                if ll!='':

                    intent_list.append(self.intent_vocab[ll])


            if len(sent_list) >= max_len:
                new_sent_list = sent_list[0:max_len]
            else:
                new_sent_list = sent_list
                ss = [0] * (max_len - len(sent_list))
                new_sent_list.extend(ss)

            if len(slot_list) >= max_len:
                new_slot_list = slot_list[0:max_len]
            else:
                new_slot_list = slot_list
                ss_l = [0] * (max_len - len(slot_list))
                new_slot_list.extend(ss_l)

            intent_normal=[0]*self.intent_num
            for e in intent_list:
                try:
                    intent_normal[e]=1
                except:
                    print(e)

            if real_len >= max_len:
                real_len = max_len

            real_len_arr.append(real_len)
            word_arr.append(new_sent_list)
            slot_arr.append(new_slot_list)
            intent_arr.append(np.array(intent_normal))

        real_len_arr = np.array(real_len_arr)
        word_arr = np.array(word_arr)
        slot_arr=np.array(slot_arr)
        intent_arr = np.array(intent_arr)
        loss_weight_arry=np.array(loss_weights)
        # intent_arr=np.reshape(intent_arr,(intent_arr.shape[0]))
        return word_arr, slot_arr, intent_arr, real_len_arr,w_indexs

    def data_deal_train(self):
        '''
        对训练样本按照长度进行排序 分箱
        :return:
        '''
        train_flie = open(self.train_path, 'r')
        data_list = [line for line in train_flie.readlines()]

        data_list.sort(key=lambda x: len(x))  # sort not shuffle

        num_batch = int(len(data_list) / int(self.batch_size))

        batch_list = []
        ss=[]
        for i in tqdm(list(range(num_batch))):
            ele = data_list[i * self.batch_size:(i + 1) * self.batch_size]
            ss.append(ele)

            # if self.use_auto_bucket:
            #     word_arr, slot_arr, intent_arr, real_len_arr = self.padd_sentences(ele)
            # else:
            #     word_arr, slot_arr, intent_arr, real_len_arr,w_indexs = self.padd_sentences_no_buckets(ele)

        # pool = ThreadPool(8)
        # requests = threadpool.makeRequests(self.padd_sentences_no_buckets, ss,exc_callback=self.callback)
        # [pool.putRequest(req) for req in requests]
        # pool.wait()

        _logger.info('数据标准化')
        pool = Pool(10)
        result=pool.map(self.padd_sentences_no_buckets,ss)
        for ele in tqdm(result):
            batch_list.append((ele[0],ele[1],ele[2],ele[3],ele[4]))


        # _logger.info('word:%s slot_shape:%s intent_shape:%s '%(word_arr.shape,slot_arr.shape,intent_arr.shape))
        # batch_list.append((word_arr, slot_arr,intent_arr, real_len_arr,w_indexs))
        return batch_list, num_batch

    def next_batch(self):
        '''

        :return:
        '''
        num_batch=len(self.sent_arr)/self.batch_size
        num_iter = num_batch
        if self.index < num_iter:
            return_sent = self.sent_arr[self.index*self.batch_size:(self.index+1)*self.batch_size]
            return_slot = self.slot_arr[self.index*self.batch_size:(self.index+1)*self.batch_size]
            return_intent = self.intent_arr[self.index*self.batch_size:(self.index+1)*self.batch_size]
            return_real_len=self.sent_len[self.index*self.batch_size:(self.index+1)*self.batch_size]
            self.index += 1
        else:
            self.index = 0
            return_sent = self.sent_arr[self.index * self.batch_size:(self.index + 1) * self.batch_size]
            return_slot = self.slot_arr[self.index * self.batch_size:(self.index + 1) * self.batch_size]
            return_intent = self.intent_arr[self.index * self.batch_size:(self.index + 1) * self.batch_size]
            return_real_len = self.sent_len[self.index * self.batch_size:(self.index + 1) * self.batch_size]

        return return_sent, return_slot, return_intent,return_real_len

    def get_dev(self):
        train_flie = open(self.dev_path, 'r')
        data_list = [line for line in train_flie.readlines()]

        data_list.sort(key=lambda x: len(x))  # sort not shuffle

        num_batch = int(len(data_list) / int(self.batch_size))

        batch_list = []
        ele = data_list[:]
        if self.use_auto_bucket:
            word_arr, slot_arr, intent_arr, real_len_arr = self.padd_sentences_infer(ele)
            _logger.info('word:%s' % len(word_arr))

        else:
            word_arr, slot_arr, intent_arr, real_len_arr,w_indexs = self.padd_sentences_no_buckets(ele)
            _logger.info('word:%s slot_shape:%s intent_shape:%s ' % (word_arr.shape, slot_arr.shape, intent_arr.shape))

        return  word_arr, slot_arr, intent_arr, real_len_arr,w_indexs

    def get_train(self):
        train_flie = open(self.train_path, 'r')
        data_list = [line for line in train_flie.readlines()]

        data_list.sort(key=lambda x: len(x))  # sort not shuffle

        num_batch = int(len(data_list) / int(self.batch_size))

        batch_list = []
        ele = data_list[:]
        if self.use_auto_bucket:
            word_arr, slot_arr, intent_arr, real_len_arr = self.padd_sentences_infer(ele)
            _logger.info('word:%s' % len(word_arr))

        else:
            word_arr, slot_arr, intent_arr, real_len_arr,w_indexs = self.padd_sentences_no_buckets(ele)
            _logger.info('word:%s slot_shape:%s intent_shape:%s ' % (word_arr.shape, slot_arr.shape, intent_arr.shape))

        return  word_arr, slot_arr, intent_arr, real_len_arr,w_indexs


    def get_sent_char(self,sent_list):

        pattern = '\d{1,3}(\\.|，|、|？)|《|》|？|。| '


        res=[]
        res_vec=[]
        sent_list = [re.subn(pattern, '', sent)[0] for sent in sent_list]
        sent_list=er.get_entity(sent_list)
        for sent in sent_list:
            ss=[]
            ss_new=['eos']
            ss_new.extend(sent)
            ss_new.append('bos')
            ss_=ss_new
            for word in ss_:
                word=word.lower()
                if word in self.vocab:
                    ss.append(self.vocab[word])
                else:
                    ss.append(self.vocab['NONE'])

            if len(ss)>=self.max_length:
                res_vec.append(self.max_length)
                ss=ss[:self.max_length]
            else:
                res_vec.append(len(ss))
                padd=[0]*(self.max_length-len(ss))
                ss.extend(padd)
            res.append(ss)
        sent_arr=np.array(res)
        sent_vec=np.array(res_vec)
        return sent_arr,sent_vec



if __name__ == '__main__':

    dd = Intent_Slot_Data(train_path="./data/gen_data_small.txt",
                              test_path="./data/dev_out_char.txt",
                              dev_path="./data/dev_out_char.txt", batch_size=20 ,max_length=30,
                          flag="train_new",use_auto_bucket=False)
    print(dd.intent_vocab)
    return_sent, return_slot, return_intent, return_real_len=dd.next_batch()
    print(return_real_len)

    # for k,v in dd.vocab.items():
    #     print(k,v)
    # ss=dd.get_sent_char(['康爱保保什么'])[0]
    # print(ss)
    # print(sent)
    # print(slot)
    # print(intent)
    # print(real_len)

    #
    #
    #
    #
    # print(sent)

    # print(loss_weight)
    # print(sent[0])
    # print(''.join([dd.id2sent[e] for e in sent[0]]))
    # #     print()
    # #     print(sent.shape,slot.shape,intent.shape,real_len.shape,cur_len.shape)
    #     # print(dd.slot_vocab)
    #     # print(cur_len)
    # sent='专业导医陪诊是什么服务'
    # sent_arr,sent_vec=dd.get_sent_char([sent])
    # # print(sent_arr)
    # print(sent_arr,sent_vec)

