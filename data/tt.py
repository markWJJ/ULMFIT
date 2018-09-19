import json
from collections import defaultdict
from tqdm import tqdm
import time
text=''
for char in tqdm(["a", "b", "c", "d"]):
    text = text + char
    time.sleep(0.5)


data_dict=defaultdict(list)
data=json.load(open('./cluster_corpus.json','rb'))
index=0
for k in tqdm(data):
    v=data[k]
    for e in v:
        if len(e)>20:
            data_dict[index].append(e.replace('\n',''))
    index+=1
#
fw=open('gen_data.txt','w')

index=0
for k in tqdm(data_dict):
    v=data_dict[k]
    for e in v:
        char=' '.join([ee for ee in e])
        fw.write(str(index))
        fw.write('\t')
        fw.write(char)
        fw.write('\t')
        fw.write('o o o')
        fw.write('\t')
        fw.write(str(k))
        fw.write('\n')
        index+=1

s=len(open('./gen_data.txt','r').readlines())
print(s)