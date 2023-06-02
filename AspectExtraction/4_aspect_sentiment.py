from cgitb import text
from re import T
from senta import Senta
import pickle
import json
from OpinionLexion.opinion_lexion_classifer import Opinion_lexion_classifer
from tqdm import tqdm
from time import time
import pandas as pd
# texts = ["I like the gril.","She is a ugly girl"]
# aspects = ["gril","girl"]
# result = my_senta.predict(texts, aspects)

# print(result)


def readfile(filename):
    f = open(filename, encoding='utf-8')
    data = []
    for line in f.readlines():
        line = json.loads(line)
        data.append(line['reviewText'])
    f.close()
    return data

def readfile2(filename):
    f = open(filename, encoding='utf-8')
    data = []
    for line in f.readlines():
        line = json.loads(line)
        one = [line['reviewerID'],line['asin']]
        data.append(one)
    f.close()
    return data


def load_as_pairs(path):
    """ 读取as.pkl文件"""
    with open(path,"rb") as f:
        res=pickle.load(f)
    return res

def build_input(reviews_path,as_pairs_path):
    """
    构建review和aspect构成的元组的列表，如果没有as_pair,则为空列表
    """
    reviews=readfile(reviews_path)
    as_pairs=load_as_pairs(as_pairs_path)
    all_inputs=[]
    for r,as_pair in zip(reviews,as_pairs):
        senta_inputs_texts=[]
        senta_inputs_aspects=[]
        olc_inputs=[]
        real_aspects=[]
        for a,a_,s in as_pair:
            senta_inputs_texts.append(r)
            senta_inputs_aspects.append(a)
            real_aspects.append(a_)
            olc_inputs.append(s)
        all_inputs.append((senta_inputs_texts,senta_inputs_aspects,real_aspects,olc_inputs))
    return all_inputs

def get_aspect_sentiments(review_file,pkl_file):

    ###################  1. 加载模型  ###################
    my_senta = Senta()
    olc=Opinion_lexion_classifer("ASPE/OpinionLexion")
    use_cuda = True
    print(my_senta.get_support_model())
    my_senta.init_model(model_class="ernie_2.0_skep_large_en", task="aspect_sentiment_classify", use_cuda=use_cuda)    
    all_inputs=build_input(review_file,pkl_file)
    all_sente_nodes=[]

    ###################  2. 情感分析  ###################
    for texts,aspects,real_aspects,senti_words in tqdm(all_inputs,ncols=100): # TODO：在这块存在性能瓶颈
        
        if len(texts)>0:
            # 1. 基于skep的情感预测
            res=my_senta.predict(texts,aspects)
            res1=[ i[1] for i in res]
            sentiments_1=[]
            for s in res1:
                if s=="postive":
                    ss=1
                elif s=="negative":
                    ss=-1
                else:
                    ss=0
                sentiments_1.append(ss)
            # 2. 基于字典的预测
            sentiments_2=olc.predict(senti_words)
            all_sentiments=[(a,b) for a,b in zip(sentiments_1,sentiments_2)]
            senti_node=["_".join(map(str,i)) for i in all_sentiments]
            all_sente_nodes.append([(a,s) for a,s in zip(real_aspects,senti_node)])
        else:
            all_sente_nodes.append("None")
        pass

def get_aspect_sentiments_quick(review_file,pkl_file):

    ###################  1. 加载模型  ###################
    my_senta = Senta()
    olc=Opinion_lexion_classifer("ASPE/OpinionLexion")
    all_inputs=build_input(review_file,pkl_file)
    all_sente_nodes=[]

    ###################  2. 情感分析  ###################
    for texts,aspects,real_aspects,senti_words in tqdm(all_inputs,ncols=100): # TODO：在这块存在性能瓶颈
        
        if len(texts)>0:
            # 2. 基于字典的预测
            sentiments_2=olc.predict(senti_words)
            all_sente_nodes.append([(a,s) for a,s in zip(real_aspects,sentiments_2)])
        else:
            all_sente_nodes.append("None")
        pass
    return all_sente_nodes

def get_aspect_sentiments2(review_file,pkl_file):
    # 加速版本

    ###################  1. 加载模型  ###################
    my_senta = Senta()
    olc=Opinion_lexion_classifer("ASPE/OpinionLexion")
    use_cuda = False
    print(my_senta.get_support_model())
    my_senta.init_model(model_class="ernie_2.0_skep_large_en", task="aspect_sentiment_classify", use_cuda=use_cuda)    
    all_inputs=build_input(review_file,pkl_file)
    all_sente_nodes=[]

    t1=[]
    a1=[]
    ###################  2. 情感分析  ###################
    for texts,aspects,senti_words in tqdm(all_inputs,ncols=100): # TODO：在这块存在性能瓶颈
        t1.extend(texts)
        a1.extend(aspects)
    t0=time()
    # 一次还不能预测太多
    batch_size=512
    for i in range(0,len(t1),batch_size):
        res=my_senta.predict(t1[i:i+batch_size],a1[i:i+batch_size])
    print(time()-t0)


def proceess_all_datasets():
    ###################  1. 数据集处理  ###################
    paths = [
    #    "Automotive_5.json",
    #     "Digital_Music_5.json",
    #     'Musical_Instruments_5.json',
    #     "Pet_Supplies_5.json",
    #     "Sports_and_Outdoors_5.json",
        "Toys_and_Games_5.json",
        # "Tools_and_Home_Improvement_5.json",
        # "Office_Products_5.json",
        # 'Grocery_and_Gourmet_Food_5.json',
        # "Video_Games_5.json",
        # "Beauty_5.json"
        # "yelp2.json"
    ]
    save_dir="data/"
    pkl_path_fmt=save_dir+"{}/as3.pkl"
    kg_path_fmt=save_dir+"{}/as_kg_Test.txt"
    data_path_fmt="data/{}"

    ###################  2. 处理逻辑  ###################
    for p in paths:
        data_path=data_path_fmt.format(p)
        pkl_path=pkl_path_fmt.format(p[:-5])
        kg_path=kg_path_fmt.format(p[:-5])

        all_sente_nodes=get_aspect_sentiments_quick(data_path,pkl_path)
        ui_pairs=readfile2(data_path)
        # 以uid, iid, aspect, senti_node
        # with open(kg_path,"w") as f:
        #     for (u,i), as_paris in zip(ui_pairs,all_sente_nodes):
        #         if as_paris=="None":
        #             continue
        #         for a,s in as_paris:
        #             f.write("{},{},{},{}\n".format(u,i,a,s)) 
        # 
        u_list=[]
        i_list=[]
        a_list=[]
        s_list=[]
        for (u,i), as_paris in zip(ui_pairs,all_sente_nodes):
            if as_paris=="None":
                continue
            for a,s in as_paris:
                u_list.append(u)  
                i_list.append(i)  
                a_list.append(a)  
                s_list.append(s)
        df=pd.DataFrame({"user":u_list,"item":i_list,"aspect":a_list,"sentiment":s_list})
        df.to_csv(kg_path,index=False,header=False)  


if __name__=="__main__":
    proceess_all_datasets()

    pass    