#!/usr/bin/env python
# coding: utf-8

# # README
# * 呼叫 `get_AML_person(content, ckip_name, mode=0, binary=0)` 即可
#     * `mode=0`, `binary` 無作用；使用模型預測 binary
#     * `mode=1`, `binary` 用來放其他模型的 binary 分類輸出 (int 1 or 0)

# In[ ]:


import pandas as pd
import ast
import numpy as np
import re
from zhon.hanzi import stops, non_stops
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from transformers import BertConfig, BertTokenizer, BertModel

import datetime #####
import math


# In[ ]:


PRETRAINED_MODEL_NAME = 'bert_wwm_pretrain_tbrain' # pretrained_bert_wwm
MODEL_PATH = './model/pre_bert_wwm_bio_only_EPOCHES_9(w359).pkl'

def clean_string(content):
    content = content.replace('\n','。').replace('\t','，').replace('!', '！').replace('?', '？')# erease white space cause English name error
    content = re.sub("[+\.\/_,$%●▼►^*(+\"\']+|[+——~@#￥%……&*（）★]", "",content)
    content = re.sub(r"[%s]+" %stops, "。",content)
    result = []
    for i in range(math.ceil(len(content) / 511)):
        result.append(content[i*512 : i*512+511])
    return result

def bio_2_string(have_AML, BIO_tagging, ckip_result, origin_text, BIO_prob):
  result = []
  if (have_AML.item() == 0):
    result.append('')
  else:
"""一個span找一次名字(慢很多)"""
#     for j in range(1, 512):
#       if (BIO_tagging[j] == 0):
#         start = j
#         end = j + 1
#         while (end < 512 and BIO_tagging[end] == 1):
#           end += 1
#         if (end > start + 1):
#           if (start <= 3):
#               s = origin_text[start-1 : end + 2] # -1 for CLS
# #               print(BIO_prob[start : end + 3])
#           else:
#               s = origin_text[start-1-1 : end + 2] # -1 for CLS
# #               print(BIO_prob[start-1 : end + 3])
# #           print('origin_span: ', origin_text[start-1 : end-1])
# #           print(s)
#           for k in range(len(ckip_result)):
#             if (len(ckip_result[k]) < 2):
#               continue
#             elif (re.findall(r"[%s]+" %non_stops, ckip_result[k]) != [] \
#                      or re.findall(r"[%s]+" %stops, ckip_result[k]) != []): # 有標點
#               continue
#             found = s.find(ckip_result[k])
#             if (found != -1):
# #               print('found: ', found)
#               result.append(ckip_result[k])

"""把span串在一起找名字(比較快)"""
    full_str = ""
    for j in range(1, 512):
      if (BIO_tagging[j] == 0):
        start = j
        end = j + 1
        while (end < 512 and BIO_tagging[end] == 1):
          end += 1
        if (end > start + 1):
          if (start <= 3):
              s = origin_text[start-1 : end + 2] # -1 for CLS
#               print(BIO_prob[start : end + 3])
          else:
              s = origin_text[start-1-1 : end + 2] # -1 for CLS
#               print(BIO_prob[start-1 : end + 3])
          print('origin_span: ', origin_text[start-1 : end-1])
          print(s)
          full_str += s
    for k in range(len(ckip_result)):
      if (len(ckip_result[k]) < 2):
        continue
      elif (re.findall(r"[%s]+" %non_stops, ckip_result[k]) != []                  or re.findall(r"[%s]+" %stops, ckip_result[k]) != []): # 有標點
        continue
      found = full_str.find(ckip_result[k])
      if (found != -1):
#       print('found: ', found)
        result.append(ckip_result[k])
      
    if (len(result) == 0):
      result.append('')
  return result

def get_predictions(model, tokens_tensors, segments_tensors, masks_tensors, ckip_names, origin_text, mode, binary):
  result = []
  with torch.no_grad():
      tokens_tensors = tokens_tensors.to("cuda:0")
      segments_tensors = segments_tensors.to("cuda:0")
      masks_tensors = masks_tensors.to("cuda:0")
#       start = datetime.datetime.now().timestamp() ######
      outputs = model(input_ids=tokens_tensors, 
                  token_type_ids=segments_tensors, 
                  attention_mask=masks_tensors)
#       end = datetime.datetime.now().timestamp()###########
#       print("through model time: ", end-start) ##########
      
      count = outputs[0].shape[0]
      for i in range(count):  # run batchsize times
        if (mode == 0):
            have_AML = outputs[0][i].argmax()
        else:
            have_AML = torch.tensor([binary])
        BIO_pred = outputs[0][i].argmax(1) # 3*512 into class label
        ckip_names_list = ast.literal_eval(ckip_names) # string to list
#         print(origin_text[i])
#         start = datetime.datetime.now().timestamp() ######
        r = bio_2_string(have_AML, BIO_pred, ckip_names_list, origin_text[i], outputs[0][i])  #####
#         end = datetime.datetime.now().timestamp()###########
#         print("bio_2_string time: ", end-start) ##########
        result.append(r)
  return result

""" model budling """
class AMLPredictModel(nn.Module):
    def __init__(self, config):
        super(AMLPredictModel, self).__init__()
        self.bert = BertModel.from_pretrained(PRETRAINED_MODEL_NAME, config = config)
        self.BIO_classifier = nn.Sequential(
                        nn.Linear(config.hidden_size, 3),
        ) # BIO tagging
        self.softmax = nn.Softmax(-1)

    def forward(self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        BIO = self.BIO_classifier(outputs[0]) # 512*HIDDENSIZE word vectors
        BIO = self.softmax(BIO)
        
        outputs = (BIO,) + outputs[2:]
        return outputs
    
config = BertConfig.from_pretrained(PRETRAINED_MODEL_NAME, output_hidden_states=True)
tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)
model = AMLPredictModel(config)
model.load_state_dict(torch.load(MODEL_PATH))
    
def get_AML_person(model, content, ckip_name, mode=0, binary=0):
#     start = datetime.datetime.now().timestamp() ######
    content = clean_string(content)
#     end = datetime.datetime.now().timestamp()###########
#     print("clean_string time: ", end-start) ##########
#     start = datetime.datetime.now().timestamp() #####
    test_input_dict = tokenizer.batch_encode_plus(content, 
                          add_special_tokens=True,
                          max_length=512,
                          return_special_tokens_mask=True,
                          pad_to_max_length=True,
                          return_tensors='pt',
                          truncation=True)
#     end = datetime.datetime.now().timestamp()###########
#     print("tokenizer time: ", end-start) ##########


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model = model.eval()
    r = get_predictions(model, test_input_dict['input_ids'], test_input_dict['token_type_ids'], test_input_dict['attention_mask'],                           ckip_name, content, mode, binary)
#     print(result)
    result = set()
    for i in range(len(r)):
        result = result | set(r[i])
    return result


# ---
# ## name classifier (QA model) by Houg Yun

# In[ ]:


"""
name classifier (QA model) by Houg Yun
input: predicted name (list), news(text), dataset num(int)
output: predict name (list)
model has been delete QAQ
"""
from transformers import BertForSequenceClassification
def qa_name_binary_ensemble(pred_name_list, news):
    num_labels = 2
    lm_path = './chinese_roberta_wwm/'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    model_path = '../../nlp/NewsClassify/QAModel/1/'
    rbt0_checkpoint = model_path + 'roberta_init0_all_data_name_qa_split_epoch0.pkl'
    rbt1_checkpoint = model_path + 'roberta_init1_all_data_name_qa_split_epoch2.pkl'
    rbt2_checkpoint = model_path + 'roberta_init2_all_data_name_qa_split_epoch2.pkl'

    model0 = BertForSequenceClassification.from_pretrained(lm_path,num_labels=num_labels)
    model0.load_state_dict(torch.load(rbt0_checkpoint))
    model0.to(device)
    model0.eval()
    
    model1 = BertForSequenceClassification.from_pretrained(lm_path,num_labels=num_labels)
    model1.load_state_dict(torch.load(rbt1_checkpoint))
    model1.to(device)
    model1.eval()
    
    model2 = BertForSequenceClassification.from_pretrained(lm_path,num_labels=num_labels)
    model2.load_state_dict(torch.load(rbt2_checkpoint))
    model2.to(device)
    model2.eval()

    tokenizer = BertTokenizer.from_pretrained(lm_path)

    def clean_string(content):
        content = content.replace('\n','').replace('\t','').replace(' ','').replace('\xa0','')
        content = re.sub("[●▼►★]", "",content)
        return content

    def cut_sent(para):
        para = re.sub('([。！？\?])([^”’])', r"\1\n\2", para)
        para = re.sub('(\.{6})([^”’])', r"\1\n\2", para) 
        para = re.sub('(\…{2})([^”’])', r"\1\n\2", para)  
        para = re.sub('([。！？\?][”’])([^，。！？\?])', r'\1\n\2', para)
        return para.split("\n")
    
    class Testset(Dataset):
        def __init__(self, input_ids, token_type_ids, attention_mask, names):
            self.input_ids = input_ids
            self.token_type_ids = token_type_ids
            self.attention_mask = attention_mask
            self.names = names

        def __getitem__(self, idx):
            inputid = self.input_ids[idx]
            tokentype = self.token_type_ids[idx]
            attentionmask = self.attention_mask[idx]
            name = self.names[idx]
            return inputid, tokentype, attentionmask, name

        def __len__(self):
            return len(self.input_ids)

    def combine_sentence(sentences, max_len):
        li = []
        string = ""
        for k in range(len(sentences)):
            sentence = sentences[k]
            if len(string) + len(sentence) < max_len:
                string = string + sentence
            else:
                #             原本是空的代表sentences太常
                if string == "":
                    n = max_len
                    tmp_li = [sentence[i : i + n] for i in range(0, len(sentence), n)]
                    string = tmp_li.pop(-1)
                    li = li + tmp_li
                else:
                    li.append(string)
                    string = sentence
        if string != "":
            li.append(string)
        return li

    train_input_ids = []
    train_token_types = []
    train_attention_mask = []
    testing_name = []

    content = clean_string(news)

    max_length = 500

    split_content = cut_sent(content)
    chunks = combine_sentence(split_content, max_length)

    for chunk in chunks:
        for name in pred_name_list:
            if len(chunk) >= max_length:
                print("error !!!! lenth > 500")
                continue
            if name not in chunk:
                continue

            input_ids = tokenizer.encode(name, chunk)
            if len(input_ids) > 512:
                continue
            sep_index = input_ids.index(tokenizer.sep_token_id)
            num_seg_a = sep_index + 1
            num_seg_b = len(input_ids) - num_seg_a
            segment_ids = [0] * num_seg_a + [1] * num_seg_b

            input_mask = [1] * len(input_ids)

            while len(input_ids) < 512:
                input_ids.append(0)
                input_mask.append(0)
                segment_ids.append(0)

            train_input_ids.append(input_ids)
            train_token_types.append(segment_ids)
            train_attention_mask.append(input_mask)
            testing_name.append(name)

    train_input_ids = np.array(train_input_ids)
    train_token_types = np.array(train_token_types)
    train_attention_mask = np.array(train_attention_mask)
    testing_name = np.array(testing_name)

    BATCH_SIZE = train_input_ids.shape[0]
    testset = Testset(
        train_input_ids, train_token_types, train_attention_mask, testing_name
    )
    testloader = DataLoader(testset, batch_size=BATCH_SIZE)

    with torch.no_grad():
        for data in testloader:
            tokens_tensors, segments_tensors, masks_tensors = [
                t.to(device) for t in data[:-1]
            ]
            name = data[-1]
            pred_name_list = np.array(name)
            
            outputs0 = model0(
                input_ids=tokens_tensors,
                token_type_ids=segments_tensors,
                attention_mask=masks_tensors,
            )
            pred0 = torch.softmax(outputs0[0], dim=-1)
            pred0 = torch.argmax(pred0, dim=-1)
            pred0 = pred0.cpu().detach().numpy()
            ans0 = list(pred_name_list[pred0 > 0])
            
            outputs1 = model1(
                input_ids=tokens_tensors,
                token_type_ids=segments_tensors,
                attention_mask=masks_tensors,
            )
            pred1 = torch.softmax(outputs1[0], dim=-1)
            pred1 = torch.argmax(pred1, dim=-1)
            pred1 = pred1.cpu().detach().numpy()
            ans1 = list(pred_name_list[pred1 > 0])
            
            
            outputs2 = model2(
                input_ids=tokens_tensors,
                token_type_ids=segments_tensors,
                attention_mask=masks_tensors,
            )
            pred2 = torch.softmax(outputs2[0], dim=-1)
            pred2 = torch.argmax(pred2, dim=-1)
            pred2 = pred2.cpu().detach().numpy()
            ans2 = list(pred_name_list[pred2 > 0])
            
            
            vote_result = []
            for name in list(set(ans0 + ans1 + ans2)):
                vote = 0
                vote += name in ans0
                vote += name in ans1
                vote += name in ans2
                if vote >=2:
                    vote_result.append(name)

            
            return vote_result


# ---
# ## single test

# In[ ]:


t = '台北地檢署偵辦台北市警中山分局中山一派出所集體貪瀆弊案，上月初判中山一派出所前、後3任所長林子芸、楊修白、劉靜怡、林子芸妻子林思賢、行賄「白手套」黃念心5人無罪，台北地檢署不服判決，對5人提起上訴。  檢調偵辦台北市中山區「立邦」酒店媒介外籍女子賣淫案，查出林子芸等官警自2004年起至2017年按月收業者4萬元賄款，以不臨檢、通風報信方式包庇酒店經營；「夜王」酒店從2007年起到2017年每月向派出所員警行賄1萬5000元，將涉案員警、業者多人起訴。  台北地院審理，上月6日將員警李石良判刑16年、判游怡如13年、判紀宏白8年、判張佳雯14年、判紀炳場14年、判陳宏洲10年10月、判莊琦良12年半，另員警曾學函、楊惠志、蔣盈君獲判緩刑，不過，林子芸、楊修白、劉靜怡、林思賢、黃念心5人獲判無罪。  台北地檢署檢察官收判後，認為法官判決林子芸等5人無罪的理由違反經驗法則、論理法則，日前向台灣高等法院提起上訴'
ckip_n = "['莊琦良', '黃念心', '判紀宏白', '張佳雯', '楊修白', '林思賢', '判紀炳場', '林子芸', '曾學函', '劉靜怡', '蔣盈君', '楊惠志', '陳宏洲', '游怡如', '李石良']"
len(t)


# In[ ]:


start = datetime.datetime.now().timestamp()
ans = get_AML_person(model, t, ckip_n, mode=1, binary=1)
end = datetime.datetime.now().timestamp()
print('ans: ', ans)
print('total time: ', end-start)


# ---
# ## multi test

# In[ ]:


df = pd.read_csv('./dataset/2020-07-29.csv')
all_ans = []
for i in range(df.shape[0]):
    t = df.loc[i, 'article']
    ckip_n = df.loc[i, 'ckip_name']
    start = datetime.datetime.now().timestamp()
    b = df.loc[i, 'binary']
    ans = get_AML_person(model, t, ckip_n, mode=1, binary=b)
    end = datetime.datetime.now().timestamp()
#     print('ans: ', ans)
#     print('total time: ', end-start)
    all_ans.append(list(ans))


# In[ ]:


result = []
for i in range(len(all_ans)):
    if (all_ans[i][0] == '' and len(all_ans[i]) == 1 or len(all_ans[i]) == 0):
        result.append([''])
        continue
    result.append(qa_name_binary_ensemble(all_ans[i] , df.loc[i,'article']))
dict = {'BIO_ans' : all_ans, 'QA_ans' : result}
df_ans = pd.DataFrame(dict)
df_ans = pd.concat([df['predict_name'], df_ans], axis = 1)
df_ans.to_csv('0729_pred_n.csv', index=False)

