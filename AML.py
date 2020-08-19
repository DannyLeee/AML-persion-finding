#!/usr/bin/env python
# coding: utf-8

# # AML Finding
# ## Data Preprocessing
# ### basic

# In[1]:


import pandas as pd
import ast
import numpy as np
import re
from zhon.hanzi import stops, non_stops
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
# PRETRAINED_MODEL_NAME = "hfl/rbtl3" # RBTL3
PRETRAINED_MODEL_NAME = "bert_wwm_pretrain_tbrain" # pretrained_bert_wwm
df_train = pd.read_csv('./dataset/tbrain_train_0.csv')
df_2 = pd.read_csv('./dataset/tbrain_test_0.csv')
df_train = pd.concat([df_train, df_2])
df_train = df_train.fillna('[\'\]')
df_train = df_train.reset_index(drop=True)
print(df_train.shape)


# In[2]:


df_train.head()


# In[3]:


def clean_string(content):
    content = content.replace('\n','。').replace('\t','，').replace('!', '！').replace('?', '？')# erease white space cause English name error
    content = re.sub("[+\.\/_,$%●▼►^*(+\"\']+|[+——~@#￥%……&*（）★]", "",content)
    content = re.sub(r"[%s]+" %stops, "。",content)
    return content


# In[4]:


def find_all(name, content):
    # +1 for [CLS]
    pos_list = [m.start()+1 for m in re.finditer(name, content)]
    count = len(pos_list)
    return pos_list , count


# In[5]:


def orgi_2_array(names, contents):
    x = []
    binary_y = []
    BIO_labels = []
    nFound_count = 0
    name_count = 0
    
    for i in range(len(contents)):
        content = contents[i]
        content = clean_string(content)

        # record names
        # name = names[i] # single
        name_list = names[i]
        names_label = ast.literal_eval(name_list) # string to list
        # debug
        

        # init pos label arr
        BIO_label = np.full((512), 2) # initial to all 2 (outside)
        
        # no AML person
        if(name_list == '[]'):
            binary_y.append(0)
            x.append(content)
            BIO_label[0] = 0 # first position 0(begin)
            BIO_labels.append(BIO_label)

        else:
            # initial position list
            start_pos = []
            end_pos = []

            # if (True): # single
            for name in names_label:
              temp, count = find_all(name, content)
              if(temp == []):
  #                 print(name + ' find error in data', i)
                  nFound_count += 1
                  continue
              for j in range(count):
                start_pos.append(temp[j])
                end_pos.append(temp[j] + len(name))

#                  01234
#                B 00100
#                I 00011
#                O 11000
            for j in range(len(start_pos)):
                if(start_pos[j] < 512 and end_pos[j] < 512):
                    BIO_label[start_pos[j]] = 0
                    BIO_label[start_pos[j]+1 : end_pos[j]] = 1
            binary_y.append(1)
            x.append(content)
            BIO_labels.append(BIO_label)
            

    x = np.array(x)
    binary_y = np.array(binary_y)
    BIO_labels = np.array(BIO_labels)
    
    print('nFound: ', nFound_count)
    print('name_count:', name_count)
    print(x.shape)
    print(binary_y.shape)
#     print(begin_pos_labels.shape)
#     print(inside_pos_labels.shape)
#     print(outside_pos_labels.shape)
    print(BIO_labels.shape)
    return x, binary_y, BIO_labels


# ### Get Data List (Train)

# In[6]:


names =  df_train['name']
contents = np.array(df_train['full_content'].tolist())
train_x, train_binary_y, train_bio_labels = orgi_2_array(names, contents)


# In[7]:


print(len(train_x),len(train_binary_y))
print(sum(train_binary_y)/len(train_x))


# In[8]:


c_0, c_1, c_2 = 0, 0, 0
for i in range(len(train_bio_labels)):
    for j in range(512):
        if (train_bio_labels[i][j] == 0):
            c_0 +=1
        elif (train_bio_labels[i][j] == 1):
            c_1 += 1
        else:
            c_2 += 1
print(c_0, c_1, c_2)
print(c_2 / c_0, c_2 / c_1, c_2 / c_2)


# In[9]:


train_bio_labels[0]


# ### Dataset Class

# In[10]:


from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class TrainDataset(Dataset):
    def __init__(self, input_dict, y , bio_labels):
        self.input_ids = input_dict['input_ids']
        self.token_type_ids = input_dict['token_type_ids']
        self.attention_mask = input_dict['attention_mask']
        self.y = y
        self.bio_labels = bio_labels
        
    def __getitem__(self,idx):
        inputid = self.input_ids[idx]
        tokentype = self.token_type_ids[idx]
        attentionmask = self.attention_mask[idx]
        bio_label = self.bio_labels[idx]
        y = self.y[idx]
        return inputid , tokentype , attentionmask, y , bio_label
    
    def __len__(self):
        return len(self.input_ids)
    
class TestDataset(Dataset):
    def __init__(self, input_dict):
        self.input_ids = input_dict['input_ids']
        self.token_type_ids = input_dict['token_type_ids']
        self.attention_mask = input_dict['attention_mask']
        
    def __getitem__(self,idx):
        inputid = self.input_ids[idx]
        tokentype = self.token_type_ids[idx]
        attentionmask = self.attention_mask[idx]
        return inputid , tokentype , attentionmask, 
    
    def __len__(self):
        return len(self.input_ids)


# ### Go Through Tokenizer (Train)

# In[11]:


from transformers import BertTokenizer


tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)
# 把input轉換成bert格式
train_input_dict = tokenizer.batch_encode_plus(train_x, 
                                         add_special_tokens=True,
                                         max_length=512,
                                         return_special_tokens_mask=True,
                                         pad_to_max_length=True,
                                         return_tensors='pt',
                                         truncation=True)


# In[19]:


train_input_dict


# ## Model Budling

# In[10]:


""" model budling """
from transformers import BertModel
import torch
import torch.nn as nn

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
#         position_ids=None,
#         head_mask=None,
#         inputs_embeds=None,
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
#             position_ids=position_ids,
#             head_mask=head_mask,
#             inputs_embeds=inputs_embeds
        )
#         have_AML = outputs[1] # pooled cls (cls token through 1 linear and tanh)
#         have_AML = self.classifier(have_AML)
        
        BIO = self.BIO_classifier(outputs[0]) # 512*HIDDENSIZE word vectors
        BIO = self.softmax(BIO)
        
        outputs = (BIO, ) + outputs[2:]
        return outputs


# ---
# ## Training

# In[11]:


""" model setting (training)"""
from transformers import BertConfig, AdamW
config = BertConfig.from_pretrained(PRETRAINED_MODEL_NAME, output_hidden_states=True)
BATCH_SIZE = 4
trainSet = TrainDataset(train_input_dict, train_binary_y, train_bio_labels)
trainLoader = DataLoader(trainSet, batch_size=BATCH_SIZE)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device:", device)
model = AMLPredictModel(config)
optimizer = AdamW(model.parameters(), lr=1e-5) # AdamW = BertAdam
binary_loss_fct = nn.CrossEntropyLoss()
weight = torch.FloatTensor([359, 510, 1]).cuda()
# 1000 900 1 ()被蓋掉
# 359 510 1
# 500 450 1
# 250 150 1 (X)
# 125 50 1 (X)
BIO_loss_fct = nn.CrossEntropyLoss(weight=weight)

# high-level 顯示此模型裡的 modules
print("""
name            module
----------------------""")
for name, module in model.named_children():
    if name == "bert":
        for n, _ in module.named_children():
            print(f"{name}:{n}")
#             print(_)
    else:
        print("{:15} {}".format(name, module))


# In[12]:


""" training """
from datetime import datetime,timezone,timedelta

model = model.to(device)
model.train() ##########################

EPOCHS = 10
dt1 = datetime.utcnow().replace(tzinfo=timezone.utc)
dt2 = dt1.astimezone(timezone(timedelta(hours=8))) # 轉換時區 -> 東八區
print(dt2)
for epoch in range(EPOCHS):
    running_loss = 0.0
    binary_running_loss = 0.0
    BIO_running_loss = 0.0
    for data in trainLoader:
    # data = testSet[21] # test model
    # if(True):
        
      tokens_tensors, segments_tensors, masks_tensors,       labels, BIO_label = [t.to(device) for t in data]

      # 將參數梯度歸零
      optimizer.zero_grad()
      
      # forward pass
      outputs = model(input_ids=tokens_tensors, 
                      token_type_ids=segments_tensors, 
                      attention_mask=masks_tensors)

      BIO_pred = outputs[0]
      BIO_pred = torch.transpose(BIO_pred, 1, 2)

      # print(BIO_pred.shape)
      # print(BIO_label.shape)
      BIO_loss = BIO_loss_fct(BIO_pred, BIO_label)
      # print(binary_loss, BIO_loss)
      loss = BIO_loss
      # print(loss)
      # break
      
      # backward
      loss.backward()
      optimizer.step()

      # 紀錄當前 batch loss
      running_loss += loss.item()
      BIO_running_loss += BIO_loss.item()
        
    CHECKPOINT_NAME = './model_final_state_dict/pre_bert_wwm_bio_only_EPOCHES_' + str(epoch) + '(w359).pkl' 
    torch.save(model.state_dict(), CHECKPOINT_NAME)
        
    # 計算分類準確率
    # _, binary_acc, bio_acc = get_predictions(model, trainLoader, compute_acc=True)
    dt1 = datetime.utcnow().replace(tzinfo=timezone.utc)
    dt2 = dt1.astimezone(timezone(timedelta(hours=8))) # 轉換時區 -> 東八區
    print('%s\t[epoch %d] loss: %.3f' %
          (dt2, epoch + 1, running_loss))


# ---
# ## Testing

# In[13]:


import pandas as pd
import ast
import numpy as np
import re
from zhon.hanzi import stops, non_stops
# df_test = pd.read_csv('./dataset/multi_tbrain_test.csv')
df_test = pd.read_csv('./dataset/tbrain_test_0.csv')

df_test.shape


# ### Get Data List (Test)

# In[15]:


names =  df_test['name']
contents = np.array(df_test['full_content'].tolist())
test_x, test_binary_y, test_bio_labels = orgi_2_array(names, contents)
test_binary_y.sum()


# In[16]:


from transformers import BertTokenizer

# PRETRAINED_MODEL_NAME = "hfl/rbtl3" # RBTL3
PRETRAINED_MODEL_NAME = "bert_wwm_pretrain_tbrain" # pretrained_bert_wwm
# MODEL_PATH = './model_3/RBTL3_bio_only_EPOCHES_9.pkl'
MODEL_PATH = './model/pre_bert_wwm_bio_only_EPOCHES_9(w359).pkl'
# MODEL_PATH = './model/pre_bert_wwm_bio_only_EPOCHES_9.pkl'


tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)


# ### Go Through Tokenizer (Test)

# In[17]:


test_input_dict = tokenizer.batch_encode_plus(test_x, 
                                         add_special_tokens=True,
                                         max_length=512,
                                         return_special_tokens_mask=True,
                                         pad_to_max_length=True,
                                         return_tensors='pt',
                                         truncation=True)


# ### Prediction Decode Function

# In[18]:


def bio_2_string(tokens_tensors, have_AML, BIO_tagging, ckip_result, BIO_prob):
  result = []
  if (have_AML.item() == 0):
    result.append('')
  else:
    for j in range(1, 512):
      if (BIO_tagging[j] == 0):
        start = j
        end = j + 1
        while (end < 512 and BIO_tagging[end] == 1):
          end += 1
        if (end > start + 1):
          if (start <= 3):
            s = tokenizer.decode(token_ids = tokens_tensors[start : end +2], skip_special_tokens = True)
          else:
            s = tokenizer.decode(token_ids = tokens_tensors[start-1 : end +2], skip_special_tokens = True)
          s = s.replace(' ', '')
          # print(s)
          for k in range(len(ckip_result)):
            if (len(ckip_result[k]) < 2):
              continue
            elif (re.findall(r"[%s]+" %non_stops, ckip_result[k]) != []                      or re.findall(r"[%s]+" %stops, ckip_result[k]) != []): # 有標點
              continue
            found = s.find(ckip_result[k])
            if (found != -1):
              if (found == 0):
                # print(s)
                prob = BIO_prob[start][0] # begin
                for i in range(1, len(ckip_result[k]) + 1):
                  p = BIO_prob[start+i][1]  # inside
                  # print(p)
                  prob *= p
                # print('! len: ', len(ckip_result[k]), '\tprobability: ', prob.item())
              else:
                # print(s)
                prob = BIO_prob[start][1] # inside
                for i in range(1, len(ckip_result[k]) + 1):
                  p = BIO_prob[start+i][1]  # inside
                  # print(p)
                  prob *= p
                # print('_ len: ', len(ckip_result[k]), '\tprobability: ', prob.item())
#               if (prob.item() >= 0.95):
              if (True):
                result.append(ckip_result[k])
    if (len(result) == 0):
      result.append('')
    # print('---')
  return result


# In[19]:


def get_predictions(model, testLoader, BATCH_SIZE):
  result = []
  total_count = 0 # 第n筆data
  with torch.no_grad():
    for data in testLoader:
      # 將所有 tensors 移到 GPU 上
      if next(model.parameters()).is_cuda:
        data = [t.to("cuda:0") for t in data if t is not None]
      
      # 別忘記前 3 個 tensors 分別為 tokens, segments 以及 masks
      # 且強烈建議在將這些 tensors 丟入 `model` 時指定對應的參數名稱
      tokens_tensors, segments_tensors, masks_tensors = data[:3]
      outputs = model(input_ids=tokens_tensors, 
                  token_type_ids=segments_tensors, 
                  attention_mask=masks_tensors)
      
      count = min(outputs[0].shape[0], BATCH_SIZE)
      for i in range(count):  # run batchsize times
#         have_AML = outputs[0][i].argmax()
        BIO_pred = outputs[0][i].argmax(1) # 3*512 into class label
        text_token = tokens_tensors[i]
        ckip_names = df_test.loc[total_count, 'ckip_names']
        ckip_names_list = ast.literal_eval(ckip_names) # string to list
        r = bio_2_string(text_token, test_binary_y[total_count], BIO_pred, ckip_names_list, outputs[0][i])
#         print(BIO_pred)
        result.append(r)
        total_count += 1
#       break
    # print(result)
  return result


# In[20]:


"""testing"""
import torch
from transformers import BertConfig
config = BertConfig.from_pretrained(PRETRAINED_MODEL_NAME, output_hidden_states=True)
# model = AMLPredictModel(config)
# model.load_state_dict(torch.load(MODEL_PATH))
# model = torch.load(MODEL_PATH)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()

BATCH_SIZE = 50
testSet = TestDataset(test_input_dict)
testLoader = DataLoader(testSet, batch_size=BATCH_SIZE)


predictions = get_predictions(model, testLoader, BATCH_SIZE)

# pred = predictions.cpu().data.numpy()
# pred = np.argmax(pred, axis=1)
# accuracy = (pred == test_binary_y).mean()
# print('Your test accuracy is %.6f' % (accuracy * 100))


# ### Get answer and Evaluation

# In[14]:


temp = df_test['name'].tolist()
ans = []
for i in range(len(temp)):
  t = ast.literal_eval(temp[i])
  if (len(t) == 0):
    t.append('')
  ans.append(t)
# ans


# In[21]:


def eval(pred, ans):
    if bool(pred) is not bool(ans):
        return 0
    elif not pred and not ans:
        return 1
    else:
        pred = set(pred)
        ans = set(ans)
        interaction_len = len(pred & ans)
        if interaction_len == 0:
            return 0

        pred_len = len(pred)
        ans_len = len(ans)
        return 2 / (pred_len / interaction_len + ans_len / interaction_len)


def eval_all(pred_list, ans_list):
    assert len(pred_list) == len(ans_list)
    return sum(eval(p, a) for p, a in zip(pred_list, ans_list)) / len(pred_list)


# ### name classfier by Mouth Han

# In[23]:


# df_ckip = pd.read_csv('./ckip/ckip_dataset3.csv')
df_ckip = pd.read_csv('./ckip/ckip.csv')
ckip_name = df_ckip.loc[df_ckip['ans'] == 1, 'name'].tolist()
# ckip_name


# In[24]:


result = []
ckip_name = set(ckip_name)
for i in range(len(predictions)):
  temp = set(predictions[i])
  r = list(ckip_name & temp)
  if (len(r) == 0):
    r.append('')
  result.append(r)
# result


# In[25]:


eval_all(result, ans)


# ### name classifier (QA model) by Houg Yun

# In[15]:


"""
name classifier (QA model) by Houg Yun
input: predicted name (list), news(text), dataset num(int)
output: predict name (list)
model has been delete QAQ
"""
def check_pred_name_is_real_ans(pred_name_list,news,dataset):
    class Testset(Dataset):
        def __init__(self, input_ids , token_type_ids , attention_mask):
            self.input_ids = input_ids
            self.token_type_ids = token_type_ids
            self.attention_mask = attention_mask
        def __getitem__(self,idx):
            inputid = self.input_ids[idx]
            tokentype = self.token_type_ids[idx]
            attentionmask = self.attention_mask[idx]

            return inputid , tokentype , attentionmask

        def __len__(self):
            return len(self.input_ids)
    
    lm_path = './bert_wwm_pretrain_tbrain/'
    tokenizer = BertTokenizer.from_pretrained(lm_path)

    content = clean_string(news)
    train_input_ids = []
    train_token_types = []
    train_attention_mask = []
        
    for name in pred_name_list:
        
        content_max_length = 512-3-len(name)
        
        if len(content) >= content_max_length:
            content = content[:content_max_length]
            
        input_ids = tokenizer.encode(name, content)
        if(len(input_ids)>512):
            continue
        sep_index = input_ids.index(tokenizer.sep_token_id)
        num_seg_a = sep_index + 1
        num_seg_b = len(input_ids) - num_seg_a
        segment_ids = [0]*num_seg_a + [1]*num_seg_b

        input_mask = [1] * len(input_ids)

        while len(input_ids) < 512:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
            
        train_input_ids.append(input_ids)
        train_token_types.append(segment_ids)
        train_attention_mask.append(input_mask)
        
    train_input_ids = np.array(train_input_ids)
    train_token_types  = np.array(train_token_types)
    train_attention_mask = np.array(train_attention_mask)
    
    
    BATCH_SIZE = train_input_ids.shape[0]
    
    testset = Testset(train_input_ids ,train_token_types , train_attention_mask)
    testloader = DataLoader(testset, batch_size=BATCH_SIZE)
    
    
    from transformers import BertForSequenceClassification
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    lm_path = './bert_wwm_pretrain_tbrain/'
    NUM_LABELS = 2
    tokenizer = BertTokenizer.from_pretrained(lm_path)
    model = BertForSequenceClassification.from_pretrained(lm_path,num_labels=NUM_LABELS)
    
    check_point = ''
    if(dataset == 0):
#         最一開始的dataset
        check_point = '../../nlp/NewsClassify/TB_multispan/Bert_wwm_ckip_name_is_ans_13.pkl'

    elif (dataset == 1):
#         tbrain_train (1).csv
        check_point = '../../nlp/NewsClassify/TB_multispan/Bert_wwm_ckip_name_is_ans_dataset1_epoch17.pkl'

    elif (dataset == 2):
#         tbrain_train (2).csv
        check_point = '../../nlp/NewsClassify/TB_multispan/Bert_wwm_ckip_name_is_ans_dataset2_epoch13.pkl'

    elif (dataset == 3):
#         tbrain_train (3).csv
        check_point = '../../nlp/NewsClassify/TB_multispan/Bert_wwm_ckip_name_is_ans_dataset3_epoch18.pkl'

    elif (dataset == 4):
#        traindata + testdata
        check_point = '../../nlp/NewsClassify/TB_multispan/Bert_wwm_ckip_name_is_ans_alldataset_epoch18.pkl'

    
    
    
#     check_point = '../../nlp/NewsClassify/TB_multispan/Bert_wwm_ckip_name_is_ans_13.pkl'
    model.load_state_dict(torch.load(check_point))
    model = model.to(device)
    model.eval()


    with torch.no_grad():
        for data in testloader:
            tokens_tensors, segments_tensors, masks_tensors = [t.to(device) for t in data]
            outputs = model(input_ids=tokens_tensors, 
                                token_type_ids=segments_tensors, 
                                attention_mask=masks_tensors)
            pred = torch.softmax(outputs[0] , dim = -1)
            torch.set_printoptions(precision=10)
            print(pred)
            pred = torch.argmax(pred,dim=-1)
            pred = pred.cpu().detach().numpy()
            pred_name_list = np.array(pred_name_list)
            return list(pred_name_list[pred>0])


# In[ ]:


result = []
for i in range(len(ans)):
    if (predictions[i][0] == ''):
        result.append([''])
        continue
    result.append(check_pred_name_is_real_ans(predictions[i] , df_test.loc[i,'full_content'], 1))
result


# In[43]:


eval_all(result, ans)


# ---
# ## Only name testing

# In[26]:


only_name_ans = []
only_name_pred = []
for i in range(len(ans)):
    if (ans[i][0] != ''):
        only_name_ans.append(ans[i])
        only_name_pred.append(result[i])
print(eval_all(only_name_pred, only_name_ans))


# ---
# ## Ensemble

# In[77]:


pre_bert_wwm_result1 = [[''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], ['賴俊吉'], [''], [''], ['張銘坤', '陳揚宗'], [''], [''], [''], [''], [''], [''], ['王益洲'], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], ['李士綸', '吳哲瑋'], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], ['林志聰', '伍政山'], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], ['張承平'], [''], [''], [''], [''], [''], [''], ['雷俊玲'], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], ['陳仕修', '黃國昌', '徐仲榮'], [''], ['林睿耆', '周漢祥', '林昱伯', '詹騏瑋', '林煒智'], [''], [''], ['秦儷舫', '童仲彥', '黃國昌'], [''], [''], [''], [''], ['蘇怡寧', '禾馨'], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], ['詹舜淇', '方俐婷', '詹逸宏', '詹雅琳', '詹雯婷'], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], ['黃顯雄', '黃世陽'], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], ['葉大慧', '吳國昌', '吳孝昌', '魏君婷', '張欽堯'], [''], [''], [''], [''], [''], [''], [''], [''], [''], ['李宛霞', '黃梅雪'], ['陳有意', '陳運作', '陳坦承', '陳武騰', '黃振榮'], ['許長裕'], [''], [''], ['蔡開宇', '李訓成', '王宇正'], [''], [''], [''], ['蔡思庭', '楊正平'], [''], [''], [''], [''], [''], [''], [''], ['林勇任', '郭雅雯', '蘇震清', '葉美麗', '茂宇', '賴麗團'], [''], ['王宇承', '陳瑞芳', '振瑞', '陳澤信'], [''], [''], ['李育英', '林煜傑', '李文潔', '劉矢口'], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], ['崔明禮', '楊敬熙'], [''], [''], ['黃聲儀', '陳功源', '黃泳學', '羅栩亮', '黃馨儀', '高兆良'], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], ['洪正義'], [''], [''], [''], [''], [''], [''], ['鍾增林', '曾國財'], [''], ['王毓雅', '羅瑞榮'], [''], [''], [''], [''], [''], ['蔡文娟'], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], ['柯賜海'], [''], [''], [''], [''], ['黃文鴻', '吳承霖', '陳玟叡', '蔡英俊'], [''], [''], [''], [''], [''], ['賴素如'], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], ['謝宥宏'], [''], [''], [''], [''], [''], [''], [''], [''], [''], ['郭政權', '楊天生', '蔡茂寅', '郭說明'], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], ['吳承霖', '陳玟叡'], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], ['何壽川'], [''], [''], [''], [''], [''], [''], [''], [''], ['李錫璋', '陳清江'], [''], [''], [''], [''], [''], [''], [''], [''], ['']]

pre_bert_wwm_result2 = [[''], [''], [''], [''], [''], [''], [''], [''], [''], [''], ['宋芷妍', '王安石'], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], ['繆竹怡'], [''], [''], ['陳建飛'], [''], [''], ['許玉秀', '吳淑珍', '王隆昌'], [''], [''], ['劉威甫'], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], ['邱嘉進'], [''], [''], [''], ['陳宣銘'], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], ['許家榮', '林詩芳', '王紀棠', '殷倜凡', '謝巧莉'], [''], [''], ['姚慶佳', '姚慶佳男'], ['吳運豐', '雲從龍'], ['李春生', '周正華', '黃建強', '高振殷', '鄭銘富', '呂家緯'], [''], [''], [''], [''], [''], [''], ['鄧超鴻', '道克明'], [''], [''], [''], [''], [''], [''], [''], [''], ['崔明禮'], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], ['蔡賜爵', '劉昌松', '畢鈞輝'], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], ['陳韋霖'], [''], [''], [''], [''], ['柯志龍', '俞小凡', '翁家明', '瓊瑤', '張興蕙', '張哲維', '夏婉君'], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], ['蒲念慈'], [''], ['詹舜淇', '詹逸宏', '詹雅琳'], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], ['侯君鍵'], [''], [''], [''], [''], ['李孟謙', '于曉艷'], [''], [''], [''], [''], [''], [''], ['楚瑞芳', '王光遠', '錢利忠'], [''], [''], [''], [''], [''], ['鍾增林', '曾國財'], [''], [''], [''], [''], [''], ['李榮華'], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], ['楚瑞芳', '王光遠', '鍾榮昌', '彭振源'], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], ['賴嚮景', '陳俊宏'], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], ['李佰全', '葉玲', '林陀桂英'], [''], [''], [''], [''], ['林政賢'], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], ['楊治渝', '李宗瑞'], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], ['吳東明', '劉吉雄', '林輝宏', '張建華', '裴振福', '呂宗南'], [''], [''], ['葉冠廷', '康明璋'], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], ['何培才'], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], ['林金龍', '黃鈺蘋', '呂翠峰', '黃子愛', '顏雪藝'], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], ['陳建三', '羅秋英', '陳斯婷', '陳斯婷批戰袍'], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], ['王益洲'], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], ['黃振龍', '蔡文旭', '張治忠', '張道銘'], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], ['']]

pre_bert_wwm_result3 = [[''], [''], [''], [''], [''], [''], ['張君豪', '李孟謙', '于曉艷'], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], ['崔培明'], [''], [''], [''], [''], [''], [''], [''], [''], ['蔡賜爵', '劉昌松', '畢鈞輝'], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], ['陳韋霖'], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], ['林欣月'], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], ['黃薪哲', '吳寶玉', '余信憲'], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], ['袁昶平', '洪勝明'], [''], [''], ['葉麗珍', '祥禾', '趙鈞震', '葉麗貞', '陳耀東'], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], ['林愈得', '陳清裕', '曾盛陽', '曾盛麟', '曾美菁'], [''], [''], [''], [''], [''], [''], [''], ['林勇任', '郭雅雯', '葉美麗', '茂宇', '賴麗團'], [''], [''], [''], ['蔡維峻', '林銘宏'], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], ['楊嘉仁', '林崇傑'], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], ['楊昇穎', '林嘉東'], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], ['卓國華'], [''], [''], [''], ['戴盛世', '宣昶孔'], [''], ['許祈文'], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], ['李維凱'], [''], [''], [''], [''], ['吳宗憲', '林清井', '湯蕙禎', '劉奕發', '歐炳辰'], ['陳發貴'], [''], [''], [''], [''], [''], [''], ['陳俊佑', '陳致銘', '王延順'], [''], [''], [''], ['孔朝'], [''], ['詹昭書'], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], ['朱啟仁'], ['戴盛世'], [''], [''], [''], ['徐詩彥'], [''], [''], [''], [''], ['陳麗珍'], [''], [''], [''], [''], ['何培才'], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], ['許玉秀', '吳淑珍', '王隆昌'], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], ['蔡思庭', '楊正平'], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], ['游美雲'], [''], [''], [''], [''], ['洪丞俊', '謝介裕', '黃丹怡'], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], ['范筱梵夫', '江智銓', '江智詮', '王柏森', '范筱梵'], [''], [''], [''], [''], [''], [''], ['吳銀嵐', '吳金虎', '張安樂', '徐宏杰', '許國楨'], [''], [''], [''], ['蒲念慈'], ['陳之漢', '紀雅玲', '林睿君'], [''], ['李榮勝', '黃錦燕'], [''], [''], ['']]


# In[78]:


RBTL3_result1 = [[''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], ['賴俊吉'], [''], [''], ['張銘坤', '陳揚宗'], [''], [''], [''], [''], [''], [''], ['王益洲'], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], ['李士綸', '吳哲瑋'], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], ['林志聰', '伍政山'], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], ['張承平'], [''], [''], [''], [''], [''], [''], ['雷俊玲'], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], ['蔡英文', '陳仕修', '黃國昌', '徐仲榮'], [''], ['林睿耆', '周漢祥', '林昱伯', '詹騏瑋', '林煒智'], [''], [''], ['秦儷舫', '童仲彥', '黃國昌'], [''], [''], [''], [''], ['禾馨'], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], ['詹舜淇', '方俐婷', '詹逸宏', '詹雅琳', '詹雯婷'], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], ['黃顯雄', '黃世陽'], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], ['葉大慧', '吳國昌', '吳孝昌', '魏君婷', '張欽堯'], [''], [''], [''], [''], [''], [''], [''], [''], [''], ['李宛霞', '黃梅雪'], ['陳運作', '陳坦承', '黃振榮', '陳武騰', '黃不滿'], ['許長裕'], [''], [''], ['蔡開宇', '李訓成', '王宇正'], [''], [''], [''], ['蔡思庭', '楊正平'], [''], [''], [''], [''], [''], [''], [''], ['林勇任', '郭雅雯', '蘇震清', '葉美麗', '茂宇', '賴麗團'], [''], ['王俊忠', '王宇承', '振瑞', '陳澤信'], [''], [''], ['李育英', '林煜傑', '李文潔', '劉矢口'], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], ['崔明禮', '楊敬熙'], [''], [''], ['黃聲儀', '陳功源', '黃泳學', '羅栩亮', '黃馨儀'], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], ['洪正義'], [''], [''], [''], [''], [''], [''], ['鍾增林', '曾國財'], [''], ['王毓雅', '羅瑞榮'], [''], [''], [''], [''], [''], ['蔡文娟'], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], ['柯賜海'], [''], [''], [''], [''], ['黃文鴻', '吳承霖', '丁偉杰', '陳玟叡', '蔡英俊'], [''], [''], [''], [''], [''], ['賴素如'], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], ['謝宥宏'], [''], [''], [''], [''], [''], [''], [''], [''], [''], ['郭政權', '楊天生', '蔡茂寅', '郭說明'], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], ['吳承霖', '陳玟叡'], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], ['何壽川'], [''], [''], [''], [''], [''], [''], [''], [''], ['李錫璋', '陳清江'], [''], [''], [''], [''], [''], [''], [''], [''], ['']]

RBTL3_result2 = [[''], [''], [''], [''], [''], [''], [''], [''], [''], [''], ['王安石'], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], ['繆竹怡'], [''], [''], ['陳建飛'], [''], [''], ['許玉秀', '吳淑珍', '王隆昌'], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], ['邱嘉進', '游芳男'], [''], [''], [''], ['陳宣銘'], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], ['許家榮', '林詩芳', '林維俊', '王紀棠', '殷倜凡', '謝巧莉'], [''], [''], ['姚慶佳', '姚慶佳男'], ['吳運豐', '雲從龍'], ['李春生', '周正華', '黃建強', '高振殷', '鄭銘富', '呂家緯'], [''], [''], [''], [''], [''], [''], ['鄧超鴻', '道克明'], [''], [''], [''], [''], [''], [''], [''], [''], ['崔明禮'], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], ['蔡賜爵', '劉昌松', '畢鈞輝'], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], ['陳韋霖'], [''], [''], [''], [''], ['柯志龍', '俞小凡', '瓊瑤', '林俊峰', '張興蕙', '張哲維', '夏婉君'], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], ['蒲念慈'], [''], ['詹雯婷', '方俐婷', '詹逸宏', '詹雅琳'], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], ['侯君鍵'], [''], [''], [''], [''], ['于曉艷'], [''], [''], [''], [''], [''], [''], ['楚瑞芳', '王光遠'], [''], [''], [''], [''], [''], ['鍾增林', '曾國財'], [''], [''], [''], [''], [''], ['李榮華', '鄭徒刑'], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], ['楚瑞芳', '彭振源'], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], ['賴嚮景', '陳俊宏'], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], ['李佰全', '葉玲', '林陀桂英'], [''], [''], [''], [''], ['林政賢'], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], ['楊治渝', '李宗瑞'], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], ['吳東明', '劉吉雄', '林輝宏', '張建華', '裴振福'], [''], [''], ['陳菊', '葉冠廷', '康明璋'], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], ['何培才'], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], ['顏則獲', '林金龍', '黃鈺蘋', '黃子愛', '顏雪藝'], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], ['陳建三', '羅秋英'], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], ['王益洲'], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], ['蔡文旭', '張道銘', '呂雅純', '張治忠', '蔡英俊', '黃振龍', '李欣潔'], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], ['']]

RBTL3_result3 = [[''], [''], [''], [''], [''], [''], ['李孟謙', '于曉艷'], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], ['崔培明'], [''], [''], [''], [''], [''], [''], [''], [''], ['蔡賜爵', '劉昌松', '畢鈞輝'], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], ['陳韋霖'], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], ['林欣月'], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], ['黃薪哲', '吳寶玉', '余信憲'], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], ['袁昶平', '洪勝明', '洪介紹'], [''], [''], ['祥禾', '葉麗珍', '葉麗貞', '趙鈞震'], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], ['陳清裕', '林愈得', '曾美菁', '曾盛麟'], [''], [''], [''], [''], [''], [''], [''], ['林勇任', '郭雅雯', '蘇震清', '葉美麗', '茂宇', '賴麗團'], [''], [''], [''], ['蔡維峻', '林銘宏'], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], ['林崇傑'], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], ['周宗賢'], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], ['許祈文'], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], ['李發布', '李維凱'], [''], [''], [''], [''], ['劉奕發', '吳宗憲', '歐炳辰', '林清井'], ['陳發貴'], [''], [''], [''], [''], [''], [''], ['陳俊佑', '陳致銘', '王延順'], [''], [''], [''], ['孔朝'], [''], ['詹昭書', '洪美秀'], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], ['朱啟仁', '楊善淵'], ['楊富巖', '戴盛世', '錢利忠'], [''], [''], [''], ['徐詩彥', '林繼蘇'], [''], [''], [''], [''], ['陳麗珍'], [''], [''], [''], [''], ['何培才'], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], ['許玉秀', '吳淑珍', '王隆昌'], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], ['蔡思庭', '楊正平'], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], ['游美雲'], [''], [''], [''], [''], ['洪丞俊', '謝介裕', '黃丹怡'], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], ['王柏森', '江智銓', '范筱梵', '江智詮'], [''], [''], [''], [''], [''], [''], ['徐宏杰', '許國楨', '吳金虎', '張安樂'], [''], [''], [''], [''], ['林睿君'], [''], ['李榮勝', '黃錦燕'], [''], [''], ['']]


# In[104]:


union_result = []
intersect_result = []
for i in range(len(ans)):
    temp1 = set(pre_bert_wwm_result3[i])
    temp2 = set(RBTL3_result3[i])
    union = list(temp1 | temp2)
    intersect = list(temp1 & temp2)
    if (len(union) == 0):
        union.append('')
    if (len(intersect) == 0):
        intersect.append('')

    union_result.append(union)
    intersect_result.append(intersect)
print(eval_all(union_result,ans))
print(eval_all(intersect_result,ans))

