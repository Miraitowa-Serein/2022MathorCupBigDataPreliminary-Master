#!/usr/bin/env python
# coding: utf-8

# # 2022 MathorCup 大数据 IssueB

# # 语音业务数据分析

# ## 初步导入相关第三方库

# In[1]:


import pandas as pd
import numpy as np
import sklearn.preprocessing as sp
import warnings

warnings.filterwarnings("ignore")

# ## 读取附件1与附件3

# In[2]:


dataOne = pd.read_excel("附件1语音业务用户满意度数据.xlsx", sheet_name='Sheet1')
dataThree = pd.read_excel("附件3语音业务用户满意度预测数据.xlsx", sheet_name='语音')

# In[3]:


dataOne

# In[4]:


dataThree

# ## 处理附件1与附件3

# ### 查看附件1与附件3表头

# In[5]:


dataOneColumnsList = list(dataOne.columns)
dataThreeColumnsList = list(dataThree.columns)

# In[6]:


dataOneColumnsList

# In[7]:


dataThreeColumnsList

# In[8]:


set(dataOneColumnsList) & set(dataThreeColumnsList)

# In[9]:


set(dataOneColumnsList) - set(dataThreeColumnsList)

# ### 对附件1增加一项指标[是否投诉]，来源于[家宽投诉 与 资费投诉]，并删除[家宽投诉 与 资费投诉]

# In[10]:


dataOne['资费投诉'] = dataOne.loc[:, ['家宽投诉', '资费投诉']].apply(lambda x1: x1.sum(), axis=1)
dataOne.drop(['家宽投诉'], axis=1, inplace=True)
dataOne.rename(columns={'资费投诉': '是否投诉'}, inplace=True)
dataOne

# In[11]:


dataOneColumnsList = list(dataOne.columns)
dataOneColumnsList

# In[12]:


dataThreeColumnsList = list(dataThree.columns)
dataThreeColumnsList

# In[13]:


set(dataOneColumnsList) - set(dataThreeColumnsList)

# ### 剔除附件1中在附件3中没有的列指标，以及剔除四项不重要列

# In[14]:


dataOne.drop(['用户id',
              '用户描述',
              '用户描述.1',
              '重定向次数',
              '重定向驻留时长',
              '语音方式',
              '是否去过营业厅',
              'ARPU（家庭宽带）',
              '是否实名登记用户',
              '当月欠费金额',
              '前第3个月欠费金额',
              '终端品牌类型'], axis=1, inplace=True)
dataOne

# ### 填补空缺值、数据利于理解化、清洗处理

# In[15]:


dataOne.info()

# In[16]:


dataOne.isnull().sum()

# In[17]:


dataOne['外省流量占比'] = dataOne['外省流量占比'].fillna(0)
dataOne["是否关怀用户"] = dataOne["是否关怀用户"].fillna(0)
dataOne["外省流量占比"] = dataOne["外省流量占比"].astype(str).replace('%', '')
dataOne["外省语音占比"] = dataOne["外省语音占比"].astype(str).replace('%', '')
dataOne

# In[18]:


dataOne.replace({"是否遇到过网络问题": {2: 0},
                 "居民小区": {-1: 0},
                 "办公室": {-1: 0, 2: 1},
                 "高校": {-1: 0, 3: 1},
                 "商业街": {-1: 0, 4: 1},
                 "地铁": {-1: 0, 5: 1},
                 "农村": {-1: 0, 6: 1},
                 "高铁": {-1: 0, 7: 1},
                 "其他，请注明": {-1: 0, 98: 1},
                 "手机没有信号": {-1: 0},
                 "有信号无法拨通": {-1: 0, 2: 1},
                 "通话过程中突然中断": {-1: 0, 3: 1},
                 "通话中有杂音、听不清、断断续续": {-1: 0, 4: 1},
                 "串线": {-1: 0, 5: 1},
                 "通话过程中一方听不见": {-1: 0, 6: 1},
                 "其他，请注明.1": {-1: 0, 98: 1},
                 "是否关怀用户": {'是': 1},
                 "是否4G网络客户（本地剔除物联网）": {'是': 1, "否": 0},
                 "是否5G网络客户": {'是': 1, "否": 0},
                 "客户星级标识": {'未评级': 0, '准星': 1, '一星': 2, '二星': 3, '三星': 4, '银卡': 5, '金卡': 6,
                                  '白金卡': 7, '钻石卡': 8}
                 }, inplace=True)
dataOne

# In[19]:


dataOne.isnull().sum()

# ### 空缺值可视化

# In[20]:


import missingno
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
missingno.bar(dataOne, color='blue')
plt.tight_layout()

# In[21]:


missingno.matrix(dataOne, color=(190 / 255, 190 / 255, 190 / 255))
plt.savefig('figuresOne\\[附件1]附件1空缺值可视化.pdf', bbox_inches='tight')

# ### 空缺值处理

# In[22]:


dataOneMiss = dataOne.isnull()
dataOne[dataOneMiss.any(axis=1) == True]

# In[23]:


dataOne.dropna(inplace=True)
dataOne = dataOne.reset_index(drop=True)
dataOne

# In[24]:


dataOne.dtypes

# ### 格式转化

# In[25]:


dataOne['外省语音占比'] = dataOne['外省语音占比'].astype('float64')
dataOne['外省流量占比'] = dataOne['外省流量占比'].astype('float64')
dataOne['是否4G网络客户（本地剔除物联网）'] = dataOne['是否4G网络客户（本地剔除物联网）'].astype('int64')
dataOne['4\\5G用户'] = dataOne['4\\5G用户'].astype(str)
dataOne['终端品牌'] = dataOne['终端品牌'].astype(str)
dataOne

# ### 标签编码，包括四项评分，视为分类问题

# In[26]:


le = sp.LabelEncoder()

OverallSatisfactionVoiceCalls = le.fit_transform(dataOne["语音通话整体满意度"])
NetworkCoverageSignalStrength = le.fit_transform(dataOne["网络覆盖与信号强度"])
VoiceCallDefinition = le.fit_transform(dataOne["语音通话清晰度"])
VoiceCallStability = le.fit_transform(dataOne["语音通话稳定性"])

FourFiveUser = le.fit_transform(dataOne["4\\5G用户"])
TerminalBrand = le.fit_transform(dataOne["终端品牌"])

dataOne["语音通话整体满意度"] = pd.DataFrame(OverallSatisfactionVoiceCalls)
dataOne["网络覆盖与信号强度"] = pd.DataFrame(NetworkCoverageSignalStrength)
dataOne["语音通话清晰度"] = pd.DataFrame(VoiceCallDefinition)
dataOne["语音通话稳定性"] = pd.DataFrame(VoiceCallStability)

dataOne["4\\5G用户"] = pd.DataFrame(FourFiveUser)
dataOne["终端品牌"] = pd.DataFrame(TerminalBrand)
dataOne


# ### 处理"是否投诉"指标

# In[27]:


def complain(x):
    if x != 0:
        return 1
    else:
        return 0


for i in range(len(dataOne)):
    dataOne.loc[i, '是否投诉'] = complain(dataOne.loc[i, '是否投诉'])

dataOne

# In[28]:


dataOne.dtypes

# ### 格式转化

# In[29]:


dataOne['是否5G网络客户'] = dataOne['是否5G网络客户'].astype('int64')
dataOne['客户星级标识'] = dataOne['客户星级标识'].astype('int64')
dataOne

# In[30]:


dataOne.describe()

# ### 数据可视化

# In[31]:


import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

box_data = dataOne[['语音通话整体满意度',
                    '网络覆盖与信号强度',
                    '语音通话清晰度',
                    '语音通话稳定性', ]]
plt.grid(True)
plt.boxplot(box_data,
            notch=True,
            sym="b+",
            vert=False,
            showmeans=True,
            labels=['语音通话整体满意度',
                    '网络覆盖与信号强度',
                    '语音通话清晰度',
                    '语音通话稳定性', ])
plt.yticks(size=14)
plt.xticks(size=14, font='Times New Roman')
plt.tight_layout()
plt.savefig('figuresOne\\[附件1][语音通话整体满意度、网络覆盖与信号强度、语音通话清晰度、语音通话稳定性]评分箱线图.pdf')

# In[32]:


import seaborn as sns

plt.style.use('ggplot')
sns.set_style('whitegrid')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
CorrDataOneAll = dataOne.corr().abs()
N = 14
ColDataOneRange = CorrDataOneAll.nlargest(N, '语音通话整体满意度')['语音通话整体满意度'].index
plt.subplots(figsize=(N, N))
plt.title('皮尔逊相关系数', size=16)
sns.heatmap(dataOne[ColDataOneRange].corr(),
            linewidths=0.1,
            vmax=1.0,
            square=True,
            cmap=plt.cm.winter,
            linecolor='white',
            annot=True,
            annot_kws={"size": 12})
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.tight_layout()
plt.savefig('figuresOne\\[附件1]皮尔逊相关系数（14个）.pdf')

# In[33]:


from yellowbrick.features.radviz import RadViz

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

x = dataOne[['是否遇到过网络问题', '居民小区', '手机没有信号', '有信号无法拨通',
             '通话过程中突然中断', '办公室', '通话过程中一方听不见', '通话中有杂音、听不清、断断续续',
             '商业街', '地铁']]
y = dataOne['语音通话整体满意度']

classes = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
visualizer = RadViz(classes=classes, colormap='winter_r')
visualizer.fit(x, y)
visualizer.transform(x)
visualizer.show(outpath='figuresOne\\[附件1]语音通话整体满意度RidViz.pdf')

# In[34]:


from yellowbrick.features.radviz import RadViz

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

x = dataOne[['是否遇到过网络问题', '居民小区', '手机没有信号', '有信号无法拨通',
             '通话过程中突然中断', '办公室', '通话过程中一方听不见', '通话中有杂音、听不清、断断续续',
             '商业街', '地铁']]
y = dataOne['网络覆盖与信号强度']

classes = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
visualizer = RadViz(classes=classes, colormap='winter_r')
visualizer.fit(x, y)
visualizer.transform(x)
visualizer.show(outpath='figuresOne\\[附件1]网络覆盖与信号强度RidViz.pdf')

# In[35]:


from yellowbrick.features.radviz import RadViz

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

x = dataOne[['是否遇到过网络问题', '居民小区', '手机没有信号', '有信号无法拨通',
             '通话过程中突然中断', '办公室', '通话过程中一方听不见', '通话中有杂音、听不清、断断续续',
             '商业街', '地铁']]
y = dataOne['语音通话清晰度']

classes = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
visualizer = RadViz(classes=classes, colormap='winter_r')
visualizer.fit(x, y)
visualizer.transform(x)
visualizer.show(outpath='figuresOne\\[附件1]语音通话清晰度RidViz.pdf')

# In[36]:


from yellowbrick.features.radviz import RadViz

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

x = dataOne[['是否遇到过网络问题', '居民小区', '手机没有信号', '有信号无法拨通',
             '通话过程中突然中断', '办公室', '通话过程中一方听不见', '通话中有杂音、听不清、断断续续',
             '商业街', '地铁']]
y = dataOne['语音通话稳定性']

classes = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
visualizer = RadViz(classes=classes, colormap='winter_r')
visualizer.fit(x, y)
visualizer.transform(x)
visualizer.show(outpath='figuresOne\\[附件1]语音通话稳定性RidViz.pdf')

# ### 数据标准化

# In[37]:


StandardTransform = dataOne[['脱网次数', 'mos质差次数', '未接通掉话次数', '4\\5G用户',
                             '套外流量（MB）', '套外流量费（元）', '语音通话-时长（分钟）', '省际漫游-时长（分钟）',
                             '终端品牌', '当月ARPU', '当月MOU',
                             '前3月ARPU', '前3月MOU', 'GPRS总流量（KB）', 'GPRS-国内漫游-流量（KB）',
                             '客户星级标识']]
StandardTransformScaler = sp.StandardScaler()
StandardTransformScaler = StandardTransformScaler.fit(StandardTransform)
StandardTransform = StandardTransformScaler.transform(StandardTransform)
StandardTransform = pd.DataFrame(StandardTransform)
StandardTransform.columns = ['脱网次数', 'mos质差次数', '未接通掉话次数', '4\\5G用户',
                             '套外流量（MB）', '套外流量费（元）', '语音通话-时长（分钟）', '省际漫游-时长（分钟）',
                             '终端品牌', '当月ARPU', '当月MOU',
                             '前3月ARPU', '前3月MOU', 'GPRS总流量（KB）', 'GPRS-国内漫游-流量（KB）',
                             '客户星级标识']
StandardTransform

# In[38]:


dataOneLeave = dataOne.loc[:, ~dataOne.columns.isin(['脱网次数', 'mos质差次数', '未接通掉话次数', '4\\5G用户',
                                                     '套外流量（MB）', '套外流量费（元）', '语音通话-时长（分钟）',
                                                     '省际漫游-时长（分钟）',
                                                     '终端品牌', '当月ARPU', '当月MOU',
                                                     '前3月ARPU', '前3月MOU', 'GPRS总流量（KB）',
                                                     'GPRS-国内漫游-流量（KB）',
                                                     '客户星级标识'])]

# In[39]:


dataOneNewStandard = pd.concat([dataOneLeave, StandardTransform], axis=1)
dataOneNewStandard

# In[40]:


dataOneNewStandard.columns = ['语音通话整体满意度', '网络覆盖与信号强度', '语音通话清晰度', '语音通话稳定性',
                              '是否遇到过网络问题', '居民小区', '办公室', '高校',
                              '商业街', '地铁', '农村', '高铁',
                              '其他，请注明', '手机没有信号', '有信号无法拨通', '通话过程中突然中断',
                              '通话中有杂音、听不清、断断续续', '串线', '通话过程中一方听不见', '其他，请注明.1',
                              '是否投诉', '是否关怀用户', '是否4G网络客户（本地剔除物联网）', '外省语音占比',
                              '外省流量占比', '是否5G网络客户', '脱网次数', 'mos质差次数',
                              '未接通掉话次数', '4\\5G用户', '套外流量（MB）', '套外流量费（元）',
                              '语音通话-时长（分钟）', '省际漫游-时长（分钟）', '终端品牌',
                              '当月ARPU', '当月MOU', '前3月ARPU', '前3月MOU',
                              'GPRS总流量（KB）', 'GPRS-国内漫游-流量（KB）', '客户星级标识']
dataOneNewStandard

# ### 数据归一化

# In[41]:


MinMaxTransform = dataOne[['脱网次数', 'mos质差次数', '未接通掉话次数', '4\\5G用户',
                           '套外流量（MB）', '套外流量费（元）', '语音通话-时长（分钟）', '省际漫游-时长（分钟）',
                           '终端品牌', '当月ARPU', '当月MOU',
                           '前3月ARPU', '前3月MOU', 'GPRS总流量（KB）', 'GPRS-国内漫游-流量（KB）',
                           '客户星级标识']]
MinMaxTransformScaler = sp.MinMaxScaler()
MinMaxTransformScaler = MinMaxTransformScaler.fit(MinMaxTransform)
MinMaxTransform = MinMaxTransformScaler.transform(MinMaxTransform)
MinMaxTransform = pd.DataFrame(MinMaxTransform)
MinMaxTransform.columns = ['脱网次数', 'mos质差次数', '未接通掉话次数', '4\\5G用户',
                           '套外流量（MB）', '套外流量费（元）', '语音通话-时长（分钟）', '省际漫游-时长（分钟）',
                           '终端品牌', '当月ARPU', '当月MOU',
                           '前3月ARPU', '前3月MOU', 'GPRS总流量（KB）', 'GPRS-国内漫游-流量（KB）',
                           '客户星级标识']
MinMaxTransform

# In[42]:


dataOneNewMinMax = pd.concat([dataOneLeave, MinMaxTransform], axis=1)
dataOneNewMinMax.columns = ['语音通话整体满意度', '网络覆盖与信号强度', '语音通话清晰度', '语音通话稳定性',
                            '是否遇到过网络问题', '居民小区', '办公室', '高校',
                            '商业街', '地铁', '农村', '高铁',
                            '其他，请注明', '手机没有信号', '有信号无法拨通', '通话过程中突然中断',
                            '通话中有杂音、听不清、断断续续', '串线', '通话过程中一方听不见', '其他，请注明.1',
                            '是否投诉', '是否关怀用户', '是否4G网络客户（本地剔除物联网）', '外省语音占比',
                            '外省流量占比', '是否5G网络客户', '脱网次数', 'mos质差次数',
                            '未接通掉话次数', '4\\5G用户', '套外流量（MB）', '套外流量费（元）',
                            '语音通话-时长（分钟）', '省际漫游-时长（分钟）', '终端品牌',
                            '当月ARPU', '当月MOU', '前3月ARPU', '前3月MOU',
                            'GPRS总流量（KB）', 'GPRS-国内漫游-流量（KB）', '客户星级标识']
dataOneNewMinMax

# ## 熵权法

# In[43]:


import copy


def ewm(data):
    label_need = data.keys()[:]
    data1 = data[label_need].values
    data2 = data1
    [m, n] = data2.shape
    data3 = copy.deepcopy(data2)
    y_min = 0.002
    y_max = 1
    for j in range(0, n):
        d_max = max(data2[:, j])
        d_min = min(data2[:, j])
        data3[:, j] = (y_max - y_min) * (data2[:, j] - d_min) / (d_max - d_min) + y_min
    p = copy.deepcopy(data3)
    for j in range(0, n):
        p[:, j] = data3[:, j] / sum(data3[:, j])
    e = copy.deepcopy(data3[0, :])
    for j in range(0, n):
        e[j] = -1 / np.log(m) * sum(p[:, j] * np.log(p[:, j]))
    w = (1 - e) / sum(1 - e)
    total = 0
    for sum_w in range(0, len(w)):
        total = total + w[sum_w]
    print(f'权重为：{w}，权重之和为：{total}')


# In[44]:


ewm(dataOneLeave.iloc[:, 4:])

# In[45]:


dataOneTransform = dataOne[['脱网次数', 'mos质差次数', '未接通掉话次数', '4\\5G用户',
                            '套外流量（MB）', '套外流量费（元）', '语音通话-时长（分钟）', '省际漫游-时长（分钟）',
                            '终端品牌', '当月ARPU', '当月MOU',
                            '前3月ARPU', '前3月MOU', 'GPRS总流量（KB）', 'GPRS-国内漫游-流量（KB）',
                            '客户星级标识']]
dataOneTransform

# In[46]:


ewm(dataOneTransform)


# ## 灰色关联

# In[47]:


def grey(data):
    label_need = data.keys()[:]
    data1 = data[label_need].values
    [m, n] = data1.shape
    data2 = data1.astype('float')
    data3 = data2
    ymin = 0.002
    ymax = 1
    for j in range(0, n):
        d_max = max(data2[:, j])
        d_min = min(data2[:, j])
        data3[:, j] = (ymax - ymin) * (data2[:, j] - d_min) / (d_max - d_min) + ymin

    for i in range(0, n):
        data3[:, i] = np.abs(data3[:, i] - data3[:, 0])
    data4 = data3
    d_max = np.max(data4)
    d_min = np.min(data4)
    a = 0.5
    data4 = (d_min + a * d_max) / (data4 + a * d_max)
    xs = np.mean(data4, axis=0)
    print(xs)


# In[48]:


grey(dataOne.loc[:, ~dataOne.columns.isin(['网络覆盖与信号强度', '语音通话清晰度', '语音通话稳定性'])])

# In[49]:


grey(dataOne.loc[:, ~dataOne.columns.isin(['语音通话整体满意度', '语音通话清晰度', '语音通话稳定性'])])

# In[50]:


grey(dataOne.loc[:, ~dataOne.columns.isin(['语音通话整体满意度', '网络覆盖与信号强度', '语音通话稳定性'])])

# In[51]:


grey(dataOne.loc[:, ~dataOne.columns.isin(['语音通话整体满意度', '网络覆盖与信号强度', '语音通话清晰度'])])

# ## 特征工程与机器学习

# ### 多输出多类别分类

# In[52]:


XdataOneMulti = dataOneNewStandard.loc[:, ~dataOneNewStandard.columns.isin(['语音通话整体满意度', '网络覆盖与信号强度',
                                                                            '语音通话清晰度', '语音通话稳定性'])]
ydataOneMulti = dataOneNewStandard[['语音通话整体满意度', '网络覆盖与信号强度', '语音通话清晰度', '语音通话稳定性']]

# In[53]:


from sklearn.model_selection import train_test_split

XdataOneMulti_train, XdataOneMulti_test, ydataOneMulti_train, ydataOneMulti_test = train_test_split(XdataOneMulti,
                                                                                                    ydataOneMulti,
                                                                                                    test_size=0.2,
                                                                                                    random_state=2022)

# In[54]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

DecisionTreeMulti = DecisionTreeClassifier(random_state=2022)
RandomForestMulti = RandomForestClassifier(random_state=2022)
DecisionTreeMulti = DecisionTreeMulti.fit(XdataOneMulti_train, ydataOneMulti_train)
RandomForestMulti = RandomForestMulti.fit(XdataOneMulti_train, ydataOneMulti_train)

# In[55]:


from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

print(f'决策树平均绝对误差：'
      f'{mean_absolute_error(ydataOneMulti_test, DecisionTreeMulti.predict(XdataOneMulti_test), sample_weight=None, multioutput="uniform_average")}\n'
      f'决策树均方误差：'
      f'{mean_squared_error(ydataOneMulti_test, DecisionTreeMulti.predict(XdataOneMulti_test), sample_weight=None, multioutput="uniform_average")}')
print(f'随机森林平均绝对误差：'
      f'{mean_absolute_error(ydataOneMulti_test, RandomForestMulti.predict(XdataOneMulti_test), sample_weight=None, multioutput="uniform_average")}\n'
      f'随机森林均方误差：'
      f'{mean_squared_error(ydataOneMulti_test, RandomForestMulti.predict(XdataOneMulti_test), sample_weight=None, multioutput="uniform_average")}')

# In[56]:


std = np.std([i.feature_importances_ for i in RandomForestMulti.estimators_], axis=0)
importances = DecisionTreeMulti.feature_importances_
feat_with_importance = pd.Series(importances, XdataOneMulti.columns)
fig, ax = plt.subplots(figsize=(12, 5))
feat_with_importance.plot.bar(yerr=std, ax=ax, color=(5 / 255, 127 / 255, 215 / 255))
ax.set_title("语音业务四项评分各指标特征重要平均程度")
ax.set_ylabel("Mean decrease in impurity", font='Times New Roman')
plt.yticks(font='Times New Roman')
plt.tight_layout()
plt.savefig('figuresOne\\[附件1]语音业务四项评分各指标特征重要平均程度.pdf')

# In[57]:


feat_with_importance

# In[58]:


from sklearn.decomposition import PCA

pca = PCA()
pca.fit(dataOneNewStandard)
evr = pca.explained_variance_ratio_

plt.figure(figsize=(12, 5))
plt.plot(range(0, len(evr)), evr.cumsum(), marker="d", linestyle="-")
plt.xlabel("Number of components", font='Times New Roman')
plt.ylabel("Cumulative explained variance", font='Times New Roman')
plt.xticks(font='Times New Roman')
plt.yticks(font='Times New Roman')
plt.tight_layout()
plt.savefig("figuresOne\\[附件1]PCA累计解释方差图.pdf")

# In[59]:


from sklearn.multioutput import MultiOutputClassifier

RandomForestMulti = MultiOutputClassifier(RandomForestClassifier(random_state=2022))
RandomForestMulti = RandomForestMulti.fit(XdataOneMulti_train, ydataOneMulti_train)
RandomForestMulti_score = RandomForestMulti.score(XdataOneMulti_test, ydataOneMulti_test)
print(f'随机森林平均绝对误差：'
      f'{mean_absolute_error(ydataOneMulti_test, RandomForestMulti.predict(XdataOneMulti_test), sample_weight=None, multioutput="uniform_average")}\n'
      f'随机森林均方误差：'
      f'{mean_squared_error(ydataOneMulti_test, RandomForestMulti.predict(XdataOneMulti_test), sample_weight=None, multioutput="uniform_average")}')
RandomForestMulti_score

# ### "语音通话整体满意度"学习

# In[60]:


XdataOneFirst = dataOneNewStandard.loc[:, ~dataOneNewStandard.columns.isin(['语音通话整体满意度', '网络覆盖与信号强度',
                                                                            '语音通话清晰度', '语音通话稳定性'])]
ydataOneFirst = dataOneNewStandard['语音通话整体满意度']
XdataOneFirst_train, XdataOneFirst_test, ydataOneFirst_train, ydataOneFirst_test = train_test_split(XdataOneFirst,
                                                                                                    ydataOneFirst,
                                                                                                    test_size=0.2,
                                                                                                    random_state=2022)

# #### 决策树，随机森林

# In[61]:


DecisionTreeFirst = DecisionTreeClassifier(random_state=2022)
RandomForestFirst = RandomForestClassifier(random_state=2022)
DecisionTreeFirst = DecisionTreeFirst.fit(XdataOneFirst_train, ydataOneFirst_train)
RandomForestFirst = RandomForestFirst.fit(XdataOneFirst_train, ydataOneFirst_train)
RandomForestFirst_score = RandomForestFirst.score(XdataOneFirst_test, ydataOneFirst_test)
RandomForestFirst_score

# In[62]:


std = np.std([i.feature_importances_ for i in RandomForestFirst.estimators_], axis=0)
importances = DecisionTreeFirst.feature_importances_
feat_with_importance = pd.Series(importances, XdataOneFirst.columns)
fig, ax = plt.subplots(figsize=(12, 5))
feat_with_importance.plot.bar(yerr=std, ax=ax, color=(5 / 255, 126 / 255, 215 / 255))
ax.set_title("语音通话整体满意度各项指标重要程度")
ax.set_ylabel("Mean decrease in impurity", font='Times New Roman')
plt.yticks(font='Times New Roman')
plt.tight_layout()
plt.savefig("figuresOne\\[附件1]语音通话整体满意度各项指标重要程度.pdf")

# In[63]:


feat_with_importance

# #### XGBoost

# In[64]:


from xgboost import XGBClassifier

XGBFirst = XGBClassifier(learning_rate=0.01,
                         n_estimators=14,
                         max_depth=5,
                         min_child_weight=1,
                         gamma=0.,
                         subsample=1,
                         colsample_btree=1,
                         scale_pos_weight=1,
                         random_state=2022,
                         slient=0)
XGBFirst.fit(XdataOneFirst_train, ydataOneFirst_train)
XGBFirst_score = XGBFirst.score(XdataOneFirst_test, ydataOneFirst_test)
XGBFirst_score

# In[65]:


from xgboost import plot_importance

fig, ax = plt.subplots(figsize=(14, 8))
plot_importance(XGBFirst, height=0.4, ax=ax)
plt.xticks(fontsize=13, font='Times New Roman')
plt.yticks(fontsize=11)
ax.set_title("")
ax.set_ylabel("")
ax.set_xlabel("")
plt.tight_layout()
plt.savefig('figuresOne\\[附件1]语音通话整体满意度各项指标重要程度（XGBoost,F-score）.pdf')

# #### KNN

# In[66]:


from sklearn.neighbors import KNeighborsClassifier

KNNFirst = KNeighborsClassifier()
KNNFirst.fit(XdataOneFirst_train, ydataOneFirst_train)
KNNFirst_score = KNNFirst.score(XdataOneFirst_test, ydataOneFirst_test)
KNNFirst_score

# In[67]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

KNN_turing_param_grid = [{'weights': ['uniform'],
                          'n_neighbors': [k for k in range(2, 20)]},
                         {'weights': ['distance'],
                          'n_neighbors': [k for k in range(2, 20)],
                          'p': [p for p in range(1, 5)]}]
KNN_turing = KNeighborsClassifier()
KNN_turing_grid_search = GridSearchCV(KNN_turing,
                                      param_grid=KNN_turing_param_grid,
                                      n_jobs=-1,
                                      verbose=2)
KNN_turing_grid_search.fit(XdataOneFirst_train, ydataOneFirst_train)

# In[68]:


KNN_turing_grid_search.best_score_

# In[69]:


KNN_turing_grid_search.best_params_

# In[70]:


KNNFirst_new = KNeighborsClassifier(n_neighbors=23, p=1, weights='distance')
KNNFirst_new.fit(XdataOneFirst_train, ydataOneFirst_train)
KNNFirst_new_score = KNNFirst_new.score(XdataOneFirst_test, ydataOneFirst_test)
KNNFirst_new_score

# #### 支持向量机

# In[71]:


from sklearn.svm import SVC

SVMFirst = SVC(random_state=2022)
SVMFirst.fit(XdataOneFirst_train, ydataOneFirst_train)
SVMFirst_score = SVMFirst.score(XdataOneFirst_test, ydataOneFirst_test)
SVMFirst_score

# #### lightgbm

# In[72]:


from lightgbm import LGBMClassifier

LightgbmFirst = LGBMClassifier(learning_rate=0.1,
                               lambda_l1=0.1,
                               lambda_l2=0.2,
                               max_depth=4,
                               objective='multiclass',
                               num_class=3,
                               random_state=2022)
LightgbmFirst.fit(XdataOneFirst_train, ydataOneFirst_train)
LightgbmFirst_score = LightgbmFirst.score(XdataOneFirst_test, ydataOneFirst_test)
LightgbmFirst_score

# #### 逻辑回归

# In[73]:


from sklearn.linear_model import LogisticRegression

LogisticRegressionFirst = LogisticRegression(multi_class="multinomial", solver="newton-cg", max_iter=1000)
LogisticRegressionFirst = LogisticRegressionFirst.fit(XdataOneFirst_train, ydataOneFirst_train)
LogisticRegressionFirst_score = LogisticRegressionFirst.score(XdataOneFirst_test, ydataOneFirst_test)
LogisticRegressionFirst_score

# In[74]:


print(f'模型一中RF平均绝对误差：'
      f'{mean_absolute_error(ydataOneFirst_test, RandomForestFirst.predict(XdataOneFirst_test), sample_weight=None, multioutput="uniform_average")}\n'
      f'模型一中RF均方误差：'
      f'{mean_squared_error(ydataOneFirst_test, RandomForestFirst.predict(XdataOneFirst_test), sample_weight=None, multioutput="uniform_average")}')
print(f'模型一中XGBoost平均绝对误差：'
      f'{mean_absolute_error(ydataOneFirst_test, XGBFirst.predict(XdataOneFirst_test), sample_weight=None, multioutput="uniform_average")}\n'
      f'模型一中XGBoost均方误差：'
      f'{mean_squared_error(ydataOneFirst_test, XGBFirst.predict(XdataOneFirst_test), sample_weight=None, multioutput="uniform_average")}')
print(f'模型一中KNN平均绝对误差：'
      f'{mean_absolute_error(ydataOneFirst_test, KNNFirst_new.predict(XdataOneFirst_test), sample_weight=None, multioutput="uniform_average")}\n'
      f'模型一中KNN均方误差：'
      f'{mean_squared_error(ydataOneFirst_test, KNNFirst_new.predict(XdataOneFirst_test), sample_weight=None, multioutput="uniform_average")}')
print(f'模型一中SVM平均绝对误差：'
      f'{mean_absolute_error(ydataOneFirst_test, SVMFirst.predict(XdataOneFirst_test), sample_weight=None, multioutput="uniform_average")}\n'
      f'模型一中SVM均方误差：'
      f'{mean_squared_error(ydataOneFirst_test, SVMFirst.predict(XdataOneFirst_test), sample_weight=None, multioutput="uniform_average")}')
print(f'模型一中LightGBM平均绝对误差：'
      f'{mean_absolute_error(ydataOneFirst_test, LightgbmFirst.predict(XdataOneFirst_test), sample_weight=None, multioutput="uniform_average")}\n'
      f'模型一中LightGBM均方误差：'
      f'{mean_squared_error(ydataOneFirst_test, LightgbmFirst.predict(XdataOneFirst_test), sample_weight=None, multioutput="uniform_average")}')
print(f'模型一中LR平均绝对误差：'
      f'{mean_absolute_error(ydataOneFirst_test, LogisticRegressionFirst.predict(XdataOneFirst_test), sample_weight=None, multioutput="uniform_average")}\n'
      f'模型一中LR均方误差：'
      f'{mean_squared_error(ydataOneFirst_test, LogisticRegressionFirst.predict(XdataOneFirst_test), sample_weight=None, multioutput="uniform_average")}')

# #### 集成学习

# In[75]:


from mlxtend.classifier import StackingCVClassifier

FirstModel = StackingCVClassifier(
    classifiers=[LogisticRegressionFirst, XGBFirst, KNNFirst_new, SVMFirst, LightgbmFirst],
    meta_classifier=RandomForestClassifier(random_state=2022), random_state=2022, cv=5)
FirstModel.fit(XdataOneFirst_train, ydataOneFirst_train)
FirstModel_score = FirstModel.score(XdataOneFirst_test, ydataOneFirst_test)
FirstModel_score

# In[76]:


print(f'模型一平均绝对误差：'
      f'{mean_absolute_error(ydataOneFirst_test, FirstModel.predict(XdataOneFirst_test), sample_weight=None, multioutput="uniform_average")}\n'
      f'模型一均方误差：'
      f'{mean_squared_error(ydataOneFirst_test, FirstModel.predict(XdataOneFirst_test), sample_weight=None, multioutput="uniform_average")}')

# ### "网络覆盖与信号强度"学习

# In[77]:


XdataOneSecond = dataOneNewStandard.loc[:, ~dataOneNewStandard.columns.isin(['语音通话整体满意度', '网络覆盖与信号强度',
                                                                             '语音通话清晰度', '语音通话稳定性'])]
ydataOneSecond = dataOneNewStandard['网络覆盖与信号强度']
XdataOneSecond_train, XdataOneSecond_test, ydataOneSecond_train, ydataOneSecond_test = train_test_split(XdataOneSecond,
                                                                                                        ydataOneSecond,
                                                                                                        test_size=0.2,
                                                                                                        random_state=2022)

# #### 决策树、随机森林

# In[78]:


DecisionTreeSecond = DecisionTreeClassifier(random_state=2022)
RandomForestSecond = RandomForestClassifier(random_state=2022)
DecisionTreeSecond = DecisionTreeSecond.fit(XdataOneSecond_train, ydataOneSecond_train)
RandomForestSecond = RandomForestSecond.fit(XdataOneSecond_train, ydataOneSecond_train)
RandomForestSecond_score = RandomForestSecond.score(XdataOneSecond_test, ydataOneSecond_test)
RandomForestSecond_score

# In[79]:


from sklearn.model_selection import cross_val_score

scorel = []
for i in range(0, 200, 10):
    RFC = RandomForestClassifier(n_estimators=i + 1,
                                 n_jobs=-1,
                                 random_state=2022)
    score = cross_val_score(RFC, XdataOneSecond, ydataOneSecond, cv=10).mean()
    scorel.append(score)

print(max(scorel), (scorel.index(max(scorel)) * 10) + 1)
plt.figure(figsize=[20, 5])
plt.plot(range(1, 201, 10), scorel)
plt.xticks(fontsize=12, font='Times New Roman')
plt.yticks(fontsize=12, font='Times New Roman')
plt.tight_layout()
plt.savefig("figuresOne\\[附件1]随机森林调参F.pdf")

# In[80]:


scorel = []
for i in range(150, 170):
    RFC = RandomForestClassifier(n_estimators=i,
                                 n_jobs=-1,
                                 random_state=2022)
    score = cross_val_score(RFC, XdataOneSecond, ydataOneSecond, cv=10).mean()
    scorel.append(score)

print(max(scorel), ([*range(150, 170)][scorel.index(max(scorel))]))
plt.figure(figsize=[20, 5])
plt.plot(range(150, 170), scorel)
plt.xticks(fontsize=12, font='Times New Roman')
plt.yticks(fontsize=12, font='Times New Roman')
plt.tight_layout()
plt.savefig("figuresOne\\[附件1]随机森林调参S.pdf")

# In[81]:


import numpy as np

param_grid = {'max_features': ['auto', 'sqrt', 'log2']}
RFC = RandomForestClassifier(n_estimators=164, random_state=2022)
GS = GridSearchCV(RFC, param_grid, cv=10)
GS.fit(XdataOneSecond, ydataOneSecond)
GS.best_params_

# In[82]:


param_grid = {'min_samples_leaf': np.arange(1, 11, 1)}
RFC = RandomForestClassifier(n_estimators=164, random_state=2022, max_features='log2')
GS = GridSearchCV(RFC, param_grid, cv=10)
GS.fit(XdataOneSecond, ydataOneSecond)
GS.best_params_

# In[83]:


param_grid = {'criterion': ['gini', 'entropy']}
RFC = RandomForestClassifier(n_estimators=164, random_state=2022, max_features='log2', min_samples_leaf=8)
GS = GridSearchCV(RFC, param_grid, cv=10)
GS.fit(XdataOneSecond, ydataOneSecond)
GS.best_params_

# In[84]:


from sklearn.model_selection import GridSearchCV

param_grid = {'max_depth': np.arange(1, 20, 1)}
RFC = RandomForestClassifier(n_estimators=164, random_state=2022, max_features='log2', min_samples_leaf=8)
GS = GridSearchCV(RFC, param_grid, cv=10)
GS.fit(XdataOneSecond, ydataOneSecond)
GS.best_params_

# In[85]:


RandomForestSecond = RandomForestClassifier(n_estimators=164, random_state=2022, min_samples_leaf=8, max_depth=19)
RandomForestSecond = RandomForestSecond.fit(XdataOneSecond_train, ydataOneSecond_train)
RandomForestSecond_score = RandomForestSecond.score(XdataOneSecond_test, ydataOneSecond_test)
RandomForestSecond_score

# In[86]:


std = np.std([i.feature_importances_ for i in RandomForestSecond.estimators_], axis=0)
importances = DecisionTreeSecond.feature_importances_
feat_with_importance = pd.Series(importances, XdataOneSecond.columns)
fig, ax = plt.subplots(figsize=(12, 5))
feat_with_importance.plot.bar(yerr=std, ax=ax, color=(5 / 255, 126 / 255, 215 / 255))
ax.set_title("网络覆盖与信号强度各项指标重要程度")
ax.set_ylabel("Mean decrease in impurity", font='Times New Roman')
plt.yticks(font='Times New Roman')
plt.tight_layout()
plt.savefig("figuresOne\\[附件1]网络覆盖与信号强度各项指标重要程度.pdf")

# In[87]:


feat_with_importance

# #### XGBoost

# In[88]:


from xgboost import XGBClassifier

XGBSecond = XGBClassifier(learning_rate=0.02,
                          n_estimators=13,
                          max_depth=8,
                          min_child_weight=1,
                          gamma=0.05,
                          subsample=1,
                          colsample_btree=1,
                          scale_pos_weight=1,
                          random_state=2022,
                          slient=0)
XGBSecond.fit(XdataOneSecond_train, ydataOneSecond_train)
XGBSecond_score = XGBSecond.score(XdataOneSecond_test, ydataOneSecond_test)
XGBSecond_score

# In[89]:


from xgboost import plot_importance

fig, ax = plt.subplots(figsize=(14, 8))
plot_importance(XGBSecond, height=0.4, ax=ax)
plt.xticks(fontsize=13, font='Times New Roman')
plt.yticks(fontsize=11)
ax.set_title("")
ax.set_ylabel("")
ax.set_xlabel("")
plt.tight_layout()
plt.savefig('figuresOne\\[附件1]网络覆盖与信号强度各项指标重要程度（XGBoost,F-score）.pdf')

# #### KNN

# In[90]:


from sklearn.neighbors import KNeighborsClassifier

KNNSecond = KNeighborsClassifier()
KNNSecond.fit(XdataOneSecond_train, ydataOneSecond_train)
KNNSecond_score = KNNSecond.score(XdataOneSecond_test, ydataOneSecond_test)
KNNSecond_score

# In[91]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

KNN_turing_param_grid = [{'weights': ['uniform'],
                          'n_neighbors': [k for k in range(40, 50)]},
                         {'weights': ['distance'],
                          'n_neighbors': [k for k in range(40, 50)],
                          'p': [p for p in range(1, 5)]}]
KNN_turing = KNeighborsClassifier()
KNN_turing_grid_search = GridSearchCV(KNN_turing,
                                      param_grid=KNN_turing_param_grid,
                                      n_jobs=-1,
                                      verbose=2)
KNN_turing_grid_search.fit(XdataOneSecond_train, ydataOneSecond_train)

# In[92]:


KNN_turing_grid_search.best_score_

# In[93]:


KNN_turing_grid_search.best_params_

# In[94]:


KNNSecond_new = KNeighborsClassifier(algorithm='auto', leaf_size=30,
                                     metric='minkowski',
                                     n_jobs=-1,
                                     n_neighbors=46, p=1,
                                     weights='distance')
KNNSecond_new.fit(XdataOneSecond_train, ydataOneSecond_train)
KNNSecond_new_score = KNNSecond_new.score(XdataOneSecond_test, ydataOneSecond_test)
KNNSecond_new_score

# #### 支持向量机

# In[95]:


from sklearn.svm import SVC

SVMSecond = SVC(random_state=2022)
SVMSecond.fit(XdataOneSecond_train, ydataOneSecond_train)
SVMSecond_score = SVMSecond.score(XdataOneSecond_test, ydataOneSecond_test)
SVMSecond_score

# #### lightgbm

# In[96]:


from lightgbm import LGBMClassifier

LightgbmSecond = LGBMClassifier(learning_rate=0.1,
                                lambda_l1=0.1,
                                lambda_l2=0.2,
                                max_depth=3,
                                objective='multiclass',
                                num_class=3,
                                random_state=2022)
LightgbmSecond.fit(XdataOneSecond_train, ydataOneSecond_train)
LightgbmSecond_score = LightgbmSecond.score(XdataOneSecond_test, ydataOneSecond_test)
LightgbmSecond_score

# #### 逻辑回归

# In[97]:


from sklearn.linear_model import LogisticRegression

LogisticRegressionSecond = LogisticRegression(multi_class="multinomial", solver="newton-cg", max_iter=2000)
LogisticRegressionSecond = LogisticRegressionSecond.fit(XdataOneSecond_train, ydataOneSecond_train)
LogisticRegressionSecond_score = LogisticRegressionSecond.score(XdataOneSecond_test, ydataOneSecond_test)
LogisticRegressionSecond_score

# In[98]:


print(f'模型二中RF平均绝对误差：'
      f'{mean_absolute_error(ydataOneSecond_test, RandomForestSecond.predict(XdataOneSecond_test), sample_weight=None, multioutput="uniform_average")}\n'
      f'模型二中RF均方误差：'
      f'{mean_squared_error(ydataOneSecond_test, RandomForestSecond.predict(XdataOneSecond_test), sample_weight=None, multioutput="uniform_average")}')
print(f'模型二中XGBoost平均绝对误差：'
      f'{mean_absolute_error(ydataOneSecond_test, XGBSecond.predict(XdataOneSecond_test), sample_weight=None, multioutput="uniform_average")}\n'
      f'模型二中XGBoost均方误差：'
      f'{mean_squared_error(ydataOneSecond_test, XGBSecond.predict(XdataOneSecond_test), sample_weight=None, multioutput="uniform_average")}')
print(f'模型二中KNN平均绝对误差：'
      f'{mean_absolute_error(ydataOneSecond_test, KNNSecond_new.predict(XdataOneSecond_test), sample_weight=None, multioutput="uniform_average")}\n'
      f'模型二中KNN均方误差：'
      f'{mean_squared_error(ydataOneSecond_test, KNNSecond_new.predict(XdataOneSecond_test), sample_weight=None, multioutput="uniform_average")}')
print(f'模型二中SVM平均绝对误差：'
      f'{mean_absolute_error(ydataOneSecond_test, SVMSecond.predict(XdataOneSecond_test), sample_weight=None, multioutput="uniform_average")}\n'
      f'模型二中SVM均方误差：'
      f'{mean_squared_error(ydataOneSecond_test, SVMSecond.predict(XdataOneSecond_test), sample_weight=None, multioutput="uniform_average")}')
print(f'模型二中LightGBM平均绝对误差：'
      f'{mean_absolute_error(ydataOneSecond_test, LightgbmSecond.predict(XdataOneSecond_test), sample_weight=None, multioutput="uniform_average")}\n'
      f'模型二中LightGBM均方误差：'
      f'{mean_squared_error(ydataOneSecond_test, LightgbmSecond.predict(XdataOneSecond_test), sample_weight=None, multioutput="uniform_average")}')
print(f'模型二中LR平均绝对误差：'
      f'{mean_absolute_error(ydataOneSecond_test, LogisticRegressionSecond.predict(XdataOneSecond_test), sample_weight=None, multioutput="uniform_average")}\n'
      f'模型二中LR均方误差：'
      f'{mean_squared_error(ydataOneSecond_test, LogisticRegressionSecond.predict(XdataOneSecond_test), sample_weight=None, multioutput="uniform_average")}')

# #### 集成学习

# In[99]:


from mlxtend.classifier import StackingCVClassifier

SecondModel = StackingCVClassifier(
    classifiers=[RandomForestSecond, XGBSecond, KNNSecond_new, SVMSecond, LogisticRegressionSecond],
    meta_classifier=LGBMClassifier(random_state=2022), random_state=2022, cv=5)
SecondModel.fit(XdataOneSecond_train, ydataOneSecond_train)
SecondModel_score = SecondModel.score(XdataOneSecond_test, ydataOneSecond_test)
SecondModel_score

# In[100]:


print(f'模型二平均绝对误差：'
      f'{mean_absolute_error(ydataOneSecond_test, SecondModel.predict(XdataOneSecond_test), sample_weight=None, multioutput="uniform_average")}\n'
      f'模型二均方误差：'
      f'{mean_squared_error(ydataOneSecond_test, SecondModel.predict(XdataOneSecond_test), sample_weight=None, multioutput="uniform_average")}')

# ### "语音通话清晰度"学习

# In[101]:


XdataOneThird = dataOneNewStandard.loc[:, ~dataOneNewStandard.columns.isin(['语音通话整体满意度', '网络覆盖与信号强度',
                                                                            '语音通话清晰度', '语音通话稳定性'])]
ydataOneThird = dataOneNewStandard['语音通话清晰度']
XdataOneThird_train, XdataOneThird_test, ydataOneThird_train, ydataOneThird_test = train_test_split(XdataOneThird,
                                                                                                    ydataOneThird,
                                                                                                    test_size=0.2,
                                                                                                    random_state=2022)

# #### 决策树、随机森林

# In[102]:


DecisionTreeThird = DecisionTreeClassifier(random_state=2022)
RandomForestThird = RandomForestClassifier(random_state=2022)
DecisionTreeThird = DecisionTreeThird.fit(XdataOneThird_train, ydataOneThird_train)
RandomForestThird = RandomForestThird.fit(XdataOneThird_train, ydataOneThird_train)
RandomForestThird_score = RandomForestThird.score(XdataOneThird_test, ydataOneThird_test)
RandomForestThird_score

# In[103]:


std = np.std([i.feature_importances_ for i in RandomForestThird.estimators_], axis=0)
importances = DecisionTreeThird.feature_importances_
feat_with_importance = pd.Series(importances, XdataOneThird.columns)
fig, ax = plt.subplots(figsize=(12, 5))
feat_with_importance.plot.bar(yerr=std, ax=ax, color=(5 / 255, 126 / 255, 215 / 255))
ax.set_title("语音通话清晰度各项指标重要程度")
ax.set_ylabel("Mean decrease in impurity", font='Times New Roman')
plt.yticks(font='Times New Roman')
plt.tight_layout()
plt.savefig("figuresOne\\[附件1]语音通话清晰度各项指标重要程度.pdf")

# In[104]:


feat_with_importance

# #### XGBoost

# In[105]:


from xgboost import XGBClassifier

XGBThird = XGBClassifier(learning_rate=0.02,
                         n_estimators=14,
                         max_depth=8,
                         min_child_weight=1,
                         gamma=0.05,
                         subsample=1,
                         colsample_btree=1,
                         scale_pos_weight=1,
                         random_state=2022,
                         slient=0)
XGBThird.fit(XdataOneThird_train, ydataOneThird_train)
XGBThird_score = XGBThird.score(XdataOneThird_test, ydataOneThird_test)
XGBThird_score

# In[106]:


from xgboost import plot_importance

fig, ax = plt.subplots(figsize=(14, 8))
plot_importance(XGBThird, height=0.4, ax=ax)
plt.xticks(fontsize=13, font='Times New Roman')
plt.yticks(fontsize=11)
ax.set_title("")
ax.set_ylabel("")
ax.set_xlabel("")
plt.tight_layout()
plt.savefig('figuresOne\\[附件1]语音通话清晰度各项指标重要程度（XGBoost,F-score）.pdf')

# #### KNN

# In[107]:


from sklearn.neighbors import KNeighborsClassifier

KNNThird = KNeighborsClassifier()
KNNThird.fit(XdataOneThird_train, ydataOneThird_train)
KNNThird_score = KNNThird.score(XdataOneThird_test, ydataOneThird_test)
KNNThird_score

# In[108]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

KNN_turing_param_grid = [{'weights': ['uniform'],
                          'n_neighbors': [k for k in range(30, 40)]},
                         {'weights': ['distance'],
                          'n_neighbors': [k for k in range(30, 40)],
                          'p': [p for p in range(1, 5)]}]
KNN_turing = KNeighborsClassifier()
KNN_turing_grid_search = GridSearchCV(KNN_turing,
                                      param_grid=KNN_turing_param_grid,
                                      n_jobs=-1,
                                      verbose=2)
KNN_turing_grid_search.fit(XdataOneThird_train, ydataOneThird_train)

# In[109]:


KNN_turing_grid_search.best_score_

# In[110]:


KNN_turing_grid_search.best_params_

# In[111]:


KNNThird_new = KNeighborsClassifier(algorithm='auto', leaf_size=30,
                                    metric='minkowski',
                                    n_jobs=-1,
                                    n_neighbors=40, p=1,
                                    weights='uniform')
KNNThird_new.fit(XdataOneThird_train, ydataOneThird_train)
KNNThird_new_score = KNNThird_new.score(XdataOneThird_test, ydataOneThird_test)
KNNThird_new_score

# #### 支持向量机

# In[112]:


from sklearn.svm import SVC

SVMThird = SVC(random_state=2022)
SVMThird.fit(XdataOneThird_train, ydataOneThird_train)
SVMThird_score = SVMThird.score(XdataOneThird_test, ydataOneThird_test)
SVMThird_score

# #### lightgbm

# In[113]:


from lightgbm import LGBMClassifier

LightgbmThird = LGBMClassifier(learning_rate=0.1,
                               lambda_l1=0.1,
                               lambda_l2=0.2,
                               max_depth=9,
                               objective='multiclass',
                               num_class=4,
                               random_state=2022)
LightgbmThird.fit(XdataOneThird_train, ydataOneThird_train)
LightgbmThird_score = LightgbmThird.score(XdataOneThird_test, ydataOneThird_test)
LightgbmThird_score

# #### 逻辑回归

# In[114]:


from sklearn.linear_model import LogisticRegression

LogisticRegressionThird = LogisticRegression(multi_class="multinomial", solver="newton-cg", max_iter=2000)
LogisticRegressionThird = LogisticRegressionThird.fit(XdataOneThird_train, ydataOneThird_train)
LogisticRegressionThird_score = LogisticRegressionThird.score(XdataOneThird_test, ydataOneThird_test)
LogisticRegressionThird_score

# In[115]:


print(f'模型三中RF平均绝对误差：'
      f'{mean_absolute_error(ydataOneThird_test, RandomForestThird.predict(XdataOneThird_test), sample_weight=None, multioutput="uniform_average")}\n'
      f'模型三中RF均方误差：'
      f'{mean_squared_error(ydataOneThird_test, RandomForestThird.predict(XdataOneThird_test), sample_weight=None, multioutput="uniform_average")}')
print(f'模型三中XGBoost平均绝对误差：'
      f'{mean_absolute_error(ydataOneThird_test, XGBThird.predict(XdataOneThird_test), sample_weight=None, multioutput="uniform_average")}\n'
      f'模型三中XGBoost均方误差：'
      f'{mean_squared_error(ydataOneThird_test, XGBThird.predict(XdataOneThird_test), sample_weight=None, multioutput="uniform_average")}')
print(f'模型三中KNN平均绝对误差：'
      f'{mean_absolute_error(ydataOneThird_test, KNNThird_new.predict(XdataOneThird_test), sample_weight=None, multioutput="uniform_average")}\n'
      f'模型三中KNN均方误差：'
      f'{mean_squared_error(ydataOneThird_test, KNNThird_new.predict(XdataOneThird_test), sample_weight=None, multioutput="uniform_average")}')
print(f'模型三中SVM平均绝对误差：'
      f'{mean_absolute_error(ydataOneThird_test, SVMThird.predict(XdataOneThird_test), sample_weight=None, multioutput="uniform_average")}\n'
      f'模型三中SVM均方误差：'
      f'{mean_squared_error(ydataOneThird_test, SVMThird.predict(XdataOneThird_test), sample_weight=None, multioutput="uniform_average")}')
print(f'模型三中LightGBM平均绝对误差：'
      f'{mean_absolute_error(ydataOneThird_test, LightgbmThird.predict(XdataOneThird_test), sample_weight=None, multioutput="uniform_average")}\n'
      f'模型三中LightGBM均方误差：'
      f'{mean_squared_error(ydataOneThird_test, LightgbmThird.predict(XdataOneThird_test), sample_weight=None, multioutput="uniform_average")}')
print(f'模型三中LR平均绝对误差：'
      f'{mean_absolute_error(ydataOneThird_test, LogisticRegressionThird.predict(XdataOneThird_test), sample_weight=None, multioutput="uniform_average")}\n'
      f'模型三中LR均方误差：'
      f'{mean_squared_error(ydataOneThird_test, LogisticRegressionThird.predict(XdataOneThird_test), sample_weight=None, multioutput="uniform_average")}')

# #### 集成学习

# In[116]:


from mlxtend.classifier import StackingCVClassifier

ThirdModel = StackingCVClassifier(
    classifiers=[XGBThird, LightgbmThird, KNNThird_new, SVMThird, LogisticRegressionThird],
    meta_classifier=RandomForestClassifier(random_state=2022), random_state=2022, cv=5)
ThirdModel.fit(XdataOneThird_train, ydataOneThird_train)
ThirdModel_score = ThirdModel.score(XdataOneThird_test, ydataOneThird_test)
ThirdModel_score

# In[117]:


print(f'模型三平均绝对误差：'
      f'{mean_absolute_error(ydataOneThird_test, ThirdModel.predict(XdataOneThird_test), sample_weight=None, multioutput="uniform_average")}\n'
      f'模型三均方误差：'
      f'{mean_squared_error(ydataOneThird_test, ThirdModel.predict(XdataOneThird_test), sample_weight=None, multioutput="uniform_average")}')

# ### "语音通话稳定性"学习

# In[118]:


XdataOneFourth = dataOneNewStandard.loc[:, ~dataOneNewStandard.columns.isin(['语音通话整体满意度', '网络覆盖与信号强度',
                                                                             '语音通话清晰度', '语音通话稳定性'])]
ydataOneFourth = dataOneNewStandard['语音通话稳定性']
XdataOneFourth_train, XdataOneFourth_test, ydataOneFourth_train, ydataOneFourth_test = train_test_split(XdataOneFourth,
                                                                                                        ydataOneFourth,
                                                                                                        test_size=0.2,
                                                                                                        random_state=2022)

# #### 决策树、随机森林

# In[119]:


DecisionTreeFourth = DecisionTreeClassifier(random_state=2022)
RandomForestFourth = RandomForestClassifier(random_state=2022)
DecisionTreeFourth = DecisionTreeFourth.fit(XdataOneFourth_train, ydataOneFourth_train)
RandomForestFourth = RandomForestFourth.fit(XdataOneFourth_train, ydataOneFourth_train)
RandomForestFourth_score = RandomForestFourth.score(XdataOneFourth_test, ydataOneFourth_test)
RandomForestFourth_score

# In[120]:


std = np.std([i.feature_importances_ for i in RandomForestFourth.estimators_], axis=0)
importances = DecisionTreeFourth.feature_importances_
feat_with_importance = pd.Series(importances, XdataOneFourth.columns)
fig, ax = plt.subplots(figsize=(12, 5))
feat_with_importance.plot.bar(yerr=std, ax=ax, color=(5 / 255, 126 / 255, 215 / 255))
ax.set_title("语音通话稳定性各项指标重要程度")
ax.set_ylabel("Mean decrease in impurity", font='Times New Roman')
plt.yticks(font='Times New Roman')
plt.tight_layout()
plt.savefig("figuresOne\\[附件1]语音通话稳定性各项指标重要程度.pdf")

# In[121]:


feat_with_importance

# #### XGBoost

# In[122]:


from xgboost import XGBClassifier

XGBFourth = XGBClassifier(learning_rate=0.02,
                          n_estimators=14,
                          max_depth=6,
                          min_child_weight=1,
                          gamma=0.05,
                          subsample=1,
                          colsample_btree=1,
                          scale_pos_weight=1,
                          random_state=2022,
                          slient=0)
XGBFourth.fit(XdataOneFourth_train, ydataOneFourth_train)
XGBFourth_score = XGBFourth.score(XdataOneFourth_test, ydataOneFourth_test)
XGBFourth_score

# In[123]:


from xgboost import plot_importance

fig, ax = plt.subplots(figsize=(14, 8))
plot_importance(XGBFourth, height=0.4, ax=ax)
plt.xticks(fontsize=13, font='Times New Roman')
plt.yticks(fontsize=11)
ax.set_title("")
ax.set_ylabel("")
ax.set_xlabel("")
plt.tight_layout()
plt.savefig('figuresOne\\[附件1]语音通话稳定性各项指标重要程度（XGBoost,F-score）.pdf')

# #### KNN

# In[124]:


from sklearn.neighbors import KNeighborsClassifier

KNNFourth = KNeighborsClassifier()
KNNFourth.fit(XdataOneFourth_train, ydataOneFourth_train)
KNNFourth_score = KNNFourth.score(XdataOneFourth_test, ydataOneFourth_test)
KNNFourth_score

# In[125]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

KNN_turing_param_grid = [{'weights': ['uniform'],
                          'n_neighbors': [k for k in range(35, 45)]},
                         {'weights': ['distance'],
                          'n_neighbors': [k for k in range(35, 45)],
                          'p': [p for p in range(1, 5)]}]
KNN_turing = KNeighborsClassifier()
KNN_turing_grid_search = GridSearchCV(KNN_turing,
                                      param_grid=KNN_turing_param_grid,
                                      n_jobs=-1,
                                      verbose=2)
KNN_turing_grid_search.fit(XdataOneFourth_train, ydataOneFourth_train)

# In[126]:


KNN_turing_grid_search.best_score_

# In[127]:


KNN_turing_grid_search.best_params_

# In[128]:


KNNFourth_new = KNeighborsClassifier(algorithm='auto', leaf_size=30,
                                     metric='minkowski',
                                     n_jobs=-1,
                                     n_neighbors=43, p=1,
                                     weights='distance')
KNNFourth_new.fit(XdataOneFourth_train, ydataOneFourth_train)
KNNFourth_new_score = KNNFourth_new.score(XdataOneFourth_test, ydataOneFourth_test)
KNNFourth_new_score

# #### 支持向量机

# In[129]:


from sklearn.svm import SVC

SVMFourth = SVC(random_state=2022)
SVMFourth.fit(XdataOneFourth_train, ydataOneFourth_train)
SVMFourth_score = SVMFourth.score(XdataOneFourth_test, ydataOneFourth_test)
SVMFourth_score

# #### lightgbm

# In[130]:


from lightgbm import LGBMClassifier

LightgbmFourth = LGBMClassifier(learning_rate=0.1,
                                lambda_l1=0.1,
                                lambda_l2=0.2,
                                max_depth=10,
                                objective='multiclass',
                                num_class=4,
                                random_state=2022)
LightgbmFourth.fit(XdataOneFourth_train, ydataOneFourth_train)
LightgbmFourth_score = LightgbmFourth.score(XdataOneFourth_test, ydataOneFourth_test)
LightgbmFourth_score

# #### 逻辑回归

# In[131]:


from sklearn.linear_model import LogisticRegression

LogisticRegressionFourth = LogisticRegression(multi_class="multinomial", solver="newton-cg", max_iter=2000)
LogisticRegressionFourth = LogisticRegressionFourth.fit(XdataOneFourth_train, ydataOneFourth_train)
LogisticRegressionFourth_score = LogisticRegressionFourth.score(XdataOneFourth_test, ydataOneFourth_test)
LogisticRegressionFourth_score

# In[132]:


print(f'模型四中RF平均绝对误差：'
      f'{mean_absolute_error(ydataOneFourth_test, RandomForestFourth.predict(XdataOneFourth_test), sample_weight=None, multioutput="uniform_average")}\n'
      f'模型四中RF均方误差：'
      f'{mean_squared_error(ydataOneFourth_test, RandomForestFourth.predict(XdataOneFourth_test), sample_weight=None, multioutput="uniform_average")}')
print(f'模型四中XGBoost平均绝对误差：'
      f'{mean_absolute_error(ydataOneFourth_test, XGBFourth.predict(XdataOneFourth_test), sample_weight=None, multioutput="uniform_average")}\n'
      f'模型四中XGBoost均方误差：'
      f'{mean_squared_error(ydataOneFourth_test, XGBFourth.predict(XdataOneFourth_test), sample_weight=None, multioutput="uniform_average")}')
print(f'模型四中KNN平均绝对误差：'
      f'{mean_absolute_error(ydataOneFourth_test, KNNFourth_new.predict(XdataOneFourth_test), sample_weight=None, multioutput="uniform_average")}\n'
      f'模型四中KNN均方误差：'
      f'{mean_squared_error(ydataOneFourth_test, KNNFourth_new.predict(XdataOneFourth_test), sample_weight=None, multioutput="uniform_average")}')
print(f'模型四中SVM平均绝对误差：'
      f'{mean_absolute_error(ydataOneFourth_test, SVMFourth.predict(XdataOneFourth_test), sample_weight=None, multioutput="uniform_average")}\n'
      f'模型四中SVM均方误差：'
      f'{mean_squared_error(ydataOneFourth_test, SVMFourth.predict(XdataOneFourth_test), sample_weight=None, multioutput="uniform_average")}')
print(f'模型四中LightGBM平均绝对误差：'
      f'{mean_absolute_error(ydataOneFourth_test, LightgbmFourth.predict(XdataOneFourth_test), sample_weight=None, multioutput="uniform_average")}\n'
      f'模型四中LightGBM均方误差：'
      f'{mean_squared_error(ydataOneFourth_test, LightgbmFourth.predict(XdataOneFourth_test), sample_weight=None, multioutput="uniform_average")}')
print(f'模型四中LR平均绝对误差：'
      f'{mean_absolute_error(ydataOneFourth_test, LogisticRegressionFourth.predict(XdataOneFourth_test), sample_weight=None, multioutput="uniform_average")}\n'
      f'模型四中LR均方误差：'
      f'{mean_squared_error(ydataOneFourth_test, LogisticRegressionFourth.predict(XdataOneFourth_test), sample_weight=None, multioutput="uniform_average")}')

# #### 集成学习

# In[133]:


from mlxtend.classifier import StackingCVClassifier

FourthModel = StackingCVClassifier(
    classifiers=[RandomForestFourth, LightgbmFourth, KNNFourth_new, LogisticRegressionFourth, SVMFourth],
    meta_classifier=XGBClassifier(random_state=2022), random_state=2022, cv=5)
FourthModel.fit(XdataOneFourth_train, ydataOneFourth_train)
FourthModel_score = FourthModel.score(XdataOneFourth_test, ydataOneFourth_test)
FourthModel_score

# In[134]:


print(f'模型四平均绝对误差：'
      f'{mean_absolute_error(ydataOneFourth_test, FourthModel.predict(XdataOneFourth_test), sample_weight=None, multioutput="uniform_average")}\n'
      f'模型四均方误差：'
      f'{mean_squared_error(ydataOneFourth_test, FourthModel.predict(XdataOneFourth_test), sample_weight=None, multioutput="uniform_average")}')

# ## 预测附件3四项评分

# In[135]:


dataThree = pd.read_excel("附件3语音业务用户满意度预测数据.xlsx", sheet_name='语音')
dataThree

# ### 附件格式统一

# In[136]:


dataThree.drop(['用户id',
                '用户描述',
                '用户描述.1',
                '性别',
                '终端品牌类型',
                '是否不限量套餐到达用户'], axis=1, inplace=True)

# In[137]:


dataThree

# In[138]:


dataThree.isnull().sum()

# In[139]:


dataThree["外省流量占比"] = dataThree["外省流量占比"].astype(str).replace('%', '')
dataThree["外省语音占比"] = dataThree["外省语音占比"].astype(str).replace('%', '')
dataThree

# In[140]:


dataThree.replace({"是否遇到过网络问题": {2: 0},
                   "居民小区": {-1: 0},
                   "办公室": {-1: 0, 2: 1},
                   "高校": {-1: 0, 3: 1},
                   "商业街": {-1: 0, 4: 1},
                   "地铁": {-1: 0, 5: 1},
                   "农村": {-1: 0, 6: 1},
                   "高铁": {-1: 0, 7: 1},
                   "其他，请注明": {-1: 0, 98: 1},
                   "手机没有信号": {-1: 0},
                   "有信号无法拨通": {-1: 0, 2: 1},
                   "通话过程中突然中断": {-1: 0, 3: 1},
                   "通话中有杂音、听不清、断断续续": {-1: 0, 4: 1},
                   "串线": {-1: 0, 5: 1},
                   "通话过程中一方听不见": {-1: 0, 6: 1},
                   "其他，请注明.1": {-1: 0, 98: 1},
                   "是否投诉": {'是': 1, '否': 0},
                   "是否关怀用户": {'是': 1, '否': 0},
                   "是否4G网络客户（本地剔除物联网）": {'是': 1, "否": 0},
                   "是否5G网络客户": {'是': 1, "否": 0},
                   "客户星级标识": {'未评级': 0, '准星': 1, '一星': 2, '二星': 3, '三星': 4, '银卡': 5, '金卡': 6,
                                    '白金卡': 7, '钻石卡': 8},
                   "终端品牌": {'苹果': 22, '华为': 11, '小米科技': 14,
                                '步步高': 18, '欧珀': 17, '三星': 4,
                                'realme': 1, '0': 0, '万普拉斯': 3,
                                '锤子': 24, '万普': 8, '中邮通信': 21,
                                '索尼爱立信': 6, '亿城': 6, '宇龙': 6,
                                '中国移动': 7, '中兴': 10, '黑鲨': 25,
                                '海信': 16, '摩托罗拉': 9, '诺基亚': 12,
                                '奇酷': 13}
                   }, inplace=True)
dataThree

# In[141]:


dataThree['外省语音占比'] = dataThree['外省语音占比'].astype('float64')
dataThree['外省流量占比'] = dataThree['外省流量占比'].astype('float64')
dataThree['是否4G网络客户（本地剔除物联网）'] = dataThree['是否4G网络客户（本地剔除物联网）'].astype('int64')
dataThree['4\\5G用户'] = dataThree['4\\5G用户'].astype(str)
dataThree

# In[142]:


le = sp.LabelEncoder()

FourFiveUser = le.fit_transform(dataThree["4\\5G用户"])
dataThree["4\\5G用户"] = pd.DataFrame(FourFiveUser)
dataThree

# In[143]:


dataThree['是否5G网络客户'] = dataThree['是否5G网络客户'].astype('int64')
dataThree['客户星级标识'] = dataThree['客户星级标识'].astype('int64')
dataThree['终端品牌'] = dataThree['终端品牌'].astype('int32')
dataThree

# In[144]:


dataThreeStandardTransform = dataThree[['脱网次数', 'mos质差次数', '未接通掉话次数', '4\\5G用户',
                                        '套外流量（MB）', '套外流量费（元）', '语音通话-时长（分钟）', '省际漫游-时长（分钟）',
                                        '终端品牌', '当月ARPU', '当月MOU',
                                        '前3月ARPU', '前3月MOU', 'GPRS总流量（KB）', 'GPRS-国内漫游-流量（KB）',
                                        '客户星级标识']]
dataThreeStandardTransformScaler = sp.StandardScaler()
dataThreeStandardTransformScaler = dataThreeStandardTransformScaler.fit(dataThreeStandardTransform)
dataThreeStandardTransform = dataThreeStandardTransformScaler.transform(dataThreeStandardTransform)
dataThreeStandardTransform = pd.DataFrame(dataThreeStandardTransform)
dataThreeStandardTransform.columns = ['脱网次数', 'mos质差次数', '未接通掉话次数', '4\\5G用户',
                                      '套外流量（MB）', '套外流量费（元）', '语音通话-时长（分钟）', '省际漫游-时长（分钟）',
                                      '终端品牌', '当月ARPU', '当月MOU',
                                      '前3月ARPU', '前3月MOU', 'GPRS总流量（KB）', 'GPRS-国内漫游-流量（KB）',
                                      '客户星级标识']
dataThreeStandardTransform

# In[145]:


dataThreeLeave = dataThree.loc[:, ~dataThree.columns.isin(['脱网次数', 'mos质差次数', '未接通掉话次数', '4\\5G用户',
                                                           '套外流量（MB）', '套外流量费（元）', '语音通话-时长（分钟）',
                                                           '省际漫游-时长（分钟）',
                                                           '终端品牌', '当月ARPU', '当月MOU',
                                                           '前3月ARPU', '前3月MOU', 'GPRS总流量（KB）',
                                                           'GPRS-国内漫游-流量（KB）',
                                                           '客户星级标识'])]
dataThreeNewStandard = pd.concat([dataThreeLeave, dataThreeStandardTransform], axis=1)
dataThreeNewStandard.columns = ['是否遇到过网络问题', '居民小区', '办公室', '高校',
                                '商业街', '地铁', '农村', '高铁',
                                '其他，请注明', '手机没有信号', '有信号无法拨通', '通话过程中突然中断',
                                '通话中有杂音、听不清、断断续续', '串线', '通话过程中一方听不见', '其他，请注明.1',
                                '是否投诉', '是否关怀用户', '是否4G网络客户（本地剔除物联网）', '外省语音占比',
                                '外省流量占比', '是否5G网络客户', '脱网次数', 'mos质差次数',
                                '未接通掉话次数', '4\\5G用户', '套外流量（MB）', '套外流量费（元）',
                                '语音通话-时长（分钟）', '省际漫游-时长（分钟）', '终端品牌',
                                '当月ARPU', '当月MOU', '前3月ARPU', '前3月MOU',
                                'GPRS总流量（KB）', 'GPRS-国内漫游-流量（KB）', '客户星级标识']
dataThreeNewStandard

# In[146]:


dataOneNewStandard

# ### 预测语音业务评分
# 需要注意到在所有预测结果上加上1，由于之前将评分编码为[0,9]，这里需要再映射回[1,10]

# In[147]:


Xpre = dataThreeNewStandard

# #### 语音通话整体满意度

# In[148]:


FirstPre = FirstModel.predict(Xpre)
FirstPre

# #### 网络覆盖与信号强度

# In[149]:


SecondPre = SecondModel.predict(Xpre)
SecondPre

# #### 语音通话清晰度

# In[150]:


ThirdPre = ThirdModel.predict(Xpre)
ThirdPre

# #### 语音通话稳定性

# In[151]:


FourthPre = FourthModel.predict(Xpre)
FourthPre

# ## 模型效果分析

# ### 混淆矩阵热力图

# #### 模型一

# In[152]:


from yellowbrick.classifier import ConfusionMatrix

classes = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
confusion_matrix = ConfusionMatrix(FirstModel, classes=classes, cmap='BuGn')
confusion_matrix.fit(XdataOneFirst_train, ydataOneFirst_train)
confusion_matrix.score(XdataOneFirst_test, ydataOneFirst_test)
plt.xticks(font='Times New Roman')
plt.yticks(font='Times New Roman')
confusion_matrix.show(outpath='figuresOne\\[附件1]模型一混淆矩阵热力图.pdf')

# #### 模型二

# In[153]:


from yellowbrick.classifier import ConfusionMatrix

classes = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
confusion_matrix = ConfusionMatrix(SecondModel, classes=classes, cmap='BuGn')
confusion_matrix.fit(XdataOneSecond_train, ydataOneSecond_train)
confusion_matrix.score(XdataOneSecond_test, ydataOneSecond_test)
plt.xticks(font='Times New Roman')
plt.yticks(font='Times New Roman')
confusion_matrix.show(outpath='figuresOne\\[附件1]模型二混淆矩阵热力图.pdf')

# #### 模型三

# In[154]:


from yellowbrick.classifier import ConfusionMatrix

classes = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
confusion_matrix = ConfusionMatrix(ThirdModel, classes=classes, cmap='BuGn')
confusion_matrix.fit(XdataOneThird_train, ydataOneThird_train)
confusion_matrix.score(XdataOneThird_test, ydataOneThird_test)
plt.xticks(font='Times New Roman')
plt.yticks(font='Times New Roman')
confusion_matrix.show(outpath='figuresOne\\[附件1]模型三混淆矩阵热力图.pdf')

# #### 模型四

# In[155]:


from yellowbrick.classifier import ConfusionMatrix

classes = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
confusion_matrix = ConfusionMatrix(FourthModel, classes=classes, cmap='BuGn')
confusion_matrix.fit(XdataOneFourth_train, ydataOneFourth_train)
confusion_matrix.score(XdataOneFourth_test, ydataOneFourth_test)
plt.xticks(font='Times New Roman')
plt.yticks(font='Times New Roman')
confusion_matrix.show(outpath='figuresOne\\[附件1]模型四混淆矩阵热力图.pdf')

# ### 分类报告

# #### 模型一

# In[156]:


from yellowbrick.classifier import ClassificationReport

classes = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
visualizer = ClassificationReport(FirstModel, classes=classes, support=True, cmap='Blues')
visualizer.fit(XdataOneFirst_train, ydataOneFirst_train)
visualizer.score(XdataOneFirst_test, ydataOneFirst_test)
plt.xticks(font='Times New Roman')
plt.yticks(font='Times New Roman')
visualizer.show(outpath='figuresOne\\[附件1]模型一分类报告.pdf')

# #### 模型二

# In[157]:


from yellowbrick.classifier import ClassificationReport

classes = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
visualizer = ClassificationReport(SecondModel, classes=classes, support=True, cmap='Blues')
visualizer.fit(XdataOneSecond_train, ydataOneSecond_train)
visualizer.score(XdataOneSecond_test, ydataOneSecond_test)
plt.xticks(font='Times New Roman')
plt.yticks(font='Times New Roman')
visualizer.show(outpath='figuresOne\\[附件1]模型二分类报告.pdf')

# #### 模型三

# In[158]:


from yellowbrick.classifier import ClassificationReport

classes = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
visualizer = ClassificationReport(ThirdModel, classes=classes, support=True, cmap='Blues')
visualizer.fit(XdataOneThird_train, ydataOneThird_train)
visualizer.score(XdataOneThird_test, ydataOneThird_test)
plt.xticks(font='Times New Roman')
plt.yticks(font='Times New Roman')
visualizer.show(outpath='figuresOne\\[附件1]模型三分类报告.pdf')

# #### 模型四

# In[159]:


from yellowbrick.classifier import ClassificationReport

classes = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
visualizer = ClassificationReport(FourthModel, classes=classes, support=True, cmap='Blues')
visualizer.fit(XdataOneFourth_train, ydataOneFourth_train)
visualizer.score(XdataOneFourth_test, ydataOneFourth_test)
plt.xticks(font='Times New Roman')
plt.yticks(font='Times New Roman')
visualizer.show(outpath='figuresOne\\[附件1]模型四分类报告.pdf')

# ### ROC AUC曲线

# #### 模型一

# In[160]:


from yellowbrick.classifier import ROCAUC

visualizer = ROCAUC(FirstModel)
visualizer.fit(XdataOneFirst_train, ydataOneFirst_train)
visualizer.score(XdataOneFirst_test, ydataOneFirst_test)
plt.xticks(font='Times New Roman')
plt.yticks(font='Times New Roman')
visualizer.show(outpath='figuresOne\\[附件1]模型一ROCAUC.pdf')

# #### 模型二

# In[161]:


from yellowbrick.classifier import ROCAUC

visualizer = ROCAUC(SecondModel)
visualizer.fit(XdataOneSecond_train, ydataOneSecond_train)
visualizer.score(XdataOneSecond_test, ydataOneSecond_test)
plt.xticks(font='Times New Roman')
plt.yticks(font='Times New Roman')
visualizer.show(outpath='figuresOne\\[附件1]模型二ROCAUC.pdf')

# #### 模型三

# In[162]:


from yellowbrick.classifier import ROCAUC

visualizer = ROCAUC(ThirdModel)
visualizer.fit(XdataOneThird_train, ydataOneThird_train)
visualizer.score(XdataOneThird_test, ydataOneThird_test)
plt.xticks(font='Times New Roman')
plt.yticks(font='Times New Roman')
visualizer.show(outpath='figuresOne\\[附件1]模型三ROCAUC.pdf')

# #### 模型四

# In[163]:


from yellowbrick.classifier import ROCAUC

visualizer = ROCAUC(FourthModel)
visualizer.fit(XdataOneFourth_train, ydataOneFourth_train)
visualizer.score(XdataOneFourth_test, ydataOneFourth_test)
plt.xticks(font='Times New Roman')
plt.yticks(font='Times New Roman')
visualizer.show(outpath='figuresOne\\[附件1]模型四ROCAUC.pdf')

# ### 平均绝对误差，均方误差

# #### 模型一

# In[164]:


print(f'模型一平均绝对误差：'
      f'{mean_absolute_error(ydataOneFirst_test, FirstModel.predict(XdataOneFirst_test), sample_weight=None, multioutput="uniform_average")}\n'
      f'模型一均方误差：'
      f'{mean_squared_error(ydataOneFirst_test, FirstModel.predict(XdataOneFirst_test), sample_weight=None, multioutput="uniform_average")}')
print(f'模型一中RF平均绝对误差：'
      f'{mean_absolute_error(ydataOneFirst_test, RandomForestFirst.predict(XdataOneFirst_test), sample_weight=None, multioutput="uniform_average")}\n'
      f'模型一中RF均方误差：'
      f'{mean_squared_error(ydataOneFirst_test, RandomForestFirst.predict(XdataOneFirst_test), sample_weight=None, multioutput="uniform_average")}')
print(f'模型一中XGBoost平均绝对误差：'
      f'{mean_absolute_error(ydataOneFirst_test, XGBFirst.predict(XdataOneFirst_test), sample_weight=None, multioutput="uniform_average")}\n'
      f'模型一中XGBoost均方误差：'
      f'{mean_squared_error(ydataOneFirst_test, XGBFirst.predict(XdataOneFirst_test), sample_weight=None, multioutput="uniform_average")}')
print(f'模型一中KNN平均绝对误差：'
      f'{mean_absolute_error(ydataOneFirst_test, KNNFirst_new.predict(XdataOneFirst_test), sample_weight=None, multioutput="uniform_average")}\n'
      f'模型一中KNN均方误差：'
      f'{mean_squared_error(ydataOneFirst_test, KNNFirst_new.predict(XdataOneFirst_test), sample_weight=None, multioutput="uniform_average")}')
print(f'模型一中SVM平均绝对误差：'
      f'{mean_absolute_error(ydataOneFirst_test, SVMFirst.predict(XdataOneFirst_test), sample_weight=None, multioutput="uniform_average")}\n'
      f'模型一中SVM均方误差：'
      f'{mean_squared_error(ydataOneFirst_test, SVMFirst.predict(XdataOneFirst_test), sample_weight=None, multioutput="uniform_average")}')
print(f'模型一中LightGBM平均绝对误差：'
      f'{mean_absolute_error(ydataOneFirst_test, LightgbmFirst.predict(XdataOneFirst_test), sample_weight=None, multioutput="uniform_average")}\n'
      f'模型一中LightGBM均方误差：'
      f'{mean_squared_error(ydataOneFirst_test, LightgbmFirst.predict(XdataOneFirst_test), sample_weight=None, multioutput="uniform_average")}')
print(f'模型一中LR平均绝对误差：'
      f'{mean_absolute_error(ydataOneFirst_test, LogisticRegressionFirst.predict(XdataOneFirst_test), sample_weight=None, multioutput="uniform_average")}\n'
      f'模型一中LR均方误差：'
      f'{mean_squared_error(ydataOneFirst_test, LogisticRegressionFirst.predict(XdataOneFirst_test), sample_weight=None, multioutput="uniform_average")}')

# #### 模型二

# In[165]:


print(f'模型二平均绝对误差：'
      f'{mean_absolute_error(ydataOneSecond_test, SecondModel.predict(XdataOneSecond_test), sample_weight=None, multioutput="uniform_average")}\n'
      f'模型二均方误差：'
      f'{mean_squared_error(ydataOneSecond_test, SecondModel.predict(XdataOneSecond_test), sample_weight=None, multioutput="uniform_average")}')
print(f'模型二中RF平均绝对误差：'
      f'{mean_absolute_error(ydataOneSecond_test, RandomForestSecond.predict(XdataOneSecond_test), sample_weight=None, multioutput="uniform_average")}\n'
      f'模型二中RF均方误差：'
      f'{mean_squared_error(ydataOneSecond_test, RandomForestSecond.predict(XdataOneSecond_test), sample_weight=None, multioutput="uniform_average")}')
print(f'模型二中XGBoost平均绝对误差：'
      f'{mean_absolute_error(ydataOneSecond_test, XGBSecond.predict(XdataOneSecond_test), sample_weight=None, multioutput="uniform_average")}\n'
      f'模型二中XGBoost均方误差：'
      f'{mean_squared_error(ydataOneSecond_test, XGBSecond.predict(XdataOneSecond_test), sample_weight=None, multioutput="uniform_average")}')
print(f'模型二中KNN平均绝对误差：'
      f'{mean_absolute_error(ydataOneSecond_test, KNNSecond_new.predict(XdataOneSecond_test), sample_weight=None, multioutput="uniform_average")}\n'
      f'模型二中KNN均方误差：'
      f'{mean_squared_error(ydataOneSecond_test, KNNSecond_new.predict(XdataOneSecond_test), sample_weight=None, multioutput="uniform_average")}')
print(f'模型二中SVM平均绝对误差：'
      f'{mean_absolute_error(ydataOneSecond_test, SVMSecond.predict(XdataOneSecond_test), sample_weight=None, multioutput="uniform_average")}\n'
      f'模型二中SVM均方误差：'
      f'{mean_squared_error(ydataOneSecond_test, SVMSecond.predict(XdataOneSecond_test), sample_weight=None, multioutput="uniform_average")}')
print(f'模型二中LightGBM平均绝对误差：'
      f'{mean_absolute_error(ydataOneSecond_test, LightgbmSecond.predict(XdataOneSecond_test), sample_weight=None, multioutput="uniform_average")}\n'
      f'模型二中LightGBM均方误差：'
      f'{mean_squared_error(ydataOneSecond_test, LightgbmSecond.predict(XdataOneSecond_test), sample_weight=None, multioutput="uniform_average")}')
print(f'模型二中LR平均绝对误差：'
      f'{mean_absolute_error(ydataOneSecond_test, LogisticRegressionSecond.predict(XdataOneSecond_test), sample_weight=None, multioutput="uniform_average")}\n'
      f'模型二中LR均方误差：'
      f'{mean_squared_error(ydataOneSecond_test, LogisticRegressionSecond.predict(XdataOneSecond_test), sample_weight=None, multioutput="uniform_average")}')

# #### 模型三

# In[166]:


print(f'模型三平均绝对误差：'
      f'{mean_absolute_error(ydataOneThird_test, ThirdModel.predict(XdataOneThird_test), sample_weight=None, multioutput="uniform_average")}\n'
      f'模型三均方误差：'
      f'{mean_squared_error(ydataOneThird_test, ThirdModel.predict(XdataOneThird_test), sample_weight=None, multioutput="uniform_average")}')
print(f'模型三中RF平均绝对误差：'
      f'{mean_absolute_error(ydataOneThird_test, RandomForestThird.predict(XdataOneThird_test), sample_weight=None, multioutput="uniform_average")}\n'
      f'模型三中RF均方误差：'
      f'{mean_squared_error(ydataOneThird_test, RandomForestThird.predict(XdataOneThird_test), sample_weight=None, multioutput="uniform_average")}')
print(f'模型三中XGBoost平均绝对误差：'
      f'{mean_absolute_error(ydataOneThird_test, XGBThird.predict(XdataOneThird_test), sample_weight=None, multioutput="uniform_average")}\n'
      f'模型三中XGBoost均方误差：'
      f'{mean_squared_error(ydataOneThird_test, XGBThird.predict(XdataOneThird_test), sample_weight=None, multioutput="uniform_average")}')
print(f'模型三中KNN平均绝对误差：'
      f'{mean_absolute_error(ydataOneThird_test, KNNThird_new.predict(XdataOneThird_test), sample_weight=None, multioutput="uniform_average")}\n'
      f'模型三中KNN均方误差：'
      f'{mean_squared_error(ydataOneThird_test, KNNThird_new.predict(XdataOneThird_test), sample_weight=None, multioutput="uniform_average")}')
print(f'模型三中SVM平均绝对误差：'
      f'{mean_absolute_error(ydataOneThird_test, SVMThird.predict(XdataOneThird_test), sample_weight=None, multioutput="uniform_average")}\n'
      f'模型三中SVM均方误差：'
      f'{mean_squared_error(ydataOneThird_test, SVMThird.predict(XdataOneThird_test), sample_weight=None, multioutput="uniform_average")}')
print(f'模型三中LightGBM平均绝对误差：'
      f'{mean_absolute_error(ydataOneThird_test, LightgbmThird.predict(XdataOneThird_test), sample_weight=None, multioutput="uniform_average")}\n'
      f'模型三中LightGBM均方误差：'
      f'{mean_squared_error(ydataOneThird_test, LightgbmThird.predict(XdataOneThird_test), sample_weight=None, multioutput="uniform_average")}')
print(f'模型三中LR平均绝对误差：'
      f'{mean_absolute_error(ydataOneThird_test, LogisticRegressionThird.predict(XdataOneThird_test), sample_weight=None, multioutput="uniform_average")}\n'
      f'模型三中LR均方误差：'
      f'{mean_squared_error(ydataOneThird_test, LogisticRegressionThird.predict(XdataOneThird_test), sample_weight=None, multioutput="uniform_average")}')

# #### 模型四

# In[167]:


print(f'模型四平均绝对误差：'
      f'{mean_absolute_error(ydataOneFourth_test, FourthModel.predict(XdataOneFourth_test), sample_weight=None, multioutput="uniform_average")}\n'
      f'模型四均方误差：'
      f'{mean_squared_error(ydataOneFourth_test, FourthModel.predict(XdataOneFourth_test), sample_weight=None, multioutput="uniform_average")}')
print(f'模型四中RF平均绝对误差：'
      f'{mean_absolute_error(ydataOneFourth_test, RandomForestFourth.predict(XdataOneFourth_test), sample_weight=None, multioutput="uniform_average")}\n'
      f'模型四中RF均方误差：'
      f'{mean_squared_error(ydataOneFourth_test, RandomForestFourth.predict(XdataOneFourth_test), sample_weight=None, multioutput="uniform_average")}')
print(f'模型四中XGBoost平均绝对误差：'
      f'{mean_absolute_error(ydataOneFourth_test, XGBFourth.predict(XdataOneFourth_test), sample_weight=None, multioutput="uniform_average")}\n'
      f'模型四中XGBoost均方误差：'
      f'{mean_squared_error(ydataOneFourth_test, XGBFourth.predict(XdataOneFourth_test), sample_weight=None, multioutput="uniform_average")}')
print(f'模型四中KNN平均绝对误差：'
      f'{mean_absolute_error(ydataOneFourth_test, KNNFourth_new.predict(XdataOneFourth_test), sample_weight=None, multioutput="uniform_average")}\n'
      f'模型四中KNN均方误差：'
      f'{mean_squared_error(ydataOneFourth_test, KNNFourth_new.predict(XdataOneFourth_test), sample_weight=None, multioutput="uniform_average")}')
print(f'模型四中SVM平均绝对误差：'
      f'{mean_absolute_error(ydataOneFourth_test, SVMFourth.predict(XdataOneFourth_test), sample_weight=None, multioutput="uniform_average")}\n'
      f'模型四中SVM均方误差：'
      f'{mean_squared_error(ydataOneFourth_test, SVMFourth.predict(XdataOneFourth_test), sample_weight=None, multioutput="uniform_average")}')
print(f'模型四中LightGBM平均绝对误差：'
      f'{mean_absolute_error(ydataOneFourth_test, LightgbmFourth.predict(XdataOneFourth_test), sample_weight=None, multioutput="uniform_average")}\n'
      f'模型四中LightGBM均方误差：'
      f'{mean_squared_error(ydataOneFourth_test, LightgbmFourth.predict(XdataOneFourth_test), sample_weight=None, multioutput="uniform_average")}')
print(f'模型四中LR平均绝对误差：'
      f'{mean_absolute_error(ydataOneFourth_test, LogisticRegressionFourth.predict(XdataOneFourth_test), sample_weight=None, multioutput="uniform_average")}\n'
      f'模型四中LR均方误差：'
      f'{mean_squared_error(ydataOneFourth_test, LogisticRegressionFourth.predict(XdataOneFourth_test), sample_weight=None, multioutput="uniform_average")}')

# ## 高频词汇云图

# In[168]:


import jieba
import wordcloud
from matplotlib.image import imread

jieba.setLogLevel(jieba.logging.INFO)
report = open('语音业务词云.txt', 'r', encoding='utf-8').read()
words = jieba.lcut(report)
txt = []
for word in words:
    if len(word) == 1:
        continue
    else:
        txt.append(word)
a = ' '.join(txt)
bg = imread("bg.jpg")
w = wordcloud.WordCloud(background_color="white", font_path="msyh.ttc", mask=bg)
w.generate(a)
w.to_file("figuresOne\\wordcloudF.png")
