# deepnlp

基于 `pytorch` 框架实现 `nlp` 各类任务的解决方案

# nlp任务

## 文本分类

### 1. 将训练数据转换为如下的 `json` 格式

```json
{
  "text": "以色列大规模空袭开始！伊朗多个军事目标遭遇打击，誓言对等反击",
  "label": "news_military"
}
```

**训练数据包含两个文件**：`train.json` 和 `dev.json`。

### 2. 通过运行 [`bash`](./examples/tc/run.sh) 命令进行模型微调

支持的模型

  - ⭐ [**`fc`**](./torchblocks/tasks/tc/sequence_classification.py)：`BERT` 后面接全连接层
  - ⭐ [**`mdp`**](./torchblocks/tasks/tc/sequence_classification_mdp.py)：`BERT` 后面使用 [`MultiSampleDropout`](https://arxiv.org/abs/1905.09788)
  - ⭐ [**`rdrop`**](./torchblocks/tasks/tc/sequence_classification_rdrop.py)：`BERT` 后面接全连接层并使用 [`R-Drop`](https://github.com/dropreg/R-Drop) 正则化

### 3. 模型预测

```python
from torchblocks.tasks.tc import TextClassificationPipeline

pipline = TextClassificationPipeline("my_bert_model_path", model_name="fc", model_type="bert")
text = "以色列大规模空袭开始！伊朗多个军事目标遭遇打击，誓言对等反击"
print(pipline(text))
```

## 命名实体识别

### 1. 将训练数据转换为如下的 `json` 格式

```json
{
  "text": "结果上周六他们主场0：3惨败给了中游球队瓦拉多利德，近7个多月以来西甲首次输球。", 
  "entities": [
    {
      "id": 0, 
      "entity": "瓦拉多利德", 
      "start_offset": 20, 
      "end_offset": 24, 
      "label": "organization"
    }, 
    {
      "id": 1, 
      "entity": "西甲", 
      "start_offset": 33, 
      "end_offset": 34, 
      "label": "organization"
    }
  ]
}
```

**训练数据包含两个文件**：`train.json` 和 `dev.json`。

### 2. 通过运行 [`bash`](./examples/ner/ner.sh) 命令进行模型微调

支持的模型

  - ⭐ [**`softmax`**](./torchblocks/tasks/ner/softmax.py)：`BERT` 后面接全连接层并使用 `BIO` 解码
  - ⭐ [**`crf`**](./torchblocks/tasks/ner/crf.py)：`BERT` 后面接全连接层和条件随机场，并使用 `BIO` 解码
  - ⭐ [**`span`**](./torchblocks/tasks/ner/span.py)：`BERT` 后面使用两个指针网络预测实体起始位置
  - ⭐ [**`global-pointer`**](./torchblocks/tasks/ner/global_pointer.py)：[GlobalPointer：用统一的方式处理嵌套和非嵌套NER](https://spaces.ac.cn/archives/8373)
  - ⭐ [**`efficient_global_pointer`**](./torchblocks/tasks/ner/global_pointer.py)：[Efficient GlobalPointer：少点参数，多点效果](https://spaces.ac.cn/archives/8877)
  - ⭐ [**`mrc`**](./torchblocks/tasks/ner/mrc.py)：[A Unified MRC Framework for Named Entity Recognition.](https://aclanthology.org/2020.acl-main.519.pdf)
  - ⭐ [**`tplinker`**](./torchblocks/tasks/ner/tplinker.py)：[TPLinker: Single-stage Joint Extraction of Entities and Relations Through Token Pair Linking.](https://aclanthology.org/2020.coling-main.138.pdf)
  - ⭐ [**`lear`**](./torchblocks/tasks/ner/lear.py)：[Enhanced Language Representation with Label Knowledge for Span Extraction.](https://aclanthology.org/2021.emnlp-main.379.pdf)
  - ⭐ [**`w2ner`**](./torchblocks/tasks/ner/w2ner.py)：[Unified Named Entity Recognition as Word-Word Relation Classification.](https://arxiv.org/pdf/2112.10070.pdf)
  - ⭐ [**`cascade-crf`**](./torchblocks/tasks/ner/crf.py)：先预测实体再预测实体类型

### 3. 模型预测

```python
from pprint import pprint
from torchblocks.tasks.ner import NERPipeline

pipline = NERPipeline("my_bert_model_path", model_name="crf", model_type="bert")
text = "结果上周六他们主场0：3惨败给了中游球队瓦拉多利德，近7个多月以来西甲首次输球。"
pprint(pipline(text))
```

## 实体关系抽取

### 1. 将训练数据转换为如下的 `json` 格式

```json
{
  "text": "查尔斯·阿兰基斯（Charles Aránguiz），1989年4月17日出生于智利圣地亚哥，智利职业足球运动员，司职中场，效力于德国足球甲级联赛勒沃库森足球俱乐部", 
  "spo_list": [
    {
      "predicate": "出生地", 
      "object_type": "地点", 
      "subject_type": "人物", 
      "object": "圣地亚哥", 
      "subject": "查尔斯·阿兰基斯"
    }, 
    {
      "predicate": "出生日期", 
      "object_type": "Date", 
      "subject_type": "人物", 
      "object": "1989年4月17日",
      "subject": "查尔斯·阿兰基斯"
    }
  ]
}
```

**训练数据包含两个文件**：`train.json` 和 `dev.json`。

### 2. 通过运行 [`bash`](./examples/re/re.sh) 命令进行模型微调

支持的模型

  - ⭐ [**`casrel`**](./torchblocks/tasks/ere/casrel.py)：[A Novel Cascade Binary Tagging Framework for Relational Triple Extraction.](https://aclanthology.org/2020.acl-main.136.pdf)
  - ⭐ [**`tplinker`**](./torchblocks/tasks/ere/tplinker.py)：[TPLinker: Single-stage Joint Extraction of Entities and Relations Through Token Pair Linking.](https://aclanthology.org/2020.coling-main.138.pdf)
  - ⭐ [**`spn`**](./torchblocks/tasks/ere/spn.py)：[Joint Entity and Relation Extraction with Set Prediction Networks.](http://xxx.itp.ac.cn/pdf/2011.01675v2)
  - ⭐ [**`prgc`**](./torchblocks/tasks/ere/prgc.py)：[PRGC: Potential Relation and Global Correspondence Based Joint Relational Triple Extraction.](https://aclanthology.org/2021.acl-long.486.pdf)
  - ⭐ [**`pfn`**](./torchblocks/tasks/ere/pfn.py)：[A Partition Filter Network for Joint Entity and Relation Extraction.](https://aclanthology.org/2021.emnlp-main.17.pdf)
  - ⭐ [**`grte`**](./torchblocks/tasks/ere/grte.py)：[A Novel Global Feature-Oriented Relational Triple Extraction Model based on Table Filling.](https://aclanthology.org/2021.emnlp-main.208.pdf)
  - ⭐ [**`gplinker`**](./torchblocks/tasks/ere/gplinker.py)：[GPLinker：基于GlobalPointer的实体关系联合抽取](https://kexue.fm/archives/8888)

### 3. 模型预测

```python
from pprint import pprint
from torchblocks.tasks.ere import REPipeline

pipline = REPipeline("my_bert_model_path", model_name="gplinker", model_type="bert")
text = "查尔斯·阿兰基斯（Charles Aránguiz），1989年4月17日出生于智利圣地亚哥，智利职业足球运动员，司职中场，效力于德国足球甲级联赛勒沃库森足球俱乐部。"
pprint(pipline(text))
```
