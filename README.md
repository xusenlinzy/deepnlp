# deepnlp

基于 `pytorch` 框架实现 `nlp` 各类任务的解决方案

# nlp任务

## 文本分类

**`STEP 1`:** 将训练数据转换为如下的 `json` 格式
```
{
  "text": "以色列大规模空袭开始！伊朗多个军事目标遭遇打击，誓言对等反击",
  "label": "news_military"
}
```

**`STEP 2`:** 选择文本分类模型，如 `BertForSequenceClassification`

