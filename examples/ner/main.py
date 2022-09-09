import sys

sys.path.append("../..")

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Union
import streamlit as st
from torchblocks.tasks.ner import NERPipeline, EnsembleNERPredictor

# 应用实例化
app = FastAPI()

MODEL_PATH_MAP = {
    "tplinker": '/home/xusenlin/nlp/deepnlp/examples/ner/outputs/cmeee/tplinkerplus/cmeee-tplinkerplus_bert_v0/checkpoint-eval_f1_micro-best',
    "crf": '/home/xusenlin/nlp/deepnlp/examples/ner/outputs/cmeee/crf/cmeee-crf_bert_v0/checkpoint-eval_f1_micro-best',
    "span": "/home/xusenlin/nlp/deepnlp/examples/ner/outputs/cmeee/span/cmeee-span_bert_v0/checkpoint-eval_f1_micro-best",
    "global-pointer": "/home/xusenlin/nlp/deepnlp/examples/ner/outputs/cmeee/efficient-global-pointer/cmeee-efficient-global-pointer_bert_v0/checkpoint-eval_f1_micro-best",
    "w2ner": "/home/xusenlin/nlp/deepnlp/examples/ner/outputs/cmeee/w2ner/cmeee-w2ner_bert_v0/checkpoint-eval_f1_micro-best",
    "lear": "/home/xusenlin/nlp/deepnlp/examples/ner/outputs/cmeee/lear/cmeee-lear_bert_v0/checkpoint-eval_f1_micro-best",
    "mrc": "/home/xusenlin/nlp/deepnlp/examples/ner/outputs/cmeee/mrc-prompt/cmeee-mrc_bert_v0/checkpoint-eval_f1_micro-best",
}

schema2prompt = {
    "dis": "疾病，主要包括疾病、中毒或受伤和器官或细胞受损",
    "sym": "临床表现，主要包括症状和体征",
    "pro": "医疗程序，主要包括检查程序、治疗或预防程序",
    "equ": "医疗设备，主要包括检查设备和治疗设备",
    "dru": "药物，是用以预防、治疗及诊断疾病的物质",
    "ite": "医学检验项目，是取自人体的材料进行血液学、细胞学等方面的检验",
    "bod": "身体，主要包括身体物质和身体部位",
    "dep": "部门科室，医院的各职能科室",
    "mic": "微生物类，一般是指细菌、病毒、真菌、支原体、衣原体、螺旋体等八类微生物"
}

LABEL_MAP = {
    "dis": "疾病",
    "sym": "症状、临床表现",
    "pro": "检查、治疗或预防程序",
    "equ": "检查设备和治疗设备",
    "dru": "药物",
    "ite": "医学检验项目",
    "bod": "身体物质和身体部位",
    "dep": "部门科室",
    "mic": "微生物"
}


@st.cache(hash_funcs={NERPipeline: id})
def load_pipline(model_name="global-pointer", max_seq_len=512, split_sentence=False, batch_size=256):
    model_name_or_path = MODEL_PATH_MAP[model_name.lower()]
    return NERPipeline(model_name_or_path, model_name.lower(), model_type="bert", schema2prompt=schema2prompt, max_seq_len=max_seq_len,
                       split_sentence=split_sentence, batch_size=batch_size)


@st.cache(hash_funcs={EnsembleNERPredictor: id})
def load_ensemble_predictor():
    predictors = [load_pipline(name) for name in
                  ["crf", "span", "global-pointer", "tplinker", "mrc", "lear", "w2ner"]]
    return EnsembleNERPredictor(predictors)


# 定义数据格式
# 定义数据格式
class Data(BaseModel):
    input: Union[str, List[str]]  # 可输入一个句子或多个句子
    model_name: str = "global-pointer"
    max_seq_len: int = 512
    threshold: float = 0.5


# uie接口
@app.post('/ner')
def uie(data: Data):
    model_name = getattr(data, "model_name", "global-pointer")
    if model_name == "ensemble":
        ner = load_ensemble_predictor()
    else:
        ner = load_pipline(model_name.lower(), max_seq_len=data.max_seq_len, split_sentence=True)

    text = data.input
    if model_name == "ensemble":
        rlt = ner.predict(text, threshold=data.threshold)
    else:
        rlt = ner.predict(text)

    if isinstance(rlt, dict):
        res = {LABEL_MAP[key]: value for key, value in rlt.items()}
    else:
        res = []
        for r in rlt:
            dic = {LABEL_MAP[key]: value for key, value in r.items()}
            res.append(dic)

    return {'success': True, 'rlt': res}


if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8080)
