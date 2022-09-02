import sys

sys.path.append("../..")

import streamlit as st
from transformers import BertTokenizerFast
from torchblocks.tasks.ner import get_auto_ner_model
from torchblocks.tasks.ner import NERPredictor, EnsembleNERPredictor, PromptNERPredictor, LearNERPredictor, \
    W2NERPredictor

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

# Using object notation
model_name = st.sidebar.radio(
    "模型框架",
    ("CRF", "SPAN", "TPLINKER", "GLOBAL-POINTER", "MRC", "LEAR", "W2NER", "ENSEMBLE")
)
st.sidebar.markdown('---')
max_seqlen = st.sidebar.number_input('句子最大长度', 0, 512, 512)
prob = st.sidebar.slider("阈值", min_value=0.0, max_value=1.0, value=0.5, step=0.01)


@st.cache(hash_funcs={BertTokenizerFast: id})
def load_tokenizer():
    return BertTokenizerFast.from_pretrained('hfl/chinese-roberta-wwm-ext', do_lower_case=True)


PREDICTOR_MAP = {"lear": LearNERPredictor, "w2ner": W2NERPredictor, "mrc": PromptNERPredictor}


def load_auto_predictor(model_name):
    predictor_class = PREDICTOR_MAP.get(model_name, NERPredictor)

    @st.cache(hash_funcs={predictor_class: id})
    def load_predictor():
        tokenizer = load_tokenizer()
        model_name_or_path = MODEL_PATH_MAP[model_name]
        model = get_auto_ner_model(model_name=model_name, model_type="bert")

        if model_name in ["lear", "mrc"]:
            return predictor_class(schema2prompt, model=model, model_name_or_path=model_name_or_path,
                                   tokenizer=tokenizer)
        else:
            return predictor_class(model, model_name_or_path, tokenizer)

    return load_predictor()


@st.cache(hash_funcs={EnsembleNERPredictor: id})
def load_ensemble_predictor():
    predictors = [load_auto_predictor(name) for name in
                  ["crf", "span", "global-pointer", "tplinker", "mrc", "lear", "w2ner"]]
    return EnsembleNERPredictor(predictors)


html_tmp = """
    <div>
    <h1 style="text-align:center;">中文医学文本命名实体识别</h1>
    </div>
"""
st.markdown(html_tmp, unsafe_allow_html=True)
st.markdown('---')

st.subheader("输入文本📖")
text = st.text_area("请输入待抽取的句子（支持多个句子输入）：")

if model_name == "ENSEMBLE":
    ner = load_ensemble_predictor()
else:
    ner = load_auto_predictor(model_name.lower())

if st.button('运行🚀'):
    text = text.split('\n')
    if model_name == "ENSEMBLE":
        out = ner.predict(text, max_length=max_seqlen, threshold=prob)
    else:
        out = ner.predict(text, max_length=max_seqlen)
    st.json(out)
    st.stop()
