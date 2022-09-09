import sys

sys.path.append("../..")

import time
import json
import streamlit as st
import pandas as pd
import seaborn as sns
from torchblocks.tasks.ner import NERPipeline
from torchblocks.tasks.ner import EnsembleNERPredictor
from torchblocks.utils.app import visualize_ner, download_button, _max_width_, make_color_palette


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


def reformat_v1(rlt):
    rlt = {LABEL_MAP[_type]: [
        {"text": ent["text"], "start": ent["start"], "end": ent["end"],
            "probability": float(ent["probability"])}
        for ent in ents] for _type, ents in rlt.items()}
    return rlt


def reformat_v2(rlt):
    rlt = {LABEL_MAP[_type]: [
        {"text": ent["text"], "start": ent["start"], "end": ent["end"]}
        for ent in ents] for _type, ents in rlt.items()}
    return rlt


@st.cache
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv(index=False).encode('utf-8')


labels = LABEL_MAP.values()
colors = make_color_palette(labels)


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


# 设置网页信息 
st.set_page_config(page_title="NER Demo", page_icon="🚀", layout="wide")

_max_width_()

c30, c31, c32 = st.columns([2.5, 1, 3])

with c30:
    # st.image("logo.png", width=400)
    st.title("🔑 中文医学命名实体识别")
    st.header("")

with st.expander("ℹ️ - 关于此APP", expanded=True):
    st.write(
        """     
-   实现多种`NER`模型抽取中文医学文本中的实体。
-   包含7种`SOTA`模型以及额外的一个集成模型。
	    """
    )

    st.markdown("")

st.markdown("")
st.markdown("## 📌 输入")

with st.form(key="my_form"):
    ce, c1, _, c2, c3 = st.columns([0.07, 1, 0.07, 5, 0.07])

    with c1:
        model_name = st.radio(
            "选择预训练模型",
            ("CRF", "SPAN", "TPLINKER", "GLOBAL-POINTER", "MRC", "LEAR", "W2NER", "ENSEMBLE"),
            help="目前支持以上八个模型。",
        )

        max_seq_len = st.number_input(
            '句子最大长度',
            0,
            512,
            512,
            help="模型输入的最大文本长度，超过该长度则截断。",
        )

        prob = st.slider(
            "阈值",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.01,
            help="模型输出实体的阈值，当概率值大于该值则输出该实体，仅对于`ensemble`模型。",
        )

        split = st.checkbox(
            '截断句子',
            value=False,
            help="对于文本长度超过最大长度的句子，将句子切分成多个句子输入模型。",
        )

    with c2:
        text = st.text_area(
            "📖请输入待抽取的句子",
            height=400, )

        file_upload_exp = st.expander("上传文件")
        uploaded_file = file_upload_exp.file_uploader("Choose a file", type=".jsonl")
        submit_button = st.form_submit_button(label="✨ 运行")

if not submit_button:
    st.stop()

if model_name == "ENSEMBLE":
    ner = load_ensemble_predictor()
else:
    ner = load_pipline(model_name.lower(), max_seq_len=max_seq_len, split_sentence=split)

if uploaded_file is not None:
    data = pd.read_json(uploaded_file, lines=True, encoding='utf-8')
    texts = data.text.values

    bar = st.progress(0.0)
    res = []
    total_batch = len(texts) // 512 + (1 if len(texts) % 512 > 0 else 0)
    for i, batch_id in enumerate(range(total_batch)):
        text = texts[batch_id * 512: (batch_id + 1) * 512]
        if model_name == "ENSEMBLE":
            rlt = ner.predict(text, threshold=prob)
        else:
            rlt = ner(text)

        bar.progress((i + 1) / total_batch)
        res.extend(rlt)

        if i == 0:
            visualize_ner(text, rlt[:10], colors=colors)

    data["prediction"] = res
    data["prediction"] = data["prediction"].apply(reformat_v1 if model_name == "ENSEMBLE" else reformat_v2)
    data["prediction"] = data["prediction"].apply(lambda x: json.dumps(x, ensure_ascii=False))
    csv = convert_df(data)

    st.download_button(
        label="📥 Download (.csv)",
        data=csv,
        file_name='ner_predict.csv',
        mime='text/csv',
    )

    st.stop()

start = time.time()
if model_name == "ENSEMBLE":
    rlt = ner.predict(text.split(), threshold=prob)
else:
    rlt = ner(text.split())
running_time = time.time() - start

res = [reformat_v1(r) if model_name == "ENSEMBLE" else reformat_v2(r) for r in rlt]

st.markdown("## 🎈 结果展示")
st.header("")

cs, c1, c2, c3, cLast = st.columns([2, 1.5, 1.5, 1.5, 2])

with c1:
    CSVButton2 = download_button(res, "Data.csv", "📥 Download (.csv)")
with c2:
    CSVButton3 = download_button(res, "Data.txt", "📥 Download (.txt)")
with c3:
    CSVButton4 = download_button(res, "Data.json", "📥 Download (.json)")

c1, c2, c3 = st.columns([1, 3, 1])

with c2:
    st.info(f'运行时间：{int(running_time * 1000)} ms', icon="✅")
    visualize_ner(text, res, colors=colors)

    json_doc_exp = st.expander("JSON")
    json_doc_exp.json(res)

    dataframe_exp = st.expander("DATAFRAME")
    columns = ["text", "start", "end", "label", "probability"]
    for r in res:
        if model_name == "ENSEMBLE":
            data = [[t[columns[0]], t[columns[1]], t[columns[2]], k, t[columns[4]]] for k, v in r.items() for t in v]
            df = pd.DataFrame(data, columns=columns).sort_values(by="probability", ascending=False).reset_index(
                drop=True)
        else:
            data = [[t[columns[0]], t[columns[1]], t[columns[2]], k] for k, v in r.items() for t in v]
            df = pd.DataFrame(data, columns=columns[:4])
        df.index += 1

        if model_name == "ENSEMBLE":
            # Add styling
            cmGreen = sns.light_palette("green", as_cmap=True)
            cmRed = sns.light_palette("red", as_cmap=True)
            df = df.style.background_gradient(
                cmap=cmGreen,
                subset=[
                    "probability",
                ],
            )
            format_dictionary = {
                "probability": "{:.1%}",
            }

            df = df.format(format_dictionary)
        dataframe_exp.table(df)
