import sys

sys.path.append("../..")

import time
import streamlit as st
import pandas as pd
import seaborn as sns
from torchblocks.tasks.uie import UIEPredictor
from torchblocks.utils.app import visualize_ner, download_button, _max_width_, make_color_palette

MODEL_CLASSES = {
    "uie": "uie_base_pytorch",
    "uie-medical": "uie_medical_base_pytorch",
    "uie-medical-finetuned": "checkpoint/uie_medical",
    "uie-finetuned": "checkpoint/uie_medical_cluener",
}


@st.cache(hash_funcs={UIEPredictor: id})
def load_ie(model_name_or_path, schema, prob, max_seq_len, device="cpu", split_sentence=False):
    return UIEPredictor(model_name_or_path=model_name_or_path, schema=schema, position_prob=prob,
                        max_seq_len=max_seq_len, device=device, split_sentence=split_sentence)


# 设置网页信息 
st.set_page_config(page_title="UIE DEMO", page_icon="🚀", layout="wide")

_max_width_()

c30, c31, c32 = st.columns([2.5, 1, 3])

with c30:
    # st.image("logo.png", width=400)
    st.title("🔑 UIE命名实体识别")
    st.header("")

with st.expander("ℹ️ - 关于此APP", expanded=True):
    st.write(
        """     
-   [UIE(Universal Information Extraction)](https://arxiv.org/pdf/2203.12277.pdf)：Yaojie Lu等人在`ACL-2022`中提出了通用信息抽取统一框架`UIE`。
-   该框架实现了实体抽取、关系抽取、事件抽取、情感分析等任务的统一建模，并使得不同任务间具备良好的迁移和泛化能力。
-   为了方便大家使用UIE的强大能力，`PaddleNLP`借鉴该论文的方法，基于`ERNIE 3.0`知识增强预训练模型，训练并开源了首个中文通用信息抽取模型`UIE`。
-   该模型可以支持不限定行业领域和抽取目标的关键信息抽取，实现零样本快速冷启动，并具备优秀的小样本微调能力，快速适配特定的抽取目标。
	    """
    )

    st.markdown("")

st.markdown("")
st.markdown("## 📌 输入")

with st.form(key="my_form"):
    ce, c1, _, c2, c3 = st.columns([0.07, 1, 0.07, 5, 0.07])

    with c1:
        model_type = st.radio(
            "选择预训练模型",
            ("uie", "uie-medical", "uie-medical-finetuned", "uie-finetuned"),
            help="""目前支持四个预训练模型，其中uie和uie-medical为没有训练的模型，
            
            uie-medical-finetuned为医疗数据微调之后的模型，uie-finetuned为微调之后的模型。""",
        )

        max_seq_len = st.number_input(
            '句子最大长度',
            0,
            512,
            512,
            help="模型输入的最大文本长度，超过该长度则截断。")

        prob = st.slider(
            "阈值",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.01,
            help="模型输出实体的阈值，当实体起、始位置的概率值大于该值则输出。")

        split = st.checkbox(
            '截断句子',
            value=False,
            help="对于文本长度超过最大长度的句子，将句子切分成多个句子输入模型。")

        gpu = st.checkbox(
            'GPU',
            value=True,
            help="使用GPU可以加快模型的推理速度。")

    with c2:
        exp = st.expander("选择实体类型")
        schema_options = exp.multiselect('候选类型',
                                         ["时间", "地点", "人物", "疾病", "症状、临床表现", "身体物质和身体部位", "药物",
                                          "检查、治疗或预防程序", "部门科室", "医学检验项目", "微生物",
                                          "检查设备和治疗设备"],
                                         ["时间", "人物"])

        schema = st.text_area(
            "🔨请输入抽取任务的实体类型（使用空格分隔）",
            height=100,
        )

        schema = (" ".join(
            schema_options) + " " + schema.strip()).strip() if schema_options is not None else schema.strip()

        text = st.text_area(
            "📖请输入待抽取的句子",
            height=200, )

        submit_button = st.form_submit_button(label="✨ 运行")

if not submit_button:
    st.stop()

schema = schema.split(' ')
device = "gpu" if gpu else "cpu"
ie = load_ie(MODEL_CLASSES[model_type], schema, prob, max_seq_len, device, split)

start = time.time()
rlt = ie(text.split('\n'))
running_time = time.time() - start

res = []
for r in rlt:
    tmp = {_type: [
        {"text": ent["text"], "start": ent["start"], "end": ent["end"], "probability": float(ent["probability"])}
        for ent in ents] for _type, ents in r.items()}
    res.append(tmp)

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
    colors = make_color_palette(schema)
    visualize_ner(text, res, colors=colors)

    json_doc_exp = st.expander("JSON")
    json_doc_exp.json(res)

    dataframe_exp = st.expander("DATAFRAME")
    columns = ["text", "start", "end", "label", "probability"]
    for r in res:
        data = [[t[columns[0]], t[columns[1]], t[columns[2]], k, t[columns[4]]] for k, v in r.items() for t in v]
        df = pd.DataFrame(data, columns=columns).sort_values(by="probability", ascending=False).reset_index(drop=True)
        df.index += 1

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
            "probability": "{:.2%}",
        }

        df = df.format(format_dictionary)
        dataframe_exp.table(df)
