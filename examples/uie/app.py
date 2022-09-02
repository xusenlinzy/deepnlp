import sys

sys.path.append("../..")

import time
import streamlit as st
from torchblocks.tasks.uie import UIEPredictor

MODEL_CLASSES = {
    "UIE": "uie_base_pytorch",
    "UIE-MEDICAL": "uie_medical_base_pytorch",
    "UIE-MEDICAL-FINETUNED": "checkpoint/uie_medical",
}


@st.cache(hash_funcs={UIEPredictor: id})
def load_ie(model_name_or_path, schema, prob, max_seq_len, device="cpu", split_sentence=False):
    return UIEPredictor(model_name_or_path=model_name_or_path, schema=schema, position_prob=prob,
                        max_seq_len=max_seq_len, device=device, split_sentence=split_sentence)


# 设置网页信息 
st.set_page_config(page_title="UIE DEMO", page_icon="🚀", layout="wide")

# st.title('UIE DEMO')
html_tmp = """
    <div>
    <h1 style="text-align:center;">UIE命名实体识别</h1>
    </div>
"""
st.markdown(html_tmp, unsafe_allow_html=True)
st.markdown('---')

st.subheader("🔨定制SCHEMA")
if candidate := st.checkbox('候选实体类型'):
    schema_options = st.multiselect('选择预定义的实体类型',
                                    ["时间", "地点", "人物", "疾病", "症状、临床表现", "身体物质和身体部位", "药物", 
                                     "检查、治疗或预防程序", "部门科室", "医学检验项目", "微生物", "检查设备和治疗设备"], None)
else:
    schema_options = None
schema = st.text_area("请输入抽取任务的实体类型（使用空格分隔）：")
schema = (" ".join(schema_options) + " " + schema) if schema_options is not None else schema

st.subheader("📖输入文本")
text = st.text_area("请输入待抽取的句子：")

html_tmp = """
    <div>
    <h2 style="text-align:center;">配置参数</h2>
    </div>
"""
st.sidebar.markdown(html_tmp, unsafe_allow_html=True)

# prob = st.sidebar.number_input('阈值', 0.0, 1.0, 0.5)
st.sidebar.markdown('---')

model_type = st.sidebar.radio(
    "预训练模型",
    ("UIE", "UIE-Medical", "UIE-Medical-Finetuned"),
)

st.sidebar.markdown('---')

max_seq_len = st.sidebar.number_input('句子最大长度', 0, 512, 512)
st.sidebar.markdown('---')
prob = st.sidebar.slider("阈值", min_value=0.0, max_value=1.0, value=0.5, step=0.01)

st.sidebar.markdown('---')
split = st.sidebar.checkbox('截断句子')
gpu = st.sidebar.checkbox('使用GPU')
plot = st.sidebar.checkbox('显示实体标签分布')

schema = schema.split(' ')
device = "gpu" if gpu else "cpu"
ie = load_ie(MODEL_CLASSES[model_type], schema, prob, max_seq_len, device, split)

if st.button('运行🚀') & (text != ''):
    text = text.split('\n')
    start = time.time()
    rlt = ie(text)
    running_time = time.time() - start
    st.text(f"Runtime for UIE: {running_time}")
    st.json(rlt)
    st.stop()
