import time
import requests
import streamlit as st
import pandas as pd
import plotly_express as px


def count_distribution(rlt) -> dict:
    """ 统计各实体类型的数量
    """
    if isinstance(rlt, dict):
        counts = {k: len(v) for k, v in rlt.items()}
    else:
        counts = {}
        for r in rlt:
            for k, v in r.items():
                if k not in counts:
                    counts[k] = len(v)
                else:
                    counts[k] += len(v)
    return counts


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
schema = (" ".join(schema_options) + " " + schema.strip()) if schema_options is not None else schema.strip()
# st.text(schema)

st.subheader("📖输入文本")
text = st.text_area("请输入待抽取的句子：")

html_tmp = """
    <div>
    <h2 style="text-align:center;">配置参数</h2>
    </div>
"""
st.sidebar.markdown(html_tmp, unsafe_allow_html=True)

# 预训练模型
st.sidebar.markdown('---')
model_type = st.sidebar.radio(
    "预训练模型",
    ("uie", "uie-medical", "uie-medical-finetuned"),
)

st.sidebar.markdown('---')
max_seq_len = st.sidebar.number_input('句子最大长度', 0, 512, 512)
prob = st.sidebar.slider("阈值", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
split = st.sidebar.checkbox('截断句子')

st.sidebar.markdown('---')
gpu = st.sidebar.checkbox('使用GPU')
engine = st.sidebar.checkbox('使用ONNX')

st.sidebar.markdown('---')
plot = st.sidebar.checkbox('显示实体标签分布')

if st.button('运行🚀') & (text != ''):
    data_bin = {
        'model_name': model_type,
        'input': text.split('\n'),
        "uie_schema": schema.strip(),
        "position_prob": prob,
        "max_seq_len": max_seq_len,
        "split_sentence": split,
        "device": "gpu" if gpu else "cpu",
        "engine": "pytorch" if not engine else "onnx",
    }
    start = time.time()
    rlt = requests.post('http://192.168.0.55:8000/uie', json=data_bin).json()['rlt']
    # rlt = requests.post('http://192.168.31.38:8000/uie', json=data_bin).json()['rlt']
    running_time = time.time() - start
    st.text(f"Runtime for UIE: {running_time}")
    st.json(rlt)

    if plot:
        counts = count_distribution(rlt)
        counts = pd.DataFrame({"频数": counts.values(), "类型": counts.keys()})
        fig = px.bar(
            counts,
            x="频数",
            y="类型",
            orientation="h",
            title="<b>实体类型分布</b>",
            color_discrete_sequence=["#4682B4"] * len(counts),
            template="plotly_white",
        )
        fig.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis=(dict(showgrid=False))
        )
        st.plotly_chart(fig, use_container_width=True)

    st.stop()
