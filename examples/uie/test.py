import requests
from pprint import pprint

data_bin = {
    'model_name': 'uie', 
    'input': "2月8日上午北京冬奥会自由式滑雪女子大跳台决赛中中国选手谷爱凌以188.25分获得金牌！", 
    "uie_schema": "时间 赛事名称 选手",
    # "position_prob": 0.5,
    # "max_seq_len": 512,
}
res = requests.post('http://192.168.0.55:8000/uie', json=data_bin).json()
pprint(res)

data_bin = {
    'model_name': 'uie-medical', 
    'input': "第十三节狂犬病狂犬病（rabies）又称恐水症（hydrophobia），是由狂犬病毒引起的中枢神经系统急性传染病，为人畜共患的自然疫源性疾病。", 
    "uie_schema": "疾病 症状 身体部位 药物 手术 检查项目 微生物",
    # "position_prob": 0.5,
    # "max_seq_len": 512,
}
res2 = requests.post('http://192.168.0.55:8000/uie', json=data_bin).json()
pprint(res2)
