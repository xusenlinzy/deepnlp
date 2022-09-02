import sys

sys.path.append("../..")

import json
import pandas as pd
from tqdm import tqdm
from torchblocks.tasks.uie import UIEPredictor


def read_json(filepath) -> pd.DataFrame:
    """ 将`json`格式的数据转换成两列，一列为文本，一列为实体集合
    """
    texts, labels = [], []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in tqdm(f):
            line = json.loads(line)
            text = line['text']
            label = {(value["label"], text[value["start_offset"]: value["end_offset"]]) for value in line["entities"]}

            texts.append(text)
            labels.append(label)
    return pd.DataFrame({"text": texts, "label": labels})


def reformat(data: dict) -> dict:
    """ `json`无法解码字典中的`numpy.float()`，将其转换为`float`
    """
    return {_type: [
        {"end": ent["end"], "start": ent["start"], "probability": float(ent["probability"]), "text": ent["text"]} for
        ent in ents] for _type, ents in data.items()}


def dict2set(result: dict) -> set:
    """ 将实体预测字典整理输出为集合
    """
    res = set()
    for k, values in result.items():
        for value in values:
            res.add((k, value['text']))
    return res


def set2json(data: set) -> str:
    """ 将实体集合根据实体类型整理输出为`json`字符串
    """
    res = {}
    for d in data:
        if d[0] not in res:
            res[d[0]] = [d[1]]
        else:
            res[d[0]].append(d[1])
    return json.dumps(res, ensure_ascii=False)


if __name__ == '__main__':
    schema = ["疾病", "症状、临床表现", "身体物质和身体部位", "药物", "检查、治疗或预防程序", "部门科室",
              "医学检验项目", "微生物", "检查设备和治疗设备"]
    uie = UIEPredictor("./checkpoint/uie_bert_v0/checkpoint-eval_f1-best", schema=schema, device="gpu", position_prob=0.5, batch_size=300)

    # medical paper predict
    df = pd.read_csv("data/medical_paper.csv")

    df['entities'] = [json.dumps(reformat(d), ensure_ascii=False) for d in uie(list(df.abstract.values))]
    df.to_csv('results/medical_paper_results.csv', index=False)

    # medical training data clean
    # df = read_json('data/medical_clean.jsonl')
    
    # df['prediction'] = [reformat(d) for d in uie(list(df.text.values))]
    # df['prediction'] = df['prediction'].apply(dict2set)
    # df['pred-true'], df['true-pred'] = df['prediction'] - df['label'], df['label'] - df['prediction']
    
    # for col in ['label', 'prediction', 'pred-true', 'true-pred']:
    #     df[col] = df[col].apply(set2json)
    
    # df.to_csv('results/medical_results.csv', index=False)
