import unicodedata
import numpy as np


def is_string(s):
    """判断是否是字符串
    """
    return isinstance(s, str)


def truncate_sequences(maxlen, indices, *sequences):
    """截断总长度至不超过maxlen
    """
    sequences = [s for s in sequences if s]
    if not isinstance(indices, (list, tuple)):
        indices = [indices] * len(sequences)

    while True:
        lengths = [len(s) for s in sequences]
        if sum(lengths) > maxlen:
            i = np.argmax(lengths)
            sequences[i].pop(indices[i])
        else:
            return sequences


def text_segmentate(text, maxlen, seps='\n', strips=None):
    """将文本按照标点符号划分为若干个短句
    """
    text = text.strip().strip(strips)
    if not seps or len(text) <= maxlen:
        return [text]
    pieces = text.split(seps[0])
    text, texts = '', []
    for i, p in enumerate(pieces):
        if text and p and len(text) + len(p) > maxlen - 1:
            texts.extend(text_segmentate(text, maxlen, seps[1:], strips))
            text = ''
        text = text + p if i + 1 == len(pieces) else text + p + seps[0]
    if text:
        texts.extend(text_segmentate(text, maxlen, seps[1:], strips))
    return texts


def lowercase_and_normalize(text):
    """转小写，并进行简单的标准化
    """
    text = text.lower()
    text = unicodedata.normalize('NFD', text)
    text = ''.join([ch for ch in text if unicodedata.category(ch) != 'Mn'])
    return text


def check_object_type(object, check_type, name):
    if not isinstance(object, check_type):
        raise TypeError(f"The type of {name} must be {check_type}, but got {type(object)}.")
