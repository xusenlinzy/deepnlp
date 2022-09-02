import scipy.stats
import numpy as np
from numpy import ndarray
from torch import Tensor
from typing import List, Union
from .base import Embeddings_Base


class BERT_WHITENING(object):
    def __init__(self, n_components=None):
        self.kernel = None
        self.bias = None
        self.n_components = n_components

    def compute_kernel_bias(self, vecs):
        """bert-whitening
        vecs: ndarray [num_samples, embedding_size]
        """
        self.bias = -vecs.mean(0, keepdims=True)
        cov = np.cov(vecs.T)  # 协方差
        u, s, vh = np.linalg.svd(cov)
        W = np.dot(u, np.diag(1 / np.sqrt(s)))
        self.kernel = W[:, :self.n_components] if self.n_components is not None else W
    
    def save_whiten(self, path):
        whiten = {'kernel': self.kernel, 'bias': self.bias}
        np.save(path, whiten)
        
    def load_whiten(self, path):
        whiten = np.load(path)
        self.kernel = whiten['kernel']
        self.bias = whiten['bias']

    def transform_and_normalize(self, vecs):
        """应用变换，然后标准化
        """
        if self.kernel is not None and self.bias is not None:
            vecs = (vecs + self.bias).dot(self.kernel)
        return vecs / (vecs**2).sum(axis=1, keepdims=True)**0.5


def compute_corrcoef(x, y):
    """Spearman相关系数
    """
    return scipy.stats.spearmanr(x, y).correlation


class BertWhitening(Embeddings_Base):
    """
    简单的线性变换（白化）操作，就可以达到BERT-flow的效果
    https://spaces.ac.cn/archives/8069
    步骤：
    1. bert_whitening = BertWhitening('bert-base-chinese', n_components=256)
    2. bert_whitening.whitening(all_sentences) # 计算kernel和bias
    3. bert_whitening.encode(example_sentence)
    """
    def __init__(self, model_name_or_path, device, pooler, n_components=None):
        super().__init__(model_name_or_path, device, pooler)
        self.whitening_ = BERT_WHITENING(n_components)

    def whitening(self, sentence: Union[str, List[str]], 
                device: str = None, 
                return_numpy: bool = True,
                normalize_to_unit: bool = False,
                keepdim: bool = False,
                batch_size: int = 64,
                max_length: int = 128) -> Union[ndarray, Tensor]:
        embeddings = super().encode(sentence, device, return_numpy, normalize_to_unit,
                                keepdim, batch_size, max_length)
        self.whitening_.compute_kernel_bias(embeddings)

    def encode(self, sentence: Union[str, List[str]], 
                device: str = None, 
                return_numpy: bool = True,
                normalize_to_unit: bool = False,
                keepdim: bool = False,
                batch_size: int = 64,
                max_length: int = 128) -> Union[ndarray, Tensor]:

        embeddings = super().encode(sentence, device, return_numpy, normalize_to_unit,
                                keepdim, batch_size, max_length)
        
        single_sentence = False
        if isinstance(sentence, str):
            embeddings = [embeddings]
            single_sentence = True

        whitening_embeddings = []
        for embedding in embeddings:
            embed = self.whitening_.transform_and_normalize(embedding)
            whitening_embeddings.append(embed)
        
        if single_sentence and not keepdim:
            whitening_embeddings = whitening_embeddings[0]
        
        if return_numpy and not isinstance(embeddings, ndarray):
            return np.array(whitening_embeddings)
        
        return whitening_embeddings

    