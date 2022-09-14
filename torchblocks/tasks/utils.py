import torch
from typing import *
from transformers.file_utils import ModelOutput
from dataclasses import dataclass
from transformers import BertModel, BertPreTrainedModel
from transformers.models.roformer.modeling_roformer import RoFormerModel, RoFormerPreTrainedModel
from transformers.models.nezha.modeling_nezha import NezhaModel, NezhaPreTrainedModel
from transformers.models.albert.modeling_albert import AlbertModel, AlbertPreTrainedModel
from transformers.models.xlnet.modeling_xlnet import XLNetModel, XLNetPreTrainedModel


MODEL_MAP = {
    "bert": (BertModel, BertPreTrainedModel),
    "ernie": (BertModel, BertPreTrainedModel),
    "roformer": (RoFormerModel, RoFormerPreTrainedModel),
    "nezha": (NezhaModel, NezhaPreTrainedModel),
    "albert": (AlbertModel, AlbertPreTrainedModel),
    "xlnet": (XLNetModel, XLNetPreTrainedModel),
}


@dataclass
class SequenceClassifierOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    predictions: List[List[List[str]]] = None
    groundtruths: List[List[List[str]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


@dataclass
class SequenceLabelingOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    predictions: List[List[List[str]]] = None
    groundtruths: List[List[List[str]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


@dataclass
class RelationExtractionOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    predictions: List[List[Tuple[str, int, int, int, int]]] = None
    groundtruths: List[List[Tuple[str, int, int, int, int]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


@dataclass
class SpanOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    start_logits: torch.FloatTensor = None
    end_logits: torch.FloatTensor = None
    span_logits: torch.FloatTensor = None
    predictions: List[List[Tuple[str, int, int]]] = None
    groundtruths: List[List[Tuple[str, int, int]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


@dataclass
class SiameseClassificationOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None


@dataclass
class SentenceEmbeddingOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    embeddings: Optional[torch.FloatTensor] = None


@dataclass
class UIEModelOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    start_prob: torch.FloatTensor = None
    end_prob: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
