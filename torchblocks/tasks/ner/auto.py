from .crf import get_auto_crf_ner_model, get_auto_cascade_crf_ner_model
from .softmax import get_auto_softmax_ner_model
from .span import get_auto_span_ner_model
from .global_pointer import get_auto_gp_ner_model
from .mrc import get_auto_mrc_ner_model
from .tplinker import get_auto_tplinker_ner_model
from .lear import get_auto_lear_ner_model
from .w2ner import get_auto_w2ner_ner_model


NER_MODEL_FN = {
    "crf": get_auto_crf_ner_model,
    "cascade-crf": get_auto_cascade_crf_ner_model,
    "softmax": get_auto_softmax_ner_model,
    "span": get_auto_span_ner_model,
    "global-pointer": get_auto_gp_ner_model,
    "mrc": get_auto_mrc_ner_model,
    "tplinker": get_auto_tplinker_ner_model,
    "lear": get_auto_lear_ner_model,
    "w2ner": get_auto_w2ner_ner_model,
}


def get_auto_ner_model(model_name: str = "crf", model_type: str = "bert"):
    fct = NER_MODEL_FN[model_name]
    return fct(model_type)
