from .sequence_classification import get_auto_fc_tc_model
from .sequence_classification_mdp import get_auto_mdp_tc_model
from .sequence_classification_rdrop import get_auto_rdrop_tc_model


NER_MODEL_FN = {
    "fc": get_auto_fc_tc_model,
    "mdp": get_auto_mdp_tc_model,
    "rdrop": get_auto_rdrop_tc_model,
}


def get_auto_tc_model(model_name: str = "fc", model_type: str = "bert"):
    fct = NER_MODEL_FN[model_name]
    return fct(model_type)
