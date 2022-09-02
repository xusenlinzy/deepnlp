from .casrel import get_auto_casrel_re_model
from .gplinker import get_auto_gplinker_re_model
from .grte import get_auto_grte_re_model
from .spn import get_auto_spn_re_model
from .tplinker import get_auto_tplinker_re_model
from .prgc import get_auto_prgc_re_model
from .pfn import get_auto_pfn_re_model

RE_MODEL_FN = {
    "casrel": get_auto_casrel_re_model,
    "gplinker": get_auto_gplinker_re_model,
    "tplinker": get_auto_tplinker_re_model,
    "grte": get_auto_grte_re_model,
    "spn": get_auto_spn_re_model,
    "prgc": get_auto_prgc_re_model,
    "pfn": get_auto_pfn_re_model,
}


def get_auto_re_model(model_name: str = "gplinker", model_type: str = "bert"):
    fct = RE_MODEL_FN[model_name]
    return fct(model_type)
