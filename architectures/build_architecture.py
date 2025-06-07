"""
To select the architecture based on a config file we need to ensure
we import each of the architectures into this file. Once we have that
we can use a keyword from the config file to build the model.
"""
from .segformer3d_esa_brats import build_segformer3d_model as build_segformer3d_model_esa
from .segformer3d_bsm_brats import build_segformer3d_model as build_segformer3d_model_bsm
from .segformer3d_epa_brats import build_segformer3d_model as build_segformer3d_model_epa
from .segformer3d_bsm_acdc import build_segformer3d_model as build_segformer3d_model_bsm_acdc
from .segformer3d_epa_acdc import build_segformer3d_model as build_segformer3d_model_epa_acdc

######################################################################
def build_architecture(config):
    if config["model"]['name'] == "segformer3d_esa":
        print("Building segformer3d with ESA")
        model = build_segformer3d_model_esa(config)
        return model
    elif config["model"]['name'] == "segformer3d_bsm":
        print("Building segformer3d with BSM")
        model = build_segformer3d_model_bsm(config)
        return model
    elif config["model"]['name'] == "segformer3d_epa":
        print("Building segformer3d with EPA")
        model = build_segformer3d_model_epa(config)
        return model
    elif config["model"]['name'] == "segformer3d_bsm_acdc":
        print("Building segformer3d with BSM on ACDC dataset")
        model = build_segformer3d_model_bsm_acdc(config)
        return model
    elif config["model"]['name'] == "segformer3d_epa_acdc":
        print("Building segformer3d with EPA on ACDC dataset")
        model = build_segformer3d_model_epa_acdc(config)
        return model
    else:
        return ValueError(
            "specified model not supported, edit build_architecture.py file"
        )