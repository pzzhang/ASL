import logging
import torch.nn as nn
import torchvision.models as tvmodels

logger = logging.getLogger(__name__)

# from ..tresnet import TResnetM, TResnetL, TResnetXL
from ..vision_transformer import *


def create_model(args):
    """Create a model
    """
    model_params = {'args': args, 'num_classes': args.num_classes}
    args = model_params['args']
    args.model_name = args.model_name.lower()

    # get torch vision models
    model_names = sorted(name for name in tvmodels.__dict__
                         if name.islower() and not name.startswith("__")
                         and callable(tvmodels.__dict__[name]))
    print("torchvision models: \n", model_names)
    # vit/deit models
    vitmodeldict = {
        'deit_tiny_patch16_224': deit_tiny_patch16_224,
        'deit_small_patch16_224': deit_small_patch16_224,
        'deit_base_patch16_224': deit_base_patch16_224,
    }
    vit_model_names = list(vitmodeldict.keys())
    print("Vision Transformer models: \n", vit_model_names)

    if args.model_name == 'tresnet_m':
        model = TResnetM(model_params)
    elif args.model_name == 'tresnet_l':
        model = TResnetL(model_params)
    elif args.model_name == 'tresnet_xl':
        model = TResnetXL(model_params)
    elif args.model_name in model_names:
        logging.info("Use torchvision predefined model")
        logging.info("=> using torchvision pre-trained model '{}'".format(args.model_name))
        model = tvmodels.__dict__[args.model_name](pretrained=True,)
        if model.fc.out_features != args.num_classes:
            model.fc = nn.Linear(model.fc.in_features, args.num_classes)
    elif args.model_name.startswith('deit_'):
        logging.info("Use vision transformer (deit) model")
        config = dict(
            drop_rate=args.drop if hasattr(args, 'drop') else 0.0,
            drop_path_rate=args.drop_path if hasattr(args, 'drop_path') else 0.2,
        )
        logging.info("=> using DEIT pre-trained model '{}'".format(args.model_name))
        model = vitmodeldict[args.model_name](pretrained=True, **config)
        if model.num_classes != args.num_classes:
            model.head = nn.Linear(model.embed_dim, args.num_classes)
    else:
        print("model: {} not found !!".format(args.model_name))
        exit(-1)

    return model
