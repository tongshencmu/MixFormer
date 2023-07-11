import torch
import logging 

def check_mismatch(self, model, ckpt):
    
    logger = logging.getLogger()
    # Load the state dictionary into the model
    all_param_names = []
    for name, param in model.named_parameters():
        all_param_names.append(name)
        if name in ckpt:
            if param.data.shape != ckpt[name].shape:
                logger.info(f"Init Weight ({model.__name__}): Size mismatch: " + name)
        else:
            logger.info(f"Init Weight ({model.__name__}): Name mismatch: " + name)
    
    all_param_names = set(all_param_names)
    
    for name, param in ckpt.items():
        if name in all_param_names:
            if param.data.shape != ckpt[name].shape:
                logger.info(f"Init Weight ({model.__name__}): Size mismatch: " + name)
        else:
            logger.info(f"Init Weight ({model.__name__}): Unused Parameter in pretrained model: " + name)
            