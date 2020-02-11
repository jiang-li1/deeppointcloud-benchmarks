from omegaconf.dictconfig import DictConfig
from omegaconf.listconfig import ListConfig


def resolve_model(model_config, dataset, tested_task):
    """ Parses the model config and evaluates any expression that may contain constants
    """
    # placeholders to subsitute
    constants = {
        "FEAT": max(dataset.feature_dimension, 3),
        "TASK": tested_task,
        "N_CLS": dataset.num_classes if hasattr(dataset, "num_classes") else None,
        "SCALE": dataset.pointcloud_scale if hasattr(dataset, "pointcloud_scale") else 1,
    }

    # user defined contants to subsitute
    if "define_constants" in model_config.keys():
        custom_constants = dict(model_config.define_constants)
        _resolve(custom_constants, constants)
        constants.update(custom_constants)

    _resolve(model_config, constants)


def _resolve(obj, constants):
    """ Resolves expressions and constants in obj.
    returns False if obj is a ListConfig or DictConfig, True is obj is a primative type.
    """
    if type(obj) == DictConfig or type(obj) == dict:
        it = (k for k in obj)
    elif type(obj) == ListConfig or type(obj) == list:
        it = range(len(obj))
    else:
        # obj is a single element
        return True

    # recursively resolve all children of obj
    for k in it:

        # if obj[k] is a primative type, evalulate it
        if _resolve(obj[k], constants):
            if type(obj[k]) is str:
                try:
                    obj[k] = eval(obj[k], constants)
                except NameError:
                    # we tried to resolve a string which isn't an expression
                    pass
                except ValueError:
                    # we tried to resolve a string which is also a builtin (e.g. max)
                    pass
                except Exception as e:
                    print(e)

    return False
