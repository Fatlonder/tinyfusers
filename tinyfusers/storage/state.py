from collections import namedtuple, OrderedDict

def update_state(obj, state_dict, prefix=''):
    if hasattr(obj, '__dict__'):
     update_state(obj.__dict__, state_dict, f"{prefix}")
    elif hasattr(obj, '_asdict'):
     update_state(obj._asdict(), state_dict, prefix)
    elif isinstance(obj, OrderedDict):
     update_state(dict(obj), state_dict, prefix)
    elif isinstance(obj, (list, tuple)):
      for i, x in enumerate(obj):
        update_state(x, state_dict, f"{prefix}.{str(i)}")
    elif isinstance(obj, dict):
      for k,v in obj.items():
        if k in {"weight", "bias"}:
          print(f"{prefix}.{k}")
          obj[k] = state_dict[f"{prefix}.{k}"]
        else:
          pre = f"{prefix}.{k}" if prefix!='' else f"{k}"
          update_state(v, state_dict, f"{pre}")