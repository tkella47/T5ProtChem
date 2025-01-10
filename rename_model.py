#credit because I am only one person 
#https://gist.github.com/the-bass/0bf8aaa302f9ba0d26798b11e4dd73e3

import torch
from collections import OrderedDict
import argparse

def rename_state_dict_keys(source, key_transformation, target=None, model=False):
    """
    source             -> Path to the saved state dict.
    key_transformation -> Function that accepts the old key names of the state
                          dict as the only argument and returns the new key name.
    target (optional)  -> Path at which the new state dict should be saved
                          (defaults to `source`)
    Example:
    Rename the key `layer.0.weight` `layer.1.weight` and keep the names of all
    other keys.
    ```py
    def key_transformation(old_key):
        if old_key == "layer.0.weight":
            return "layer.1.weight"
        return old_key
    rename_state_dict_keys(state_dict_path, key_transformation)
    ```
    """
    if target is None:
        target = source

    state_dict = torch.load(source)
    new_state_dict = OrderedDict()

    for key, value in state_dict.items():
        new_key = key_transformation(key, model)
        new_state_dict[new_key] = value

    torch.save(new_state_dict, target)

def rename(key, model=False):
    if model:
        return "model." + key[22:]
    else:
        return key[22:]
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", type=str)
    parser.add_argument("output_path", type=str)
    args = parser.parse_args()
    rename_state_dict_keys(args.model_path, rename, args.output_path)
    rename_state_dict_keys(args.model_path, rename, args.output_path + ".model", True)
