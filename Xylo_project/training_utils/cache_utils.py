import hashlib
import os

def generate_hash(*args) -> str:
    """Process given arguments as strings and create a hash string."""
    __hash = hashlib.md5()
    for arg in args:
        __hash.update(str(arg).encode("utf-8"))
    return __hash.hexdigest()

def create_config_str(params_dict, not_cacheing_params,cache_path):
    config_str = ""
    for key in params_dict.keys():
        # drop arguments that are not necessary to identify unique configuration
        if key not in not_cacheing_params:
            config_str += key + str(params_dict[key])

    config_str_hash = hashlib.sha1(config_str.encode('utf-8')).hexdigest()

    cache_dir = os.path.join(cache_path, "cache", config_str_hash)
    # Only save a .txt file if the cache folders have been generated
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    # If .txt file with the cacheing info does not exist, create one
    if not os.path.exists(cache_dir + '/cache_info.txt'):
        with open(cache_dir + '/cache_info.txt', 'w') as f:
            for key in params_dict.keys():
                if key not in not_cacheing_params:
                    f.write(key)
                    f.write(':')
                    f.write(str(params_dict[key]))
                    f.write('\n')

    return config_str_hash