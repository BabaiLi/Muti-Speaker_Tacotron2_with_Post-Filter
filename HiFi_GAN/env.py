from pathlib import Path
import shutil

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def build_env(config, config_name, path):
    t_path = Path(path, config_name)
    if config != t_path:
        Path.mkdir(path, exist_ok=True)
        shutil.copyfile(config, Path(path, config_name))
