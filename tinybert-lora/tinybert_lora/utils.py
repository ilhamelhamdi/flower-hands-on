import tomllib
from omegaconf import OmegaConf

def load_config(file_path="pyproject.toml", cli_args=None):
    with open(file_path, "rb") as f:
        config = tomllib.load(f)
    conf = OmegaConf.create(config["tool"]["training"])
    if cli_args:
        cli_config = OmegaConf.from_dotlist(cli_args)
        conf = OmegaConf.merge(conf, cli_config)
    return conf

def to_dict(conf):
    """Convert OmegaConf/DictConfig to a standard Python dict."""
    return OmegaConf.to_container(conf, resolve=True)