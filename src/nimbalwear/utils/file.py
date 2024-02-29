"""Utilities that manipulate files.

Functions:
    convert_json_to_toml(json_path, toml_path) --> dict

"""

import io
from pathlib import Path
import json

from msoffcrypto import OfficeFile
import toml
import pandas as pd


def convert_json_to_toml(json_path, toml_path):
    """Converts a json file to a toml file.

    Parameters
    ----------
    json_path : Path or str
        Path to the json file to be read.
    toml_path : Path or str
        Path to the toml file to be written.

    Returns
    -------
    dict
        Dictionary of file contents

    """

    json_path = Path(json_path)
    toml_path = Path(toml_path)

    with open(json_path, 'r') as f:
        contents = json.load(f)

    with open(toml_path, 'w') as f:
        toml.dump(contents, f)

    return contents


def read_excel_pwd(file_path, password, **kwargs):

    file = OfficeFile(open(file_path, "rb"))
    file.load_key(password=password)

    decrypted = io.BytesIO()
    file.decrypt(decrypted)

    df = pd.read_excel(decrypted, **kwargs)

    return df
