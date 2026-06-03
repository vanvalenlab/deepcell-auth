import yaml


__all__ = [
    "download_cellsam_evaluation_dataset",
]
__version__ = "0.1.0-dev"


def download_cellsam_evaluation_dataset(version=None):
    """Download the evaluation data for the CellSAM model.

    The compressed dataset will be downloaded to the canonical location:
    ``$HOME/.deepcell/data``.

    Parameters
    ----------
    version : str, optional, default=latest
       Which version of the dataset to download. If not specified, the latest
       published version will be downloaded. Available versions:

         - 1.2 (latest)
         - 1.0
    """
    from ._auth import fetch_data

    # Load manifest
    with open("asset_manifest.yaml") as fh:
        manifest = yaml.safe_load(fh.read())["models"]["cellsam"]

    version = "1.2" if version is None else version
    try:
        record = manifest[version]
    except KeyError:
        raise KeyError(
            f"Version {version} not found. Available versions: {list(manifest)}"
        )

    fetch_data(
        record["asset_key"], cache_subdir="data", file_hash=record["asset_hash"]
    )
