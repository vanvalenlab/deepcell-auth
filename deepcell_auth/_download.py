__all__ = [
    "download_cellsam_evaluation_dataset",
    "download_cellsam_model",
]

def download_cellsam_model(version=None):
    """Download pre-trained weights for the CellSAM model.

    Parameters
    ----------
    version : str, optional, default=latest
       Which version of the model weights to download. If not specified, the latest
       published version will be downloaded. Available versions:

         - 1.2 (latest)
         - 1.0
    """
    from ._auth import load_manifest, fetch_data

    manifest = load_manifest()

    version = "1.2" if version is None else version
    try:
        record = manifest["models"]["cellsam"][version]
    except KeyError:
        raise KeyError(
            f"Version {version} not found. Available versions: {list(manifest)}"
        )

    fetch_data(
        record["asset_key"], cache_subdir="models", file_hash=record["asset_hash"]
    )

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
    from ._auth import load_manifest, fetch_data

    manifest = load_manifest()

    version = "1.2" if version is None else version
    try:
        record = manifest["datasets"]["cellsam"][version]
    except KeyError:
        raise KeyError(
            f"Version {version} not found. Available versions: {list(manifest)}"
        )

    fetch_data(
        record["asset_key"], cache_subdir="data", file_hash=record["asset_hash"]
    )
