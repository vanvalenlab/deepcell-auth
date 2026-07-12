__all__ = [
    "download_cellsam_evaluation_dataset",
    "download_cellsam_model",
    "download_deepcell_types_model",
    "download_deepcell_types_data",
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


def download_deepcell_types_model(version=None):
    """Download pre-trained weights for the deepcell-types model.

    Parameters
    ----------
    version : str, optional, default=latest
       Which version of the model weights to download. If not specified, the latest
       published version will be downloaded. Available versions:

         - '2026-06-15'  (latest)
         - '2026-06-23'
    """
    from ._auth import load_manifest, fetch_data

    manifest = load_manifest()
    dct_models = manifest["models"]["deepcell-types"]

    version = "2026-06-15" if version is None else version
    try:
        record = dct_models[version]
    except KeyError:
        raise KeyError(
            f"Version {version} not found. Available versions: {list(dct_models)}"
        )

    fetch_data(
        record["asset_key"], cache_subdir="models", file_hash=record["asset_hash"]
    )


def download_deepcell_types_baseline(name):
    """Download a deepcell-types comparison-baseline checkpoint.

    Parameters
    ----------
    name : str
        Baseline identifier. One of 'cellsighter', 'maps', or 'xgboost'.
        Some baselines ship companion files (maps -> _stats.npz;
        xgboost -> .remap.json), so a list of local paths is returned.

        'nimbus' is intentionally not served here: its pretrained weights are
        distributed upstream (angelolab/Nimbus-Inference) and are not
        re-hosted by this project.

    Returns
    -------
    list
        Local paths to every file downloaded for this baseline.
    """
    from ._auth import load_manifest, fetch_data

    manifest = load_manifest()
    baselines = manifest["models"]["deepcell-types-baselines"]

    try:
        records = baselines[name]
    except KeyError:
        raise KeyError(
            f"Baseline {name} not found. Available baselines: {list(baselines)}"
        )

    return [
        fetch_data(
            record["asset_key"], cache_subdir="models", file_hash=record["asset_hash"]
        )
        for record in records
    ]


def download_deepcell_types_data(version=None):
    """Download training dataset for the deepcell-types project.

    The compressed dataset will be downloaded to the canonical location:
    ``$HOME/.deepcell/data``.

    Parameters
    ----------
    version : str, optional, default=latest
       Which version of the dataset to download. If not specified, the latest
       published version will be downloaded. Available versions:

         - 1.1 (latest)
    """
    from ._auth import load_manifest, fetch_data

    manifest = load_manifest()
    dct_datasets = manifest["datasets"]["deepcell-types"]

    version = "1.1" if version is None else version
    try:
        record = dct_datasets[version]
    except KeyError:
        raise KeyError(
            f"Version {version} not found. Available versions: {list(dct_datasets)}"
        )

    fetch_data(record["asset_key"], cache_subdir="data")

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
