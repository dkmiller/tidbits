"""
Home-brewed Shrike.
"""


from azure.identity import DefaultAzureCredential
from azure.ml import dsl, MLClient
from pathlib import Path
from typing import Callable


def get_ml_client(workspace_name: str) -> MLClient:
    cred = DefaultAzureCredential()

    if workspace_name == "Alexander256V100":
        rv = MLClient(
            cred, "79f57c16-00fe-48da-87d4-5192e86cd047", "Alexander256", workspace_name
        )
    elif workspace_name == "aml1p-ml-wus2":
        rv = MLClient(
            cred, "48bbc269-ce89-4f6f-9a12-c6f91fcb772d", "aml1p-rg", workspace_name
        )
    else:
        raise ValueError(f"Workspace {workspace_name} is not known")

    return rv


def load_local_component(folder_name: str) -> Callable:
    """
    Load a local component from `component/{folder_name}/component.yaml`.
    """
    components_dir = Path(__file__).parent.parent / "component"
    component_file = components_dir / f"{folder_name}/component.yaml"
    component_path_raw = str(component_file.absolute())
    rv = dsl.load_component(yaml_file=component_path_raw)
    return rv
