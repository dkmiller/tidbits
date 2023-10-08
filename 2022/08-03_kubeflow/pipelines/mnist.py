import kfp
import logging
from pathlib import Path

from kubeflow_assistant import KubeflowAssistant


log = logging.getLogger(__name__)


@kfp.dsl.pipeline(name="mnist")
def sample_pipeline():
    assistant = KubeflowAssistant(Path(__file__).parent.parent)

    download_mnist = assistant.build_and_load_component("download_mnist")

    data_step = download_mnist()


def main():
    host = "http://localhost:8080"
    client = kfp.Client(host=host)
    run = client.create_run_from_pipeline_func(sample_pipeline, arguments={})

    run_link = f"{host}/#/runs/details/{run.run_id}"
    log.info(f"Submitted:\n\n\t{run_link}\n")


if __name__ == "__main__":
    logging.basicConfig(level="INFO")
    main()
