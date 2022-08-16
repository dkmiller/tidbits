import kfp
import logging
from pathlib import Path

from kubeflow_assistant import KubeflowAssistant


log = logging.getLogger(__name__)


@kfp.dsl.pipeline(name="sample")
def sample_pipeline(n_file: int = 25):
    assistant = KubeflowAssistant(Path(__file__).parent.parent)

    gen_data = assistant.build_and_load_component("gen_data")
    show_data = assistant.build_and_load_component("show_data")
    show_data_r = assistant.build_and_load_component("show_data_r")

    gen_data_step = gen_data(n_files=n_file)

    show_data_step = show_data(input=gen_data_step.outputs["random_files"], sleep_seconds=60)
    show_data_r_step = show_data_r(input=gen_data_step.outputs["random_files"])


def main():
    host = "http://localhost:8080"
    client = kfp.Client(host=host)
    run = client.create_run_from_pipeline_func(sample_pipeline, arguments={})

    run_link = f"{host}/#/runs/details/{run.run_id}"
    log.info(f"Submitted:\n\n\t{run_link}\n")


if __name__ == "__main__":
    logging.basicConfig(level="INFO")
    main()
