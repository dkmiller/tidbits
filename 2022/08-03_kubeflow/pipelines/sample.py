import kfp
import logging
from pathlib import Path

from kubeflow_assistant import KubeflowAssistant


log = logging.getLogger(__name__)


@kfp.dsl.pipeline(name="sample")
def sample_pipeline(n_file: int):
    assistant = KubeflowAssistant(Path(__file__).parent.parent)

    create_step_gen_data = assistant.build_and_load_component("gen_data")
    create_step_show_data = assistant.build_and_load_component("show_data")

    gen_data_step = create_step_gen_data(n_files=n_file)

    show_data_step = create_step_show_data(input=gen_data_step.outputs["random_files"])


def main():
    client = kfp.Client(host="http://localhost:8080")
    run = client.create_run_from_pipeline_func(
        sample_pipeline, arguments={"n_file": 20}
    )
    log.info(f"Submitted: {run}")


if __name__ == "__main__":
    logging.basicConfig(level="INFO")
    main()
