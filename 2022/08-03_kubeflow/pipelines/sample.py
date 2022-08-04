import kfp
import kfp.components as comp


def component(name: str):
    path = f"components/{name}/component.yaml"
    rv = comp.load_component_from_file(path)
    return rv


def sample_pipeline():
    create_step_gen_data = component("gen_data")
    create_step_show_data = component("show_data")

    gen_data_step = create_step_gen_data(n_files=10)

    show_data_step = create_step_show_data(input=gen_data_step.outputs["random_files"])


def main():
    client = kfp.Client(host="http://localhost:8080")
    run = client.create_run_from_pipeline_func(sample_pipeline, arguments={})
    print(f"Submitted: {run}")


if __name__ == "__main__":
    main()
