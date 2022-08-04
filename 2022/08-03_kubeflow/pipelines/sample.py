import kfp
import kfp.components as comp


def sample_pipeline():
    create_step_gen_data = comp.load_component_from_file(
        "components/gen_data/component.yaml"
    )

    gen_data_step = create_step_gen_data(n_files=10)


def main():
    client = kfp.Client(host="http://localhost:8080/")
    run = client.create_run_from_pipeline_func(sample_pipeline, arguments={})
    print(f"Submitted: {run}")


if __name__ == "__main__":
    main()
