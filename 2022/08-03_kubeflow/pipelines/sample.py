import kfp
import kfp.components as comp
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

    show_data_step = show_data(input=gen_data_step.outputs["random_files"])
    show_data_r_step = show_data_r(input=gen_data_step.outputs["random_files"])

    # TODO: imitate these for distributed job.
    # - https://github.com/kubeflow/pipelines/blob/master/components/kubeflow/pytorch-launcher/sample.py
    # - https://github.com/kubeflow/pipelines/blob/master/samples/core/resource_ops/resource_ops.py
    # - https://github.com/kubeflow/training-operator/blob/master/examples/pytorch/simple.yaml
    # - https://github.com/kubeflow/pipelines/blob/master/samples/contrib/kubeflow-e2e-mnist/kubeflow-e2e-mnist.ipynb
    # - https://cloud.google.com/blog/topics/developers-practitioners/scalable-ml-workflows-using-pytorch-kubeflow-pipelines-and-vertex-pipelines
    #     - https://github.com/kubeflow/pipelines/blob/master/samples/contrib/pytorch-samples/Pipeline-Bert.ipynb

    pytorch_job_op = comp.load_component_from_url(
        "https://raw.githubusercontent.com/kubeflow/pipelines/master/components/kubeflow/pytorch-launcher/component.yaml"
    )

    # Copy-paste from: https://github.com/kubeflow/pipelines/blob/master/samples/contrib/pytorch-samples/Pipeline-Bert-Dist.ipynb
    namespace = "default"
    dataset_path = "dataset_path"
    checkpoint_dir = "__checkpoint_dir__"
    num_samples = "__num_samples__"
    tensorboard_root = "__tensorboard_root__"
    max_epochs = "__max_epochs__"
    gpus = 0
    num_nodes = "__num_nodes__"
    # confusion_matrix_url = "__confusion_matrix_url__"
    volume_mount_path = "__volume_mount_path__"
    dist_volume = "__dist_volume__"

    train_task = pytorch_job_op(
        name="pytorch-bert-dist", 
        namespace=namespace, 
        master_spec=
        {
          "replicas": 1,
          "imagePullPolicy": "Always",
          "restartPolicy": "OnFailure",
          "template": {
            "metadata": {
              "annotations": {
                "sidecar.istio.io/inject": "false"
              }
            },
            "spec": {
              "containers": [
                {
                  "name": "pytorch",
                  "image": "public.ecr.aws/pytorch-samples/kfp_samples:latest-gpu",
                  "command": ["python3", "bert/agnews_classification_pytorch.py"],
                  "args": [
                    "--dataset_path", dataset_path,
                    "--checkpoint_dir", checkpoint_dir,
                    "--script_args", f"model_name=bert.pth,num_samples={num_samples}",
                    "--tensorboard_root", tensorboard_root,
                    "--ptl_args", f"max_epochs={max_epochs},profiler=pytorch,gpus={gpus},accelerator=ddp,num_nodes={num_nodes}"
                  ],
                  "env": [
                    {
                        "name": "MINIO_ACCESS_KEY",
                        "valueFrom": {
                            "secretKeyRef": {
                                "name": "mlpipeline-minio-artifact",
                                "key": "accesskey",
                            }
                        },
                    },
                    {
                        "name": "MINIO_SECRET_KEY",
                        "valueFrom": {
                            "secretKeyRef": {
                                "name": "mlpipeline-minio-artifact",
                                "key": "secretkey",
                            }
                        },
                    }
                  ],
                  "ports": [
                    {
                      "containerPort": 24456,
                      "name": "pytorchjob-port"
                    }
                  ],
                  "resources": {
                    "limits": {
                      "nvidia.com/gpu": 2
                    }
                  },
                  "volumeMounts": [
                    {
                      "mountPath": volume_mount_path,
                      "name": "model-volume"
                    }
                  ]
                }
              ],
              "volumes": [
                {
                  "name": "model-volume",
                  "persistentVolumeClaim": {
                    "claimName": dist_volume
                  }
                }
              ]
            }
          }
        }, 
        worker_spec=
        {
          "replicas": 1,
          "imagePullPolicy": "Always",
          "restartPolicy": "OnFailure",
          "template": {
            "metadata": {
              "annotations": {
                "sidecar.istio.io/inject": "false"
              }
            },
            "spec": {
              "containers": [
                {
                  "name": "pytorch",
                  "image": "public.ecr.aws/pytorch-samples/kfp_samples:latest-gpu",
                  "command": ["python3", "bert/agnews_classification_pytorch.py"],
                  "args": [
                    "--dataset_path", dataset_path,
                    "--checkpoint_dir", checkpoint_dir,
                    "--script_args", f"model_name=bert.pth,num_samples={num_samples}",
                    "--tensorboard_root", tensorboard_root,
                    "--ptl_args", f"max_epochs={max_epochs},profiler=pytorch,gpus={gpus},accelerator=ddp,num_nodes={num_nodes}"
                  ],
                  "env": [
                    {
                        "name": "MINIO_ACCESS_KEY",
                        "valueFrom": {
                            "secretKeyRef": {
                                "name": "mlpipeline-minio-artifact",
                                "key": "accesskey",
                            }
                        },
                    },
                    {
                        "name": "MINIO_SECRET_KEY",
                        "valueFrom": {
                            "secretKeyRef": {
                                "name": "mlpipeline-minio-artifact",
                                "key": "secretkey",
                            }
                        },
                    }
                  ],
                  "ports": [
                    {
                      "containerPort": 24456,
                      "name": "pytorchjob-port"
                    }
                  ],
                  "resources": {
                    "limits": {
                      "nvidia.com/gpu": 2
                    }
                  },
                  "volumeMounts": [
                    {
                      "mountPath": volume_mount_path,
                      "name": "model-volume"
                    }
                  ]
                }
              ],
              "volumes": [
                {
                  "name": "model-volume",
                  "persistentVolumeClaim": {
                    "claimName": dist_volume
                  }
                }
              ]
            }
          }
        },
        delete_after_done=False
    ).after(gen_data_step)


def main():
    host = "http://localhost:8080"
    client = kfp.Client(host=host)
    run = client.create_run_from_pipeline_func(
        sample_pipeline, arguments={"n_file": 100}  # type: ignore
    )

    run_link = f"{host}/#/runs/details/{run.run_id}"
    log.info(f"Submitted:\n\n\t{run_link}\n")


if __name__ == "__main__":
    logging.basicConfig(level="INFO")
    main()
