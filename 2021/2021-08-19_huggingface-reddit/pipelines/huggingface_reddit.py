from azure.ml.component import dsl
from shrike.pipeline import AMLPipelineHelper


class HuggingfaceReddit(AMLPipelineHelper):
    def build(self, config):
        download_reddit_data = self.component_load("download-reddit-data")
        prepare_json_data = self.component_load("prepare-json-data")
        data_read = self.component_load("canary-data-read")
        gpu = self.component_load("canary-gpu")

        @dsl.pipeline()
        def huggingface_reddit():
            conf = config.download_reddit
            reddit_step = download_reddit_data(**conf)
            reddit_output = reddit_step.outputs.output_data
            reddit_output.register_as(name="reddit-data", description=f"{conf}")

            read_step = data_read(input_data=reddit_output)

            prepare_step = prepare_json_data(
                input_directory=reddit_output, **config.prepare_json_data
            )
            read_prepare_step = data_read(input_data=prepare_step.outputs.output_data)

            # gpu_step = gpu()
            # self.apply_smart_runsettings(gpu_step, gpu=True, node_count=2)

        return huggingface_reddit

    def pipeline_instance(self, pipeline_function, config):
        return pipeline_function()


if __name__ == "__main__":
    HuggingfaceReddit.main()
