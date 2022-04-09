from azure.ml.component import dsl
from shrike.pipeline import AMLPipelineHelper


class HuggingfaceReddit(AMLPipelineHelper):
    def build(self, config):
        download_reddit_data = self.component_load("download-reddit-data")
        prepare_json_data = self.component_load("prepare-json-data")

        data_read = self.component_load("canary-data-read")
        split = self.component_load("split-data")

        @dsl.pipeline()
        def huggingface_reddit():
            reddit_conf = config.download_reddit
            reddit_step = download_reddit_data(**reddit_conf)
            reddit_output = reddit_step.outputs.output_data
            reddit_output.register_as(name="reddit-data", description=f"{reddit_conf}")

            prepare_step = prepare_json_data(
                input_directory=reddit_output, **config.prepare_json_data
            )
            _ = data_read(input_data=prepare_step.outputs.output_data)

        return huggingface_reddit

    def pipeline_instance(self, pipeline_function, config):
        return pipeline_function()


if __name__ == "__main__":
    HuggingfaceReddit.main()
