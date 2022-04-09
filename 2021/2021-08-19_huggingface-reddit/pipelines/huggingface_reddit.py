from azure.ml.component import dsl
from shrike.pipeline import AMLPipelineHelper


class HuggingfaceReddit(AMLPipelineHelper):
    def build(self, config):
        download_reddit_data = self.component_load("download-reddit-data")

        @dsl.pipeline()
        def huggingface_reddit():
            reddit_step = download_reddit_data(**config.download_reddit)

        return huggingface_reddit

    def pipeline_instance(self, pipeline_function, config):
        return pipeline_function()


if __name__ == "__main__":
    HuggingfaceReddit.main()
