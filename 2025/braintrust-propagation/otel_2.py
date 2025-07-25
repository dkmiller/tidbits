import logging
import os, json

from braintrust import Eval, init_dataset, current_span
from braintrust.otel import BraintrustSpanProcessor
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider

from metrics import precision_recall_score

# logging.basicConfig(level="DEBUG")

# Set up the tracer provider
provider = TracerProvider()
trace.set_tracer_provider(provider)

# Create and add the Braintrust span processor
processor = BraintrustSpanProcessor(
    api_key=os.environ["BRAINTRUST_API_KEY"],
    api_url=os.environ["BRAINTRUST_API_URL"],
    parent=f"project_name:{os.environ['BRAINTRUST_PROJECT_NAME']}",
    filter_ai_spans=False  # Keep all spans for this example
)
# The BraintrustSpanProcessor implements the SpanProcessor interface
provider.add_span_processor(processor)  # type: ignore

# Create a tracer
tracer = trace.get_tracer(__name__)


def fake_agent(input: str) -> list[str]:
    # Get the current Braintrust span and export it to get the parent ID
    current_braintrust_span = current_span()
    parent_export = current_braintrust_span.export()

    # print(f"Current Braintrust span export: {parent_export}")

    # Create a temporary processor with the current span as parent
    temp_processor = BraintrustSpanProcessor(
        parent=parent_export,
        filter_ai_spans=False
    )

    # Add the temporary processor
    provider.add_span_processor(temp_processor)  # type: ignore

    try:
        # Create an OpenTelemetry span that should be a child of the current Braintrust span
        with tracer.start_as_current_span("fake_agent") as otel_span:
            # Use the correct Braintrust attribute names for OpenTelemetry
            result = ["blah"]
            otel_span.set_attribute("braintrust.input_json", json.dumps(input))
            otel_span.set_attribute("braintrust.output_json", json.dumps(result))
            # these other attributes are not necessary but this is how you would set them if you wanted to
            otel_span.set_attribute("braintrust.metadata", json.dumps({"source": "otel_span"}))
            otel_span.set_attribute("braintrust.span_attributes", json.dumps({"some_attribute": "some value"}))
            otel_span.set_attribute("braintrust.scores", json.dumps({"some_score": "some value"}))
            otel_span.set_attribute("braintrust.expected", json.dumps({"expected": "expected value if you pass it"}))

            return result
    finally:
        # Clean up the temporary processor
        temp_processor.shutdown()


def dummy_task(input: str) -> list[str]:
    # The task span will be created by Braintrust Eval framework
    # fake_agent will create a child span of the current active span
    return fake_agent(input)

test_dataset = [
    {
        "name": "Perfect match",
        "input": "Extract fruits from this text: I like apples, bananas, and oranges.",
        "output": ["apples", "bananas", "oranges"],
        "expected": ["apples", "bananas", "oranges"],
        "description": "Output exactly matches expected - should get perfect precision and recall (1.0 each)"
    },
    {
        "name": "Partial match - missing items",
        "input": "Extract fruits from this text: I like apples, bananas, and oranges.",
        "output": ["apples", "bananas"],
        "expected": ["apples", "bananas", "oranges"],
        "description": "Output missing one expected item - high precision, lower recall"
    },
    {
        "name": "Partial match - extra items",
        "input": "Extract fruits from this text: I like apples, bananas, and oranges.",
        "output": ["apples", "bananas", "oranges", "grapes"],
        "expected": ["apples", "bananas", "oranges"],
        "description": "Output has extra item - lower precision, perfect recall"
    },
    {
        "name": "Mixed partial match",
        "input": "Extract colors from this text: The sky is blue, grass is green, and roses are red.",
        "output": ["blue", "green", "yellow"],
        "expected": ["blue", "green", "red"],
        "description": "Some correct, some missing, some extra - tests balanced precision/recall"
    },
    {
        "name": "No matches",
        "input": "Extract animals from this text: I like apples, bananas, and oranges.",
        "output": ["dog", "cat"],
        "expected": ["elephant", "tiger"],
        "description": "No overlap between output and expected - should get 0 precision and recall"
    },
    {
        "name": "Empty output",
        "input": "Extract numbers from this text: Hello world!",
        "output": [],
        "expected": ["one", "two", "three"],
        "description": "Empty output with non-empty expected - 0 precision, 0 recall"
    },
    {
        "name": "Empty expected",
        "input": "Extract nothing from this text: Hello world!",
        "output": ["hello", "world"],
        "expected": [],
        "description": "Non-empty output with empty expected - 0 precision, perfect recall"
    },
    {
        "name": "Both empty",
        "input": "Extract nothing from empty text.",
        "output": [],
        "expected": [],
        "description": "Both empty - should handle gracefully"
    },
    {
        "name": "Duplicate items in output",
        "input": "Extract repeated words: apple apple banana banana.",
        "output": ["apple", "apple", "banana"],
        "expected": ["apple", "banana"],
        "description": "Tests how duplicates in output are handled"
    },
    {
        "name": "Case sensitivity test",
        "input": "Extract names: John, MARY, bob.",
        "output": ["John", "mary", "Bob"],
        "expected": ["John", "MARY", "bob"],
        "description": "Tests case sensitivity in matching"
    },
    {
        "name": "Large lists",
        "input": "Extract many items from a long list.",
        "output": [f"item_{i}" for i in range(1, 21)],  # item_1 to item_20
        "expected": [f"item_{i}" for i in range(10, 31)],  # item_10 to item_30
        "description": "Tests performance with larger lists - partial overlap"
    },
    {
        "name": "Single item correct",
        "input": "Extract the main topic: Machine Learning",
        "output": ["machine learning"],
        "expected": ["machine learning"],
        "description": "Single item perfect match"
    },
    {
        "name": "Single item incorrect",
        "input": "Extract the main topic: Machine Learning",
        "output": ["artificial intelligence"],
        "expected": ["machine learning"],
        "description": "Single item complete miss"
    }
]

Eval(
    os.environ['BRAINTRUST_PROJECT_NAME'],
    data=test_dataset,
    task=dummy_task,
    scores=[precision_recall_score],
    experiment_name="new_otel_test",
)