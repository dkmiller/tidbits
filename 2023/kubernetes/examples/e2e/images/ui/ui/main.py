from pathlib import Path

from streamlit.web import bootstrap


def main():
    # https://stackoverflow.com/a/76130057/
    script = (Path(__file__).parent / "app.py").absolute()
    bootstrap.run(str(script), False, [], {})
