# Python &mapsto; executable

[Create a single executable from a Python project](https://stackoverflow.com/q/12059509/)

https://stackoverflow.com/a/60875372/

| Program | Size |
| - | - |
| Hello world | 11 MB |
| OmegaConf YAML parsing | 12 MB |

## Running

Run `gen.sh` to generate a clean Docker image `pyc` capable of compiling the Python
code, then invoking this image to generate an executable `cli`.

Next, run `run.sh` to invoke the generated executable in a minimal, non-Python Docker
image.
