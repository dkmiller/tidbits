# Namespace packages in Python

Figure out how to have multiple Python packages depending on each other like below,
with "native" `pyproject.toml` files and an easy development experience.

```mermaid
flowchart TD
    oryx --> oryx_core
    oryx_cli --> oryx
    oryx_cli --> oryx_core
```

## Setup

Simply run

```bash
pip install -e oryx-cli

# Simpler
pip install -e ./oryx/

# Works???
pip install -e oryx-core/ oryx/

pip install -e oryx-core -e oryx -e oryx-cli

# Works!
pip install -e oryx-core --config-settings editable_mode=strict

# Works!
pip install -e oryx-core -e oryx-operations -e oryx-cli --config-settings editable_mode=strict
```

## Links

https://stackoverflow.com/a/76616926/

https://packaging.python.org/en/latest/guides/packaging-namespace-packages/

https://peps.python.org/pep-0508/

https://stackoverflow.com/q/75159453/

https://github.com/pypa/pip/issues/6658

https://github.com/pdm-project/pdm/discussions/600

https://setuptools.pypa.io/en/latest/userguide/package_discovery.html

https://github.com/dkmiller/modern-python-package/

https://stackoverflow.com/a/76214384/

- VS Code approach is better than `editable_mode=strict` approach; no obvious way to do multiple
  of the latter in single `requirements.txt` file.

https://setuptools.pypa.io/en/latest/userguide/development_mode.html
