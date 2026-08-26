import logging
from pathlib import Path
from jinja2 import Template

import frontmatter
import yaml

log = logging.getLogger(__name__)


class JinjaRenderer:
    def render_files(self, root: Path, files: list[str]) -> list[dict]:
        """
        Build messages from `.md` prompt files and render them with the variables
        from `.yaml` files. Use `name.yaml@0` to select one index of a variable file.
        """
        log.debug("root = %s, files = %s", root, files)

        messages = []
        variables = {}

        for file in files:
            file, index = file.split("@") if "@" in file else (file, None)
            path = root / file
            if path.suffix == ".md":
                post = frontmatter.loads(path.read_text())
                content = Template(post.content).render(**variables)

                messages.append({"role": post["role"], "content": content})
            elif path.suffix == ".yaml":
                vars = yaml.safe_load(path.read_text())
                if index:
                    vars = vars[int(index)]
                variables |= vars
            else:
                raise RuntimeError(f"Unknown file type: {path.suffix}")

        return messages
