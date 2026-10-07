from dataclasses import dataclass

from openai import AsyncOpenAI
from rich.console import Console
from rich.text import Text


class RichStreamPrinter:
    """Immediately print styled streaming text, wrapped on word boundaries.

    A line breaks at the first whitespace at or after column `width`, so lines
    can run past `width` but words are never split.
    """

    def __init__(self, width: int = 60) -> None:
        if width < 1:
            raise ValueError("width must be at least 1")

        self.width = width
        self.column = 0
        self._just_wrapped = False
        self.console = Console(width=width, highlight=False)

    def write(self, text: str, *, style: str = "") -> None:
        while text:
            # Preserve newlines already present in the model output.
            if text[0] == "\n":
                text = text[1:]

                # Avoid two line breaks when an explicit newline occurs
                # exactly where the wrap occurred.
                if self._just_wrapped:
                    self._just_wrapped = False
                else:
                    self.console.print()
                    self.column = 0

                continue

            # The line break replaces the whitespace it wrapped on, so a
            # wrapped line never starts with the rest of that whitespace run.
            if self._just_wrapped and text[0].isspace():
                text = text[1:]
                continue

            self._just_wrapped = False

            # Only consider text up to the next explicit newline.
            newline_at = text.find("\n")
            current_line = text if newline_at == -1 else text[:newline_at]

            if self.column < self.width:
                # Room left before the wrap column: print up to it, then let
                # the next pass look for whitespace to break on.
                fragment = current_line[: self.width - self.column]
                wrap_at = None
            else:
                # At or past the wrap column: break on the first whitespace,
                # or keep the word going if none has arrived yet.
                wrap_at = self._first_space(current_line)
                fragment = current_line if wrap_at is None else current_line[:wrap_at]

            if fragment:
                self.console.print(
                    Text(fragment, style=style),
                    end="",
                    soft_wrap=True,
                )

                self.column += len(fragment)

            text = text[len(fragment) :]

            if wrap_at is not None:
                self.console.print()
                self.column = 0
                self._just_wrapped = True

                # Drop the whitespace the line break stands in for.
                text = text[1:]

    @staticmethod
    def _first_space(line: str) -> int | None:
        for index, character in enumerate(line):
            if character.isspace():
                return index

        return None

    def finish(self) -> None:
        """End the final partial line, if there is one."""
        if self.column:
            self.console.print()

        self.column = 0
        self._just_wrapped = False


@dataclass
class ModelResponse:
    content: str
    reasoning_content: str | None = None


@dataclass
class ModelInvoker:
    model: str
    max_tokens: int
    endpoint: str
    printer: RichStreamPrinter = RichStreamPrinter(width=60)

    @property
    def openai(self):
        return AsyncOpenAI(base_url=self.endpoint, api_key="placeholder")

    async def invoke(self, messages: list[dict]) -> ModelResponse:
        stream = await self.openai.chat.completions.create(
            model=self.model,
            messages=messages,  # type: ignore
            max_completion_tokens=self.max_tokens,
            stream=True,
        )

        rv_content = []
        rv_reasoning_content = []

        async for chunk in stream:
            if not chunk.choices:
                continue

            delta = chunk.choices[0].delta
            if content := delta.content:
                self.printer.write(content)
                rv_content.append(content)
            elif reasoning_content := getattr(delta, "reasoning_content", None):
                self.printer.write(reasoning_content, style="italic")
                rv_reasoning_content.append(reasoning_content)
            else:
                self.printer.write(f"{delta=}\n")

        self.printer.finish()

        return ModelResponse(
            content="".join(rv_content),
            reasoning_content="".join(rv_reasoning_content)
            if rv_reasoning_content
            else None,
        )
