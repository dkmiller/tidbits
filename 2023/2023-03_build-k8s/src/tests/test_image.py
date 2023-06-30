from pathlib import Path

import pytest

from ptah.core.image import ImageDefinition


@pytest.mark.parametrize(
    "path,expected",
    [
        ("a/b/Dockerfile", "b"),
        ("a/b/dockerfile", "b"),
        ("a/b/c.Dockerfile", "c"),
    ],
)
def test_image_definition_name(path, expected):
    definition = ImageDefinition(Path(path))
    assert definition.name == expected


def test_image_definition_tag(tmpdir):
    p = tmpdir.join("Dockerfile")
    p.write("FROM python:3.8\n")

    definition = ImageDefinition(Path(tmpdir))
    tag1 = definition.tag

    p.write("# comment\n")

    tag2 = definition.tag

    assert tag1 != tag2, "Changing the snapshot should change the tag"
