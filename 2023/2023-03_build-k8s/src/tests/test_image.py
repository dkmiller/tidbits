from pathlib import Path

import pytest

from ptah.core.image import ImageClient, ImageDefinition


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


def test_image_client_image_definitions(tmpdir):
    p1 = tmpdir.mkdir("a").join("Dockerfile")
    p1.write("FROM python:3.8\n")

    p2 = tmpdir.mkdir("sub").join("b.dockerfile")
    p2.write("FROM python:3.9")

    client = ImageClient(tmpdir)
    images = client.image_definitions()
    assert sorted([i.name for i in images]) == ["a", "b"]
