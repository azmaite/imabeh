from setuptools import setup, find_packages

setup(
    name="imabeh",
    version="0.0.1",
    packages=[
        "imabeh", "imabeh.run"
    ],
    author="Maite Azcorra",
    author_email="maite.azcorrasedano@epfl.ch",
    description="Pipeline to pre-process simulanesouly recorded two-photon and behavioural data in flies.",

    long_description_content_type="text/x-rst",
    url="https://github.com/azcorra/imabeh",
    python_requires='>=3.7'
)
