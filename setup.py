import os.path
import setuptools

DESCRIPTION = "A Tensorflow implementation of Fastformer: Additive Attention Can Be All You Need in TensorFlow"

curr_directory = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(curr_directory, "requirements.in")) as requirement_file:
    REQUIRED_PACKAGES = requirement_file.read().splitlines()

with open(os.path.join(curr_directory, "VERSION")) as version_file:
    VERSION = version_file.read().strip()

with open(os.path.join(curr_directory, "README.md"), encoding="utf-8") as f:
    LONG_DESCRIPTION = f.read()

setuptools.setup(
    name="fastformer-tf",
    author="Alaa Awad",
    author_email="alaa.awad.mail@gmail.com",
    url="https://github.com/awadalaa/Fastformer",
    description=DESCRIPTION,
    long_description_content_type="text/markdown",
    long_description=LONG_DESCRIPTION,
    version=VERSION,
    install_requires=REQUIRED_PACKAGES,
    extras_require={
        "dev": [
            "check-manifest",
            "black",
        ],
    },
    packages=setuptools.find_packages(),
)