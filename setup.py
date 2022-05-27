import pathlib
from setuptools import setup

HERE = pathlib.Path(__file__).parent

README = (HERE / "README.md").read_text()

setup(setup_requires=['wheel'],
      name="polyloss",
      version="0.0.2",
      description="pytorch implementation of poly loss",
      long_description=README,
      long_description_content_type="text/markdown",
      url="https://github.com/jahongir7174/PolyLoss",
      author="Jahongir Yunusov",
      author_email="jahongir7174@gmail.com",
      license="MIT",
      classifiers=["License :: OSI Approved :: MIT License",
                   "Programming Language :: Python :: 3.6",
                   "Programming Language :: Python :: 3.9"],
      packages=["polyloss"],
      install_requires=["torch"])
