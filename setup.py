from setuptools import setup
from setuptools import find_packages

with open("requirements.txt") as f:
    content = f.readlines()
requirements = [x.strip() for x in content if "git+" not in x]

setup(name='mlb',
      description="predicitng outcomes in MLB",
      install_requires=requirements,
      packages=find_packages(),
      )
