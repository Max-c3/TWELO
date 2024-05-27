from setuptools import find_packages
from setuptools import setup

with open("requirements.txt") as f:
    content = f.readlines()
requirements = [x.strip() for x in content if "git+" not in x]

setup(name='kestrix',
      version="0.0.1",
      description="Identification and Obfuscation of personal data in drone images",
      license="MIT",
      author="François-Xavier Foray, Tim Hildebrandt, Max Kreuz, Tatiana Lupashina",
      author_email="contact@lewagon.org",
      url="https://github.com/Max-c3/Kestrix_Project",
      install_requires=requirements,
      packages=find_packages(),
    #   test_suite="tests",
      # include_package_data: to install data from MANIFEST.in
      include_package_data=True,
      zip_safe=False)
