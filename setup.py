import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()
    
setuptools.setup(
    name='',
    version='1.0.0',
    author='552 A4',
    author_email='joseph.horeczy@mail.mcgill.ca',
    description='A4',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/chavezrodz/A4',
    license='MIT',
    packages=['models, data'],
    install_requires=[
      'torch',
      'pandas',
      'pytorch_lightning',
      'pytorch_forecast'
    ],
)
