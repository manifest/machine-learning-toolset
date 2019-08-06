import pathlib
from setuptools import setup
from setuptools import find_packages

README = (pathlib.Path(__file__).parent / "README.md").read_text()

setup(
    name='mlts',
    version='0.3.0',
    description='Machine Learning Toolset',
    long_description=README,
    long_description_content_type="text/markdown",
    author='Andrei Nesterov',
    author_email='ae.nesterov@gmail.com',
    url='https://github.com/manifest/machine-learning-toolset',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
    ],
    python_requires='>=3.6',
    install_requires=[
        'pandas',
        'numpy',
    ],
    packages=find_packages()
)
