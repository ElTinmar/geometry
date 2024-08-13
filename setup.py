from distutils.core import setup

setup(
    name='geometry',
    author='Martin Privat',
    version='0.1.8',
    packages=['geometry','geometry.tests'],
    license='Creative Commons Attribution-Noncommercial-Share Alike license',
    description='geometry functions',
    long_description=open('README.md').read(),
    install_requires=[
        "numpy"
    ]
)