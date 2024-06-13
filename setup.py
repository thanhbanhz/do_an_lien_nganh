from setuptools import setup, find_packages

setup(
    name='my_project',
    version='0.1',
    packages=find_packages(),
    package_data={'main_scr': ['main_scr/*']},
    install_requires=[
        # Add your project dependencies here
    ],
    author='thanh',
    author_email='21011080',
    description='daln',
)
