from setuptools import setup

setup(
    name='deepmoji',
    version='1.0',
    packages=['deepmoji'],
    description='DeepMoji library',
    install_requires=[
        'emoji==0.4.5',
        'numpy==1.18.2',
        'scikit-learn==0.22.2.post1',
        'text-unidecode==1.0',
    ],
)
