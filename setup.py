from setuptools import setup

setup(
    name='deepmoji',
    version='1.0',
    description='DeepMoji library',
    include_package_data=True,
    install_requires=[
        'emoji==0.4.5',
        'h5py==2.9.0',
        'Keras==2.0.9',
        'numpy==1.18.2',
        'scikit-learn==0.22.2.post1',
        'text-unidecode==1.0',
    ],
)
