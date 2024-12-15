import sys
from setuptools import setup, find_packages

if sys.version_info.major != 3:
    print("This Python is only compatible with Python 3, but you are running "
          "Python {}. The installation will likely fail.".format(sys.version_info.major))


setup(
    name='HandsOnVLM',
    version='0.1.0',
    packages=find_packages(),
    url='https://github.com/Kami-code/HandsOnVLM-release',
    license='',
    install_requires=[
        'transformers==4.31.0',
        'accelerate==0.21.0',
        'decord==0.6.0',
        'deepspeed==0.9.5',
        'einops==0.6.1',
        'einops-exts==0.0.4',
        'lmdb==1.5.1',
        'lmdbdict==0.2.2',
        'numpy==1.26.4',
        'ninja==1.11.1.1',
        'openai==1.54.4',
        'opencv-python==4.10.0.84',
        'peft==0.4.0',
        'scikit-learn==1.2.2',
        'scipy==1.13.1',
        'tensorboardX==2.6.2.2',
        'timm==0.6.13',
        'wandb==0.18.1',
        'pandas==2.2.2'
    ],
)