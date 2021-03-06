from setuptools import find_packages
from setuptools import setup

setup(name='mobile-unet',
      version='0.0.1',
      description='An ML model with U-shaped architecture with MobileNetV2 based encoders',
      url='https://github.com/Basars/mobile-unet.git',
      author='OrigamiDream',
      packages=find_packages(),
      zip_safe=False,
      include_package_data=True,
      install_requires=[
          'tensorflow>=2.0',
          'numpy',
      ],
      extra_require={
          'tests': ['pytest']
      })
