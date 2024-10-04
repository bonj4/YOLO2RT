#!/usr/bin/env python
import os


from setuptools import setup
import warnings


with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)

here = os.path.abspath(os.path.dirname(__file__))

def read_requirements():
    with open('requirements.txt') as fp:
        return [row.strip() for row in fp if row.strip()]


about = {}
with open(os.path.join(here, 'YOLO2RT', 'version.py'), 'r') as f:
    exec(f.read(), about)


setup(name='YOLO2RT',
      version=about['VERSION'],
      description='YOLO2RT using TensorRT accelerate !',
      long_description=open('README.md').read(),
      long_description_content_type='text/markdown',
      url='https://github.com/bonj4/YOLO2RT',
      author='bonj4',
      license='MIT License',
      python_requires=">=3.8",
      packages=['YOLO2RT', 'YOLO2RT.commands', 'YOLO2RT.export','YOLO2RT.inferences', 'YOLO2RT.models'],
      entry_points='''
      [console_scripts]
      YOLO2RT=YOLO2RT.__main__:main
      ''',
      install_requires=read_requirements(),
      )