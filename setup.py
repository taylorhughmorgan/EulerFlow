#!/usr/bin/env python

from setuptools import setup

setup(name='EulerFlow',
      version='0.1.0',
      description='Solve the Euler Equations for inviscid compressible flow in 1D.',
      long_description='Solve the Euler Equations for inviscid, incompressible flow in 1D cartesian, cylindrical, and spherical coordinates. Validate solution against Taylor-Von Neumann-Sedov solution.',
      author='Hugh Morgan',
      author_email='serebrum@gmail.com',
      url='https://github.com/taylorhughmorgan/EulerFlow',
      packages=['EulerFlow'],
     )