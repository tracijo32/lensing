from setuptools import setup

setup(name='lensing',
      version='0.1',
      description='Useful functions for gravitational lensing',
      url='http://github.com/tracijo32/lensing',
      author='Traci Johnson',
      author_email='tljohn@umich.edu',
      license='MIT',
      packages=['lensing'],
      install_requires=['numpy','astropy','scipy'],
      zip_safe=False)
