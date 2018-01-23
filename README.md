# lensing
Python library of useful functions for gravitational lensing.

Installation
1. Download directory
2. ```cd``` to directory with the ```setup.py``` file.
3. type ```pip install .``` in command line
4. You're done. You can now import the functions as so.

```
from lensing.formalism import dlsds
zl=0.5
zs=2
lensing_ratio = dlsds(zl,zs)
'''

