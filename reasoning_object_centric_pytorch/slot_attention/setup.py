# -*- coding: utf-8 -*-
from setuptools import setup

packages = \
['slot_attention']

package_data = \
{'': ['*']}

install_requires = \
[]

setup_kwargs = {
    'name': 'slot-attention',
    'version': '0.1.0',
    'description': 'PyTorch implementation of "Object-Centric Learning with Slot Attention"',
    'long_description': None,
    'author': 'Bryden Fogelman',
    'author_email': 'bryden1995@gmail.com',
    'maintainer': None,
    'maintainer_email': None,
    'url': None,
    'packages': packages,
    'package_data': package_data,
    'install_requires': install_requires,
    'python_requires': '>=3.7',
}


setup(**setup_kwargs)