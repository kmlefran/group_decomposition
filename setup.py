import setuptools
setuptools.setup(name='group_decomposition',
version='0.3.1',
description='Extract fragments and counts of fragments from molecule SMILES codes',
url='https://github.com/kmlefran/group_decomposition',
author='Kevin M. Lefrancois-Gagnon',
install_requires=['numpy','pandas','rdkit'],
author_email='kgagnon@lakeheadu.ca',
packages=setuptools.find_packages(),
zip_safe=False)