# Copyright 2008, 2021 European Space Agency
#
# This file is part of pyoptgra, a pygmo affiliated library.
#
# This Source Code Form is available under two different licenses.
# You may choose to license and use it under version 3 of the
# GNU General Public License or under the
# ESA Software Community Licence (ESCL) 2.4 Weak Copyleft.
# We explicitly reserve the right to release future versions of
# Pyoptgra and Optgra under different licenses.
# If copies of GPL3 and ESCL 2.4 were not distributed with this
# file, you can obtain them at https://www.gnu.org/licenses/gpl-3.0.txt
# and https://essr.esa.int/license/european-space-agency-community-license-v2-4-weak-copyleft

import sys

from skbuild import setup

# load version number
about = dict()
exec(open("pyoptgra/_about.py").read(), about)
version = about["__version__"]

setup(
    name="pyoptgra",
    version=version,
    packages=["pyoptgra"],
    setup_requires=["cmake", "ninja"],
    install_requires=["pygmo>=2.16"],
    cmake_install_dir="pyoptgra/core",
)
