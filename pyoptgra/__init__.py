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

from ._about import __version__  # noqa
from .khan import (  # noqa
    base_khan_function,
    inverse_triangular_wave,
    khan_function_sin,
    khan_function_tanh,
    khan_function_triangle,
    triangular_wave_fourier,
    triangular_wave_fourier_grad,
)
from .optgra import optgra  # noqa
