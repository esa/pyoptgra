#!/bin/bash

set -ex

function repair_wheel {
    wheel="$1"
    if ! auditwheel show "$wheel"; then
        echo "Skipping non-platform wheel $wheel"
    else
        auditwheel repair "$wheel" -w /wheelhouse/repaired
    fi
}

for PY in cp310-cp310 cp36-cp36m cp37-cp37m cp38-cp38 cp39-cp39; do
    mkdir -p /wheelhouse/built/$PY
    /opt/python/$PY/bin/pip wheel . --no-deps -w /wheelhouse/built/$PY -v
    repair_wheel "/wheelhouse/built/$PY/"*.whl
done