#!/bin/bash

if ! command -v docker >/dev/null; then
    echo "please install docker with user rights (sudo pacman -Sy docker)"
    exit 1
fi

if ! command -v pandoc >/dev/null; then
    echo "please install pandoc (sudo pacman -Sy pandoc)"
    exit 1
fi

if ! command -v gcc >/dev/null; then
    echo "please install gcc (sudo pacman -Sy gcc)"
    exit 1
fi

if ! command -v make >/dev/null; then
    echo "please install make (sudo pacman -Sy make)"
    exit 1
fi

if ! python -c 'import optuna'; then
    echo "please install optuna (pip install optuna)"
    exit 1
fi

if ! python -c 'import sklearn'; then
    echo "please install sklearn (pip install sklearn)"
    exit 1
fi

if ! python -c 'import pandas'; then
    echo "please install pandas (pip install pandas)"
    exit 1
fi

if ! python -c 'import Cython'; then
    echo "please install Cython (pip install Cython)"
    exit 1
fi

if ! python -c 'import matplotlib'; then
    echo "please install matplotlib (pip install matplotlib)"
    exit 1
fi

if ! python -c 'import MySQLdb'; then
    echo "please install MySQLdb (sudo pacman -Sy python-mysqlclient)"
    exit 1
fi

echo "config:"
cat ./config.sh

set -e
mkdir -p ./result

if [ ! -f ./result/OeSNN-A.json ]; then
    pushd ./OeSNN-A
    bash run_with_optuna.sh
    popd
    cp -vf ./OeSNN-A/result.json ./result/OeSNN-A.json
fi

if [ ! -f ./result/OeSNN-B.json ]; then
    pushd ./OeSNN-B
    bash run_with_optuna.sh
    popd
    cp -vf ./OeSNN-B/result.json ./result/OeSNN-B.json
fi

if [ ! -f ./result/OeSNN-C.json ]; then
    pushd ./OeSNN-C
    bash run_with_optuna.sh
    popd
    cp -vf ./OeSNN-C/result.json ./result/OeSNN-C.json
fi

if [ ! -f ./result/OeSNN-D.json ]; then
    pushd ./OeSNN-D
    bash run_with_optuna.sh
    popd
    cp -vf ./OeSNN-D/result.json ./result/OeSNN-D.json
fi

if [ ! -f ./result/SPIRIT.json ]; then
    pushd ./Spirit
    rm -rf build
    mkdir -p build
    pushd build
    cmake ..
    make
    cp -f libspirit.so ..
    cp -f spirit_module.cpython-*-x86_64-linux-gnu.so ..
    popd
    python test.py
    popd
    cp -vf ./Spirit/result.json ./result/SPIRIT.json
fi

python visu_result.py
pandoc result.md -o result.pdf --metadata-file=markdown.yml --pdf-engine=xelatex
echo "done"
