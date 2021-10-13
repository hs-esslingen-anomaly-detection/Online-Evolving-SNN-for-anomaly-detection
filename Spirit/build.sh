#!/bin/bash

rm -rf build
mkdir -p build
cd build
cmake ..
make
cp -fv libspirit.so ..
cp -fv spirit_module.cpython-*.so ..
cd ..
