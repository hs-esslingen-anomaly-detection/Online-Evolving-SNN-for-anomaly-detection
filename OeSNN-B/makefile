.PHONY: build
build:
	python3 setup.py build_ext --inplace

.PHONY: test
test:
	python3 test.py

.PHONY: update
update:
	git submodule update --remote

.PHONY: clean
clean:
	clear
	rm -rf build
	rm -rf __pycache__
	rm -f ./out/*
	rm -f ./src/OeSNNPythonWrapper.cpp
	rm -f OeSNNPythonWrapper.cpython*
	rm -rf log
	rm -rf .ipynb_checkpoints
