CXX ?= g++
CXXFLAGS ?= -O3 -march=native -std=c++17 -Wall -Wextra -Wpedantic
PYTHON ?= .venv/bin/python
QAT_PYTHON ?= .venv/qat/bin/python

.PHONY: all test test-qat benchmark sanitize clean
all: libpentanary.so native_tests

libpentanary.so: pentanary_loader.cpp kernel.cpp pentanary.h
	$(CXX) $(CXXFLAGS) -flto -fPIC -shared pentanary_loader.cpp kernel.cpp -o $@

native_tests: tests/native_tests.cpp pentanary_loader.cpp kernel.cpp pentanary.h
	$(CXX) $(CXXFLAGS) -I. tests/native_tests.cpp pentanary_loader.cpp kernel.cpp -o $@

test: all
	./native_tests
	$(PYTHON) -m pytest -q

test-qat: all
	OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 $(QAT_PYTHON) -m pytest -q tests/test_qat.py

benchmark: all
	$(PYTHON) benchmark.py

sanitize:
	$(CXX) -O1 -g -std=c++17 -fsanitize=address,undefined -fno-omit-frame-pointer -I. tests/native_tests.cpp pentanary_loader.cpp kernel.cpp -o native_tests_sanitized
	./native_tests_sanitized

clean:
	rm -f libpentanary.so native_tests native_tests_sanitized
