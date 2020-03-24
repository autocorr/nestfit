all:
	python setup.py build_ext --inplace

clean:
	-rm -rf build/
	-rm nestfit/wrapper.c nestfit/wrapper.html nestfit/wrapped*.so
