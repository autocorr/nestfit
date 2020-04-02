all:
	python3 setup.py build_ext --inplace

.PHONY : clean
clean : setup.py
	-rm -rf build/
	python3 setup.py clean --all
