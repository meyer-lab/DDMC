PACKAGE_NAME=pomegranate
PY3_ENV=py3.6

default:
	echo no default

.PHONY: install test bigclean nbtest nbclean
.PHONY: bigbuild py2build py3build
.PHONY: biginstall py2install py3install
.PHONY: biguninstall py2uninstall py3uninstall
.PHONY: bigtest py2test py3test
.PHONY: bignbtest py2nbtest py3nbtest

install:
	python setup.py install

test:
	python3 setup.py test

bigclean:
	rm -rf build
	rm -rf dist
	rm -rf .eggs
	rm -rf $(PACKAGE_NAME).egg-info
	find . -name '*.pyc' -print -delete
	find . -name '*.so' -print -delete
	find . -name '*.c' -print -delete


bigbuild: py3build
biginstall: py3install
biguninstall: py3uninstall
bigtest: py3test


py3build:
	(source activate $(PY3_ENV) ; python setup.py build ; python setup.py build_ext --inplace )

py3install:
	(source activate $(PY3_ENV) ; python setup.py install )

py3uninstall:
	rm -rf ~/miniconda2/envs/$(PY3_ENV)/lib/python3.6/site-packages/$(PACKAGE_NAME)*

py3test:
	(source activate $(PY3_ENV) ; python setup.py test )

EXECUTE_TIMEOUT=600
ALLOW_ERRORS=
