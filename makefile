flist = $(wildcard msresist/figures/figureM*.py)

all: $(patsubst msresist/figures/figure%.py, output/figure%.svg, $(flist))

# Figure rules
output/figure%.svg: venv genFigure.py msresist/figures/figure%.py
	mkdir -p output
	. venv/bin/activate && ./genFigure.py $*

venv: venv/bin/activate

venv/bin/activate: requirements.txt
	test -d venv || virtualenv venv
	. venv/bin/activate && pip install --prefer-binary -Uqr requirements.txt
	touch venv/bin/activate

test: venv
	. venv/bin/activate && pytest -s -v -x msresist

testcover: venv
	. venv/bin/activate && pytest --junitxml=junit.xml --cov=msresist --cov-report xml:coverage.xml

clean:
	rm -rf *.pdf venv pylint.log output
