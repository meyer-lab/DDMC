
all: figure1.svg

# Figure rules
figure%.svg: genFigure.py msresist/figures/figure%.py
	./genFigure.py $*

test:
	pytest

testcover:
	pytest --junitxml=junit.xml --cov=msresist --cov-report xml:coverage.xml

clean:
	rm figure1.svg