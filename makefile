
all: figure1.svg figure2.svg figure3.svg figure4.svg

# Figure rules
figure%.svg: genFigure.py msresist/figures/figure%.py
	./genFigure.py $*

$(fdir)/figure%pdf: $(fdir)/figure%svg
	rsvg-convert --keep-image-data -f pdf $< -o $@

test:
	pytest

testcover:
	pytest --junitxml=junit.xml --cov=msresist --cov-report xml:coverage.xml

clean:
	rm figure*.svg
