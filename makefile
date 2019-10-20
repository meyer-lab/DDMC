
all: figure1.svg figure2.svg figure3.svg figure4.svg

# Figure rules
figure%.svg: genFigure.py msresist/figures/figure%.py
	./genFigure.py $*

$(fdir)/figure%pdf: $(fdir)/figure%svg
	rsvg-convert --keep-image-data -f pdf $< -o $@

test:
	pytest

testprofile:
	python3 -m cProfile -o profile /usr/local/bin/pytest
	gprof2dot -f pstats --node-thres=5.0 profile | dot -Tsvg -o profile.svg

testcover:
	pytest --junitxml=junit.xml --cov=msresist --cov-report xml:coverage.xml

clean:
	rm figure*.svg
