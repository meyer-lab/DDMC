flist = $(wildcard ddmc/figures/figureM*.py)

all: $(patsubst ddmc/figures/figure%.py, output/figure%.svg, $(flist))

output/figure%.svg: ddmc/figures/figure%.py
	@ mkdir -p ./output
	poetry run fbuild $*

test:
	poetry run pytest -s -x -v

testprofile:
	poetry run python3 -m cProfile -o profile -m pytest -s -v -x
	gprof2dot -f pstats --node-thres=5.0 profile | dot -Tsvg -o profile.svg

coverage.xml:
	poetry run pytest --cov=ddmc --cov-report=xml

clean:
	rm -rf *.pdf pylint.log output

mypy:
	poetry run mypy --install-types --non-interactive --ignore-missing-imports ddmc
