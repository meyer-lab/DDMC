flist = $(wildcard ddmc/figures/figureM*.py)

all: $(patsubst ddmc/figures/figure%.py, output/figure%.svg, $(flist))

output/figure%.svg: ddmc/figures/figure%.py
	@ mkdir -p ./output
	uv run fbuild $*

test:
	uv run pytest -s -x -v

testprofile:
	uv run python3 -m cProfile -o profile -m pytest -s -v -x
	gprof2dot -f pstats --node-thres=5.0 profile | dot -Tsvg -o profile.svg

coverage:
	uv run pytest --cov=ddmc --cov-report=term

clean:
	rm -rf *.pdf pylint.log output

lint:
	uv run ruff check ddmc
	uv run ruff format --check ddmc

typecheck:
	uv run ty check ddmc

docs:
	uv run mkdocs serve

docs-build:
	uv run mkdocs build --strict
