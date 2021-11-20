flist = 2 S1 S2 S3 S4

all: $(patsubst %, output/biol/figure%.svg, $(flist)) $(patsubst %, output/method/figure%.svg, $(fmlist))

# Figure rules
output/method/figureM%.svg: venv genFigure.py msresist/figures/figureM%.py
	. venv/bin/activate && ./genFigure.py M$*

output/biol/figure%.svg: venv genFigure.py msresist/figures/figure%.py
	. venv/bin/activate && ./genFigure.py $*

venv: venv/bin/activate

venv/bin/activate: requirements.txt
	test -d venv || virtualenv venv
	. venv/bin/activate && pip install --prefer-binary -Uqr requirements.txt
	touch venv/bin/activate

output/%/manuscript.md: venv manuscripts/%/*.md
	. venv/bin/activate && manubot process --content-directory=manuscripts/$*/ --output-directory=output/$*/ --cache-directory=cache --skip-citations --log-level=INFO
	git remote rm rootstock

output/%/manuscript.html: venv output/%/manuscript.md $(patsubst %, output/biol/figure%.svg, $(flist)) $(patsubst %, output/method/figure%.svg, $(fmlist))
	. venv/bin/activate && pandoc --verbose \
		--defaults=common.yaml \
		--defaults=./common/templates/manubot/pandoc/html.yaml \
		--output=output/$*/manuscript.html output/$*/manuscript.md

output/%/manuscript.docx: venv output/%/manuscript.md $(patsubst %, output/biol/figure%.svg, $(flist)) $(patsubst %, output/method/figure%.svg, $(fmlist))
	. venv/bin/activate && pandoc --verbose \
		--defaults=common.yaml \
		--defaults=./common/templates/manubot/pandoc/docx.yaml \
		--output=output/$*/manuscript.docx output/$*/manuscript.md

test: venv
	. venv/bin/activate && pytest -s -v -x msresist

testprofile: venv
	. venv/bin/activate && python3 -m cProfile -o profile /usr/local/bin/pytest -s
	. venv/bin/activate && gprof2dot -f pstats --node-thres=5.0 profile | dot -Tsvg -o profile.svg

figprofile: venv
	. venv/bin/activate && python3 -m cProfile -o profile genFigure.py M2
	. venv/bin/activate && python3 -m gprof2dot -f pstats --node-thres=5.0 profile | dot -Tsvg -o profile.svg

testcover: venv
	. venv/bin/activate && pytest --junitxml=junit.xml --cov=msresist --cov-report xml:coverage.xml

%.pdf: %.ipynb
	. venv/bin/activate && jupyter nbconvert --execute --ExecutePreprocessor.timeout=6000 --to pdf $< --output $@

clean:
	rm -rf *.pdf venv pylint.log
	git checkout HEAD -- output
	git clean -ffdx output
