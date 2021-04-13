flist = 1 2 3 S1 S2 S3 M2 M3 M4 M5 MS2 MS3 MS4 MS5 MS6 MS7

all: $(patsubst %, figure%.svg, $(flist))

# Figure rules
figure%.svg: venv genFigure.py msresist/figures/figure%.py
	. venv/bin/activate && ./genFigure.py $*

venv: venv/bin/activate

venv/bin/activate: requirements.txt
	test -d venv || virtualenv venv
	. venv/bin/activate && pip install Cython scipy==1.6.2 numpy==1.20.2
	. venv/bin/activate && pip install --prefer-binary -Uqr requirements.txt
	touch venv/bin/activate

output/%/manuscript.md: venv manuscripts/%/*.md
	mkdir -p ./output/%
	. venv/bin/activate && manubot process --content-directory=manuscripts/$*/ --output-directory=output/$*/ --cache-directory=cache --skip-citations --log-level=INFO
	git remote rm rootstock

output/%/manuscript.html: venv output/%/manuscript.md $(patsubst %, figure%.svg, $(flist))
	cp *.svg output/$*/
	. venv/bin/activate && pandoc --verbose \
		--defaults=./common/templates/manubot/pandoc/common.yaml \
		--defaults=./common/templates/manubot/pandoc/html.yaml \
		--output=output/$*/manuscript.html output/$*/manuscript.md

output/%/manuscript.docx: venv output/%/manuscript.md $(patsubst %, figure%.svg, $(flist))
	cp *.svg output/$*/
	. venv/bin/activate && pandoc --verbose \
		--defaults=./common/templates/manubot/pandoc/common.yaml \
		--defaults=./common/templates/manubot/pandoc/docx.yaml \
		--output=output/$*/manuscript.docx output/$*/manuscript.md

test: venv
	. venv/bin/activate && pytest -s -v -x

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
	rm -rf *.pdf output venv pylint.log figure*.svg