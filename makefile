
all: figure1.svg figure2.svg figure3.svg figure4.svg output/method/manuscript.html output/biol/manuscript.html

# Figure rules
figure%.svg: venv genFigure.py msresist/figures/figure%.py
	. venv/bin/activate && ./genFigure.py $*

venv: venv/bin/activate

venv/bin/activate: requirements.txt
	test -d venv || virtualenv --system-site-packages venv
	. venv/bin/activate && pip install -Uqr requirements.txt
	touch venv/bin/activate

output/%/manuscript.md: venv manuscripts/%/*.md
	mkdir -p ./output/%
	. venv/bin/activate && manubot process --content-directory=manuscripts/$*/ --output-directory=output/$*/ --log-level=WARNING

output/%/manuscript.html: venv output/%/manuscript.md
	. venv/bin/activate && pandoc \
		--from=markdown --to=html5 --filter=pandoc-fignos --filter=pandoc-eqnos --filter=pandoc-tablenos \
		--bibliography=output/$*/references.json \
		--csl=common/templates/manubot/style.csl \
		--metadata link-citations=true \
		--include-after-body=common/templates/manubot/default.html \
		--include-after-body=common/templates/manubot/plugins/table-scroll.html \
		--include-after-body=common/templates/manubot/plugins/anchors.html \
		--include-after-body=common/templates/manubot/plugins/accordion.html \
		--include-after-body=common/templates/manubot/plugins/tooltips.html \
		--include-after-body=common/templates/manubot/plugins/jump-to-first.html \
		--include-after-body=common/templates/manubot/plugins/link-highlight.html \
		--include-after-body=common/templates/manubot/plugins/table-of-contents.html \
		--include-after-body=common/templates/manubot/plugins/lightbox.html \
		--mathjax \
		--variable math="" \
		--include-after-body=common/templates/manubot/plugins/math.html \
		--include-after-body=common/templates/manubot/plugins/hypothesis.html \
		--output=output/$*/manuscript.html output/$*/manuscript.md

test: venv
	. venv/bin/activate && pytest

testprofile: venv
	. venv/bin/activate && python3 -m cProfile -o profile /usr/local/bin/pytest
	. venv/bin/activate && gprof2dot -f pstats --node-thres=5.0 profile | dot -Tsvg -o profile.svg

testcover: venv
	. venv/bin/activate && pytest --junitxml=junit.xml --cov=msresist --cov-report xml:coverage.xml

pylint.log: venv
	. venv/bin/activate && (pylint --rcfile=./common/pylintrc msresist > pylint.log || echo "pylint exited with $?")

%.pdf: %.ipynb
	. venv/bin/activate && jupyter nbconvert --execute --ExecutePreprocessor.timeout=6000 --to pdf $< --output $@

clean:
	rm -rf *.pdf output venv pylint.log
