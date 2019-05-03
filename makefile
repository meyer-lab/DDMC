




test:
	pytest

testcover:
	pytest --junitxml=junit.xml --cov=msresist --cov-report xml:coverage.xml

docs:
	sphinx-apidoc -o doc/source msresist
	sphinx-build doc/source doc/build