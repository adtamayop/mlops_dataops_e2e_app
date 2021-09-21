#!/bin/bash
coverage run -m pytest --disable-warnings
coverage html -d "test/coverage_report"
