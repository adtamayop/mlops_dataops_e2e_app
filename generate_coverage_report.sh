#!/bin/bash
coverage run -m pytest
coverage html -d "test/coverage_report"
