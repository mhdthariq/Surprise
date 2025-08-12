#!/bin/sh

set -ex

# Show tool versions for debugging
black --version
usort --version
flake8 --version

echo "Running usort..."
usort format surprise
usort format tests
usort format examples
usort format setup.py

echo "Running black..."
black surprise
black tests
black examples
black setup.py

echo "Running flake8..."
# Use configuration from pyproject.toml via command line flags for compatibility
flake8 --max-line-length 88 --ignore E203,E231,E241,E402,W503,W504,F821,E501 surprise
flake8 --max-line-length 88 --ignore E203,E231,E241,E402,W503,W504,F821,E501 tests
flake8 --max-line-length 88 --ignore E203,E231,E241,E402,W503,W504,F821,E501 examples
flake8 --max-line-length 88 --ignore E203,E231,E241,E402,W503,W504,F821,E501 setup.py

echo "Linting completed successfully!"
