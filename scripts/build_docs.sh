#!/bin/bash
# run this script from the project root using `./scripts/build_docs.sh`

echo "----------------------------------------"
echo "Generating API documentation with Sphinx"
echo "----------------------------------------"

poetry run make -C docs html

echo "-----------------------------------------"
echo "Moving Sphinx documentation to Docusaurus"
echo "-----------------------------------------"

SPHINX_HTML_DIR="website/static/api/"
cp -R "./docs/build/html/" "./${SPHINX_HTML_DIR}"
echo "Sucessfully moved Sphinx docs to ${SPHINX_HTML_DIR}"
