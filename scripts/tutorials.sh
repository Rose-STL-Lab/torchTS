#!/bin/bash

for i in $(find ../examples -name '*.ipynb'); do
    jupyter nbconvert --to markdown $i
    modified=${i::-6}
    echo $modified
    files="${modified}_files"
    echo $files
    out="$(basename $files)"
    echo $out
    sed -i "s/$out/\/img/g" "$modified.md"
    mv $files/* ../website/static/img
    mv "$modified.md" ../website/tutorials
done
