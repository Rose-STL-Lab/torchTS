#!/bin/bash

for i in $(find ./examples -name '*.ipynb'); do

    # Converting notebook to markdown
    jupyter nbconvert --to markdown $i

    # Removing ipynb extension from file path
    modified=${i::-6}
    echo $modified

    # Setting variable for folder which contains images
    files="${modified}_files"
    echo $files

    # Extracts just the filename from large path variable
    out="$(basename $files)"
    echo $out

    # Fixes images path
    sed -i "s/$out/\/img/g" "$modified.md"

    #Moves images into static folder of website
    mv $files/* website/static/img

    # Moves Markdown file into tutorial folder
    mv "$modified.md" website/tutorials
done
