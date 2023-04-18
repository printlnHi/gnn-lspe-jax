#!/bin/zsh

exclude_files=("./data/molecules.py" "./data/ogb_mol.py" "./notebook.py")
exclusions=$(printf "-o -path %s " $exclude_files)
#TODO: Count jupyter notebook

count=0
for file in $(find . -type f \( -name "*.sh" -o -name "*.py" \) -not \( -name "scratch*" ${=exclusions} \) )
; do
    loc=$(wc -l < $file | xargs)
    echo $file":" $loc
    count=$((count + loc))
done

echo $count