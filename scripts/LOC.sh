#!/bin/zsh
cd `dirname $0`/..
exclude_files=("./data/molecules.py" "./data/ogb_mol.py" "./notebook.py" "./wandb/*")
exclusions=$(printf "-o -path %s " $exclude_files)

count=0
for file in $(find . -type f \( -name "*.sh" -o -name "*.py" -o -path "*environment_specs/*" \) -not \( -name "scratch*" ${=exclusions} \) )
; do
    loc=$(wc -l < $file | xargs)
    echo $file":" $loc
    count=$((count + loc))
done

echo $count