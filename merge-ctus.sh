#!/bin/bash

verbose=0


if [[ $# -ne 2 ]]; then
    echo No input given. Please provide \<fname\> \<skip-header-lines + 1\>.
    exit 1
fi

fname=$1
skiphead=$2

# Find ctu (sub-)directories
ctudirs=$(ls -d */ | sort -V) 2> /dev/null || echo ""
if [[ -z $ctudirs ]]; then
    if [[ ${verbose} -eq 1 ]]; then
        echo "Nothing to merge in $(pwd)"
    fi
    exit
fi

i=0
for ctu in ${ctudirs}; do
    #echo "ctu = $ctu" # reduce verbosity
    # Check whether ctu-like directories are available
    if [[ $ctu == "ctu_*" ]]; then
        if [[ ${verbose} -eq 1 ]]; then
            echo "No ctu directories found! Skipping merge-ctus."
        fi
        break
    # Check for non-ctu directories in ctudirs
    elif [[ ${ctu:0:3} != "ctu" ]]; then
        if [[ ${verbose} -eq 1 ]]; then
            echo "$ctu likely not a ctu directory. Skipping this directory."
        fi
        continue
    elif [[ ${ctu} == "*crash*" ]]; then
        if [[ ${verbose} -eq 1 ]]; then
            echo "$ctu is a crashed directory. Skipping this directory."
        fi
        continue
    fi

    #echo i=$i, ctu = $ctu # reduce verbosity

    path=$ctu/${fname}
    path=${path/\/\//\/} # Replace double forward slash

    # Check whether cputime exists
    if [[ ! -e $path ]]; then
        if [[ ${verbose} -eq 1 ]]; then
            echo "Skipping $path. File does not exist."
        fi
        continue
    fi

    # Create initial file
    if [[ $i -eq 0 ]]; then
        tail -n +1 $path | paste > ${fname}
    # Append all other files
    else
        tail -n +${skiphead} $path | paste >> ${fname}
    fi
    i=$(($i+1));
done

