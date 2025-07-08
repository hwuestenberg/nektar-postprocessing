#!/bin/bash

verbose=0

# Check for logfile
logfile=$(ls *.l*)
if [[ -z $logfile ]]; then
    logfile=$(ls log*)
    if [[ -z $logfile ]]; then
        #echo "Could not find any log file with either \"*.l*\" or \"log*\". 
        #Please check directory $(pwd) for logfile. Exiting logcrawler.sh."
        exit
    fi
fi

# Extract timing and number of dofs
#OUT=timings.dat
#echo "Crawling through ${logfile}..."
#echo "ncpu time localdof globaldof" > $OUT
#ncpu=$(cat $logfile | grep "Num. Processes" | tr -s ' ' | cut -d' ' -f 4)
#timeintegration=$(cat $logfile | grep "Time-integration" | tr -s ' ' | cut -d ' ' -f 3 | cut -d 's' -f 1) 
#localdof=$(cat $logfile | grep "Assembly map statistics for field u:" -A1 | awk 'NR==2' | tr -s ' ' | cut -d' ' -f 8)
#globaldof=$(cat $logfile | grep "Assembly map statistics for field u:" -A1 | awk 'NR==2' | tr -s ' ' | cut -d' ' -f 9)
#echo "$ncpu $timeintegration $localdof $globaldof" >> $OUT


# Check for PETSc Full systems
npetsc=$(grep "PETSc" $logfile | wc -l)
npetsc=$(python3 -c "print(int( $npetsc/2 ))")
if [[ ${verbose} -eq 1 ]]; then
    echo Corrected npetsc = $npetsc
fi

## Print header, sequentially add more information
header="step cputime phystime"
echo $header > cputime.dat
steps=$(cat $logfile | grep "  CPU Time" | tr -s ' ' | cut -d ' ' -f 2)
phystimes=$(cat $logfile | grep "  CPU Time" | tr -s ' ' | cut -d ' ' -f 4)
cputimes=$(cat $logfile | grep "  CPU Time" | tr -s ' ' | cut -d ' ' -f 7 | cut -d 's' -f 1)
paste <(echo "$steps") <(echo "$cputimes") <(echo "$phystimes") -d ' ' >> cputime.dat


# Check for CFL
if [[ -n $(cat $logfile | grep "CFL:" | tr -s ' ' | cut -d ' ' -f 2) ]]; then
    if [[ ${verbose} -eq 1 ]]; then
        echo "Found CFL information"
    fi
    header="cfl"
    echo $header > cfl.dat
    cfls=$(cat $logfile | grep "CFL:" | tr -s ' ' | cut -d ' ' -f 2)
    paste <(echo "$cfls") -d ' ' >> cfl.dat
fi


### Check for iteration count
# Get the number of variables for iterations counts to check 2D or 3D
nvars=$(grep "Variable" $logfile | wc -l)
echo nvars = $nvars
if [[ nvars -gt 4 ]]; then # Assume mpi is used if more than 3 vars detected
    nvars=$(python3 -c "print(int( $nvars/2 ))")
fi
if [[ ${verbose} -eq 1 ]]; then
    echo Corrected nvars = $nvars
fi

# Extract for each step: CPU time, iterations, CFL, ...
if [[ $nvars -eq 3 ]]; then
    header="niteru niterv" > iterations.dat
else
    header="niterp niteru niterv niterw" > iterations.dat
fi
echo $header > iterations.dat

# Print data at each iteration
# Add logic for 3D and 2.5D variables to extract iteration counts
if [[ $nvars -eq 3 ]]; then
    niteru=$(grep "iterations made" $logfile | awk -v nvar="2" 'NR % nvar - 1 == 0' | tr -s ' ' | awk '{$1=$1};1' | cut -d ' ' -f 5)
    niterv=$(grep "iterations made" $logfile | awk -v nvar="2" 'NR % nvar - 0 == 0' | tr -s ' ' | awk '{$1=$1};1' | cut -d ' ' -f 5)
    paste <(echo "$niteru") <(echo "$niterv") -d ' ' >> iterations.dat
else
    # If petsc solves are detected, assume all velocity solves use PETSc output
    if [[ $npetsc -eq 3 ]]; then
        niterp=$(grep "iterations" $logfile | awk -v nvar="$nvars" 'NR % nvar - 1 == 0' | tr -s ' ' | awk '{$1=$1};1' | cut -d ' ' -f 5)
        niteru=$(grep "iterations" $logfile | awk -v nvar="$nvars" 'NR % nvar - 2 == 0' | tr -s ' ' | awk '{$1=$1};1' | cut -d ' ' -f 8)
        niterv=$(grep "iterations" $logfile | awk -v nvar="$nvars" 'NR % nvar - 3 == 0' | tr -s ' ' | awk '{$1=$1};1' | cut -d ' ' -f 8)
        niterw=$(grep "iterations" $logfile | awk -v nvar="$nvars" 'NR % nvar - 0 == 0' | tr -s ' ' | awk '{$1=$1};1' | cut -d ' ' -f 8)
        paste <(echo "$niterp") <(echo "$niteru") <(echo "$niterv") <(echo "$niterw") -d ' ' >> iterations.dat
    else
        niterp=$(grep "iterations" $logfile | awk -v nvar="$nvars" 'NR % nvar - 1 == 0' | tr -s ' ' | awk '{$1=$1};1' | cut -d ' ' -f 5)
        niteru=$(grep "iterations" $logfile | awk -v nvar="$nvars" 'NR % nvar - 2 == 0' | tr -s ' ' | awk '{$1=$1};1' | cut -d ' ' -f 5)
        niterv=$(grep "iterations" $logfile | awk -v nvar="$nvars" 'NR % nvar - 3 == 0' | tr -s ' ' | awk '{$1=$1};1' | cut -d ' ' -f 5)
        niterw=$(grep "iterations" $logfile | awk -v nvar="$nvars" 'NR % nvar - 0 == 0' | tr -s ' ' | awk '{$1=$1};1' | cut -d ' ' -f 5)
        paste <(echo "$niterp") <(echo "$niteru") <(echo "$niterv") <(echo "$niterw") -d ' ' >> iterations.dat
    fi
fi

if [[ ${verbose} -eq 1 ]]; then
    echo "Crawler finished with ${logfile}."
fi
