#!/usr/bin/bash

#==== variables ====#
# generate
NUMS=10000
MAXIMUM=1000000
INFILE="data.txt"
# kmeans
CLUSTERS=3
COMMANDS="serial omp mpi hybrid"
RESULT="result"
OUTFILE="data.txt"
HOSTFILE="hosts"

#==== change default values from arguments ====#
while getopts "c:f:" argv; do
    case $argv in
        c) CLUSTERS=$OPTARG                 ;;
        f) INFILE=$OPTARG  ; OUTFILE=$OPTARG;;
    esac
done

#==== these lines should be executed before parallel-scp ====#
# generate points
#python3 generate.py -n $NUMS -m $MAXIMUM -f $INFILE
# exeucute program
#make clean && make

#==== statistic====#
> $RESULT
for COMMAND in $COMMANDS
do 
    if [ "$COMMAND" = "mpi" ] || [ "$COMMAND" = "hybrid" ]
    then
        mpirun --quiet -np 4 --hostfile $HOSTFILE --bind-to core:overload-allowed\
        ./kmeans -c $CLUSTERS -f $INFILE --no-output $COMMAND | tee -a $RESULT
    else
        ./kmeans -c $CLUSTERS -f $INFILE --no-output $COMMAND | tee -a $RESULT
    fi
done

# draw kmeans result (uncomment it if needed)
#python3 draw.py -c $CLUSTERS -f $OUTFILE

# output result
echo
echo "K-Means clustering statistics"
echo
echo "==========================="
awk '\
    BEGIN {printf "implementation\ttime(s)\n"; printf "--------------\t-----------\n"}\
    /Total/ { printf "%s\t%s\n", $5, $7 }' $RESULT\
    | column -t
echo "==========================="