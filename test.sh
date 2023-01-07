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
HOSTFILE="hosts"

#==== change default values from arguments ====#
while getopts "c:f:" argv; do
    case $argv in
        c) CLUSTERS=$OPTARG ;;
        f) INFILE=$OPTARG   ;;
    esac
done

#==== these lines should be executed before parallel-scp ====#
# generate points
#python3 generate.py -n $NUMS -m $MAXIMUM -f $INFILE
# exeucute program
#make clean && make

#==== statistic ====#
> $RESULT
for COMMAND in $COMMANDS
do 
    if [ "$COMMAND" = "mpi" ]; then
        mpirun -np 4 -x OMP_NUM_THREADS=4 --hostfile $HOSTFILE --bind-to core\
        ./kmeans -c $CLUSTERS -f $INFILE --no-output $COMMAND | tee -a $RESULT
    elif [ "$COMMAND" = "hybrid" ]; then
        export OMP_PROC_BIND=true; export OMP_NUM_THREADS=4;
        mpirun -np 4 -pernode --hostfile $HOSTFILE --bind-to none \
        ./kmeans -c $CLUSTERS -f $INFILE --no-output $COMMAND | tee -a $RESULT
    else
        ./kmeans -c $CLUSTERS -f $INFILE --no-output $COMMAND | tee -a $RESULT
    fi
done

# draw kmeans result (uncomment it if needed)
#python3 draw.py -c $CLUSTERS -f $INFILE

#==== output result ====#
echo
echo "K-Means clustering statistics"
echo
echo "======================================"
awk -v BASETIME=$(awk 'NR==1{print $7}' $RESULT)\
    '\
    BEGIN {printf "implementation\ttime(s)\tspeedup\n";\
    printf "--------------\t-----------\t---------\n"}\
    /Total/ { printf "%s\t%ss\t%.3fX\n", $5, $7, BASETIME / $7 }\
    '\
    $RESULT | column -t
echo "======================================"
echo