#Run goostats on each of input files

for datafile in "$@"
do
    echo $datafile stats-$datafile
    bash goostats $datafile stats-$datafile
done
