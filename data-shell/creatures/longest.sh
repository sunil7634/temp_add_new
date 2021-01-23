for filename in *.dat
do
    wc -l $filename | sort -n | tail -n 2 | head -n 1
done