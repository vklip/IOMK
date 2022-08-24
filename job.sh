#!/bin/bash
python fit.py it0_mem 6 1000
#setting number of iterations to 5
for ((i=0; i<=5; i++))
do
mkdir it${i}
pushd it${i}
cp ../CG.in .
cp ../data.in .
cp ../nb.table .
cp ../it${i}_mem_fit/Amatrix .

#here you might want to change the executable or choose parallelization options
lmp -i CG.in
python ../vacf.py dump.h5 ./ 

popd
python update.py it${i}/vacf a_mem t_mem  it${i}_mem_fit/fit it$((i+1))_mem
python fit.py it$((i+1))_mem 6 1000
done
