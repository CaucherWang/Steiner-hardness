
cd ..

g++ -o ./src/get_delta0_point ./src/get_delta0_point.cpp -I ./src/ -O3 -mavx2 -fopenmp -lCbcSolver -lCbc -lOsiClp -lOsi -lCoinUtils -lClp

# g++ -pg -o ./src/get_delta0_point ./src/get_delta0_point.cpp -I ./src/ -O3 -mavx2 -fopenmp


echo "Get ME"
cd src

data=gist
for method in 13
do
    for recall in 0.94
    do
        for k in 50
        do
            for prob in 0.82 0.86 0.90 0.94 0.98
            do 
                echo "data: ${data}, method: ${method}, recall: ${recall}, k: ${k}, prob: ${prob}"
                ./get_delta0_point -d ${data} -r ${recall} -k ${k} -o 0 -m ${method} -p ${prob}
                ./get_delta0_point -d ${data} -r ${recall} -k ${k} -o 1 -m ${method} -p ${prob}
            done 
        done
    done
done
