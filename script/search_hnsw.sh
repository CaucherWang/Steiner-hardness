
cd ..

# g++ ./src/search_hnsw.cpp -O3 -o ./src/search_hnsw -I ./src
# g++ ./src/search_hnsw.cpp -O3 -o ./src/search_hnsw -I ./src -march=native
# g++ ./src/search_hnsw.cpp -O3 -o ./src/search_hnsw -I ./src -ffast-math -march=native
g++ ./src/search_hnsw.cpp -O3 -o ./src/search_hnsw -I ./src -ffast-math -march=native -fopenmp
# g++ ./src/search_hnsw.cpp -O3 -o ./src/search_hnsw -I ./src -lprofiler
# path=./data/
# result_path=./results/

# data='gist'
# ef=500
# M=16
# k=20

# for randomize in {0..2}
# do
# if [ $randomize == "1" ]
# then 
#     echo "HNSW++"
#     index="${path}/${data}/O${data}_ef${ef}_M${M}.index"
# elif [ $randomize == "2" ]
# then 
#     echo "HNSW+"
#     index="${path}/${data}/O${data}_ef${ef}_M${M}.index"
# else
#     echo "HNSW"
#     index="${path}/${data}/${data}_ef${ef}_M${M}.index"    
# fi

# res="${result_path}/${data}_ef${ef}_M${M}_${randomize}.log"
# query="${path}/${data}/${data}_query.fvecs"
# gnd="${path}/${data}/${data}_groundtruth.ivecs"
# trans="${path}/${data}/O.fvecs"

# ./src/search_hnsw -d ${randomize} -n ${data} -i ${index} -q ${query} -g ${gnd} -r ${res} -t ${trans} -k ${k}
cd src

# for recall in 0.94
# do
#     for k in 100 # 20 50 100 200 500
#     do
        # dataset="gist"
        # ef=1000
        # M=32
        # ./search_hnsw -e ${ef} -m ${M} -r ${recall} -k ${k} -d ${dataset} >> ${dataset}.out &
        # dataset="rand100"
        # ef=2000
        # M=100
        # ./search_hnsw -e ${ef} -m ${M} -r ${recall} -k ${k} -d ${dataset} >> ${dataset}.out &
#         dataset="glove-100"
#         ef=1000
#         M=60
#         ./search_hnsw -e ${ef} -m ${M} -r ${recall} -k ${k} -d ${dataset} >> ${dataset}.out &
#     done
# done


k=50
# recall=0.94

dataset="deep"
ef=500
M=16

# dataset="glove-100"
# ef=1000
# M=60

# dataset="gist"
# ef=1000
# M=32

# dataset="rand100"
# ef=2000
# M=100

for recall in 0.42
do
    ./search_hnsw -e ${ef} -m ${M} -r ${recall} -k ${k} -d ${dataset} >> ${dataset}.out &
done

echo "done"


