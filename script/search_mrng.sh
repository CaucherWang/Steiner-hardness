
cd ..
# g++ ./src/search_mrng.cpp -O3 -o ./src/search_mrng -I ./src
# g++ ./src/search_mrng.cpp -O3 -o ./src/search_mrng -I ./src -march=native
g++ ./src/search_mrng.cpp -O3 -o ./src/search_mrng -I ./src -ffast-math -march=native -fopenmp
# g++ ./src/search_mrng.cpp -O3 -o ./src/search_mrng -I ./src -lprofiler
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
#     echo "mrng++"
#     index="${path}/${data}/O${data}_ef${ef}_M${M}.index"
# elif [ $randomize == "2" ]
# then 
#     echo "mrng+"
#     index="${path}/${data}/O${data}_ef${ef}_M${M}.index"
# else
#     echo "mrng"
#     index="${path}/${data}/${data}_ef${ef}_M${M}.index"    
# fi

# res="${result_path}/${data}_ef${ef}_M${M}_${randomize}.log"
# query="${path}/${data}/${data}_query.fvecs"
# gnd="${path}/${data}/${data}_groundtruth.ivecs"
# trans="${path}/${data}/O.fvecs"

# ./src/search_mrng -d ${randomize} -n ${data} -i ${index} -q ${query} -g ${gnd} -r ${res} -t ${trans} -k ${k}
cd src
./search_mrng

echo "done"


