
cd ..

# g++ ./src/search_deg.cpp -O3 -o ./src/search_deg -I ./src
# g++ ./src/search_deg.cpp -O3 -o ./src/search_deg -I ./src -march=native
# g++ ./src/search_deg.cpp -O3 -o ./src/search_deg -I ./src -ffast-math -march=native
g++ ./src/search_deg.cpp -O3 -o ./src/search_deg -I ./src -ffast-math -march=native -fopenmp
# g++ ./src/search_deg.cpp -O3 -o ./src/search_deg -I ./src -lprofiler
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
#     echo "deg++"
#     index="${path}/${data}/O${data}_ef${ef}_M${M}.index"
# elif [ $randomize == "2" ]
# then 
#     echo "deg+"
#     index="${path}/${data}/O${data}_ef${ef}_M${M}.index"
# else
#     echo "deg"
#     index="${path}/${data}/${data}_ef${ef}_M${M}.index"    
# fi

# res="${result_path}/${data}_ef${ef}_M${M}_${randomize}.log"
# query="${path}/${data}/${data}_query.fvecs"
# gnd="${path}/${data}/${data}_groundtruth.ivecs"
# trans="${path}/${data}/O.fvecs"

# ./src/search_deg -d ${randomize} -n ${data} -i ${index} -q ${query} -g ${gnd} -r ${res} -t ${trans} -k ${k}
cd src
# ./search_deg

dataset=glove-100
recall=0.98
degree=30
k=50
# ./search_deg -r ${recall} -k ${k} -d ${dataset} -g ${degree}

for recall in 0.98
do
    for k in 50
    do
        echo "recall=${recall}, k=${k}"
        # nohup \
        ./search_deg -r ${recall} -k ${k} -g ${degree} -d ${dataset}
        # 2>&1 >> ${recall}.out &
    done
done

# for recall in 0.86 0.90 0.94 0.98
# do
#     for k in 50
#     do
#         echo "recall=${recall}, k=${k}"
#         # nohup  \
#         ./search_deg -r ${recall} -k ${k} -d ${dataset} -p 1
#         # 2>&1 >> no_shuf.out &
#     done
# done


echo "done"


