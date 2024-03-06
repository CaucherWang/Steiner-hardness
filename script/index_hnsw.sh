
cd ..

efConstruction=1000
M=60
data='glove-100'

g++ -o ./src/index_hnsw ./src/index_hnsw.cpp -I ./src/ -O3 -mavx2 -fopenmp


echo "Indexing - ${data}"

data_path=./data/${data}
index_path=./data/${data}
postfix=

# data_file="${data_path}/${data}_base.fvecs${postfix}"
# index_file="${index_path}/${data}_ef${efConstruction}_M${M}.index_plain${postfix}"
# nohup \
# ./src/index_hnsw -t float -d $data_file -i $index_file -e $efConstruction -m $M
# 2>&1 >> nohup.out &

for i in {3..11}
do
    postfix="_shuf${i}"
    data_file="${data_path}/${data}_base.fvecs${postfix}"
    index_file="${index_path}/${data}_ef${efConstruction}_M${M}.index_plain${postfix}"
    nohup ./src/index_hnsw -t float -d $data_file -i $index_file -e $efConstruction -m $M 2>&1 >> {i}.out &
done

echo "done"

# data_file="${data_path}/O${data}_base.fvecs"
# index_file="${index_path}/O${data}_ef${efConstruction}_M${M}.index"
# ./src/index_hnsw -d $data_file -i $index_file -e $efConstruction -m $M

# data_file="${data_path}/PCA_${data}_base.fvecs"
# index_file="${index_path}/PCA_${data}_ef${efConstruction}_M${M}.index"
# ./src/index_hnsw -d $data_file -i $index_file -e $efConstruction -m $M

# data_file="${data_path}/DWT_${data}_base.fvecs"
# index_file="${index_path}/DWT_${data}_ef${efConstruction}_M${M}.index"
# ./src/index_hnsw -d $data_file -i $index_file -e $efConstruction -m $M
