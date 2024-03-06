
cd ..

efConstruction=500
K=16
data='rand200'

g++ -o ./src/index_nsw ./src/index_nsw.cpp -I ./src/ -O3 -mavx2 -fopenmp


echo "Indexing - ${data}"

data_path=./data/${data}
index_path=./data/${data}

data_file="${data_path}/${data}_base.fvecs"
index_file="${index_path}/${data}_ef${efConstruction}_K${K}.nsw.index"
./src/index_nsw -d $data_file -i $index_file -e $efConstruction -k $K

# data_file="${data_path}/O${data}_base.fvecs"
# index_file="${index_path}/O${data}_ef${efConstruction}_M${M}.index"
# ./src/index_hnsw -d $data_file -i $index_file -e $efConstruction -m $M

# data_file="${data_path}/PCA_${data}_base.fvecs"
# index_file="${index_path}/PCA_${data}_ef${efConstruction}_M${M}.index"
# ./src/index_hnsw -d $data_file -i $index_file -e $efConstruction -m $M

# data_file="${data_path}/DWT_${data}_base.fvecs"
# index_file="${index_path}/DWT_${data}_ef${efConstruction}_M${M}.index"
# ./src/index_hnsw -d $data_file -i $index_file -e $efConstruction -m $M
