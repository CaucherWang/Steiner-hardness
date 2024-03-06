
cd ..

data='glove-100'
kod='_2048'
m=0

g++ -o ./src/index_mrng ./src/index_mrng.cpp -I ./src/ -O3 -mavx2 -ffast-math -march=native -fopenmp


echo "MRNG Indexing - ${data} - ${kod}"


./src/index_mrng -k ${kod} -d $data -m $m

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
