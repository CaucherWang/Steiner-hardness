
cd ..

g++ -o ./src/get_delta0_point ./src/get_delta0_point.cpp -I ./src/ -O3 -mavx2 -fopenmp

echo "Start runnning"
cd src || return

./get_delta0_point -d "glove-100" -k 100 -r 0.94 -p 0.86 -m 0 -o 0
./get_delta0_point -d "glove-100" -k 100 -r 0.94 -p 0.86 -m 0 -o 1

# hnsw

# ./get_delta0_point -d "gist" -k 25 -r 0.98 -p 0.98 -m 1 -o 0
# ./get_delta0_point -d "gist" -k 25 -r 0.98 -p 0.98 -m 1 -o 1

# ./get_delta0_point -d "gist" -k 25 -r 0.94 -p 0.98 -m 1 -o 0
# ./get_delta0_point -d "gist" -k 25 -r 0.94 -p 0.98 -m 1 -o 1

# ./get_delta0_point -d "rand100" -k 25 -r 0.98 -p 0.86 -m 1 -o 0
# ./get_delta0_point -d "rand100" -k 25 -r 0.98 -p 0.86 -m 1 -o 1

# ./get_delta0_point -d "rand100" -k 25 -r 0.94 -p 0.86 -m 1 -o 0
# ./get_delta0_point -d "rand100" -k 25 -r 0.94 -p 0.86 -m 1 -o 1

# ./get_delta0_point -d "glove-100" -k 25 -r 0.98 -p 0.86 -m 1 -o 0
# ./get_delta0_point -d "glove-100" -k 25 -r 0.98 -p 0.86 -m 1 -o 1

# ./get_delta0_point -d "glove-100" -k 25 -r 0.94 -p 0.86 -m 1 -o 0
# ./get_delta0_point -d "glove-100" -k 25 -r 0.94 -p 0.86 -m 1 -o 1

# ./get_delta0_point -d "glove-100" -k 100 -r 0.94 -p 0.86 -m 1 -o 0
# ./get_delta0_point -d "glove-100" -k 100 -r 0.94 -p 0.86 -m 1 -o 1

# nsg

# ./get_delta0_point -d "gist" -k 25 -r 0.98 -p 0.98 -m 2 -o 0
# ./get_delta0_point -d "gist" -k 25 -r 0.98 -p 0.98 -m 2 -o 1

# ./get_delta0_point -d "gist" -k 25 -r 0.94 -p 0.98 -m 2 -o 0
# ./get_delta0_point -d "gist" -k 25 -r 0.94 -p 0.98 -m 2 -o 1

# ./get_delta0_point -d "rand100" -k 25 -r 0.98 -p 0.86 -m 2 -o 0
# ./get_delta0_point -d "rand100" -k 25 -r 0.98 -p 0.86 -m 2 -o 1

# ./get_delta0_point -d "rand100" -k 25 -r 0.94 -p 0.86 -m 2 -o 0
# ./get_delta0_point -d "rand100" -k 25 -r 0.94 -p 0.86 -m 2 -o 1

# ./get_delta0_point -d "glove-100" -k 25 -r 0.98 -p 0.86 -m 2 -o 0
# ./get_delta0_point -d "glove-100" -k 25 -r 0.98 -p 0.86 -m 2 -o 1

# ./get_delta0_point -d "glove-100" -k 25 -r 0.94 -p 0.86 -m 2 -o 0
# ./get_delta0_point -d "glove-100" -k 25 -r 0.94 -p 0.86 -m 2 -o 1

# ./get_delta0_point -d "glove-100" -k 100 -r 0.94 -p 0.86 -m 2 -o 0
# ./get_delta0_point -d "glove-100" -k 100 -r 0.94 -p 0.86 -m 2 -o 1

# kgraph

# ./get_delta0_point -d "gist" -k 100 -r 0.98 -p 0.98 -m 0 -o 0
# ./get_delta0_point -d "gist" -k 100 -r 0.98 -p 0.98 -m 0 -o 1

# ./get_delta0_point -d "gist" -k 25 -r 0.98 -p 0.98 -m 0 -o 0
# ./get_delta0_point -d "gist" -k 25 -r 0.98 -p 0.98 -m 0 -o 1

# ./get_delta0_point -d "gist" -k 25 -r 0.94 -p 0.98 -m 0 -o 0
# ./get_delta0_point -d "gist" -k 25 -r 0.94 -p 0.98 -m 0 -o 1

# ./get_delta0_point -d "rand100" -k 25 -r 0.98 -p 0.86 -m 0 -o 0
# ./get_delta0_point -d "rand100" -k 25 -r 0.98 -p 0.86 -m 0 -o 1

# ./get_delta0_point -d "rand100" -k 25 -r 0.94 -p 0.86 -m 0 -o 0
# ./get_delta0_point -d "rand100" -k 25 -r 0.94 -p 0.86 -m 0 -o 1

# ./get_delta0_point -d "glove-100" -k 25 -r 0.98 -p 0.86 -m 0 -o 0
# ./get_delta0_point -d "glove-100" -k 25 -r 0.98 -p 0.86 -m 0 -o 1

# ./get_delta0_point -d "glove-100" -k 25 -r 0.94 -p 0.86 -m 0 -o 0
# ./get_delta0_point -d "glove-100" -k 25 -r 0.94 -p 0.86 -m 0 -o 1

# ./get_delta0_point -d "glove-100" -k 100 -r 0.94 -p 0.86 -m 0 -o 0
# ./get_delta0_point -d "glove-100" -k 100 -r 0.94 -p 0.86 -m 0 -o 1

# tau

# ./get_delta0_point -d "gist" -k 100 -r 0.98 -p 0.98 -m 3 -o 0
# ./get_delta0_point -d "gist" -k 100 -r 0.98 -p 0.98 -m 3 -o 1

# ./get_delta0_point -d "gist" -k 100 -r 0.94 -p 0.98 -m 3 -o 0
# ./get_delta0_point -d "gist" -k 100 -r 0.94 -p 0.98 -m 3 -o 1

# ./get_delta0_point -d "gist" -k 25 -r 0.98 -p 0.98 -m 3 -o 0
# ./get_delta0_point -d "gist" -k 25 -r 0.98 -p 0.98 -m 3 -o 1

# ./get_delta0_point -d "gist" -k 25 -r 0.94 -p 0.98 -m 3 -o 0
# ./get_delta0_point -d "gist" -k 25 -r 0.94 -p 0.98 -m 3 -o 1

# ./get_delta0_point -d "rand100" -k 25 -r 0.98 -p 0.86 -m 3 -o 0
# ./get_delta0_point -d "rand100" -k 25 -r 0.98 -p 0.86 -m 3 -o 1

# ./get_delta0_point -d "rand100" -k 25 -r 0.94 -p 0.86 -m 3 -o 0
# ./get_delta0_point -d "rand100" -k 25 -r 0.94 -p 0.86 -m 3 -o 1

# ./get_delta0_point -d "rand100" -k 100 -r 0.94 -p 0.86 -m 3 -o 0
# ./get_delta0_point -d "rand100" -k 100 -r 0.94 -p 0.86 -m 3 -o 1

# ./get_delta0_point -d "glove-100" -k 25 -r 0.98 -p 0.86 -m 3 -o 0
# ./get_delta0_point -d "glove-100" -k 25 -r 0.98 -p 0.86 -m 3 -o 1

# ./get_delta0_point -d "glove-100" -k 25 -r 0.94 -p 0.86 -m 3 -o 0
# ./get_delta0_point -d "glove-100" -k 25 -r 0.94 -p 0.86 -m 3 -o 1

# ./get_delta0_point -d "glove-100" -k 100 -r 0.94 -p 0.86 -m 3 -o 0
# ./get_delta0_point -d "glove-100" -k 100 -r 0.94 -p 0.86 -m 3 -o 1