g++ main.cc ../../src/core/*.cc ../../src/graph/*.cc ../../src/graph/ops/*.cc -I ../../src/ -o main -std=c++17 && ./main < data/$1.txt && rm -f main
