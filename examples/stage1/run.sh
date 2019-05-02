g++ -g main.cc ../../src/core/*.cc ../../src/graph/*.cc -I ../../src/ -o main -std=c++17 && ./main < data/$1.txt && rm -f main
