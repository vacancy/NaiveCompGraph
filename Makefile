all:
	g++ examples/stage1/main.cc src/core/*.cc src/graph/*.cc -I src/ -o main -std=c++17

clean:
	rm main

# vim:ft=make
#
