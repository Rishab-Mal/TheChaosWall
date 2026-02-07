build:
	g++ -std=c++20 -o executables/main src/*.cpp

clean:
	rm executables/*
	clear

run:
	./executables/main