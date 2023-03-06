CXX = g++
CXXFLAGS = -std=c++11 -O3 -march=native

RM = rm -f
MKDIRS = mkdir -p

TARGETS: MMult1 val_test01_solved val_test02_solved fast-sin

all : $(TARGETS)

%:%.cpp
	$(CXX) $(CXXFLAGS) $^ -o $@

clean:
	$(RM) $(TARGETS)

fast_sin: fast_sin.cpp
	$(CC) $(CXXFLAGS) -o fast_sin fast_sin.cpp

MMult1: MMult1.cpp
	$(CC) $(CXXFLAGS) -o MMult1 MMult1.cpp

val_test01_solved: val_test01_solved.cpp
	$(CC) $(CXXFLAGS) -o val_test01_solved val_test01_solved.cpp

val_test02_solved: val_test02_solved.cpp
	$(CC) $(CXXFLAGS) -o val_test02_solved val_test02_solved.cpp

clean:
	$(RM) $(TARGETS)