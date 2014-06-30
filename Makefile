# Path where Blitz++ is installed
BZDIR = /usr/include/blitz

CXX = g++
DEBUG = 1
ARCH := $(shell getconf LONG_BIT)

# Flags for optimized executables

CXXFLAGS = -ftemplate-depth-100 -I$(BZDIR) -std=c++11 -D_GLIBCXX_USE_NANOSLEEP `pkg-config gtkmm-3.0 blitz --cflags --libs`

DEBUGING =  -ggdb -DBZ_DEBUG -Wall -Wextra -pedantic -O0
OPTIMIZE = -O2

LIBS = -L$(BZDIR)/lib -lblitz -lm -lncurses


TARGETS = simulation

.SUFFIXES: .o .cpp

.PHONY: all clean

.cpp.o:	
ifeq ($(DEBUG),1)
	$(CXX) $(CXXFLAGS) $(DEBUGING) -c $*.cpp -o $*.o_$(ARCH)
else
	$(CXX) $(CXXFLAGS) $(OPTIMIZE) -c $*.cpp -o $*.o_$(ARCH)
endif

$(TARGETS): 
	$(CXX)  $(addsuffix _$(ARCH),$^) -o $@_$(ARCH) $(LIBS)

all: 
	$(TARGETS)


simulation:  simulation.o predator.o bptt.o vgl.o rprop.o  NN/neural_network.o gui.o config.o environment.o distance.o age.o reward.o time.o energy.o gaussian_enc.o inv_gaussian_enc.o

clean: 
	-rm -f *.o $(TARGETS)
