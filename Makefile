export PETSC_DIR=$(PWD)/petsc
export SLEPC_DIR=$(PWD)/slepc
# export PETSC_ARCH=arch-linux2-c-opt
export PETSC_ARCH=arch-linux2-cxx-opt

SRCFILES = $(wildcard ./src/*cpp)
INCLUDE_DIR = ./include
OBJFILES= $(SRCFILES:.cpp=.o)
CPPFLAGS = ${PETSC_CC_INCLUDES} ${SLEPC_CC_INCLUDES} -std=c++11 -Wall -I$(INCLUDE_DIR) -c
CFLAGS = ${PETSC_CC_INCLUDES} ${SLEPC_CC_INCLUDES}
FFLAGS = ${PETSC_FC_INCLUDES} ${SLEPC_FC_INCLUDES}

EXECUTABLE = a

all: $(SRCFILES) $(EXECUTABLE)

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${SLEPC_DIR}/lib/slepc/conf/slepc_variables

CPP = mpicxx

$(EXECUTABLE): $(OBJFILES)
	$(CLINKER) $(OBJFILES) ${PETSC_LIB} -o $(EXECUTABLE) 

docs:
	doxygen

clean:
	rm -f src/*o $(EXECUTABLE)
	rm -rf doc/html

.cpp.o:
	$(CPP) $(CPPFLAGS) $< -o $@
