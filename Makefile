export PETSC_DIR=$(PWD)/petsc
export SLEPC_DIR=$(PWD)/slepc
# export PETSC_ARCH=arch-linux2-c-opt
export PETSC_ARCH=arch-linux2-cxx-opt

SRCFILES = $(wildcard ./src/*cpp)
INCLUDE_DIR = ./include
OBJFILES= $(SRCFILES:.cpp=.o)
#CPPFLAGS = ${PETSC_CC_INCLUDES} ${SLEPC_CC_INCLUDES} -std=c++11 -Wall -I$(INCLUDE_DIR) -c
#CFLAGS = ${PETSC_CC_INCLUDES} ${SLEPC_CC_INCLUDES}
#FFLAGS = ${PETSC_FC_INCLUDES} ${SLEPC_FC_INCLUDES}
CPPFLAGS = ${SLEPC_CC_INCLUDES} -std=c++11 -Wall -I$(INCLUDE_DIR) -c
CFLAGS = ${SLEPC_CC_INCLUDES}
FFLAGS = ${SLEPC_FC_INCLUDES}

EXECUTABLE = PETSC_LIB.exec

all: $(SRCFILES) $(EXECUTABLE)

# include ${PETSC_DIR}/lib/petsc/conf/variables
include ${SLEPC_DIR}/lib/slepc/conf/slepc_variables

CPP = mpicxx

$(EXECUTABLE): $(OBJFILES)
	$(CLINKER) $(OBJFILES) ${SLEPC_LIB} -o $(EXECUTABLE) 
	@#$(CLINKER) $(OBJFILES) ${PETSC_LIB} -o $(EXECUTABLE) 

doc:
	doxygen

clean:
	rm -f src/*o $(EXECUTABLE) *dat *dat.info *hdf5 
	rm -rf docs/*

.cpp.o:
	$(CPP) $(CPPFLAGS) $< -o $@
