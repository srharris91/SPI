#include "main.hpp"
// include tests.hpp if running the tests function that tests all the functionality of the SPI classes
#include "tests.hpp"

static char help[] = "SPI class to wrap PETSc Mat variables \n\n";

int main(int argc, char **args){
    PetscErrorCode ierr;
    ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;
    ierr = SlepcInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;

    if(1){// set to 1 if wanting to test SPI Mat and Vec and other operations found in tests.cpp
        tests();
    }
    else{// started working on derivative operators, but it didn't quite work yet
        SPI::SPIVec s(5,"X1");
        s = SPI::arange(-2,3,1.);
        PetscInt d=1;
        // get_D_coeffs function with inputs s,d
        PetscInt N = s.rows;
        SPI::SPIVec spow(N);
        spow = s;
        SPI::SPIMat A(N,"A");
        for(PetscInt i=0; i<N; i++){
            std::cout<<"i = "<<i<<std::endl;
            spow = s^((PetscScalar)i);
            spow.print();
            for(PetscInt j=0; j<N; j++){
                std::cout<<"i,j = "<<i<<","<<j<<std::endl;
                A(i,j,spow(j));
            }
        }
        A();
        A.print();
    }

    ierr = PetscFinalize();CHKERRQ(ierr);
    ierr = SlepcFinalize();CHKERRQ(ierr);

    return 0;
}
