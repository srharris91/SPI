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
    else{
        SPI::SPIVec X1(61,"X1");
        X1 = SPI::linspace(0,24,61);
        X1.print();
    }

    ierr = PetscFinalize();CHKERRQ(ierr);
    ierr = SlepcFinalize();CHKERRQ(ierr);

    return 0;
}
