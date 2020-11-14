#include "main.hpp"
// include tests.hpp if running the tests function that tests all the functionality of the SPI classes
#include "tests.hpp"

static char help[] = "SPI class to wrap PETSc Mat variables \n\n";

PetscInt factorial(PetscInt n){
    PetscInt value = 1;
    for(int i=1; i<=n; ++i) value *= i;
    return value;
}

int main(int argc, char **args){
    PetscErrorCode ierr;
    std::cout<<"-----------------Petsc Slepc Init Starting---------------"<<std::endl;
    ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;
    ierr = SlepcInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;
    SPI::printf("\n\n\n");
    SPI::printf("-----------------Petsc Slepc Init Complete---------------");

    if(0){// set to 1 if wanting to test SPI Mat and Vec and other operations found in tests.cpp
        tests();
    }
    else{// started working on derivative operators, but it didn't quite work yet
        SPI::SPIVec s(5,"X1");
        s = SPI::arange(-2,3,1.);
        PetscInt d=1;
        // get_D_coeffs function with inputs s,d
        PetscInt N = s.rows;
        SPI::SPIVec spow(N,"spow");
        spow = SPI::ones(N);
        SPI::SPIMat A(N,"A");
        for(PetscInt i=0; i<N; i++){
            //std::cout<<"i = "<<i<<std::endl;
            if(i>=1){
                //spow = (s^i); // doesn't work.... :(
                spow *= s;
            }
            //SPI::draw(spow);
            //spow.print();
            for(PetscInt j=0; j<N; j++){
                //std::cout<<"i,j = "<<i<<","<<j<<std::endl;
                A(i,j,spow(j,PETSC_TRUE));
            }
        }
        A();
        A.print();
    }

    ierr = PetscFinalize();CHKERRQ(ierr);
    ierr = SlepcFinalize();CHKERRQ(ierr);

    return 0;
}
