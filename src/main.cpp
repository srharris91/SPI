#include "main.hpp"
// include tests.hpp if running the tests function that tests all the functionality of the SPI classes
#include "tests.hpp"

static char help[] = "SPI class to wrap PETSc Mat variables \n\n";


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
        // get_D_Coeffs
        //SPI::SPIVec s(SPI::arange(-2,3,1.),"s");
        PetscInt d=1;
        //get_D_Coeffs(s,d).print();
        // set_D
        PetscInt order=4;
        PetscInt n=61;
        PetscScalar y_max=21.;
        SPI::SPIVec ytmp(SPI::linspace(0.,y_max,n),"y");
        PetscScalar delta=2.0001;
        SPI::SPIVec y(y_max*(1.+(SPI::tanh(delta*((ytmp/y_max) - 1.))/tanh(delta))),"y");
        y.print();
        SPI::SPIMat D(set_D(y,d,order));
        //D.print();
        (D*(y^2)).print();



    }

    ierr = PetscFinalize();CHKERRQ(ierr);
    ierr = SlepcFinalize();CHKERRQ(ierr);

    return 0;
}
