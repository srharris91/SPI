#include "main.hpp"

static char help[] = "SPE class to wrap PETSc Mat variables \n\n";

int main(int argc, char **args){
    PetscInt m=4,n=4;
    PetscErrorCode ierr;
    ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;



    //SPEMat B(argc,args);
    //ierr = B.Init(m,n);CHKERRQ(ierr);
    SPE::SPEMat B(m,n),C(m,n),D,E(4*m,4*n);
    C(0,1,1.);
    B(1,1,1.0);
    B();
    C();
    B=(3.4+PETSC_i*4.2)*B;
    C+=B;
    SPE::SPEMat CT;
    C.T(CT);
    D = CT*B*C;
    D *= 4.;
    D = (C*B);
    B.T();
    B.print();
    C.print();
    D.print();
    D();
    E(4,7,B);// insert
    E(0,0,C);
    E(1,5,D);
    E();
    E.print();
    B.~SPEMat();
    C.~SPEMat();
    D.~SPEMat();
    E.~SPEMat();
    CT.~SPEMat();



    ierr = PetscFinalize();CHKERRQ(ierr);

    return 0;
}
