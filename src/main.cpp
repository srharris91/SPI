#include "main.hpp"
// include tests.hpp if running the tests function that tests all the functionality of the SPI classes
#include "tests.hpp"

static char help[] = "SPI class to wrap PETSc Mat variables \n\n";

PetscInt factorial(PetscInt n){ // needed for get_D_Coeffs(s,d)
    PetscInt value = 1;
    for(int i=1; i<=n; ++i) value *= i;
    return value;
}
SPI::SPIVec get_D_Coeffs(SPI::SPIVec &s,PetscInt d){
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
        //A.print();
        SPI::SPIVec b(SPI::zeros(N),"b");
        b(d,factorial(d));
        b();
        //b.print();
        SPI::SPIVec x(N,"x");
        x = (b/A);
        x();
        //x.print();

        return x;
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
        // get_D_Coeffs
        //SPI::SPIVec s(SPI::arange(-2,3,1.),"s");
        PetscInt d=1;
        //get_D_Coeffs(s,d).print();
        // set_D
        PetscInt order=4;
        PetscInt n=21;
        SPI::SPIVec xi = (SPI::linspace(0.,1.,n));
        PetscScalar h = xi(1,PETSC_TRUE)-xi(0,PETSC_TRUE);
        SPI::SPIVec ones(SPI::ones(n),"ones");
        SPI::SPIMat I(SPI::eye(n),"I");
        PetscInt N = order+d;
        if(N>n) {
            PetscErrorCode ierr=1;
            CHKERRXX(ierr);
        }
        PetscInt Nm1 = N-1;
        if (d%2 != 0) Nm1 += 1; // increase for odd derivative
        SPI::SPIVec s(SPI::arange(Nm1)-(Nm1-1)/2,"s"); // set stencil
        PetscInt smax = s(s.rows-1,PETSC_TRUE).real();

        SPI::SPIVec Coeffs(get_D_Coeffs(s,d),"Coeffs");

        SPI::SPIMat D(n,"D");
        D();
        for(PetscInt i=0; i<s.rows; i++){
            //diag_to_add.~SPIMat(); // destroy to free memory
            PetscInt k=(PetscInt)s(i,PETSC_TRUE).real();
            PetscInt nmk=n-std::abs(k);
            D += SPI::diag(Coeffs(i,PETSC_TRUE)*SPI::ones(nmk),k);

            //SPI::diag(Coeffs(0,PETSC_TRUE)*SPI::ones(10),-1).print()
        }
        D.print();
        // BCs
        D.ierr = MatSetOption(D.mat,MAT_NEW_NONZERO_LOCATIONS,PETSC_TRUE);CHKERRXX(D.ierr);
        for(PetscInt i=0; i<smax; i++){
            // for ith row
            s.~SPIVec(); // deallocate
            if(d%2!=0){// odd derivative
                s = (SPI::arange(Nm1-1)-i); // stencil for shifted diff of order-1
            }
            else{
                s = SPI::arange(Nm1)-i;// stencil for shifted diff of order-1
            }
            Coeffs.~SPIVec();
            Coeffs = get_D_Coeffs(s,d);
            D.zero_row(i);
            for(PetscInt j=0; j<s.rows; j++){
                PetscInt sj=s(j,PETSC_TRUE).real();
                SPI::printf("setting D(%d,%d)=%g",i,sj+i,Coeffs(j,PETSC_TRUE));
                D(i,sj+i,Coeffs(j,PETSC_TRUE));
            }
            D();
            // for -ith-1 row
            s.~SPIVec(); // deallocate
            if(d%2!=0){// odd derivative
                s = -1*(SPI::arange(Nm1-1)-i); // stencil for shifted diff of order-1
            }
            else{
                s = -1*SPI::arange(Nm1)-i;// stencil for shifted diff of order-1
            }
            Coeffs.~SPIVec();
            Coeffs = get_D_Coeffs(s,d);
            D.zero_row(D.rows-1-i);
            for(PetscInt j=0; j<s.rows; j++){
                PetscInt sj=s(j,PETSC_TRUE).real();
                SPI::printf("setting D(%d,%d)=%g",i,sj+i,Coeffs(j,PETSC_TRUE));
                D(D.rows-1-i,D.cols-1+sj-i,Coeffs(j,PETSC_TRUE));
            }
            D();
        }
        D.ierr = MatSetOption(D.mat,MAT_NEW_NONZERO_LOCATIONS,PETSC_FALSE);CHKERRXX(D.ierr);
        D();
        D*=(1./std::pow(h,d));
        D.print();
        (D*xi).print();

    }

    ierr = PetscFinalize();CHKERRQ(ierr);
    ierr = SlepcFinalize();CHKERRQ(ierr);

    return 0;
}
