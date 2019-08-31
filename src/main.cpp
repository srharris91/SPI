#include "main.hpp"
#include <tuple>

static char help[] = "SPE class to wrap PETSc Mat variables \n\n";

int main(int argc, char **args){
    PetscInt m=4,n=4;
    PetscErrorCode ierr;
    ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;
    ierr = SlepcInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;

    // Vec tests
    if(1){
        SPE::printf("------------ Vec tests start-------------");
        // initialize SPEVec and Init function
        SPE::SPEVec X1(4,"X1"),X2(4,"X2"),X3("X3");

        // () operators
        X1(0,1.+1.*PETSC_i);// PetscScalar
        X1(1,1.);           // const double
        X1(2,1);            // const int
        X1.set(3,1.+1.*PETSC_i); // set function

        // assemble and print
        X1.print();

        SPE::SPEVec X5(X1,"X5_copy_X1"); // initialize with SPEVec
        X5.print(); // assemble and print

        // equals operator
        X2=X1;
        X2.print();

        // +- operators
        X2+=X1;
        X2.print();
        X3 = X2+X1;
        X3.print();
        X3.axpy(1.,X1);
        X3.print();
        X3-=X1;
        X3.print();
        SPE::SPEVec X4("X4=0");
        X4 = X2-X1;
        X4 -= X1;
        X4.print();

        // * operators
        SPE::SPEVec X6("X6=i*X1"), X7("X7"), X8("X8");
        X6 = (X1*(0.+1.*PETSC_i));// PetscScalar
        X6.print();
        X7 = X1*7.;             // double
        X8 = X1*8;              // int
        X7.print();
        X8.print();
        X8*=0.+1.*PETSC_i;
        X8.print();
        SPE::SPEVec X9("X9=X1*X2");
        X9 = X1*X2;
        X9.print();

        // / operators
        SPE::SPEVec X10("X10=X2/2.");
        SPE::SPEVec X11("X11=X3/3.");
        X10 = X2/(2.+0.*PETSC_i);
        X11 = X3/3.;
        X10.print();
        X11.print();
        X11/=2.;
        X11.name="X11=X3/6.";
        X11.print();

        SPE::printf("X5(0) = %g+%gi \nX5(1) = %g+%gi \nX5(2) = %g+%gi\nX5(3) = %g+%gi ",X5(0,PETSC_TRUE),X5(1,PETSC_TRUE),X5(2,PETSC_TRUE),X5(3,PETSC_TRUE));

        // conj and max operations
        X11.conj().print();
        SPE::SPEVec X12(4,"X12");
        for (int i=0; i<4; i++) X12(i,i);
        X12.print();
        SPE::printf("max(X12) = %g+%gi",X12.max());

        // other functions
        SPE::SPEVec X13("X13 = 13.*X1");
        X13 = 13.*X1;
        X13.print();
        SPE::SPEVec X15("X15 = 14.+X1");
        X15 = 14.+X1;
        X15.print();
        X15 = 1. - X1;
        X15.name = "X15 = 1. - X1";
        X15.print();
        SPE::SPEVec X16("X16 = X1 * conj(X1)");
        X16 = X1 * SPE::conj(X1);
        X16.print();

        (4.*SPE::ones(4)).print();
        (4.*SPE::ones(4)/4.).print();
        SPE::zeros(4).print();
        // combine all functions in one line to see
        SPE::SPEVec X14("X14 = 1");
        //X14 = (1.+1.*PETSC_i) + 4.*((X2 + 2.*X1) - 4.*X5)*3. + 13.*(X2*X7)/14. - X13 - (1.+1.*PETSC_i);
        X14 = (1.+1.*PETSC_i) + 4.*((X2 + 2.*X1) - 4.*X5)*3. + 1.*(X1+X1)/2. - X1 - (1.+1.*PETSC_i) - X1 + X1 + (1.+0.*PETSC_i);
        X14.print();

        if(
                    (PetscRealPart(X14(0,PETSC_TRUE)) == 1.)
                &&  (PetscRealPart(X14(1,PETSC_TRUE)) == 1.)
                &&  (PetscRealPart(X14(2,PETSC_TRUE)) == 1.)
                &&  (PetscRealPart(X14(3,PETSC_TRUE)) == 1.)
          ){
            //SPE::printf("Passed");
            SPE::printf("----------- SPEVec tests passed---------------");
        }
        SPE::printf("%g + %gi",X14(0,PETSC_TRUE));
        SPE::printf("%g + %gi",X14(1,PETSC_TRUE));
        SPE::printf("%g + %gi",X14(2,PETSC_TRUE));
        SPE::printf("%g + %gi",X14(3,PETSC_TRUE));

        SPE::linspace(0,4,13).print();

        SPE::sin(SPE::linspace(0.,2.*PETSC_PI,14)).print();

        SPE::printf("------------ Vec tests end  -------------");
    }

    if(0){ // Mat tests
        SPE::printf("------------ Mat tests start ---------------");

        // test constructors
        SPE::SPEMat D("D");
        SPE::SPEMat I(SPE::eye(m),"I");
        SPE::SPEMat B(m,"B");
        SPE::SPEMat A2(m,n,"A2"), E(4*m,4*n,"E");

        // set operators
        for (int i=0; i<m; i++){
            A2(i,i,1.+PETSC_i);
        }
        A2(0,n-1,0.43);
        A2.print();
        B(0,1,1);
        B();
        B.print();
        B=(3.4+PETSC_i*4.2)*A2+4.*A2;
        B.print();
        E(0,0,A2);
        E(m,n,B);
        E.print();
        A2+=B;
        A2.print();
        SPE::SPEMat A2T("A2T");
        A2.T(A2T);
        A2T.print();
        D = A2T*B;
        D.print();
        D *= 4.;
        D.print();
        D.~SPEMat();
        A2*B;
        D = A2*I;
        D.print();
        B.T();
        B.print();
        A2.print();
        D.print();
        D();
        D=I;
        D.print();
        E(4,7,B);// insert
        E(0,0,A2);
        E(1,5,D);
        E();
        E.print();
        E.H().print();
        E.conj().print();
        E.T().print();
        SPE::SPEMat F(E,"F");
        F.print();
        D.diag().print();
        //F.~SPEMat();
        //A2.~SPEMat();
        //B.~SPEMat();
        //D.~SPEMat();
        //E.~SPEMat();
        //CT.~SPEMat();
        //I.~SPEMat();
        std::cout<<"------------ Mat tests end   ---------------"<<std::endl;
    }

    // test A*x
    if(0){
        std::cout<<"------------ A*x tests start ---------------"<<std::endl;
        SPE::SPEMat A(4,4,"A");
        SPE::SPEVec x(4,"x"),b;

        for(int i=0; i<4; i++){
            A(i,i,1.+i);
            x(i,2.3+PETSC_i*1.);
        }
        A.print();
        x.print();
        b=A*x;
        b.print();

        std::cout<<"------------ A*x tests end   ---------------"<<std::endl;
    }

    // linear system solver test Ax=b solved with x=b/A
    if(0){
        std::cout<<"------------ A*x=b tests start ---------------"<<std::endl;
        SPE::SPEMat A(4,4,"A");
        SPE::SPEVec b(4,"x"),x;

        for(int i=0; i<4; i++){
            A(i,i,1.+i);
            b(i,2.);
        }
        A.print();
        b.print();
        x=b/A;
        x.print();
        std::cout<<"------------ A*x=b tests end   ---------------"<<std::endl;
    }
    // check Mat functions (eye, kron, diag)
    if(0){
        std::cout<<"------------ Mat func tests start-------------"<<std::endl;
        SPE::SPEMat I(SPE::eye(4),"I-identity");
        //I=SPE::eye(4);
        I.print();

        SPE::SPEVec two(3,"two");
        for (int i=0; i<3; i++) two(i,2.);
        two.print();

        SPE::SPEMat A("A");
        A = SPE::diag(two);

        A.print();

        SPE::SPEMat C("C");
        C=SPE::kron(I,A);
        C.print();

        SPE::SPEMat D(2,2,"D");
        D(0,0,1.); D(0,1,2.);
        D(1,0,3.); D(1,1,4.);
        D();

        SPE::kron(D,I).print();
        std::cout<<"------------ Mat func tests end  -------------"<<std::endl;
    }
    if(0){// eig test
        std::cout<<"------------ Mat eig tests start-------------"<<std::endl;
        SPE::SPEMat A(2,"A");
        A(0,0,1.);
        A(0,1,1.);
        A(1,1,1.);
        A.print();
        SPE::SPEMat B(SPE::eye(2),"I-identity");
        B.print();
        PetscScalar alpha;
        SPE::SPEVec eig(2);
        std::tie(alpha,eig) = SPE::eig(A,B,1.0+PETSC_i*0.5);
        eig.print();
        eig /= eig.max();
        eig.print();
        eig.conj().print();
        A.conj().print();
        std::cout<<"------------ Mat eig tests end  -------------"<<std::endl;
    }
    if(0){// I/O using hdf5
        std::cout<<"------------ I/O tests start  -------------"<<std::endl;
        SPE::SPEVec A(2,"A_Vec");
        A(0,1.);
        A(1,2.);
        A.print();
        SPE::save(A,"saved_data.hdf5");
        SPE::SPEVec B(2,"B_Vec");
        B(0,1.+PETSC_i*0.5);
        B(1,2.*PETSC_i);
        B.print();
        SPE::save(B,"saved_data.hdf5");
        std::cout<<"------------ I/O tests end    -------------"<<std::endl;
    }
    if(0){
        std::cout<<"------------ I/O tests2 start  -------------"<<std::endl;
        SPE::SPEVec A(2,"A_Vec");
        SPE::load(A,"saved_data.hdf5");
        A.print();
        SPE::SPEVec B(2,"B_Vec");
        SPE::load(B,"saved_data.hdf5");
        B.print();
        std::cout<<"------------ I/O tests2 end    -------------"<<std::endl;
    }
    if(0){
        SPE::SPEMat A(2,"A");
        A(0,0,2.);
        A(0,1,3.);
        A(1,1,4.);
        A.print();
        SPE::SPEMat B(SPE::eye(2),"I-identity");
        B.print();

        SPE::block({{A,B},{A,B}}).print();
    }



    ierr = PetscFinalize();CHKERRQ(ierr);
    ierr = SlepcFinalize();CHKERRQ(ierr);

    return 0;
}
