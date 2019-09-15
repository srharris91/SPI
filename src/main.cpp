#include "main.hpp"

static char help[] = "SPE class to wrap PETSc Mat variables \n\n";

void test_if_true(PetscBool test,std::string name){
    if (test) { SPE::printf("\x1b[32m"+name+" test passed"+"\x1b[0m"); }
    else{ std::cout<<"\x1b[31m"+name+" test failed"+"\x1b[0m"<<std::endl;}
}
void test_if_close(PetscScalar value,PetscScalar golden, std::string name){
    PetscReal tol=1.E-13;
    PetscReal valuer=PetscRealPart(value);
    PetscReal goldenr=PetscRealPart(golden);
    if ((goldenr-tol<=valuer) && (valuer<=goldenr+tol)) { SPE::printf("\x1b[32m"+name+" test passed"+"\x1b[0m"); }
    else{ std::cout<<"\x1b[31m"+name+" test failed"+"\x1b[0m"<<std::endl;}
}

int main(int argc, char **args){
    PetscInt m=4,n=4;
    PetscErrorCode ierr;
    ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;
    ierr = SlepcInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;

    // Vec tests
    if(0){
        SPE::printf("------------ Vec tests start-------------");
        // initialize SPEVec and Init function
        SPE::SPEVec X1(4,"X1"),X2(4,"X2"),X3("X3");

        // () operators
        X1(0,1.+1.*PETSC_i);// PetscScalar
        X1(1,1.);           // const double
        X1(2,1);            // const int
        X1.set(3,1.+1.*PETSC_i); // set function
        X1();               // assemble
        test_if_close(X1(0,PETSC_TRUE),1.,"SPEVec() 0");
        test_if_close(X1(1,PETSC_TRUE),1.,"SPEVec() 1");
        test_if_close(X1(2,PETSC_TRUE),1.,"SPEVec() 2");
        test_if_close(X1(3,PETSC_TRUE),1.,"SPEVec() 3");

        SPE::SPEVec X5(X1,"X5_copy_X1"); // initialize with SPEVec
        test_if_close(X5(2,PETSC_TRUE),1.,"SPEVec(SPEVec)");

        // equals operator
        X2=X1;
        test_if_close(X2(2,PETSC_TRUE),1.,"SPEVec=SPEVec");

        // +- operators
        X2+=X1;
        test_if_close(X2(2,PETSC_TRUE),2.,"SPEVec+=SPEVec");
        X3 = X2+X1;
        test_if_close(X3(2,PETSC_TRUE),3.,"SPEVec+SPEVec");
        X3.axpy(1.,X1);
        test_if_close(X3(2,PETSC_TRUE),4.,"SPEVec.axpy(PetscScalar,SPEVec)");
        X3-=X1;
        test_if_close(X3(2,PETSC_TRUE),3.,"SPEVec-=SPEVec");
        SPE::SPEVec X4("X4=0");
        X4 = X2-X1;
        X4 -= X1;
        test_if_close(X4(2,PETSC_TRUE),0.,"SPEVec-SPEVec");

        // * operators
        SPE::SPEVec X6("X6=i*X1"), X7("X7"), X8("X8");
        X6 = (X1*(0.+1.*PETSC_i));// PetscScalar
        test_if_close(X6(3,PETSC_TRUE),-1.,"SPEVec*PetscScalar");
        X7 = X1*7.;             // double
        test_if_close(X7(3,PETSC_TRUE),7.,"SPEVec*double");
        X8 = X1*8;              // int
        test_if_close(X8(3,PETSC_TRUE),8.,"SPEVec*int");
        X8*=0.+1.*PETSC_i;
        test_if_close(X8(3,PETSC_TRUE),-8.,"SPEVec*=PetscScalar");
        SPE::SPEVec X9("X9=X1*X2");
        X9 = X1*X2;
        test_if_close(X9(2,PETSC_TRUE),2.,"SPEVec*SPEVec");

        // / operators
        SPE::SPEVec X10("X10=X2/2.");
        SPE::SPEVec X11("X11=X3/3.");
        X10 = X2/(2.+0.*PETSC_i);
        X11 = X3/3.;
        test_if_close(X10(2,PETSC_TRUE),1.,"SPEVec/PetscScalar");
        test_if_close(X11(2,PETSC_TRUE),1.,"SPEVec/double");
        X11/=2.;
        X11.name="X11=X3/6.";
        test_if_close(X11(2,PETSC_TRUE),0.5,"SPEVec/=double");

        //SPE::printf("X5(0) = %g+%gi \nX5(1) = %g+%gi \nX5(2) = %g+%gi\nX5(3) = %g+%gi ",X5(0,PETSC_TRUE),X5(1,PETSC_TRUE),X5(2,PETSC_TRUE),X5(3,PETSC_TRUE));

        // conj and max operations
        X11.conj();
        test_if_close(PetscImaginaryPart(X11(3,PETSC_TRUE)),-0.5,"SPEVec.conj()");

        SPE::SPEVec X12(4,"X12");
        for (int i=0; i<4; i++) X12(i,i);
        X12();
        test_if_close(X12.max(),3.,"SPEVec.max()");

        // other functions
        SPE::SPEVec X13("X13 = 13.*X1");
        X13 = 13.*X1;
        test_if_close(X13(3,PETSC_TRUE),13.,"PetscScalar*SPEVec");
        SPE::SPEVec X15("X15 = 14.+X1");
        X15 = 14.+X1;
        test_if_close(X15(3,PETSC_TRUE),15.,"PetscScalar+SPEVec");
        X15 = 1. - X1;
        test_if_close(X15(3,PETSC_TRUE),0.,"PetscScalar-SPEVec");
        X15.name = "X15 = 1. - X1";
        SPE::SPEVec X16("X16 = X1 * conj(X1)");
        X16 = X1 * SPE::conj(X1);
        test_if_close(X16(3,PETSC_TRUE),2.,"SPEVec*SPEVec");

        test_if_close((4.*SPE::ones(4))(3,PETSC_TRUE),4.,"ones(PetscInt)");
        test_if_close((4.*SPE::ones(4)/4.)(1,PETSC_TRUE),1.,"PetscScalar*ones(PetscInt)/PetscScalar");
        test_if_close(SPE::zeros(4)(1,PETSC_TRUE),0.,"zeros(PetscInt)");
        // combine all functions in one line to see
        SPE::SPEVec X14("X14 = 1");
        //X14 = (1.+1.*PETSC_i) + 4.*((X2 + 2.*X1) - 4.*X5)*3. + 13.*(X2*X7)/14. - X13 - (1.+1.*PETSC_i);
        X14 = (1.+1.*PETSC_i) + 4.*((X2 + 2.*X1) - 4.*X5)*3. + 1.*(X1+X1)/2. - X1 - (1.+1.*PETSC_i) - X1 + X1 + (1.+0.*PETSC_i);
        X14();
        test_if_close(X14(3,PETSC_TRUE),1.,"combo of functions");

        //if(
                    //(PetscRealPart(X14(0,PETSC_TRUE)) == 1.)
                //&&  (PetscRealPart(X14(1,PETSC_TRUE)) == 1.)
                //&&  (PetscRealPart(X14(2,PETSC_TRUE)) == 1.)
                //&&  (PetscRealPart(X14(3,PETSC_TRUE)) == 1.)
          //){
            ////SPE::printf("Passed");
            //SPE::printf("----------- SPEVec tests passed---------------");
        //}
        //SPE::printf("%g + %gi",X14(0,PETSC_TRUE));
        //SPE::printf("%g + %gi",X14(1,PETSC_TRUE));
        //SPE::printf("%g + %gi",X14(2,PETSC_TRUE));
        //SPE::printf("%g + %gi",X14(3,PETSC_TRUE));

        //SPE::linspace(0,4,13).print();

        SPE::SPEVec zero_to_2pi(SPE::linspace(0.,2.*PETSC_PI,101));
        SPE::SPEVec zero_to_2(SPE::linspace(0.,2.,101));
        SPE::SPEVec n1_to_1(SPE::linspace(-1.,1.,101));

        test_if_close(SPE::sin(zero_to_2pi)()(25,PETSC_TRUE),   1.,         "sin(SPEVec)");
        test_if_close(SPE::cos(zero_to_2pi)()(0, PETSC_TRUE),   1.,         "cos(SPEVec)");
        test_if_close(SPE::tan(n1_to_1)()(50,PETSC_TRUE),       0.,         "tan(SPEVec)");
        test_if_close(SPE::exp(zero_to_2pi)()(0,PETSC_TRUE),    1.,         "exp(SPEVec)");
        test_if_close(SPE::sinh(zero_to_2pi)()(0,PETSC_TRUE),   0.,         "sinh(SPEVec)");
        test_if_close(SPE::cosh(zero_to_2pi)()(0,PETSC_TRUE),   1.,         "cosh(SPEVec)");
        test_if_close(SPE::tanh(zero_to_2pi)()(0,PETSC_TRUE),   0.,         "tanh(SPEVec)");
        test_if_close(SPE::asin(n1_to_1)()(100,PETSC_TRUE),     PETSC_PI/2.,"asin(SPEVec)");
        test_if_close(SPE::acos(n1_to_1)()(50,PETSC_TRUE),      PETSC_PI/2.,"acos(SPEVec)");
        test_if_close(SPE::atan(n1_to_1)()(50,PETSC_TRUE),      0.,         "atan(SPEVec)");
        test_if_close(SPE::asinh(n1_to_1)()(50,PETSC_TRUE),     0.,         "asinh(SPEVec)");
        test_if_close(SPE::acosh(SPE::linspace(1.,10.,101))()(0,PETSC_TRUE),0.,"acosh(SPEVec)");
        test_if_close(SPE::atanh(SPE::linspace(-0.5,0.5,101))()(50,PETSC_TRUE),0.,"atanh(SPEVec)");
        SPE::SPEVec twos(2.*SPE::ones(101));
        test_if_close(SPE::pow(zero_to_2,twos)()(100,PETSC_TRUE),4.,        "pow(SPEVec,SPEVec)");
        test_if_close(SPE::pow(zero_to_2,2.)()(100,PETSC_TRUE), 4.,         "pow(SPEVec,PetscScalar)");
        test_if_close((zero_to_2^2.)(100,PETSC_TRUE),           4.,         "SPEVec^PetscScalar");
        test_if_close((zero_to_2^(2.*SPE::ones(101)))(100,PETSC_TRUE),4.,   "SPEVec^SPEVec");

        SPE::SPEVec x1(zero_to_2);
        x1.set(4.);
        test_if_close(x1(10,PETSC_TRUE),                        4.,         "SPEVec.set(PetscScalar)");
        test_if_close(SPE::abs(SPE::linspace(-1.,1.,101)())(0,PETSC_TRUE),1.,"abs(SPEVec) test passed");
        test_if_close(SPE::L2(SPE::ones(100)),                  10.,        "L2(SPEVec)");
        test_if_close(SPE::L2(SPE::ones(100)*2,SPE::ones(100)), 10.,        "L2(SPEVec,SPEVec)");
        test_if_true(SPE::linspace(0.,2.,11)==SPE::linspace(0.,2.,11),      "SPEVec==SPEVec");
        test_if_close(SPE::diff(zero_to_2)()(4,PETSC_TRUE),     0.02,       "diff(SPEVec)");

        test_if_close(SPE::sum(zero_to_2),                      101.,       "sum(SPEVec)");
        test_if_close(SPE::trapz(zero_to_2,SPE::linspace(0,4,101)),4.,"trapz(SPEVec,SPEVec)");
        test_if_close(SPE::trapz(zero_to_2),100.,"trapz(SPEVec)");
        test_if_close((SPE::linspace(0.,2.,11)+PETSC_i*SPE::linspace(4,5,11)).real()()(10,PETSC_TRUE),2.,"SPEVec.real()");
        test_if_close((SPE::linspace(0.,2.,11)+PETSC_i*SPE::linspace(4,5,11)).imag()()(10,PETSC_TRUE),5.,"SPEVec.imag()");
        test_if_close(SPE::real(SPE::linspace(0.,2.,11)+PETSC_i*SPE::linspace(4,5,11))()(10,PETSC_TRUE),2.,"real(SPEVec)");
        test_if_close(SPE::imag(SPE::linspace(0.,2.,11)+PETSC_i*SPE::linspace(4,5,11))()(10,PETSC_TRUE),5.,"imag(SPEVec)");
        test_if_close(SPE::dot(SPE::linspace(0,10,11),SPE::linspace(0,10,11)),385.,"dot(SPEVec,SPEVec)");
        test_if_close(SPE::linspace(0,10,11).dot(SPE::linspace(0,10,11)),385.,"SPEVec.dot(SPEVec)");

        test_if_close(SPE::arange(0,5,0.33)(4,PETSC_TRUE),  1.32,   "arange(int,int,PetscScalar) 1");
        test_if_close(SPE::arange(0,5,0.5)(4,PETSC_TRUE),   2.,     "arange(int,int,PetscScalar) 2");

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
        A2();
        test_if_close(PetscImaginaryPart(A2(2,2,PETSC_TRUE)),1.,"SPEMat(PetscInt,PetscInt,PETSC_TRUE)");
        B(0,1,1);
        B();
        test_if_close(B(0,1,PETSC_TRUE),1.,"SPEMat(PetscInt,PetscInt,PetscInt)");
        B=(3.4+PETSC_i*4.2)*A2+4.*A2;
        test_if_close(B(1,1,PETSC_TRUE),3.2,"PetscScalar*SPEMat+double*SPEMat");
        E(0,0,A2);
        E(m,n,B);
        E();
        test_if_close(E(1,1,PETSC_TRUE),1.,     "SPEMat(PetscInt,PetscInt,SPEMat) 1");
        test_if_close(E(4,7,PETSC_TRUE),3.182,  "SPEMat(PetscInt,PetscInt,SPEMat) 2");
        A2+=B;
        test_if_close(A2(1,1,PETSC_TRUE),4.2,  "SPEMat+=SPEMat");
        SPE::SPEMat A2T("A2T");
        A2.T(A2T);
        test_if_close(A2T(3,0,PETSC_TRUE),3.612,  "SPEMat.T(SPEMat)");
        D = A2T*B;
        test_if_close(D(0,3,PETSC_TRUE),-9.3912,  "SPEMat*SPEMat");
        D *= 4.;
        test_if_close(D(0,3,PETSC_TRUE),4.*-9.3912,  "SPEMat*=PetscScalar");
        D.~SPEMat();
        D = A2*I;
        test_if_close(A2(0,3,PETSC_TRUE),3.612,  "SPEMat*eye(SPEMat)");
        B.T();
        test_if_close(B(3,0,PETSC_TRUE),3.182,  "SPEMat.T()");
        D=I;
        test_if_close(D(1,1,PETSC_TRUE),1.,  "SPEMat=SPEMat");
        E(4,8,B);// insert
        E();
        test_if_close(E(7,8,PETSC_TRUE),3.182,  "SPEMat(PetscInt,PetscInt,SPEMat) 3");
        test_if_close(PetscImaginaryPart(E(6,10,PETSC_TRUE)),11.6,  "SPEMat(PetscInt,PetscInt,SPEMat) 4");
        E.H();
        test_if_close(PetscImaginaryPart(E(10,6,PETSC_TRUE)),-11.6,  "SPEMat.H()");
        E.conj();
        test_if_close(PetscImaginaryPart(E(10,6,PETSC_TRUE)),11.6,  "SPEMat.conj()");
        E.T();
        test_if_close(PetscImaginaryPart(E(6,10,PETSC_TRUE)),11.6,  "SPEMat.T()");
        SPE::SPEMat F(E,"F");
        test_if_close(PetscImaginaryPart(F(6,10,PETSC_TRUE)),11.6,  "SPEMat(SPEMat)");
        D.diag();
        test_if_close(D.diag()(1,PETSC_TRUE),1.,  "diag(SPEMat)");
        SPE::printf("------------ Mat tests end   ---------------");
    }

    // test A*x 
    if(0){
        SPE::printf("------------ A*x tests start ---------------");
        SPE::SPEMat A(4,4,"A");
        SPE::SPEVec x(4,"x"),b;

        for(int i=0; i<4; i++){
            A(i,i,1.+i);
            x(i,2.3+PETSC_i*1.);
        }
        A();
        x();
        b=A*x;
        test_if_close(b(2,PETSC_TRUE),6.9,"SPEMat*SPEVec");

        SPE::printf("------------ A*x tests end   ---------------");
    }

    // linear system solver test Ax=b solved with x=b/A
    if(0){
        SPE::printf("------------ A*x=b tests start ---------------");
        SPE::SPEMat A(4,4,"A");
        SPE::SPEVec b(4,"x"),x;

        for(int i=0; i<4; i++){
            A(i,i,1.+i);
            b(i,2.);
        }
        A();
        b();
        x=b/A;
        test_if_close(x(3,PETSC_TRUE),0.5,"SPEVec/SPEMat");
        std::cout<<"------------ A*x=b tests end   ---------------"<<std::endl;
    }
    // check Mat functions (eye, kron, diag)
    if(0){
        std::cout<<"------------ Mat func tests start-------------"<<std::endl;
        SPE::SPEMat I(SPE::eye(4),"I-identity");
        test_if_close(I(1,1,PETSC_TRUE),1.,"eye(PetscInt)");

        SPE::SPEVec two(3,"two");
        for (int i=0; i<3; i++) two(i,2.);
        two();

        SPE::SPEMat A("A");
        A = SPE::diag(two);
        test_if_close(A(2,2,PETSC_TRUE),2.,"diag(SPEVec)");

        SPE::SPEMat C("C");
        C=SPE::kron(I,A);
        test_if_close(C(6,6,PETSC_TRUE),2.,"kron(SPEMat,SPEMat) 1");

        SPE::SPEMat D(2,2,"D");
        D(0,0,1.); D(0,1,2.);
        D(1,0,3.); D(1,1,4.);
        D();

        test_if_close(SPE::kron(D,I)(5,1,PETSC_TRUE),3.,"kron(SPEMat,SPEMat) 2");
        SPE::printf("------------ Mat func tests end  -------------");
    }
    if(1){// eig test
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
