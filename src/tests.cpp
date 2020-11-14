#include "main.hpp"
#include "tests.hpp"

void test_if_true(PetscBool test,std::string name){
    if (test) { SPI::printf("\x1b[32m"+name+" test passed"+"\x1b[0m"); }
    else{ std::cout<<"\x1b[31m"+name+" test failed"+"\x1b[0m"<<std::endl;}
}

void test_if_close(PetscScalar value,PetscScalar golden, std::string name, PetscReal tol){
    PetscReal valuer=PetscRealPart(value);
    PetscReal goldenr=PetscRealPart(golden);
    if ((goldenr-tol<=valuer) && (valuer<=goldenr+tol)) { SPI::printf("\x1b[32m"+name+" test passed"+"\x1b[0m"); }
    else{ std::cout<<"\x1b[31m"+name+" test failed"+"\x1b[0m"<<std::endl;
            std::cout<<"      valuer="<<valuer<<std::endl;
            std::cout<<"      goldenr="<<goldenr<<std::endl;}
}

int tests(){
    PetscInt m=4, n=4;
    // Vec tests
    if(1){
        SPI::printf("------------ Vec tests start-------------");
        // initialize SPIVec and Init function
        SPI::SPIVec X1(4,"X1"),X2(4,"X2"),X3("X3");

        // () operators
        X1(0,1.+1.*PETSC_i);// PetscScalar
        X1(1,1.);           // const double
        X1(2,1);            // const int
        X1.set(3,1.+1.*PETSC_i); // set function
        X1();               // assemble
        test_if_close(X1(0,PETSC_TRUE),1.,"SPIVec() 0");
        test_if_close(X1(1,PETSC_TRUE),1.,"SPIVec() 1");
        test_if_close(X1(2,PETSC_TRUE),1.,"SPIVec() 2");
        test_if_close(X1(3,PETSC_TRUE),1.,"SPIVec() 3");

        SPI::SPIVec X5(X1,"X5_copy_X1"); // initialize with SPIVec
        test_if_close(X5(2,PETSC_TRUE),1.,"SPIVec(SPIVec)");

        // equals operator
        X2=X1;
        test_if_close(X2(2,PETSC_TRUE),1.,"SPIVec=SPIVec");

        // +- operators
        X2+=X1;
        test_if_close(X2(2,PETSC_TRUE),2.,"SPIVec+=SPIVec");
        X3 = X2+X1;
        test_if_close(X3(2,PETSC_TRUE),3.,"SPIVec+SPIVec");
        X3.axpy(1.,X1);
        test_if_close(X3(2,PETSC_TRUE),4.,"SPIVec.axpy(PetscScalar,SPIVec)");
        X3-=X1;
        test_if_close(X3(2,PETSC_TRUE),3.,"SPIVec-=SPIVec");
        SPI::SPIVec X4("X4=0");
        X4 = X2-X1;
        X4 -= X1;
        test_if_close(X4(2,PETSC_TRUE),0.,"SPIVec-SPIVec");

        // * operators
        SPI::SPIVec X6("X6=i*X1"), X7("X7"), X8("X8");
        X6 = (X1*(0.+1.*PETSC_i));// PetscScalar
        test_if_close(X6(3,PETSC_TRUE),-1.,"SPIVec*PetscScalar");
        X7 = X1*7.;             // double
        test_if_close(X7(3,PETSC_TRUE),7.,"SPIVec*double");
        X8 = X1*8;              // int
        test_if_close(X8(3,PETSC_TRUE),8.,"SPIVec*int");
        X8*=0.+1.*PETSC_i;
        test_if_close(X8(3,PETSC_TRUE),-8.,"SPIVec*=PetscScalar");
        SPI::SPIVec X9("X9=X1*X2");
        X9 = X1*X2;
        test_if_close(X9(2,PETSC_TRUE),2.,"SPIVec*SPIVec");
    }
    if(1){
        // Mat tests
        SPI::printf("------------ Mat tests start ---------------");

        // test constructors
        SPI::SPIMat D("D");
        SPI::SPIMat I(SPI::eye(m),"I");
        SPI::SPIMat B(m,"B");
        SPI::SPIMat A2(m,n,"A2"), E(4*m,4*n,"E");

        // set operators
        for (int i=0; i<m; i++){
            A2(i,i,1.+PETSC_i);
        }
        A2(0,n-1,0.43);
        A2();
        test_if_close(PetscImaginaryPart(A2(2,2,PETSC_TRUE)),1.,"SPIMat(PetscInt,PetscInt,PETSC_TRUE)");
        B(0,1,1);
        B();
        test_if_close(B(0,1,PETSC_TRUE),1.,"SPIMat(PetscInt,PetscInt,PetscInt)");
        B=(3.4+PETSC_i*4.2)*A2+4.*A2;
        test_if_close(B(1,1,PETSC_TRUE),3.2,"PetscScalar*SPIMat+double*SPIMat");
        E(0,0,A2);
        E(m,n,B);
        E();
        test_if_close(E(1,1,PETSC_TRUE),1.,     "SPIMat(PetscInt,PetscInt,SPIMat) 1");
        test_if_close(E(4,7,PETSC_TRUE),3.182,  "SPIMat(PetscInt,PetscInt,SPIMat) 2");
        A2+=B;
        test_if_close(A2(1,1,PETSC_TRUE),4.2,  "SPIMat+=SPIMat");
        SPI::SPIMat A2T("A2T");
        A2.T(A2T);
        test_if_close(A2T(3,0,PETSC_TRUE),3.612,  "SPIMat.T(SPIMat)");
        D = A2T*B;
        test_if_close(D(0,3,PETSC_TRUE),-9.3912,  "SPIMat*SPIMat");
        D *= 4.;
        test_if_close(D(0,3,PETSC_TRUE),4.*-9.3912,  "SPIMat*=PetscScalar");
        D.~SPIMat();
        D = A2*I;
        test_if_close(A2(0,3,PETSC_TRUE),3.612,  "SPIMat*eye(SPIMat)");
        B.T();
        test_if_close(B(3,0,PETSC_TRUE),3.182,  "SPIMat.T()");
        D=I;
        test_if_close(D(1,1,PETSC_TRUE),1.,  "SPIMat=SPIMat");
        E(4,8,B);// insert
        E();
        test_if_close(E(7,8,PETSC_TRUE),3.182,  "SPIMat(PetscInt,PetscInt,SPIMat) 3");
        test_if_close(PetscImaginaryPart(E(6,10,PETSC_TRUE)),11.6,  "SPIMat(PetscInt,PetscInt,SPIMat) 4");
        E.H();
        test_if_close(PetscImaginaryPart(E(10,6,PETSC_TRUE)),-11.6,  "SPIMat.H()");
        E.conj();
        test_if_close(PetscImaginaryPart(E(10,6,PETSC_TRUE)),11.6,  "SPIMat.conj()");
        E.T();
        test_if_close(PetscImaginaryPart(E(6,10,PETSC_TRUE)),11.6,  "SPIMat.T()");
        SPI::SPIMat F(E,"F");
        test_if_close(PetscImaginaryPart(F(6,10,PETSC_TRUE)),11.6,  "SPIMat(SPIMat)");
        D.diag();
        test_if_close(D.diag()(1,PETSC_TRUE),1.,  "diag(SPIMat)");
        SPI::printf("------------ Mat tests end   ---------------");
    }
    if(1){
        SPI::printf("------------ A*x tests start ---------------");
        SPI::SPIMat A(4,4,"A");
        SPI::SPIVec x(4,"x"),b;

        for(int i=0; i<4; i++){
            A(i,i,1.+i);
            x(i,2.3+PETSC_i*1.);
        }
        A();
        x();
        b=A*x;
        test_if_close(b(2,PETSC_TRUE),6.9,"SPIMat*SPIVec");

        SPI::printf("------------ A*x tests end   ---------------");
    }

    if(1){
        SPI::printf("------------ A*x=b tests start ---------------");
        SPI::SPIMat A(4,4,"A");
        SPI::SPIVec b(4,"x"),x;

        for(int i=0; i<4; i++){
            A(i,i,1.+i);
            b(i,2.);
        }
        A();
        b();
        x=b/A;
        test_if_close(x(3,PETSC_TRUE),0.5,"SPIVec/SPIMat");
        SPI::printf("------------ A*x=b tests end   ---------------");
    }
    if(1){
        SPI::printf("------------ Mat func tests start-------------");
        SPI::SPIMat I(SPI::eye(4),"I-identity");
        test_if_close(I(1,1,PETSC_TRUE),1.,"eye(PetscInt)");

        SPI::SPIVec two(3,"two");
        for (int i=0; i<3; i++) two(i,2.);
        two();

        SPI::SPIMat A("A");
        A = SPI::diag(two);
        test_if_close(A(2,2,PETSC_TRUE),2.,"diag(SPIVec)");

        SPI::SPIMat C("C");
        C=SPI::kron(I,A);
        test_if_close(C(6,6,PETSC_TRUE),2.,"kron(SPIMat,SPIMat) 1");

        SPI::SPIMat D(2,2,"D");
        D(0,0,1.); D(0,1,2.);
        D(1,0,3.); D(1,1,4.);
        D();

        test_if_close(SPI::kron(D,I)(5,1,PETSC_TRUE),3.,"kron(SPIMat,SPIMat) 2");
        SPI::printf("------------ Mat func tests end  -------------");
    }
    if(1){
        SPI::printf("------------ Mat eig tests start-------------");
        SPI::SPIMat A(2,"A");
        A(0,0,1.);
        A(0,1,1.);
        A(1,1,1.);
        A();
        SPI::SPIMat B(SPI::eye(2),"I-identity");
        PetscScalar alpha;
        SPI::SPIVec eig(2,"eig1");
        std::tie(alpha,eig) = SPI::eig(A,B,1.0+PETSC_i*0.5);
        eig /= eig.max(); // normalize by max amplitude
        test_if_close(alpha,1.,"eig(SPIMat,SPIMat,PetscScalar) 1",1.E-8);
        PetscScalar alpha2;
        SPI::SPIVec eig2(2,"eig2");
        std::tie(alpha2,eig2) = SPI::eig(A,B,-1.0+PETSC_i*0.00005,1.E-19,10);
        eig2 /= eig2.max(); // normalize by max amplitude
        test_if_close(alpha2,1.,"eig(SPIMat,SPIMat,PetscScalar) 2",1.E-8);

        std::tie(alpha,eig) = SPI::polyeig({A,-B},1.0+0.5*PETSC_i);
        test_if_close(alpha,1.,"polyeig(std::vector<SPIMat>,PetscScalar) 1",1.E-8);
        std::tie(alpha,eig) = SPI::polyeig({A},1.0+0.5*PETSC_i);
        test_if_close(alpha,1.,"polyeig(std::vector<SPIMat>,PetscScalar) 2",1.E-8);

        SPI::SPIMat M(4,"M"),C(4,"C"),K(4,"K"); // (M*lambda^2 + C*lambda + K)*x = 0 from MatLab mathworks documentation
        // M
        M(0,0,3.);
                    M(1,1,1.);
                                M(2,2,3.);
                                            M(3,3,1.);
        // C
        C(0,0,0.4);             C(0,2,-0.3);

        C(2,0,-0.3);            C(2,2,0.5); C(2,3,-0.2);
                                C(3,2,-0.2);C(3,3,0.2);
        // K
        K(0,0,-7.); K(0,1,2.);  K(0,2,4);
        K(1,0,2);   K(1,1,-4);  K(1,2,2);
        K(2,0,4);   K(2,1,2);   K(2,2,-9);  K(2,3,3);
                                K(3,2,3);   K(3,3,-3);
        // assemble
        M();C();K();


        PetscScalar alpha4;
        SPI::SPIVec eig4(4);
        std::tie(alpha4,eig4) = SPI::polyeig({K,C,M},-2.5+0.5*PETSC_i);
        test_if_close(alpha4,-2.44985,"polyeig(std::vector<SPIMat>,PetscScalar) 3",1.E-5);
        std::tie(alpha4,eig4) = SPI::polyeig({K,C,M},0.33+0.005*PETSC_i);
        test_if_close(alpha4,0.3353,"polyeig(std::vector<SPIMat>,PetscScalar) 4",1.E-5);

        SPI::printf("------------ Mat eig tests end  -------------");
    }

    if(0){// I/O using hdf5
        SPI::printf("------------ I/O tests start  -------------");
        SPI::SPIVec A(2,"A_Vec");
        A(0,1.);
        A(1,2.);
        A();
        SPI::save(A,"saved_data.hdf5");
        SPI::SPIVec B(2,"B_Vec");
        B(0,1.+PETSC_i*0.5);
        B(1,2.*PETSC_i);
        B();
        SPI::save(B,"saved_data.hdf5");
        SPI::printf("------------ I/O tests end    -------------");
    }
    if(0){
        SPI::printf("------------ I/O tests2 start  -------------");
        SPI::SPIVec A_read(2,"A_Vec");
        SPI::load(A_read,"saved_data.hdf5");
        test_if_close(A_read(0,PETSC_TRUE),1.,"load(SPIVec,std::string) 1");
        test_if_close(A_read(1,PETSC_TRUE),2.,"load(SPIVec,std::string) 2");
        SPI::SPIVec B_read(2,"B_Vec");
        SPI::load(B_read,"saved_data.hdf5");
        test_if_close(PetscImaginaryPart(B_read(0,PETSC_TRUE)),0.5,"load(SPIVec,std::string) 3");
        test_if_close(B_read(1,PETSC_TRUE),0.,"load(SPIVec,std::string) 4");
        SPI::printf("------------ I/O tests2 end    -------------");
    }
    if(0){// I/O using binary for Mat
        SPI::printf("------------ I/O tests3 start  -------------");
        SPI::SPIMat A(2,2,"A");
        A(0,0,1.);
        A(1,0,2.);
        A(1,1,3.+4.59*PETSC_i);
        A();
        SPI::save(A,"saved_data_mat.dat");
        SPI::SPIMat B(2,2,"B");
        B(0,0,3.);
        B(1,0,4.);
        B(1,1,5.+4.89*PETSC_i);
        B();
        SPI::save(B,"saved_data_mat.dat");
        SPI::printf("------------ I/O tests3 end    -------------");
    }
    if(0){
        SPI::printf("------------ I/O tests4 start  -------------");
        SPI::SPIMat A_read(2,2,"A_Mat");
        SPI::load(A_read,"saved_data_mat.dat");
        test_if_close(A_read(0,0,PETSC_TRUE),1.,"load(SPIMat,std::string) 1");
        test_if_close(A_read(1,0,PETSC_TRUE),2.,"load(SPIMat,std::string) 2");
        test_if_close(A_read(1,1,PETSC_TRUE),3.,"load(SPIMat,std::string) 3");
        test_if_close(PetscImaginaryPart(A_read(1,1,PETSC_TRUE)),4.59,"load(SPIMat,std::string) 4");
        std::vector<SPI::SPIMat> AB(2,SPI::eye(2)*0.);
        SPI::load(AB,"saved_data_mat.dat");
        test_if_close(AB[0](0,0,PETSC_TRUE),1.,"load(std::vector<SPIMat>,std::string) 1");
        test_if_close(AB[0](1,0,PETSC_TRUE),2.,"load(std::vector<SPIMat>,std::string) 2");
        test_if_close(AB[0](1,1,PETSC_TRUE),3.,"load(std::vector<SPIMat>,std::string) 3");
        test_if_close(PetscImaginaryPart(AB[0](1,1,PETSC_TRUE)),4.59,"load(std::vector<SPIMat>,std::string) 4");
        test_if_close(AB[1](0,0,PETSC_TRUE),3.,"load(std::vector<SPIMat>,std::string) 5");
        test_if_close(AB[1](1,0,PETSC_TRUE),4.,"load(std::vector<SPIMat>,std::string) 6");
        test_if_close(AB[1](1,1,PETSC_TRUE),5.,"load(std::vector<SPIMat>,std::string) 7");
        test_if_close(PetscImaginaryPart(AB[1](1,1,PETSC_TRUE)),4.89,"load(std::vector<SPIMat>,std::string) 8");

        SPI::draw(AB[1]);

        SPI::printf("------------ I/O tests4 end    -------------");
    }
    if(1){
        SPI::printf("------------ block test start  -------------");
        SPI::SPIMat A(2,"A");
        A(0,0,2.);
        A(0,1,3.);
        A(1,1,4.);
        A();
        SPI::SPIMat B(SPI::eye(2),"I-identity");

        SPI::SPIMat block(SPI::block({{A,B},{A,B}}));
        test_if_close(block(0,0,PETSC_TRUE),2.,"block(std::vector<std::vector<SPIMat>>) 1");
        test_if_close(block(0,2,PETSC_TRUE),1.,"block(std::vector<std::vector<SPIMat>>) 2");
        test_if_close(block(3,1,PETSC_TRUE),4.,"block(std::vector<std::vector<SPIMat>>) 3");
        test_if_close(block(3,3,PETSC_TRUE),1.,"block(std::vector<std::vector<SPIMat>>) 4");
        SPI::printf("------------ block test end    -------------");
    }

    return 0;
}
