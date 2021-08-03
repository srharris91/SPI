#include "SPImain.hpp"
#include "tests.hpp"
#include <time.h>
#include <slepcsvd.h>

void test_if_true(PetscBool test,std::string name){
    if (test) { SPI::printf("\x1b[32m"+name+" test passed"+"\x1b[0m"); }
    else{ std::cout<<"\x1b[31m"+name+" test failed"+"\x1b[0m"<<std::endl;}
}

void test_if_close(PetscScalar value,PetscScalar golden, std::string name, PetscReal tol){
    PetscReal valuer=PetscRealPart(value);
    PetscReal goldenr=PetscRealPart(golden);
    if ((goldenr-tol<=valuer) && (valuer<=goldenr+tol)) { SPI::printf("\x1b[32m"+name+" test passed"+"\x1b[0m"); }
    else{ //std::cout<<"\x1b[31m"+name+" test failed"+"\x1b[0m"<<std::endl;
        //std::cout<<"      valuer="<<valuer<<std::endl;
        //std::cout<<"      goldenr="<<goldenr<<std::endl;
        SPI::printf("\x1b[31m"+name+" test failed"+"\x1b[0m");
        SPI::printf("\x1b[31m     goldenr=%.20f \x1b[0m",goldenr);
        SPI::printf("\x1b[31m     valuer =%.20f \x1b[0m",valuer );
    }
}

int tests(){
    PetscInt m=4, n=4;
    PetscBool alltests=PETSC_FALSE;
    // Vec tests
    if(alltests){
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
    if(alltests){
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
    if(alltests){
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
        //A.print();
        //x.print();
        //b.print();
        //Vec b2;
        //VecCreate(PETSC_COMM_WORLD,&b2);
        //VecSetSizes(b2,PETSC_DECIDE,4);
        //VecSetType(b2,VECMPICUDA);
        //MatMult(A.mat,x.vec,b2);
        //VecView(b2,PETSC_VIEWER_STDOUT_WORLD);

        SPI::printf("------------ A*x tests end   ---------------");
    }

    if(alltests){
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
    if(alltests){
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
    if(alltests){
        SPI::printf("------------ Mat eig tests start-------------");
        SPI::SPIMat A(2,"A");
        A(0,0,1.);
        A(0,1,1.);
        A(1,1,1.);
        A();
        SPI::SPIMat B(SPI::eye(2),"I-identity");
        PetscScalar alpha;
        SPI::SPIVec eig(2,"eig1");
        SPI::SPIVec eig2(2,"eig2");
        std::tie(alpha,eig,eig2) = SPI::eig(A,B,1.0+PETSC_i*0.5);
        eig /= eig.max(); // normalize by max amplitude
        test_if_close(alpha,1.,"eig(SPIMat,SPIMat,PetscScalar) 1",1.E-8);
        PetscScalar alpha2;
        std::tie(alpha2,eig2,eig) = SPI::eig(A,B,-1.0+PETSC_i*0.00005,1.E-19,10);
        eig2 /= eig2.max(); // normalize by max amplitude
        test_if_close(alpha2,1.,"eig(SPIMat,SPIMat,PetscScalar) 2",1.E-7);
        // eig_right
        std::tie(alpha,eig) = SPI::eig_right(A,B,1.0+PETSC_i*0.5);
        test_if_close(alpha,1.,"eig_right(SPIMat,SPIMat,PetscScalar) 1",1.E-8);
        std::tie(alpha2,eig) = SPI::eig_right(A,B,-1.0+PETSC_i*0.00005,1.E-19,10);
        test_if_close(alpha2,1.,"eig_right(SPIMat,SPIMat,PetscScalar) 2",1.E-7);
        // eig_init
        std::tie(alpha,eig,eig2) = SPI::eig_init(A,B,1.0+PETSC_i*0.5,eig.conj(),eig);
        test_if_close(alpha,1.,"eig_init(SPIMat,SPIMat,PetscScalar,SPIVec) 1",1.E-8);
        std::tie(alpha2,eig2,eig) = SPI::eig_init(A,B,-1.0+PETSC_i*0.00005,eig.conj(),eig,1.E-19,10);
        test_if_close(alpha2,1.,"eig_init(SPIMat,SPIMat,PetscScalar,SPIVec,PetscReal,PetscInt) 2",1.E-7);
        // eig_init_right
        std::tie(alpha,eig) = SPI::eig_init_right(A,B,1.0+PETSC_i*0.5,eig);
        test_if_close(alpha,1.,"eig_init_right(SPIMat,SPIMat,PetscScalar,SPIVec) 1",1.E-8);
        std::tie(alpha2,eig) = SPI::eig_init_right(A,B,-1.0+PETSC_i*0.00005,eig,1.E-19,10);
        test_if_close(alpha2,1.,"eig_init_right(SPIMat,SPIMat,PetscScalar,SPIVec,PetscReal,PetscInt) 2",1.E-7);

        // TODO these cases don't work as it is a linear problem
        //std::tie(alpha,eig) = SPI::polyeig({A,-B},1.0+0.5*PETSC_i);
        //test_if_close(alpha,1.,"polyeig(std::vector<SPIMat>,PetscScalar) 1",1.E-8);
        //std::tie(alpha,eig) = SPI::polyeig({A},1.0+0.5*PETSC_i);
        //test_if_close(alpha,1.,"polyeig(std::vector<SPIMat>,PetscScalar) 2",1.E-8);

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
        SPI::SPIVec eig4(4),eig5(4);
        std::tie(alpha4,eig4) = SPI::polyeig({K,C,M},-2.5+0.5*PETSC_i);
        test_if_close(alpha4,-2.44985,"polyeig(std::vector<SPIMat>,PetscScalar) 1",1.E-5);
        std::tie(alpha4,eig4) = SPI::polyeig({K,C,M},0.33+0.005*PETSC_i);
        test_if_close(alpha4,0.3353,"polyeig(std::vector<SPIMat>,PetscScalar) 2",1.E-5);
        // polyeig_init
        //std::tie(alpha,eig) = SPI::polyeig_init({A,-B},1.0+0.5*PETSC_i,eig);
        //test_if_close(alpha,1.,"polyeig_init(std::vector<SPIMat>,PetscScalar,SPIVec) 1",1.E-8);
        //std::tie(alpha,eig) = SPI::polyeig_init({A},1.0+0.5*PETSC_i,eig);
        //test_if_close(alpha,1.,"polyeig_init(std::vector<SPIMat>,PetscScalar,SPIVec) 2",1.E-8);
        std::tie(alpha4,eig4) = SPI::polyeig_init({K,C,M},-2.5+0.5*PETSC_i,eig4);
        test_if_close(alpha4,-2.44985,"polyeig_init(std::vector<SPIMat>,PetscScalar) 1",1.E-5);
        std::tie(alpha4,eig4) = SPI::polyeig_init({K,C,M},0.33+0.005*PETSC_i,eig4);
        test_if_close(alpha4,0.3353,"polyeig_init(std::vector<SPIMat>,PetscScalar) 2",1.E-5);

        SPI::printf("------------ Mat eig tests end  -------------");
    }

    if(alltests){// I/O using hdf5
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
    if(alltests){
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
    if(alltests){// I/O using binary for Mat
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
    if(alltests){
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

        //SPI::draw(AB[1]); // test failed TODO

        SPI::printf("------------ I/O tests4 end    -------------");
    }
    if(alltests){
        SPI::printf("------------ block test start  -------------");
        SPI::SPIMat A(2,"A");
        A(0,0,2.);
        A(0,1,3.);
        A(1,1,4.);
        A();
        SPI::SPIMat B(SPI::eye(2),"I-identity");

        SPI::SPIMat block(SPI::block({{A,B},{A,B}}));
        block();
        test_if_close(block(0,0,PETSC_TRUE),2.,"block(std::vector<std::vector<SPIMat>>) 1");
        test_if_close(block(0,2,PETSC_TRUE),1.,"block(std::vector<std::vector<SPIMat>>) 2");
        test_if_close(block(3,1,PETSC_TRUE),4.,"block(std::vector<std::vector<SPIMat>>) 3");
        test_if_close(block(3,3,PETSC_TRUE),1.,"block(std::vector<std::vector<SPIMat>>) 4");
        SPI::printf("------------ block test end    -------------");
    }
    if(alltests){
        SPI::printf("------------ LST_temporal test start  -------------");
        // create grid and derivatives using chebyshev polynomials
        PetscInt n=128;
        SPI::SPIVec y(SPI::set_Cheby_y(n),"yCheby");
        SPI::SPIgrid1D grid(y,"grid",SPI::Chebyshev);

        // channel flow Plane Poiseuille solution
        SPI::SPIMat U(SPI::diag(1.0-((grid.y)^2)),"U");
        SPI::SPIMat Uy(SPI::diag(-2.*grid.y),"Uy");
        SPI::SPIVec o(U.diag()*0.0,"o"); // zero vector
        SPI::SPIbaseflow channel(U.diag(),o,o,o,Uy.diag(),o,o,o,o,o,o,o,o,o);

        // parameters for Orr-Sommerfield eq.
        PetscScalar Re=2000.0;
        PetscScalar alpha=1.0;
        PetscScalar beta=0.0;

        // set parameters into param struct
        SPI::SPIparams params("channel parameter");
        params.Re = Re;
        params.omega = 0.3121002078-0.0197986590*PETSC_i;
        params.alpha = alpha;
        params.beta = beta;

        // solve eigenvalue problem
        SPI::SPIVec eigenfunction(grid.y.rows*4,"q");
        PetscScalar eigenvalue;
        eigenfunction.name = "eigenfunction";
        std::tie(eigenvalue,eigenfunction) = SPI::LST_temporal(params,grid,channel);
        //SPI::printfc("eigenvalue is: %.10f+%.10fi",eigenvalue);
        //eigenfunction.print();
        test_if_close(eigenvalue,0.3121002979-0.0197986590*PETSC_i,"LST_temporal 1",1e-9);
        std::tie(eigenvalue,eigenfunction) = SPI::LST_temporal(params,grid,channel,eigenfunction);
        //SPI::printfc(" eigenvalue is: %.10f+%.10fi",params.omega);
        test_if_close(eigenvalue,0.3121002979-0.0197986590*PETSC_i,"LST_temporal 2",1e-9);
        SPI::printf("------------ LST_temporal test end    -------------");
    }
    if(alltests){
        SPI::printf("------------ LST_spatial test start   -------------");
        PetscInt n=128;
        SPI::SPIVec y(SPI::set_Cheby_y(n),"yCheby");
        SPI::SPIgrid1D grid(y,"grid",SPI::Chebyshev);
        // channel flow Orr-Sommerfeld solution
        SPI::SPIMat U(SPI::diag(1.0-((grid.y)^2)),"U");
        SPI::SPIMat Uy(SPI::diag(-2.*grid.y),"Uy");
        //SPI::SPIMat Uyy(SPI::diag(-2.*SPI::ones(grid.y.rows)),"Uyy");

        SPI::SPIparams params("channel parameter");
        params.Re = 2000.0;
        params.omega = 0.3;
        params.alpha = 0.97875+0.044394*PETSC_i;
        params.beta = 0.0;

        SPI::SPIVec eigenfunction(grid.y.rows*6,"q");
        PetscScalar eigenvalue;
        eigenfunction.name = "eigenfunction";

        SPI::SPIVec o(U.diag()*0.0,"o");
        SPI::SPIbaseflow channel(U.diag(),o,o,o,Uy.diag(),o,o,o,o,o,o,o,o,o);

        std::tie(eigenvalue,eigenfunction) = SPI::LST_spatial(params,grid,channel);
        test_if_close(eigenvalue,0.97875+0.044394*PETSC_i,"LST_spatial 1",1e-5);
        test_if_close(params.alpha,0.97875+0.044394*PETSC_i,"LST_spatial 1",1e-5);
        std::tie(eigenvalue,eigenfunction) = SPI::LST_spatial(params,grid,channel,eigenfunction); // with initial guess
        test_if_close(eigenvalue,0.97875+0.044394*PETSC_i,"LST_spatial 1",1e-5);
        test_if_close(params.alpha,0.97875+0.044394*PETSC_i,"LST_spatial 1",1e-5);
        params.alpha = 0.34312+0.049677*PETSC_i;
        std::tie(eigenvalue,eigenfunction) = SPI::LST_spatial(params,grid,channel,eigenfunction);
        test_if_close(eigenvalue,0.34312+0.049677*PETSC_i,"LST_spatial 2",1e-5);
        test_if_close(params.alpha,0.34312+0.049677*PETSC_i,"LST_spatial 2",1e-5);
        params.alpha = 0.61+0.1*PETSC_i;
        std::tie(eigenvalue,eigenfunction) = SPI::LST_spatial(params,grid,channel,eigenfunction);
        test_if_close(eigenvalue,0.61167+0.140492*PETSC_i,"LST_spatial 3",1e-5);
        test_if_close(params.alpha,0.61167+0.140492*PETSC_i,"LST_spatial 3",1e-5);
        SPI::printf("------------ LST_spatial test end     -------------");
    }
    if(alltests){
        SPI::printf("------------ LSTNP_spatial test start   -------------");
        PetscInt n=64;
        SPI::SPIVec y(SPI::set_Cheby_y(n),"yCheby");
        SPI::SPIgrid1D grid(y,"grid",SPI::Chebyshev);
        // channel flow Orr-Sommerfeld solution
        SPI::SPIVec U((1.0-(grid.y*grid.y)),"U");
        SPI::SPIVec Uy((-2.*grid.y),"Uy");
        //SPI::SPIMat Uyy(SPI::diag(-2.*SPI::ones(grid.y.rows)),"Uyy");

        SPI::SPIparams params("channel parameter");
        params.Re = 2000.0;
        params.omega = 0.3;
        params.alpha = 0.97875+0.044394*PETSC_i;
        params.beta = 0.0;

        SPI::SPIVec eigenfunction(grid.y.rows*16,"q");
        SPI::SPIVec eigenfunction8(grid.y.rows*8,"q");
        SPI::SPIVec leigenfunction(grid.y.rows*16,"q");
        PetscScalar eigenvalue;
        eigenfunction.name = "eigenfunction";

        SPI::SPIVec o(U*0.0,"o");
        SPI::SPIbaseflow channel(U,o,o,o,Uy,o,o,o,o,o,o,o,o,o);

        PetscScalar cg;
        std::tie(eigenvalue,cg,leigenfunction,eigenfunction) = SPI::LSTNP_spatial(params,grid,channel);
        SPI::printfc("cg = %g+%gi",cg);
        test_if_close(eigenvalue,0.978748+0.04439397*PETSC_i,"LSTNP_spatial 1",1e-5);
        test_if_close(params.alpha,0.978748+0.04439397*PETSC_i,"LSTNP_spatial 1",1e-5);
        // LSTNP_spatial_right
        std::tie(eigenvalue,eigenfunction8) = SPI::LSTNP_spatial_right(params,grid,channel);
        test_if_close(params.alpha,0.978748+0.04439397*PETSC_i,"LSTNP_spatial_right 1",1e-5);
        //SPI::printfc("eigenvalue is %.10f + %.10fi",eigenvalue);
        params.alpha = 0.978748+0.04439397*PETSC_i;
        std::tie(eigenvalue,cg,leigenfunction,eigenfunction) = SPI::LSTNP_spatial(params,grid,channel,eigenfunction.conj(),eigenfunction); // with initial guess
        SPI::printfc("cg = %g+%gi",cg);
        test_if_close(eigenvalue,0.978748+0.04439397*PETSC_i,"LSTNP_spatial 2",1e-5);
        test_if_close(params.alpha,0.978748+0.04439397*PETSC_i,"LSTNP_spatial 2",1e-5);
        std::tie(eigenvalue,eigenfunction8) = SPI::LSTNP_spatial_right(params,grid,channel,eigenfunction);
        test_if_close(params.alpha,0.978748+0.04439397*PETSC_i,"LSTNP_spatial_right 2",1e-5);
        //SPI::printfc("eigenvalue is %.10f + %.10fi",eigenvalue);
        params.alpha = 0.34305+0.0498376872*PETSC_i;
        std::tie(eigenvalue,cg,leigenfunction,eigenfunction) = SPI::LSTNP_spatial(params,grid,channel,leigenfunction,eigenfunction);
        SPI::printfc("cg = %g+%gi",cg);
        test_if_close(eigenvalue,0.34305+0.049837*PETSC_i,"LSTNP_spatial 3",1e-5);
        std::tie(eigenvalue,eigenfunction8) = SPI::LSTNP_spatial_right(params,grid,channel,eigenfunction);
        test_if_close(eigenvalue,0.34305+0.049837*PETSC_i,"LSTNP_spatial_right 3",1e-5);
        //test_if_close(params.alpha,0.34305+0.049837*PETSC_i,"LSTNP_spatial 2",1e-5);
        //SPI::printfc("eigenvalue is %.10f + %.10fi",eigenvalue);
        params.alpha = 0.6116672+0.140493*PETSC_i;
        std::tie(eigenvalue,cg,leigenfunction,eigenfunction) = SPI::LSTNP_spatial(params,grid,channel,leigenfunction,eigenfunction);
        SPI::printfc("cg = %g+%gi",cg);
        test_if_close(eigenvalue,0.6116672+0.140493*PETSC_i,"LSTNP_spatial 4",1e-5);
        std::tie(eigenvalue,eigenfunction8) = SPI::LSTNP_spatial_right(params,grid,channel,eigenfunction);
        test_if_close(eigenvalue,0.6116672+0.140493*PETSC_i,"LSTNP_spatial_right 4",1e-5);
        //test_if_close(params.alpha,0.635797+0.08405*PETSC_i,"LSTNP_spatial 3",1e-5);
        //SPI::printfc("eigenvalue is %.10f + %.10fi",eigenvalue);
        SPI::printf("------------ LSTNP_spatial test end     -------------");
    }
    if(0){
        SPI::printf("------------ Chebyshev derivatives start -----------");
        PetscInt n=16;
        //SPI::SPIVec y(SPI::set_Cheby_y(n),"yCheby");
        SPI::SPIVec y(SPI::set_Cheby_mapped_y(0,61.,n) ,"yCheby");
        SPI::SPIgrid1D grid(y,"grid",SPI::Chebyshev);
        grid.Dyy.print();
        SPI::printf("------------ Chebyshev derivatives end   -----------");
    }
    if(alltests){
        SPI::printf("------------ LSTNP Blasius boundary layer start -----------");
        PetscInt n=168;
        //SPI::SPIVec y(SPI::set_Cheby_mapped_y(0.,61.,n) ,"yCheby");
        //SPI::SPIgrid1D grid(y,"grid",SPI::Chebyshev);
        //SPI::SPIVec y(SPI::linspace(0.,61.,n) ,"yFD");
        //SPI::printf("n = %d",n);
        SPI::SPIVec y(SPI::set_FD_stretched_y(61.,n) ,"yFD");
        SPI::SPIgrid1D grid(y,"grid",SPI::FD);
        //grid.print();
        //(grid.Dy*grid.y).print();

        SPI::SPIparams params("Blasius parameters");
        params.Re = 400.0;
        params.omega = 86.*params.Re/(1000000.);
        params.nu = 1./params.Re;
        params.x = params.Re;
        params.x_start = params.x;
        params.alpha = 0.094966+0.004564*PETSC_i;
        //params.alpha = 0.106654+0.0018979*PETSC_i;
        params.beta = 0.0;
        //params.print();

        SPI::SPIVec eigenfunction(grid.y.rows*16,"q");
        //SPI::SPIVec eig_vec(grid.y.rows*8,"q");
        SPI::SPIVec leigenfunction(grid.y.rows*16,"q");
        PetscScalar eigenvalue;
        eigenfunction.name = "eigenfunction";

        SPI::SPIbaseflow bl_flow = SPI::blasius(params,grid);
        if(0){ // set to parallel baseflow
            SPI::SPIVec o(SPI::zeros(n),"zero");
            bl_flow.Ux = o;
            bl_flow.Uxy = o;
            bl_flow.V = o;
            bl_flow.Vy = o;
        }
        //bl_flow.print();

        PetscScalar cg;
        std::tie(eigenvalue,cg,leigenfunction,eigenfunction) = SPI::LSTNP_spatial(params,grid,bl_flow);
        SPI::printfc("cg = %g+%gi",cg);
        //std::tie(eigenvalue,eig_vec) = SPI::LST_spatial(params,grid,bl_flow);
        test_if_close(eigenvalue,0.094966+0.004564*PETSC_i,"LSTNP_spatial 1",1e-5);
        test_if_close(params.alpha,0.094966+0.004564*PETSC_i,"LSTNP_spatial 1",1e-5);
        std::tie(eigenvalue,eigenfunction) = SPI::LSTNP_spatial_right(params,grid,bl_flow);
        test_if_close(params.alpha,0.094966+0.004564*PETSC_i,"LSTNP_spatial_right 1",1e-5);
        //SPI::printfc("eigenvalue is %.10f + %.10fi",eigenvalue);

        std::tie(eigenvalue,cg,leigenfunction,eigenfunction) = SPI::LSTNP_spatial(params,grid,bl_flow,leigenfunction,eigenfunction); // with initial guess
        SPI::printfc("cg = %g+%gi",cg);
        test_if_close(eigenvalue,0.094966+0.004564*PETSC_i,"LSTNP_spatial 2",1e-5);
        std::tie(eigenvalue,eigenfunction) = SPI::LSTNP_spatial_right(params,grid,bl_flow,eigenfunction); // with initial guess
        test_if_close(eigenvalue,0.094966+0.004564*PETSC_i,"LSTNP_spatial_right 2",1e-5);
        //SPI::printfc("eigenvalue is %.10f + %.10fi",eigenvalue);

        params.alpha = 0.1067+0.001898*PETSC_i;
        std::tie(eigenvalue,cg,leigenfunction,eigenfunction) = SPI::LSTNP_spatial(params,grid,bl_flow); 
        SPI::printfc("cg = %g+%gi",cg);
        test_if_close(eigenvalue,0.10665+0.0018979*PETSC_i,"LSTNP_spatial 3",1e-5);
        std::tie(eigenvalue,eigenfunction) = SPI::LSTNP_spatial_right(params,grid,bl_flow); 
        test_if_close(eigenvalue,0.10665+0.0018979*PETSC_i,"LSTNP_spatial_right 3",1e-5);
        //SPI::printfc("eigenvalue is %.10f + %.10fi",eigenvalue);
        std::tie(eigenvalue,cg,leigenfunction,eigenfunction) = SPI::LSTNP_spatial(params,grid,bl_flow,leigenfunction,eigenfunction); // with initial guess
        SPI::printfc("cg = %g+%gi",cg);
        test_if_close(eigenvalue,0.10665+0.0018979*PETSC_i,"LSTNP_spatial 4",1e-5);
        SPI::printfc("eigenvalue is %.10f + %.10fi",eigenvalue);
        std::tie(eigenvalue,eigenfunction) = SPI::LSTNP_spatial_right(params,grid,bl_flow,eigenfunction); // with initial guess
        test_if_close(eigenvalue,0.10665+0.0018979*PETSC_i,"LSTNP_spatial_right 4",1e-5);

        SPI::printf("------------ LSTNP Blasius boundary layer end   -----------");
    }
    if(0){
        SPI::printf("------------ LSTNP Blasius boundary layer start -----------");
        PetscInt n=168;
        SPI::SPIVec y(SPI::set_Cheby_mapped_y(0.,61.,n) ,"yCheby");
        SPI::SPIgrid1D grid(y,"grid",SPI::UltraS);
        //SPI::SPIVec y(SPI::linspace(0.,61.,n) ,"yFD");
        //SPI::printf("n = %d",n);
        //SPI::SPIVec y(SPI::set_FD_stretched_y(61.,n) ,"yFD");
        //SPI::SPIgrid1D grid(y,"grid",SPI::FD);
        //grid.print();
        //(grid.Dy*grid.y).print();

        SPI::SPIparams params("Blasius parameters");
        params.Re = 400.0;
        params.omega = 86.*params.Re/(1000000.);
        params.nu = 1./params.Re;
        params.x = params.Re;
        params.x_start = params.x;
        params.alpha = 0.094966+0.004564*PETSC_i;
        //params.alpha = 0.106654+0.0018979*PETSC_i;
        params.beta = 0.0;
        //params.print();

        SPI::SPIVec eigenfunction(grid.y.rows*16,"q");
        //SPI::SPIVec eig_vec(grid.y.rows*8,"q");
        SPI::SPIVec leigenfunction(grid.y.rows*16,"q");
        PetscScalar eigenvalue;
        eigenfunction.name = "eigenfunction";

        SPI::SPIbaseflow bl_flow = SPI::blasius(params,grid);
        if(0){ // set to parallel baseflow
            SPI::SPIVec o(SPI::zeros(n),"zero");
            bl_flow.Ux = o;
            bl_flow.Uxy = o;
            bl_flow.V = o;
            bl_flow.Vy = o;
        }
        //bl_flow.print();

        PetscScalar cg;
        std::tie(eigenvalue,cg,leigenfunction,eigenfunction) = SPI::LSTNP_spatial(params,grid,bl_flow);
        SPI::printfc("cg = %g+%gi",cg);
        //std::tie(eigenvalue,eig_vec) = SPI::LST_spatial(params,grid,bl_flow);
        test_if_close(eigenvalue,0.094966+0.004564*PETSC_i,"LSTNP_spatial 1",1e-5);
        test_if_close(params.alpha,0.094966+0.004564*PETSC_i,"LSTNP_spatial 1",1e-5);
        //std::tie(eigenvalue,eigenfunction) = SPI::LSTNP_spatial_right(params,grid,bl_flow);
        //test_if_close(params.alpha,0.094966+0.004564*PETSC_i,"LSTNP_spatial_right 1",1e-5);
        //SPI::printfc("eigenvalue is %.10f + %.10fi",eigenvalue);

        std::tie(eigenvalue,cg,leigenfunction,eigenfunction) = SPI::LSTNP_spatial(params,grid,bl_flow,leigenfunction,eigenfunction); // with initial guess
        SPI::printfc("cg = %g+%gi",cg);
        test_if_close(eigenvalue,0.094966+0.004564*PETSC_i,"LSTNP_spatial 2",1e-5);
        //std::tie(eigenvalue,eigenfunction) = SPI::LSTNP_spatial_right(params,grid,bl_flow,eigenfunction); // with initial guess
        //test_if_close(eigenvalue,0.094966+0.004564*PETSC_i,"LSTNP_spatial_right 2",1e-5);
        //SPI::printfc("eigenvalue is %.10f + %.10fi",eigenvalue);

        params.alpha = 0.1067+0.001898*PETSC_i;
        std::tie(eigenvalue,cg,leigenfunction,eigenfunction) = SPI::LSTNP_spatial(params,grid,bl_flow); 
        SPI::printfc("cg = %g+%gi",cg);
        test_if_close(eigenvalue,0.10665+0.0018979*PETSC_i,"LSTNP_spatial 3",1e-5);
        //std::tie(eigenvalue,eigenfunction) = SPI::LSTNP_spatial_right(params,grid,bl_flow); 
        //test_if_close(eigenvalue,0.10665+0.0018979*PETSC_i,"LSTNP_spatial_right 3",1e-5);
        //SPI::printfc("eigenvalue is %.10f + %.10fi",eigenvalue);
        std::tie(eigenvalue,cg,leigenfunction,eigenfunction) = SPI::LSTNP_spatial(params,grid,bl_flow,leigenfunction,eigenfunction); // with initial guess
        SPI::printfc("cg = %g+%gi",cg);
        test_if_close(eigenvalue,0.10665+0.0018979*PETSC_i,"LSTNP_spatial 4",1e-5);
        SPI::printfc("eigenvalue is %.10f + %.10fi",eigenvalue);
        //std::tie(eigenvalue,eigenfunction) = SPI::LSTNP_spatial_right(params,grid,bl_flow,eigenfunction); // with initial guess
        //test_if_close(eigenvalue,0.10665+0.0018979*PETSC_i,"LSTNP_spatial_right 4",1e-5);

        SPI::printf("------------ LSTNP Blasius boundary layer end   -----------");
    }
    if(alltests){
        SPI::printf("------------ LST Blasius boundary layer start -----------");
        PetscInt n=200;
        SPI::SPIVec y(SPI::set_Cheby_mapped_y(0.,21.,n) ,"yCheby");
        SPI::SPIgrid1D grid(y,"grid",SPI::Chebyshev);
        //SPI::SPIVec y(SPI::linspace(0.,61.,n) ,"yFD");
        //SPI::printf("n = %d",n);
        //SPI::SPIVec y(SPI::set_FD_stretched_y(21.,n) ,"yFD");
        //SPI::SPIgrid1D grid(y,"grid",SPI::FD);
        //grid.print();
        //(grid.Dy*grid.y).print();

        SPI::SPIparams params("Blasius parameters");
        params.Re = 1000.0/1.7208;
        params.omega = 0.26/1.7208;
        params.nu = 1./params.Re;
        params.x = params.Re;
        params.x_start = params.x;
        params.alpha = (0.74155+0.345132*PETSC_i)/1.7208;
        params.beta = 0.0;

        SPI::SPIVec eigenfunction(grid.y.rows*8,"q");
        PetscScalar eigenvalue;
        eigenfunction.name = "eigenfunction";

        SPI::SPIbaseflow bl_flow = SPI::blasius(params,grid);
        if(1){ // set to parallel baseflow
            SPI::SPIVec o(SPI::zeros(n),"zero");
            bl_flow.Ux = o;
            bl_flow.Uxy = o;
            bl_flow.V = o;
            bl_flow.Vy = o;
        }
        //bl_flow.print();

        std::tie(eigenvalue,eigenfunction) = SPI::LST_spatial(params,grid,bl_flow);
        test_if_close(eigenvalue*1.7208,(0.74155+0.345132*PETSC_i),"LST_spatial 1",1e-5);
        test_if_close(params.alpha*1.7208,(0.74155+0.345132*PETSC_i),"LST_spatial 1",1e-5);
        SPI::printfc("eigenvalue is %.10f + %.10fi",eigenvalue*1.7208);

        params.alpha = (0.54213+0.083968*PETSC_i)/1.7208;
        std::tie(eigenvalue,eigenfunction) = SPI::LST_spatial(params,grid,bl_flow); 
        test_if_close(eigenvalue*1.7208,(0.54213+0.083968*PETSC_i),"LST_spatial 2",1e-5);
        SPI::printfc("eigenvalue is %.10f + %.10fi",eigenvalue*1.7208);

        params.alpha = (0.29967+0.230773*PETSC_i)/1.7208;
        std::tie(eigenvalue,eigenfunction) = SPI::LST_spatial(params,grid,bl_flow); 
        test_if_close(eigenvalue*1.7208,(0.29967+0.083968*PETSC_i),"LST_spatial 3",1e-5);
        SPI::printfc("eigenvalue is %.10f + %.10fi",eigenvalue*1.7208);

        SPI::printf("------------ LST Blasius boundary layer end   -----------");
    }
    if(alltests){
        SPI::printf("------------ LSTNP parallel Blasius boundary layer start -----------");
        PetscInt n=1000; // needs 1700 if using finite difference with default stretching...., 1000 if using delta=1.01 stretching
        //SPI::SPIVec y(SPI::set_Cheby_mapped_y(0.,21.,n) ,"yCheby");
        //SPI::SPIgrid1D grid(y,"grid",SPI::Chebyshev);
        //SPI::SPIVec y(SPI::linspace(0.,61.,n) ,"yFD");
        //SPI::printf("n = %d",n);
        SPI::SPIVec y(SPI::set_FD_stretched_y(21.,n,1.01) ,"yFD");
        SPI::SPIgrid1D grid(y,"grid",SPI::FD);
        //grid.print();
        //(grid.Dy*grid.y).print();

        SPI::SPIparams params("Blasius parameters");
        params.Re = 1000.0/1.7208;
        params.omega = 0.26/1.7208;
        params.nu = 1./params.Re;
        params.x = params.Re;
        params.x_start = params.x;
        params.alpha = (0.74155+0.345132*PETSC_i)/1.7208;
        params.beta = 0.0;

        SPI::SPIVec eigenfunction(grid.y.rows*8,"q");
        PetscScalar eigenvalue;
        eigenfunction.name = "eigenfunction";

        SPI::SPIbaseflow bl_flow = SPI::blasius(params,grid);
        if(1){ // set to parallel baseflow
            SPI::SPIVec o(SPI::zeros(n),"zero");
            bl_flow.Ux = o;
            bl_flow.Uxy = o;
            bl_flow.V = o;
            bl_flow.Vy = o;
        }
        //bl_flow.print();

        std::tie(eigenvalue,eigenfunction) = SPI::LSTNP_spatial_right(params,grid,bl_flow);
        test_if_close(eigenvalue*1.7208,(0.74155+0.345132*PETSC_i),"LSTNP_spatial_right 1",1e-5);
        test_if_close(params.alpha*1.7208,(0.74155+0.345132*PETSC_i),"LSTNP_spatial_right 1",1e-5);
        SPI::printfc("eigenvalue is %.10f + %.10fi",eigenvalue*1.7208);

        params.alpha = (0.54213+0.083968*PETSC_i)/1.7208;
        std::tie(eigenvalue,eigenfunction) = SPI::LSTNP_spatial_right(params,grid,bl_flow); 
        test_if_close(eigenvalue*1.7208,(0.54213+0.083968*PETSC_i),"LSTNP_spatial_right 2",1e-5);
        SPI::printfc("eigenvalue is %.10f + %.10fi",eigenvalue*1.7208);

        params.alpha = (0.29967+0.230773*PETSC_i)/1.7208;
        std::tie(eigenvalue,eigenfunction) = SPI::LSTNP_spatial_right(params,grid,bl_flow); 
        test_if_close(eigenvalue*1.7208,(0.29967+0.083968*PETSC_i),"LSTNP_spatial_right 3",1e-5);
        SPI::printfc("eigenvalue is %.10f + %.10fi",eigenvalue*1.7208);

        SPI::printf("------------ LSTNP parallel Blasius boundary layer end   -----------");
    }
    if(alltests){
        SPI::printf("------------ Fourier Collocated derivative operator start -----------");
        PetscInt n=8; // must be even number for Fourier derivatives
        SPI::SPIVec t(SPI::set_Fourier_t(2.0*M_PI,n) ,"tFT");
        SPI::SPIMat Dt(SPI::set_D_Fourier(t),"Dt");
        SPI::SPIVec y(sin(t));
        SPI::SPIVec yp(cos(t));
        test_if_close((Dt*y)(2,PETSC_TRUE),yp(2,PETSC_TRUE),"set_D_Fourier 1",1e-12);
        test_if_close(SPI::L2((Dt*y)-yp),0.0,"set_D_Fourier 2",1e-12);
        Dt.print();
        SPI::printf("------------ Fourier Collocated derivative operator end   -----------");
    }
    if(alltests){
        SPI::printf("------------ BV Orthogonalize start -----------");
        //SPI::SPIMat A(2,2,"A");
        //A(0,0,1.);
        //A(0,1,1.);
        //A(1,0,2.);
        //A(1,1,1.);
        //A();
        //A.print();
        SPI::SPIVec A1(2,"A1"),A2(2,"A2");
        A1(0,1.0+0.5*PETSC_i); A2(0,1.);
        A1(1,2.0+0.5*PETSC_i); A2(1,1.);
        A1();A2();
        Vec As[]={A1.vec,A2.vec};
        // orthogonalize
        if(0){
            PetscErrorCode ierr;
            BV bv;
            ierr = BVCreate(PETSC_COMM_WORLD,&bv); CHKERRXX(ierr);
            PetscInt m=2;
            BVSetSizesFromVec(bv,A1.vec,m);
            ierr = BVSetFromOptions(bv);CHKERRXX(ierr);
            ierr = BVInsertVecs(bv,0,&m,As,PETSC_TRUE);
            SPI::SPIMat AorthH("AorthH");
            ierr = BVCreateMat(bv,&AorthH.mat); CHKERRXX(ierr);
            AorthH.rows=A1.rows;
            AorthH.cols=m;
            AorthH.print();
            ierr = BVDestroy(&bv); CHKERRXX(ierr);
        }
        SPI::SPIMat Aorth(SPI::orthogonalize({A1,A2}));
        test_if_close(Aorth(1,0,PETSC_TRUE),0.85280286542244166,"orthogonalize 1",1e-12);

        SPI::SPIMat Aorth2(2,2,"Aorth2");
        Aorth2 = SPI::orthogonalize({A1,A2});
        //Aorth.print();
        test_if_close(Aorth2(1,0,PETSC_TRUE),0.85280286542244166,"orthogonalize 2",1e-12);
        //ierr = MatSetType(AorthH.mat,MATMPIAIJ);CHKERRXX(ierr);
        //AorthH.print();
        //SPI::SPIMat Aorth(AorthH,"Aorth");
        //AorthH.H();
        //AorthH.print();
        //(AorthH*A1).print();
        //(AorthH*AorthH).print();

        SPI::printf("------------ BV Orthogonalize end   -----------");
    }
    if(alltests){
        SPI::printf("------------ Gram-Schmidt Orthogonalize start -----------");
        PetscInt n=8;
        SPI::SPIVec y(SPI::set_Cheby_mapped_y(-1.,1.,n) ,"yCheby");
        SPI::SPIgrid1D grid(y,"grid",SPI::UltraS);
        SPI::SPIVec A1(grid.That*(1.0-(grid.y^2)),"A1"),A2(grid.That*(1.0+(grid.y^2)),"A2");
        std::vector<SPI::SPIVec> A={A1,A2};
        //SPI::SPIMat Aorth(SPI::orthogonalize(A,grid));
        //test_if_close(Aorth(2,0,PETSC_TRUE),-0.484122918275927,"orthogonalize 1",1e-12);
        //test_if_close(Aorth(2,1,PETSC_TRUE),1.082531754730548,"orthogonalize 2",1e-12);
        std::vector<SPI::SPIVec> Aorth(SPI::orthogonalize(A,grid));
        test_if_close(Aorth[0](2,PETSC_TRUE),-0.484122918275927,"orthogonalize 1",1e-12);
        test_if_close(Aorth[1](2,PETSC_TRUE),1.082531754730548,"orthogonalize 2",1e-12);

        SPI::printf("------------ Gram-Schmidt Orthogonalize end   -----------");
    }
    if(alltests){
        SPI::printf("------------ Gram-Schmidt Orthogonalize start -----------");
        PetscInt n=8;
        SPI::SPIVec y(SPI::set_Cheby_mapped_y(-1.,1.,n) ,"yCheby");
        SPI::SPIgrid1D grid(y,"grid",SPI::UltraS);
        SPI::SPIVec A1(grid.That*(1.0-(grid.y^2)),"A1"),A2(grid.That*(1.0+(grid.y^2)),"A2");
        std::vector<SPI::SPIVec> A={A1,A2};
        //SPI::SPIMat Aorth(SPI::orthogonalize(A,grid));
        //test_if_close(Aorth(2,0,PETSC_TRUE),-0.484122918275927,"orthogonalize 1",1e-12);
        //test_if_close(Aorth(2,1,PETSC_TRUE),1.082531754730548,"orthogonalize 2",1e-12);
        std::vector<SPI::SPIVec> Aorth(SPI::orthogonalize(A,grid));
        test_if_close(Aorth[0](2,PETSC_TRUE),-0.484122918275927,"orthogonalize 1",1e-12);
        test_if_close(Aorth[1](2,PETSC_TRUE),1.082531754730548,"orthogonalize 2",1e-12);

        // now try multiple length projection and integration
        SPI::SPIVec A1l(SPI::block({
                    {SPI::diag(A1),grid.O},
                    {grid.O,SPI::diag(A2)}
                    })()*SPI::ones(2*n),"A1l");
        SPI::SPIVec A2l(SPI::block({
                    {SPI::diag(A2),grid.O},
                    {grid.O,SPI::diag(A1)}
                    })()*SPI::ones(2*n),"A2l");
        //A1.print();
        //A2.print();
        std::vector<SPI::SPIVec> Al = {A1l,A2l};
        std::vector<SPI::SPIVec> Alorth(SPI::orthogonalize(Al,grid));
        test_if_close(Alorth[0](2,PETSC_TRUE),-0.228217732293819,"orthogonalize 3",1e-12);
        test_if_close(Alorth[0](10,PETSC_TRUE),0.228217732293819,"orthogonalize 4",1e-12);
        test_if_close(Alorth[1](0,PETSC_TRUE),0.714434508311760,"orthogonalize 5",1e-12);
        test_if_close(Alorth[1](8,PETSC_TRUE),-0.306186217847897,"orthogonalize 6",1e-12);
        //Alorth[0].print();
        //Alorth[1].print();
        SPI::SPIVec A1ll(SPI::block({
                    {SPI::diag(A1),grid.O,grid.O,grid.O},
                    {grid.O,SPI::diag(A2),grid.O,grid.O},
                    {grid.O,grid.O,SPI::diag(A1),grid.O},
                    {grid.O,grid.O,grid.O,SPI::diag(A2)},
                    })()*SPI::ones(4*n),"A1ll");
        SPI::SPIVec A2ll(SPI::block({
                    {SPI::diag(A2),grid.O,grid.O,grid.O},
                    {grid.O,SPI::diag(A1),grid.O,grid.O},
                    {grid.O,grid.O,SPI::diag(A2),grid.O},
                    {grid.O,grid.O,grid.O,SPI::diag(A1)},
                    })()*SPI::ones(4*n),"A2ll");
        std::vector<SPI::SPIVec> All = {A1ll,A2ll};
        std::vector<SPI::SPIVec> Allorth(SPI::orthogonalize(All,grid));
        test_if_close(Allorth[0](2,PETSC_TRUE),-0.161374306091976,"orthogonalize 7",1e-12);
        test_if_close(Allorth[0](8,PETSC_TRUE),0.484122918275927,"orthogonalize 8",1e-12);
        test_if_close(Allorth[0](10,PETSC_TRUE),0.161374306091976,"orthogonalize 9",1e-12);
        test_if_close(Allorth[0](18,PETSC_TRUE),-0.161374306091976,"orthogonalize 10",1e-12);
        test_if_close(Allorth[0](24,PETSC_TRUE),0.484122918275927,"orthogonalize 11",1e-12);
        test_if_close(Allorth[0](26,PETSC_TRUE),0.161374306091976,"orthogonalize 12",1e-12);
        test_if_close(Allorth[1](2,PETSC_TRUE),0.360843918243516,"orthogonalize 13",1e-12);
        test_if_close(Allorth[1](8,PETSC_TRUE),-0.216506350946110,"orthogonalize 14",1e-12);
        test_if_close(Allorth[1](10,PETSC_TRUE),-0.360843918243516,"orthogonalize 15",1e-12);
        test_if_close(Allorth[1](18,PETSC_TRUE),0.360843918243516,"orthogonalize 16",1e-12);
        test_if_close(Allorth[1](24,PETSC_TRUE),-0.216506350946110,"orthogonalize 17",1e-12);
        test_if_close(Allorth[1](26,PETSC_TRUE),-0.360843918243516,"orthogonalize 18",1e-12);

        SPI::printf("------------ Gram-Schmidt Orthogonalize end   -----------");
    }
    if(alltests){
        SPI::printf("------------ Gram-Schmidt Orthogonalize 2D start -----------");
        PetscInt n=8;
        SPI::SPIVec y(SPI::set_Cheby_mapped_y(-1.,1.,n) ,"yCheby");
        SPI::SPIVec t(SPI::set_Cheby_mapped_y(0.,3.,n) ,"tCheby");
        SPI::SPIgrid2D grid(y,t,"grid",SPI::Chebyshev,SPI::Chebyshev);
        SPI::SPIgrid2D gridFD(y,t,"grid",SPI::FD,SPI::FD);
        SPI::SPIgrid2D gridUltraS(y,t,"grid",SPI::UltraS,SPI::FD);

        // test integration
        SPI::SPIVec A1((1.0-(grid.y)),"A1"),A2(grid.grid1Dy.That*(1.0+(grid.y)),"A2");
        SPI::SPIVec A1l(SPIVec1Dto2D(grid,A1));
        SPI::SPIVec A2l(SPIVec1Dto2D(grid,A2));
        //std::cout<<"intChebyshev = "<<SPI::integrate(A1l,grid)<<std::endl;
        //std::cout<<"intFD = "<<SPI::integrate(A1l,gridFD)<<std::endl;
        test_if_close(SPI::integrate(A1l,grid),6.0,"integrate SPIgrid2D Chebyshev 1",1e-12);
        test_if_close(SPI::integrate(A1l,gridFD),6.0,"integrate SPIgrid2D FD 2",1e-12);
        test_if_close(SPI::integrate(A2l,gridUltraS),6.0,"integrate SPIgrid2D UltraS 3",1e-12);
        //std::cout<<"intUltraS = "<<SPI::integrate(A2l,gridUltraS)<<std::endl;
        //std::cout<<"int_1D = "<<SPI::integrate(A1,grid.grid1Dy)<<std::endl;
        test_if_close(SPI::integrate(A1,grid.grid1Dy),2.0,"integrate SPIgrid1D Chebyshev 4",1e-12);

        // test projection
        SPI::SPIVec yp1(y+1.0);
        SPI::SPIVec nym1(1.0-y);
        A2l = SPIVec1Dto2D(grid,yp1);
        SPI::SPIVec A1lp,A2lp;
        A1lp = SPI::proj(A1l,A2l,grid);
        //A2lp = SPI::proj(A1l,A2l,gridFD);
        //A1lp.print();
        //A2lp.print();
        //SPI::proj(nym1,yp1,grid.grid1Dy).print();
        test_if_close(A1lp(1,PETSC_TRUE),SPI::proj(nym1,yp1,grid.grid1Dy)(1,PETSC_TRUE),"proj SPIgrid2D 1",1e-12);
        //std::tie(eigenvalue,eigenfunction) = SPI::LST_spatial(params,grid,bl_flow); 
        //std::cout<<"norm = "<<std::sqrt(SPI::integrate(SPI::abs(A1lp)^2,grid))<<std::endl;
        //A1l = SPIVec1Dto2D(grid,nym1);
        //A2l = SPIVec1Dto2D(grid,yp1);
        std::vector<SPI::SPIVec> A12ls = {A1l,A2l};
        std::vector<SPI::SPIVec> A12lso;
        A12lso = SPI::orthogonalize(A12ls,grid);
        //for(PetscInt i=0; i<A12lso.size(); ++i){
            //A12lso[i].print();
        //}
        //for(PetscInt i=0; i<A12lso.size(); ++i){
            //std::cout<<"norm = "<<std::sqrt(SPI::integrate(SPI::abs(A12lso[i])^2,grid))<<std::endl;
        //}
        test_if_close(std::sqrt(SPI::integrate(SPI::abs(A12lso[0])^2,grid)),1.0,"norm SPIgrid2D after projection 1",1e-12);
        test_if_close(std::sqrt(SPI::integrate(SPI::abs(A12lso[1])^2,grid)),1.0,"norm SPIgrid2D after orthogonalize 1",1e-12);
            



        /*
        SPI::SPIVec A1(grid.That*(1.0-(grid.y^2)),"A1"),A2(grid.That*(1.0+(grid.y^2)),"A2");
        std::vector<SPI::SPIVec> A={A1,A2};
        std::vector<SPI::SPIVec> Aorth(SPI::orthogonalize(A,grid));
        test_if_close(Aorth[0](2,PETSC_TRUE),-0.484122918275927,"orthogonalize 1",1e-12);
        test_if_close(Aorth[1](2,PETSC_TRUE),1.082531754730548,"orthogonalize 2",1e-12);

        // now try multiple length projection and integration
        SPI::SPIVec A1l(SPI::block({
                    {SPI::diag(A1),grid.O},
                    {grid.O,SPI::diag(A2)}
                    })()*SPI::ones(2*n),"A1l");
        SPI::SPIVec A2l(SPI::block({
                    {SPI::diag(A2),grid.O},
                    {grid.O,SPI::diag(A1)}
                    })()*SPI::ones(2*n),"A2l");
        std::vector<SPI::SPIVec> Al = {A1l,A2l};
        std::vector<SPI::SPIVec> Alorth(SPI::orthogonalize(Al,grid));
        test_if_close(Alorth[0](2,PETSC_TRUE),-0.228217732293819,"orthogonalize 3",1e-12);
        test_if_close(Alorth[0](10,PETSC_TRUE),0.228217732293819,"orthogonalize 4",1e-12);
        test_if_close(Alorth[1](0,PETSC_TRUE),0.714434508311760,"orthogonalize 5",1e-12);
        test_if_close(Alorth[1](8,PETSC_TRUE),-0.306186217847897,"orthogonalize 6",1e-12);
        SPI::SPIVec A1ll(SPI::block({
                    {SPI::diag(A1),grid.O,grid.O,grid.O},
                    {grid.O,SPI::diag(A2),grid.O,grid.O},
                    {grid.O,grid.O,SPI::diag(A1),grid.O},
                    {grid.O,grid.O,grid.O,SPI::diag(A2)},
                    })()*SPI::ones(4*n),"A1ll");
        SPI::SPIVec A2ll(SPI::block({
                    {SPI::diag(A2),grid.O,grid.O,grid.O},
                    {grid.O,SPI::diag(A1),grid.O,grid.O},
                    {grid.O,grid.O,SPI::diag(A2),grid.O},
                    {grid.O,grid.O,grid.O,SPI::diag(A1)},
                    })()*SPI::ones(4*n),"A2ll");
        std::vector<SPI::SPIVec> All = {A1ll,A2ll};
        std::vector<SPI::SPIVec> Allorth(SPI::orthogonalize(All,grid));
        test_if_close(Allorth[0](2,PETSC_TRUE),-0.161374306091976,"orthogonalize 7",1e-12);
        test_if_close(Allorth[0](8,PETSC_TRUE),0.484122918275927,"orthogonalize 8",1e-12);
        test_if_close(Allorth[0](10,PETSC_TRUE),0.161374306091976,"orthogonalize 9",1e-12);
        test_if_close(Allorth[0](18,PETSC_TRUE),-0.161374306091976,"orthogonalize 10",1e-12);
        test_if_close(Allorth[0](24,PETSC_TRUE),0.484122918275927,"orthogonalize 11",1e-12);
        test_if_close(Allorth[0](26,PETSC_TRUE),0.161374306091976,"orthogonalize 12",1e-12);
        test_if_close(Allorth[1](2,PETSC_TRUE),0.360843918243516,"orthogonalize 13",1e-12);
        test_if_close(Allorth[1](8,PETSC_TRUE),-0.216506350946110,"orthogonalize 14",1e-12);
        test_if_close(Allorth[1](10,PETSC_TRUE),-0.360843918243516,"orthogonalize 15",1e-12);
        test_if_close(Allorth[1](18,PETSC_TRUE),0.360843918243516,"orthogonalize 16",1e-12);
        test_if_close(Allorth[1](24,PETSC_TRUE),-0.216506350946110,"orthogonalize 17",1e-12);
        test_if_close(Allorth[1](26,PETSC_TRUE),-0.360843918243516,"orthogonalize 18",1e-12);
        */

        SPI::printf("------------ Gram-Schmidt Orthogonalize 2D end   -----------");
    }
    if(alltests){
        SPI::printf("------------ SPIMat.H(SPIVec) start -----------");
        SPI::SPIMat A(4,2,"A");
        A(0,0,4.); A(0,1,3.);
        A(1,0,3.+4.0*PETSC_i); A(1,1,3.);
        A(2,0,2.); A(2,1,3.);
        A(3,0,1.); A(3,1,3.);
        A();
        //A.print();
        SPI::SPIVec AHq(A.H(SPI::ones(4)));
        test_if_close(AHq(0,PETSC_TRUE),10.0,"SPIMat.H(SPIVec) 1",1e-12);
        test_if_close(PetscImaginaryPart(AHq(0,PETSC_TRUE)),-4.0,"SPIMat.H(SPIVec) 2",1e-12);
        test_if_close(AHq(1,PETSC_TRUE),12.0,"SPIMat.H(SPIVec) 3",1e-12);
        test_if_close(PetscImaginaryPart(AHq(1,PETSC_TRUE)),0.0,"SPIMat.H(SPIVec) 3",1e-12);
        SPI::printf("------------ SPIMat.H(SPIVec) end   -----------");
    }
    if(alltests){
        SPI::printf("------------ long skinny mat * short vec SPIMat*SPIVec start -----------");
        PetscInt ny=3;
        SPI::SPIMat A(ny,2,"A");
        for(PetscInt i=0; i<ny; ++i){
            A(i,0,(PetscScalar)i);
            A(i,1,(PetscScalar)(2*i));
        }
        A();
        SPI::SPIVec x(SPI::arange(2));
        SPI::SPIVec y(A*x);
        test_if_close(y(1,PETSC_TRUE),2.0,"SPIMat*SPIVec 1",1e-12);
        SPI::printf("------------ long skinny mat * short vec SPIMat*SPIVec start -----------");
    }
    if(alltests){
        SPI::printf("------------ abs(SPIMat) start -----------");
        PetscInt ny=3;
        SPI::SPIMat A(ny,2,"A");
        for(PetscInt i=0; i<ny; ++i){
            A(i,0,-(PetscScalar)i+0.0*PETSC_i);
            A(i,1,-(PetscScalar)(2*i) - 2.0*PETSC_i);
        }
        A();
        //A.print();
        //SPI::abs(A).print();
        test_if_close(SPI::abs(A)(1,1,PETSC_TRUE),sqrt(8.0),"abs(SPIMat) 1",1e-12);
        test_if_close(SPI::abs(A)(2,1,PETSC_TRUE),sqrt(20.0),"abs(SPIMat) 2",1e-12);
        SPI::printf("------------ abs(SPIMat) end   -----------");
    }
    if(alltests){
        SPI::printf("------------ inv(SPIMat) start -----------");
        SPI::SPIMat A(3,3,"A");
        A(0,0, 0.0); A(0,1,-3.0); A(0,2,-2.0);
        A(1,0, 1.0); A(1,1,-4.0); A(1,2,-2.0);
        A(2,0,-3.0); A(2,1, 4.0); A(2,2, 1.0);
        A();
        SPI::SPIVec yy(3);
        MatGetColumnVector(A.mat,yy.vec,1);
        SPI::SPIMat Ainv = inv(A);
        test_if_close(Ainv(2,1,PETSC_TRUE),9.0,"inv(SPIMat) 1",1e-12);
        SPI::printf("------------ inv(SPIMat) end   -----------");
    }
    if(alltests){
        SPI::printf("------------ UltraSpherical ops start -----------");
        //SPI::SPIVec y(SPI::set_Cheby_y(5),"yCheby");
        SPI::SPIVec y(SPI::set_Cheby_mapped_y(0,10,5),"yCheby");
        SPI::SPIMat S0(5,5,"S0"),D1(5,5,"D1");
        SPI::SPIMat S1(5,5,"S1"),D2(5,5,"D2");
        std::tie(S0,D1) = SPI::set_D_UltraS(y,1);  
        std::tie(S1,D2) = SPI::set_D_UltraS(y,2);  
        //S0.print();
        //(S1*D1).print();
        //(D2).print();
        SPI::SPIMat T(5,5,"T"),That(5,5,"That");
        std::tie(T,That) = SPI::set_T_That(y.rows);
        //T.print();
        //That.print();
        //(T*inv(S0)*D1*That).print();
        test_if_close((S1*D1)(1,2,PETSC_TRUE),0.2,"set_D_UltraS(SPIMat) 1",1e-12);
        test_if_close((S1*D1)(1,4,PETSC_TRUE),-0.2,"set_D_UltraS(SPIMat) 2",1e-12);
        test_if_close(D2(1,3,PETSC_TRUE),0.24,"set_D_UltraS(SPIMat) 3",1e-12);
        test_if_close((T*inv(S0)*D1*That)(1,2,PETSC_TRUE),0.282842712,"set_D_UltraS(SPIMat) 4",1e-7);
        test_if_close((T*inv(S0)*inv(S1)*D2*That)(1,3,PETSC_TRUE),-0.08,"set_D_UltraS(SPIMat) 4",1e-7);
        //(T*inv(S0)*inv(S1)*D2*That).print();
        //((T*inv(S0)*inv(S1)*D2*That)*y).print();
        //inv(S0).print();
        //inv(S1).print();
        //(inv(S0)*inv(S1)).print();
        test_if_close((inv(S0)*inv(S1))(2,4,PETSC_TRUE),16.0,"inv(S0)*inv(S1) 1",1e-7);
        SPI::printf("------------ UltraSpherical ops end   -----------");
    }
    if(alltests){
        SPI::printf("------------ LST_temporal channel UltraS start -----------");
        PetscInt n=64;
        SPI::SPIVec y(SPI::set_Cheby_mapped_y(-1.,1.,n) ,"yCheby");
        SPI::SPIgrid1D grid(y,"grid",SPI::UltraS);
        //grid.print();
        //exit(0);
        //SPI::SPIVec y(SPI::linspace(0.,61.,n) ,"yFD");
        //SPI::printf("n = %d",n);
        //grid.print();
        //(grid.Dy*grid.y).print();

        SPI::SPIparams params("Channel parameters");
        params.Re = 2000.0;
        params.omega = 0.3121-0.01978*PETSC_i;
        params.nu = 1./params.Re;
        params.x = 1.0;
        params.x_start = params.x;
        params.alpha = 1.0;
        params.beta = 0.0;

        SPI::SPIVec eigenfunction(grid.y.rows*4,"q");
        //SPI::SPIVec eig_vec(grid.y.rows*8,"q");
        PetscScalar eigenvalue;
        eigenfunction.name = "eigenfunction";

        //SPI::SPIbaseflow bl_flow = SPI::blasius(params,grid);
        SPI::SPIVec U((1.0-((grid.y)^2)),"U");
        SPI::SPIVec Uy((-2.*grid.y),"Uy");
        SPI::SPIVec o(U*0.0,"o"); // zero vector
        SPI::SPIbaseflow channel(U,o,o,o,Uy,o,o,o,o,o,o,o,o,o);
        if(0){ // set to parallel baseflow
            //SPI::SPIVec o(SPI::zeros(n),"zero");
            //bl_flow.Ux = o;
            //bl_flow.Uxy = o;
            //bl_flow.V = o;
            //bl_flow.Vy = o;
        }
        //bl_flow.print();

        PetscScalar cg;
        std::tie(eigenvalue,eigenfunction) = SPI::LST_temporal(params,grid,channel);
        SPI::printfc("eigenvalue = %g+%gi",eigenvalue);
        //std::tie(eigenvalue,eig_vec) = SPI::LST_spatial(params,grid,channel);
        test_if_close(eigenvalue,0.3121-0.01978*PETSC_i,"LST_temporal UltraS 1",1e-5);
        test_if_close(params.omega,0.3121+0.01978*PETSC_i,"LST_temporal UltraS 2",1e-5);
        //std::tie(eigenvalue,eigenfunction) = SPI::LSTNP_spatial_right(params,grid,channel);
        //test_if_close(params.alpha,0.094966+0.004564*PETSC_i,"LSTNP_spatial_right 1",1e-5);
        //SPI::printfc("eigenvalue is %.10f + %.10fi",eigenvalue);

        //std::tie(eigenvalue,cg,leigenfunction,eigenfunction) = SPI::LSTNP_spatial(params,grid,bl_flow,leigenfunction,eigenfunction); // with initial guess
        //SPI::printfc("cg = %g+%gi",cg);
        //test_if_close(eigenvalue,0.094966+0.004564*PETSC_i,"LSTNP_spatial 2",1e-5);
        //std::tie(eigenvalue,eigenfunction) = SPI::LSTNP_spatial_right(params,grid,bl_flow,eigenfunction); // with initial guess
        //test_if_close(eigenvalue,0.094966+0.004564*PETSC_i,"LSTNP_spatial_right 2",1e-5);
        //SPI::printfc("eigenvalue is %.10f + %.10fi",eigenvalue);

        //params.alpha = 0.1067+0.001898*PETSC_i;
        //std::tie(eigenvalue,cg,leigenfunction,eigenfunction) = SPI::LSTNP_spatial(params,grid,bl_flow); 
        //SPI::printfc("cg = %g+%gi",cg);
        //test_if_close(eigenvalue,0.10665+0.0018979*PETSC_i,"LSTNP_spatial 3",1e-5);
        //std::tie(eigenvalue,eigenfunction) = SPI::LSTNP_spatial_right(params,grid,bl_flow); 
        //test_if_close(eigenvalue,0.10665+0.0018979*PETSC_i,"LSTNP_spatial_right 3",1e-5);
        //SPI::printfc("eigenvalue is %.10f + %.10fi",eigenvalue);
        //std::tie(eigenvalue,cg,leigenfunction,eigenfunction) = SPI::LSTNP_spatial(params,grid,bl_flow,leigenfunction,eigenfunction); // with initial guess
        //SPI::printfc("cg = %g+%gi",cg);
        //test_if_close(eigenvalue,0.10665+0.0018979*PETSC_i,"LSTNP_spatial 4",1e-5);
        //SPI::printfc("eigenvalue is %.10f + %.10fi",eigenvalue);
        //std::tie(eigenvalue,eigenfunction) = SPI::LSTNP_spatial_right(params,grid,bl_flow,eigenfunction); // with initial guess
        //test_if_close(eigenvalue,0.10665+0.0018979*PETSC_i,"LSTNP_spatial_right 4",1e-5);

        SPI::printf("------------ LST_temporal channel UltraS end   -----------");
    }
    if(alltests){
        SPI::printf("------------ LST_spatial channel flow UltraS start -----------");
        PetscInt n=64;
        //SPI::SPIVec y(SPI::set_Cheby_mapped_y(0.,21.,n) ,"yCheby");
        SPI::SPIVec y(SPI::set_Cheby_mapped_y(-1.,1.,n) ,"yCheby");
        SPI::SPIgrid1D grid(y,"grid",SPI::UltraS);
        //grid.print();
        //exit(0);
        //SPI::SPIVec y(SPI::linspace(0.,61.,n) ,"yFD");
        //SPI::printf("n = %d",n);
        //grid.print();
        //(grid.Dy*grid.y).print();

        SPI::SPIparams params("Blasius parameters");
        //params.Re = 400.0;
        //params.omega = 86.*params.Re/(1000000.);
        //params.Re = 1000.0/1.7208;
        //params.omega = 0.26/1.7208;
        params.Re = 2000.0;
        params.omega = 0.3;
        params.nu = 1./params.Re;
        params.x = params.Re;
        params.x_start = params.x;
        //params.alpha = (0.74155+0.345132*PETSC_i)/1.7208;
        params.alpha = 0.97875+0.044394*PETSC_i;
        params.beta = 0.0;

        SPI::SPIVec eigenfunction(grid.y.rows*8,"q");
        //SPI::SPIVec eig_vec(grid.y.rows*8,"q");
        SPI::SPIVec leigenfunction(grid.y.rows*8,"q");
        PetscScalar eigenvalue;
        eigenfunction.name = "eigenfunction";

        //SPI::SPIbaseflow bl_flow = SPI::blasius(params,grid);
        SPI::SPIbaseflow bl_flow = SPI::channel(params,grid);
        if(0){ // set to parallel baseflow
            SPI::SPIVec o(SPI::zeros(n),"zero");
            bl_flow.Ux = o;
            bl_flow.Uxy = o;
            bl_flow.V = o;
            bl_flow.Vy = o;
        }

        std::tie(eigenvalue,eigenfunction) = SPI::LST_spatial(params,grid,bl_flow);
        test_if_close(params.alpha,(0.97875+0.044394*PETSC_i),"LST_spatial 1",1e-5);

        params.alpha = 0.34312+0.049677*PETSC_i;
        std::tie(eigenvalue,eigenfunction) = SPI::LST_spatial(params,grid,bl_flow);
        test_if_close(params.alpha,(0.34312+0.049677*PETSC_i),"LST_spatial 2",1e-5);

        params.alpha = 0.61167+0.140492*PETSC_i;
        std::tie(eigenvalue,eigenfunction) = SPI::LST_spatial(params,grid,bl_flow);
        test_if_close(params.alpha,(0.61167+0.140492*PETSC_i),"LST_spatial 3",1e-5);

        SPI::printf("------------ LST_spatial channel flow UltraS end   -----------");
    }
    if(alltests){ // requires -mat_mumps_icntl_14 25 in command line call to work
        SPI::printf("------------ LST_spatial Blasius boundary layer UltraS start -----------");
        PetscInt n=200;
        SPI::SPIVec y(SPI::set_Cheby_mapped_y(0.,21.,n) ,"yCheby");
        //SPI::SPIVec y(SPI::set_Cheby_mapped_y(-1.,1.,n) ,"yCheby");
        SPI::SPIgrid1D grid(y,"grid",SPI::UltraS);
        //grid.print();
        //exit(0);
        //SPI::SPIVec y(SPI::linspace(0.,61.,n) ,"yFD");
        //SPI::printf("n = %d",n);
        //grid.print();
        //(grid.Dy*grid.y).print();

        SPI::SPIparams params("Blasius parameters");
        //params.Re = 400.0;
        //params.omega = 86.*params.Re/(1000000.);
        params.Re = 1000.0/1.7208;
        params.omega = 0.26/1.7208;
        params.nu = 1./params.Re;
        params.x = params.Re;
        params.x_start = params.x;
        params.alpha = (0.74155+0.345132*PETSC_i)/1.7208;
        params.beta = 0.0;
        //params.print();

        SPI::SPIVec eigenfunction(grid.y.rows*8,"q");
        //SPI::SPIVec eig_vec(grid.y.rows*8,"q");
        SPI::SPIVec leigenfunction(grid.y.rows*8,"q");
        PetscScalar eigenvalue;
        eigenfunction.name = "eigenfunction";

        SPI::SPIbaseflow bl_flow = SPI::blasius(params,grid);
        //SPI::SPIbaseflow bl_flow = SPI::channel(params,grid);
        if(1){ // set to parallel baseflow
            SPI::SPIVec o(SPI::zeros(n),"zero");
            bl_flow.Ux = o;
            bl_flow.Uxy = o;
            bl_flow.V = o;
            bl_flow.Vy = o;
        }
        //bl_flow.print();

        //std::tie(eigenvalue,cg,leigenfunction,eigenfunction) = SPI::LSTNP_spatial(params,grid,bl_flow);
        std::tie(eigenvalue,eigenfunction) = SPI::LST_spatial(params,grid,bl_flow);
        test_if_close(params.alpha*1.7208,(0.74155+0.345132*PETSC_i),"LST_spatial 1",1e-5);

        params.alpha = (0.54213+0.083968*PETSC_i)/1.7208;
        std::tie(eigenvalue,eigenfunction) = SPI::LST_spatial(params,grid,bl_flow);
        test_if_close(params.alpha*1.7208,(0.54213+0.083968*PETSC_i),"LST_spatial 2",1e-5);

        params.alpha = (0.29967+0.230773*PETSC_i)/1.7208;
        std::tie(eigenvalue,eigenfunction) = SPI::LST_spatial(params,grid,bl_flow);
        test_if_close(params.alpha*1.7208,(0.29967+0.230773*PETSC_i),"LST_spatial 3",1e-5);

        SPI::printf("------------ LST_spatial Blasius boundary layer UltraS end   -----------");
    }
    if(0){
        // noticed block and eye commands were the slowest part of the LSTNP_spatial_right eigenvalue solver, so let's speed those up here
        SPI::printf("------------ eye and block timings start -----------");
        PetscInt n=900000;
        SPI::SPIMat I(SPI::eye(n*8),"I");
        SPI::SPIMat O(SPI::zeros(n*8,n*8),"O");
        std::cout<<"created I and O"<<std::endl;
        SPI::block({
                {I,O,O},
                {I,-I,I},
                {I,I,O}
                })();
        std::cout<<"created block"<<std::endl;
        SPI::block({
                {I,O,O},
                {O,O,O},
                {O,O,O}
                })();
        std::cout<<"created block P"<<std::endl;
        SPI::SPIMat B(3,3);
        B(0,0,1.0)();
        B(0,1,2.0)();
        SPI::kron(B,I);
        std::cout<<"created block one kron"<<std::endl;
        SPI::SPIMat B1(3,3);
        B1(0,0,1.0)();
        SPI::SPIMat B2(3,3);
        B2(0,1,2.0)();
        SPI::kron(B1,I) + SPI::kron(B2,I);
        std::cout<<"created block using 2 kron"<<std::endl;

        SPI::printf("------------ eye and block timings end   -----------");
    }
    if(alltests){
        SPI::printf("------------ LSTNP_spatial_right Blasius boundary layer UltraS start -----------");
        PetscInt n=169;
        SPI::SPIVec y(SPI::set_Cheby_mapped_y(0.,21.,n) ,"yCheby");
        //SPI::SPIVec y(SPI::set_Cheby_mapped_y(-1.,1.,n) ,"yCheby");
        SPI::SPIgrid1D grid(y,"grid",SPI::UltraS);
        //grid.print();
        //exit(0);
        //SPI::SPIVec y(SPI::linspace(0.,61.,n) ,"yFD");
        //SPI::printf("n = %d",n);
        //grid.print();
        //(grid.Dy*grid.y).print();

        SPI::SPIparams params("Blasius parameters");
        //params.Re = 400.0;
        //params.omega = 86.*params.Re/(1000000.);
        params.Re = 1000.0/1.7208;
        params.omega = 0.26/1.7208;
        params.nu = 1./params.Re;
        params.x = params.Re;
        params.x_start = params.x;
        params.alpha = (0.74155+0.345132*PETSC_i)/1.7208;
        params.beta = 0.0;
        //params.print();

        SPI::SPIVec eigenfunction(grid.y.rows*8,"q");
        //SPI::SPIVec eig_vec(grid.y.rows*8,"q");
        //SPI::SPIVec leigenfunction(grid.y.rows*16,"q");
        PetscScalar eigenvalue;
        eigenfunction.name = "eigenfunction";

        SPI::SPIbaseflow bl_flow = SPI::blasius(params,grid);
        //SPI::SPIbaseflow bl_flow = SPI::channel(params,grid);
        if(1){ // set to parallel baseflow
            SPI::SPIVec o(SPI::zeros(n),"zero");
            bl_flow.Ux = o;
            bl_flow.Uxy = o;
            bl_flow.V = o;
            bl_flow.Vy = o;
        }
        //bl_flow.print();

        std::tie(eigenvalue,eigenfunction) = SPI::LSTNP_spatial_right(params,grid,bl_flow);
        //std::tie(eigenvalue,eigenfunction) = SPI::LST_spatial(params,grid,bl_flow);
        test_if_close(params.alpha*1.7208,(0.74155+0.345132*PETSC_i),"LSTNP_spatial_right 1",1e-5);

        params.alpha = (0.54213+0.083968*PETSC_i)/1.7208;
        std::tie(eigenvalue,eigenfunction) = SPI::LSTNP_spatial_right(params,grid,bl_flow);
        test_if_close(params.alpha*1.7208,(0.54213+0.083968*PETSC_i),"LSTNP_spatial_right 2",1e-5);

        params.alpha = (0.29967+0.230773*PETSC_i)/1.7208;
        std::tie(eigenvalue,eigenfunction) = SPI::LSTNP_spatial_right(params,grid,bl_flow);
        test_if_close(params.alpha*1.7208,(0.29967+0.230773*PETSC_i),"LSTNP_spatial_right 3",1e-5);

        SPI::printf("------------ LSTNP_spatial_right Blasius boundary layer UltraS end   -----------");
    }
    if(alltests){
        SPI::printf("------------ LST_spatial_cg Blasius boundary layer physical vs UltraS start -----------");
        PetscInt n=169;
        SPI::SPIVec y(SPI::set_Cheby_mapped_y(0.,21.,n) ,"yCheby");
        //SPI::SPIVec y(SPI::set_Cheby_mapped_y(-1.,1.,n) ,"yCheby");
        SPI::SPIgrid1D grid(y,"grid",SPI::UltraS);
        SPI::SPIgrid1D grid2(y,"grid",SPI::Chebyshev);
        //grid.print();
        //exit(0);
        //SPI::SPIVec y(SPI::linspace(0.,61.,n) ,"yFD");
        //SPI::printf("n = %d",n);
        //grid.print();
        //(grid.Dy*grid.y).print();

        SPI::SPIparams params("Blasius parameters");
        //params.Re = 400.0;
        //params.omega = 86.*params.Re/(1000000.);
        params.Re = 1000.0/1.7208;
        params.omega = 0.26/1.7208;
        params.nu = 1./params.Re;
        params.x = params.Re;
        params.x_start = params.x;
        params.alpha = (0.74155+0.345132*PETSC_i)/1.7208;
        params.beta = 0.0;
        //params.print();

        SPI::SPIVec eigenfunction(grid.y.rows*8,"eigenfunction");
        SPI::SPIVec leigenfunction(grid.y.rows*8,"leigenfunction");
        //SPI::SPIVec eig_vec(grid.y.rows*8,"q");
        //SPI::SPIVec leigenfunction(grid.y.rows*16,"q");
        PetscScalar eigenvalue;
        eigenfunction.name = "eigenfunction";

        SPI::SPIbaseflow bl_flow = SPI::blasius(params,grid);
        //SPI::SPIbaseflow bl_flow = SPI::channel(params,grid);
        if(0){ // set to parallel baseflow
            SPI::SPIVec o(SPI::zeros(n),"zero");
            bl_flow.Ux = o;
            bl_flow.Uxy = o;
            bl_flow.V = o;
            bl_flow.Vy = o;
        }
        //bl_flow.print();

        // LST_spatial_cg UltraS
        PetscScalar cg;
        std::tie(eigenvalue,cg,leigenfunction,eigenfunction) = SPI::LST_spatial_cg(params,grid,bl_flow);
        SPI::printfc("cg = %.10f + %.10fi",cg);
        //std::tie(eigenvalue,eigenfunction) = SPI::LST_spatial(params,grid,bl_flow);
        test_if_close(params.alpha*1.7208,(0.74155+0.345132*PETSC_i),"LST_spatial_cg UltraS 1",1e-5);
        SPI::SPIMat &O = grid.O;
        SPI::SPIMat T(SPI::block({
                    {grid.T,O,O,O,O,O,O,O},
                    {O,grid.T,O,O,O,O,O,O},
                    {O,O,grid.T,O,O,O,O,O},
                    {O,O,O,grid.T,O,O,O,O},
                    {O,O,O,O,grid.T,O,O,O},
                    {O,O,O,O,O,grid.T,O,O},
                    {O,O,O,O,O,O,grid.T,O},
                    {O,O,O,O,O,O,O,grid.T},
                    })());
        SPI::SPIVec eigenfunctionout(T*eigenfunction,"UltraS");
        SPI::SPIVec leigenfunctionout(T*leigenfunction,"UltraSl");
        // normalize
        eigenfunctionout /= SPI::L2(eigenfunctionout);
        leigenfunctionout /= SPI::L2(leigenfunctionout);
        SPI::save(eigenfunctionout,"eigenfunction.h5");
        SPI::save(leigenfunctionout,"eigenfunction.h5");
        // LST_spatial physical
        SPI::SPIVec leigenfunction8(8*n,"Physicall");
        SPI::SPIVec eigenfunction8(8*n,"Physical");
        std::tie(eigenvalue,cg,leigenfunction8,eigenfunction8) = SPI::LST_spatial_cg(params,grid2,bl_flow);
        SPI::printfc("cg = %.10f + %.10fi",cg);
        test_if_close(params.alpha*1.7208,(0.74155+0.345132*PETSC_i),"LST_spatial_cg physical 1",1e-5);
        SPI::save(eigenfunction8,"eigenfunction.h5");
        SPI::save(leigenfunction8,"eigenfunction.h5");
        SPI::save(grid.y,"eigenfunction.h5");
        // finite difference what is the group velocity of this mode?
        PetscScalar deltaOmega = 1e-2;
        params.omega += deltaOmega;
        PetscScalar deltaeigenvalue;
        std::tie(deltaeigenvalue,cg,leigenfunction8,eigenfunction8) = SPI::LST_spatial_cg(params,grid2,bl_flow);
        SPI::printfc("with delta_omega cg = %.10f + %.10fi",cg);
        cg = deltaOmega/(deltaeigenvalue-eigenvalue);
        SPI::printfc("with finite diff delta_omega cg = %.10f + %.10fi",cg);
        params.omega -= 2.0*deltaOmega;
        PetscScalar deltaeigenvaluem;
        std::tie(deltaeigenvaluem,cg,leigenfunction8,eigenfunction8) = SPI::LST_spatial_cg(params,grid,bl_flow);
        cg = (2.0*deltaOmega)/(deltaeigenvalue-deltaeigenvaluem);
        SPI::printfc("with finite diff delta_omega2 cg = %.10f + %.10fi",cg);
        // LSTNP_spatial physical
        //std::tie(eigenvalue,cg,leigenfunction,eigenfunction) = SPI::LSTNP_spatial(params,grid2,bl_flow);
        //SPI::printfc("cg = %.10f + %.10fi",cg);
        //test_if_close(params.alpha*1.7208,(0.74155+0.345132*PETSC_i),"LSTNP_spatial physical 1",1e-4);

        // these following 2 cases don't work
        //params.alpha = (0.54213+0.083968*PETSC_i)/1.7208;
        //std::tie(eigenvalue,cg,leigenfunction,eigenfunction) = SPI::LSTNP_spatial(params,grid,bl_flow);
        //SPI::printfc("cg = %.10f + %.10fi",cg);
        //test_if_close(params.alpha*1.7208,(0.54213+0.083968*PETSC_i),"LSTNP_spatial UltraS 2",1e-5);
        //std::tie(eigenvalue,cg,leigenfunction,eigenfunction) = SPI::LST_spatial_cg(params,grid2,bl_flow);
        //SPI::printfc("cg = %.10f + %.10fi",cg);
        //test_if_close(params.alpha*1.7208,(0.54213+0.083968*PETSC_i),"LST_spatial_cg 2",1e-5);

        //params.alpha = (0.29967+0.230773*PETSC_i)/1.7208;
        //std::tie(eigenvalue,cg,leigenfunction,eigenfunction) = SPI::LSTNP_spatial(params,grid,bl_flow);
        //SPI::printfc("cg = %.10f + %.10fi",cg);
        //test_if_close(params.alpha*1.7208,(0.29967+0.230773*PETSC_i),"LSTNP_spatial UltraS 3",1e-5);
        //std::tie(eigenvalue,cg,leigenfunction,eigenfunction) = SPI::LST_spatial_cg(params,grid,bl_flow);
        //SPI::printfc("cg = %.10f + %.10fi",cg);
        //test_if_close(params.alpha*1.7208,(0.29967+0.230773*PETSC_i),"LST_spatial_cg 3",1e-5);

        SPI::printf("------------ LST_spatial_cg Blasius boundary layer UltraS end   -----------");
    }
    if(alltests){
        SPI::printf("------------ LSTNP_spatial Blasius boundary layer UltraS start -----------");
        PetscInt n=169;
        SPI::SPIVec y(SPI::set_Cheby_mapped_y(0.,21.,n) ,"yCheby");
        //SPI::SPIVec y(SPI::set_Cheby_mapped_y(-1.,1.,n) ,"yCheby");
        SPI::SPIgrid1D grid(y,"grid",SPI::UltraS);
        SPI::SPIgrid1D grid2(y,"grid",SPI::Chebyshev);
        //grid.print();
        //exit(0);
        //SPI::SPIVec y(SPI::linspace(0.,61.,n) ,"yFD");
        //SPI::printf("n = %d",n);
        //grid.print();
        //(grid.Dy*grid.y).print();

        SPI::SPIparams params("Blasius parameters");
        //params.Re = 400.0;
        //params.omega = 86.*params.Re/(1000000.);
        params.Re = 1000.0/1.7208;
        params.omega = 0.26/1.7208;
        params.nu = 1./params.Re;
        params.x = params.Re;
        params.x_start = params.x;
        params.alpha = (0.74155+0.345132*PETSC_i)/1.7208;
        params.beta = 0.0;
        //params.print();

        SPI::SPIVec eigenfunction(grid.y.rows*16,"eigenfunction");
        SPI::SPIVec leigenfunction(grid.y.rows*16,"leigenfunction");
        //SPI::SPIVec eig_vec(grid.y.rows*8,"q");
        //SPI::SPIVec leigenfunction(grid.y.rows*16,"q");
        PetscScalar eigenvalue;
        eigenfunction.name = "eigenfunction";

        SPI::SPIbaseflow bl_flow = SPI::blasius(params,grid);
        //SPI::SPIbaseflow bl_flow = SPI::channel(params,grid);
        if(1){ // set to parallel baseflow
            SPI::SPIVec o(SPI::zeros(n),"zero");
            bl_flow.Ux = o;
            bl_flow.Uxy = o;
            bl_flow.V = o;
            bl_flow.Vy = o;
        }
        //bl_flow.print();

        PetscScalar cg;
        std::tie(eigenvalue,cg,leigenfunction,eigenfunction) = SPI::LSTNP_spatial(params,grid,bl_flow);
        SPI::printfc("cg = %.10f + %.10fi",cg);
        //std::tie(eigenvalue,eigenfunction) = SPI::LST_spatial(params,grid,bl_flow);
        test_if_close(params.alpha*1.7208,(0.74155+0.345132*PETSC_i),"LSTNP_spatial UltraS 1",1e-4);
        SPI::SPIMat &O = grid.O;
        SPI::SPIMat &t = grid.T;
        SPI::SPIMat T(SPI::block({
                    {t,O,O,O,O,O,O,O,O,O,O,O,O,O,O,O},
                    {O,t,O,O,O,O,O,O,O,O,O,O,O,O,O,O},
                    {O,O,t,O,O,O,O,O,O,O,O,O,O,O,O,O},
                    {O,O,O,t,O,O,O,O,O,O,O,O,O,O,O,O},
                    {O,O,O,O,t,O,O,O,O,O,O,O,O,O,O,O},
                    {O,O,O,O,O,t,O,O,O,O,O,O,O,O,O,O},
                    {O,O,O,O,O,O,t,O,O,O,O,O,O,O,O,O},
                    {O,O,O,O,O,O,O,t,O,O,O,O,O,O,O,O},
                    {O,O,O,O,O,O,O,O,t,O,O,O,O,O,O,O},
                    {O,O,O,O,O,O,O,O,O,t,O,O,O,O,O,O},
                    {O,O,O,O,O,O,O,O,O,O,t,O,O,O,O,O},
                    {O,O,O,O,O,O,O,O,O,O,O,t,O,O,O,O},
                    {O,O,O,O,O,O,O,O,O,O,O,O,t,O,O,O},
                    {O,O,O,O,O,O,O,O,O,O,O,O,O,t,O,O},
                    {O,O,O,O,O,O,O,O,O,O,O,O,O,O,t,O},
                    {O,O,O,O,O,O,O,O,O,O,O,O,O,O,O,t},
                    })());
        SPI::SPIVec eigenfunctionout(T*eigenfunction,"UltraS");
        SPI::SPIVec leigenfunctionout(T*leigenfunction,"UltraSl");
        //SPI::save(eigenfunctionout,"eigenfunctionNP.h5");
        //SPI::save(leigenfunctionout,"eigenfunctionNP.h5");
        //SPI::save(grid.y,"eigenfunctionNP.h5");
        // LST_spatial physical
        SPI::SPIVec leigenfunction16(16*n);
        SPI::SPIVec eigenfunction16(16*n);
        std::tie(eigenvalue,cg,leigenfunction16,eigenfunction16) = SPI::LSTNP_spatial(params,grid2,bl_flow);
        SPI::printfc("cg = %.10f + %.10fi",cg);
        test_if_close(params.alpha*1.7208,(0.74155+0.345132*PETSC_i),"LSTNP_spatial physical 1",1e-4);
        // LSTNP_spatial physical
        std::tie(eigenvalue,cg,leigenfunction,eigenfunction) = SPI::LSTNP_spatial(params,grid2,bl_flow);
        SPI::printfc("cg = %.10f + %.10fi",cg);
        test_if_close(params.alpha*1.7208,(0.74155+0.345132*PETSC_i),"LSTNP_spatial physical 1",1e-4);

        params.alpha = (0.54213+0.083968*PETSC_i)/1.7208;
        std::tie(eigenvalue,cg,leigenfunction,eigenfunction) = SPI::LSTNP_spatial(params,grid,bl_flow);
        SPI::printfc("cg = %.10f + %.10fi",cg);
        test_if_close(params.alpha*1.7208,(0.54213+0.083968*PETSC_i),"LSTNP_spatial UltraS 2",1e-5);

        params.alpha = (0.29967+0.230773*PETSC_i)/1.7208;
        std::tie(eigenvalue,cg,leigenfunction,eigenfunction) = SPI::LSTNP_spatial(params,grid,bl_flow);
        SPI::printfc("cg = %.10f + %.10fi",cg);
        test_if_close(params.alpha*1.7208,(0.29967+0.230773*PETSC_i),"LSTNP_spatial UltraS 3",1e-5);

        SPI::printf("------------ LSTNP_spatial Blasius boundary layer UltraS end   -----------");
    }
    if(alltests){
        SPI::printf("------------ LSTNP_spatial_right Blasius boundary layer UltraS start -----------");
        PetscInt n=169;
        SPI::SPIVec y(SPI::set_Cheby_mapped_y(0.,21.,n) ,"yCheby");
        //SPI::SPIVec y(SPI::set_Cheby_mapped_y(-1.,1.,n) ,"yCheby");
        SPI::SPIgrid1D grid(y,"grid",SPI::UltraS);
        SPI::SPIgrid1D grid2(y,"grid",SPI::Chebyshev);
        //grid.print();
        //exit(0);
        //SPI::SPIVec y(SPI::linspace(0.,61.,n) ,"yFD");
        //SPI::printf("n = %d",n);
        //grid.print();
        //(grid.Dy*grid.y).print();

        SPI::SPIparams params("Blasius parameters");
        //params.Re = 400.0;
        //params.omega = 86.*params.Re/(1000000.);
        params.Re = 1000.0/1.7208;
        params.omega = 0.26/1.7208;
        params.nu = 1./params.Re;
        params.x = params.Re;
        params.x_start = params.x;
        params.alpha = (0.74155+0.345132*PETSC_i)/1.7208;
        params.beta = 0.0;
        //params.print();

        SPI::SPIVec eigenfunction(grid.y.rows*16,"eigenfunction");
        //SPI::SPIVec eig_vec(grid.y.rows*8,"q");
        //SPI::SPIVec leigenfunction(grid.y.rows*16,"q");
        PetscScalar eigenvalue;
        eigenfunction.name = "eigenfunction";

        SPI::SPIbaseflow bl_flow = SPI::blasius(params,grid);
        //SPI::SPIbaseflow bl_flow = SPI::channel(params,grid);
        if(1){ // set to parallel baseflow
            SPI::SPIVec o(SPI::zeros(n),"zero");
            bl_flow.Ux = o;
            bl_flow.Uxy = o;
            bl_flow.V = o;
            bl_flow.Vy = o;
        }
        //bl_flow.print();

        std::tie(eigenvalue,eigenfunction) = SPI::LSTNP_spatial_right(params,grid,bl_flow);
        //std::tie(eigenvalue,eigenfunction) = SPI::LST_spatial(params,grid,bl_flow);
        test_if_close(params.alpha*1.7208,(0.74155+0.345132*PETSC_i),"LSTNP_spatial_right UltraS 1",1e-4);
        SPI::SPIMat &O = grid.O;
        SPI::SPIMat &t = grid.T;
        SPI::SPIMat T(SPI::block({
                    {t,O,O,O,O,O,O,O,O,O,O,O,O,O,O,O},
                    {O,t,O,O,O,O,O,O,O,O,O,O,O,O,O,O},
                    {O,O,t,O,O,O,O,O,O,O,O,O,O,O,O,O},
                    {O,O,O,t,O,O,O,O,O,O,O,O,O,O,O,O},
                    {O,O,O,O,t,O,O,O,O,O,O,O,O,O,O,O},
                    {O,O,O,O,O,t,O,O,O,O,O,O,O,O,O,O},
                    {O,O,O,O,O,O,t,O,O,O,O,O,O,O,O,O},
                    {O,O,O,O,O,O,O,t,O,O,O,O,O,O,O,O},
                    {O,O,O,O,O,O,O,O,t,O,O,O,O,O,O,O},
                    {O,O,O,O,O,O,O,O,O,t,O,O,O,O,O,O},
                    {O,O,O,O,O,O,O,O,O,O,t,O,O,O,O,O},
                    {O,O,O,O,O,O,O,O,O,O,O,t,O,O,O,O},
                    {O,O,O,O,O,O,O,O,O,O,O,O,t,O,O,O},
                    {O,O,O,O,O,O,O,O,O,O,O,O,O,t,O,O},
                    {O,O,O,O,O,O,O,O,O,O,O,O,O,O,t,O},
                    {O,O,O,O,O,O,O,O,O,O,O,O,O,O,O,t},
                    })());
        SPI::SPIVec eigenfunctionout(T*eigenfunction,"UltraS");
        //SPI::save(eigenfunctionout,"eigenfunctionNP.h5");
        //SPI::save(grid.y,"eigenfunctionNP.h5");
        // LST_spatial physical
        SPI::SPIVec leigenfunction16(16*n);
        SPI::SPIVec eigenfunction16(16*n);
        std::tie(eigenvalue,eigenfunction16) = SPI::LSTNP_spatial_right(params,grid2,bl_flow);
        test_if_close(params.alpha*1.7208,(0.74155+0.345132*PETSC_i),"LSTNP_spatial_right physical 1",1e-4);
        // LSTNP_spatial physical
        std::tie(eigenvalue,eigenfunction) = SPI::LSTNP_spatial_right(params,grid2,bl_flow);
        test_if_close(params.alpha*1.7208,(0.74155+0.345132*PETSC_i),"LSTNP_spatial_right physical 1",1e-4);

        params.alpha = (0.54213+0.083968*PETSC_i)/1.7208;
        std::tie(eigenvalue,eigenfunction) = SPI::LSTNP_spatial_right(params,grid,bl_flow);
        test_if_close(params.alpha*1.7208,(0.54213+0.083968*PETSC_i),"LSTNP_spatial_right UltraS 2",1e-5);

        params.alpha = (0.29967+0.230773*PETSC_i)/1.7208;
        std::tie(eigenvalue,eigenfunction) = SPI::LSTNP_spatial_right(params,grid,bl_flow);
        test_if_close(params.alpha*1.7208,(0.29967+0.230773*PETSC_i),"LSTNP_spatial_right UltraS 3",1e-5);

        SPI::printf("------------ LSTNP_spatial_right Blasius boundary layer UltraS end   -----------");
    }
    if(alltests){
        SPI::printf("------------ LSTNP_spatials_right Blasius boundary layer UltraS start -----------");
        PetscInt n=169;
        SPI::SPIVec y(SPI::set_Cheby_mapped_y(0.,21.,n) ,"yCheby");
        //SPI::SPIVec y(SPI::set_Cheby_mapped_y(-1.,1.,n) ,"yCheby");
        SPI::SPIgrid1D grid(y,"grid",SPI::UltraS);
        //SPI::SPIgrid1D grid2(y,"grid",SPI::Chebyshev);
        //grid.print();
        //exit(0);
        //SPI::SPIVec y(SPI::linspace(0.,61.,n) ,"yFD");
        //SPI::printf("n = %d",n);
        //grid.print();
        //(grid.Dy*grid.y).print();

        SPI::SPIparams params("Blasius parameters");
        //params.Re = 400.0;
        //params.omega = 86.*params.Re/(1000000.);
        params.Re = 1000.0/1.7208;
        params.omega = 0.26/1.7208;
        params.nu = 1./params.Re;
        params.x = params.Re;
        params.x_start = params.x;
        params.alpha = (0.74155+0.345132*PETSC_i)/1.7208;
        params.beta = 0.0;
        //params.print();

        std::vector<SPI::SPIVec> eigenfunction(1); //grid.y.rows*16,"eigenfunction");
        //SPI::SPIVec eig_vec(grid.y.rows*8,"q");
        eigenfunction[0] = SPI::ones(grid.y.rows*16);
        std::vector<PetscScalar> eigenvalue(1);
        std::vector<PetscScalar> eigenvalues;
        std::vector<SPI::SPIVec> eigenfunctions;
        eigenvalue[0] = params.alpha;
        //eigenfunction[0].name = "eigenfunction";

        SPI::SPIbaseflow bl_flow = SPI::blasius(params,grid);
        //SPI::SPIbaseflow bl_flow = SPI::channel(params,grid);
        if(1){ // set to parallel baseflow
            SPI::SPIVec o(SPI::zeros(n),"zero");
            bl_flow.Ux = o;
            bl_flow.Uxy = o;
            bl_flow.V = o;
            bl_flow.Vy = o;
        }
        //bl_flow.print();

        std::tie(eigenvalue,eigenfunction) = SPI::LSTNP_spatials_right(params,grid,bl_flow,eigenvalue,eigenfunction);
        eigenvalues.push_back(params.alpha);
        eigenfunctions.push_back(eigenfunction[0]);
        //std::tie(eigenvalue,eigenfunction) = SPI::LST_spatial(params,grid,bl_flow);
        test_if_close(eigenvalue[0]*1.7208,(0.74155+0.345132*PETSC_i),"LSTNP_spatials_right UltraS 1",1e-4);
        //std::cout<<"eigenvalue = "<<eigenvalue[0]*1.7208<<std::endl;
        SPI::SPIMat &O = grid.O;
        SPI::SPIMat &t = grid.T;
        SPI::SPIMat T(SPI::block({
                    {t,O,O,O,O,O,O,O,O,O,O,O,O,O,O,O},
                    {O,t,O,O,O,O,O,O,O,O,O,O,O,O,O,O},
                    {O,O,t,O,O,O,O,O,O,O,O,O,O,O,O,O},
                    {O,O,O,t,O,O,O,O,O,O,O,O,O,O,O,O},
                    {O,O,O,O,t,O,O,O,O,O,O,O,O,O,O,O},
                    {O,O,O,O,O,t,O,O,O,O,O,O,O,O,O,O},
                    {O,O,O,O,O,O,t,O,O,O,O,O,O,O,O,O},
                    {O,O,O,O,O,O,O,t,O,O,O,O,O,O,O,O},
                    {O,O,O,O,O,O,O,O,t,O,O,O,O,O,O,O},
                    {O,O,O,O,O,O,O,O,O,t,O,O,O,O,O,O},
                    {O,O,O,O,O,O,O,O,O,O,t,O,O,O,O,O},
                    {O,O,O,O,O,O,O,O,O,O,O,t,O,O,O,O},
                    {O,O,O,O,O,O,O,O,O,O,O,O,t,O,O,O},
                    {O,O,O,O,O,O,O,O,O,O,O,O,O,t,O,O},
                    {O,O,O,O,O,O,O,O,O,O,O,O,O,O,t,O},
                    {O,O,O,O,O,O,O,O,O,O,O,O,O,O,O,t},
                    })());
        SPI::SPIVec eigenfunctionout(T*eigenfunction[0],"UltraS");
        SPI::save(eigenfunctionout,"eigenfunctionNP.h5");
        SPI::save(grid.y,"eigenfunctionNP.h5");
        // LST_spatial physical
        //std::vector<SPI::SPIVec> eigenfunction16;
        //std::tie(eigenvalue,eigenfunction16) = SPI::LSTNP_spatials_right(params,grid2,bl_flow);
        //test_if_close(eigenvalue[0]*1.7208,(0.74155+0.345132*PETSC_i),"LSTNP_spatials_right physical 1",1e-4);
        //std::cout<<"eigenvalue = "<<eigenvalue<<std::endl;
        // LSTNP_spatial physical
        //std::tie(eigenvalue,eigenfunction) = SPI::LSTNP_spatials_right(params,grid2,bl_flow);
        //test_if_close(eigenvalue[0]*1.7208,(0.74155+0.345132*PETSC_i),"LSTNP_spatials_right physical 1",1e-4);
        //std::cout<<"eigenvalue = "<<eigenvalue<<std::endl;

        params.alpha = (0.54213+0.083968*PETSC_i)/1.7208;
        eigenvalue[0] = params.alpha;
        std::tie(eigenvalue,eigenfunction) = SPI::LSTNP_spatials_right(params,grid,bl_flow,eigenvalue,eigenfunction);
        eigenvalues.push_back(params.alpha);
        eigenfunctions.push_back(eigenfunction[0]);
        test_if_close(eigenvalue[0]*1.7208,(0.54213+0.083968*PETSC_i),"LSTNP_spatials_right UltraS 2",1e-5);
        //std::cout<<"eigenvalue = "<<eigenvalue[0]*1.7208<<std::endl;

        params.alpha = (0.29967+0.230773*PETSC_i)/1.7208;
        eigenvalue[0] = params.alpha;
        std::tie(eigenvalue,eigenfunction) = SPI::LSTNP_spatials_right(params,grid,bl_flow,eigenvalue,eigenfunction);
        eigenvalues.push_back(params.alpha);
        eigenfunctions.push_back(eigenfunction[0]);
        test_if_close(eigenvalue[0]*1.7208,(0.29967+0.230773*PETSC_i),"LSTNP_spatials_right UltraS 3",1e-4);
        //std::cout<<"eigenvalue = "<<eigenvalue[0]*1.7208<<std::endl;
        //std::tie(eigenvalue,eigenfunction) = SPI::LSTNP_spatials_right(params,grid,bl_flow,eigenvalue,eigenfunction);
        //test_if_close(eigenvalue[0]*1.7208,(0.29967+0.230773*PETSC_i),"LSTNP_spatials_right UltraS 3",1e-5);
        //std::cout<<"eigenvalue = "<<eigenvalue[0]*1.7208<<std::endl;

        std::vector<PetscScalar> eigenvalues2;
        std::vector<SPI::SPIVec> eigenfunctions2;
        std::tie(eigenvalues2,eigenfunctions2) = SPI::LSTNP_spatials_right(params,grid,bl_flow,eigenvalues,eigenfunctions);
        test_if_close(eigenvalues2[0]*1.7208,(0.74155+0.345132*PETSC_i),"LSTNP_spatials_right UltraS 1",1e-4);
        test_if_close(eigenvalues2[1]*1.7208,(0.54213+0.083968*PETSC_i),"LSTNP_spatials_right UltraS 2",1e-5);
        test_if_close(eigenvalues2[2]*1.7208,(0.29967+0.230773*PETSC_i),"LSTNP_spatials_right UltraS 3",1e-5);

        SPI::printf("------------ LSTNP_spatials_right Blasius boundary layer UltraS end   -----------");
    }
    if(alltests){
        SPI::printf("------------ UltraS int start -----------");
        PetscInt n=11;
        SPI::SPIVec y(SPI::set_Cheby_mapped_y(-1.,1.,n) ,"yCheby");
        SPI::SPIgrid1D grid(y,"grid",SPI::UltraS);
        SPI::SPIVec a(grid.That*(y^2));
        SPI::SPIVec a2(y^2);
        //std::cout<<"integrate = "<<integrate_coeffs(a)<<std::endl;
        test_if_close(integrate_coeffs(a),2.0/3.0,"integrate_coeffs 1",1e-12);
        test_if_close(integrate_coeffs(a),2.0/3.0,"integrate_coeffs 1",1e-12);
        a = (grid.That*((y^2)+(4.0*y)+1.0));
        test_if_close(integrate_coeffs(a),8.0/3.0,"integrate_coeffs 2",1e-12);
        a = (grid.That*((y^2)+(4.0*y)));
        test_if_close(integrate_coeffs(a),2.0/3.0,"integrate_coeffs 3",1e-12);
        a = (grid.That*(4.0*(y^2)+(y)+10.0));
        test_if_close(integrate_coeffs(a),68.0/3.0,"integrate_coeffs 4",1e-12);

        SPI::SPIVec y2(SPI::set_Cheby_mapped_y(0.,10.,n) ,"yCheby");
        SPI::SPIgrid1D grid2(y2,"grid",SPI::UltraS);
        a = grid2.That*(4.0*(y2^2)+(y2)+10.0);
        test_if_close(integrate_coeffs(a,grid2.y),4450.0/3.0,"integrate_coeffs 5",1e-12);

        SPI::SPIVec y3(SPI::set_Cheby_mapped_y(-10.,10.,n) ,"yCheby");
        SPI::SPIgrid1D grid3(y3,"grid",SPI::UltraS);
        a = grid3.That*(4.0*(y3^2)+(y3)+10.0);
        test_if_close(integrate_coeffs(a,grid3.y),8600.0/3.0,"integrate_coeffs 6",1e-12);

        SPI::SPIVec y4(SPI::set_Cheby_mapped_y(-1.,10.,n) ,"yCheby");
        SPI::SPIgrid1D grid4(y4,"grid",SPI::UltraS);
        a = grid4.That*(4.0*(y4^2)+(y4)+10.0);
        test_if_close(integrate_coeffs(a,grid4.y),8965.0/6.0,"integrate_coeffs 7",1e-12);

        SPI::SPIVec y5(SPI::set_Cheby_mapped_y(-1.,10.,n) ,"yCheby");
        SPI::SPIgrid1D grid5(y5,"grid",SPI::UltraS);
        SPI::SPIgrid1D grid5_2(y5,"grid",SPI::Chebyshev);
        a = grid5.That*(4.0*(y5^2)+(y5)+10.0);
        a2 = (4.0*(y5^2)+(y5)+10.0);
        test_if_close(integrate(a,grid5),8965.0/6.0,"integrate 1 UltraS",1e-12);
        test_if_close(integrate(a2,grid5_2),8965.0/6.0,"integrate 1 Chebyshev",1e-12);

        SPI::SPIVec y6(SPI::set_Cheby_mapped_y(-1.,10.,201) ,"yCheby");
        SPI::SPIgrid1D grid6(y6,"grid",SPI::Chebyshev);
        a2 = (4.0*(y6^2)+(y6)+10.0);
        test_if_close(integrate(a2,grid6),8965.0/6.0,"integrate 2",1e-1);

        SPI::SPIVec y7(SPI::set_Cheby_mapped_y(0.,21.,41) ,"yCheby");
        SPI::SPIgrid1D grid7(y7,"grid",SPI::UltraS);
        SPI::SPIVec a7(41*4,"a3");
        //SPI::SPIVec tmp(grid7.That*(1.0-(y7^2)));
        SPI::SPIVec tmp(grid7.That*(1.0-(y7^2)+y7*y7*y7+y7*PETSC_i));
        for(PetscInt i=0; i<4; ++i){
            for(PetscInt j=0; j<41; ++j){
                a7(i*41+j,tmp(j,PETSC_TRUE));
            }
        }
        a7();
        //test_if_close(integrate(a7,grid7),-3066.0*4.0,"integrate 3",1e-11);
        test_if_close(integrate(a7,grid7),182217.0,"integrate 3",1e-11);
        test_if_close(PetscImaginaryPart(integrate(a7,grid7)),441.0*2.0,"integrate 3 imaginary",1e-11);
        SPI::printf("------------ UltraS int end   -----------");
    }
    if(alltests){ // and timing of LSTNP_spatials_right vs LSTNP_spatial_right
        SPI::printf("------------ LSTNP_spatials_right Blasius boundary layer UltraS start -----------");
        PetscInt n=119;
        SPI::SPIVec y(SPI::set_Cheby_mapped_y(0.,21.,n) ,"yCheby");
        //SPI::SPIVec y(SPI::set_Cheby_mapped_y(-1.,1.,n) ,"yCheby");
        SPI::SPIgrid1D grid(y,"grid",SPI::UltraS);
        //grid.print();
        //exit(0);
        //SPI::SPIVec y(SPI::linspace(0.,61.,n) ,"yFD");
        //SPI::printf("n = %d",n);
        //grid.print();
        //(grid.Dy*grid.y).print();

        SPI::SPIparams params("Blasius parameters");
        //params.Re = 400.0;
        //params.omega = 86.*params.Re/(1000000.);
        params.Re = 1000.0/1.7208;
        params.omega = 0.26/1.7208;
        params.nu = 1./params.Re;
        params.x = params.Re;
        params.x_start = params.x;
        params.alpha = (0.74155+0.345132*PETSC_i)/1.7208;
        params.beta = 0.0;
        //params.print();

        SPI::SPIVec eigenfunction(grid.y.rows*16,"eigenfunction");
        //SPI::SPIVec eig_vec(grid.y.rows*8,"q");
        //SPI::SPIVec leigenfunction(grid.y.rows*16,"q");
        PetscScalar eigenvalue;
        eigenfunction.name = "eigenfunction";

        SPI::SPIbaseflow bl_flow = SPI::blasius(params,grid);
        //SPI::SPIbaseflow bl_flow = SPI::channel(params,grid);
        if(1){ // set to parallel baseflow
            SPI::SPIVec o(SPI::zeros(n),"zero");
            bl_flow.Ux = o;
            bl_flow.Uxy = o;
            bl_flow.V = o;
            bl_flow.Vy = o;
        }
        //bl_flow.print();
        // timing
        time_t timer1,timer2,timer3;
        double seconds;
        time(&timer1); // get current time

        std::vector<PetscScalar> alphas_guess(3);
        std::vector<SPI::SPIVec> eigenfunction_guess(3);
        std::tie(eigenvalue,eigenfunction) = SPI::LSTNP_spatial_right(params,grid,bl_flow);
        alphas_guess[0] = eigenvalue;
        eigenfunction_guess[0] = eigenfunction;
        //std::tie(eigenvalue,eigenfunction) = SPI::LST_spatial(params,grid,bl_flow);
        test_if_close(params.alpha*1.7208,(0.74155+0.345132*PETSC_i),"LSTNP_spatial_right 1",1e-4);

        params.alpha = (0.54213+0.083968*PETSC_i)/1.7208;
        std::tie(eigenvalue,eigenfunction) = SPI::LSTNP_spatial_right(params,grid,bl_flow);
        alphas_guess[1] = eigenvalue;
        eigenfunction_guess[1] = eigenfunction;
        test_if_close(params.alpha*1.7208,(0.54213+0.083968*PETSC_i),"LSTNP_spatial_right 2",1e-5);

        params.alpha = (0.29967+0.230773*PETSC_i)/1.7208;
        std::tie(eigenvalue,eigenfunction) = SPI::LSTNP_spatial_right(params,grid,bl_flow);
        alphas_guess[2] = eigenvalue;
        eigenfunction_guess[2] = eigenfunction;
        test_if_close(params.alpha*1.7208,(0.29967+0.230773*PETSC_i),"LSTNP_spatial_right 3",1e-3);
        time(&timer2); // get current time

        std::vector<PetscScalar> alpha_spatials_right;
        std::vector<SPI::SPIVec> eig_vecs_spatials_right;
        std::tie(alpha_spatials_right,eig_vecs_spatials_right) = SPI::LSTNP_spatials_right(params,grid,bl_flow,alphas_guess,eigenfunction_guess);
        test_if_close(alpha_spatials_right[0]*1.7208,(0.74155+0.345132*PETSC_i),"LSTNP_spatials_right 1",1e-4);
        test_if_close(alpha_spatials_right[1]*1.7208,(0.54213+0.083968*PETSC_i),"LSTNP_spatials_right 2",1e-5);
        test_if_close(alpha_spatials_right[2]*1.7208,(0.29967+0.230773*PETSC_i),"LSTNP_spatials_right 3",1e-3);
        //std::cout<<"alpha_spatials_right[0] = "<<alpha_spatials_right[0]<<std::endl;
        //std::cout<<"alpha_spatials_right[1] = "<<alpha_spatials_right[1]<<std::endl;
        //std::cout<<"alpha_spatials_right[2] = "<<alpha_spatials_right[2]<<std::endl;
        //std::cout<<"alphas_guess[0] = "<<alphas_guess[0]<<std::endl;
        //std::cout<<"alphas_guess[1] = "<<alphas_guess[1]<<std::endl;
        //std::cout<<"alphas_guess[2] = "<<alphas_guess[2]<<std::endl;
        time(&timer3); // get current time
        seconds = difftime(timer2,timer1);
        SPI::printf("%.10f seconds for LSTNP_spatial_right 3 solves on Blasius boundary layer",seconds);
        seconds = difftime(timer3,timer2);
        SPI::printf("%.10f seconds for LSTNP_spatials_right 3 solves on Blasius boundary layer",seconds);

        SPI::printf("------------ LSTNP_spatials_right Blasius boundary layer UltraS end   -----------");
    }
    if(alltests){
        SPI::printf("------------ A*x=b grid UltraS start -----------");
        PetscInt n=20;
        SPI::SPIVec y(SPI::set_Cheby_mapped_y(0.,1.,n) ,"yCheby");
        SPI::SPIgrid1D grid(y,"grid",SPI::Chebyshev);
        //SPI::SPIgrid1D grid(y,"grid",SPI::FD);
        SPI::SPIgrid1D grid2(y,"grid",SPI::UltraS);
        SPI::SPIMat A;
        A = grid.Dyy + grid.Dy + grid.I;
        SPI::SPIVec b(SPI::zeros(n));
        std::vector<PetscInt> rowBCs = {0*n,n-1};
        A.eye_rows(rowBCs);
        b(rowBCs[0],0.0);
        b(rowBCs[1],1.0);
        b();
        SPI::SPIVec x;
        x = SPI::solve(A,b);
        //x.print();

        // exact solution
        SPI::SPIVec xexact(SPI::zeros(n));
        PetscScalar i=PETSC_i;
        PetscScalar a=std::exp(0.5)/(2.0*i*std::sin(std::sqrt(3.0)/2.0));
        //std::cout<<"a = "<<a<<std::endl;
        xexact = SPI::exp(-0.5*grid.y)*((a*SPI::exp((i*std::sqrt(3.0)/2.0)*grid.y)) - (a*SPI::exp((-i*std::sqrt(3.0)/2.0)*grid.y )));
        //(x-xexact).print();
        std::cout<<"error physical = "<<SPI::L2(x-xexact)<<std::endl;

        // UltraS solution
        SPI::SPIMat A2(n,n);
        //MatMPIAIJSetPreallocation(A.mat,20,NULL,20,NULL);
        A2 = grid2.Dyy + grid2.Dy + grid2.I;
        //MatMPIAIJSetPreallocation(A.mat,20,NULL,20,NULL);
        b = SPI::zeros(n);
        rowBCs = {n-2,n-1};
        A2.zero_rows(rowBCs);
        MatSetOption(A2.mat,MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE);
        for(PetscInt j=0; j<n; ++j){
            A2(rowBCs[0],j,grid.T(0,j,PETSC_TRUE));      // u at the wall
            A2(rowBCs[1],j,grid.T(n-1,j,PETSC_TRUE));      // u at the wall
        }
        b(rowBCs[0],0.0);
        b(rowBCs[1],1.0);
        b();
        A2();
        x = SPI::solve(A2,b);
        std::cout<<"error UltraS = "<<SPI::L2((grid.T*x)-xexact)<<std::endl;
        xexact = (SPI::exp(0.5-grid.y/2.0)/std::sin(std::sqrt(3.0)/2.0))*SPI::sin(std::sqrt(3.0)/2.0 * grid.y);
        std::cout<<"error UltraS = "<<SPI::L2((grid.T*x)-xexact)<<std::endl;
        //std::cout<<"erf(1) = "<<std::erf(1.0)<<std::endl;
        //std::cout<<"erf(1) = "<<std::erf((double)1.0)<<std::endl;
        //std::cout<<"erf(1) = "<<std::erf(std::complex<double>(1.0))<<std::endl;
        //SPI::erf(SPI::ones(n)).print();
        //
        // solve u'' + x*u' + u = (4*y + 17)*e^(4*x) with u(0) = 1 and u(1) = e^4
        // solution is u=e^(4*y)
        A = grid.Dyy + SPI::diag(y)*grid.Dy + grid.I;
        b = (4.0*y + 17.0) * SPI::exp(4.0*y);
        rowBCs = {0*n,n-1};
        A.eye_rows(rowBCs);
        b(rowBCs[0],1.0);
        b(rowBCs[1],std::exp(4.0));
        b();
        x = SPI::solve(A,b);

        // exact solution
        xexact = SPI::exp(4.0*y);

        std::cout<<"error non-const = "<<SPI::L2(x - xexact)<<std::endl;

        // solve non-const using UltraS
        A = grid2.Dyy + grid2.S1S0That*(SPI::diag(y)*(grid2.T*grid2.S0invS1inv*grid2.Dy)) + grid2.I;
        b = grid2.S1S0That*(((4.0*y) + 17.0) * SPI::exp(4.0*y));
        rowBCs = {n-2,n-1};
        A.zero_rows(rowBCs);
        MatSetOption(A.mat,MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE);
        for(PetscInt j=0; j<n; ++j){
            A(rowBCs[0],j,grid.T(0,j,PETSC_TRUE));      // u at the wall
            A(rowBCs[1],j,grid.T(n-1,j,PETSC_TRUE));      // u at the wall
        }
        A();
        b(rowBCs[0],1.0);
        b(rowBCs[1],std::exp(4.0));
        b();
        x = SPI::solve(A,b);

        std::cout<<"error non-const UltraS = "<<SPI::L2(grid.T*x - xexact)<<std::endl;

        SPI::printf("------------ A*x=b grid UltraS end   -----------");
    }
    if(alltests){ // and timing of LSTNP_spatials_right vs LSTNP_spatial_right
        SPI::printf("------------ LSTNP_spatials_right non-Parallel Blasius boundary layer UltraS start -----------");
        PetscInt n=169;
        SPI::SPIVec y(SPI::set_Cheby_mapped_y(0.,61.,n) ,"yCheby");
        //SPI::SPIVec y(SPI::set_Cheby_mapped_y(-1.,1.,n) ,"yCheby");
        SPI::SPIgrid1D grid(y,"grid",SPI::UltraS);
        //grid.print();
        //exit(0);
        //SPI::SPIVec y(SPI::linspace(0.,61.,n) ,"yFD");
        //SPI::printf("n = %d",n);
        //grid.print();
        //(grid.Dy*grid.y).print();

        SPI::SPIparams params("Blasius parameters");
        params.Re = 400.0;
        params.omega = 86.*params.Re/(1000000.);
        params.nu = 1./params.Re;
        params.x = params.Re;
        params.x_start = params.x;
        params.alpha = (0.094966+0.004564*PETSC_i);
        params.beta = 0.0;
        //params.print();

        SPI::SPIVec eigenfunction(grid.y.rows*16,"eigenfunction");
        //SPI::SPIVec eig_vec(grid.y.rows*8,"q");
        SPI::SPIVec leigenfunction(grid.y.rows*16,"q");
        PetscScalar eigenvalue,cg;
        eigenfunction.name = "eigenfunction";

        SPI::SPIbaseflow bl_flow = SPI::blasius(params,grid);
        //SPI::SPIbaseflow bl_flow = SPI::channel(params,grid);
        if(0){ // set to parallel baseflow
            SPI::SPIVec o(SPI::zeros(n),"zero");
            bl_flow.Ux = o;
            bl_flow.Uxy = o;
            bl_flow.V = o;
            bl_flow.Vy = o;
        }
        //bl_flow.print();
        // timing
        time_t timer1,timer2;
        double seconds;

        time(&timer1); // get current time
        std::tie(eigenvalue,eigenfunction) = SPI::LSTNP_spatial_right(params,grid,bl_flow);
        time(&timer2); // get current time
        seconds = difftime(timer2,timer1);
        SPI::printf("%.10f seconds for LSTNP_spatial_right 1 solves on Blasius boundary layer",seconds);
        test_if_close(params.alpha,(0.094966355495876+0.004564261943353*PETSC_i),"LSTNP_spatial_right 1",1e-8);

        std::vector<PetscScalar> alphas_guess(2);
        std::vector<SPI::SPIVec> eigenfunction_guess(2);
        time(&timer1); // get current time
        std::tie(eigenvalue,eigenfunction) = SPI::LSTNP_spatial_right(params,grid,bl_flow);
        time(&timer2); // get current time
        seconds = difftime(timer2,timer1);
        SPI::printf("%.10f seconds for LSTNP_spatial_right 1 solves on Blasius boundary layer",seconds);
        alphas_guess[0] = eigenvalue;
        eigenfunction_guess[0] = eigenfunction;
        //std::tie(eigenvalue,eigenfunction) = SPI::LST_spatial(params,grid,bl_flow);
        test_if_close(params.alpha,(0.094966355495876+0.004564261943353*PETSC_i),"LSTNP_spatial_right 1",1e-8);
        //SPI::printf("LSTNP_spatial_right 1 non-Parallel Blasius boundary layer eigenvalue = %.10f + %.10fi",eigenvalue);

        time(&timer1); // get current time
        std::tie(eigenvalue,cg,leigenfunction,eigenfunction) = SPI::LSTNP_spatial(params,grid,bl_flow);
        time(&timer2); // get current time
        seconds = difftime(timer2,timer1);
        SPI::printf("%.10f seconds for LSTNP_spatial 2 solves on Blasius boundary layer",seconds);
        test_if_close(params.alpha,(0.094966355495876+0.004564261943353*PETSC_i),"LSTNP_spatial 2",1e-8);

        params.alpha = (0.10665444+0.00189793*PETSC_i);
        time(&timer1); // get current time
        std::tie(eigenvalue,eigenfunction) = SPI::LSTNP_spatial_right(params,grid,bl_flow);
        time(&timer2); // get current time
        seconds = difftime(timer2,timer1);
        SPI::printf("%.10f seconds for LSTNP_spatial_right 3 solves on Blasius boundary layer",seconds);
        alphas_guess[1] = eigenvalue;
        eigenfunction_guess[1] = eigenfunction;
        test_if_close(params.alpha,(0.106654447306241+0.001897936897103*PETSC_i),"LSTNP_spatial_right 3",1e-8);
        //SPI::printf("LSTNP_spatial_right 2 non-Parallel Blasius boundary layer eigenvalue = %.10f + %.10fi",eigenvalue);

        time(&timer1); // get current time
        std::tie(eigenvalue,cg,leigenfunction,eigenfunction) = SPI::LSTNP_spatial(params,grid,bl_flow);
        time(&timer2); // get current time
        seconds = difftime(timer2,timer1);
        SPI::printf("%.10f seconds for LSTNP_spatial 4 solves on Blasius boundary layer",seconds);
        test_if_close(params.alpha,(0.106654447306241+0.001897936897103*PETSC_i),"LSTNP_spatial 4",1e-8);

        std::vector<PetscScalar> eigenvalues(2);
        std::vector<SPI::SPIVec> eigenfunctions(2);
        time(&timer1); // get current time
        std::tie(eigenvalues,eigenfunctions) = SPI::LSTNP_spatials_right(params,grid,bl_flow,alphas_guess,eigenfunction_guess);
        time(&timer2); // get current time
        seconds = difftime(timer2,timer1);
        SPI::printf("%.10f seconds for LSTNP_spatials_right 5,6 solves on Blasius boundary layer",seconds);
        test_if_close(eigenvalues[0],(0.094966355495876+0.004564261943353*PETSC_i),"LSTNP_spatials_right 5",1e-8);
        test_if_close(eigenvalues[1],(0.106654447306241+0.001897936897103*PETSC_i),"LSTNP_spatials_right 6",1e-8);

        SPI::printf("------------ LSTNP_spatials_right non-Parallel Blasius boundary layer UltraS end   -----------");
    }
    if(alltests){ // and timing of LSTNP_spatials_right vs LSTNP_spatial_right
        SPI::printf("------------ LSTNP_spatials_right non-Parallel Blasius boundary layer UltraS start -----------");
        PetscInt n=169;
        SPI::SPIVec y(SPI::set_Cheby_mapped_y(0.,61.,n) ,"yCheby");
        //SPI::SPIVec y(SPI::set_Cheby_mapped_y(-1.,1.,n) ,"yCheby");
        SPI::SPIgrid1D grid(y,"grid",SPI::UltraS);
        //grid.print();
        //exit(0);
        //SPI::SPIVec y(SPI::linspace(0.,61.,n) ,"yFD");
        //SPI::printf("n = %d",n);
        //grid.print();
        //(grid.Dy*grid.y).print();

        SPI::SPIparams params("Blasius parameters");
        params.Re = 400.0;
        params.omega = 86.*params.Re/(1000000.);
        params.nu = 1./params.Re;
        params.x = params.Re;
        params.x_start = params.x;
        params.alpha = (0.094966+0.004564*PETSC_i);
        params.beta = 0.0;
        //params.print();

        SPI::SPIVec eigenfunction(grid.y.rows*16,"eigenfunction");
        //SPI::SPIVec eig_vec(grid.y.rows*8,"q");
        SPI::SPIVec leigenfunction(grid.y.rows*16,"q");
        PetscScalar eigenvalue,cg;
        eigenfunction.name = "eigenfunction";

        SPI::SPIbaseflow bl_flow = SPI::blasius(params,grid);
        //SPI::SPIbaseflow bl_flow = SPI::channel(params,grid);
        if(0){ // set to parallel baseflow
            SPI::SPIVec o(SPI::zeros(n),"zero");
            bl_flow.Ux = o;
            bl_flow.Uxy = o;
            bl_flow.V = o;
            bl_flow.Vy = o;
        }
        //bl_flow.print();
        // timing
        time_t timer1,timer2;
        double seconds;

        time(&timer1); // get current time
        std::tie(eigenvalue,eigenfunction) = SPI::LSTNP_spatial_right2(params,grid,bl_flow);
        time(&timer2); // get current time
        seconds = difftime(timer2,timer1);
        SPI::printf("%.10f seconds for LSTNP_spatial_right 1 solves on Blasius boundary layer",seconds);
        test_if_close(params.alpha,(0.094966355495876+0.004564261943353*PETSC_i),"LSTNP_spatial_right 1",1e-8);
        SPI::printf("------------ LSTNP_spatials_right non-Parallel Blasius boundary layer UltraS end   -----------");
    }
    if(0){ // and timing of LSTNP_spatials_right vs LSTNP_spatial_right
        SPI::printf("------------ LSTNP_spatials_right non-Parallel Blasius boundary layer UltraS start -----------");
        PetscInt n=169;
        SPI::SPIVec y(SPI::set_Cheby_mapped_y(0.,61.,n) ,"yCheby");
        //SPI::SPIVec y(SPI::set_Cheby_mapped_y(-1.,1.,n) ,"yCheby");
        //SPI::SPIgrid1D grid0(y,"grid",SPI::UltraS); // good now
        //SPI::SPIgrid1D grid2(y,"grid",SPI::Chebyshev); // good now
        //SPI::SPIgrid1D grid3(y,"grid",SPI::FD); // good now
        //SPI::SPIVec y2;
        //2.0*y;
        //2.0+y;
        //2.0/y;
        //y/2.0;
        //y2 = y;
        // test grid making routines
        //SPI::SPIMat Dy_1(SPI::set_D(y,1)); // this is good now
        //SPI::SPIMat Dy_2(SPI::set_D_Chebyshev(y,1,PETSC_TRUE)); // this is good now
        //SPI::SPIMat Dyy_1(SPI::set_D(y,2)); // this is good now
        //SPI::SPIMat Dyy_2(SPI::set_D_Chebyshev(y,2,PETSC_TRUE)); // this is good now
        //SPI::SPIVec xi = (SPI::ones(n));// good
        //SPI::SPIMat I(SPI::eye(n),"I");// good
        //SPI::SPIVec s(SPI::arange(4));// good
        //SPI::SPIMat D(n);// good
        //D = SPI::zeros(n,n);
        //SPI::SPIVec Coeffs(SPI::get_D_Coeffs(s,1)); // this is good now
        //Coeffs.~SPIVec();
        //Coeffs = SPI::arange(4);
        //Coeffs.~SPIVec();
        //Coeffs = SPI::arange(4);
        //D*=4.0;
        //SPI::map_D(D,y,1,4); // good now
        //SPI::SPIMat Dy(SPI::map_D(D,y,1,4)); // good now
        //SPI::SPIMat D1(SPI::eye(n)); // good
        //SPI::diag(1./(D*y)); // good
        //SPI::diag(1./(D*y))*D; // good now;
        //D*D; // good now
        //y^3;
        //y^y;
        //D*y;
        //D+D;
        //SPI::SPIMat Dy1(SPI::set_D(y,1,4,PETSC_TRUE));
        //-1.*(D*y);
        //SPI::diag(y^2);
        //s *= y; // good
        //SPI::factorial(3);
        //SPI::SPIMat A(3);
        //A(0,0,1.0); A(0,1,1.0); A(0,2,1.0);
        //A(1,0,1.0); A(1,1,2.0); A(1,2,3.0);
        //A(2,0,1.0); A(2,1,4.0); A(2,2,9.0);
        //SPI::SPIVec b(3);
        //b(1,1.0);
        //A();b();
        //SPI::solve(A,b); // good now
        //b/A; // good now
        //SPI::block({
                //{A,A},
                //{A,A}
                //})();
        //SPI::set_T_That(y.rows); // good now
        //SPI::meshgrid(y,y); // good now
        //D%D; // good now
        //SPI::SPIMat A2(4,3); // good
        //A2();
        //SPI::SPIMat A3(A2); // good
        //SPI::SPIMat Y1,Y2;
        //std::tie(Y1,Y2) = SPI::meshgrid(y,y); // good
        //cos(Y1%acos(Y2));
        //SPI::SPIMat Y3;
        //Y1.T(Y3);
        //SPI::SPIgrid1D grid(y,"grid",SPI::UltraS); // works
        SPI::SPIgrid1D grid(y,"grid",SPI::Chebyshev); // works
        //grid.print();
        //exit(0);
        //SPI::SPIVec y(SPI::linspace(0.,61.,n) ,"yFD");
        //SPI::printf("n = %d",n);
        //grid.print();
        //(grid.Dy*grid.y).print();

        SPI::SPIparams params("Blasius parameters");
        params.Re = 400.0;
        params.omega = 86.*params.Re/(1000000.);
        params.nu = 1./params.Re;
        params.x = params.Re;
        params.x_start = params.x;
        params.alpha = (0.094966+0.004564*PETSC_i);
        params.beta = 0.0;
        //params.print();

        SPI::SPIVec eigenfunction(grid.y.rows*16,"eigenfunction");
        SPI::SPIVec eigenfunction2(grid.y.rows*8,"eigenfunction");
        //SPI::SPIVec eig_vec(grid.y.rows*8,"q");
        SPI::SPIVec leigenfunction(grid.y.rows*16,"q");
        PetscScalar eigenvalue,eigenvalue2,cg;
        eigenfunction.name = "eigenfunction";

        SPI::SPIbaseflow bl_flow = SPI::blasius(params,grid);
        //SPI::SPIbaseflow bl_flow = SPI::channel(params,grid);
        if(0){ // set to parallel baseflow
            SPI::SPIVec o(SPI::zeros(n),"zero");
            bl_flow.Ux = o;
            bl_flow.Uxy = o;
            bl_flow.V = o;
            bl_flow.Vy = o;
        }
        //bl_flow.print();
        // timing
        time_t timer1,timer2;
        double seconds;

        time(&timer1); // get current time
        params.print();
        //grid.print();
        //(grid.Dy*grid.y).print();
        //(grid.Dyy*grid.y).print();
        //bl_flow.print();
        //grid.T.print();
        //grid.That.print();
        //std::tie(eigenvalue,eigenfunction) = SPI::LSTNP_spatial_right2(params,grid,bl_flow); // good
        std::tie(eigenvalue,eigenfunction) = SPI::LSTNP_spatial_right(params,grid,bl_flow);
        std::tie(eigenvalue2,eigenfunction2) = SPI::LST_spatial(params,grid,bl_flow);
        time(&timer2); // get current time
        seconds = difftime(timer2,timer1);
        SPI::printf("%.10f seconds for LSTNP_spatial_right 1 solves on Blasius boundary layer",seconds);
        test_if_close(eigenvalue,(0.094966355495876+0.004564261943353*PETSC_i),"LSTNP_spatial_right 1",1e-8);
        SPI::printf("------------ LSTNP_spatials_right non-Parallel Blasius boundary layer UltraS end   -----------");

        SPI::SPIMat M(4),C(4),K(4);
        M(0,0,3.0); M(1,1,1.0); M(2,2,3.0); M(3,3,1.0); M();
        C(0,0,0.4); C(0,2,-0.3);
        C(2,0,-0.3); C(2,2,0.5); C(2,3,-0.2);
        C(3,2,-0.2); C(3,3,0.2);
        C();
        K(0,0,-7.0); K(0,1, 2.0); K(0,2, 4.0);
        K(1,0, 2.0); K(1,1,-4.0); K(1,2, 2.0);
        K(2,0, 4.0); K(2,1, 2.0); K(2,2,-9.0); K(2,3, 3.0);
                                  K(3,2, 3.0); K(3,3,-3.0);
        K();
        SPI::SPIVec eigenfunction_4(4);
        std::tie(eigenvalue,eigenfunction_4) = SPI::polyeig({K,C,M},-2.4498); // good
        SPI::printfc("eigenvalue = %.10f + %.10f",eigenvalue);
        K.zero_rows({1,2});
        M.eye_row(1);
        K = M*K;
        //eigenfunction_4.print();
        //(0.1828*eigenfunction_4/eigenfunction_4(0,PETSC_TRUE)).print();
        //(eigenvalue*eigenvalue*(M*eigenfunction_4) + eigenvalue*(C*eigenfunction_4) + (K*eigenfunction_4)).print();
    }
    if(alltests){
        SPI::printf("------------ SVD start -----------");
        SPI::SPIVec x(4,"x");
        x(0,0.0); x(1,1.0); x(2,2.0); x(3,3.0);
        x();
        SPI::SPIVec y(4,"y");
        y(0,-1.0); y(1,0.2); y(2,0.9); y(3,2.1);
        y();
        //y.print();
        SPI::SPIMat A(4,2,"A");
        A(0,0,0); A(0,1,1.0);
        A(1,0,1); A(1,1,1.0);
        A(2,0,2); A(2,1,1.0);
        A(3,0,3); A(3,1,1.0);
        A();

        std::vector<PetscReal> sigma;
        std::vector<SPI::SPIVec> u;
        std::vector<SPI::SPIVec> v;

        //std::cout<<"made it here before svd"<<std::endl;
        //A.print();
        std::tie(sigma,u,v) = SPI::svd(A);
        //SPI::svd(A);
        //std::cout<<"made it here after  svd"<<std::endl;
        SPI::SPIVec xi(SPI::zeros(2));
        for(PetscInt j=0; j<2; ++j){
            xi += ((y.dot(u[j]))/sigma[j]) * v[j];
            //std::cout<<"sigma["<<j<<"] = "<<sigma[j]<<std::endl;
            //u[j].print();
            //v[j].print();
        }
        //xi.print();
        test_if_close(xi(0,PETSC_TRUE),1.0,"SVD 1",1e-14);
        test_if_close(xi(1,PETSC_TRUE),-0.95,"SVD 2",1e-14);

        //(A*v[0] - sigma[0]*u[0]).print();
        //(A*v[1] - sigma[1]*u[1]).print();
        //(A*xi).print();

        SPI::printf("------------ SVD end   -----------");
    }
    if(alltests){
        SPI::printf("------------ least squares start -----------");
        SPI::SPIVec x;
        //SPI::SPIVec x(4,"x");
        //x(0,0.0); x(1,1.0); x(2,2.0); x(3,3.0);
        //x();
        SPI::SPIVec y(4,"y");
        y(0,-1.0); y(1,0.2); y(2,0.9); y(3,2.1);
        y();
        SPI::SPIMat A(4,2,"A");
        A(0,0,0); A(0,1,1.0);
        A(1,0,1); A(1,1,1.0);
        A(2,0,2); A(2,1,1.0);
        A(3,0,3); A(3,1,1.0);
        A();
        
        x = lstsq(A,y);
        test_if_close(x(0,PETSC_TRUE),1.0,"lstsq 1",1e-14);
        test_if_close(x(1,PETSC_TRUE),-0.95,"lstsq 2",1e-14);
        //x.print();

        SPI::printf("------------ least squares end   -----------");
    }
    if(alltests){
        SPI::printf("------------ set_col start -----------");
        SPI::SPIVec x1(4,"x1");
        x1(0,0.0); x1(1,1.0); x1(2,2.0); x1(3,3.0);
        x1();
        //x1.print();
        SPI::SPIVec x2(4,"x2");
        x2(0,1.0); x2(1,1.0); x2(2,1.0); x2(3,1.0);
        x2();
        //x2.print();
        SPI::SPIMat A(4,2,"A");
        A.set_col(0,x1);
        A.set_col(1,x2);
        A();
        //A.print();
        test_if_close(A(2,0,PETSC_TRUE),2.0,"set_col 1",1e-14);
        test_if_close(A(2,1,PETSC_TRUE),1.0,"set_col 2",1e-14);
        SPI::printf("------------ set_col end   -----------");
    }
    if(alltests){
        SPI::printf("------------ SPIMat(std::vector<SPIVec>) and lstsq(std::vector<SPIVec>,SPIVec) start -----------");
        SPI::SPIVec y(4,"y");
        y(0,-1.0); y(1,0.2); y(2,0.9); y(3,2.1);
        y();
        SPI::SPIVec x1(4,"x1");
        x1(0,0.0); x1(1,1.0); x1(2,2.0); x1(3,3.0);
        x1();
        SPI::SPIVec x2(4,"x2");
        x2(0,1.0); x2(1,1.0); x2(2,1.0); x2(3,1.0);
        x2();
        std::vector<SPI::SPIVec> x = {x1,x2};
        SPI::SPIMat A(x);
        //A.print();
        test_if_close(A(2,0,PETSC_TRUE),2.0,"set_col 1",1e-14);
        test_if_close(A(2,1,PETSC_TRUE),1.0,"set_col 2",1e-14);

        SPI::SPIVec xi(SPI::lstsq(x,y));
        test_if_close(xi(0,PETSC_TRUE),1.0,"lstsq(std::vector<SPIVec>,SPIVec) 1",1e-14);
        test_if_close(xi(1,PETSC_TRUE),-0.95,"lstsq(std::vector<SPIVec>,SPIVec) 2",1e-14);
        SPI::printf("------------ SPIMat(std::vector<SPIVec>) and lstsq(std::vector<SPIVec>,SPIVec) end   -----------");
    }
    if(alltests){
        SPI::printf("------------ SPIMat save start -----------");
        SPI::SPIMat A(4,2,"A");
        A(0,0,0.2+3.1*PETSC_i); A(0,1,1.0+4.0*PETSC_i);
        A(1,0,1.0+4.2*PETSC_i); A(1,1,2.0+3.0*PETSC_i);
        A(2,0,2.0+3.3*PETSC_i); A(2,1,3.0+2.0*PETSC_i);
        A(3,0,3.0+4.4*PETSC_i); A(3,1,4.0+1.0*PETSC_i);
        A();
        A.print();
        SPI::save(A,"A.dat");
        //SPI::load(A,"A.dat");
        //A.print();
        SPI::printf("------------ SPIMat save end   -----------");
    }
    if(1){
        SPI::printf("------------ SPIgrid2D.avgt start -----------");
        PetscInt ny=400;
        PetscInt nt=4;
        SPI::SPIVec y(SPI::set_FD_stretched_y(61.,ny,1.01) ,"yFD");
        SPI::SPIVec t(SPI::set_Fourier_t((2.0*M_PI)/0.0344,nt) ,"t");
        SPI::SPIgrid2D grid(y,t,"grid",SPI::FD,SPI::FT);
        //(grid.avgt*SPI::ones(400*4)).print();
        test_if_close(grid.avgt(1,1,PETSC_TRUE),0.25,"SPIgrid2D.avgt 1",1e-14);
        //t.print();
        SPI::printf("------------ SPIgrid2D.avgt end   -----------");
    }


    return 0;
}
