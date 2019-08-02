#include "main.hpp"
#include <tuple>

static char help[] = "SPE class to wrap PETSc Mat variables \n\n";

int main(int argc, char **args){
    PetscInt m=4,n=4;
    PetscErrorCode ierr;
    ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;

    // Vec tests
    if(0){
        SPE::SPEVec X1(m,"X1"),X2(m,"X2"),X3("X3");

        X1(0,0.+4.*PETSC_i);
        X1(1,1.+3.*PETSC_i);
        X1(2,2.+2.*PETSC_i);
        X1.set(3,3.+1.*PETSC_i);

        X1.print();

        X2=2.*X1;
        X3=X1;
        X2.print();
        SPE::SPEVec X4("X4");
        X3.print();
        X4=2.*X2*X3+X1;
        X4.print();
        X1.print();
        X3=X2+X1;
        X3.print();
        std::cout<<X4(2)<<std::endl;
        //X2.print();

        //X3 = X2+X1;

        //X3.print();

        // destroy (unless wrapped in if statment, then destructor will do it)
        //X1.~SPEVec();
        //X2.~SPEVec();
        //X3.~SPEVec();
        //X4.~SPEVec();
    }

    if(0){ // Mat tests
        //SPEMat B(argc,args);
        //ierr = B.Init(m,n);CHKERRQ(ierr);
        SPE::SPEMat I(m,n,"I"),A2(m,n,"A"),B(m,n,"B"),C(m,n,"C"),D("D"),E(4*m,4*n,"E");
        for (int i=0; i<m; i++){
            A2(i,i,1.+PETSC_i);
            I(i,i,1);
        }
        I();
        C(0,1,1.);
        A2.print();
        B(1,1,1.0);
        B();
        B.print();
        C();
        C.print();
        B=(3.4+PETSC_i*4.2)*C+4.*C;
        B.print();
        C+=B;
        C.print();
        SPE::SPEMat CT("CT");
        C.T(CT);
        CT.print();
        D = CT*B;
        D.print();
        D *= 4.;
        D.print();
        D.~SPEMat();
        C*B;
        D = C*B;
        D.print();
        B.T();
        B.print();
        C.print();
        D.print();
        D();
        D=I;
        D.print();
        E(4,7,B);// insert
        E(0,0,C);
        E(1,5,D);
        E();
        E.print();
        E.H();
        E.print();
        E.conj();
        E.print();
        E.T();
        E.print();
        SPE::SPEMat F(E);
        F.print();
        //F.~SPEMat();
        //A2.~SPEMat();
        //B.~SPEMat();
        //C.~SPEMat();
        //D.~SPEMat();
        //E.~SPEMat();
        //CT.~SPEMat();
        //I.~SPEMat();
    }

    // test A*x
    if(0){
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

    }

    // linear system solver test Ax=b solved with x=b/A
    if(0){
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
    }
    // check Mat functions (eye, kron, diag)
    if(1){
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
    }



    ierr = PetscFinalize();CHKERRQ(ierr);

    return 0;
}
