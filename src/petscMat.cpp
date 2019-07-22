#include <iostream>
#include <petscksp.h>
// #include <petscsys.h>

static char help[] = "SPE class to wrap PETSc Mat variables \n\n";

struct SPEMat{
    PetscInt rows;
    PetscInt cols;

    // constructors
    SPEMat(){
    }
    SPEMat(PetscInt rowscols){
        rows=rowscols;
        cols=rowscols;
        Init(rows,cols);
    }
    SPEMat(PetscInt rowsm, PetscInt colsn){
        Init(rowsm,colsn);
    }

    Mat mat;

    PetscErrorCode ierr;

    // Initialize matrix
    PetscInt Init(PetscInt m,PetscInt n){
        rows=m;
        cols=n;
        ierr = MatCreate(PETSC_COMM_WORLD,&mat);CHKERRQ(ierr);
        ierr = MatSetSizes(mat,PETSC_DECIDE,PETSC_DECIDE,m,n);CHKERRQ(ierr);
        //ierr = MatSetFromOptions(mat);CHKERRQ(ierr);
        ierr = MatSetType(mat,MATMPIAIJ);CHKERRQ(ierr);
        ierr = MatSetUp(mat);CHKERRQ(ierr);
        return 0;
    }

    PetscInt set(PetscInt m, PetscInt n,PetscScalar v){
        ierr = MatSetValue(mat,m,n,v,INSERT_VALUES);CHKERRQ(ierr);
        return 0;
    }

    // overloaded operators, get
    PetscScalar operator()(PetscInt m, PetscInt n) {
        PetscScalar v;
        ierr = MatGetValues(mat,1,&m, 1,&n, &v);
        return v;
    }
    // overloaded operator, set
    PetscInt operator()(PetscInt m, PetscInt n,PetscScalar v){
        ierr = MatSetValue(mat,m,n,v,INSERT_VALUES);CHKERRQ(ierr);
        return 0;
    }
    // overloaded operator, set
    PetscInt operator()(PetscInt m, PetscInt n,SPEMat& Asub){
        InsertMode addv = INSERT_VALUES;
        PetscInt rowoffset = m;
        PetscInt coloffset = n;
        PetscInt nsub = Asub.rows;
        PetscInt Isubstart,Isubend;
        PetscInt ncols;
        const PetscInt *cols;
        const PetscScalar *vals;

        // get ranges for each matrix
        MatGetOwnershipRange(Asub.mat,&Isubstart,&Isubend);

        PetscErrorCode ierr;
        for (PetscInt i=Isubstart; i<Isubend && i<nsub; i++){
            ierr = MatGetRow(Asub.mat,i,&ncols,&cols,&vals);CHKERRQ(ierr);
            PetscInt offcols[ncols];
            PetscScalar avals[ncols];
            for (PetscInt j=0; j<ncols; j++) {
                offcols[j] = cols[j]+coloffset;
                avals[j] = vals[j];
            }
            //printScalar(vals,ncols);
            PetscInt rowoffseti = i+rowoffset;
            ierr = MatSetValues(mat,1,&rowoffseti,ncols,offcols,avals,addv);CHKERRQ(ierr);
            
            MatRestoreRow(Asub.mat,i,&ncols,&cols,&vals);
        }
        (*this)();//assemble
        return 0;
    }
    // overloaded operator, assemble
    PetscInt operator()(){
        ierr = MatAssemblyBegin(mat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
        ierr = MatAssemblyEnd(mat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
        return 0;
    }
    // overloaded operator, MatAXPY
    SPEMat& operator+=(const SPEMat &X){
        ierr = MatAXPY(this->mat,1.,X.mat,DIFFERENT_NONZERO_PATTERN);CHKERRXX(ierr);
        return *this;
    }
    // overloaded operator, MatAXPY
    SPEMat operator+(const SPEMat &X){
        SPEMat A;
        A=*this;
        ierr = MatAXPY(A.mat,1.,X.mat,DIFFERENT_NONZERO_PATTERN);CHKERRXX(ierr);
        return A;
    }
    // overloaded operator, MatAXPY
    SPEMat& operator-=(const SPEMat &X){
        ierr = MatAXPY(this->mat,-1.,X.mat,DIFFERENT_NONZERO_PATTERN);CHKERRXX(ierr);
        return *this;
    }
    // overloaded operator, MatAXPY
    SPEMat operator-(const SPEMat &X){
        SPEMat A;
        A=*this;
        ierr = MatAXPY(A.mat,-1.,X.mat,DIFFERENT_NONZERO_PATTERN);CHKERRXX(ierr);
        return A;
    }
    // overload operator, scale with scalar
    SPEMat operator*(const PetscScalar a){
        SPEMat A;
        A=*this;
        ierr = MatScale(A.mat,a);CHKERRXX(ierr);
        return A;
    }
    // overload operator, scale with scalar
    SPEMat& operator*=(const PetscScalar a){
        ierr = MatScale(this->mat,a);CHKERRXX(ierr);
        return *this;
    }
    // overload operator, matrix multiply
    SPEMat operator*(const SPEMat& A){
        SPEMat C;
        C.rows=rows;
        C.cols=cols;
        ierr = MatMatMult(mat,A.mat,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&C.mat); CHKERRXX(ierr);
        return C;
    }
    // overload operator, copy and initialize
    SPEMat& operator=(const SPEMat A){
        rows=A.rows;
        cols=A.cols;
        //this->mat = A.mat;
         ierr = MatConvert(A.mat,MATSAME,MAT_INITIAL_MATRIX,&mat);CHKERRXX(ierr);
        return *this;
    }
    // overload % for element wise multiplication
    //SPEMat operator%(SPEMat A){
        //return *this;
        //}     
    PetscInt T(SPEMat& A){
        ierr = MatTranspose(mat,MAT_INITIAL_MATRIX,&A.mat);CHKERRQ(ierr);
        return 0;
    }
    PetscInt T(){
        ierr = MatTranspose(mat,MAT_INPLACE_MATRIX,&mat);CHKERRQ(ierr);
        return 0;
    }
    // print matrix to screen
    PetscInt print(){
        ierr = MatView(mat,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
        return 0;
    }

    ~SPEMat(){
        MatDestroy(&mat);
    }

};

// overload operator, scale with scalar
SPEMat operator*(const PetscScalar &a, SPEMat &A){
    //PetscErrorCode ierr;
    //SPEMat B;
    //B=A;
    //ierr = MatScale(B.mat,a);CHKERRXX(ierr);
    //return B;
    return A*a;
}

int main(int argc, char **args){
    PetscInt m=4,n=4;
    PetscErrorCode ierr;
    ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;



    //SPEMat B(argc,args);
    //ierr = B.Init(m,n);CHKERRQ(ierr);
    SPEMat B(m,n),C(m,n),D,E(4*m,4*n);
    C(0,1,1.);
    B(1,1,1.0);
    B();
    C();
    B=(3.4+PETSC_i*4.2)*B;
    C+=B;
    D = C*B;
    B.print();
    C.print();
    D.print();
    D.T();
    D.print();
    E(3,3,D);
    E.print();
    B.~SPEMat();
    C.~SPEMat();
    D.~SPEMat();
    E.~SPEMat();



    ierr = PetscFinalize();CHKERRQ(ierr);

    return 0;
}
