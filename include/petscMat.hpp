#ifndef SPEMAT_H
#define SPEMAT_H

namespace SPE{
    struct SPEMat{
        PetscInt rows;
        PetscInt cols;

        // constructors
        SPEMat();       ///< constructor with no arguments (no initialization)
        SPEMat(PetscInt rowscols); ///< constructor with one arguement to make square matrix
        SPEMat(PetscInt rowsm, PetscInt colsn); ///< constructor of rectangular matrix
        Mat mat; ///< petsc Mat data

        PetscErrorCode ierr; ///< ierr for various routines and operators

        // Initialize matrix
        PetscInt Init(PetscInt m,PetscInt n); ///< initialize the matrix of size m by n
        PetscInt set(PetscInt m, PetscInt n,PetscScalar v); ///< set a scalar value at position row m and column n

        // overloaded operators, get
        PetscScalar operator()(PetscInt m, PetscInt n); ///< get local value at row m, column n
        // overloaded operator, set
        PetscInt operator()(PetscInt m, PetscInt n,PetscScalar v); ///< operator the same as set function
        // overloaded operator, set
        PetscInt operator()(PetscInt m, PetscInt n,SPEMat& Asub); ///< set submatrix into matrix at row m, col n
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

    }


}



#endif // 
