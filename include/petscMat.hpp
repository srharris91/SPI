#ifndef SPEMAT_H
#define SPEMAT_H
#include <iostream>
#include <petscksp.h>

namespace SPE{
    struct SPEMat{
        PetscInt rows;              ///< number of rows in mat
        PetscInt cols;              ///< number of columns in mat

        // Constructors
        SPEMat();                   ///< constructor with no arguments (no initialization)
        SPEMat(PetscInt rowscols);  ///< constructor with one arguement to make square matrix
        SPEMat(PetscInt rowsm, PetscInt colsn); ///< constructor of rectangular matrix

        Mat mat;                ///< petsc Mat data
        PetscErrorCode ierr;    ///< ierr for various routines and operators
        
        PetscInt Init(PetscInt m,PetscInt n); ///< initialize the matrix of size m by n
        PetscInt set(PetscInt m, PetscInt n,PetscScalar v); ///< set a scalar value at position row m and column n
        // () operators
        PetscScalar operator()(PetscInt m, PetscInt n);     ///< get local value at row m, column n
        PetscInt operator()(PetscInt m, PetscInt n,PetscScalar v);  ///< set operator the same as set function
        PetscInt operator()(PetscInt m, PetscInt n,SPEMat& Asub, InsertMode addv=ADD_VALUES);   ///< set submatrix into matrix at row m, col n
        PetscInt operator()();                                      ///< assemble the matrix
        // +- operators
        SPEMat& operator+=(const SPEMat &X); ///< MatAXPY,  Y = 1.*X + Y operation
        SPEMat operator+(const SPEMat &X); ///< Y + X operation
        SPEMat& operator-=(const SPEMat &X); ///< Y = -1.*X + Y operation
        SPEMat operator-(const SPEMat &X); ///< Y - X operation
        // * operators
        SPEMat operator*(const PetscScalar a); ///< Y*a operation
        SPEMat& operator*=(const PetscScalar a); ///< Y = Y*a operation
        SPEMat operator*(const SPEMat& A); ///< Y*A operation
        // = operator
        SPEMat& operator=(const SPEMat A); ///< Y=X with initialization of Y using MatConvert
        // overload % for element wise multiplication
        //SPEMat operator%(SPEMat A); 
        // Transpose functions
        PetscInt T(SPEMat& A); ///< A = Transpose(*this.mat) operation with initialization of A
        PetscInt T(); ///< Transpose the current mat
        PetscInt print(); ///< print mat to screen using PETSC_VIEWER_STDOUT_WORLD

        ~SPEMat(); /// destructor to delete memory

    };
    SPEMat operator*(const PetscScalar a, SPEMat &A); ///< a*A operation to be equivalent to A*a
}



#endif // SPEMAT_H
