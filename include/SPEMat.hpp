#ifndef SPEMAT_H
#define SPEMAT_H
#include <iostream>
#include <petscksp.h>
#include <string>
#include "SPEVec.hpp"

namespace SPE{
    struct SPEMat{
        PetscInt rows;              ///< number of rows in mat
        PetscInt cols;              ///< number of columns in mat

        // Constructors
        SPEMat(std::string _name="SPEMat");                   ///< constructor with no arguments (no initialization)
        SPEMat(const SPEMat  &A, std::string _name="SPEMat");      ///< constructor using another SPEMat
        SPEMat(PetscInt rowscols, std::string _name="SPEMat");  ///< constructor with one arguement to make square matrix
        SPEMat(PetscInt rowsm, PetscInt colsn, std::string _name="SPEMat"); ///< constructor of rectangular matrix

        Mat mat;                ///< petsc Mat data
        PetscErrorCode ierr;    ///< ierr for various routines and operators

        // flags
        PetscBool flag_init=PETSC_FALSE;    ///< flag if it has been initialized
        std::string name;            ///< Matrix name
        
        PetscInt Init(PetscInt m,PetscInt n, std::string name="SPEMat"); ///< initialize the matrix of size m by n
        PetscInt set(PetscInt m, PetscInt n,const PetscScalar v); ///< set a scalar value at position row m and column n
        PetscInt add(PetscInt m, PetscInt n,const PetscScalar v); ///< add a scalar value at position row m and column n
        // () operators
        PetscScalar operator()(PetscInt m, PetscInt n);     ///< get local value at row m, column n
        PetscInt operator()(PetscInt m, PetscInt n,const PetscScalar v);  ///< set operator the same as set function
        PetscInt operator()(PetscInt m, PetscInt n,const double v);  ///< set operator the same as set function
        PetscInt operator()(PetscInt m, PetscInt n,const int v);  ///< set operator the same as set function
        PetscInt operator()(PetscInt m, PetscInt n,const SPEMat &Asub, InsertMode addv=ADD_VALUES);   ///< set submatrix into matrix at row m, col n
        PetscInt operator()();                                      ///< assemble the matrix
        // +- operators
        SPEMat& operator+=(const SPEMat &X); ///< MatAXPY,  Y = 1.*X + Y operation
        SPEMat operator+(const SPEMat &X); ///< Y + X operation
        PetscInt axpy(const PetscScalar a, const SPEMat &X); ///< MatAXPY function call to add a*X to the current mat
        SPEMat& operator-=(const SPEMat &X); ///< Y = -1.*X + Y operation
        SPEMat operator-(const SPEMat &X); ///< Y - X operation
        // * operators
        SPEMat operator*(const PetscScalar a); ///< Y*a operation
        SPEMat operator*(const double a); ///< Y*a operation
        SPEVec operator*(const SPEVec &x); ///< A*x operation to return a vector
        SPEMat& operator*=(const PetscScalar a); ///< Y = Y*a operation
        SPEMat operator*(const SPEMat &A); ///< Y*A operation
        // = operator
        SPEMat& operator=(const SPEMat &A); ///< Y=X with initialization of Y using MatConvert
        // overload % for element wise multiplication
        //SPEMat operator%(SPEMat A); 
        // Transpose functions
        PetscInt T(SPEMat &A); ///< A = Transpose(*this.mat) operation with initialization of A
        PetscInt T(); ///< Transpose the current mat
        // conjugate
        PetscInt H(SPEMat &A); ///< A = Hermitian Transpose(*this.mat) operation with initialization of A (tranpose and complex conjugate)
        PetscInt H(); ///< Hermitian Transpose the current mat
        PetscInt conj(); ///< elemenwise conjugate current matrix
        PetscInt print(); ///< print mat to screen using PETSC_VIEWER_STDOUT_WORLD

        ~SPEMat(); /// destructor to delete memory

    };
    SPEMat operator*(const PetscScalar a, const SPEMat A); ///< a*A operation to be equivalent to A*a
    SPEMat operator*(const SPEMat A, const PetscScalar a); ///< A*a operation to be equivalent to A*a
    SPEVec operator/(const SPEVec &b, const SPEMat &A); ///< Solve linear system, Ax=b using b/A notation
    SPEMat eye(const PetscInt n); ///< create, form, and return identity matrix of size n
    SPEMat diag(const SPEVec &diag); ///< set diagonal of matrix
    SPEMat kron(const SPEMat &A, const SPEMat &B); ///< set kronecker inner product of two matrices
}



#endif // SPEMAT_H
