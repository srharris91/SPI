#ifndef SPEVEC_H
#define SPEVEC_H
#include <iostream>
#include <petscksp.h>
#include <string>

namespace SPE{
    struct SPEVec{
        PetscInt rows;              ///< number of rows in mat

        // Constructors
        SPEVec(std::string _name="SPEVec");                   ///< constructor with no arguments (no initialization)
        SPEVec(const SPEVec& x, std::string _name="SPEVec");      ///< constructor using another SPEVec
        SPEVec(PetscInt rows, std::string _name="SPEVec");  ///< constructor with one arguement to make vector of length rows

        Vec vec;                ///< petsc Mat data
        PetscErrorCode ierr;    ///< ierr for various routines and operators

        // flags
        PetscBool flag_init=PETSC_FALSE;    ///< flag if it has been initialized
        std::string name;            ///< Vec name
        
        PetscInt Init(PetscInt _rows, std::string name="SPEVec"); ///< initialize the matrix of size _rows
        PetscInt set(const PetscInt _row, const PetscScalar v); ///< set a scalar value at position row 
        PetscInt add(PetscInt _row, const PetscScalar v); ///< add a scalar value at position row 
        // () operators
        PetscScalar operator()(PetscInt _row);     ///< get local value at row
        PetscInt operator()(PetscInt _row, const PetscScalar v);  ///< set operator the same as set function
        PetscInt operator()(PetscInt _row, const double v);  ///< set operator the same as set function
        PetscInt operator()(PetscInt _row, const int v);  ///< set operator the same as set function
        PetscInt operator()();                                      ///< assemble the vector
        // +- operators
        SPEVec& operator+=(const SPEVec &X); ///< VecAXPY,  Y = 1.*X + Y operation
        SPEVec operator+(const SPEVec &X); ///< Y + X operation
        PetscInt axpy(const PetscScalar a, const SPEVec &X); ///< VecAXPY function call to add a*X to the current vec
        SPEVec& operator-=(const SPEVec &X); ///< Y = -1.*X + Y operation
        SPEVec operator-(const SPEVec &X); ///< Y - X operation
        // * operators
        SPEVec operator*(const PetscScalar a); ///< Y*a operation
        SPEVec operator*(const double a); ///< Y*a operation
        SPEVec& operator*=(const PetscScalar a); ///< Y = Y*a operation
        SPEVec operator*(const SPEVec &A); ///< Y*A operation
        // / operators
        SPEVec operator/(const PetscScalar a); ///< Y*a operation
        SPEVec operator/(const double a); ///< Y*a operation
        SPEVec& operator/=(const PetscScalar a); ///< Y = Y*a operation
        // = operator
        SPEVec& operator=(const SPEVec &A); ///< Y=X with initialization of Y using MatConvert
        // overload % for element wise multiplication
        //SPEVec operator%(SPEVec A); 
        // Transpose functions
        // conjugate
        SPEVec& conj(); ///< elemenwise conjugate current vector
        PetscScalar max(); ///< return maximum value of vector
        PetscInt print(); ///< print mat to screen using PETSC_VIEWER_STDOUT_WORLD

        ~SPEVec(); /// destructor to delete memory

    };
    SPEVec operator*(const PetscScalar a, SPEVec &A); ///< a*A operation to be equivalent to A*a
}



#endif // SPEVEC_H
