#ifndef SPEVEC_H
#define SPEVEC_H
#include <iostream>
#include <petscksp.h>
#include <string>
#include <fstream>

namespace SPE{
    struct SPEVec{
        PetscInt rows;              ///< number of rows in vec

        // Constructors
        SPEVec( std::string _name="SPEVec" );// constructor with no arguments (no initialization)
        SPEVec( const SPEVec& A, std::string _name="SPEVec"); // constructor using another SPEVec
        SPEVec( PetscInt rows,   std::string _name="SPEVec" );  // constructor with one arguement to make vector of length rows

        Vec vec;                ///< petsc Vec data
        PetscErrorCode ierr;    ///< ierr for various routines and operators

        // flags
        PetscBool flag_init=PETSC_FALSE;    ///< flag if it has been initialized
        std::string name;            ///< Vec name (important for hdf5 i/o)
        
        PetscInt Init( PetscInt _rows, std::string name="SPEVec"); // initialize the vector of size _rows
        PetscInt set(const PetscInt _row, const PetscScalar v); // set a scalar value at position row 
        PetscInt add(PetscInt _row, const PetscScalar v); // add a scalar value at position row 
        // () operators
        PetscScalar operator()(PetscInt _row, PetscBool global=PETSC_FALSE);     // get value at row
        PetscInt operator()(PetscInt _row, const PetscScalar v);  // set operator the same as set function
        PetscInt operator()(PetscInt _row, const double v);  // set operator the same as set function
        PetscInt operator()(PetscInt _row, const int v);  // set operator the same as set function
        SPEVec& operator()();                             // assemble the vector
        // +- operators
        SPEVec& operator+=(const SPEVec &X); // VecAXPY,  Y = 1.*X + Y operation
        SPEVec& axpy(const PetscScalar a, const SPEVec &X); // VecAXPY function call to add a*X to the current vec
        SPEVec operator+(const SPEVec &X); // Y + X operation
        SPEVec operator+(const PetscScalar a); // Y + a operation
        SPEVec operator-(const PetscScalar a); // Y - a operation
        SPEVec& operator-=(const SPEVec &X); // Y = -1.*X + Y operation
        SPEVec operator-(const SPEVec &X); // Y - X operation
        // * operators
        SPEVec operator*(const PetscScalar a); // Y*a operation
        SPEVec operator*(const double a); // Y*a operation
        SPEVec& operator*=(const PetscScalar a); // Y = Y*a operation
        SPEVec operator*(const SPEVec &X); // Y*X operation
        // / operators
        SPEVec operator/(const PetscScalar a); // Y/a operation
        SPEVec operator/(const double a); // Y*a operation
        SPEVec& operator/=(const PetscScalar a); // Y = Y/a operation
        // = operator
        SPEVec& operator=(const SPEVec &X); // Y=X with initialization of Y
        // == vecequal operator
        PetscBool operator==(const SPEVec &x2); // check if this==x2
        // overload % for element wise multiplication
        //SPEVec operator%(SPEVec A); 
        // Transpose functions
        // conjugate
        SPEVec& conj(); // elemenwise conjugate current vector
        PetscScalar max(); // return maximum value of vector
        PetscInt print(); // print vec to screen using PETSC_VIEWER_STDOUT_WORLD

        ~SPEVec(); // destructor to delete memory

    };
    SPEVec operator*(const PetscScalar a, const SPEVec &A); // a*A operation to be equivalent to A*a
    SPEVec operator+(const PetscScalar a, const SPEVec &A); // a+A operation to be equivalent to A+a
    SPEVec operator-(const PetscScalar a, const SPEVec &A); // a-A operation to be equivalent to A-a
    PetscInt save(const SPEVec &A, std::string filename); // save A to hdf5 to filename as variable A.name
    PetscInt load( SPEVec &A, const std::string filename); // load A from hdf5 filename using variable A.name, be sure it has the right size first before loading
    SPEVec ones(const PetscInt rows); // return a vector of size rows full of ones
    SPEVec zeros(const PetscInt rows); // return a vector of size rows full of zeros
    SPEVec conj(const SPEVec &A); // return the conjugate vector
    SPEVec linspace(const PetscScalar begin, const PetscScalar end, const PetscInt rows); // return linspace of number of rows equally spaced points between begin and end
    template <class T>
    SPEVec _Function_on_each_element(T (*f)(T const&), const SPEVec &A); // take the function of element in vector
    SPEVec sin(const SPEVec &A); // take the sin of element
    SPEVec cos(const SPEVec &A); // take the cos of element
    SPEVec tan(const SPEVec &A); // take the tan of element
    SPEVec exp(const SPEVec &A); // take the exp of element
    SPEVec log(const SPEVec &A); // take the log (natural log) of element
    SPEVec log10(const SPEVec &A); // take the log10 of element
    SPEVec sinh(const SPEVec &A); // take the sinh of element
    SPEVec cosh(const SPEVec &A); // take the cosh of element
    SPEVec tanh(const SPEVec &A); // take the tanh of element
    SPEVec asin(const SPEVec &A); // take the asin of element
    SPEVec acos(const SPEVec &A); // take the acos of element
    SPEVec atan(const SPEVec &A); // take the atan of element
    SPEVec asinh(const SPEVec &A); // take the asinh of element
    SPEVec acosh(const SPEVec &A); // take the acosh of element
    SPEVec atanh(const SPEVec &A); // take the atanh of element
    // template for scalar function on each element
    template <class T>
    SPEVec _Function_on_each_element(T (*f)(T const&,T const&), const SPEVec &A, SPEVec &B); // take the function of elements in vectors e.g. (*f)(A(i),B(i))
    SPEVec pow(const SPEVec &A, SPEVec &B); // take the pow(A(i),B(i)) of element
    SPEVec abs(const SPEVec &A);
    PetscScalar sum(SPEVec x); // sum of vector
    PetscReal L2(SPEVec x1, const SPEVec x2, NormType type=NORM_2);
    PetscReal L2(const SPEVec x1, NormType type=NORM_2);
    SPEVec diff(SPEVec x1); // diff of vector (will be size x1.rows-1)
    PetscScalar trapz(const SPEVec y); // trapezoidal integration of y with x coordinates  \int y dx
    PetscScalar trapz(const SPEVec y, const SPEVec x); // trapezoidal integration of y with x coordinates  \int y dx
}



#endif // SPEVEC_H
