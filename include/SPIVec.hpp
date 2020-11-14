#ifndef SPIVEC_H
#define SPIVEC_H
#include <iostream>
#include <petscksp.h>
#include <string>
#include <fstream>

namespace SPI{
    struct SPIVec{
        PetscInt rows;              ///< number of rows in vec

        // Constructors
        SPIVec( std::string _name="SPIVec" );// constructor with no arguments (no initialization)
        SPIVec( const SPIVec& A, std::string _name="SPIVec"); // constructor using another SPIVec
        SPIVec( PetscInt rows,   std::string _name="SPIVec" );  // constructor with one arguement to make vector of length rows

        Vec vec;                ///< petsc Vec data
        PetscErrorCode ierr;    ///< ierr for various routines and operators

        // flags
        PetscBool flag_init=PETSC_FALSE;    ///< flag if it has been initialized
        std::string name;            ///< Vec name (important for hdf5 i/o)
        
        PetscInt Init( PetscInt _rows, std::string name="SPIVec"); // initialize the vector of size _rows
        PetscInt set(const PetscInt _row, const PetscScalar v); // set a scalar value at position row 
        PetscInt set(const PetscScalar v); // set a scalar value at all positions
        PetscInt add(PetscInt _row, const PetscScalar v); // add a scalar value at position row 
        // get info operators
        PetscInt size(); // get the size of the vector using VecGetSize
        // () operators
        PetscScalar operator()(PetscInt _row, PetscBool global=PETSC_FALSE);     // get value at row
        PetscInt operator()(PetscInt _row, const PetscScalar v);  // set operator the same as set function
        PetscInt operator()(PetscInt _row, const double v);  // set operator the same as set function
        PetscInt operator()(PetscInt _row, const int v);  // set operator the same as set function
        SPIVec& operator()();                             // assemble the vector
        // +- operators
        SPIVec& operator+=(const SPIVec &X); // VecAXPY,  Y = 1.*X + Y operation
        SPIVec& axpy(const PetscScalar a, const SPIVec &X); // VecAXPY function call to add a*X to the current vec
        SPIVec operator+(const SPIVec &X); // Y + X operation
        SPIVec operator+(const PetscScalar a); // Y + a operation
        SPIVec operator-(const PetscScalar a); // Y - a operation
        SPIVec& operator-=(const SPIVec &X); // Y = -1.*X + Y operation
        SPIVec operator-(const SPIVec &X); // Y - X operation
        // * operators
        SPIVec operator*(const PetscScalar a); // Y*a operation
        SPIVec operator*(const double a); // Y*a operation
        SPIVec& operator*=(const PetscScalar a); // Y = Y*a operation
        SPIVec operator*(const SPIVec &X); // Y*X operation
        // / operators
        SPIVec operator/(const PetscScalar a); // Y/a operation
        SPIVec operator/(const double a); // Y*a operation
        SPIVec& operator/=(const PetscScalar a); // Y = Y/a operation
        // ^ operators
        SPIVec operator^(const PetscScalar p); // Y^p operation
        SPIVec operator^(const double p); // Y^p operation
        SPIVec operator^(SPIVec p); // elementwise Y^p operation
        // = operator
        SPIVec& operator=(const SPIVec &X); // Y=X with initialization of Y
        // == vecequal operator
        PetscBool operator==(const SPIVec &x2); // check if this==x2
        // overload % for element wise multiplication
        //SPIVec operator%(SPIVec A); 
        // Transpose functions
        // conjugate
        SPIVec& conj(); // elemenwise conjugate current vector
        PetscScalar max(); // return maximum value of vector
        SPIVec& real(); // real part of current vector
        SPIVec& imag(); // real part of current vector
        PetscScalar dot(SPIVec y); // take inner dot product (this,y) or y^H this, where H is the complex conjugate transpose
        PetscInt print(); // print vec to screen using PETSC_VIEWER_STDOUT_WORLD

        ~SPIVec(); // destructor to delete memory

    };
    SPIVec operator*(const PetscScalar a, const SPIVec &A); // a*A operation to be equivalent to A*a
    SPIVec operator+(const PetscScalar a, const SPIVec &A); // a+A operation to be equivalent to A+a
    SPIVec operator-(const PetscScalar a, const SPIVec &A); // a-A operation to be equivalent to A-a
    PetscInt save(const SPIVec &A, std::string filename); // save A to hdf5 to filename as variable A.name
    PetscInt load( SPIVec &A, const std::string filename); // load A from hdf5 filename using variable A.name, be sure it has the right size first before loading
    SPIVec ones(const PetscInt rows); // return a vector of size rows full of ones
    SPIVec zeros(const PetscInt rows); // return a vector of size rows full of zeros
    SPIVec conj(const SPIVec &A); // return the conjugate vector
    SPIVec real(const SPIVec &A); // return real part of vector
    SPIVec imag(const SPIVec &A); // return imaginary part of vector
    SPIVec linspace(const PetscScalar begin, const PetscScalar end, const PetscInt rows); // return linspace of number of rows equally spaced points between begin and end
    SPIVec arange(const PetscScalar begin, const PetscScalar end, const PetscScalar stepsize=1); // return a range of number of rows equally spaced points between begin and end of step stepsize
    template <class T>
    SPIVec _Function_on_each_element(T (*f)(T const&), const SPIVec &A); // take the function of element in vector
    SPIVec sin(const SPIVec &A); // take the sin of element
    SPIVec cos(const SPIVec &A); // take the cos of element
    SPIVec tan(const SPIVec &A); // take the tan of element
    SPIVec exp(const SPIVec &A); // take the exp of element
    SPIVec log(const SPIVec &A); // take the log (natural log) of element
    SPIVec log10(const SPIVec &A); // take the log10 of element
    SPIVec sinh(const SPIVec &A); // take the sinh of element
    SPIVec cosh(const SPIVec &A); // take the cosh of element
    SPIVec tanh(const SPIVec &A); // take the tanh of element
    SPIVec asin(const SPIVec &A); // take the asin of element
    SPIVec acos(const SPIVec &A); // take the acos of element
    SPIVec atan(const SPIVec &A); // take the atan of element
    SPIVec asinh(const SPIVec &A); // take the asinh of element
    SPIVec acosh(const SPIVec &A); // take the acosh of element
    SPIVec atanh(const SPIVec &A); // take the atanh of element
    // template for scalar function on each element
    template <class T>
    SPIVec _Function_on_each_element(T (*f)(T const&,T const&), const SPIVec &A, SPIVec &B); // take the function of elements in vectors e.g. (*f)(A(i),B(i))
    SPIVec pow(const SPIVec &A, SPIVec &B); // take the pow(A(i),B(i)) of element
    SPIVec pow(const SPIVec &A, PetscScalar b); // take the pow(A(i),b) of element
    SPIVec abs(const SPIVec &A); // take absolute value of vector
    PetscScalar sum(SPIVec x); // sum of vector
    PetscScalar dot(SPIVec x, SPIVec y); // inner dot product of the two vectors (i.e. y^H x)
    PetscReal L2(SPIVec x1, const SPIVec x2, NormType type=NORM_2);
    PetscReal L2(const SPIVec x1, NormType type=NORM_2);
    SPIVec diff(SPIVec x1); // diff of vector (will be size x1.rows-1)
    PetscScalar trapz(const SPIVec y); // trapezoidal integration of y with x coordinates  \int y dx
    PetscScalar trapz(const SPIVec y, const SPIVec x); // trapezoidal integration of y with x coordinates  \int y dx
}



#endif // SPIVEC_H
