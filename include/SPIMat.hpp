#ifndef SPIMAT_H
#define SPIMAT_H
#include <iostream>
#include <vector>
#include <petscksp.h>
#include <slepceps.h>
#include <slepcpep.h>
#include <string>
#include <tuple>
#include "SPIVec.hpp"

namespace SPI{
    template <class T>
        using Block2D = std::vector<std::vector<T>>;

    struct SPIMat{
        PetscInt rows;              ///< number of rows in mat
        PetscInt cols;              ///< number of columns in mat

        // Constructors
        SPIMat(std::string _name="SPIMat");                     // constructor with no arguments (no initialization)
        SPIMat(const SPIMat  &A, std::string _name="SPIMat");   // constructor using another SPIMat
        SPIMat(PetscInt rowscols, std::string _name="SPIMat");  // constructor with one arguement to make square matrix
        SPIMat(PetscInt rowsm, PetscInt colsn, std::string _name="SPIMat"); // constructor of rectangular matrix

        Mat mat;                ///< petsc Mat data
        PetscErrorCode ierr;    ///< ierr for various routines and operators

        // flags
        PetscBool flag_init=PETSC_FALSE;    ///< flag if it has been initialized
        std::string name;            ///< Matrix name
        
        PetscInt Init(PetscInt m,PetscInt n, std::string name="SPIMat"); // initialize the matrix of size m by n
        SPIMat& set(PetscInt m, PetscInt n,const PetscScalar v); // set a scalar value at position row m and column n
        SPIMat& add(PetscInt m, PetscInt n,const PetscScalar v); // add a scalar value at position row m and column n
        // () operators
        PetscScalar operator()(PetscInt m, PetscInt n, PetscBool global=PETSC_FALSE);             // get local value at row m, column n
        SPIMat& operator()(PetscInt m, PetscInt n,const PetscScalar v);  // set operator the same as set function
        SPIMat& operator()(PetscInt m, PetscInt n,const double v);  // set operator the same as set function
        SPIMat& operator()(PetscInt m, PetscInt n,const int v);     // set operator the same as set function
        SPIMat& operator()(PetscInt m, PetscInt n,const SPIMat &Asub, InsertMode addv=ADD_VALUES);   // set submatrix into matrix at row m, col n
        SPIMat& operator()();                                       // assemble the matrix
        // +- operators
        SPIMat& operator+=(const SPIMat &X); // MatAXPY,  Y = 1.*X + Y operation
        SPIMat& axpy(const PetscScalar a, const SPIMat &X); // MatAXPY function call to add a*X to the current mat
        SPIMat operator+(const SPIMat &X); // Y + X operation
        SPIMat& operator-=(const SPIMat &X); // Y = -1.*X + Y operation
        SPIMat operator-(const SPIMat &X); // Y - X operation
        SPIMat operator-() const; // -X operation
        // * operators
        SPIMat operator*(const PetscScalar a); // Y*a operation
        SPIMat operator*(const double a); // Y*a operation
        SPIVec operator*(const SPIVec &x); // A*x operation to return a vector
        SPIMat& operator*=(const PetscScalar a); // Y = Y*a operation
        SPIMat& operator*=(const double a); // Y = Y*a operation
        SPIMat& operator/=(const PetscScalar a); // Y = Y/a operation
        SPIMat operator/(const PetscScalar a); // Z = Y/a operation
        SPIMat operator*(const SPIMat &A); // Y*A operation
        // = operator
        SPIMat& operator=(const SPIMat &A); // Y=X with initialization of Y using MatConvert
        // overload % for element wise multiplication
        //SPIMat operator%(SPIMat A); 
        // Transpose functions
        PetscInt T(SPIMat &A); // A = Transpose(*this.mat) operation with initialization of A
        SPIMat& T(); // Transpose the current mat
        // conjugate
        PetscInt H(SPIMat &A); // A = Hermitian Transpose(*this.mat) operation with initialization of A (tranpose and complex conjugate)
        SPIMat& H(); // Hermitian Transpose the current mat
        SPIMat& conj(); // elemenwise conjugate current matrix
        SPIMat& real(); // take the real part of the matrix (alters current matrix)
        SPIVec diag(); // get diagonal of matrix
        SPIMat& zero_row(const PetscInt row); // zero a row
        SPIMat& eye_row(const PetscInt row); // zero a row and put 1 in the diagonal entry
        SPIMat& zero_row_full(const PetscInt row); // zero a row
        SPIMat& zero_rows(std::vector<PetscInt> rows); // zero every row
        SPIMat& eye_rows(std::vector<PetscInt> rows); // zero every row
        PetscInt print(); // print mat to screen using PETSC_VIEWER_STDOUT_WORLD

        ~SPIMat(); // destructor to delete memory

    };
    SPIMat operator*(const PetscScalar a, const SPIMat A); // a*A operation to be equivalent to A*a
    SPIMat operator*(const SPIMat A, const PetscScalar a); // A*a operation to be equivalent to A*a
    SPIVec operator/(const SPIVec &b, const SPIMat &A); // Solve linear system, Ax=b using b/A notation
    //SPIVec solve(const SPIVec &b, const SPIMat &A); // Solve linear system, Ax=b using solve(A,b) notation
    SPIVec solve(const SPIMat &A, const SPIVec &b); // Solve linear system, Ax=b using solve(A,b) notation
    SPIMat eye(const PetscInt n); // create, form, and return identity matrix of size n
    SPIMat zeros(const PetscInt m,const PetscInt n); // create, form, and return zero matrix of size mxn
    SPIMat diag(const SPIVec &diag,const PetscInt k=0); // set diagonal of matrix
    SPIMat kron(const SPIMat &A, const SPIMat &B); // set kronecker inner product of two matrices
    std::tuple<PetscScalar,SPIVec,SPIVec> eig(const SPIMat &A, const SPIMat &B, const PetscScalar target,const PetscReal tol=-1,const PetscInt max_iter=-1); // solve general eigenvalue problem of Ax = kBx and return a tuple of tie(PetscScalar alpha, SPIVec eig_vector)
    std::tuple<PetscScalar,SPIVec, SPIVec> eig_init(const SPIMat &A, const SPIMat &B, const PetscScalar target,const SPIVec &ql, const SPIVec &qr, PetscReal tol=-1,const PetscInt max_iter=-1); // solve general eigenvalue problem of Ax = kBx and return a tuple of tie(PetscScalar alpha, SPIVec eig_vector) using initial subspace from q
    std::tuple<PetscScalar,SPIVec> polyeig(const std::vector<SPIMat> &As, const PetscScalar target,const PetscReal tol=-1.,const PetscInt max_iter=-1); // solve general polynomial eigenvalue problem of (A0 + A1x + A2x^2 ...) = 0 and return a tuple of tie(PetscScalar alpha, SPIVec eig_vector)
    std::tuple<PetscScalar,SPIVec> polyeig_init(const std::vector<SPIMat> &As, const PetscScalar target, const SPIVec &qr,  const PetscReal tol=-1.,const PetscInt max_iter=-1); // solve general polynomial eigenvalue problem of (A0 + A1x + A2x^2 ...) = 0 and return a tuple of tie(PetscScalar alpha, SPIVec eig_vector) using initial subspace from q
    //SPIMat block(const SPIMat Blocks[], const PetscInt rows,const PetscInt cols); // set block matrices using an input array of size rows*cols.  Fills rows first
    //SPIMat block(const std::vector<std::vector<SPIMat>> Blocks); // set block matrices using an input array of size rows*cols.  Fills rows first
    SPIMat block(const Block2D<SPIMat> Blocks); // set block matrices using an input array of size rows*cols.  Fills rows first
    PetscInt save(const SPIMat &A, const std::string filename); // save matrix to filename to binary format
    PetscInt save(const std::vector<SPIMat> &A, const std::string filename); // save matrix to filename to binary format
    PetscInt load(SPIMat &A, const std::string filename); // load matrix to filename from binary format
    PetscInt load(std::vector<SPIMat> &A, const std::string filename); // load matrix to filename from binary format
    PetscInt draw(const SPIMat &A); // draw nonzero structure and wait at command line input
}



#endif // SPIMAT_H
