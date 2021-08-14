#include "SPIMat.hpp"
#include "SPIprint.hpp"
#include <petscviewerhdf5.h>
#include <petscmath.h>
#include <slepcsvd.h>
// use 1 to use the GPU and 0 or anything else to only use CPU and MPI.  Be sure this matches what is in SPIVec.cpp
#define USE_GPU 0

namespace SPI{

    // constructors
    /** \brief constructor with no arguments (no initialization) */
    SPIMat::SPIMat(
            std::string _name   ///< [in] name of SPIMat
            ){name=_name; }
    /** constructor using another SPIMat */
    SPIMat::SPIMat(
            const SPIMat &A,    ///< [in] another SPIMat to copy into this new SPIMat
            std::string _name   ///< [in] name of SPIMat
            ){
        name=_name; 
        (*this) = A;
    }
    /** constructor using a vector of column vectors*/
    SPIMat::SPIMat(
            const std::vector<SPIVec> &A,
            std::string _name
            ){
        PetscInt m=A[0].rows;
        PetscInt n=A.size();
        Init(m,n,_name); // initialize the matrix
        for(PetscInt j=0; j<n; ++j){
            set_col(j,A[j]);
        }
        (*this)();
    }
    /** constructor with one arguement to make square matrix */
    SPIMat::SPIMat(
            PetscInt rowscols,  ///< [in] number of rows and columns to make the square matrix
            std::string _name   ///< [in] name of SPIMat
            ){
        Init(rowscols,rowscols,_name);
    }
    /** \brief constructor of rectangular matrix */
    SPIMat::SPIMat(
            PetscInt rowsm,     ///< [in] number of rows in matrix
            PetscInt colsn,     ///< [in] number of columns in matrix
            std::string _name   ///< [in] name of SPIMat
            ){
        Init(rowsm,colsn,_name);
    }

    // Initialize matrix
    /** initialize the matrix of size m by n \return 0 if successful */
    PetscInt SPIMat::Init(
            PetscInt m,         ///< [in] number of rows
            PetscInt n,         ///< [in] number of columns
            std::string _name   ///< [in] name of SPIMat
            ){
        name=_name;
        rows=m;
        cols=n;
        ierr = MatCreate(PETSC_COMM_WORLD,&mat);CHKERRXX(ierr);
        ierr = MatSetSizes(mat,PETSC_DECIDE,PETSC_DECIDE,m,n);CHKERRXX(ierr);
        //ierr = MatSetFromOptions(mat);CHKERRXX(ierr);
#if UUSE_GPU == 1
        ierr = MatSetType(mat,MATMPIAIJCUSPARSE);CHKERRXX(ierr);
#else
        ierr = MatSetType(mat,MATMPIAIJ);CHKERRXX(ierr);
#endif
        ierr = MatSetUp(mat);CHKERRXX(ierr);
        flag_init=PETSC_TRUE;
        return 0;
    }

    /** set a scalar value at position row m and column n \return current SPIMat after setting value */
    SPIMat& SPIMat::set(
            PetscInt m,         ///< [in] row to insert scalar
            PetscInt n,         ///< [in] column to insert scalar
            const PetscScalar v ///< [in] scalar to insert in matrix
            ){
        ierr = MatSetValue(mat,m,n,v,INSERT_VALUES);CHKERRXX(ierr);
        return (*this);
    }
    /** set a column vector into a matrix */
    SPIMat& SPIMat::set_col(
            const PetscInt col, ///< [in] 
            const SPIVec &v     ///< [in] 
            ){
        for(PetscInt i=0; i<rows; ++i){
            (*this)(i,col,v(i,PETSC_TRUE));
        }
        //(*this)();
        return (*this);
    }
    /** \brief add a scalar value at position row m and column n \return current matrix after adding the value*/
    SPIMat& SPIMat::add(
            PetscInt m,         ///< [in] row to add scalar
            PetscInt n,         ///< [in] column to add scalar
            const PetscScalar v ///< [in] scalar to add in matrix
            ){
        ierr = MatSetValue(mat,m,n,v,ADD_VALUES);CHKERRXX(ierr);
        return (*this);
    }
    /** \brief get local value at row m, column n \return scalar at specified location */
    PetscScalar SPIMat::operator()(
            PetscInt m,         ///< [in] row to get scalar
            PetscInt n,         ///< [in] column to get scalar
            PetscBool global    ///< [in] whether to broadcast value to all processors or not (default is false)
            )const{
        PetscErrorCode ierr2;
        PetscScalar v,v_global=0.;
        PetscInt low,high;
        ierr2 = MatGetOwnershipRange(mat,&low, &high);CHKERRXX(ierr2);
        if ((low<=m) && (m<high)){
            ierr2 = MatGetValues(mat,1,&m, 1,&n, &v);
        }
        if (global){ // if broadcast to all processors
            MPIU_Allreduce(&v,&v_global,1,MPIU_SCALAR,MPIU_SUM,PETSC_COMM_WORLD);
        }
        else{
            v_global=v; // return local value
        }
        return v_global;
    }

    // overloaded operators, get
    /** \brief get local value at row m, column n \return scalar at specified location */
    PetscScalar SPIMat::operator()(
            PetscInt m,         ///< [in] row to get scalar
            PetscInt n,         ///< [in] column to get scalar
            PetscBool global    ///< [in] whether to broadcast value to all processors or not (default is false)
            ){
        PetscScalar v,v_global=0.;
        PetscInt low,high;
        ierr = MatGetOwnershipRange(mat,&low, &high);CHKERRXX(ierr);
        if ((low<=m) && (m<high)){
            ierr = MatGetValues(mat,1,&m, 1,&n, &v);
        }
        if (global){ // if broadcast to all processors
            MPIU_Allreduce(&v,&v_global,1,MPIU_SCALAR,MPIU_SUM,PETSC_COMM_WORLD);
        }
        else{
            v_global=v; // return local value
        }
        return v_global;
    }
    // overloaded operator, set
    /** \brief set operator the same as set function \return current matrix after setting value */
    SPIMat& SPIMat::operator()(
            PetscInt m,         ///< [in] row to set scalar
            PetscInt n,         ///< [in] column to set scalar
            const PetscScalar v ///< [in] scalar to set in matrix
            ){
        ierr = MatSetValue(mat,m,n,v,INSERT_VALUES);CHKERRXX(ierr);
        //(*this)(); // assemble after every insertion
        return (*this);
    }
    /** \brief set operator the same as set function \return current matrix after setting value */
    SPIMat& SPIMat::operator()(
            PetscInt m,         ///< [in] row to set scalar
            PetscInt n,         ///< [in] column to set scalar
            const double v      ///< [in] scalar to set in matrix
            ){
        //ierr = (*this)(m,n,(PetscScalar)v);CHKERRXX(ierr);
        return (*this)(m,n,(PetscScalar)(v+0.0*PETSC_i));
    }
    /** \brief set operator the same as set function \return current matrix after setting value */
    SPIMat& SPIMat::operator()(
            PetscInt m,         ///< [in] row to set scalar
            PetscInt n,         ///< [in] column to set scalar
            const int v         ///< [in] scalar to set in matrix
            ){
        //ierr = (*this)(m,n,(PetscScalar)v);CHKERRXX(ierr);
        return (*this)(m,n,(PetscScalar)((double)v+0.0*PETSC_i));
    }

    // overloaded operator, set
    /** \brief set submatrix into matrix at row m, col n \return current matrix after setting value */
    SPIMat& SPIMat::operator()(
            PetscInt m,         ///< [in] row to set submatrix
            PetscInt n,         ///< [in] column to set submatrix
            const SPIMat &Asub, ///< [in] submatrix to set in matrix
            InsertMode addv     ///< [in] default to ADD_VALUES in submatrix, can do INSERT_VALUES instead
            ){
        //InsertMode addv = INSERT_VALUES;
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
            ierr = MatGetRow(Asub.mat,i,&ncols,&cols,&vals);CHKERRXX(ierr);
            PetscInt offcols[ncols];
            PetscScalar avals[ncols];
            for (PetscInt j=0; j<ncols; j++) {
                offcols[j] = cols[j]+coloffset;
                avals[j] = vals[j];
            }
            //printScalar(vals,ncols);
            PetscInt rowoffseti = i+rowoffset;
            ierr = MatSetValues(mat,1,&rowoffseti,ncols,offcols,avals,addv);CHKERRXX(ierr);

            MatRestoreRow(Asub.mat,i,&ncols,&cols,&vals);
        }
        //(*this)();//assemble
        return (*this);
    }
    // overloaded operator, assemble
    /** \brief assmelbe the matrix \return the current matrix */
    SPIMat& SPIMat::operator()(){
        ierr = MatAssemblyBegin(mat,MAT_FINAL_ASSEMBLY);CHKERRXX(ierr);
        ierr = MatAssemblyEnd(mat,MAT_FINAL_ASSEMBLY);CHKERRXX(ierr);
        return (*this);
    }
    // overloaded operator, MatAXPY
    /** \brief MatAXPY,  Y = 1.*X + Y operation \return current matrix after operation */
    SPIMat& SPIMat::operator+=(
            const SPIMat &X ///< [in] X in Y+=X operation
            ){
        ierr = MatAXPY(this->mat,1.,X.mat,DIFFERENT_NONZERO_PATTERN);CHKERRXX(ierr);
        return *this;
    }
    /** \brief MatAXPY function call to add a*X to the current mat \return current matrix after operation */
    SPIMat& SPIMat::axpy(
            const PetscScalar a,    ///< [in] scalar a in Y = a*X + Y operation
            const SPIMat &X         ///< [in] matrix X in Y = a*X + Y operation
            ){
        ierr = MatAXPY(this->mat,a,X.mat,DIFFERENT_NONZERO_PATTERN);CHKERRXX(ierr);
        return (*this);
    }
    // overloaded operator, MatAXPY
    /** \brief Y + X operation \return new matrix after operation */
    SPIMat SPIMat::operator+(
            const SPIMat &X ///< [in] X in Y+X operation
            ){
        SPIMat A;
        A=*this;
        ierr = MatAXPY(A.mat,1.,X.mat,DIFFERENT_NONZERO_PATTERN);CHKERRXX(ierr);
#if USE_GPU == 1
        ierr = MatSetType(A.mat,MATMPIAIJCUSPARSE);CHKERRXX(ierr);
#else
        ierr = MatSetType(A.mat,MATMPIAIJ);CHKERRXX(ierr);
#endif
        return A;
    }
    // overloaded operator, MatAXPY
    /** \brief Y = -1.*X + Y operation \return Y after operation */
    SPIMat& SPIMat::operator-=(
            const SPIMat &X ///< [in] X in Y = -1.*X + Y operation
            ){
        ierr = MatAXPY(this->mat,-1.,X.mat,DIFFERENT_NONZERO_PATTERN);CHKERRXX(ierr);
        return *this;
    }
    // overloaded operator, MatAXPY
    /** \brief Y - X operation \return new matrix after operation */
    SPIMat SPIMat::operator-(
            const SPIMat &X     ///< [in] X in Y-X operation
            ){
        SPIMat A;
        A=*this;
        ierr = MatAXPY(A.mat,-1.,X.mat,DIFFERENT_NONZERO_PATTERN);CHKERRXX(ierr);
#if USE_GPU == 1
        ierr = MatSetType(A.mat,MATMPIAIJCUSPARSE);CHKERRXX(ierr);
#else
        ierr = MatSetType(A.mat,MATMPIAIJ);CHKERRXX(ierr);
#endif
        return A;
    }
    /** \brief -X operation \return new matrix after operation */
    SPIMat SPIMat::operator-() const {
        SPIMat A;
        A=-1.*(*this);
        return A;
    }
    // overload operator, scale with scalar
    /** \brief Y*a operation \return new matrix after operation */
    SPIMat SPIMat::operator*(
            const PetscScalar a     ///< [in] scalar
            ){
        SPIMat A;
        A=*this;
        ierr = MatScale(A.mat,a);CHKERRXX(ierr);
        return A;
    }
    /** \brief Y*a operation \return new matrix after operation */
    SPIMat SPIMat::operator*(
            const double a          ///< [in] scalar
            ){
        SPIMat A;
        A=*this;
        ierr = MatScale(A.mat,a);CHKERRXX(ierr);
        return A;
    }
    /** \brief A*x operation to return a vector \return new vector after operation */
    SPIVec SPIMat::operator*(
            const SPIVec &x         ///< [in] x in A*x matrix vector multiplication
            ){
        //SPIVec b(x.rows); // only works for square matrices
        SPIVec b(rows); // works for non-square matrices
        ierr = MatMult(mat,x.vec,b.vec);CHKERRXX(ierr);
        return b;
    }
    // overload operator, scale with scalar
    /** \brief Y = Y*a operation \return Y after operation */
    SPIMat& SPIMat::operator*=(
            const double a     ///< [in] scalar in Y*a operation
            ){
        ierr = MatScale(this->mat,(PetscScalar)(a+0.*PETSC_i));CHKERRXX(ierr);
        return *this;
    }
    /** \brief Y = Y*a operation \return Y after operation */
    SPIMat& SPIMat::operator*=(
            const PetscScalar a     ///< [in] scalar in Y*a operation
            ){
        ierr = MatScale(this->mat,a);CHKERRXX(ierr);
        return *this;
    }
    /** \brief Y = Y/a operation \return Y after operation */
    SPIMat& SPIMat::operator/=(
            const PetscScalar a     ///< [in] scalar in Y*a operation
            ){
        ierr = MatScale(this->mat,1./a);CHKERRXX(ierr);
        return *this;
    }
    /** \brief Z = Y/a operation \return Z after operation */
    SPIMat SPIMat::operator/(
            const PetscScalar a     ///< [in] scalar in Y*a operation
            ){
        return (1./a)*(*this);
    }
    /** \brief Z = Y/A operation \return Z after operation */
    SPIMat SPIMat::operator/(
            const SPIMat &A     ///< [in] A in Y/A operation
            ){
        SPIMat Z(A.rows,A.cols);
        for(PetscInt i=0; i<Z.rows; ++i){
            for(PetscInt j=0; j<Z.cols; ++j){
                Z(i,j,((*this)(i,j,PETSC_TRUE))/A(i,j,PETSC_TRUE));
            }
        }
        Z();
        return Z;
    }
    // overload operator, matrix multiply
    /** \brief Y*A operation \return new matrix after matrix matrix multiply */
    SPIMat SPIMat::operator*(
            const SPIMat &A         ///< [in] A matrix in Y*A operation
            ){
        SPIMat C;
        C.rows=rows;
        C.cols=A.cols;
        ierr = MatMatMult(mat,A.mat,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&C.mat); CHKERRXX(ierr);
        C.flag_init=PETSC_TRUE;
        //ierr = MatSetType(C.mat,MATMPIAIJ);CHKERRXX(ierr);
        return C;
    }
    // overload operator, copy and initialize
    /** \brief Y=X with initialization of Y using MatConvert \return Y after operation */
    SPIMat& SPIMat::operator=(
            const SPIMat &A         ///< [in] A in Y=A operation
            ){
        if(flag_init){
            this->~SPIMat();
            //ierr = MatCopy(A.mat,mat,DIFFERENT_NONZERO_PATTERN);CHKERRXX(ierr);
        }
        //else
        rows=A.rows;
        cols=A.cols;
        ierr = MatConvert(A.mat,MATSAME,MAT_INITIAL_MATRIX,&mat);CHKERRXX(ierr);
#if USE_GPU == 1
        ierr = MatSetType(mat,MATMPIAIJCUSPARSE);CHKERRXX(ierr);
#else
        ierr = MatSetType(mat,MATMPIAIJ);CHKERRXX(ierr);
#endif
        flag_init=PETSC_TRUE;
        //
        return *this;
    }
    // overload % for element wise multiplication
    //SPIMat operator%(SPIMat A)
    //return *this;
    //]     
    /** \brief A = Transpose(*this.mat) operation with initialization of A \return 0 if solves */
    PetscInt SPIMat::T(
            SPIMat &A       ///< [out] transpose of current matrix
            ){
        //ierr = MatTranspose(mat,MAT_INITIAL_MATRIX,&A.mat);CHKERRXX(ierr);
        //A();
        //A.Init(cols,rows);
        ierr = MatTranspose(mat,MAT_INITIAL_MATRIX,&A.mat);CHKERRXX(ierr);
        A.rows=this->cols;
        A.cols=this->rows;
        A.flag_init=PETSC_TRUE;
        return 0;
    }
    /** \brief Transpose the current mat \return current matrix after transpose */
    SPIMat& SPIMat::T(){
        ierr = MatTranspose(mat,MAT_INPLACE_MATRIX,&mat);CHKERRXX(ierr);
        return (*this);
        //SPIMat T1;
        //ierr = MatCreateTranspose(mat,&T1.mat);CHKERRXX(ierr);
        //SPIMat T1;
        //ierr = MatTranspose(mat,MAT_INITIAL_MATRIX,&T1.mat);CHKERRXX(ierr);
        //return T1;
    }
    /** \brief A = Hermitian Transpose(*this.mat) operation with initialization of A (tranpose and complex conjugate) \return current matrix without hermitian transpose */
    PetscInt SPIMat::H(
            SPIMat &A       ///< [out] hermitian transpose of current matrix saved in new initialized matrix
            ){ // A = Hermitian Transpose(*this.mat) operation with initialization of A (tranpose and complex conjugate)
        ierr = MatHermitianTranspose(mat,MAT_INITIAL_MATRIX,&A.mat);CHKERRXX(ierr);
        return 0;
    }
    /** \brief Hermitian Transpose the current mat \return current matrix after transpose */
    SPIMat& SPIMat::H(){ // Hermitian Transpose the current mat
        ierr = MatHermitianTranspose(mat,MAT_INPLACE_MATRIX,&mat);CHKERRXX(ierr);
        return (*this);
    }
    /** \brief Hermitian Transpose multiplication with the current mat \return Vector y after y=A^H*x operation */
    SPIVec SPIMat::H(
            const SPIVec &x   ///< [in] x vector to multiply by the Hermitian Transpose.  y = A^H*x
            ){ // Hermitian Transpose the current mat
        SPIVec y(cols);
        ierr = MatMultHermitianTranspose(mat,x.vec,y.vec);CHKERRXX(ierr);
        return y;
    }
    /** \brief elemenwise conjugate current matrix \return current matrix after conjugate of each element */
    SPIMat& SPIMat::conj(){
        ierr = MatConjugate(mat);CHKERRXX(ierr);
        return (*this);
    }
    /** \brief take the real part of the vector \returns the vector after taking the real part of it */
    SPIMat& SPIMat::real(){
        ierr = MatRealPart(mat); CHKERRXX(ierr);
        return (*this);
    }
    /** \brief get diagonal of matrix \return vector of current diagonal */
    SPIVec SPIMat::diag(){ // get diagonal of matrix
        SPIVec d(rows);
        ierr = MatGetDiagonal(mat,d.vec); CHKERRXX(ierr);
        return d;
    }
    /** \brief set a row to zero \return matrix after setting the row to zero */
    SPIMat& SPIMat::zero_row(
            const PetscInt row ///< [in] which row to zero out of the matrix
            ){
        ierr = MatZeroRows(mat,1,&row,0.,0,0); CHKERRXX(ierr);
        return (*this);
    }
    /** \brief set a row to zero and set 1 in diagonal entry \return matrix after setting the row to zero and setting 1 in diagonal */
    SPIMat& SPIMat::eye_row(
            const PetscInt row ///< [in] which row to zero out of the matrix
            ){
        ierr = MatZeroRows(mat,1,&row,1.,0,0); CHKERRXX(ierr);
        return (*this);
    }
    /** \brief set a row to zero using dense format \return matrix after setting the row to zero */
    SPIMat& SPIMat::zero_row_full(
            const PetscInt row ///< [in] which row to zero out of the matrix
            ){
        for(PetscInt j=0; j<cols; j++){
            (*this)(row,j,0.0);
        }
        //ierr = MatZeroRows(mat,1,&row,0.,0,0); CHKERRXX(ierr);
        return (*this);
    }
    /** \brief set rows to zero \return matrix after setting the rows to zero */
    SPIMat& SPIMat::zero_rows(
            std::vector<PetscInt> rows ///< [in] which rows to zero out of the matrix
            ){
        ierr = MatZeroRows(mat,rows.size(),rows.data(),0.,0,0); CHKERRXX(ierr);
        return (*this);
    }
    /** \brief set rows to zero and set main diagonal to 1 \return matrix after setting the rows to zero and main diagonals to 1 */
    SPIMat& SPIMat::eye_rows(
            std::vector<PetscInt> rows ///< [in] which rows to zero out of the matrix
            ){
        ierr = MatZeroRows(mat,rows.size(),rows.data(),1.,0,0); CHKERRXX(ierr);
        return (*this);
    }
    /** \brief get column vector \return SPIVec of column vector */
    SPIVec SPIMat::col(
            const PetscInt i
            ){
        SPIVec yy(rows);
        yy.ierr = MatGetColumnVector(mat,yy.vec,i); CHKERRXX(yy.ierr);
        return yy;
    }
    // print matrix to screen
    /** \brief print mat to screen using PETSC_VIEWER_STDOUT_WORLD \return 0 if successful */
    PetscInt SPIMat::print(){
        (*this)();
        PetscPrintf(PETSC_COMM_WORLD,("\n---------------- "+name+"---start------\n").c_str());
        SPI::printf("shape = "+std::to_string(this->rows)+" x "+std::to_string(this->cols));
        ierr = MatView(mat,PETSC_VIEWER_STDOUT_WORLD);CHKERRXX(ierr);
        PetscPrintf(PETSC_COMM_WORLD,("---------------- "+name+"---done-------\n\n").c_str());
        return 0;
    }

    /** \brief destructor to delete memory */
    SPIMat::~SPIMat(){
        if(flag_init){
            flag_init=PETSC_FALSE;
            ierr = MatDestroy(&mat);CHKERRXX(ierr);
        }
    }

    // overload operator, scale with scalar
    /** \brief a*A operation to be equivalent to A*a \return new matrix of a*A */
    SPIMat operator*(
            const PetscScalar a,    ///< [in] scalar a in a*A
            const SPIMat A          ///< [in] matrix A in a*A
            ){
        SPIMat B;
        B=A;
        B.ierr = MatScale(B.mat,a);CHKERRXX(B.ierr);
        return B;
    }
    /** \brief A*a operation to be equivalent to A*a \return new matrix of A*a operation */
    SPIMat operator*(const SPIMat A, const PetscScalar a){
        SPIMat B;
        B=A;
        B.ierr = MatScale(B.mat,a);CHKERRXX(B.ierr);
        return B;
    }
    // overload operator, Linear System solve Ax=b
    /** \brief Solve linear system, Ax=b using b/A notation \return x vector in Ax=b solved linear system */
    SPIVec operator/(
            const SPIVec &b,    ///< [in] b in Ax=b
            const SPIMat &A     ///< [in] A in Ax=b
            ){
        SPIVec x;
        x.rows=b.rows;
        KSP    ksp;  // Linear solver context
        PetscErrorCode ierr;
        ierr = VecDuplicate(b.vec,&x.vec);CHKERRXX(ierr);

        // Create the linear solver and set various options
        // Create linear solver context
        ierr = KSPCreate(PETSC_COMM_WORLD,&ksp);CHKERRXX(ierr);
        // Set operators. Here the matrix that defines the linear system
        // also serves as the preconditioning matrix.
        ierr = KSPSetOperators(ksp,A.mat,A.mat);CHKERRXX(ierr);
        // Set runtime options, e.g.,
        // -ksp_type <type> -pc_type <type> -ksp_monitor -ksp_rtol <rtol> -ksp_type <type> -pc_type <type>
        //ierr = KSPSetFromOptions(ksp);CHKERRXX(ierr);
        //ierr = KSPSetType(ksp,KSPPREONLY);CHKERRXX(ierr);

        PC pc;
        ierr = KSPGetPC(ksp,&pc);CHKERRXX(ierr);
        ierr = PCSetType(pc,PCLU);CHKERRXX(ierr);
        ierr = KSPSetType(ksp,KSPPREONLY);CHKERRXX(ierr);
        //ierr = PCSetOperators(pc,A.mat,A.mat); CHKERRXX(ierr);


        // Solve the linear system
        ierr = KSPSolve(ksp,b.vec,x.vec);CHKERRXX(ierr);
        x.flag_init = PETSC_TRUE;

        // output iterations
        //PetscInt its;
        //ierr = KSPGetIterationNumber(ksp,&its);CHKERRXX(ierr);
        //PetscPrintf(PETSC_COMM_WORLD,"KSP Solved in %D iterations \n",its);
        // Free work space.  All PETSc objects should be destroyed when they
        //set_Vec(x);
        // are no longer needed.
        ierr = KSPDestroy(&ksp);CHKERRXX(ierr);
        return x;
    }
    /** \brief Y=a^A operation \return Y matrix in Y=a^A */
    SPIMat operator^(
            const PetscScalar a,   ///< [in] a in Y=a^A
            const SPIMat &A         ///< [in] A in Y=a^A
            ){
        SPIMat Y(A.rows,A.cols);
        for(PetscInt i=0; i<Y.rows; ++i){
            for(PetscInt j=0; j<Y.cols; ++j){
                Y(i,j,std::pow<PetscReal>(a,A(i,j,PETSC_TRUE)));
                //std::cout<<"A("<<i<<","<<j<<") = "<<A(i,j,PETSC_TRUE)<<std::endl;
                //std::cout<<"Y("<<i<<","<<j<<") = "<<Y(i,j,PETSC_TRUE)<<std::endl;
                //Y();
            }
        }
        Y();
        //Y.real();
        return Y;
    }
    /** \brief Solve linear system, Ax=b using solve(A,b) notation \return x vector in Ax=b solved linear system */
    SPIVec solve(
            const SPIMat &A,    ///< [in] A in Ax=b
            const SPIVec &b     ///< [in] b in Ax=b
            ){
        return b/A;
    }
    /** \brief Solve linear system A*Ainv=B using MatMatSolve \return Ainv matrix */
    SPIMat inv(
            const SPIMat &A     ///< [in] A in A A^-1 = B
            ){
        if(0){ // only works on single processor and with Dense matrices
            SPIMat Acopy;
            Acopy.rows=A.rows;
            Acopy.cols=A.cols;
            Acopy.name=A.name;
            Mat Amat;
            Acopy.ierr = MatCreate(PETSC_COMM_WORLD,&Amat);CHKERRXX(Acopy.ierr);
            Acopy.ierr = MatSetSizes(Amat,PETSC_DECIDE,PETSC_DECIDE,A.rows,A.cols);CHKERRXX(Acopy.ierr);
            Acopy.ierr = MatSetType(Amat,MATDENSE);CHKERRXX(Acopy.ierr);
            Acopy.ierr = MatSetUp(Amat);CHKERRXX(Acopy.ierr);
            Acopy.flag_init=PETSC_TRUE;
            Acopy.mat = Amat;
            Acopy(0,0,A);
            //Acopy.ierr = MatSetType(Acopy.mat,MATMPIDENSE);CHKERRXX(Acopy.ierr);
            Acopy();CHKERRXX(Acopy.ierr);
            Acopy.ierr = MatLUFactor(Acopy.mat,NULL,NULL,NULL); CHKERRXX(Acopy.ierr);
            SPIMat B;
            Mat Bmat;
            B.ierr = MatCreate(PETSC_COMM_WORLD,&Bmat);CHKERRXX(B.ierr);
            B.ierr = MatSetSizes(Bmat,PETSC_DECIDE,PETSC_DECIDE,A.rows,A.cols);CHKERRXX(B.ierr);
            B.ierr = MatSetType(Bmat,MATDENSE);CHKERRXX(B.ierr);
            B.ierr = MatSetUp(Bmat);CHKERRXX(B.ierr);
            B.mat = Bmat;
            B(0,0,eye(A.rows));
            SPIMat Ainv;
            Mat Ainvmat;
            Ainv.ierr = MatCreate(PETSC_COMM_WORLD,&Ainvmat);CHKERRXX(Ainv.ierr);
            Ainv.ierr = MatSetSizes(Ainvmat,PETSC_DECIDE,PETSC_DECIDE,A.rows,A.cols);CHKERRXX(Ainv.ierr);
            Ainv.ierr = MatSetType(Ainvmat,MATDENSE);CHKERRXX(Ainv.ierr);
            Ainv.ierr = MatSetUp(Ainvmat);CHKERRXX(Ainv.ierr);
            Ainv.mat = Ainvmat;
            Ainv.rows=A.rows;
            Ainv.cols=A.cols;
            Ainv.name = A.name + "^-1";
            //Ainv.ierr = MatSetType(Ainv.mat,MATSEQDENSE);CHKERRXX(Ainv.ierr);
            //Ainv();
            Ainv.ierr = MatMatSolve(Acopy.mat,B.mat,Ainv.mat); CHKERRXX(Ainv.ierr);
            return Ainv;
        }
        else{ // solve using ksp multiple times
            SPIMat B(eye(A.rows));
            SPIVec b(B.col(0));
            SPIVec x;
            x.rows=b.rows;
            KSP    ksp;  // Linear solver context
            PetscErrorCode ierr;
            ierr = VecDuplicate(b.vec,&x.vec);CHKERRXX(ierr);
            x.flag_init=PETSC_TRUE;

            // Create the linear solver and set various options
            // Create linear solver context
            ierr = KSPCreate(PETSC_COMM_WORLD,&ksp);CHKERRXX(ierr);
            // Set operators. Here the matrix that defines the linear system
            // also serves as the preconditioning matrix.
            ierr = KSPSetOperators(ksp,A.mat,A.mat);CHKERRXX(ierr);
            // Set runtime options, e.g.,
            // -ksp_type <type> -pc_type <type> -ksp_monitor -ksp_rtol <rtol> -ksp_type <type> -pc_type <type>
            //ierr = KSPSetFromOptions(ksp);CHKERRXX(ierr);
            //ierr = KSPSetType(ksp,KSPPREONLY);CHKERRXX(ierr);

            PC pc;
            ierr = KSPGetPC(ksp,&pc);CHKERRXX(ierr);
            ierr = PCSetType(pc,PCLU);CHKERRXX(ierr);
            ierr = KSPSetType(ksp,KSPPREONLY);CHKERRXX(ierr);
            //ierr = PCSetOperators(pc,A.mat,A.mat); CHKERRXX(ierr);


            // Solve the linear system multiple times to get Ainv
            SPIMat Ainv(A.rows,A.cols,A.name+"^-1");
            for(PetscInt j=0; j<A.cols; ++j){
                ierr = KSPSolve(ksp,B.col(j).vec,x.vec);CHKERRXX(ierr);
                for(PetscInt i=0; i<A.cols; ++i){
                    PetscScalar value=x(i,PETSC_TRUE);
                    if(value != 0.) Ainv(i,j,value); // try to preserve zero structure
                }
            }
            Ainv();

            // output iterations
            //PetscInt its;
            //ierr = KSPGetIterationNumber(ksp,&its);CHKERRXX(ierr);
            //PetscPrintf(PETSC_COMM_WORLD,"KSP Solved in %D iterations \n",its);
            // Free work space.  All PETSc objects should be destroyed when they
            //set_Vec(x);
            // are no longer needed.
            ierr = KSPDestroy(&ksp);CHKERRXX(ierr);
            return Ainv;
        }
    }
    // identity matrix formation
    /** \brief create, form, and return identity matrix of size n \return identity matrix of size nxn */
    SPIMat eye(
            const PetscInt n        ///< [in] n size of square identity matrix
            ){
        SPIMat I(n,"I");
        SPIVec one(n);
        I.ierr = VecSet(one.vec,1.);CHKERRXX(I.ierr);
        I.ierr = MatDiagonalSet(I.mat,one.vec,INSERT_VALUES);CHKERRXX(I.ierr);

        return I;
    }
    /** \brief create, form, and return zeros matrix of size mxn \return zero matrix of size mxn */
    SPIMat zeros(
            const PetscInt m,       ///< [in] m size of zero matrix
            const PetscInt n        ///< [in] n size of zero matrix
            ){
        SPIMat O(m,n,"zero");
        O();
        //O.ierr = MatZeroEntries(O.mat); CHKERRXX(O.ierr);
        // set main diagonal to zero... but these didn't quite work.  
        //SPIVec zero(m);
        //O.ierr = VecSet(zero.vec,1.);CHKERRXX(O.ierr);
        //zero *= 0.;
        //O.ierr = MatDiagonalSet(O.mat,zero.vec,INSERT_VALUES);CHKERRXX(O.ierr);
        //O();
        //for(PetscInt i=0; i<m; i++)
        //O(i,i,1.0);
        //
        return O;
    }
    // diagonal matrix
    /** \brief set diagonal of matrix \return new matrix with  a diagonal vector set as the main diagonal */
    SPIMat diag(
            const SPIVec &d,     ///< [in] diagonal vector to set along main diagonal
            const PetscInt k     ///< [in] diagonal to set, k=0 is main diagonal, k=1 is offset one in positive direction
            ){ // set diagonal of matrix
        if(k==0){
            SPIMat A(d.rows);
            A.ierr = MatDiagonalSet(A.mat,d.vec,INSERT_VALUES);CHKERRXX(A.ierr);
            return A;
        }
        else if(k>0){
            PetscInt r0=d.rows, r1=k;
            PetscInt c0=k,      c1=d.rows;
            SPIMat A00(zeros(r0,c0)), A01(r0,c1),
                   A10(zeros(r1,c0)), A11(zeros(r1,c1));
            A01.ierr = MatDiagonalSet(A01.mat,d.vec,INSERT_VALUES);CHKERRXX(A01.ierr);
            //SPIMat A(SPI::block(A00,A01,
            //A10,A11));
            SPIMat A(r0+r1,c0+c1);
            A(0,c0,A01);
            A();
            return A;

        }
        else if(k<0){
            PetscInt r0=-k,     r1=d.rows;
            PetscInt c0=d.rows, c1=-k;
            SPIMat A00(zeros(r0,c0)); CHKERRXX(A00.ierr);
            SPIMat A01(zeros(r0,c1)); CHKERRXX(A01.ierr);
            SPIMat A10(r1,c0);        CHKERRXX(A10.ierr);
            SPIMat A11(zeros(r1,c1)); CHKERRXX(A11.ierr);
            //CHKERRXX(A00.ierr);
            //CHKERRXX(A10.ierr);
            //CHKERRXX(A01.ierr);
            //CHKERRXX(A11.ierr);
            A10.ierr = MatDiagonalSet(A10.mat,d.vec,INSERT_VALUES);CHKERRXX(A10.ierr);
            //A00();CHKERRXX(A00.ierr);
            //A01();CHKERRXX(A01.ierr);
            //A10();CHKERRXX(A10.ierr);
            //A11();CHKERRXX(A11.ierr);
            SPIMat A(SPI::block({{A00,A01},
                        {A10,A11}}));
            A();
            return A();

        }
        else{
            exit(0);
        }
    }

    // kron inner product
    /** \brief set kronecker inner product of two matrices \return kronecker inner product of the two matrices */
    SPIMat kron(
            const SPIMat &A,    ///< [in] A in A kron B operation
            const SPIMat &B     ///< [in] B in A kron B operation
            ){
        PetscErrorCode ierr;

        // get A,B information
        PetscInt m,n,p,q;
        MatGetSize(A.mat,&m,&n);
        MatGetSize(B.mat,&p,&q);

        // assume square matrices A and B, so we can use set_Mat for the square submatrices
        PetscInt na=m, nb=p,nc;
        nc=m*p;

        // init C
        SPIMat C(nc);

        // kron function C=kron(A,B)
        PetscInt ncols;
        const PetscInt *cols;
        const PetscScalar *vals;
        PetscInt Isubstart,Isubend;
        ierr = MatGetOwnershipRange(A.mat,&Isubstart,&Isubend);CHKERRXX(ierr);
        for (PetscInt rowi=0; rowi<na; rowi++){
            //PetscPrintf(PETSC_COMM_WORLD,"kron rowi=%i of %i\n",rowi,m);
            bool onprocessor=(Isubstart<=rowi) and (rowi<Isubend);
            if(onprocessor){
                // extract row of one A
                ierr = MatGetRow(A.mat,rowi,&ncols,&cols,&vals);CHKERRXX(ierr); // extract the one row of A if owned by processor
            }
            else{
                ncols=0;
            }
            PetscInt ncols2=0;
            MPI_Allreduce(&ncols,&ncols2,1,MPIU_INT,MPI_SUM,PETSC_COMM_WORLD);

            // set global vals2 array
            PetscScalar vals2[ncols2];
            PetscInt cols2[ncols2];
            PetscScalar *vals_temp=new PetscScalar[ncols2] ();// new array and set to 0 with ()
            PetscInt *cols_temp=new PetscInt[ncols2] ();
            if(onprocessor){ 
                for(PetscInt i=0; i<ncols2; i++){
                    vals_temp[i]=vals[i];
                    cols_temp[i]=cols[i];
                }

            }
            MPIU_Allreduce(vals_temp,vals2,ncols2,MPIU_SCALAR,MPIU_SUM,PETSC_COMM_WORLD);
            MPI_Allreduce(cols_temp,cols2,ncols2,MPIU_INT,MPI_SUM,PETSC_COMM_WORLD);

            // every processor calls set_Mat for each col in row
            for(PetscInt i=0; i<ncols2; i++){
                // ierr = set_Mat(vals2[i],B,nb,C,nc,rowi*nb,cols2[i]*nb,INSERT_VALUES);CHKERRXX(ierr);
                C(rowi*nb,cols2[i]*nb,B*vals2[i],INSERT_VALUES);
            }

            if(onprocessor){
                // restore row
                ierr = MatRestoreRow(A.mat,rowi,&ncols,&cols,&vals); CHKERRXX(ierr);
            }
            delete[] vals_temp;
            delete[] cols_temp;

        }
        C();
        return C;
    }

    /** \brief solve SVD problem of A=U*E*V^H for skinny A matrix \return tuple of sigma,u,v   std::tie(sigma, u, v) = svd(A) */
    std::tuple<std::vector<PetscReal>, std::vector<SPIVec>, std::vector<SPIVec>> svd(
            const SPIMat &A         ///< [in] A to perform SVD decomposition A=U*E*V^H problem
            ){
        // set m to be the number of rows and k to be number of columns
        PetscInt m=A.rows; // number of rows
        PetscInt n=A.cols; // number of columns
        SVD     svd;    // SVD context
        //PetscReal sigmai;
        //SPIVec ui(m,"u");
        //SPIVec vi(n,"v");
        std::vector<PetscReal> sigma(n);
        std::vector<SPIVec> u(n);
        std::vector<SPIVec> v(n);
        PetscInt nconv;
        //PetscReal error;

        SVDCreate(PETSC_COMM_WORLD, &svd);
        SVDSetOperator(svd, A.mat);
        //SVDSetProblemType(svd, SVD_STANDARD);
        SVDSetFromOptions(svd);
        SVDSolve(svd);
        //MatCreateVecs(A.mat,&vi.vec,&ui.vec);
        //ui.flag_init=PETSC_TRUE;
        //vi.flag_init=PETSC_TRUE;
        SVDGetConverged(svd, &nconv);
        //std::cout<<"n="<<n<<" nconv = "<<nconv<<std::endl;
        for(PetscInt j=0; j<nconv; j++){
            //std::cout<<"j = "<<j<<std::endl;
            MatCreateVecs(A.mat,&v[j].vec,&u[j].vec);
            SVDGetSingularTriplet(svd,j,&(sigma[j]),u[j].vec,v[j].vec);
            u[j].flag_init=PETSC_TRUE;
            //u[j].name = "u";
            u[j].rows = m;

            v[j].flag_init=PETSC_TRUE;
            //v[j].name = "v";
            v[j].rows = n;
            //SVDGetSingularTriplet(svd,j,&sigmai,ui.vec,vi.vec);
            //SVDComputeError(svd,j,SVD_ERROR_RELATIVE,&error);
            //sigma.push_back(sigmai);
            //u.push_back(ui);
            //v.push_back(vi);
            //sigma[j] = sigmai;
            //u[j] = ui;
            //v[j] = vi;
            //std::cout<<"j = "<<j<<" sigma = "<<sigma[j]<<" error = "<<error<<std::endl;
        }
        //std::cout<<"made it here"<<std::endl;
        SVDDestroy(&svd);
        //std::cout<<"made it here deallocate"<<std::endl;
        return std::make_tuple(sigma,u,v);
    }
    /** \brief solve least squares problem of A*x=y for skinny A matrix using SVD \return x */
    SPIVec lstsq(
            const SPIMat &A,    ///< [in] A matrix in A*x=y least squares problem
            SPIVec &y           ///< [in] y in A*x=y least squares problem
            ){
        PetscInt n=A.cols;
        std::vector<PetscReal> sigma;
        std::vector<SPIVec> u,v;
        SPIVec x(zeros(n));
        std::tie(sigma,u,v) = svd(A);
        for(PetscInt j=0; j<n; ++j){
            x += ((y.dot(u[j]))/sigma[j]) * v[j];
        }
        return x;
    }
    /** \brief solve least squares problem of A*x=y for skinny A matrix (or vectors of columns vectors) using SVD \return x */
    SPIVec lstsq(
            const std::vector<SPIVec> &A,   ///< [in] vectors of columns vectors making up A matrix
            SPIVec &y                       ///< [in] y in A*x=y least squares problem
            ){
        SPIMat S(A);
        return lstsq(S,y);
    }
    /** \brief solve general eigenvalue problem of Ax = kBx and return a tuple of tie(PetscScalar alpha, SPIVec eig_vector) \return tuple of eigenvalue and eigenvector closest to the target value e.g.  std::tie(alpha, eig_vector) = eig(A,B,0.1+0.4*PETSC_i) */
    std::tuple<PetscScalar, SPIVec, SPIVec> eig(
            const SPIMat &A,        ///< [in] A in Ax=kBx generalized eigenvalue problem
            const SPIMat &B,        ///< [in] B in Ax=kBx generalized eigenvalue problem
            const PetscScalar target,   ///< [in] target eigenvalue to solve for
            const PetscReal tol,    ///< [in] tolerance of eigenvalue solver
            const PetscInt max_iter ///< [in] maximum number of iterations
            ){
        //std::cout<<"target = "<<target<<std::endl;
        PetscInt rows=A.rows;
        EPS             eps;        /* eigenproblem solver context slepc */
        //ST              st;
        //EPSType         type;
        //KSP             ksp;        /* linear solver context petsc */
        PetscErrorCode  ierr;
        PetscScalar ki,alpha;
        SPIVec eigl_vec(rows),eigr_vec(rows);

        //PetscScalar kr_temp, ki_temp;
        PetscScalar kr_temp;

        // Create the eigenvalue solver and set various options
        // Create solver contexts
        ierr = EPSCreate(PETSC_COMM_WORLD,&eps);CHKERRXX(ierr);
        // Set operators. Here the matrix that defines the eigenvalue system
        // swap operators such that LHS matrix is singular, and RHS matrix can be inverted (for slepc)
        // this will make the Ax=lambda Bx into the problem Bx = (1./lambda) Ax, thus our eigenvalues are inverted
        ierr = EPSSetOperators(eps,A.mat,B.mat);CHKERRXX(ierr);
        //ierr = EPSSetOperators(eps,B,A);CHKERRXX(ierr);
        // Set runtime options, e.g.,
        // -
        ierr = EPSSetFromOptions(eps);CHKERRXX(ierr);
        //std::cout<<"After KSPSetFromOptions"<<std::endl;
        //
        // set convergence type
        EPSWhich which=EPS_TARGET_MAGNITUDE;
        PetscInt nev=1;
        EPSSetWhichEigenpairs(eps,which);
        EPSSetDimensions(eps,nev,PETSC_DEFAULT,PETSC_DEFAULT);
        if ((max_iter==-1) && (tol==-1.)){
            EPSSetTolerances(eps,PETSC_DEFAULT,PETSC_DEFAULT);
        }
        else if(tol==-1.){
            EPSSetTolerances(eps,PETSC_DEFAULT,max_iter);
        }
        else if(max_iter==-1){
            EPSSetTolerances(eps,tol,PETSC_DEFAULT);
        }
        else{
            EPSSetTolerances(eps,tol,max_iter);
        }
        if (
                which==EPS_TARGET_REAL ||
                which==EPS_TARGET_IMAGINARY ||
                which==EPS_TARGET_MAGNITUDE){
            // PetscScalar target=0.-88.5*PETSC_i;
            EPSSetTarget(eps,target);
            //EPSSetTolerances(eps,1.E-8,100000);
        }


        // set spectral transform to shift-and-invert (seems to work best for LST_spatial)
        ST              st;
        EPSGetST(eps,&st);
        STSetType(st,STSINVERT);
        // Solve the system with left and right
        EPSSetTwoSided(eps,PETSC_TRUE);
        // Solve the system
        ierr = EPSSolve(eps);CHKERRXX(ierr);
        //std::cout<<"After KSPSolve"<<std::endl;

        // output iterations
        //PetscInt its, maxit, i;
        PetscInt nconv;
        //PetscReal error, tol, re, im;
        //PetscReal tol2;
        /* Optional: Get some information from the solver and display it */
        // ierr = EPSGetIterationNumber(eps,&its);CHKERRXX(ierr);
        // ierr = PetscPrintf(PETSC_COMM_WORLD," Number of iterations of the method: %D\n",its);CHKERRXX(ierr);
        // ierr = EPSGetType(eps,&type);CHKERRXX(ierr);
        // ierr = PetscPrintf(PETSC_COMM_WORLD," Solution method: %s\n\n",type);CHKERRXX(ierr);
        // ierr = EPSGetDimensions(eps,&nev,NULL,NULL);CHKERRXX(ierr);
        // ierr = PetscPrintf(PETSC_COMM_WORLD," Number of requested eigenvalues: %D\n",nev);CHKERRXX(ierr);
        // ierr = EPSGetTolerances(eps,&tol2,&maxit);CHKERRXX(ierr);
        // ierr = PetscPrintf(PETSC_COMM_WORLD," Stopping condition: tol=%.4g, maxit=%D\n",(double)tol2,maxit);CHKERRXX(ierr);

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
           Display solution and clean up
           - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
        /*
           Get number of converged approximate eigenpairs
           */
        ierr = EPSGetConverged(eps,&nconv);CHKERRXX(ierr);
        // ierr = PetscPrintf(PETSC_COMM_WORLD," Number of converged eigenpairs: %D\n\n",nconv);CHKERRXX(ierr);

        if (nconv>0) {
            /*
               Display eigenvalues and relative errors
               */
            // ierr = PetscPrintf(PETSC_COMM_WORLD,
            //         "      k                ||Ax-kx||/||kx||\n"
            //         "   ----------------- ------------------\n");CHKERRXX(ierr);

            // for (PetscInt i=0;i<nconv;i++) 
            //     /*
            //        Get converged eigenpairs: i-th eigenvalue is stored in kr (real part) and
            //        ki (imaginary part)
            //        */
            //     ierr = EPSGetEigenpair(eps,i,&alpha,&ki,eig_vec.vec,xi.vec);CHKERRXX(ierr);
            //     /*
            //        Compute the relative error associated to each eigenpair
            //        */
            //     PetscReal error, re, im;
            //     ierr = EPSComputeError(eps,i,EPS_ERROR_RELATIVE,&error);CHKERRXX(ierr);
            //     re = PetscRealPart(alpha);
            //     im = PetscImaginaryPart(alpha);
            //     if (im!=0.0) 
            //         ierr = PetscPrintf(PETSC_COMM_WORLD," (%9e+%9ei)  %12g\n",(double)re,(double)im,(double)error);CHKERRXX(ierr);
            //       else 
            //         ierr = PetscPrintf(PETSC_COMM_WORLD,"   %12e       %12g\n",(double)re,(double)error);CHKERRXX(ierr);
            //      
            //  
            // ierr = PetscPrintf(PETSC_COMM_WORLD,"\n");CHKERRXX(ierr);

            ierr = EPSGetEigenpair(eps,0,&alpha,PETSC_NULL,eigr_vec.vec,PETSC_NULL);CHKERRXX(ierr);
            ierr = EPSGetLeftEigenvector(eps,0,eigl_vec.vec,PETSC_NULL);CHKERRXX(ierr);
        }

        // PetscInt its;
        // ierr = EPSGetIterationNumber(eps,&its);CHKERRXX(ierr);
        // //ierr = PetscPrintf(PETSC_COMM_WORLD,"ksp iterations %D\n",its);CHKERRXX(ierr);
        // ierr = PetscPrintf(PETSC_COMM_WORLD,"EPS Solved in %D iterations \n",its); CHKERRXX(ierr);
        // Free work space.  All PETSc objects should be destroyed when they
        // are no longer needed.
        //set_Vec(x);
        ierr = EPSDestroy(&eps);CHKERRXX(ierr);
        //ierr = VecDestroy(&x);CHKERRXX(ierr);
        //ierr = VecDestroy(&b);CHKERRXX(ierr); 
        //ierr = MatDestroy(&A);CHKERRXX(ierr);
        //ierr = PetscFinalize();

        return std::make_tuple(alpha,eigl_vec,eigr_vec);
        //return std::make_tuple(alpha,alpha);
    }
    /** \brief solve general eigenvalue problem of Ax = kBx and return a tuple of tie(PetscScalar alpha, SPIVec eig_vector) \return tuple of eigenvalue and eigenvector closest to the target value e.g.  std::tie(alpha, eig_vector) = eig(A,B,0.1+0.4*PETSC_i) */
    std::tuple<PetscScalar, SPIVec> eig_right(
            const SPIMat &A,        ///< [in] A in Ax=kBx generalized eigenvalue problem
            const SPIMat &B,        ///< [in] B in Ax=kBx generalized eigenvalue problem
            const PetscScalar target,   ///< [in] target eigenvalue to solve for
            const PetscReal tol,    ///< [in] tolerance of eigenvalue solver
            const PetscInt max_iter ///< [in] maximum number of iterations
            ){
        //std::cout<<"target = "<<target<<std::endl;
        PetscInt rows=A.rows;
        EPS             eps;        /* eigenproblem solver context slepc */
        //ST              st;
        //EPSType         type;
        //KSP             ksp;        /* linear solver context petsc */
        PetscErrorCode  ierr;
        PetscScalar ki,alpha;
        SPIVec eigr_vec(rows);

        //PetscScalar kr_temp, ki_temp;
        PetscScalar kr_temp;

        // Create the eigenvalue solver and set various options
        // Create solver contexts
        ierr = EPSCreate(PETSC_COMM_WORLD,&eps);CHKERRXX(ierr);
        // Set operators. Here the matrix that defines the eigenvalue system
        // swap operators such that LHS matrix is singular, and RHS matrix can be inverted (for slepc)
        // this will make the Ax=lambda Bx into the problem Bx = (1./lambda) Ax, thus our eigenvalues are inverted
        ierr = EPSSetOperators(eps,A.mat,B.mat);CHKERRXX(ierr);
        //ierr = EPSSetOperators(eps,B,A);CHKERRXX(ierr);
        // Set runtime options, e.g.,
        // -
        ierr = EPSSetFromOptions(eps);CHKERRXX(ierr);
        //std::cout<<"After KSPSetFromOptions"<<std::endl;
        //
        // set convergence type
        EPSWhich which=EPS_TARGET_MAGNITUDE;
        PetscInt nev=1;
        EPSSetWhichEigenpairs(eps,which);
        EPSSetDimensions(eps,nev,PETSC_DEFAULT,PETSC_DEFAULT);
        if ((max_iter==-1) && (tol==-1.)){
            EPSSetTolerances(eps,PETSC_DEFAULT,PETSC_DEFAULT);
        }
        else if(tol==-1.){
            EPSSetTolerances(eps,PETSC_DEFAULT,max_iter);
        }
        else if(max_iter==-1){
            EPSSetTolerances(eps,tol,PETSC_DEFAULT);
        }
        else{
            EPSSetTolerances(eps,tol,max_iter);
        }
        if (
                which==EPS_TARGET_REAL ||
                which==EPS_TARGET_IMAGINARY ||
                which==EPS_TARGET_MAGNITUDE){
            // PetscScalar target=0.-88.5*PETSC_i;
            EPSSetTarget(eps,target);
            //EPSSetTolerances(eps,1.E-8,100000);
        }


        // set spectral transform to shift-and-invert (seems to work best for LST_spatial)
        ST              st;
        EPSGetST(eps,&st);
        STSetType(st,STSINVERT);
        // Solve the system with left and right
        //EPSSetTwoSided(eps,PETSC_TRUE);
        // Solve the system
        ierr = EPSSolve(eps);CHKERRXX(ierr);
        //std::cout<<"After KSPSolve"<<std::endl;

        // output iterations
        //PetscInt its, maxit, i;
        PetscInt nconv;
        //PetscReal error, tol, re, im;
        //PetscReal tol2;
        /* Optional: Get some information from the solver and display it */
        // ierr = EPSGetIterationNumber(eps,&its);CHKERRXX(ierr);
        // ierr = PetscPrintf(PETSC_COMM_WORLD," Number of iterations of the method: %D\n",its);CHKERRXX(ierr);
        // ierr = EPSGetType(eps,&type);CHKERRXX(ierr);
        // ierr = PetscPrintf(PETSC_COMM_WORLD," Solution method: %s\n\n",type);CHKERRXX(ierr);
        // ierr = EPSGetDimensions(eps,&nev,NULL,NULL);CHKERRXX(ierr);
        // ierr = PetscPrintf(PETSC_COMM_WORLD," Number of requested eigenvalues: %D\n",nev);CHKERRXX(ierr);
        // ierr = EPSGetTolerances(eps,&tol2,&maxit);CHKERRXX(ierr);
        // ierr = PetscPrintf(PETSC_COMM_WORLD," Stopping condition: tol=%.4g, maxit=%D\n",(double)tol2,maxit);CHKERRXX(ierr);

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
           Display solution and clean up
           - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
        /*
           Get number of converged approximate eigenpairs
           */
        ierr = EPSGetConverged(eps,&nconv);CHKERRXX(ierr);
        // ierr = PetscPrintf(PETSC_COMM_WORLD," Number of converged eigenpairs: %D\n\n",nconv);CHKERRXX(ierr);

        if (nconv>0) {
            /*
               Display eigenvalues and relative errors
               */
            // ierr = PetscPrintf(PETSC_COMM_WORLD,
            //         "      k                ||Ax-kx||/||kx||\n"
            //         "   ----------------- ------------------\n");CHKERRXX(ierr);

            // for (PetscInt i=0;i<nconv;i++) 
            //     /*
            //        Get converged eigenpairs: i-th eigenvalue is stored in kr (real part) and
            //        ki (imaginary part)
            //        */
            //     ierr = EPSGetEigenpair(eps,i,&alpha,&ki,eig_vec.vec,xi.vec);CHKERRXX(ierr);
            //     /*
            //        Compute the relative error associated to each eigenpair
            //        */
            //     PetscReal error, re, im;
            //     ierr = EPSComputeError(eps,i,EPS_ERROR_RELATIVE,&error);CHKERRXX(ierr);
            //     re = PetscRealPart(alpha);
            //     im = PetscImaginaryPart(alpha);
            //     if (im!=0.0) 
            //         ierr = PetscPrintf(PETSC_COMM_WORLD," (%9e+%9ei)  %12g\n",(double)re,(double)im,(double)error);CHKERRXX(ierr);
            //       else 
            //         ierr = PetscPrintf(PETSC_COMM_WORLD,"   %12e       %12g\n",(double)re,(double)error);CHKERRXX(ierr);
            //      
            //  
            // ierr = PetscPrintf(PETSC_COMM_WORLD,"\n");CHKERRXX(ierr);

            ierr = EPSGetEigenpair(eps,0,&alpha,PETSC_NULL,eigr_vec.vec,PETSC_NULL);CHKERRXX(ierr);
            eigr_vec.rows=rows;
            //ierr = EPSGetLeftEigenvector(eps,0,eigl_vec.vec,PETSC_NULL);CHKERRXX(ierr);
        }

        // PetscInt its;
        // ierr = EPSGetIterationNumber(eps,&its);CHKERRXX(ierr);
        // //ierr = PetscPrintf(PETSC_COMM_WORLD,"ksp iterations %D\n",its);CHKERRXX(ierr);
        // ierr = PetscPrintf(PETSC_COMM_WORLD,"EPS Solved in %D iterations \n",its); CHKERRXX(ierr);
        // Free work space.  All PETSc objects should be destroyed when they
        // are no longer needed.
        //set_Vec(x);
        ierr = EPSDestroy(&eps);CHKERRXX(ierr);
        //ierr = VecDestroy(&x);CHKERRXX(ierr);
        //ierr = VecDestroy(&b);CHKERRXX(ierr); 
        //ierr = MatDestroy(&A);CHKERRXX(ierr);
        //ierr = PetscFinalize();

        return std::make_tuple(alpha,eigr_vec);
        //return std::make_tuple(alpha,alpha);
    }
    /** \brief solve general eigenvalue problem of Ax = kBx and return a tuple of tie(PetscScalar alpha, SPIVec eig_vector) \return tuple of eigenvalue and eigenvector closest to the target value e.g.  std::tie(alpha, eig_vector) = eig(A,B,0.1+0.4*PETSC_i) using initial subspace defined by a vector */
    std::tuple<PetscScalar, SPIVec, SPIVec> eig_init(
            const SPIMat &A,        ///< [in] A in Ax=kBx generalized eigenvalue problem
            const SPIMat &B,        ///< [in] B in Ax=kBx generalized eigenvalue problem
            const PetscScalar target,   ///< [in] target eigenvalue to solve for
            const SPIVec &ql,        ///< [in] initial subspace vector for EPS solver (left eigenvector)
            const SPIVec &qr,        ///< [in] initial subspace vector for EPS solver (right eigenvector)
            PetscReal tol,    ///< [in] tolerance of eigenvalue solver
            const PetscInt max_iter ///< [in] maximum number of iterations
            ){
        //std::cout<<"target = "<<target<<std::endl;
        PetscInt rows=A.rows;
        EPS             eps;        /* eigenproblem solver context slepc */
        //ST              st;
        //EPSType         type;
        //KSP             ksp;        /* linear solver context petsc */
        PetscErrorCode  ierr;
        PetscScalar alpha;
        SPIVec eigl_vec(rows),eigr_vec(rows);

        PetscScalar kr_temp, ki_temp;

        // Create the eigenvalue solver and set various options
        // Create solver contexts
        ierr = EPSCreate(PETSC_COMM_WORLD,&eps);CHKERRXX(ierr);
        // Set operators. Here the matrix that defines the eigenvalue system
        // swap operators such that LHS matrix is singular, and RHS matrix can be inverted (for slepc)
        // this will make the Ax=lambda Bx into the problem Bx = (1./lambda) Ax, thus our eigenvalues are inverted
        ierr = EPSSetOperators(eps,A.mat,B.mat);CHKERRXX(ierr);
        //ierr = EPSSetOperators(eps,B,A);CHKERRXX(ierr);
        // Set runtime options, e.g.,
        // -
        ierr = EPSSetFromOptions(eps);CHKERRXX(ierr);
        //std::cout<<"After KSPSetFromOptions"<<std::endl;
        //
        // set convergence type
        EPSWhich which=EPS_TARGET_MAGNITUDE;
        PetscInt nev=1;
        EPSSetWhichEigenpairs(eps,which);
        EPSSetDimensions(eps,nev,PETSC_DEFAULT,PETSC_DEFAULT);
        if ((max_iter==-1) && (tol==-1.)){
            EPSSetTolerances(eps,PETSC_DEFAULT,PETSC_DEFAULT);
        }
        else if(tol==-1.){
            EPSSetTolerances(eps,PETSC_DEFAULT,max_iter);
        }
        else if(max_iter==-1){
            EPSSetTolerances(eps,tol,PETSC_DEFAULT);
        }
        else{
            EPSSetTolerances(eps,tol,max_iter);
        }
        if (
                which==EPS_TARGET_REAL ||
                which==EPS_TARGET_IMAGINARY ||
                which==EPS_TARGET_MAGNITUDE){
            // PetscScalar target=0.-88.5*PETSC_i;
            EPSSetTarget(eps,target);
            //EPSSetTolerances(eps,1.E-8,100000);
        }


        // set spectral transform to shift-and-invert (seems to work best for LST_spatial)
        ST              st;
        EPSGetST(eps,&st);
        STSetType(st,STSINVERT);
        // set initial guess
        std::vector<Vec> qrvec(1);
        qrvec[0] = qr.vec;
        ierr = EPSSetInitialSpace(eps,1,qrvec.data());CHKERRXX(ierr);
        std::vector<Vec> qlvec(1);
        qlvec[0] = ql.vec;
        ierr = EPSSetLeftInitialSpace(eps,1,qlvec.data());CHKERRXX(ierr); // TODO doesn't work using 3.12.1 version, but should work in 3.14 documentation

        // Solve the system with left and right
        EPSSetTwoSided(eps,PETSC_TRUE);
        ierr = EPSSolve(eps);CHKERRXX(ierr);
        //std::cout<<"After KSPSolve"<<std::endl;

        // output iterations
        //PetscInt its, maxit;
        PetscInt nconv;
        //PetscReal error, tol, re, im;
        /* Optional: Get some information from the solver and display it */
        //ierr = EPSGetIterationNumber(eps,&its);CHKERRXX(ierr);
        //ierr = PetscPrintf(PETSC_COMM_WORLD," Number of iterations of the method: %D\n",its);CHKERRXX(ierr);
        //ierr = EPSGetType(eps,&type);CHKERRXX(ierr);
        //ierr = PetscPrintf(PETSC_COMM_WORLD," Solution method: %s\n\n",type);CHKERRXX(ierr);
        //ierr = EPSGetDimensions(eps,&nev,NULL,NULL);CHKERRXX(ierr);
        //ierr = PetscPrintf(PETSC_COMM_WORLD," Number of requested eigenvalues: %D\n",nev);CHKERRXX(ierr);
        //ierr = EPSGetTolerances(eps,&tol,&maxit);CHKERRXX(ierr);
        //ierr = PetscPrintf(PETSC_COMM_WORLD," Stopping condition: tol=%.4g, maxit=%D\n",(double)tol,maxit);CHKERRXX(ierr);

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
           Display solution and clean up
           - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
        /*
           Get number of converged approximate eigenpairs
           */
        ierr = EPSGetConverged(eps,&nconv);CHKERRXX(ierr);
        //ierr = PetscPrintf(PETSC_COMM_WORLD," Number of converged eigenpairs: %D\n\n",nconv);CHKERRXX(ierr);

        if (nconv>0) {
            /*
               Display eigenvalues and relative errors
               */
            // ierr = PetscPrintf(PETSC_COMM_WORLD,
            //         "      k                ||Ax-kx||/||kx||\n"
            //         "   ----------------- ------------------\n");CHKERRXX(ierr);

            // for (PetscInt i=0;i<nconv;i++) 
            //     /*
            //        Get converged eigenpairs: i-th eigenvalue is stored in kr (real part) and
            //        ki (imaginary part)
            //        */
            //     ierr = EPSGetEigenpair(eps,i,&alpha,&ki,eigr_vec.vec,eigl_vec.vec);CHKERRXX(ierr);
            //     /*
            //        Compute the relative error associated to each eigenpair
            //        */
            //     PetscReal error, re, im;
            //     ierr = EPSComputeError(eps,i,EPS_ERROR_RELATIVE,&error);CHKERRXX(ierr);
            //     re = PetscRealPart(alpha);
            //     im = PetscImaginaryPart(alpha);
            //     if (im!=0.0) 
            //         ierr = PetscPrintf(PETSC_COMM_WORLD," (%9e+%9ei)  %12g\n",(double)re,(double)im,(double)error);CHKERRXX(ierr);
            //       else 
            //         ierr = PetscPrintf(PETSC_COMM_WORLD,"   %12e       %12g\n",(double)re,(double)error);CHKERRXX(ierr);
            //      
            //  
            // ierr = PetscPrintf(PETSC_COMM_WORLD,"\n");CHKERRXX(ierr);

            ierr = EPSGetEigenpair(eps,0,&alpha,PETSC_NULL,eigr_vec.vec,PETSC_NULL);CHKERRXX(ierr);
            ierr = EPSGetLeftEigenvector(eps,0,eigl_vec.vec,PETSC_NULL);CHKERRXX(ierr);
        }

        // PetscInt its;
        // ierr = EPSGetIterationNumber(eps,&its);CHKERRXX(ierr);
        // //ierr = PetscPrintf(PETSC_COMM_WORLD,"ksp iterations %D\n",its);CHKERRXX(ierr);
        // ierr = PetscPrintf(PETSC_COMM_WORLD,"EPS Solved in %D iterations \n",its); CHKERRXX(ierr);
        // Free work space.  All PETSc objects should be destroyed when they
        // are no longer needed.
        //set_Vec(x);
        ierr = EPSDestroy(&eps);CHKERRXX(ierr);
        //ierr = VecDestroy(&x);CHKERRXX(ierr);
        //ierr = VecDestroy(&b);CHKERRXX(ierr); 
        //ierr = MatDestroy(&A);CHKERRXX(ierr);
        //ierr = PetscFinalize();

        return std::make_tuple(alpha,eigl_vec,eigr_vec);
        //return std::make_tuple(alpha,alpha);
    }
    /** \brief solve general eigenvalue problem of Ax = kBx and return a tuple of tie(PetscScalar alpha, SPIVec eig_vector) \return tuple of eigenvalue and eigenvector closest to the target value e.g.  std::tie(alpha, eig_vector) = eig(A,B,0.1+0.4*PETSC_i) using initial subspace defined by a vector */
    std::tuple<PetscScalar, SPIVec> eig_init_right(
            const SPIMat &A,        ///< [in] A in Ax=kBx generalized eigenvalue problem
            const SPIMat &B,        ///< [in] B in Ax=kBx generalized eigenvalue problem
            const PetscScalar target,   ///< [in] target eigenvalue to solve for
            const SPIVec &qr,        ///< [in] initial subspace vector for EPS solver (right eigenvector)
            PetscReal tol,    ///< [in] tolerance of eigenvalue solver
            const PetscInt max_iter ///< [in] maximum number of iterations
            ){
        //std::cout<<"target = "<<target<<std::endl;
        PetscInt rows=A.rows;
        EPS             eps;        /* eigenproblem solver context slepc */
        //ST              st;
        //EPSType         type;
        //KSP             ksp;        /* linear solver context petsc */
        PetscErrorCode  ierr;
        PetscScalar alpha;
        SPIVec eigr_vec(rows);

        PetscScalar kr_temp, ki_temp;

        // Create the eigenvalue solver and set various options
        // Create solver contexts
        ierr = EPSCreate(PETSC_COMM_WORLD,&eps);CHKERRXX(ierr);
        // Set operators. Here the matrix that defines the eigenvalue system
        // swap operators such that LHS matrix is singular, and RHS matrix can be inverted (for slepc)
        // this will make the Ax=lambda Bx into the problem Bx = (1./lambda) Ax, thus our eigenvalues are inverted
        ierr = EPSSetOperators(eps,A.mat,B.mat);CHKERRXX(ierr);
        //ierr = EPSSetOperators(eps,B,A);CHKERRXX(ierr);
        // Set runtime options, e.g.,
        // -
        ierr = EPSSetFromOptions(eps);CHKERRXX(ierr);
        //std::cout<<"After KSPSetFromOptions"<<std::endl;
        //
        // set convergence type
        EPSWhich which=EPS_TARGET_MAGNITUDE;
        PetscInt nev=1;
        EPSSetWhichEigenpairs(eps,which);
        EPSSetDimensions(eps,nev,PETSC_DEFAULT,PETSC_DEFAULT);
        if ((max_iter==-1) && (tol==-1.)){
            EPSSetTolerances(eps,PETSC_DEFAULT,PETSC_DEFAULT);
        }
        else if(tol==-1.){
            EPSSetTolerances(eps,PETSC_DEFAULT,max_iter);
        }
        else if(max_iter==-1){
            EPSSetTolerances(eps,tol,PETSC_DEFAULT);
        }
        else{
            EPSSetTolerances(eps,tol,max_iter);
        }
        if (
                which==EPS_TARGET_REAL ||
                which==EPS_TARGET_IMAGINARY ||
                which==EPS_TARGET_MAGNITUDE){
            // PetscScalar target=0.-88.5*PETSC_i;
            EPSSetTarget(eps,target);
            //EPSSetTolerances(eps,1.E-8,100000);
        }


        // set spectral transform to shift-and-invert (seems to work best for LST_spatial)
        ST              st;
        EPSGetST(eps,&st);
        STSetType(st,STSINVERT);
        // set initial guess
        std::vector<Vec> qrvec(1);
        qrvec[0] = qr.vec;
        ierr = EPSSetInitialSpace(eps,1,qrvec.data());CHKERRXX(ierr);
        //std::vector<Vec> qlvec(1);
        //qlvec[0] = ql.vec;
        //ierr = EPSSetLeftInitialSpace(eps,1,qlvec.data());CHKERRXX(ierr); // TODO doesn't work using 3.12.1 version, but should work in 3.14 documentation

        // Solve the system with left and right
        //EPSSetTwoSided(eps,PETSC_TRUE);
        ierr = EPSSolve(eps);CHKERRXX(ierr);
        //std::cout<<"After KSPSolve"<<std::endl;

        // output iterations
        //PetscInt its, maxit;
        PetscInt nconv;
        //PetscReal error, tol, re, im;
        /*
Optional: Get some information from the solver and display it
*/
        //ierr = EPSGetIterationNumber(eps,&its);CHKERRXX(ierr);
        //ierr = PetscPrintf(PETSC_COMM_WORLD," Number of iterations of the method: %D\n",its);CHKERRXX(ierr);
        //ierr = EPSGetType(eps,&type);CHKERRXX(ierr);
        //ierr = PetscPrintf(PETSC_COMM_WORLD," Solution method: %s\n\n",type);CHKERRXX(ierr);
        //ierr = EPSGetDimensions(eps,&nev,NULL,NULL);CHKERRXX(ierr);
        //ierr = PetscPrintf(PETSC_COMM_WORLD," Number of requested eigenvalues: %D\n",nev);CHKERRXX(ierr);
        //ierr = EPSGetTolerances(eps,&tol,&maxit);CHKERRXX(ierr);
        //ierr = PetscPrintf(PETSC_COMM_WORLD," Stopping condition: tol=%.4g, maxit=%D\n",(double)tol,maxit);CHKERRXX(ierr);

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
           Display solution and clean up
           - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
        /*
           Get number of converged approximate eigenpairs
           */
        ierr = EPSGetConverged(eps,&nconv);CHKERRXX(ierr);
        //ierr = PetscPrintf(PETSC_COMM_WORLD," Number of converged eigenpairs: %D\n\n",nconv);CHKERRXX(ierr);

        if (nconv>0) {
            /*
               Display eigenvalues and relative errors
               */
            // ierr = PetscPrintf(PETSC_COMM_WORLD,
            //         "      k                ||Ax-kx||/||kx||\n"
            //         "   ----------------- ------------------\n");CHKERRXX(ierr);

            // for (PetscInt i=0;i<nconv;i++) 
            //     /*
            //        Get converged eigenpairs: i-th eigenvalue is stored in kr (real part) and
            //        ki (imaginary part)
            //        */
            //     ierr = EPSGetEigenpair(eps,i,&alpha,&ki,eigr_vec.vec,eigl_vec.vec);CHKERRXX(ierr);
            //     /*
            //        Compute the relative error associated to each eigenpair
            //        */
            //     PetscReal error, re, im;
            //     ierr = EPSComputeError(eps,i,EPS_ERROR_RELATIVE,&error);CHKERRXX(ierr);
            //     re = PetscRealPart(alpha);
            //     im = PetscImaginaryPart(alpha);
            //     if (im!=0.0) 
            //         ierr = PetscPrintf(PETSC_COMM_WORLD," (%9e+%9ei)  %12g\n",(double)re,(double)im,(double)error);CHKERRXX(ierr);
            //       else 
            //         ierr = PetscPrintf(PETSC_COMM_WORLD,"   %12e       %12g\n",(double)re,(double)error);CHKERRXX(ierr);
            //      
            //  
            // ierr = PetscPrintf(PETSC_COMM_WORLD,"\n");CHKERRXX(ierr);

            ierr = EPSGetEigenpair(eps,0,&alpha,PETSC_NULL,eigr_vec.vec,PETSC_NULL);CHKERRXX(ierr);
            eigr_vec.rows=rows;
            //ierr = EPSGetLeftEigenvector(eps,0,eigl_vec.vec,PETSC_NULL);CHKERRXX(ierr);
        }

        // PetscInt its;
        // ierr = EPSGetIterationNumber(eps,&its);CHKERRXX(ierr);
        // //ierr = PetscPrintf(PETSC_COMM_WORLD,"ksp iterations %D\n",its);CHKERRXX(ierr);
        // ierr = PetscPrintf(PETSC_COMM_WORLD,"EPS Solved in %D iterations \n",its); CHKERRXX(ierr);
        // Free work space.  All PETSc objects should be destroyed when they
        // are no longer needed.
        //set_Vec(x);
        ierr = EPSDestroy(&eps);CHKERRXX(ierr);
        //ierr = VecDestroy(&x);CHKERRXX(ierr);
        //ierr = VecDestroy(&b);CHKERRXX(ierr); 
        //ierr = MatDestroy(&A);CHKERRXX(ierr);
        //ierr = PetscFinalize();

        return std::make_tuple(alpha,eigr_vec);
        //return std::make_tuple(alpha,alpha);
    }
    /** \brief solve general eigenvalue problem of Ax = kBx and return a tuple of tie(PetscScalar alpha, SPIVec eig_vector) \return tuple of eigenvalue and eigenvector closest to the target value e.g.  std::tie(alpha, eig_vector) = eig(A,B,0.1+0.4*PETSC_i) using initial subspace defined by a vector */
    std::tuple<std::vector<PetscScalar>, std::vector<SPIVec>> eig_init_rights(
            const SPIMat &A,                            ///< [in] A in Ax=kBx generalized eigenvalue problem
            const SPIMat &B,                            ///< [in] B in Ax=kBx generalized eigenvalue problem
            const std::vector<PetscScalar> targets,     ///< [in] target eigenvalue to solve for
            const std::vector<SPIVec> &qrs,             ///< [in] initial subspace vector for EPS solver (right eigenvector)
            PetscReal tol,                              ///< [in] tolerance of eigenvalue solver
            const PetscInt max_iter                     ///< [in] maximum number of iterations
            ){
        //std::cout<<"target = "<<target<<std::endl;
        PetscInt rows=A.rows;
        EPS             eps;        /* eigenproblem solver context slepc */
        //ST              st;
        //EPSType         type;
        //KSP             ksp;        /* linear solver context petsc */
        PetscErrorCode  ierr;
        std::vector<PetscScalar> alphas(qrs.size());
        PetscScalar alpha;
        std::vector<SPIVec> eigr_vecs(qrs.size());
        SPIVec eigr_vec(rows);

        PetscScalar kr_temp, ki_temp;

        // Create the eigenvalue solver and set various options
        // Create solver contexts
        ierr = EPSCreate(PETSC_COMM_WORLD,&eps);CHKERRXX(ierr);
        // Set operators. Here the matrix that defines the eigenvalue system
        // swap operators such that LHS matrix is singular, and RHS matrix can be inverted (for slepc)
        // this will make the Ax=lambda Bx into the problem Bx = (1./lambda) Ax, thus our eigenvalues are inverted
        ierr = EPSSetOperators(eps,A.mat,B.mat);CHKERRXX(ierr);
        //ierr = EPSSetOperators(eps,B,A);CHKERRXX(ierr);
        // Set runtime options, e.g.,
        // -
        ierr = EPSSetFromOptions(eps);CHKERRXX(ierr);
        //std::cout<<"After KSPSetFromOptions"<<std::endl;
        //
        // set convergence type
        EPSWhich which=EPS_TARGET_MAGNITUDE;
        PetscInt nev=1;
        EPSSetWhichEigenpairs(eps,which);
        EPSSetDimensions(eps,nev,PETSC_DEFAULT,PETSC_DEFAULT);
        if ((max_iter==-1) && (tol==-1.)){
            EPSSetTolerances(eps,PETSC_DEFAULT,PETSC_DEFAULT);
        }
        else if(tol==-1.){
            EPSSetTolerances(eps,PETSC_DEFAULT,max_iter);
        }
        else if(max_iter==-1){
            EPSSetTolerances(eps,tol,PETSC_DEFAULT);
        }
        else{
            EPSSetTolerances(eps,tol,max_iter);
        }
        if (
                which==EPS_TARGET_REAL ||
                which==EPS_TARGET_IMAGINARY ||
                which==EPS_TARGET_MAGNITUDE){
            // PetscScalar target=0.-88.5*PETSC_i;
            EPSSetTarget(eps,targets[0]);
            //EPSSetTolerances(eps,1.E-8,100000);
        }


        // set spectral transform to shift-and-invert (seems to work best for LST_spatial)
        ST              st;
        EPSGetST(eps,&st);
        STSetType(st,STSINVERT);
        // set initial guess
        PetscInt k=qrs.size();
        std::vector<Vec> qrvec(k);
        for(PetscInt i=0; i<k; ++i){
            qrvec[i] = qrs[i].vec;
        }
        ierr = EPSSetInitialSpace(eps,qrvec.size(),qrvec.data());CHKERRXX(ierr);
        //std::vector<Vec> qlvec(1);
        //qlvec[0] = ql.vec;
        //ierr = EPSSetLeftInitialSpace(eps,1,qlvec.data());CHKERRXX(ierr); // TODO doesn't work using 3.12.1 version, but should work in 3.14 documentation

        // Solve the system with left and right
        //EPSSetTwoSided(eps,PETSC_TRUE);
        PetscInt nconv;
        for(PetscInt i=0; i<k; ++i){
            EPSSetWhichEigenpairs(eps,EPS_LARGEST_REAL); // have to add this to switch the eigenvalue solver to a new target
            EPSSetWhichEigenpairs(eps,which);
            EPSSetTarget(eps,targets[i]);
            ierr = EPSSolve(eps);CHKERRXX(ierr);
            //std::cout<<"After KSPSolve"<<std::endl;

            // output iterations
            //PetscInt its, maxit;
            //PetscReal error, tol, re, im;
            /*
Optional: Get some information from the solver and display it
*/
            //ierr = EPSGetIterationNumber(eps,&its);CHKERRXX(ierr);
            //ierr = PetscPrintf(PETSC_COMM_WORLD," Number of iterations of the method: %D\n",its);CHKERRXX(ierr);
            //ierr = EPSGetType(eps,&type);CHKERRXX(ierr);
            //ierr = PetscPrintf(PETSC_COMM_WORLD," Solution method: %s\n\n",type);CHKERRXX(ierr);
            //ierr = EPSGetDimensions(eps,&nev,NULL,NULL);CHKERRXX(ierr);
            //ierr = PetscPrintf(PETSC_COMM_WORLD," Number of requested eigenvalues: %D\n",nev);CHKERRXX(ierr);
            //ierr = EPSGetTolerances(eps,&tol,&maxit);CHKERRXX(ierr);
            //ierr = PetscPrintf(PETSC_COMM_WORLD," Stopping condition: tol=%.4g, maxit=%D\n",(double)tol,maxit);CHKERRXX(ierr);

            /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
               Display solution and clean up
               - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
            /*
               Get number of converged approximate eigenpairs
               */
            ierr = EPSGetConverged(eps,&nconv);CHKERRXX(ierr);
            //ierr = PetscPrintf(PETSC_COMM_WORLD," Number of converged eigenpairs: %D\n\n",nconv);CHKERRXX(ierr);

            if (nconv>0) {
                /*
                   Display eigenvalues and relative errors
                   */
                // ierr = PetscPrintf(PETSC_COMM_WORLD,
                //         "      k                ||Ax-kx||/||kx||\n"
                //         "   ----------------- ------------------\n");CHKERRXX(ierr);

                // for (PetscInt i=0;i<nconv;i++) 
                //     /*
                //        Get converged eigenpairs: i-th eigenvalue is stored in kr (real part) and
                //        ki (imaginary part)
                //        */
                //     ierr = EPSGetEigenpair(eps,i,&alpha,&ki,eigr_vec.vec,eigl_vec.vec);CHKERRXX(ierr);
                //     /*
                //        Compute the relative error associated to each eigenpair
                //        */
                //     PetscReal error, re, im;
                //     ierr = EPSComputeError(eps,i,EPS_ERROR_RELATIVE,&error);CHKERRXX(ierr);
                //     re = PetscRealPart(alpha);
                //     im = PetscImaginaryPart(alpha);
                //     if (im!=0.0) 
                //         ierr = PetscPrintf(PETSC_COMM_WORLD," (%9e+%9ei)  %12g\n",(double)re,(double)im,(double)error);CHKERRXX(ierr);
                //       else 
                //         ierr = PetscPrintf(PETSC_COMM_WORLD,"   %12e       %12g\n",(double)re,(double)error);CHKERRXX(ierr);
                //      
                //  
                // ierr = PetscPrintf(PETSC_COMM_WORLD,"\n");CHKERRXX(ierr);

                ierr = EPSGetEigenpair(eps,0,&alpha,PETSC_NULL,eigr_vec.vec,PETSC_NULL);CHKERRXX(ierr);
                alphas[i] = alpha;
                //EPSErrorView(eps,EPS_ERROR_RELATIVE,NULL);
                eigr_vecs[i] = eigr_vec;
                //ierr = EPSGetLeftEigenvector(eps,0,eigl_vec.vec,PETSC_NULL);CHKERRXX(ierr);
            }
        }

        // PetscInt its;
        // ierr = EPSGetIterationNumber(eps,&its);CHKERRXX(ierr);
        // //ierr = PetscPrintf(PETSC_COMM_WORLD,"ksp iterations %D\n",its);CHKERRXX(ierr);
        // ierr = PetscPrintf(PETSC_COMM_WORLD,"EPS Solved in %D iterations \n",its); CHKERRXX(ierr);
        // Free work space.  All PETSc objects should be destroyed when they
        // are no longer needed.
        //set_Vec(x);
        ierr = EPSDestroy(&eps);CHKERRXX(ierr);
        //ierr = VecDestroy(&x);CHKERRXX(ierr);
        //ierr = VecDestroy(&b);CHKERRXX(ierr); 
        //ierr = MatDestroy(&A);CHKERRXX(ierr);
        //ierr = PetscFinalize();

        return std::make_tuple(alphas,eigr_vecs);
        //return std::make_tuple(alpha,alpha);
    }
    /** \brief solve general polynomial eigenvalue problem of (A0 + A1*alpha + A2*alpha^2 + ...)*x = 0 and return a tuple of tie(PetscScalar alpha, SPIVec eig_vector) \return tuple of eigenvalue and eigenvector closest to the target value e.g.  std::tie(alpha, eig_vector) = eig([A0,A1,A2...],0.1+0.4*PETSC_i) */
    std::tuple<PetscScalar, SPIVec> polyeig(
            const std::vector<SPIMat> &As,        ///< [in] [A0,A1,A2...] in generalized polynomial eigenvalue problem
            const PetscScalar target,   ///< [in] target eigenvalue to solve for
            const PetscReal tol,    ///< [in] tolerance of eigenvalue solver
            const PetscInt max_iter ///< [in] maximum number of iterations
            ){
        // if linear eigenvalue problem, use EPS
        if(0){//As.size()==1)
            //return eig(As[0],SPI::eye(As[0].rows),target,tol,max_iter);
        }
        else if(0){//As.size()==2)
            //return eig(As[0],-As[1],target,tol,max_iter);
        }
        else{// else polynomial eigenvalue problem use PEP
            PetscInt rows=As[0].rows;
            PEP             pep;        /* polynomial eigenproblem solver context slepc */
            PetscErrorCode  ierr;
            PetscScalar ki,alpha;
            //SPIVec eigl_vec(rows),eigr_vec(rows);
            SPIVec eigr_vec(rows);
            PetscScalar kr_temp;
            // Create the eigenvalue solver and set various options
            ierr = PEPCreate(PETSC_COMM_WORLD,&pep);CHKERRXX(ierr);
            // Set operators. Here the matrix that defines the eigenvalue system
            std::vector<Mat> AsMat(As.size());
            for (unsigned i=0; i<As.size(); ++i){ AsMat[i]=As[i].mat; }
            ierr = PEPSetOperators(pep,AsMat.size(),AsMat.data());CHKERRXX(ierr);
            // Set runtime options, e.g.,
            ierr = PEPSetFromOptions(pep);CHKERRXX(ierr);
            // set convergence type
            PEPWhich which=PEP_TARGET_MAGNITUDE;
            PetscInt nev=1;
            PEPSetWhichEigenpairs(pep,which);
            PEPSetDimensions(pep,nev,PETSC_DEFAULT,PETSC_DEFAULT);
            if ((max_iter==-1) && (tol==-1.)){  PEPSetTolerances(pep,PETSC_DEFAULT, PETSC_DEFAULT); }
            else if(tol==-1.){                  PEPSetTolerances(pep,PETSC_DEFAULT, max_iter); }
            else if(max_iter==-1){              PEPSetTolerances(pep,tol,           PETSC_DEFAULT); }
            else{                               PEPSetTolerances(pep,tol,           max_iter); }
            PEPSetTarget(pep,target);
            // set spectral transform to shift-and-invert (seems to work best for LST_spatial)
            ST              st;
            PEPGetST(pep,&st);
            STSetType(st,STSINVERT);
            // get EPS and set two sided
            //EPS eps;
            //ierr = PEPLinearGetEPS(pep,&eps);// CHKERRXX(ierr);
            //EPSSetTwoSided(eps,PETSC_TRUE); // this doesn't work TODO
            // Solve the system
            ierr = PEPSolve(pep);CHKERRXX(ierr);
            //ierr = EPSSolve(eps);CHKERRXX(ierr);
            // output iterations
            PetscInt nconv;
            ierr = PEPGetConverged(pep,&nconv);CHKERRXX(ierr);
            //ierr = EPSGetConverged(eps,&nconv);CHKERRXX(ierr);
            if (nconv>0) { 
                ierr = PEPGetEigenpair(pep,0,&alpha,PETSC_NULL,eigr_vec.vec,PETSC_NULL);CHKERRXX(ierr); 
                eigr_vec.flag_init=PETSC_TRUE;
                //ierr = EPSGetEigenpair(eps,0,&alpha,PETSC_NULL,eigr_vec.vec,PETSC_NULL);CHKERRXX(ierr); 
                //ierr = EPSGetLeftEigenvector(eps,0,eigl_vec.vec,PETSC_NULL);CHKERRXX(ierr);

            }
            // destroy pep
            ierr = PEPDestroy(&pep);CHKERRXX(ierr);
            // return
            return std::make_tuple(alpha,eigr_vec);
        }
    }
    /** \brief solve general polynomial eigenvalue problem of (A0 + A1*alpha + A2*alpha^2 + ...)*x = 0 and return a tuple of tie(PetscScalar alpha, SPIVec eig_vector) using initial subspace vector \return tuple of eigenvalue and eigenvector closest to the target value e.g.  std::tie(alpha, eig_vector) = eig([A0,A1,A2...],0.1+0.4*PETSC_i) */
    std::tuple<PetscScalar, SPIVec> polyeig_init(
            const std::vector<SPIMat> &As,        ///< [in] [A0,A1,A2...] in generalized polynomial eigenvalue problem
            const PetscScalar target,   ///< [in] target eigenvalue to solve for
            const SPIVec &qr,         ///< [in] initial subspace vector for right eigenvalue problem
            const PetscReal tol,    ///< [in] tolerance of eigenvalue solver
            const PetscInt max_iter ///< [in] maximum number of iterations
            ){
        // if linear eigenvalue problem, use EPS
        if(0){//As.size()==1)
            //return eig(As[0],SPI::eye(As[0].rows),target,tol,max_iter);
        }
        else if(0){//As.size()==2)
            //return eig(As[0],-As[1],target,tol,max_iter);
        }
        else{// else polynomial eigenvalue problem use PEP
            PetscInt rows=As[0].rows;
            PEP             pep;        /* polynomial eigenproblem solver context slepc */
            PetscErrorCode  ierr;
            PetscScalar alpha;
            //SPIVec eigl_vec(rows),eigr_vec(rows);
            SPIVec eigr_vec(rows);
            PetscScalar kr_temp;
            // Create the eigenvalue solver and set various options
            ierr = PEPCreate(PETSC_COMM_WORLD,&pep);CHKERRXX(ierr);
            // Set operators. Here the matrix that defines the eigenvalue system
            std::vector<Mat> AsMat(As.size());
            for (unsigned i=0; i<As.size(); ++i){ AsMat[i]=As[i].mat; }
            ierr = PEPSetOperators(pep,AsMat.size(),AsMat.data());CHKERRXX(ierr);
            // Set runtime options, e.g.,
            ierr = PEPSetFromOptions(pep);CHKERRXX(ierr);
            // set convergence type
            PEPWhich which=PEP_TARGET_MAGNITUDE;
            PetscInt nev=1;
            PEPSetWhichEigenpairs(pep,which);
            PEPSetDimensions(pep,nev,PETSC_DEFAULT,PETSC_DEFAULT);
            if ((max_iter==-1) && (tol==-1.)){  PEPSetTolerances(pep,PETSC_DEFAULT, PETSC_DEFAULT); }
            else if(tol==-1.){                  PEPSetTolerances(pep,PETSC_DEFAULT, max_iter); }
            else if(max_iter==-1){              PEPSetTolerances(pep,tol,           PETSC_DEFAULT); }
            else{                               PEPSetTolerances(pep,tol,           max_iter); }
            PEPSetTarget(pep,target);
            // set spectral transform to shift-and-invert (seems to work best for LST_spatial)
            ST              st;
            PEPGetST(pep,&st);
            STSetType(st,STSINVERT);
            // set initial guess for right eigenvalue problem
            std::vector<Vec> qrvec(1);
            qrvec[0] = qr.vec;
            ierr = PEPSetInitialSpace(pep,1,qrvec.data());CHKERRXX(ierr);
            // now for left (adjoint) eigenvalue problem initial guess
            //EPS eps;
            //ierr = PEPLinearGetEPS(pep,&eps); CHKERRXX(ierr);
            //std::vector<Vec> qlvec(1);
            //qlvec[0] = ql.vec;
            //ierr = EPSSetLeftInitialSpace(eps,1,qlvec.data()); CHKERRXX(ierr);

            // Solve the system
            ierr = PEPSolve(pep);CHKERRXX(ierr);
            // output iterations
            PetscInt nconv;
            ierr = PEPGetConverged(pep,&nconv);CHKERRXX(ierr);
            if (nconv>0) { 
                ierr = PEPGetEigenpair(pep,0,&alpha,PETSC_NULL,eigr_vec.vec,PETSC_NULL);CHKERRXX(ierr); 
                //ierr = EPSGetLeftEigenvector(eps,0,eigl_vec.vec,PETSC_NULL);CHKERRXX(ierr);
            }
            // destroy pep
            ierr = PEPDestroy(&pep);CHKERRXX(ierr);
            // return
            return std::make_tuple(alpha,eigr_vec);
        }
    }
    // /** \brief set block matrices using an input array of size rows*cols.  Fills rows first \return new matrix with inserted blocks */
    //SPIMat block(
    //        const SPIMat Blocks[],  ///< [in] array of matrices to set into larger matrix e.g. [ A, B, C, D ]
    //        const PetscInt rows,    ///< [in] number of rows of submatrices e.g. 2
    //        const PetscInt cols     ///< [in] number of columns of submatrices e.g. 2
    //        ) 
    //    PetscInt m[rows];
    //    PetscInt msum=Blocks[0].rows;
    //    PetscInt n[cols];
    //    PetscInt nsum=Blocks[0].cols;
    //    m[0]=n[0]=0;
    //    for (PetscInt i=1; i<rows; ++i)[
    //        m[i] = m[i-1]+Blocks[i-1].rows;
    //        msum += m[i];
    //     
    //    for (PetscInt j=1; j<cols; ++j)[
    //        n[j] = n[j-1]+Blocks[j*rows].cols;
    //        nsum += n[j];
    //     

    //    // TODO check if all rows and columns match for block matrix....

    //    SPIMat A(msum,nsum);

    //    for (PetscInt j=0; j<cols; ++j)[
    //        for(PetscInt i=0; i<rows; ++i)[
    //            A(m[i],n[j],Blocks[i+j*rows]);
    //         
    //     

    //    return A;
    // 
    /** \brief set block matrices using an input array of size rows*cols.  Fills rows first \return new matrix with inserted blocks */
    SPIMat block(
            //std::vector<std::vector<SPIMat>> Blocks  ///< [in] array of matrices to set into larger matrix e.g. [ A, B, C, D ]
            const Block2D<SPIMat> Blocks                        ///< [in] array of matrices to set into larger matrix e.g. [ A, B, C, D ]
            ){
        const PetscInt rows=Blocks.size();    // number of rows of submatrices e.g. 2
        const PetscInt cols=Blocks[0].size(); // number of columns of submatrices e.g. 2
        PetscInt m[rows];
        PetscInt msum=Blocks[0][0].rows;
        PetscInt n[cols];
        PetscInt nsum=Blocks[0][0].cols;
        m[0]=n[0]=0;
        for (PetscInt i=1; i<rows; ++i) m[i] = m[i-1]+Blocks[i-1][0].rows;
        for (PetscInt i=1; i<rows; ++i) msum += Blocks[i][0].rows;
        for (PetscInt j=1; j<cols; ++j) nsum += Blocks[0][j].cols;
        for (PetscInt j=1; j<cols; ++j) n[j] = n[j-1]+Blocks[0][j-1].cols;

        // TODO check if all rows and columns match for block matrix.... user error catch

        SPIMat A(msum,nsum);
        //printf("msum,nsum = %d,%d",msum,nsum);
        //for (PetscInt j=0; j<cols; ++j)[
        //for(PetscInt i=0; i<rows; ++i)[
        //printf("m[%d],n[%d] = %d,%d",i,j,m[i],n[j]);
        //]
        //]

        for(PetscInt i=0; i<rows; ++i){
            for (PetscInt j=0; j<cols; ++j){
                //printf("setting A[%d,%d] with shape=%dx%d",m[i],n[j],Blocks[i][j].rows,Blocks[i][j].cols);
                A(m[i],n[j],Blocks[i][j],INSERT_VALUES);
            }
        }
        //A();

        return A;
    }
    /* brief create meshgrid with ij indexing \brief tuple of X and Y matrices */
    std::tuple<SPIMat,SPIMat> meshgrid(
            SPIVec &x,    ///< [in] x input array
            SPIVec &y     ///< [in] y input array
            ){
        PetscInt m=x.rows;
        PetscInt n=y.rows;
        SPIMat X(m,n);
        SPIMat Y(m,n);
        for(PetscInt i=0; i<m; ++i){
            for(PetscInt j=0; j<n; ++j){
                X(i,j,x(i,PETSC_TRUE));
                Y(i,j,y(j,PETSC_TRUE));
            }
        }
        X();
        Y();
        return std::make_tuple(X,Y);
    }

    /** \brief save matrix to filename in binary format (see Petsc documentation for format )
     * Format is (from Petsc documentation):
     * int    MAT_FILE_CLASSID
     * int    number of rows
     * int    number of columns
     * int    total number of nonzeros
     * int    *number nonzeros in each row
     * int    *column indices of all nonzeros (starting index is zero)
     * PetscScalar *values of all nonzeros
     *
     * \returns 0 if successful */
    PetscInt save(
            const SPIMat &A,        ///< [in] A to save in 
            const std::string filename ///< [in] filename to save data to
            ){
        PetscViewer     viewer;
        //PetscViewerASCIIOpen(PETSC_COMM_WORLD,name.c_str(),&viewer);
        PetscErrorCode ierr;
        std::ifstream f(filename.c_str());
        if(f.good()){
            ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,filename.c_str(),FILE_MODE_APPEND,&viewer);CHKERRXX(ierr);
        }
        else{
            ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,filename.c_str(),FILE_MODE_WRITE,&viewer);CHKERRXX(ierr);
        }
        //ierr = PetscViewerPushFormat(viewer,format);CHKERRXX(ierr);
        ierr = MatView(A.mat,viewer);CHKERRXX(ierr);
        ierr = PetscViewerDestroy(&viewer);CHKERRXX(ierr);
        return 0;
    }
    /** \brief save matrices to filename in binary format (see Petsc documentation for format \returns 0 if successful */
    PetscInt save(
            const std::vector<SPIMat> &As,        ///< [in] A to save in 
            const std::string filename ///< [in] filename to save data to
            ){
        PetscViewer     viewer;
        //PetscViewerASCIIOpen(PETSC_COMM_WORLD,name.c_str(),&viewer);
        PetscErrorCode ierr;
        std::ifstream f(filename.c_str());
        if(f.good()){
            ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,filename.c_str(),FILE_MODE_APPEND,&viewer);CHKERRXX(ierr);
        }
        else{
            ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,filename.c_str(),FILE_MODE_WRITE,&viewer);CHKERRXX(ierr);
        }
        //ierr = PetscViewerPushFormat(viewer,format);CHKERRXX(ierr);
        for(unsigned i=0; i<As.size(); ++i){
            ierr = MatView(As[i].mat,viewer);CHKERRXX(ierr);
        }
        ierr = PetscViewerDestroy(&viewer);CHKERRXX(ierr);
        return 0;
    }
    /** \brief load matrix from filename from binary format (works with save(SPIMat,std::string) function \returns 0 if successful */
    PetscInt load(
            SPIMat &A,        ///< [inout] A to load data into (must be initialized to the right size)
            const std::string filename ///< [in] filename to read
            ){
        PetscViewer viewer;
        //std::ifstream f(filename.c_str());
        A.ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD, filename.c_str(), FILE_MODE_READ, &viewer); CHKERRXX(A.ierr);
        A.ierr = MatLoad(A.mat,viewer); CHKERRXX(A.ierr);
        A.ierr = PetscViewerDestroy(&viewer); CHKERRXX(A.ierr);
        return 0;
    }
    /** \brief load matrix from filename from binary format (works with save(SPIMat,std::string) function \returns 0 if successful */
    PetscInt load(
            std::vector<SPIMat> &As,         ///< [inout] matrices to load data into (must be initialized to the right size)
            const std::string filename      ///< [in] filename to read
            ){
        PetscViewer viewer;
        //std::ifstream f(filename.c_str());
        PetscErrorCode ierr;
        ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD, filename.c_str(), FILE_MODE_READ, &viewer); CHKERRXX(ierr);
        for(unsigned i=0; i<As.size(); ++i){
            ierr = MatLoad(As[i].mat,viewer); CHKERRXX(ierr);
        }
        ierr = PetscViewerDestroy(&viewer); CHKERRXX(ierr);
        return 0;
    }

    /** \brief draw nonzero structure of matrix \returns 0 if successful */
    PetscInt draw(
            const SPIMat &A         ///< [in] A to draw nonzero structure
            ){
        PetscViewer     viewer;
        PetscErrorCode ierr;
        ierr = PetscViewerDrawOpen(PETSC_COMM_WORLD,NULL,A.name.c_str(),PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE,&viewer);CHKERRXX(ierr);
        ierr = MatView(A.mat,viewer);CHKERRXX(ierr);

        // pause until user inputs at command line
        SPI::printf("  draw(SPIMat) with title=%s, hit ENTER to continue",A.name.c_str());
        std::cin.ignore();

        ierr = PetscViewerDestroy(&viewer);CHKERRXX(ierr);
        return 0;
    }
    /** \brief take the function of each element in a matrix, e.g. (*f)(A(i,j)) for each i,j */
    template <class T>
        SPIMat _Function_on_each_element(
                T (*f)(T const&),   ///< [in] function handle to pass in e.g. std::sin<PetscReal>
                const SPIMat &A           ///< [in] matrix to perform function on each element
                ){
            SPIMat out(A.rows,A.cols);
            for (PetscInt i=0; i<out.rows; ++i){
                for (PetscInt j=0; j<out.cols; ++j){
                    out(i,j,(*f)(A(i,j,PETSC_TRUE)));                // TODO speed up by getting all values at once on local processor and looping through those
                }
            }
            out();
            return out;
        }
    /** \brief take the sin of each element in a matrix */
    SPIMat sin(const SPIMat &A){ return _Function_on_each_element(&std::sin<PetscReal>, A); }
    /** \brief take the cos of each element in a matrix */
    SPIMat cos(const SPIMat &A){ return _Function_on_each_element(&std::cos<PetscReal>, A); }
    /** \brief take the arccos of each element in a matrix */
    SPIMat acos(const SPIMat &A){ return _Function_on_each_element(&std::acos<PetscReal>, A); }
    /** \brief take the tan of each element in a matrix */
    SPIMat tan(const SPIMat &A){ return _Function_on_each_element(&std::tan<PetscReal>, A); }
    /** \brief take the elementwise multiplication of each element in a matrix */
    SPIMat operator%(const SPIMat &A, const SPIMat &B){ 
        SPIMat out(A.rows,A.cols);
        for (PetscInt i=0; i<out.rows; ++i){
            for (PetscInt j=0; j<out.cols; ++j){
                out(i,j,A(i,j,PETSC_TRUE)*B(i,j,PETSC_TRUE));                // TODO speed up by getting all values at once on local processor and looping through those
            }
        }
        out();
        return out;
        //return _Function_on_each_element2(&std::multiplies<PetscReal>, A,B);  
    }
    /** \brief take the abs of each element in a matrix */
    SPIMat abs(const SPIMat &A){ 
        SPIMat out(A.rows,A.cols);
        for (PetscInt i=0; i<out.rows; ++i){
            for (PetscInt j=0; j<out.cols; ++j){
                out(i,j,std::abs(A(i,j,PETSC_TRUE)));                // TODO speed up by getting all values at once on local processor and looping through those
            }
        }
        out();
        return out;
        //return _Function_on_each_element(&std::abs<PetscScalar>, A); 
    }
    /* \brief orthogonalize a basis dense matrix from an array of vec using SLEPc BV */
    SPIMat orthogonalize(
            const std::vector<SPIVec> &x  ///< [in] array of vectors to orthogonalize 
            ){
        PetscInt m=x[0].rows;   // number of rows
        PetscInt n=x.size();    // number of columns
        Vec xvec[n];
        for(PetscInt i=0; i<n; ++i) xvec[i]=x[i].vec;
        SPIMat E("E");
        // create and initialize BV
        BV bv;
        E.ierr = BVCreate(PETSC_COMM_WORLD,&bv); CHKERRXX(E.ierr);
        E.ierr = BVSetSizesFromVec(bv,x[0].vec,n); CHKERRXX(E.ierr);
        E.ierr = BVSetFromOptions(bv);CHKERRXX(E.ierr);
        E.ierr = BVInsertVecs(bv,0,&n,xvec,PETSC_TRUE);
        //SPI::SPIMat AorthH("AorthH");
        E.ierr = BVOrthogonalize(bv,PETSC_NULL); CHKERRXX(E.ierr);
        E.ierr = BVCreateMat(bv,&E.mat); CHKERRXX(E.ierr);
#if USE_GPU == 1
        E.ierr = MatConvert(E.mat,MATMPIAIJCUSPARSE,MAT_INPLACE_MATRIX,&E.mat); CHKERRXX(E.ierr);
#else
        E.ierr = MatConvert(E.mat,MATMPIAIJ,MAT_INPLACE_MATRIX,&E.mat); CHKERRXX(E.ierr);
#endif
        E.rows=m;
        E.cols=n;
        E();
        return E;
    }

}


