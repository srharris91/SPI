#include "SPEMat.hpp"
#include <petscviewerhdf5.h>

namespace SPE{

    // constructors
    /** \brief constructor with no arguments (no initialization) */
    SPEMat::SPEMat(
            std::string _name   ///< [in] name of SPEMat
            ){name=_name; }
    /** constructor using another SPEMat */
    SPEMat::SPEMat(
            const SPEMat &A,    ///< [in] another SPEMat to copy into this new SPEMat
            std::string _name   ///< [in] name of SPEMat
            ){
        name=_name; 
        (*this) = A;
    }
    /** constructor with one arguement to make square matrix */
    SPEMat::SPEMat(
            PetscInt rowscols,  ///< [in] number of rows and columns to make the square matrix
            std::string _name   ///< [in] name of SPEMat
            ){
        Init(rowscols,rowscols,_name);
    }
    /** \brief constructor of rectangular matrix */
    SPEMat::SPEMat(
            PetscInt rowsm,     ///< [in] number of rows in matrix
            PetscInt colsn,     ///< [in] number of columns in matrix
            std::string _name   ///< [in] name of SPEMat
            ){
        Init(rowsm,colsn,_name);
    }

    // Initialize matrix
    /** initialize the matrix of size m by n \return 0 if successful */
    PetscInt SPEMat::Init(
            PetscInt m,         ///< [in] number of rows
            PetscInt n,         ///< [in] number of columns
            std::string _name   ///< [in] name of SPEMat
            ){
        name=_name;
        rows=m;
        cols=n;
        ierr = MatCreate(PETSC_COMM_WORLD,&mat);CHKERRQ(ierr);
        ierr = MatSetSizes(mat,PETSC_DECIDE,PETSC_DECIDE,m,n);CHKERRQ(ierr);
        //ierr = MatSetFromOptions(mat);CHKERRQ(ierr);
        ierr = MatSetType(mat,MATMPIAIJ);CHKERRQ(ierr);
        ierr = MatSetUp(mat);CHKERRQ(ierr);
        flag_init=PETSC_TRUE;
        return 0;
    }

    /** set a scalar value at position row m and column n \return current SPEMat after setting value */
    SPEMat& SPEMat::set(
            PetscInt m,         ///< [in] row to insert scalar
            PetscInt n,         ///< [in] column to insert scalar
            const PetscScalar v ///< [in] scalar to insert in matrix
            ){
        ierr = MatSetValue(mat,m,n,v,INSERT_VALUES);CHKERRXX(ierr);
        return (*this);
    }
    /** \brief add a scalar value at position row m and column n \return current matrix after adding the value*/
    SPEMat& SPEMat::add(
            PetscInt m,         ///< [in] row to add scalar
            PetscInt n,         ///< [in] column to add scalar
            const PetscScalar v ///< [in] scalar to add in matrix
            ){
        ierr = MatSetValue(mat,m,n,v,ADD_VALUES);CHKERRXX(ierr);
        return (*this);
    }

    // overloaded operators, get
    /** \brief get local value at row m, column n \return scalar at specified location */ // TODO: make global on all processors
    PetscScalar SPEMat::operator()(
            PetscInt m,         ///< [in] row to get scalar
            PetscInt n          ///< [in] column to get scalar
            ){
        PetscScalar v;
        ierr = MatGetValues(mat,1,&m, 1,&n, &v);
        return v;
    }
    // overloaded operator, set
    /** \brief set operator the same as set function \return current matrix after setting value */
    SPEMat& SPEMat::operator()(
            PetscInt m,         ///< [in] row to set scalar
            PetscInt n,         ///< [in] column to set scalar
            const PetscScalar v ///< [in] scalar to set in matrix
            ){
        ierr = MatSetValue(mat,m,n,v,INSERT_VALUES);CHKERRXX(ierr);
        return (*this);
    }
    /** \brief set operator the same as set function \return current matrix after setting value */
    SPEMat& SPEMat::operator()(
            PetscInt m,         ///< [in] row to set scalar
            PetscInt n,         ///< [in] column to set scalar
            const double v      ///< [in] scalar to set in matrix
            ){
        //ierr = (*this)(m,n,(PetscScalar)v);CHKERRXX(ierr);
        return (*this)(m,n,(PetscScalar)v);
    }
    /** \brief set operator the same as set function \return current matrix after setting value */
    SPEMat& SPEMat::operator()(
            PetscInt m,         ///< [in] row to set scalar
            PetscInt n,         ///< [in] column to set scalar
            const int v         ///< [in] scalar to set in matrix
            ){
        //ierr = (*this)(m,n,(PetscScalar)v);CHKERRXX(ierr);
        return (*this)(m,n,(PetscScalar)v);
    }

    // overloaded operator, set
    /** \brief set submatrix into matrix at row m, col n \return current matrix after setting value */
    SPEMat& SPEMat::operator()(
            PetscInt m,         ///< [in] row to set submatrix
            PetscInt n,         ///< [in] column to set submatrix
            const SPEMat &Asub, ///< [in] submatrix to set in matrix
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
        (*this)();//assemble
        return (*this);
    }
    // overloaded operator, assemble
    /** \brief assmelbe the matrix \return the current matrix */
    SPEMat& SPEMat::operator()(){
        ierr = MatAssemblyBegin(mat,MAT_FINAL_ASSEMBLY);CHKERRXX(ierr);
        ierr = MatAssemblyEnd(mat,MAT_FINAL_ASSEMBLY);CHKERRXX(ierr);
        return (*this);
    }
    // overloaded operator, MatAXPY
    /** \brief MatAXPY,  Y = 1.*X + Y operation \return current matrix after operation */
    SPEMat& SPEMat::operator+=(
            const SPEMat &X ///< [in] X in Y+=X operation
            ){
        ierr = MatAXPY(this->mat,1.,X.mat,DIFFERENT_NONZERO_PATTERN);CHKERRXX(ierr);
        return *this;
    }
    /** \brief MatAXPY function call to add a*X to the current mat \return current matrix after operation */
    SPEMat& SPEMat::axpy(
            const PetscScalar a,    ///< [in] scalar a in Y = a*X + Y operation
            const SPEMat &X         ///< [in] matrix X in Y = a*X + Y operation
            ){
        ierr = MatAXPY(this->mat,a,X.mat,DIFFERENT_NONZERO_PATTERN);CHKERRXX(ierr);
        return (*this);
    }
    // overloaded operator, MatAXPY
    /** \brief Y + X operation \return new matrix after operation */
    SPEMat SPEMat::operator+(
            const SPEMat &X ///< [in] X in Y+X operation
            ){
        SPEMat A;
        A=*this;
        ierr = MatAXPY(A.mat,1.,X.mat,DIFFERENT_NONZERO_PATTERN);CHKERRXX(ierr);
        ierr = MatSetType(A.mat,MATMPIAIJ);CHKERRXX(ierr);
        return A;
    }
    // overloaded operator, MatAXPY
    /** \brief Y = -1.*X + Y operation \return Y after operation */
    SPEMat& SPEMat::operator-=(
            const SPEMat &X ///< [in] X in Y = -1.*X + Y operation
            ){
        ierr = MatAXPY(this->mat,-1.,X.mat,DIFFERENT_NONZERO_PATTERN);CHKERRXX(ierr);
        return *this;
    }
    // overloaded operator, MatAXPY
    /** \brief Y - X operation \return new matrix after operation */
    SPEMat SPEMat::operator-(
            const SPEMat &X     ///< [in] X in Y-X operation
            ){
        SPEMat A;
        A=*this;
        ierr = MatAXPY(A.mat,-1.,X.mat,DIFFERENT_NONZERO_PATTERN);CHKERRXX(ierr);
        ierr = MatSetType(A.mat,MATMPIAIJ);CHKERRXX(ierr);
        return A;
    }
    // overload operator, scale with scalar
    /** \brief Y*a operation \return new matrix after operation */
    SPEMat SPEMat::operator*(
            const PetscScalar a     ///< [in] scalar
            ){
        SPEMat A;
        A=*this;
        ierr = MatScale(A.mat,a);CHKERRXX(ierr);
        return A;
    }
    /** \brief Y*a operation \return new matrix after operation */
    SPEMat SPEMat::operator*(
            const double a          ///< [in] scalar
            ){
        SPEMat A;
        A=*this;
        ierr = MatScale(A.mat,a);CHKERRXX(ierr);
        return A;
    }
    /** \brief A*x operation to return a vector \return new vector after operation */
    SPEVec SPEMat::operator*(
            const SPEVec &x         ///< [in] x in A*x matrix vector multiplication
            ){
        SPEVec b(x.rows);
        ierr = MatMult(mat,x.vec,b.vec);CHKERRXX(ierr);
        return b;
    }
    // overload operator, scale with scalar
    /** \brief Y = Y*a operation \return Y after operation */
    SPEMat& SPEMat::operator*=(
            const PetscScalar a     ///< [in] scalar in Y*a operation
            ){
        ierr = MatScale(this->mat,a);CHKERRXX(ierr);
        return *this;
    }
    // overload operator, matrix multiply
    /** \brief Y*A operation \return new matrix after matrix matrix multiply */
    SPEMat SPEMat::operator*(
            const SPEMat &A         ///< [in] A matrix in Y*A operation
            ){
        SPEMat C;
        C.rows=rows;
        C.cols=cols;
        ierr = MatMatMult(mat,A.mat,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&C.mat); CHKERRXX(ierr);
        ierr = MatSetType(C.mat,MATMPIAIJ);CHKERRXX(ierr);
        return C;
    }
    // overload operator, copy and initialize
    /** \brief Y=X with initialization of Y using MatConvert \return Y after operation */
    SPEMat& SPEMat::operator=(
            const SPEMat &A         ///< [in] A in Y=A operation
            ){
        if(flag_init){
            this->~SPEMat();
            //ierr = MatCopy(A.mat,mat,DIFFERENT_NONZERO_PATTERN);CHKERRXX(ierr);
        }
        //else{
            rows=A.rows;
            cols=A.cols;
            ierr = MatConvert(A.mat,MATSAME,MAT_INITIAL_MATRIX,&mat);CHKERRXX(ierr);
            //ierr = MatSetType(mat,MATMPIAIJ);CHKERRXX(ierr);
            flag_init=PETSC_TRUE;
        //}
        return *this;
    }
    // overload % for element wise multiplication
    //SPEMat operator%(SPEMat A){
    //return *this;
    //}     
    /** \brief A = Transpose(*this.mat) operation with initialization of A \return transpose of current matrix */
    PetscInt SPEMat::T(
            SPEMat &A       ///< [out] transpose of current matrix
            ){
        //ierr = MatTranspose(mat,MAT_INITIAL_MATRIX,&A.mat);CHKERRQ(ierr);
        //A();
        //A.Init(cols,rows);
        ierr = MatTranspose(mat,MAT_INITIAL_MATRIX,&A.mat);CHKERRQ(ierr);
        return 0;
    }
    /** \brief Transpose the current mat \return current matrix after transpose */
    SPEMat& SPEMat::T(){
        ierr = MatTranspose(mat,MAT_INPLACE_MATRIX,&mat);CHKERRXX(ierr);
        return (*this);
        //SPEMat T1;
        //ierr = MatCreateTranspose(mat,&T1.mat);CHKERRXX(ierr);
        //SPEMat T1;
        //ierr = MatTranspose(mat,MAT_INITIAL_MATRIX,&T1.mat);CHKERRXX(ierr);
        //return T1;
    }
    /** \brief A = Hermitian Transpose(*this.mat) operation with initialization of A (tranpose and complex conjugate) \return current matrix without hermitian transpose */
    PetscInt SPEMat::H(
            SPEMat &A       ///< [out] hermitian transpose of current matrix saved in new initialized matrix
            ){ // A = Hermitian Transpose(*this.mat) operation with initialization of A (tranpose and complex conjugate)
        ierr = MatHermitianTranspose(mat,MAT_INITIAL_MATRIX,&A.mat);CHKERRQ(ierr);
        return 0;
    }
    /** \brief Hermitian Transpose the current mat \return current matrix after transpose */
    SPEMat& SPEMat::H(){ // Hermitian Transpose the current mat
        ierr = MatHermitianTranspose(mat,MAT_INPLACE_MATRIX,&mat);CHKERRXX(ierr);
        return (*this);
    }
    /** \brief elemenwise conjugate current matrix \return current matrix after conjugate of each element */
    SPEMat& SPEMat::conj(){
        ierr = MatConjugate(mat);CHKERRXX(ierr);
        return (*this);
    }
    /** \brief get diagonal of matrix \return vector of current diagonal */
    SPEVec SPEMat::diag(){ // get diagonal of matrix
        SPEVec d(rows);
        ierr = MatGetDiagonal(mat,d.vec); CHKERRXX(ierr);
        return d;
    }
    // print matrix to screen
    /** \brief print mat to screen using PETSC_VIEWER_STDOUT_WORLD \return 0 if successful */
    PetscInt SPEMat::print(){
        (*this)();
        PetscPrintf(PETSC_COMM_WORLD,("\n---------------- "+name+"---start------\n").c_str());
        ierr = MatView(mat,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
        PetscPrintf(PETSC_COMM_WORLD,("---------------- "+name+"---done-------\n\n").c_str());
        return 0;
    }

    /** \brief destructor to delete memory */
    SPEMat::~SPEMat(){
        flag_init=PETSC_FALSE;
        ierr = MatDestroy(&mat);CHKERRXX(ierr);
    }

    // overload operator, scale with scalar
    /** \brief a*A operation to be equivalent to A*a \return new matrix of a*A */
    SPEMat operator*(
            const PetscScalar a,    ///< [in] scalar a in a*A
            const SPEMat A          ///< [in] matrix A in a*A
            ){
        SPEMat B;
        B=A;
        B.ierr = MatScale(B.mat,a);CHKERRXX(B.ierr);
        return B;
    }
    /** \brief A*a operation to be equivalent to A*a \return new matrix of A*a operation */
    SPEMat operator*(const SPEMat A, const PetscScalar a){
        SPEMat B;
        B=A;
        B.ierr = MatScale(B.mat,a);CHKERRXX(B.ierr);
        return B;
    }
    // overload operator, Linear System solve Ax=b
    /** \brief Solve linear system, Ax=b using b/A notation \return x vector in Ax=b solved linear system */
    SPEVec operator/(
            const SPEVec &b,    ///< [in] b in Ax=b
            const SPEMat &A     ///< [in] A in Ax=b
            ){
        SPEVec x;
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
        //ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);
        //ierr = KSPSetType(ksp,KSPPREONLY);CHKERRXX(ierr);

        PC pc;
        ierr = KSPGetPC(ksp,&pc);CHKERRXX(ierr);
        ierr = PCSetType(pc,PCLU);CHKERRXX(ierr);
        ierr = KSPSetType(ksp,KSPPREONLY);CHKERRXX(ierr);
        //ierr = PCSetOperators(pc,A.mat,A.mat); CHKERRXX(ierr);


        // Solve the linear system
        ierr = KSPSolve(ksp,b.vec,x.vec);CHKERRXX(ierr);

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
    // identity matrix formation
    /** \brief create, form, and return identity matrix of size n \return identity matrix of size nxn */
    SPEMat eye(
            const PetscInt n        ///< [in] n size of square identity matrix
            ){
        SPEMat I(n,"I");
        SPEVec one(n);
        I.ierr = VecSet(one.vec,1.);CHKERRXX(I.ierr);
        I.ierr = MatDiagonalSet(I.mat,one.vec,INSERT_VALUES);CHKERRXX(I.ierr);

        return I;
    }
    // diagonal matrix
    /** \brief set diagonal of matrix \return new matrix with  a diagonal vector set as the main diagonal */
    SPEMat diag(
            const SPEVec &d     ///< [in] diagonal vector to set along main diagonal
            ){ // set diagonal of matrix
        SPEMat A(d.rows);
        A.ierr = MatDiagonalSet(A.mat,d.vec,INSERT_VALUES);CHKERRXX(A.ierr);
        return A;
    }

    // kron inner product
    /** \brief set kronecker inner product of two matrices \return kronecker inner product of the two matrices */
    SPEMat kron(
            const SPEMat &A,    ///< [in] A in A kron B operation
            const SPEMat &B     ///< [in] B in A kron B operation
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
        SPEMat C(nc);

        // kron function C=kron(A,B)
        PetscInt ncols;
        const PetscInt *cols;
        const PetscScalar *vals;
        PetscInt Isubstart,Isubend;
        ierr = MatGetOwnershipRange(A.mat,&Isubstart,&Isubend);CHKERRXX(ierr);
        for (PetscInt rowi=0; rowi<na; rowi++){
            PetscPrintf(PETSC_COMM_WORLD,"kron rowi=%i of %i\n",rowi,m);
            bool onprocessor=(Isubstart<=rowi) and (rowi<Isubend);
            if(onprocessor){
                // extract row of one A
                ierr = MatGetRow(A.mat,rowi,&ncols,&cols,&vals);CHKERRXX(ierr); // extract the one row of A if owned by processor
            }
            else{
                ncols=0;
            }
            PetscInt ncols2=0;
            MPIU_Allreduce(&ncols,&ncols2,1,MPIU_INT,MPIU_SUM,PETSC_COMM_WORLD);

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
            MPIU_Allreduce(cols_temp,cols2,ncols2,MPIU_INT,MPIU_SUM,PETSC_COMM_WORLD);

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

    /** \brief solve general eigenvalue problem of Ax = kBx and return a tuple of tie(PetscScalar alpha, SPEVec eig_vector) \return tuple of eigenvalue and eigenvector closest to the target value \example \snippet std::tie(alpha, eig_vector) = eig(A,B,0.1+0.4*PETSC_i) */
    std::tuple<PetscScalar, SPEVec> eig(
            const SPEMat &A,        ///< [in] A in Ax=kBx generalized eigenvalue problem
            const SPEMat &B,        ///< [in] B in Ax=kBx generalized eigenvalue problem
            const PetscScalar target    ///< [in] target eigenvalue to solve for
            ){
        //TODO use PEP in slepc (polynomial eigenvalue problem) instead of the generalized eigenvalue problem.  Will make things easier, and faster hopefully
        PetscInt rows=A.rows;
        EPS             eps;        /* eigenproblem solver context slepc */
        //ST              st;
        EPSType         type;
        //KSP             ksp;        /* linear solver context petsc */
        PetscErrorCode  ierr;
        PetscScalar ki,alpha;
        SPEVec xi(rows),eig_vec(rows);

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
        if (
                which==EPS_TARGET_REAL ||
                which==EPS_TARGET_IMAGINARY ||
                which==EPS_TARGET_MAGNITUDE){
            // PetscScalar target=0.-88.5*PETSC_i;
            EPSSetTarget(eps,target);
            //EPSSetTolerances(eps,1.E-8,100000);
        }


        // Solve the system
        ierr = EPSSolve(eps);CHKERRXX(ierr);
        //std::cout<<"After KSPSolve"<<std::endl;

        // output iterations
        PetscInt its, maxit, i, nconv;
        PetscReal error, tol, re, im;
        /*
            Optional: Get some information from the solver and display it
        */
        ierr = EPSGetIterationNumber(eps,&its);CHKERRXX(ierr);
        ierr = PetscPrintf(PETSC_COMM_WORLD," Number of iterations of the method: %D\n",its);CHKERRXX(ierr);
        ierr = EPSGetType(eps,&type);CHKERRXX(ierr);
        ierr = PetscPrintf(PETSC_COMM_WORLD," Solution method: %s\n\n",type);CHKERRXX(ierr);
        ierr = EPSGetDimensions(eps,&nev,NULL,NULL);CHKERRXX(ierr);
        ierr = PetscPrintf(PETSC_COMM_WORLD," Number of requested eigenvalues: %D\n",nev);CHKERRXX(ierr);
        ierr = EPSGetTolerances(eps,&tol,&maxit);CHKERRXX(ierr);
        ierr = PetscPrintf(PETSC_COMM_WORLD," Stopping condition: tol=%.4g, maxit=%D\n",(double)tol,maxit);CHKERRXX(ierr);

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
           Display solution and clean up
           - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
        /*
           Get number of converged approximate eigenpairs
           */
        ierr = EPSGetConverged(eps,&nconv);CHKERRXX(ierr);
        ierr = PetscPrintf(PETSC_COMM_WORLD," Number of converged eigenpairs: %D\n\n",nconv);CHKERRXX(ierr);

        if (nconv>0) {
            /*
               Display eigenvalues and relative errors
               */
            ierr = PetscPrintf(PETSC_COMM_WORLD,
                    "      k                ||Ax-kx||/||kx||\n"
                    "   ----------------- ------------------\n");CHKERRXX(ierr);

            for (i=0;i<nconv;i++) {
                /*
                   Get converged eigenpairs: i-th eigenvalue is stored in kr (real part) and
                   ki (imaginary part)
                   */
                ierr = EPSGetEigenpair(eps,i,&alpha,&ki,eig_vec.vec,xi.vec);CHKERRXX(ierr);
                /*
                   Compute the relative error associated to each eigenpair
                   */
                ierr = EPSComputeError(eps,i,EPS_ERROR_RELATIVE,&error);CHKERRXX(ierr);

                re = PetscRealPart(alpha);
                im = PetscImaginaryPart(alpha);
                if (im!=0.0) {
                    ierr = PetscPrintf(PETSC_COMM_WORLD," (%9e+%9ei)  %12g\n",(double)re,(double)im,(double)error);CHKERRXX(ierr);
                } else {
                    ierr = PetscPrintf(PETSC_COMM_WORLD,"   %12e       %12g\n",(double)re,(double)error);CHKERRXX(ierr);
                }
            }
            ierr = PetscPrintf(PETSC_COMM_WORLD,"\n");CHKERRXX(ierr);

            ierr = EPSGetEigenpair(eps,0,&alpha,&ki,eig_vec.vec,xi.vec);CHKERRXX(ierr);
        }

        ierr = EPSGetIterationNumber(eps,&its);CHKERRXX(ierr);
        //ierr = PetscPrintf(PETSC_COMM_WORLD,"ksp iterations %D\n",its);CHKERRXX(ierr);
        PetscPrintf(PETSC_COMM_WORLD,"EPS Solved in %D iterations \n",its);
        // Free work space.  All PETSc objects should be destroyed when they
        // are no longer needed.
        //set_Vec(x);
        ierr = EPSDestroy(&eps);CHKERRXX(ierr);
        //ierr = VecDestroy(&x);CHKERRXX(ierr);
        //ierr = VecDestroy(&b);CHKERRXX(ierr); 
        //ierr = MatDestroy(&A);CHKERRXX(ierr);
        //ierr = PetscFinalize();

        return std::make_tuple(alpha,eig_vec);
        //return std::make_tuple(alpha,alpha);
    }
    /** \brief set block matrices using an input array of size rows*cols.  Fills rows first \return new matrix with inserted blocks */
    SPEMat block(
            const SPEMat Blocks[],  ///< [in] array of matrices to set into larger matrix e.g. { A, B, C, D }
            const PetscInt rows,    ///< [in] number of rows of submatrices e.g. 2
            const PetscInt cols     ///< [in] number of columns of submatrices e.g. 2
            ){
        PetscInt m[rows];
        PetscInt msum=Blocks[0].rows;
        PetscInt n[cols];
        PetscInt nsum=Blocks[0].cols;
        m[0]=n[0]=0;
        for (PetscInt i=1; i<rows; ++i){
            m[i] = m[i-1]+Blocks[i-1].rows;
            msum += m[i];
        }
        for (PetscInt i=1; i<cols; ++i){
            n[i] = n[i-1]+Blocks[i*rows].cols;
            nsum += n[i];
        }

        // TODO check if all rows and columns match for block matrix....

        SPEMat A(msum,nsum);

        for (PetscInt j=0; j<cols; ++j){
            for(PetscInt i=0; i<rows; ++i){
                A(m[i],n[j],Blocks[i+j*rows]);
            }
        }

        return A;
    }
}


