#include "SPEMat.hpp"

namespace SPE{

    // constructors
    SPEMat::SPEMat(std::string _name){name=_name; }
    SPEMat::SPEMat(const SPEMat &A, std::string _name){
        name=_name; 
        (*this) = A;
    }
    SPEMat::SPEMat(PetscInt rowscols, std::string _name){
        Init(rowscols,rowscols,_name);
    }
    SPEMat::SPEMat(PetscInt rowsm, PetscInt colsn, std::string _name){
        Init(rowsm,colsn,_name);
    }

    // Initialize matrix
    PetscInt SPEMat::Init(PetscInt m,PetscInt n, std::string _name){
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

    SPEMat& SPEMat::set(PetscInt m, PetscInt n,const PetscScalar v){
        ierr = MatSetValue(mat,m,n,v,INSERT_VALUES);CHKERRXX(ierr);
        return (*this);
    }
    SPEMat& SPEMat::add(PetscInt m, PetscInt n, const PetscScalar v){
        ierr = MatSetValue(mat,m,n,v,ADD_VALUES);CHKERRXX(ierr);
        return (*this);
    }

    // overloaded operators, get
    PetscScalar SPEMat::operator()(PetscInt m, PetscInt n) {
        PetscScalar v;
        ierr = MatGetValues(mat,1,&m, 1,&n, &v);
        return v;
    }
    // overloaded operator, set
    PetscInt SPEMat::operator()(PetscInt m, PetscInt n, const PetscScalar v){
        ierr = MatSetValue(mat,m,n,v,INSERT_VALUES);CHKERRQ(ierr);
        return 0;
    }
    PetscInt SPEMat::operator()(PetscInt m, PetscInt n, const double v){
        ierr = (*this)(m,n,(PetscScalar)v);CHKERRQ(ierr);
        return 0;
    }
    PetscInt SPEMat::operator()(PetscInt m, PetscInt n, const int v){
        ierr = (*this)(m,n,(PetscScalar)v);CHKERRQ(ierr);
        return 0;
    }

    // overloaded operator, set
    PetscInt SPEMat::operator()(PetscInt m, PetscInt n,const SPEMat &Asub, InsertMode addv){
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
    SPEMat& SPEMat::operator()(){
        ierr = MatAssemblyBegin(mat,MAT_FINAL_ASSEMBLY);CHKERRXX(ierr);
        ierr = MatAssemblyEnd(mat,MAT_FINAL_ASSEMBLY);CHKERRXX(ierr);
        return (*this);
    }
    // overloaded operator, MatAXPY
    SPEMat& SPEMat::operator+=(const SPEMat &X){
        ierr = MatAXPY(this->mat,1.,X.mat,DIFFERENT_NONZERO_PATTERN);CHKERRXX(ierr);
        return *this;
    }
    SPEMat& SPEMat::axpy(const PetscScalar a, const SPEMat &X){
        ierr = MatAXPY(this->mat,a,X.mat,DIFFERENT_NONZERO_PATTERN);CHKERRXX(ierr);
        return (*this);
    }
    // overloaded operator, MatAXPY
    SPEMat SPEMat::operator+(const SPEMat &X){
        SPEMat A;
        A=*this;
        ierr = MatAXPY(A.mat,1.,X.mat,DIFFERENT_NONZERO_PATTERN);CHKERRXX(ierr);
        ierr = MatSetType(A.mat,MATMPIAIJ);CHKERRXX(ierr);
        return A;
    }
    // overloaded operator, MatAXPY
    SPEMat& SPEMat::operator-=(const SPEMat &X){
        ierr = MatAXPY(this->mat,-1.,X.mat,DIFFERENT_NONZERO_PATTERN);CHKERRXX(ierr);
        return *this;
    }
    // overloaded operator, MatAXPY
    SPEMat SPEMat::operator-(const SPEMat &X){
        SPEMat A;
        A=*this;
        ierr = MatAXPY(A.mat,-1.,X.mat,DIFFERENT_NONZERO_PATTERN);CHKERRXX(ierr);
        ierr = MatSetType(A.mat,MATMPIAIJ);CHKERRXX(ierr);
        return A;
    }
    // overload operator, scale with scalar
    SPEMat SPEMat::operator*(const PetscScalar a){
        SPEMat A;
        A=*this;
        ierr = MatScale(A.mat,a);CHKERRXX(ierr);
        return A;
    }
    SPEMat SPEMat::operator*(const double a){
        SPEMat A;
        A=*this;
        ierr = MatScale(A.mat,a);CHKERRXX(ierr);
        return A;
    }
    SPEVec SPEMat::operator*(const SPEVec &x){
        SPEVec b(x.rows);
        ierr = MatMult(mat,x.vec,b.vec);CHKERRXX(ierr);
        return b;
    }
    // overload operator, scale with scalar
    SPEMat& SPEMat::operator*=(const PetscScalar a){
        ierr = MatScale(this->mat,a);CHKERRXX(ierr);
        return *this;
    }
    // overload operator, matrix multiply
    SPEMat SPEMat::operator*(const SPEMat &A){
        SPEMat C;
        C.rows=rows;
        C.cols=cols;
        ierr = MatMatMult(mat,A.mat,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&C.mat); CHKERRXX(ierr);
        ierr = MatSetType(C.mat,MATMPIAIJ);CHKERRXX(ierr);
        return C;
    }
    // overload operator, copy and initialize
    SPEMat& SPEMat::operator=(const SPEMat &A){
        if(flag_init){
            this->~SPEMat();
            //ierr = MatCopy(A.mat,mat,DIFFERENT_NONZERO_PATTERN);CHKERRXX(ierr);
        }
        //else{
            rows=A.rows;
            cols=A.cols;
            ierr = MatConvert(A.mat,MATSAME,MAT_INITIAL_MATRIX,&mat);CHKERRXX(ierr);
            ierr = MatSetType(mat,MATMPIAIJ);CHKERRXX(ierr);
            flag_init=PETSC_TRUE;
        //}
        return *this;
    }
    // overload % for element wise multiplication
    //SPEMat operator%(SPEMat A){
    //return *this;
    //}     
    PetscInt SPEMat::T(SPEMat &A){
        //ierr = MatTranspose(mat,MAT_INITIAL_MATRIX,&A.mat);CHKERRQ(ierr);
        //A();
        //A.Init(cols,rows);
        ierr = MatTranspose(mat,MAT_INITIAL_MATRIX,&A.mat);CHKERRQ(ierr);
        return 0;
    }
    SPEMat& SPEMat::T(){
        ierr = MatTranspose(mat,MAT_INPLACE_MATRIX,&mat);CHKERRXX(ierr);
        //SPEMat T1;
        //ierr = MatCreateTranspose(mat,&T1.mat);CHKERRXX(ierr);
        return (*this);
    }
    PetscInt SPEMat::H(SPEMat &A){ // A = Hermitian Transpose(*this.mat) operation with initialization of A (tranpose and complex conjugate)
        ierr = MatHermitianTranspose(mat,MAT_INITIAL_MATRIX,&A.mat);
        return 0;
    }
    SPEMat& SPEMat::H(){ // Hermitian Transpose the current mat
        ierr = MatHermitianTranspose(mat,MAT_INPLACE_MATRIX,&mat);CHKERRXX(ierr);
        return (*this);
    }
    SPEMat& SPEMat::conj(){
        ierr = MatConjugate(mat);CHKERRXX(ierr);
        return (*this);
    }
    SPEVec SPEMat::diag(){ // get diagonal of matrix
        SPEVec d(rows);
        ierr = MatGetDiagonal(mat,d.vec); CHKERRXX(ierr);
        return d;
    }
    // print matrix to screen
    PetscInt SPEMat::print(){
        (*this)();
        PetscPrintf(PETSC_COMM_WORLD,("\n---------------- "+name+"---start------\n").c_str());
        ierr = MatView(mat,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
        PetscPrintf(PETSC_COMM_WORLD,("---------------- "+name+"---done-------\n\n").c_str());
        return 0;
    }

    SPEMat::~SPEMat(){
        flag_init=PETSC_FALSE;
        ierr = MatDestroy(&mat);CHKERRXX(ierr);
    }

    // overload operator, scale with scalar
    SPEMat operator*(const PetscScalar a, const SPEMat A){
        SPEMat B;
        B=A;
        B.ierr = MatScale(B.mat,a);CHKERRXX(B.ierr);
        return B;
    }
    SPEMat operator*(const SPEMat A, const PetscScalar a){
        SPEMat B;
        B=A;
        B.ierr = MatScale(B.mat,a);CHKERRXX(B.ierr);
        return B;
    }
    // overload operator, Linear System solve Ax=b
    SPEVec operator/(const SPEVec &b, const SPEMat &A){
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
    SPEMat eye(const PetscInt n){
        SPEMat I(n,"I");
        SPEVec one(n);
        I.ierr = VecSet(one.vec,1.);CHKERRXX(I.ierr);
        I.ierr = MatDiagonalSet(I.mat,one.vec,INSERT_VALUES);CHKERRXX(I.ierr);

        return I;
    }
    // diagonal matrix
    SPEMat diag(const SPEVec &d){ // set diagonal of matrix
        SPEMat A(d.rows);
        A.ierr = MatDiagonalSet(A.mat,d.vec,INSERT_VALUES);CHKERRXX(A.ierr);
        return A;
    }

    // kron inner product
    SPEMat kron(const SPEMat &A, const SPEMat &B){
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

    std::tuple<PetscScalar, SPEVec> eig(const SPEMat &A, const SPEMat &B, const PetscScalar target){
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


}


