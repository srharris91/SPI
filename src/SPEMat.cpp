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

    PetscInt SPEMat::set(PetscInt m, PetscInt n,const PetscScalar v){
        ierr = MatSetValue(mat,m,n,v,INSERT_VALUES);CHKERRQ(ierr);
        return 0;
    }
    PetscInt SPEMat::add(PetscInt m, PetscInt n, const PetscScalar v){
        ierr = MatSetValue(mat,m,n,v,ADD_VALUES);CHKERRQ(ierr);
        return 0;
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
    PetscInt SPEMat::operator()(){
        ierr = MatAssemblyBegin(mat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
        ierr = MatAssemblyEnd(mat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
        return 0;
    }
    // overloaded operator, MatAXPY
    SPEMat& SPEMat::operator+=(const SPEMat &X){
        ierr = MatAXPY(this->mat,1.,X.mat,DIFFERENT_NONZERO_PATTERN);CHKERRXX(ierr);
        return *this;
    }
    PetscInt SPEMat::axpy(const PetscScalar a, const SPEMat &X){
        ierr = MatAXPY(this->mat,a,X.mat,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
        return 0;
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
    PetscInt SPEMat::T(){
        ierr = MatTranspose(mat,MAT_INPLACE_MATRIX,&mat);CHKERRQ(ierr);
        //SPEMat T1;
        //ierr = MatCreateTranspose(mat,&T1.mat);CHKERRXX(ierr);
        return 0;
    }
    PetscInt SPEMat::H(SPEMat &A){ // A = Hermitian Transpose(*this.mat) operation with initialization of A (tranpose and complex conjugate)
        ierr = MatHermitianTranspose(mat,MAT_INITIAL_MATRIX,&A.mat);
        return 0;
    }
    PetscInt SPEMat::H(){ // Hermitian Transpose the current mat
        ierr = MatHermitianTranspose(mat,MAT_INPLACE_MATRIX,&mat);
        return 0;
    }
    PetscInt SPEMat::conj(){
        ierr = MatConjugate(mat);CHKERRQ(ierr);
        return 0;
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


}


