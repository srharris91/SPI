#include "SPEVec.hpp"
#include <petscviewerhdf5.h>
#include "SPEprint.hpp"

namespace SPE{

    // constructors
    /** \brief constructor with no arguments (no initialization) */
    SPEVec::SPEVec(
            std::string _name ///< [in] name of SPEVec (important with hdf5 i/o)
            ){name=_name; }
    /** \brief constructor using another SPEVec */
    SPEVec::SPEVec(
            const SPEVec &A, ///< [in] SPEVec to copy and initialize from
            std::string _name ///< [in] name of SPEVec (important with hdf5 i/o)
            ){
        name=_name; 
        (*this) = A;
    }
    /** \brief constructor with one arguement to make vector of length rows */
    SPEVec::SPEVec(
            PetscInt _rows,  ///< [in] number of rows to initialize vector
            std::string _name///< [in] name of SPEVec (important with hdf5 i/o)
            ){
        Init(_rows,_name);
    }

    // Initialize vector
    /** \brief initialize the matrix of size _rows \return 0 if successful */
    PetscInt SPEVec::Init(
            PetscInt _rows,     ///< [in] number of rows to initialize vector
            std::string _name   ///< [in] name of SPEVec (important with hdf5 i/o)
            ){
        name=_name;
        rows=_rows;
        ierr = VecCreate(PETSC_COMM_WORLD,&vec);CHKERRQ(ierr);
        ierr = VecSetSizes(vec,PETSC_DECIDE,_rows);CHKERRQ(ierr);
        ierr = VecSetType(vec,VECMPI);CHKERRQ(ierr);
        flag_init=PETSC_TRUE;
        return 0;
    }

    /** set a scalar value at a position row  \return 0 if successful */
    PetscInt SPEVec::set(
            const PetscInt _row,  ///< [in] position to set value
            const PetscScalar v   ///< [in] value to set in vec
            ){
        ierr = VecSetValue(vec,_row,v,INSERT_VALUES);CHKERRQ(ierr);
        return 0;
    }
    /** add a scalar value at position row \return 0 if successful */
    PetscInt SPEVec::add(
            PetscInt _row,      ///< [in] position to add value
            const PetscScalar v ///< [in] value to add at position _row
            ){
        ierr = VecSetValue(vec,_row,v,ADD_VALUES);CHKERRQ(ierr);
        return 0;
    }

    // overloaded operators, get
    /** get value at row (on all processors) \return scalar value at row specified */
    PetscScalar SPEVec::operator()(
            PetscInt _row, ///< what row to get value
            PetscBool global ///< [in] whether to broadcast value to all processors or not (default is false)
            ) {
        PetscScalar v,v_global=0.;
        ierr = VecGetValues(vec,1,&_row,&v);
        if (global){
            MPIU_Allreduce(&v,&v_global,1,MPIU_SCALAR,MPIU_SUM,PETSC_COMM_WORLD);
        }
        else{
            v_global=v; // return local value
        }
        return v_global;
    }
    // overloaded operator, set
    /** set operator the same as set function \return 0 if successful */
    PetscInt SPEVec::operator()(
            PetscInt _row,      ///< [in] row to set the value
            const PetscScalar v ///< [in] value to set in the row
            ){
        ierr = VecSetValue(vec,_row,v,INSERT_VALUES);CHKERRQ(ierr);
        return 0;
    }
    /** same as above */
    PetscInt SPEVec::operator()(PetscInt _row, const double v){
        ierr = (*this)(_row,(PetscScalar)v);CHKERRQ(ierr);
        return 0;
    }
    /** same as above */
    PetscInt SPEVec::operator()(PetscInt _row, const int v){
        ierr = (*this)(_row,(PetscScalar)v);CHKERRQ(ierr);
        return 0;
    }

    // overloaded operator, assemble
    /** assemble the vector \return SPEVec of assembled vector */
    SPEVec& SPEVec::operator()(){
        ierr = VecAssemblyBegin(vec);CHKERRXX(ierr);
        ierr = VecAssemblyEnd(vec);CHKERRXX(ierr);
        return (*this);
    }
    // overloaded operator, VecAXPY
    /** VecAXPY Y = 1.*X + Y operation \return SPEVec Y */
    SPEVec& SPEVec::operator+=(
            const SPEVec &X ///< [in] X in Y += X operation
            ){
        ierr = VecAXPY(this->vec,1.,X.vec);CHKERRXX(ierr);
        return *this;
    }
    /** VecAXPY Y=a*X+Y operation to add a*X to the current vec \return SPEVec Y */
    SPEVec& SPEVec::axpy(
            const PetscScalar a,    ///< [in] scalar a in Y=a*X+Y operation
            const SPEVec &X         ///< [in] vec X in Y=a*X+Y operation
            ){
        ierr = VecAXPY(this->vec,a,X.vec);CHKERRXX(ierr);
        return (*this);
    }
    // overloaded operator, VecAXPY
    /** Y+X operation \return SPEVec Z=Y+X */
    SPEVec SPEVec::operator+(
            const SPEVec &X ///< [in] X in Z=Y+X operation
            ){
        SPEVec A;
        A=*this;
        ierr = VecAXPY(A.vec,1.,X.vec);CHKERRXX(ierr);
        ierr = VecSetType(A.vec,VECMPI);CHKERRXX(ierr);
        return A;
    }
    /** Y+a operation \return SPEVec Z=Y+a */
    SPEVec SPEVec::operator+(
            const PetscScalar a ///< [in] scalar a in Y+a operation
            ){ // Y + a operation
        SPEVec A;
        A=(*this);
        A += a*ones(rows);
        return A;
    }
    /** Y-a operation \return Z in Z=Y-a */
    SPEVec SPEVec::operator-(
            const PetscScalar a ///< [in] scalar a in Y-a operation
            ){ // Y - a operation
        SPEVec A;
        A=(*this);
        A -= a*ones(rows);
        return A;
    }
    // overloaded operator, VecAXPY
    /** Y = -1.*X + Y operation \return Y in Y-=X */
    SPEVec& SPEVec::operator-=(
            const SPEVec &X ///< [in] X in Y=-1*X + Y operation
            ){
        ierr = VecAXPY(this->vec,-1.,X.vec);CHKERRXX(ierr);
        return *this;
    }
    // overloaded operator, VecAXPY
    /** Y - X operation \return Z in Z=Y-X operation */
    SPEVec SPEVec::operator-(
            const SPEVec &X ///< [in] X in Y-X operation
            ){
        SPEVec A;
        A=*this;
        ierr = VecAXPY(A.vec,-1.,X.vec);CHKERRXX(ierr);
        ierr = VecSetType(A.vec,VECMPI);CHKERRXX(ierr);
        return A;
    }
    // overload operator, scale with scalar
    /** Y*a operation \return Z in Z=Y*a */
    SPEVec SPEVec::operator*(
            const PetscScalar a ///< [in] a in Z=Y*a operation
            ){
        SPEVec A;
        A=(*this);
        ierr = VecScale(A.vec,a);CHKERRXX(ierr);
        return A;
    }
    /** same as above */
    SPEVec SPEVec::operator*(const double a){
        PetscScalar as=a;
        SPEVec A;
        A=*this;
        ierr = VecScale(A.vec,as);CHKERRXX(ierr);
        return A;
    }
    // overload operator, scale with scalar
    /** Y = Y*a operation \return Y in Y*=a */
    SPEVec& SPEVec::operator*=(
            const PetscScalar a ///< [in] scalar a in Y*=a operation
            ){
        ierr = VecScale(this->vec,a);CHKERRXX(ierr);
        return *this;
    }
    // overload operator, pointwise multiply
    /** Y*X pointwise multiply operation \return Z in Z=Y*X operation */
    SPEVec SPEVec::operator*(
            const SPEVec& X ///< [in] X in Z=Y*X operation
            ){
        SPEVec C;
        C.rows=rows;
        C=(*this);
        ierr = VecPointwiseMult(C.vec,X.vec,(*this).vec);CHKERRXX(ierr);
        // ierr = VecSetType(C.vec,VECMPI);CHKERRXX(ierr);
        return C;
    }
    // overload operator, pointwise divide
    /** pointwise divide by scalar a \return Z in Z=Y/a operation */
    SPEVec SPEVec::operator/(
            const PetscScalar a ///< [in] scalar a in Z=Y/a operation
            ){
        SPEVec A;
        A=(*this);
        ierr = VecScale(A.vec,1./a);CHKERRXX(ierr);
        return A;
    }
    /** same as above */
    SPEVec SPEVec::operator/(const double a){
        PetscScalar as=a;
        SPEVec A;
        A=*this;
        ierr = VecScale(A.vec,1./as);CHKERRXX(ierr);
        return A;
    }
    // overload operator, scale with scalar
    /** Y = Y*a pointwise divide operation \return Y in Y/=a operation */
    SPEVec& SPEVec::operator/=(
            const PetscScalar a ///< [in] scalar a in Y/=a operation
            ){
        ierr = VecScale(this->vec,1./a);CHKERRXX(ierr);
        return *this;
    }
    // overload operator, copy and initialize
    /** Y=X with initialization of Y using MatConvert \return Y initialized and copied of X */
    SPEVec& SPEVec::operator=(const SPEVec &X){
        if(flag_init){
            ierr = VecCopy(X.vec,vec);CHKERRXX(ierr);
        }
        else{
            rows=X.rows;
            ierr = VecDuplicate(X.vec,&vec); CHKERRXX(ierr);
            ierr = VecCopy(X.vec,vec); CHKERRXX(ierr);
            //ierr = VecSetType(vec,VECMPI);CHKERRXX(ierr);
            flag_init=PETSC_TRUE;
        }
        return (*this);
    }
    // overload % for inner product
    //SPEVec operator%(SPEVec A){
    //return *this;
    //}     
    /** elementwise conjugate current vector \return current vector after conjugate \see conj(const SPEVec&) */
    SPEVec& SPEVec::conj(){
        ierr = VecConjugate(vec);CHKERRXX(ierr);
        return (*this);
    }

    /** maximum value of vector \return scalar maximum value of the vector (broadcasted to all processors) */
    PetscScalar SPEVec::max(){
        PetscInt argmax;
        PetscReal max=0.;
        PetscScalar maxscalar;
        ierr = VecMax(this->vec,&argmax,&max);CHKERRXX(ierr);
        maxscalar = (*this)(argmax,PETSC_TRUE);
        //PetscScalar maxscalar_global = 0.;
        //MPIU_Allreduce(&maxscalar,&maxscalar_global,1,MPIU_SCALAR,MPIU_SUM,PETSC_COMM_WORLD);

        return maxscalar;
    }
    // print vector to screen
    /** print vec to screen using PETSC_VIEWER_STDOUT_WORLD \return 0 if successful */
    PetscInt SPEVec::print(){
        (*this)();// assemble
        printf("\n---------------- "+name+"---start------");
        //PetscPrintf(PETSC_COMM_WORLD,("\n---------------- "+name+"---start------\n").c_str());
        ierr = VecView(vec,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
        //PetscPrintf(PETSC_COMM_WORLD,("---------------- "+name+"---done-------\n\n").c_str());
        printf("---------------- "+name+"---done-------\n");
        return 0;
    }

    /** destructor to delete memory */
    SPEVec::~SPEVec(){
        ierr = VecDestroy(&vec);CHKERRXX(ierr);
    }

    // overload operator, scale with scalar
    /** Z=a*Y operation to be equivalent to Y*a \return Z in Z=a*Y operation */
    SPEVec operator*(
            const PetscScalar a, ///< [in] scalar a in a*Y operation
            const SPEVec &Y     ///< [in] Y in Z=a*Y
            ){
        SPEVec B;
        B=Y;
        B.ierr = VecScale(B.vec,a);CHKERRXX(B.ierr);
        return B;
    }

    /** Z=a+Y operation to be equivalent to Y+a \return Z in Z=a+Y operation */
    SPEVec operator+(
            const PetscScalar a,  ///< [in] scalar a in a+Y operation
            const SPEVec &Y         ///< [in] Y in a+Y operation
            ){
        SPEVec B;
        B = Y;
        B += a*ones(B.rows);
        return B;
    }

    /** Z=a-Y operation to be equivalent to Y-a \return Z in Z=a-Y operation */
    SPEVec operator-(
            const PetscScalar a, ///< [in] scalar a in a-Y operation
            const SPEVec &Y     ///< [in] Y in a-Y operation
            ){
        SPEVec B;
        B = Y;
        B -= a*ones(B.rows);
        return B;
    }

    /** save A to hdf5 to filename as variable A.name (note: this will append if filename already exists) \return 0 if successful */
    PetscInt save(
            const SPEVec &A,            ///< [in] A to save in hdf5 format under A.name variable
            const std::string filename  ///< [in] filename to save
            ){ // save A to hdf5 to filename as variable A.name
        PetscErrorCode ierr;
        ierr = PetscObjectSetName((PetscObject)A.vec, A.name.c_str());CHKERRQ(ierr);
        PetscViewer viewer;
        std::ifstream f(filename.c_str());
        if(f.good()){
            ierr = PetscViewerHDF5Open(PETSC_COMM_WORLD, filename.c_str(), FILE_MODE_APPEND, &viewer); CHKERRQ(ierr);
        }
        else{
            ierr = PetscViewerHDF5Open(PETSC_COMM_WORLD, filename.c_str(), FILE_MODE_WRITE, &viewer); CHKERRQ(ierr);
        }
        ierr = VecView(A.vec,viewer); CHKERRQ(ierr);
        ierr = PetscViewerDestroy(&viewer); CHKERRQ(ierr);
        return 0;
    }

    /** load A from hdf5 filename using variable A.name, be sure it has the right size first before loading \return 0 if successful */
    PetscInt load( 
            SPEVec &A,                  ///< [inout] vector to load data into (must be initialized to the right size)
            const std::string filename  ///< [in] filename to read
            ){
        A.ierr = PetscObjectSetName((PetscObject)A.vec, A.name.c_str());CHKERRQ(A.ierr);
        PetscViewer viewer;
        std::ifstream f(filename.c_str());
        A.ierr = PetscViewerHDF5Open(PETSC_COMM_WORLD, filename.c_str(), FILE_MODE_READ, &viewer); CHKERRQ(A.ierr);
        A.ierr = VecLoad(A.vec,viewer); CHKERRQ(A.ierr);
        A.ierr = PetscViewerDestroy(&viewer); CHKERRQ(A.ierr);
        return 0;
    }

    /** createa vector of size rows full of ones \return vector of size rows */
    SPEVec ones(
            const PetscInt rows ///< [in] number of rows for vector size
            ){
        SPEVec A(rows);
        A.ierr = VecSet(A.vec,1.);CHKERRXX(A.ierr);
        return A;
    }

    /** create and return a vector of size rows full of zeros \return vector of size zeros */
    SPEVec zeros(
            const PetscInt rows ///< [in] size of vector to create
            ){
        SPEVec A(rows);
        A.ierr = VecSet(A.vec,0.);CHKERRXX(A.ierr);
        return A;
    }
    /** return the conjugate of the vector \return conjugate of A \see SPEVec::conj() */
    SPEVec conj(
            const SPEVec &A ///< [in] vector to conjugate
            ){
        SPEVec B;
        B=A;
        B.ierr = VecConjugate(B.vec);CHKERRXX(B.ierr);
        return B;
    }
}


