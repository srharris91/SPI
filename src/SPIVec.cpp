#include "SPIVec.hpp"
// use 1 to use the GPU and 0 or anything else to only use CPU and MPI.  Be sure this matches what is in SPIMat.cpp
#define USE_GPU 0

namespace SPI{

    // constructors
    /** \brief constructor with no arguments (no initialization) */
    SPIVec::SPIVec(
            std::string _name ///< [in] name of SPIVec (important with hdf5 i/o)
            ){name=_name; }
    /** \brief constructor using another SPIVec */
    SPIVec::SPIVec(
            const SPIVec &A, ///< [in] SPIVec to copy and initialize from
            std::string _name ///< [in] name of SPIVec (important with hdf5 i/o)
            ){
        name=_name; 
        (*this) = A;
    }
    /** \brief constructor with one arguement to make vector of length rows */
    SPIVec::SPIVec(
            const PetscInt _rows,  ///< [in] number of rows to initialize vector
            const std::string _name///< [in] name of SPIVec (important with hdf5 i/o)
            ){
        //this->Init(_rows,_name);
        this->name=_name;
        this->rows=_rows;
        //PetscInt _rows2 = this->rows;
        //SPI::printf("_rows at initialization %D",_rows);
        //SPI::printf(" rows at initialization %D",_rows2);
        //std::cout<<" rows at initialization "<<this->rows<<std::endl;
        ierr = VecCreate(PETSC_COMM_WORLD,&vec);CHKERRXX(ierr);
        ierr = VecSetSizes(vec,PETSC_DECIDE,_rows);CHKERRXX(ierr);
#if USE_GPU == 1
        ierr = VecSetType(vec,VECMPICUDA);CHKERRXX(ierr);
#else
        ierr = VecSetType(vec,VECMPI);CHKERRXX(ierr);
#endif
        flag_init=PETSC_TRUE;
    }

    // Initialize vector
    /** \brief initialize the vector of size _rows \return 0 if successful */
    PetscInt SPIVec::Init(
            PetscInt _rows,     ///< [in] number of rows to initialize vector
            const std::string _name   ///< [in] name of SPIVec (important with hdf5 i/o)
            ){
        this->name=_name;
        rows=_rows;
        SPI::printf("_rows at initialization %D",_rows);
        SPI::printf("rows at initialization %D",this->rows);
        ierr = VecCreate(PETSC_COMM_WORLD,&vec);CHKERRQ(ierr);
        ierr = VecSetSizes(vec,PETSC_DECIDE,_rows);CHKERRQ(ierr);
#if USE_GPU == 1
        ierr = VecSetType(vec,VECMPICUDA);CHKERRQ(ierr);
#else
        ierr = VecSetType(vec,VECMPI);CHKERRQ(ierr);
#endif
        flag_init=PETSC_TRUE;
        return 0;
    }

    /** set a scalar value at a position row if owned by processor  \return 0 if successful */
    PetscInt SPIVec::set(
            const PetscInt _row,  ///< [in] position to set value
            const PetscScalar v   ///< [in] value to set in vec
            ){
        PetscInt low,high;
        VecGetOwnershipRange(vec,&low,&high);
        if ((low <= _row) && (_row < high)){
            ierr = VecSetValue(vec,_row,v,INSERT_VALUES);CHKERRQ(ierr);
        }
        return 0;
    }
    /** set a scalar value at all positions \return 0 if successful */
    PetscInt SPIVec::set(
            const PetscScalar v   ///< [in] value to set in vec
            ){
        ierr = VecSet(vec,v); CHKERRQ(ierr);
        return 0;
    }
    /** add a scalar value at position row if owned by processor \return 0 if successful */
    PetscInt SPIVec::add(
            PetscInt _row,      ///< [in] position to add value
            const PetscScalar v ///< [in] value to add at position _row
            ){
        PetscInt low,high;
        VecGetOwnershipRange(vec,&low,&high);
        if ((low <= _row) && (_row < high)){
            ierr = VecSetValue(vec,_row,v,ADD_VALUES);CHKERRQ(ierr);
        }
        return 0;
    }
    // get size of vector
    /** get the global size of the vector */
    PetscInt SPIVec::size(){
        PetscInt n;
        ierr = VecGetSize(vec,&n);CHKERRXX(ierr);
        rows=n; // update rows
        return n;
    }

    // overloaded operators, get
    /** get value at row (on all processors) \return scalar value at row specified */
    PetscScalar SPIVec::operator()(
            PetscInt _row, ///< what row to get value
            PetscBool global ///< [in] whether to broadcast value to all processors or not (default is false)
            )const{
        PetscScalar v,v_global=0.;
        PetscInt low,high;
        PetscErrorCode ierr2;
        ierr2 = VecGetOwnershipRange(vec,&low, &high);CHKERRXX(ierr2);
        if ((low<=_row) && (_row<high)){
            ierr2 = VecGetValues(vec,1,&_row,&v);CHKERRXX(ierr2);
        }
        if (global){
            MPIU_Allreduce(&v,&v_global,1,MPIU_SCALAR,MPIU_SUM,PETSC_COMM_WORLD);
        }
        else{
            v_global=v; // return local value
        }
        return v_global;
    }
    /** get value at row (on all processors) \return scalar value at row specified */
    PetscScalar SPIVec::operator()(
            PetscInt _row, ///< what row to get value
            PetscBool global ///< [in] whether to broadcast value to all processors or not (default is false)
            ){
        PetscScalar v,v_global=0.;
        PetscInt low,high;
        ierr = VecGetOwnershipRange(vec,&low, &high);CHKERRXX(ierr);
        if ((low<=_row) && (_row<high)){
            ierr = VecGetValues(vec,1,&_row,&v);CHKERRXX(ierr);
        }
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
    PetscInt SPIVec::operator()(
            PetscInt _row,      ///< [in] row to set the value
            const PetscScalar v ///< [in] value to set in the row
            ){
        PetscInt low,high;
        VecGetOwnershipRange(vec,&low,&high);
        if ((low <= _row) && (_row < high)){
            ierr = VecSetValue(vec,_row,v,INSERT_VALUES);CHKERRQ(ierr);
        }
        //(*this)(); // assemble after every insertion
        return 0;
    }
    /** same as above */
    PetscInt SPIVec::operator()(PetscInt _row, const double v){
        ierr = (*this)(_row,(PetscScalar)(v+0.0*PETSC_i));CHKERRQ(ierr);
        return 0;
    }
    /** same as above */
    PetscInt SPIVec::operator()(PetscInt _row, const int v){
        ierr = (*this)(_row,(PetscScalar)((double)v+0.0*PETSC_i));CHKERRQ(ierr);
        return 0;
    }

    // overloaded operator, assemble
    /** assemble the vector \return SPIVec of assembled vector */
    SPIVec& SPIVec::operator()(){
        ierr = VecAssemblyBegin(vec);CHKERRXX(ierr);
        ierr = VecAssemblyEnd(vec);CHKERRXX(ierr);
        return (*this);
    }
    // overloaded operator, VecAXPY
    /** VecAXPY Y = 1.*X + Y operation \return SPIVec Y */
    SPIVec& SPIVec::operator+=(
            const SPIVec &X ///< [in] X in Y += X operation
            ){
        ierr = VecAXPY(this->vec,1.,X.vec);CHKERRXX(ierr);
        return *this;
    }
    /** VecAXPY Y=a*X+Y operation to add a*X to the current vec \return SPIVec Y */
    SPIVec& SPIVec::axpy(
            const PetscScalar a,    ///< [in] scalar a in Y=a*X+Y operation
            const SPIVec &X         ///< [in] vec X in Y=a*X+Y operation
            ){
        ierr = VecAXPY(this->vec,a,X.vec);CHKERRXX(ierr);
        return (*this);
    }
    // overloaded operator, VecAXPY
    /** Y+X operation \return SPIVec Z=Y+X */
    SPIVec SPIVec::operator+(
            const SPIVec &X ///< [in] X in Z=Y+X operation
            ){
        SPIVec A;
        A=*this;
        ierr = VecAXPY(A.vec,1.,X.vec);CHKERRXX(ierr);
#if USE_GPU == 1
        ierr = VecSetType(A.vec,VECMPICUDA);CHKERRXX(ierr);
#else
        ierr = VecSetType(A.vec,VECMPI);CHKERRXX(ierr);
#endif
        return A;
    }
    /** Y+a operation \return SPIVec Z=Y+a */
    SPIVec SPIVec::operator+(
            const PetscScalar a ///< [in] scalar a in Y+a operation
            ){ // Y + a operation
        SPIVec A;
        A=(*this);
        A += a*ones(rows);
        return A;
    }
    /** Y+a operation \return SPIVec Z=Y+a */
    SPIVec SPIVec::operator+(
            const double a ///< [in] scalar a in Y+a operation
            ){ // Y + a operation
        SPIVec A;
        A=(*this);
        A += a*ones(rows);
        return A;
    }
    /** Y-a operation \return Z in Z=Y-a */
    SPIVec SPIVec::operator-(
            const PetscScalar a ///< [in] scalar a in Y-a operation
            ){ // Y - a operation
        SPIVec A;
        A=(*this);
        A -= a*ones(rows);
        return A;
    }
    /** Y-a operation \return Z in Z=Y-a */
    SPIVec SPIVec::operator-(
            const PetscInt a ///< [in] scalar a in Y-a operation
            ){ // Y - a operation
        SPIVec A;
        A=(*this);
        A -= a*ones(rows);
        return A;
    }
    // overloaded operator, VecAXPY
    /** Y = -1.*X + Y operation \return Y in Y-=X */
    SPIVec& SPIVec::operator-=(
            const SPIVec &X ///< [in] X in Y=-1*X + Y operation
            ){
        ierr = VecAXPY(this->vec,-1.,X.vec);CHKERRXX(ierr);
        return *this;
    }
    // overloaded operator, VecAXPY
    /** Y - X operation \return Z in Z=Y-X operation */
    SPIVec SPIVec::operator-(
            const SPIVec &X ///< [in] X in Y-X operation
            ){
        SPIVec A;
        A=*this;
        ierr = VecAXPY(A.vec,-1.,X.vec);CHKERRXX(ierr);
#if USE_GPU == 1
        ierr = VecSetType(A.vec,VECMPICUDA);CHKERRXX(ierr);
#else
        ierr = VecSetType(A.vec,VECMPI);CHKERRXX(ierr);
#endif
        return A;
    }
    /** -X operation \return Z in Z=-X operation */
    SPIVec SPIVec::operator-() const{
        return -1.*(*this);
    }
    // overload operator, scale with scalar
    /** Y*a operation \return Z in Z=Y*a */
    SPIVec SPIVec::operator*(
            const PetscScalar a ///< [in] a in Z=Y*a operation
            ){
        SPIVec A;
        A=(*this);
        ierr = VecScale(A.vec,a);CHKERRXX(ierr);
        return A;
    }
    /** same as above */
    SPIVec SPIVec::operator*(const double a){
        PetscScalar as=a;
        SPIVec A;
        A=*this;
        ierr = VecScale(A.vec,as);CHKERRXX(ierr);
        return A;
    }
    // overload operator, scale with scalar
    /** Y = Y*a operation \return Y in Y*=a */
    SPIVec& SPIVec::operator*=(
            const PetscScalar a ///< [in] scalar a in Y*=a operation
            ){
        ierr = VecScale(this->vec,a);CHKERRXX(ierr);
        return *this;
    }
    /** Y = Y*a operation \return Y in Y*=a */
    SPIVec& SPIVec::operator*=(
            const double a ///< [in] scalar a in Y*=a operation
            ){
        ierr = VecScale(this->vec,(PetscScalar)(a+0.*PETSC_i));CHKERRXX(ierr);
        return *this;
    }
    /** Y = Y*a operation \return Y in Y*=a */
    SPIVec& SPIVec::operator*=(
            const SPIVec &a ///< [in] SPIVec a in Y*=a operation
            ){
        ierr = VecPointwiseMult(vec,a.vec,(*this).vec);CHKERRXX(ierr);
        return *this;
    }
    // overload operator, pointwise multiply
    /** Y*X pointwise multiply operation \return Z in Z=Y*X operation */
    SPIVec SPIVec::operator*(
            const SPIVec& X ///< [in] X in Z=Y*X operation
            ){
        SPIVec C;
        C.rows=rows;
        C=(*this);
        ierr = VecPointwiseMult(C.vec,X.vec,(*this).vec);CHKERRXX(ierr);
        // ierr = VecSetType(C.vec,VECMPI);CHKERRXX(ierr);
        return C;
    }
    // overload operator, pointwise divide
    /** pointwise divide by scalar a \return Z in Z=Y/a operation */
    SPIVec SPIVec::operator/(
            const PetscScalar a ///< [in] scalar a in Z=Y/a operation
            ){
        SPIVec A;
        A=(*this);
        ierr = VecScale(A.vec,1./a);CHKERRXX(ierr);
        return A;
    }
    /** same as above */
    SPIVec SPIVec::operator/(const double a){
        PetscScalar as=a;
        SPIVec A;
        A=*this;
        ierr = VecScale(A.vec,1./as);CHKERRXX(ierr);
        return A;
    }
    /** pointwise divide by X  \return Z in Z=Y/X operation */
    SPIVec SPIVec::operator/(
            const SPIVec X ///< [in] X in Z=Y/X operation
            ){
        SPIVec Z(X.rows);
        ierr = VecPointwiseDivide(Z.vec,this->vec,X.vec);CHKERRXX(ierr);
        return Z;
    }
    // overload operator, scale with scalar
    /** Y = Y*a pointwise divide operation \return Y in Y/=a operation */
    SPIVec& SPIVec::operator/=(
            const PetscScalar a ///< [in] scalar a in Y/=a operation
            ){
        ierr = VecScale(this->vec,1./a);CHKERRXX(ierr);
        return *this;
    }
    // ^ operator
    /** \brief pow operation pow(this,p) */
    SPIVec SPIVec::operator^(
            const PetscScalar p ///< [in] exponent of this^p operation
            ){
        return pow(*this,p);
    }
    /** \brief pow operation pow(this,p) */
    SPIVec SPIVec::operator^(
            const double p ///< [in] exponent of this^p operation
            ){
        return pow(*this,(PetscScalar)p);
    }
    /** \brief pow operation pow(this,p) */
    SPIVec SPIVec::operator^(
            const int p ///< [in] exponent of this^p operation
            ){
        return pow(*this,(PetscScalar)p);
    }
    /** \brief pow operation pow(this,p) */
    SPIVec SPIVec::operator^(
            SPIVec p ///< [in] exponent of this^p operation
            ){
        return pow(*this,p);
    }
    // overload operator, copy and initialize
    /** Y=X with initialization of Y using VecCopy and VecDuplicate \return Y initialized and copied of X */
    SPIVec& SPIVec::operator=(const SPIVec &X){
        if(flag_init){
            if(X.rows==this->rows){
                ierr = VecCopy(X.vec,vec);CHKERRXX(ierr); // use copy if size matches
            }
            else{
                this->~SPIVec(); // destroy and recreate from scratch
                this->rows=X.rows;
                ierr = VecDuplicate(X.vec,&vec); CHKERRXX(ierr);
                ierr = VecCopy(X.vec,vec); CHKERRXX(ierr);
            }
        }
        else{
            this->rows=X.rows;
            ierr = VecDuplicate(X.vec,&vec); CHKERRXX(ierr);
            ierr = VecCopy(X.vec,vec); CHKERRXX(ierr);
            flag_init=PETSC_TRUE;
        }
        //ierr = VecSetType(vec,VECMPI);CHKERRXX(ierr);
        return (*this);
    }
    /** \brief == VecEqual test if this==x2 \returns PETSC_TRUE if this==x2 */
    PetscBool SPIVec::operator==(
            const SPIVec &x2    ///< [in] x2 in test
            ){
        PetscBool iftrue;
        ierr = VecEqual(vec,x2.vec,&iftrue); CHKERRXX(ierr);
        return iftrue;

    }


    // overload % for inner product
    //SPIVec operator%(SPIVec A){
    //return *this;
    //}     
    /** elementwise conjugate current vector \return current vector after conjugate \see conj(const SPIVec&) */
    SPIVec& SPIVec::conj(){
        ierr = VecConjugate(vec);CHKERRXX(ierr);
        return (*this);
    }

    /** maximum value of vector \return scalar maximum value of the vector (broadcasted to all processors) */
    PetscScalar SPIVec::max(){
        PetscInt argmax;
        PetscReal max=0.;
        PetscScalar maxscalar;
        ierr = VecMax(this->vec,&argmax,&max);CHKERRXX(ierr);
        maxscalar = (*this)(argmax,PETSC_TRUE);
        //PetscScalar maxscalar_global = 0.;
        //MPIU_Allreduce(&maxscalar,&maxscalar_global,1,MPIU_SCALAR,MPIU_SUM,PETSC_COMM_WORLD);

        return maxscalar;
    }
    /** maximum index value of vector \return integer index of maximum value of the vector */
    PetscInt SPIVec::argmax(){
        PetscInt argmax;
        PetscReal max=0.;
        //PetscScalar maxscalar;
        ierr = VecMax(this->vec,&argmax,&max);CHKERRXX(ierr);
        //maxscalar = (*this)(argmax,PETSC_TRUE);
        //PetscScalar maxscalar_global = 0.;
        //MPIU_Allreduce(&maxscalar,&maxscalar_global,1,MPIU_SCALAR,MPIU_SUM,PETSC_COMM_WORLD);

        return argmax;
    }
    /** \brief take the real part of the vector \returns the vector after taking the real part of it */
    SPIVec& SPIVec::real(){
        ierr = VecRealPart(vec); CHKERRXX(ierr);
        return (*this);
    }
    /** \brief take the imaginary part of the vector \returns the vector after taking the imaginary part of it */
    SPIVec& SPIVec::imag(){
        ierr = VecImaginaryPart(vec); CHKERRXX(ierr);
        return (*this);
    }
    /** \brief take the inner product of two vectors \returns y^H this  where H is the complex conjugate transpose*/
    PetscScalar SPIVec::dot(
            SPIVec y    ///< [in] second vector in inner product (x,y) or y^H x
            ){
        PetscScalar val;
        ierr = VecDot(vec,y.vec,&val); CHKERRXX(ierr);
        return val;
    }


    // print vector to screen
    /** print vec to screen using PETSC_VIEWER_STDOUT_WORLD \return 0 if successful */
    PetscInt SPIVec::print(){
        (*this)();// assemble
        printf("\n---------------- "+this->name+"---start------");
        //PetscPrintf(PETSC_COMM_WORLD,("\n---------------- "+name+"---start------\n").c_str());
        //SPI::printf("shape = %d x 1",this->rows);
        SPI::printf("shape = "+std::to_string(this->rows)+" x 1");
        ierr = VecView(vec,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
        //PetscPrintf(PETSC_COMM_WORLD,("---------------- "+name+"---done-------\n\n").c_str());
        printf("---------------- "+this->name+"---done-------\n");
        return 0;
    }

    /** destructor to delete memory */
    SPIVec::~SPIVec(){
        if(flag_init){
            flag_init=PETSC_FALSE;
            ierr = VecDestroy(&vec);CHKERRXX(ierr);
        }
        //else{
            //(*this)=zeros(1);
            //ierr = VecDestroy(&vec);CHKERRXX(ierr);

        //}
    }

    // overload operator, scale with scalar
    /** Z=a/Y operation \return Z in Z=a/Y operation */
    SPIVec operator/(
            const PetscScalar a, ///< [in] scalar a in a/Y operation
            const SPIVec &Y     ///< [in] Y in Z=a/Y
            ){
        SPIVec B(Y.rows);
        SPIVec A(ones(Y.rows)*a);
        //B=Y;
        //B.ierr = VecScale(B.vec,a);CHKERRXX(B.ierr);
        B.ierr = VecPointwiseDivide(B.vec,A.vec,Y.vec);CHKERRXX(B.ierr);
        return B;
    }

    /** Z=a*Y operation to be equivalent to Y*a \return Z in Z=a*Y operation */
    SPIVec operator*(
            const PetscScalar a, ///< [in] scalar a in a*Y operation
            const SPIVec &Y     ///< [in] Y in Z=a*Y
            ){
        SPIVec B;
        B=Y;
        B.ierr = VecScale(B.vec,a);CHKERRXX(B.ierr);
        return B;
    }

    /** Z=a+Y operation to be equivalent to Y+a \return Z in Z=a+Y operation */
    SPIVec operator+(
            const PetscScalar a,  ///< [in] scalar a in a+Y operation
            const SPIVec &Y         ///< [in] Y in a+Y operation
            ){
        SPIVec B;
        B = Y;
        B += a*ones(B.rows);
        return B;
    }

    /** Z=a-Y operation to be equivalent to Y-a \return Z in Z=a-Y operation */
    SPIVec operator-(
            const PetscScalar a, ///< [in] scalar a in a-Y operation
            const SPIVec &Y     ///< [in] Y in a-Y operation
            ){
        SPIVec B;
        B = -1.*Y;
        B += a*ones(B.rows);
        return B;
    }

    /** save A to hdf5 to filename as variable A.name (note: this will append if filename already exists) \return 0 if successful */
    PetscInt save(
            const SPIVec &A,            ///< [in] A to save in hdf5 format under A.name variable
            const std::string filename  ///< [in] filename to save
            ){ // save A to hdf5 to filename as variable A.name
        PetscErrorCode ierr;
        ierr = PetscObjectSetName((PetscObject)A.vec, A.name.c_str());CHKERRQ(ierr);
        PetscViewer viewer;
        std::ifstream f(filename.c_str());
        if(f.good()){// if filename exists, then append
            ierr = PetscViewerHDF5Open(PETSC_COMM_WORLD, filename.c_str(), FILE_MODE_APPEND, &viewer); CHKERRQ(ierr);
        }
        else{ // if not, then write a new file
            ierr = PetscViewerHDF5Open(PETSC_COMM_WORLD, filename.c_str(), FILE_MODE_WRITE, &viewer); CHKERRQ(ierr);
        }
        ierr = VecView(A.vec,viewer); CHKERRQ(ierr);
        ierr = PetscViewerDestroy(&viewer); CHKERRQ(ierr);
        return 0;
    }
    /** save A to hdf5 to filename as variable A.name (note: this will append if filename already exists) \return 0 if successful */
    PetscInt save(
            std::vector<PetscScalar> A, ///< [in] variable to save to hdf5 file
            std::string variablename,   ///< [in] the name of the variable to save in the file
            std::string filename        ///< [in] hdf5 filename
            ){
        PetscInt n=A.size();
        SPIVec Avec(n,variablename);
        for(PetscInt i=0; i<n; i++){
            Avec(i,A[i]);
        }
        save(Avec,filename);
        return 0;
    }
    /** save A to hdf5 to filename as variable A.name (note: this will append if filename already exists) \return 0 if successful */
    PetscInt save(
            std::vector<SPIVec> A,      ///< [in] variable to save to hdf5 file
            std::string variablename,   ///< [in] the name of the variable to save in the file
            std::string filename        ///< [in] hdf5 filename
            ){
        PetscInt n=A.size();
        std::string sep="_";
        for(PetscInt i=0; i<n; i++){
            A[i].name = variablename+sep+std::to_string(i);
            save(A[i],filename);
        }
        return 0;
    }

    /** load A from hdf5 filename using variable A.name, be sure it has the right size first before loading \return 0 if successful */
    PetscInt load( 
            SPIVec &A,                  ///< [inout] vector to load data into (must be initialized to the right size)
            const std::string filename  ///< [in] filename to read
            ){
        A.ierr = PetscObjectSetName((PetscObject)A.vec, A.name.c_str());CHKERRQ(A.ierr);
        PetscViewer viewer;
        //std::ifstream f(filename.c_str());
        A.ierr = PetscViewerHDF5Open(PETSC_COMM_WORLD, filename.c_str(), FILE_MODE_READ, &viewer); CHKERRQ(A.ierr);
        A.ierr = VecLoad(A.vec,viewer); CHKERRQ(A.ierr);
        A.ierr = PetscViewerDestroy(&viewer); CHKERRQ(A.ierr);
        return 0;
    }

    /** create a vector of size rows full of ones \return vector of size rows */
    SPIVec ones(
            const PetscInt rows ///< [in] number of rows for vector size
            ){
        SPIVec A(rows);
        A.ierr = VecSet(A.vec,1.);CHKERRXX(A.ierr);
        return A;
    }

    /** create and return a vector of size rows full of zeros \return vector of size zeros */
    SPIVec zeros(
            const PetscInt rows ///< [in] size of vector to create
            ){
        SPIVec A(rows);
        A.ierr = VecSet(A.vec,0.);CHKERRXX(A.ierr);
        return A;
    }
    /** return the conjugate of the vector \return conjugate of A \see SPIVec::conj() */
    SPIVec conj(
            const SPIVec &A ///< [in] vector to conjugate
            ){
        SPIVec B;
        B=A;
        B.ierr = VecConjugate(B.vec);CHKERRXX(B.ierr);
        return B;
    }
    /** \brief return the real part of the vector */
    SPIVec real(
            const SPIVec &A     ///< [in] vector to take real part of
            ){
        SPIVec B(A);
        return B.real();
    }
    /** \brief return the imaginary part of the vector */
    SPIVec imag(
            const SPIVec &A     ///< [in] vector to take imaginary part of
            ){
        SPIVec B(A);
        return B.imag();
    }
    /** \brief return linspace of number of rows equally spaced points between begin and end */
    SPIVec linspace(
            const PetscScalar begin,    ///< [in] beginning scalar of equally spaced points
            const PetscScalar end,      ///< [in] end scalar of equally spaced points
            const PetscInt _rows         ///< [in] how many points in array
            ){ // return linspace of number of rows equally spaced points between begin and end
        SPIVec y(_rows);
        //PetscInt _rows2 = y.rows;
        //SPI::printf("y.rows in linspace = %D",_rows2);
        PetscScalar step = (end-begin)/((PetscScalar)(_rows-1));
        PetscScalar value=begin;
        //PetscInt i=0;
        for (PetscInt i=0; i<_rows; ++i){
            y(i,value);
            value += step;
        }
        y();
        //_rows2 = y.rows;
        //SPI::printf("y.rows in linspace = %D",_rows2);
        //SPI::printf("_rows in linspace = %D",_rows);
        return y;
    }
    /** \brief return a range of number of equally spaced points between begin and end by step size step*/
    SPIVec arange(
            const PetscScalar begin,    ///< [in] beginning scalar of equally spaced points
            const PetscScalar end,      ///< [in] end scalar of equally spaced points
            const PetscScalar stepsize  ///< [in] step size for equally spaced points
            ){ // return linspace of number of rows equally spaced points between begin and end
        PetscInt _rows=ceil(PetscRealPart((end-begin)/stepsize));
        SPIVec y(_rows);
        //PetscScalar step = (end-begin)/((PetscScalar)(_rows-1));
        PetscScalar value=begin;
        //PetscInt i=0;
        for (PetscInt i=0; i<_rows; ++i){
            y(i,value);
            value += stepsize;
        }
        y();
        return y;
    }
    SPIVec arange(
            const PetscScalar end      ///< [in] end scalar of equally spaced points
            ){ // return linspace of number of rows equally spaced points between begin and end
        PetscScalar begin=0.;
        return arange(begin,end);
    }

    /** \brief take the function of each element in a vector, e.g. (*f)(A(i)) for each i */
    template <class T>
        SPIVec _Function_on_each_element(
                T (*f)(T const&),   ///< [in] function handle to pass in e.g. std::sin<PetscReal>
                const SPIVec &A     ///< [in] vector to perform function on each element
                ){
            SPIVec out(A);
            for (PetscInt i=0; i<out.rows; ++i){
                out(i,(*f)(out(i)));                // TODO speed up by getting all values at once on local processor and looping through those
            }
            out();
            return out;
        }

    /** \brief take the sin of each element in a vector */
    SPIVec sin(const SPIVec &A){ return _Function_on_each_element(&std::sin<PetscReal>, A); }
    /** \brief take the cos of each element in a vector */
    SPIVec cos(const SPIVec &A){ return _Function_on_each_element(&std::cos<PetscReal>, A); }
    /** \brief take the tan of each element in a vector */
    SPIVec tan(const SPIVec &A){ return _Function_on_each_element(&std::tan<PetscReal>, A); }
    /** \brief take the exp of each element in a vector */
    SPIVec exp(const SPIVec &A){ 
        //return _Function_on_each_element(&std::exp<PetscReal>, A); 
        SPIVec B(A);
        B.ierr = VecExp(B.vec); CHKERRXX(B.ierr);
        return B;
    }
    /** \brief take the log (natural log) of each element in a vector */
    SPIVec log(const SPIVec &A){ return _Function_on_each_element(&std::log<PetscReal>, A); }
    /** \brief take the log10 of each element in a vector */
    SPIVec log10(const SPIVec &A){ return _Function_on_each_element(&std::log10<PetscReal>, A); }
    /** \brief take the sinh of each element in a vector */
    SPIVec sinh(const SPIVec &A){ return _Function_on_each_element(&std::sinh<PetscReal>, A); }
    /** \brief take the cosh of each element in a vector */
    SPIVec cosh(const SPIVec &A){ return _Function_on_each_element(&std::cosh<PetscReal>, A); }
    /** \brief take the tanh of each element in a vector */
    SPIVec tanh(const SPIVec &A){ return _Function_on_each_element(&std::tanh<PetscReal>, A); }
    /** \brief take the asin of each element in a vector */
    SPIVec asin(const SPIVec &A){ return _Function_on_each_element(&std::asin<PetscReal>, A); }
    /** \brief take the acos of each element in a vector */
    SPIVec acos(const SPIVec &A){ return _Function_on_each_element(&std::acos<PetscReal>, A); }
    /** \brief take the atan of each element in a vector */
    SPIVec atan(const SPIVec &A){ return _Function_on_each_element(&std::atan<PetscReal>, A); }
    /** \brief take the asinh of each element in a vector */
    SPIVec asinh(const SPIVec &A){ return _Function_on_each_element(&std::asinh<PetscReal>, A); }
    /** \brief take the acosh of each element in a vector */
    SPIVec acosh(const SPIVec &A){ return _Function_on_each_element(&std::acosh<PetscReal>, A); }
    /** \brief take the atanh of each element in a vector */
    SPIVec atanh(const SPIVec &A){ return _Function_on_each_element(&std::atanh<PetscReal>, A); }
    /** \brief take the atanh of each element in a vector */
    SPIVec sqrt(const SPIVec &A){ return _Function_on_each_element(&std::sqrt<PetscReal>, A); }
    /** \brief take the erf of each element in a vector */
    SPIVec erf(const SPIVec &A){
            SPIVec out(A);
            for (PetscInt i=0; i<out.rows; ++i){
                out(i,std::erf((double)(PetscRealPart(out(i)))));                // TODO speed up by getting all values at once on local processor and looping through those
            }
            out();
            return out;
    }
    /** \brief take the erfc(z) = 1-erf(z) of each element in a vector */
    SPIVec erfc(const SPIVec &A){
            SPIVec out(A);
            for (PetscInt i=0; i<out.rows; ++i){
                out(i,std::erfc((double)(PetscRealPart(out(i)))));                // TODO speed up by getting all values at once on local processor and looping through those
            }
            out();
            return out;
    }
    /** \brief function to take element by element of two vectors e.g. (*f)(A(i),B(i)) for all i */
    template <class T>
        SPIVec _Function_on_each_element(
                T (*f)(T const&, T const&),     ///< [in] function handle to pass in e.g. std::pow<PetscReal>
                const SPIVec &A,                ///< [in] first vector to perform function on each element
                SPIVec &B                 ///< [in] second vector 
                ){
            SPIVec out(A);
            for (PetscInt i=0; i<out.rows; ++i){
                out(i,(*f)(out(i),B(i)));                // TODO speed up by getting all values at once on local processor and looping through those
            }
            out();
            return out;
        }
    /** \brief take the pow of each element in the vectors */
    SPIVec pow(const SPIVec &A,SPIVec &B){ return _Function_on_each_element(&std::pow<PetscReal>, A,B); }
    /** \brief take the pow of each element in the vector (A^b) \returns A^b */
    SPIVec pow(
            const SPIVec &A, ///< [in] vector to raise to the power
            PetscScalar b       ///< [in] the exponenet
            ){
        SPIVec B(A);
        B.ierr = VecPow(B.vec,b);CHKERRXX(B.ierr);
        return B;

    }

    /** \brief take the inner product of the two vectors (i.e. y^H x) where ^H is the complex conjugate transpose*/
    PetscScalar dot(
            SPIVec x,   ///< [in] first vector in inner product
            SPIVec y    ///< [in] second vector in inner product (this one gets the complex conjugate transpose)
            ){
        PetscScalar innerproduct;
        x.ierr = VecDot(x.vec,y.vec,&innerproduct); CHKERRXX(x.ierr);
        return innerproduct;
    }

    /** \brief take the absolute value of a vector */
    SPIVec abs(const SPIVec &A){ 
        SPIVec B(A);
        VecAbs(B.vec);
        return B;
    }

    /** \brief take the sum of a vector */
    PetscScalar sum(
            SPIVec x1   ///< [in] vector to sum
            ){
        PetscScalar sum;
        x1.ierr = VecSum(x1.vec,&sum); CHKERRXX(x1.ierr);
        return sum;
    }
    /** \brief integrate a vector of chebyshev Coefficients on a grid */
    PetscScalar integrate_coeffs(
            const SPIVec &a     ///< [in] chebyshev coefficients to integrate
            ){
        PetscInt n=a.rows;
        PetscInt N=n-1;
        SPIVec d(n+1,"d");
        PetscScalar value=0.0,d0=0.0;
        for(PetscInt i=1; i<=N+1; ++i){
            if(i==1){
                value=0.5*(2.0*a(i-1,PETSC_TRUE)-a(i+1,PETSC_TRUE));
            }
            else if(i==N+1){
                value = 0.5/((PetscScalar)i)*(2.0*a(i-1,PETSC_TRUE));
            }
            else if(i==N){
                value = 0.5/((PetscScalar)i)*(1.0*a(i-1,PETSC_TRUE));
            }
            else{
                value = 0.5/((PetscScalar)i)*(1.0*a(i-1,PETSC_TRUE) - a(i+1,PETSC_TRUE));
            }
            d(i,value);
            if((i%2)==0){ // even
                d0 -= value;
            }
            else{ // odd
                d0 += value;
            }
        }
        d(0,d0);
        d(); // assemble
        return sum(d);
    }
    /** \brief integrate a vector of chebyshev Coefficients on a stretched grid */
    PetscScalar integrate_coeffs(
            const SPIVec &a,      ///< [in] chebyshev coefficients
            const SPIVec &y       ///< [in] Cheby_mapped_y grid
            ){
        PetscScalar ai=y(0,PETSC_TRUE);
        PetscScalar bi=y(y.rows-1,PETSC_TRUE);
        PetscScalar mul = (bi-ai)/2.0;
        return mul*integrate_coeffs(a);
    }



    /** \brief calculate the \f$ L_2 \f$ norm of the difference between \f$x_1\f$ and \f$x_2\f$ vectors.  \returns \f$L_2\f$ norm of the difference */
    PetscReal L2(
            SPIVec x1,      ///< [in] \f$x_1\f$
            const SPIVec x2,      ///< [in] \f$x_2\f$
            NormType type   ///< [in] type of norm (default NORM_2 \f$\sqrt{\sum |x_1 - x_2|^2}\f$) (NORM_1 denotes sum_i |x_i|), (NORM_2 denotes sqrt(sum_i |x_i|^2)), (NORM_INFINITY denotes max_i |x_i|)
            ){
        PetscReal error;
        VecNorm((x1-x2).vec,type,&error);
        return error;
    }


    /** \brief calculate the \f$ L_2 \f$ norm of the vector \returns \f$L_2\f$ norm of the vector */
    PetscReal L2(
            const SPIVec x1,        ///< [in] \f$x_1\f$ l
            NormType type           ///< [in] type of norm (default NORM_2 \f$\sqrt{\sum x_1^2}\f$)
            ){
        PetscReal error;
        VecNorm(x1.vec,type,&error);
        return error;
    }

    /** \brief diff of the vector (see numpy.diff) \returns y[i] = x[i+1]-x[i] for i=0,1,...,x.rows-2 */
    SPIVec diff( 
            SPIVec x       ///< [in] vector to diff (x[i+1]-x[i])
            ){ 
        SPIVec x0(x.rows-1), x1(x.rows-1);
        // set x0=x[i] and x1=x[i+1]
        for (PetscInt i=0; i<x.rows-1; ++i){
            x0(i,x(i,PETSC_TRUE));
            x1(i,x(i+1,PETSC_TRUE));
        }
        // assemble
        x0();
        x1();
        // return difference x[i+1]-x[i]
        return x1-x0;
    }

    /** \brief trapezoidal integration of y with unity coordinate spacing, \f$\int y dx \f$ \returns integrated value */
    PetscScalar trapz(
            SPIVec y      ///< [in] vector to integrate, assuming default spacing of one
            ){
        SPIVec y0(y.rows-1), y1(y.rows-1);
        // set y0=y[i] and y1=y[i+1]
        for (PetscInt i=0; i<y.rows-1; ++i){
            y0(i,y(i,PETSC_TRUE));
            y1(i,y(i+1,PETSC_TRUE));
        }
        // assemble
        y0();
        y1();
        // return trapezoidal rule sum((y[i+1]+y[i])/2.
        return sum((y1+y0)/2.);
    }

    /** \brief trapezoidal integration of y with x coordinates, \f$\int y dx \f$ \returns integrated value */
    PetscScalar trapz(
            SPIVec y,     ///< [in] vector to integrate
            SPIVec x      ///< [in] optional, coordinates to integrate over, must be same size as y, and defaults to spacing of one if not given
            ){
        SPIVec y0(y.rows-1), y1(y.rows-1);
        // set y0=y[i] and y1=y[i+1]
        for (PetscInt i=0; i<y.rows-1; ++i){
            y0(i,y(i,PETSC_TRUE));
            y1(i,y(i+1,PETSC_TRUE));
        }
        // assemble
        y0();
        y1();
        // return trapezoidal rule sum((y[i+1]+y[i])/2. * diff(x))
        return sum((y1+y0)/2. * diff(x));
    }
    /** \brief draw nonzero structure and wait at command line input */
    PetscInt draw(
            const SPIVec &x     ///< [in] vector to draw
            ){
        PetscViewer viewer;
        PetscErrorCode ierr;
        ierr = PetscViewerDrawOpen(PETSC_COMM_WORLD,NULL,x.name.c_str(),PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE,&viewer);CHKERRQ(ierr);
        ierr = VecView(x.vec,viewer);CHKERRQ(ierr);
        SPI::printf("  draw(SPIVec) with title=%s, hit Enter to continue",x.name.c_str());
        std::cin.ignore();

        ierr = PetscViewerDestroy(&viewer); CHKERRQ(ierr);
        return 0;
    }
    
}


