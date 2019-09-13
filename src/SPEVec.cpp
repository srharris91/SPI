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
    /** \brief initialize the vector of size _rows \return 0 if successful */
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

    /** set a scalar value at a position row if owned by processor  \return 0 if successful */
    PetscInt SPEVec::set(
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
    PetscInt SPEVec::set(
            const PetscScalar v   ///< [in] value to set in vec
            ){
        ierr = VecSet(vec,v); CHKERRQ(ierr);
        return 0;
    }
    /** add a scalar value at position row if owned by processor \return 0 if successful */
    PetscInt SPEVec::add(
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

    // overloaded operators, get
    /** get value at row (on all processors) \return scalar value at row specified */
    PetscScalar SPEVec::operator()(
            PetscInt _row, ///< what row to get value
            PetscBool global ///< [in] whether to broadcast value to all processors or not (default is false)
            ) {
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
    PetscInt SPEVec::operator()(
            PetscInt _row,      ///< [in] row to set the value
            const PetscScalar v ///< [in] value to set in the row
            ){
        PetscInt low,high;
        VecGetOwnershipRange(vec,&low,&high);
        if ((low <= _row) && (_row < high)){
            ierr = VecSetValue(vec,_row,v,INSERT_VALUES);CHKERRQ(ierr);
        }
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
    // ^ operator
    /** \brief pow operation pow(this,p) */
    SPEVec SPEVec::operator^(
            const PetscScalar p ///< [in] exponent of this^p operation
            ){
        return pow(*this,p);
    }
    /** \brief pow operation pow(this,p) */
    SPEVec SPEVec::operator^(
            const double p ///< [in] exponent of this^p operation
            ){
        return pow(*this,(PetscScalar)p);
    }
    /** \brief pow operation pow(this,p) */
    SPEVec SPEVec::operator^(
            SPEVec p ///< [in] exponent of this^p operation
            ){
        return pow(*this,p);
    }
    // overload operator, copy and initialize
    /** Y=X with initialization of Y using VecCopy and VecDuplicate \return Y initialized and copied of X */
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
    /** \brief == VecEqual test if this==x2 \returns PETSC_TRUE if this==x2 */
    PetscBool SPEVec::operator==(
            const SPEVec &x2    ///< [in] x2 in test
            ){
        PetscBool iftrue;
        ierr = VecEqual(vec,x2.vec,&iftrue); CHKERRXX(ierr);
        return iftrue;

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
    /** \brief take the real part of the vector \returns the vector after taking the real part of it */
    SPEVec& SPEVec::real(){
        ierr = VecRealPart(vec); CHKERRXX(ierr);
        return (*this);
    }
    /** \brief take the imaginary part of the vector \returns the vector after taking the imaginary part of it */
    SPEVec& SPEVec::imag(){
        ierr = VecImaginaryPart(vec); CHKERRXX(ierr);
        return (*this);
    }
    /** \brief take the inner product of two vectors \returns y^H this  where H is the complex conjugate transpose*/
    PetscScalar SPEVec::dot(
            SPEVec y    ///< [in] second vector in inner product (x,y) or y^H x
            ){
        PetscScalar val;
        ierr = VecDot(vec,y.vec,&val); CHKERRXX(ierr);
        return val;
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
    /** \brief return the real part of the vector */
    SPEVec real(
            const SPEVec &A     ///< [in] vector to take real part of
            ){
        SPEVec B(A);
        return B.real();
    }
    /** \brief return the imaginary part of the vector */
    SPEVec imag(
            const SPEVec &A     ///< [in] vector to take imaginary part of
            ){
        SPEVec B(A);
        return B.imag();
    }
    /** \brief return linspace of number of rows equally spaced points between begin and end */
    SPEVec linspace(
            const PetscScalar begin,    ///< [in] beginning scalar of equally spaced points
            const PetscScalar end,      ///< [in] end scalar of equally spaced points
            const PetscInt rows         ///< [in] how many points in array
            ){ // return linspace of number of rows equally spaced points between begin and end
        SPEVec y(rows);
        PetscScalar step = (end-begin)/((PetscScalar)(rows-1));
        PetscScalar value=begin;
        //PetscInt i=0;
        for (PetscInt i=0; i<rows; ++i){
            y(i,value);
            value += step;
        }
        y();
        return y;
    }

    /** \brief take the function of each element in a vector, e.g. (*f)(A(i)) for each i */
    template <class T>
        SPEVec _Function_on_each_element(
                T (*f)(T const&),   ///< [in] function handle to pass in e.g. std::sin<PetscReal>
                const SPEVec &A     ///< [in] vector to perform function on each element
                ){
            SPEVec out(A);
            for (PetscInt i=0; i<out.rows; ++i){
                out(i,(*f)(out(i)));                // TODO speed up by getting all values at once on local processor and looping through those
            }
            out();
            return out;
        }

    /** \brief take the sin of each element in a vector */
    SPEVec sin(const SPEVec &A){ return _Function_on_each_element(&std::sin<PetscReal>, A); }
    /** \brief take the cos of each element in a vector */
    SPEVec cos(const SPEVec &A){ return _Function_on_each_element(&std::cos<PetscReal>, A); }
    /** \brief take the tan of each element in a vector */
    SPEVec tan(const SPEVec &A){ return _Function_on_each_element(&std::tan<PetscReal>, A); }
    /** \brief take the exp of each element in a vector */
    SPEVec exp(const SPEVec &A){ 
        //return _Function_on_each_element(&std::exp<PetscReal>, A); 
        SPEVec B(A);
        B.ierr = VecExp(B.vec); CHKERRXX(B.ierr);
        return B;
    }
    /** \brief take the log (natural log) of each element in a vector */
    SPEVec log(const SPEVec &A){ return _Function_on_each_element(&std::log<PetscReal>, A); }
    /** \brief take the log10 of each element in a vector */
    SPEVec log10(const SPEVec &A){ return _Function_on_each_element(&std::log10<PetscReal>, A); }
    /** \brief take the sinh of each element in a vector */
    SPEVec sinh(const SPEVec &A){ return _Function_on_each_element(&std::sinh<PetscReal>, A); }
    /** \brief take the cosh of each element in a vector */
    SPEVec cosh(const SPEVec &A){ return _Function_on_each_element(&std::cosh<PetscReal>, A); }
    /** \brief take the tanh of each element in a vector */
    SPEVec tanh(const SPEVec &A){ return _Function_on_each_element(&std::tanh<PetscReal>, A); }
    /** \brief take the asin of each element in a vector */
    SPEVec asin(const SPEVec &A){ return _Function_on_each_element(&std::asin<PetscReal>, A); }
    /** \brief take the acos of each element in a vector */
    SPEVec acos(const SPEVec &A){ return _Function_on_each_element(&std::acos<PetscReal>, A); }
    /** \brief take the atan of each element in a vector */
    SPEVec atan(const SPEVec &A){ return _Function_on_each_element(&std::atan<PetscReal>, A); }
    /** \brief take the asinh of each element in a vector */
    SPEVec asinh(const SPEVec &A){ return _Function_on_each_element(&std::asinh<PetscReal>, A); }
    /** \brief take the acosh of each element in a vector */
    SPEVec acosh(const SPEVec &A){ return _Function_on_each_element(&std::acosh<PetscReal>, A); }
    /** \brief take the atanh of each element in a vector */
    SPEVec atanh(const SPEVec &A){ return _Function_on_each_element(&std::atanh<PetscReal>, A); }
    /** \brief function to take element by element of two vectors e.g. (*f)(A(i),B(i)) for all i */
    template <class T>
        SPEVec _Function_on_each_element(
                T (*f)(T const&, T const&),     ///< [in] function handle to pass in e.g. std::pow<PetscReal>
                const SPEVec &A,                ///< [in] first vector to perform function on each element
                SPEVec &B                 ///< [in] second vector 
                ){
            SPEVec out(A);
            for (PetscInt i=0; i<out.rows; ++i){
                out(i,(*f)(out(i),B(i)));                // TODO speed up by getting all values at once on local processor and looping through those
            }
            out();
            return out;
        }
    /** \brief take the pow of each element in the vectors */
    SPEVec pow(const SPEVec &A,SPEVec &B){ return _Function_on_each_element(&std::pow<PetscReal>, A,B); }
    /** \brief take the pow of each element in the vector (A^b) \returns A^b */
    SPEVec pow(
            const SPEVec &A, ///< [in] vector to raise to the power
            PetscScalar b       ///< [in] the exponenet
            ){
        SPEVec B(A);
        B.ierr = VecPow(B.vec,b);CHKERRXX(B.ierr);
        return B;

    }

    /** \brief take the inner product of the two vectors (i.e. y^H x) where ^H is the complex conjugate transpose*/
    PetscScalar dot(
            SPEVec x,   ///< [in] first vector in inner product
            SPEVec y    ///< [in] second vector in inner product (this one gets the complex conjugate transpose)
            ){
        PetscScalar innerproduct;
        x.ierr = VecDot(x.vec,y.vec,&innerproduct); CHKERRXX(x.ierr);
        return innerproduct;
    }

    /** \brief take the absolute value of a vector */
    SPEVec abs(const SPEVec &A){ 
        SPEVec B(A);
        VecAbs(B.vec);
        return B;
    }

    /** \brief take the sum of a vector */
    PetscScalar sum(
            SPEVec x1   ///< [in] vector to sum
            ){
        PetscScalar sum;
        x1.ierr = VecSum(x1.vec,&sum); CHKERRXX(x1.ierr);
        return sum;
    }



    /** \brief calculate the \f$ L_2 \f$ norm of the difference between \f$x_1\f$ and \f$x_2\f$ vectors.  \returns \f$L_2\f$ norm of the difference */
    PetscReal L2(
            SPEVec x1,      ///< [in] \f$x_1\f$
            const SPEVec x2,      ///< [in] \f$x_2\f$
            NormType type   ///< [in] type of norm (default NORM_2 \f$\sqrt{\sum |x_1 - x_2|^2}\f$) (NORM_1 denotes sum_i |x_i|), (NORM_2 denotes sqrt(sum_i |x_i|^2)), (NORM_INFINITY denotes max_i |x_i|)
            ){
        PetscReal error;
        VecNorm((x1-x2).vec,type,&error);
        return error;
    }


    /** \brief calculate the \f$ L_2 \f$ norm of the vector \returns \f$L_2\f$ norm of the vector */
    PetscReal L2(
            const SPEVec x1,        ///< [in] \f$x_1\f$ l
            NormType type           ///< [in] type of norm (default NORM_2 \f$\sqrt{\sum x_1^2}\f$)
            ){
        PetscReal error;
        VecNorm(x1.vec,type,&error);
        return error;
    }

    /** \brief diff of the vector (see numpy.diff) \returns y[i] = x[i+1]-x[i] for i=0,1,...,x.rows-2 */
    SPEVec diff( 
            SPEVec x       ///< [in] vector to diff (x[i+1]-x[i])
            ){ 
        SPEVec x0(x.rows-1), x1(x.rows-1);
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
            SPEVec y      ///< [in] vector to integrate, assuming default spacing of one
            ){
        SPEVec y0(y.rows-1), y1(y.rows-1);
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
            SPEVec y,     ///< [in] vector to integrate
            SPEVec x      ///< [in] optional, coordinates to integrate over, must be same size as y, and defaults to spacing of one if not given
            ){
        SPEVec y0(y.rows-1), y1(y.rows-1);
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
}


