#include "SPEVec.hpp"

namespace SPE{

    // constructors
    SPEVec::SPEVec(std::string _name){name=_name; }
    SPEVec::SPEVec(SPEVec &A, std::string _name){
        name=_name; 
        (*this) = A;
    }
    SPEVec::SPEVec(PetscInt _rows, std::string _name){
        Init(_rows,_name);
    }

    // Initialize vector
    PetscInt SPEVec::Init(PetscInt _rows, std::string _name){
        name=_name;
        rows=_rows;
        ierr = VecCreate(PETSC_COMM_WORLD,&vec);CHKERRQ(ierr);
        ierr = VecSetSizes(vec,PETSC_DECIDE,_rows);CHKERRQ(ierr);
        ierr = VecSetType(vec,VECMPI);CHKERRQ(ierr);
        flag_init=PETSC_TRUE;
        return 0;
    }

    PetscInt SPEVec::set(const PetscInt _row, const PetscScalar v){
        ierr = VecSetValue(vec,_row,v,INSERT_VALUES);CHKERRQ(ierr);
        return 0;
    }
    PetscInt SPEVec::add(PetscInt _row, const PetscScalar v){
        ierr = VecSetValue(vec,_row,v,ADD_VALUES);CHKERRQ(ierr);
        return 0;
    }

    // overloaded operators, get
    PetscScalar SPEVec::operator()(PetscInt _row) {
        PetscScalar v;
        ierr = VecGetValues(vec,1,&_row,&v);
        return v;
    }
    // overloaded operator, set
    PetscInt SPEVec::operator()(PetscInt _row, const PetscScalar v){
        ierr = VecSetValue(vec,_row,v,INSERT_VALUES);CHKERRQ(ierr);
        return 0;
    }
    PetscInt SPEVec::operator()(PetscInt _row, const double v){
        ierr = (*this)(_row,(PetscScalar)v);CHKERRQ(ierr);
        return 0;
    }
    PetscInt SPEVec::operator()(PetscInt _row, const int v){
        ierr = (*this)(_row,(PetscScalar)v);CHKERRQ(ierr);
        return 0;
    }

    // overloaded operator, assemble
    PetscInt SPEVec::operator()(){
        ierr = VecAssemblyBegin(vec);CHKERRQ(ierr);
        ierr = VecAssemblyEnd(vec);CHKERRQ(ierr);
        return 0;
    }
    // overloaded operator, VecAXPY
    SPEVec& SPEVec::operator+=(const SPEVec &X){
        ierr = VecAXPY(this->vec,1.,X.vec);CHKERRXX(ierr);
        return *this;
    }
    PetscInt SPEVec::axpy(const PetscScalar a, const SPEVec &X){
        ierr = VecAXPY(this->vec,a,X.vec);CHKERRQ(ierr);
        return 0;
    }
    // overloaded operator, VecAXPY
    SPEVec SPEVec::operator+(const SPEVec &X){
        SPEVec A;
        A=*this;
        ierr = VecAXPY(A.vec,1.,X.vec);CHKERRXX(ierr);
        ierr = VecSetType(A.vec,VECMPI);CHKERRXX(ierr);
        return A;
    }
    // overloaded operator, VecAXPY
    SPEVec& SPEVec::operator-=(const SPEVec &X){
        ierr = VecAXPY(this->vec,-1.,X.vec);CHKERRXX(ierr);
        return *this;
    }
    // overloaded operator, VecAXPY
    SPEVec SPEVec::operator-(const SPEVec &X){
        SPEVec A;
        A=*this;
        ierr = VecAXPY(A.vec,-1.,X.vec);CHKERRXX(ierr);
        ierr = VecSetType(A.vec,VECMPI);CHKERRXX(ierr);
        return A;
    }
    // overload operator, scale with scalar
    SPEVec SPEVec::operator*(const PetscScalar a){
        SPEVec A;
        (*this).print();
        A=(*this);
        A.print();
        ierr = VecScale(A.vec,a);CHKERRXX(ierr);
        return A;
    }
    SPEVec SPEVec::operator*(const double a){
        PetscScalar as=a;
        SPEVec A;
        A=*this;
        ierr = VecScale(A.vec,as);CHKERRXX(ierr);
        return A;
    }
    // overload operator, scale with scalar
    SPEVec& SPEVec::operator*=(const PetscScalar a){
        ierr = VecScale(this->vec,a);CHKERRXX(ierr);
        return *this;
    }
    // overload operator, pointwise multiply
    SPEVec SPEVec::operator*(const SPEVec& A){
        SPEVec C;
        C.rows=rows;
        C=(*this);
        ierr = VecPointwiseMult(C.vec,A.vec,(*this).vec);CHKERRXX(ierr);
        // ierr = VecSetType(C.vec,VECMPI);CHKERRXX(ierr);
        return C;
    }
    // overload operator, copy and initialize
    SPEVec& SPEVec::operator=(const SPEVec &A){
        if(flag_init){
            ierr = VecCopy(A.vec,vec);CHKERRXX(ierr);
        }
        else{
            rows=A.rows;
            ierr = VecDuplicate(A.vec,&vec); CHKERRXX(ierr);
            ierr = VecCopy(A.vec,vec); CHKERRXX(ierr);
            //ierr = VecSetType(vec,VECMPI);CHKERRXX(ierr);
            flag_init=PETSC_TRUE;
        }
        return (*this);
    }
    // overload % for inner product
    //SPEVec operator%(SPEVec A){
    //return *this;
    //}     
    PetscInt SPEVec::conj(){
        ierr = VecConjugate(vec);CHKERRQ(ierr);
        return 0;
    }
    // print vector to screen
    PetscInt SPEVec::print(){
        (*this)();// assemble
        PetscPrintf(PETSC_COMM_WORLD,("\n---------------- "+name+"---start------\n").c_str());
        ierr = VecView(vec,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
        PetscPrintf(PETSC_COMM_WORLD,("---------------- "+name+"---done-------\n\n").c_str());
        return 0;
    }

    SPEVec::~SPEVec(){
        ierr = VecDestroy(&vec);CHKERRXX(ierr);
    }

    // overload operator, scale with scalar
    SPEVec operator*(const PetscScalar a, SPEVec &A){
        A.ierr = VecScale(A.vec,a);CHKERRXX(A.ierr);
        return A;
    }
}


