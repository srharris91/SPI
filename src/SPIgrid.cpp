#include "SPIgrid.hpp"

namespace SPI{

    /** \brief compute the factorial of n.  This is needed for get_D_Coeffs function \return factorial of n*/
    PetscInt factorial(
            PetscInt n ///< [in] integer to compute factorial of
            ) {
        PetscInt value = 1;
        for(int i=1; i<=n; ++i) value *= i;
        return value;
    }

    /** \brief get the coefficients of the given stencil. \return coefficients of stencil for derivative at 0 in s n*/
    SPIVec get_D_Coeffs(
            SPIVec &s, ///< [in] vector of stencil points e.g. [-3,-2,-1,0,1]
            PetscInt d ///< [in] order of desired derivative e.g. 1 or 2
            ){
        // get_D_coeffs function with inputs s,d
        PetscInt N = s.rows;
        SPIVec spow(N,"spow");
        spow = ones(N);
        SPIMat A(N,"A");
        for(PetscInt i=0; i<N; i++){
            //std::cout<<"i = "<<i<<std::endl;
            if(i>=1){
                //spow = (s^i); // doesn't work.... :(
                spow *= s;
            }
            //SPI::draw(spow);
            //spow.print();
            for(PetscInt j=0; j<N; j++){
                //std::cout<<"i,j = "<<i<<","<<j<<std::endl;
                A(i,j,spow(j,PETSC_TRUE));
            }
        }
        A();
        //A.print();
        SPIVec b(zeros(N),"b");
        b(d,factorial(d));
        b();
        //b.print();
        SPIVec x(N,"x");
        x = (b/A);
        x();
        //x.print();

        return x;
    }

    /** \brief map the derivative operator to the proper y grid \return mapped derivative to streched grid */
    SPIMat map_D(
            SPIMat D,  ///< [in] derivative operator on uniform grid from 0 to 1 (xi grid)
            SPIVec y,  ///< [in] potentially non-uniform grid from 0 to y_max
            PetscInt d,     ///< [in] order of derivative
            PetscInt order ///< [in] order of accuracy for finite difference stencil (default 4)
            ){
        // map_D
        if(d==1){
            //SPI::SPIVec dydxi(D*y);
            //SPI::SPIMat dxidy(SPI::diag(1./(D*y)));
            //dxidy.print();
            SPIMat Dy(diag(1./(D*y))*D);
            //(Dy*y).print();
            return Dy;
        }
        else if(d==2){
            SPIMat D1(set_D(y,1,order,PETSC_TRUE),"D1");
            SPIVec dxidy(1./(D1*y),"dxidy");
            SPIVec d2xidy2(-1.*(D*y)*(dxidy^3),"d2xidy2");
            SPIMat Dy(diag(dxidy^2)*D + (diag(d2xidy2)*D1),"Dy");
            //(Dy*(y)).print();
            return Dy;
        }
        else{
        }
    }

    /** \brief set the derivative operator for the proper y grid if uniform=false.  Uses map_D function \return mapped derivative to streched grid */
    SPIMat set_D(
            SPIVec &y,          ///< [in] grid points
            PetscInt d,         ///< [in] order of derivative
            PetscInt order,     ///< [in] order of accuracy of derivative (default 4)
            PetscBool uniform   ///< [in] is this for a uniform grid? (default false)
            ){
        //PetscInt d=1;
        // set_D
        //PetscInt order=4;
        //PetscInt n=21;
        PetscInt n=y.size();
        SPIVec xi = (linspace(0.,1.,n));
        PetscScalar h = xi(1,PETSC_TRUE)-xi(0,PETSC_TRUE);
        SPIVec one(ones(n),"ones");
        SPIMat I(eye(n),"I");
        PetscInt N = order+d;
        if(N>n) {
            PetscErrorCode ierr=1;
            CHKERRXX(ierr);
        }
        PetscInt Nm1 = N-1;
        if (d%2 != 0) Nm1 += 1; // increase for odd derivative
        SPIVec s(arange(Nm1)-(Nm1-1)/2,"s"); // set stencil
        PetscInt smax = s(s.rows-1,PETSC_TRUE).real();

        SPIVec Coeffs(get_D_Coeffs(s,d),"Coeffs");

        SPIMat D(n,"D");
        D();
        for(PetscInt i=0; i<s.rows; i++){
            //diag_to_add.~SPIMat(); // destroy to free memory
            PetscInt k=(PetscInt)s(i,PETSC_TRUE).real();
            PetscInt nmk=n-std::abs(k);
            D += diag(Coeffs(i,PETSC_TRUE)*ones(nmk),k);

            //SPI::diag(Coeffs(0,PETSC_TRUE)*SPI::one(10),-1).print()
        }
        //D.print();
        // BCs
        D.ierr = MatSetOption(D.mat,MAT_NEW_NONZERO_LOCATIONS,PETSC_TRUE);CHKERRXX(D.ierr);
        for(PetscInt i=0; i<smax; i++){
            // for ith row
            s.~SPIVec(); // deallocate
            if(d%2!=0){// odd derivative
                s = (arange(Nm1-1)-i); // stencil for shifted diff of order-1
            }
            else{
                s = arange(Nm1)-i;// stencil for shifted diff of order-1
            }
            Coeffs.~SPIVec();
            Coeffs = get_D_Coeffs(s,d);
            D.zero_row(i);
            for(PetscInt j=0; j<s.rows; j++){
                PetscInt sj=s(j,PETSC_TRUE).real();
                //SPI::printf("setting D(%d,%d)=%g",i,sj+i,Coeffs(j,PETSC_TRUE));
                D(i,sj+i,Coeffs(j,PETSC_TRUE));
            }
            D();
            // for -ith-1 row
            s.~SPIVec(); // deallocate
            if(d%2!=0){// odd derivative
                s = -1*(arange(Nm1-1)-i); // stencil for shifted diff of order-1
            }
            else{
                s = -1*arange(Nm1)-i;// stencil for shifted diff of order-1
            }
            Coeffs.~SPIVec();
            Coeffs = get_D_Coeffs(s,d);
            D.zero_row(D.rows-1-i);
            for(PetscInt j=0; j<s.rows; j++){
                PetscInt sj=s(j,PETSC_TRUE).real();
                //SPI::printf("setting D(%d,%d)=%g",i,sj+i,Coeffs(j,PETSC_TRUE));
                D(D.rows-1-i,D.cols-1+sj-i,Coeffs(j,PETSC_TRUE));
            }
            D();
        }
        D.ierr = MatSetOption(D.mat,MAT_NEW_NONZERO_LOCATIONS,PETSC_FALSE);CHKERRXX(D.ierr);
        D();
        D*=(1./std::pow(h,d));
        D();
        if(uniform==PETSC_FALSE){
            SPIMat Dy(map_D(D,y,d,order));
            return Dy;
        }
        else{
            return D;
        }
    }


}
