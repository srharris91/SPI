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
        //x();
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
            SPIMat Dy(diag(dxidy*dxidy)*D + (diag(d2xidy2)*D1),"Dy");
            //(Dy*(y)).print();
            return Dy;
        }
        else{
            SPI::printf("--------------Something wrong with map_D call -------------");
            exit(0);
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
        if ((d%2) != 0) Nm1 += 1; // increase for odd derivative
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
            if((d%2)!=0){// odd derivative
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
            if((d%2)!=0){// odd derivative
                s = -(arange(Nm1-1)-i); // stencil for shifted diff of order-1
            }
            else{
                s = -(arange(Nm1)-i);// stencil for shifted diff of order-1
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

    /** \brief set the derivative operator for the proper periodic grid assuming uniform discretization.  \return derivative operator for uniform periodic grid */
    SPIMat set_D_periodic(
            SPIVec &y,          ///< [in] grid points
            PetscInt d,         ///< [in] order of derivative
            PetscInt order      ///< [in] order of accuracy of derivative (default 4)
            ){
        PetscInt n=y.size();
        PetscScalar h = y(1,PETSC_TRUE)-y(0,PETSC_TRUE);
        SPIVec one(ones(n),"ones");
        SPIMat I(eye(n),"I");
        PetscInt N = order+d;
        if(N>n) {
            PetscErrorCode ierr=1;
            CHKERRXX(ierr);
        }
        PetscInt Nm1 = N-1;
        if ((d%2) != 0) Nm1 += 1; // increase for odd derivative
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
            if(k>0){
                D += diag(Coeffs(i,PETSC_TRUE)*ones(k),-nmk);
            }
            else if(k<0){
                D += diag(Coeffs(i,PETSC_TRUE)*ones(-k),k+n);
            }

        }
        D();
        D*=(1./std::pow(h,d));
        D();
        return D;
    }
    /** \brief set stretched grid from [0,y_max] using tanh stretching for use with finite difference operators \return y stretched grid */
    SPIVec set_FD_stretched_y(
            PetscScalar y_max,      ///< [in] upper bound of y in [0,y_max] 
            PetscInt ny,            ///< [in] number of points to use in the domain
            PetscScalar delta       ///< [in] stretching parameter, near zero is no stretching, (default 2.0001)
            ){
        //SPI::SPIVec y((y_max*(1.+(SPI::tanh(delta*((linspace(0.,y_max,ny)/y_max) - 1.))/tanh(delta)))));
        //return y;
        return y_max*(1.+(SPI::tanh(delta*((linspace(0.,y_max,ny)/y_max) - 1.))/tanh(delta)));
    }

    /** \brief set a Chebyshev collocated operator acting with respect to the collocated grid \return Chebyshev collocated derivative operator */
    SPIMat set_D_Chebyshev(
            SPIVec &x,          ///< [in] grid (created from set_Cheby_stretched_y or set_Cheby_y or similar example)
            PetscInt d,         ///< [in] order of the derivative (default 1)
            PetscBool need_map  ///< [in] need mapping?  (default false)
            ){
        if(need_map){
            SPIMat D(map_D_Chebyshev(x,d));
            return D;
        }
        else{
            PetscInt N=x.rows-1;
            //PetscInt order=-1;
            SPIMat D(zeros(N+1,N+1));
            SPIVec c(ones(N+1));
            c(0,2.);
            c(N,2.);
            for(PetscInt j=0; j<N+1; j++){
                PetscScalar cj=c(j,PETSC_TRUE);
                PetscScalar xj=x(j,PETSC_TRUE);
                for(PetscInt k=0; k<N+1; k++){
                    PetscScalar ck=c(k,PETSC_TRUE);
                    PetscScalar xk=x(k,PETSC_TRUE);
                    if(j!=k){
                        D(j,k,cj*pow(-1.,(PetscScalar)((double)(j+k)+0.0*PETSC_i)) / (ck*(xj-xk)));
                    }
                    else if((j==k) && ((j!=0) && (j!=N))){
                        D(j,k,-xj/(2.*(1.-(pow(xj,2)))));
                    }
                    else if((j==k) && (j==0)){
                        D(j,k,-1.0*(2.*pow((PetscScalar)((double)N+0.0*PETSC_i),2.)+1.)/6.); // make sure xj goes from -1 to 1
                    }
                    else if((j==k) && (j==N)){
                        D(j,k,1.0*(2.*pow((PetscScalar)((double)N+0.0*PETSC_i),2.)+1.)/6.);
                    }
                    else{
                        std::cout<<"you messed up in set_D_Chebyshev"<<std::endl;
                        exit(0);
                    }
                }

            }
            D();
            if(d==2){
                SPIMat D2(D*D,"Dyy");
                return D2;
            }else{
                return D;
            }
        }
    }


    /** \brief map a Chebyshev collocated operator acting with respect to the stretched collocated grid \return Chebyshev collocated derivative operator */
    SPIMat map_D_Chebyshev(
            SPIVec &x,          ///< [in] grid (created from set_Cheby_stretched_y or set_Cheby_y or similar example)
            PetscInt d          ///< [in] order of the derivative (default 1)
            ){
        PetscInt N=x.rows-1;
        SPIVec xi(set_Cheby_y(N+1));
        if(d==1){
            SPIMat D1(set_D_Chebyshev(xi,1,PETSC_FALSE));
            SPIMat D(diag(1./(D1*x))*D1,"Dy");
            return D;
        }
        else if(d==2){
            SPIMat D1(set_D_Chebyshev(xi,1,PETSC_FALSE));
            SPIMat D2(D1*D1);
            SPIVec dxdxi(D1*x);
            SPIVec dxidx(1./dxdxi);
            SPIVec d2xidx2(-1.*(D2*x)*(dxidx^3));
            SPIMat D((diag(dxidx^2))*D2 + (diag(d2xidx2)*D1),"Dyy");
            return D;
        }
        else{
            std::cout<<"order of derivative is not implemented in map_D_Chebyshev"<<std::endl;
            exit(0);
        }
    }

    /** \brief create a stretched Chebyshev grid from [0,y_max] with yi being the midpoint location for the number of points \return stretched grid using Chebyshev collocated points */
    SPIVec set_Cheby_stretched_y(
            PetscScalar y_max,      ///< [in] maximum range for [0, y_max] stretching
            PetscInt ny,            ///< [in] number of points
            PetscScalar yi          ///< [in] midpoint location for the number of points (i.e. ny/2 are above and below this point)
            ){
        SPIVec xi(set_Cheby_y(ny)); // default grid [-1,1]
        PetscScalar a=yi*y_max/(y_max-2.*yi);
        PetscScalar b=1.+2.*a/y_max;
        SPIVec y(a*(1.+xi)/(b-xi),"y");
        return y;
    }

    /** \brief create a stretched Chebyshev grid from [a,b] using default Chebfun like mapping \return grid in [a,b] using Chebyshev collocated points */
    SPIVec set_Cheby_mapped_y(
            PetscScalar a,          ///< [in] lower bound of grid domain [a,b]
            PetscScalar b,          ///< [in] upper bound of grid domain [a,b]
            PetscInt ny             ///< [in] number of points on grid
            ){
        SPIVec xi(set_Cheby_y(ny)); // default grid [-1,1]
        SPIVec y(b*(xi+1.0)/2.0+a*(1.0-xi)/2.0,"y"); // grid from [a,b]
        return y;
    }

    /** \brief create a Chebyshev grid from [-1,1] \return grid using Chebyshev collocated points */
    SPIVec set_Cheby_y(
            PetscInt ny             ///< [in] number of points
            ){
        //PetscScalar pi = 2.*std::acos(0.0);
        SPIVec y(cos(PETSC_PI*arange((PetscScalar)ny-1.,-1.,-1.)/((PetscScalar)ny-1.)),"y");
        return y;
    }

    /** \brief set a UltraSpherical operator acting with respect to the collocated grid (keeps everything in UltraSpherical space), take in Chebyshev coefficients and outputs coefficients in C(d) coefficient space \return UltraSpherical derivative operator */
    std::tuple<SPIMat,SPIMat> set_D_UltraS(
            SPIVec &x,         ///< [in] grid (must be created from set_Cheby_y or set_Cheby_mapped_y)
            PetscInt d         ///< [in] order of the derivative (default 1)
            ){
        PetscInt n=x.rows;
        PetscScalar a=x(0,PETSC_TRUE);
        PetscScalar b=x(n-1,PETSC_TRUE);
        SPIMat dxidx(diag(ones(n)*(2.0/(b-a))),"dxi/dx"); // assuming it is made from set_Cheby_mapped_y (identity matrix if created from set_Cheby_y)
        //SPIVec xi(cos(PETSC_PI*arange((PetscScalar)n-1.,-1.,-1.)/((PetscScalar)n-1.)),"y");
        if(d==1){
            SPIMat S0(diag(ones(n)*0.5) + diag(ones(n-2)*-0.5,2),"S0");
            S0(0,0,1.0);
            S0();
            //std::cout<<"in set_D_UltraS"<<std::endl;
            //S0.print();
            SPIMat D1((diag(arange(1,n),1))*dxidx,"D1");
            return std::make_tuple(S0,D1);
        }
        else if(d==2){
            SPIMat S1(diag(1.0/(arange(n)+1)) + diag(-1.0/(arange(2,n)+1),2),"S1");
            SPIMat D2((2.0*diag(arange(2,n),2))*(dxidx*dxidx),"D2");
            return std::make_tuple(S1,D2);
        }
        else{
            SPI::printf("Warning: d>2 not implemented in set_D_UltraS function");
            exit(1);
        }
    }
    /** \brief set a T and That operators acting with respect to the Chebyshev collocated grid, That: physical -> Chebyshev coefficient, T: Chebyshev coefficient -> physical \return UltraSpherical derivative operator */
    std::tuple<SPIMat,SPIMat> set_T_That(
            PetscInt n           ///< [in] number of discretized points in Chebyshev grid
            ){
        PetscInt N=n-1;
        SPIVec xi(cos(PETSC_PI*arange((PetscScalar)n-1.,-1.,-1.)/((PetscScalar)n-1.)),"y");
        SPIMat X(n,n);
        SPIMat Ns(n,n); // X and Ns meshgrid for T and That creation
        SPIVec ns(arange(n));
        std::tie(X,Ns) = meshgrid(xi,ns);
        SPIMat T(cos(Ns%acos(X)),"T");
        SPIMat That; // initialized in next line
        T.T(That);  // initialize That and set it to be T.T()
        SPIMat Thatcopy(That);
        for(PetscInt i=0; i<n; ++i) That(i,0,Thatcopy(i,0,PETSC_TRUE)/2.0);
        for(PetscInt i=0; i<n; ++i) That(i,n-1,Thatcopy(i,n-1,PETSC_TRUE)/2.0);
        That();
        Thatcopy=That;
        for(PetscInt i=0; i<n; ++i) That(0,i,Thatcopy(0,i,PETSC_TRUE)/2.0);
        for(PetscInt i=0; i<n; ++i) That(n-1,i,Thatcopy(n-1,i,PETSC_TRUE)/2.0);
        That();
        That *= (2.0/((PetscScalar)N));
        return std::make_tuple(T,That);
    }

    /** \brief create a Fourier grid from [0,T] \return grid using Fourier collocated points without the last point of linspace(0,T,n+1)[:-1] */
    SPIVec set_Fourier_t(
            PetscScalar T,          ///< [in] period or end point for [0,T] domain
            PetscInt nt             ///< [in] number of points
            ){
        PetscScalar step = T/(PetscScalar)nt;
        SPIVec t(nt);
        PetscScalar value=0.;
        for(PetscInt i=0; i<nt; ++i){
            t(i,value);
            value += step;
        }
        t();
        return t;
    }
    /** \brief create a Fourier derivative operator acting on grid t \return derivative operator using Fourier collocated points */
    SPIMat set_D_Fourier(
            SPIVec t,               ///< [in] grid point created from set_Fourier_t
            PetscInt d              ///< [in] order of derivatives 
            ){
        //PetscScalar pi = PETSC_PI;
        PetscScalar dt = t(1,PETSC_TRUE)-t(0,PETSC_TRUE); // uniform grid spacing
        PetscInt npts=t.rows;
        if(npts%2==0){ // only works with even number of grid points
            PetscScalar nptss = (PetscScalar)npts;
            PetscScalar T = t(npts-1,PETSC_TRUE) + dt;
            SPIMat D;
            SPIMat N,J;
            SPIVec n(arange(0,npts));
            SPIVec j(arange(0,npts));
            std::tie(N,J) = meshgrid(n,j);
            SPIMat NmJ(N-J,"NmJ");
            D = ((PETSC_PI/T)*((-1.0)^(NmJ)))/tan((PETSC_PI/nptss)*(NmJ));
            //std::cout<<"PETSC_PI/T = "<<PETSC_PI/T<<std::endl;
            //((-1.0)^(NmJ)).print();
            //((PETSC_PI/T)*((-1.0)^(NmJ))).print();
            //tan((PETSC_PI/nptss)*(NmJ)).print();
            //((((PETSC_PI/T)*((-1.0)^(NmJ))).real())/tan((PETSC_PI/nptss)*(NmJ)).real()).print();
            //((((PETSC_PI/T)*((-1.0)^(NmJ))).real())*(1.0/(tan((PETSC_PI/nptss)*(NmJ)).real()))).print();
            //std::cout<<"made it here"<<std::endl;
            //(1.0/(tan((PETSC_PI/nptss)*(NmJ)).real())).print();
            //std::cout<<"made it here2"<<std::endl;

            //D.print();
            D.ierr = MatDiagonalSet(D.mat,zeros(npts).vec,INSERT_VALUES);CHKERRXX(D.ierr);
            D();
            D.real();
            if(d==2) D = D*D; // then return Dyy
            //D.print();
            return D;
        }
        else{
            exit(1);
        }
    }

    /** \brief constructor with at no arguments */
    SPIgrid1D::SPIgrid1D(){}
    /** \brief constructor with at least one argument (set default values) */
    SPIgrid1D::SPIgrid1D(
            SPIVec &y,  ///< [in] grid to save
            std::string name, ///< [in] name of grid (default to SPIgrid1D)
            gridtype _ytype ///< [in] what type of grid (default finite difference FD)
            ){
        // set grid name and type
        this->name = name;
        this->ytype = _ytype;
        // set grid 
        this->set_grid(y);
        // set respective derivatives
        this->set_derivatives();
        // set respective operators
        this->set_operators();
    }


    /** \brief saves grid to internal grid */
    void SPIgrid1D::print(){
    SPI::printf("---------------- "+this->name+" start --------------------------");
        if(this->flag_set_grid) {
            //PetscInt ny2 = this->ny;
            //SPI::printf("ny = %D",ny2);
            this->y.print();
        }
        if(this->flag_set_derivatives){
            this->Dy.print();
            this->Dyy.print();
            if(this->ytype==UltraS){
                this->S0.print();
                this->S1.print();
                this->T.print();
                this->That.print();
            }
        }
        if(this->flag_set_operators){
            this->O.print();
            this->I.print();
        }
    SPI::printf("---------------- "+this->name+" done ---------------------------");
    }

    /** \brief saves grid to internal grid */
    void SPIgrid1D::set_grid(
            SPIVec &y ///< [in] grid to save
            ){
        this->y=y;
        this->y.name=std::string("y");
        this->ny = y.rows;
        this->flag_set_grid=PETSC_TRUE;
    }

    /** \brief sets derivatives Dy and Dyy using saved grid */
    void SPIgrid1D::set_derivatives(
            PetscInt order      ///< [in] order of accuracy of finite difference derivative (default 4)
            ){
        if(this->ytype==FD){
            this->Dy=set_D(this->y,1);   // default of fourth order nonuniform grid
            this->Dy.name=std::string("Dy");
            this->Dyy=set_D(this->y,2); // default of fourth order nonuniform grid
            this->Dyy.name=std::string("Dyy");
            this->Dy.real(); // just take only real part
            this->Dyy.real(); // just take only real part
            this->flag_set_derivatives=PETSC_TRUE;
        }
        else if(this->ytype==FDperiodic){
            this->Dy=set_D_periodic(this->y,1);   // default of fourth order nonuniform grid
            this->Dy.name=std::string("Dy");
            this->Dyy=set_D_periodic(this->y,2); // default of fourth order nonuniform grid
            this->Dyy.name=std::string("Dyy");
            this->Dy.real(); // just take only real part
            this->Dyy.real(); // just take only real part
            this->flag_set_derivatives=PETSC_TRUE;
        }
        else if(this->ytype==Chebyshev){
            std::tie(T,That) = set_T_That(y.rows);      // get Chebyshev operators to take back and forth from physical space
            this->Dy=set_D_Chebyshev(this->y,1,PETSC_TRUE);   // default Chebyshev operator on non-uniform grid
            this->Dy.name=std::string("Dy");
            this->Dyy=set_D_Chebyshev(this->y,2,PETSC_TRUE);   // default Chebyshev operator on non-uniform grid
            this->Dyy.name=std::string("Dyy");
            this->Dy.real(); // just take only real part
            this->Dyy.real(); // just take only real part
            this->flag_set_derivatives=PETSC_TRUE;
        }
        else if(this->ytype==Fourier){
            this->Dy=set_D_Fourier(this->y,1);   // default Fourier operator on uniform grid
            this->Dy.name=std::string("Dy");
            this->Dyy=set_D_Fourier(this->y,2);   // default Chebyshev operator on uniform grid
            this->Dyy.name=std::string("Dyy");
            this->Dy.real(); // just take only real part
            this->Dyy.real(); // just take only real part
            this->flag_set_derivatives=PETSC_TRUE;
        }
        else if(this->ytype==UltraS){
            std::tie(T,That) = set_T_That(y.rows);      // get Chebyshev operators to take back and forth from physical space
            std::tie(S0,Dy) = set_D_UltraS(this->y,1);   // default UltraSpherical operators on non-uniform grid
            std::tie(S1,Dyy) = set_D_UltraS(this->y,2);   // default UltraSpherical operators on non-uniform grid
            this->Dy.real(); // just take only real part
            this->Dyy.real(); // just take only real part
            this->Dy = S1*Dy; // make it output C^(2) coefficient space
            this->S0.real(); // just take only real part 
            this->S1.real(); // just take only real part
            this->T.real(); // just take only real part
            this->That.real(); // just take only real part
            this->S1S0That = this->S1*this->S0*this->That;
            this->S0invS1inv = inv(this->S1*this->S0);

            this->Dy.name=std::string("Dy");
            this->S0.name=std::string("S0");
            this->S1.name=std::string("S1");
            this->Dyy.name=std::string("Dyy");
            this->T.name=std::string("T");
            this->That.name=std::string("That");
            this->S1S0That.name=std::string("S1*S0*That");
            this->S0invS1inv.name=std::string("inv(S0)*inv(S1)");
            //this->PS1S0That.name=std::string("PS1S0That");
            this->flag_set_derivatives=PETSC_TRUE;
        }
        else{
            SPI::printf("Warning this type of ytype grid is not implemented in SPIgrid1D.set_derivatives");
        }
    }

    /** \brief sets zero and identity operators for grid */
    void SPIgrid1D::set_operators(){
        PetscInt m = Dy.rows;
        PetscInt n = Dy.cols;
        this->O = zeros(m,n);;   // default Chebyshev operator on non-uniform grid
        this->O.name = "zero";
        if(this->ytype==UltraS){
            this->P = block({
                    {zeros(2,ny-2),zeros(2,2)},
                    {eye(ny-2),zeros(ny-2,2)}
                    })();
            P(0,ny-2,1.0);
            P(1,ny-1,1.0);
            P();
            this->I = S1*S0;    // get it up to second order coefficients (C^(2))
            // permute such that the bottom two rows are now the top two rows (good for LU factorization and pivoting)
            //PS1S0That = P*S1*S0*That;
            //Dy = P*Dy;
            //Dyy = P*Dyy;
            //S0 = P*S0;
            //S1 = P*S1;
        }else{
            this->I = eye(m);   // default Chebyshev operator on non-uniform grid
        }
        this->I.name="eye";
        this->flag_set_operators=PETSC_TRUE;
    }

    /** \brief destructor of saved SPIVec and SPIMat */
    SPIgrid1D::~SPIgrid1D(){
        // destroy grid variables and reset flags
        if (this->flag_set_grid){
            this->y.~SPIVec();
            this->flag_set_grid=PETSC_FALSE;
        }
        // destroy derivative operators and reset flags
        if (this->flag_set_derivatives){
            this->Dy.~SPIMat();
            this->Dyy.~SPIMat();
            this->S0.~SPIMat();
            this->S1.~SPIMat();
            this->T.~SPIMat();
            this->That.~SPIMat();
            this->S1S0That.~SPIMat();
            this->S0invS1inv.~SPIMat();
            this->flag_set_derivatives=PETSC_FALSE;
        }
        // destroy operators and reset flags
        if (this->flag_set_operators){
            this->O.~SPIMat();
            this->I.~SPIMat();
            this->P.~SPIMat();
            this->FTinv.~SPIMat();
            this->FT.~SPIMat();
            this->Ihalf.~SPIMat();
            this->Ihalfn.~SPIMat();
            this->flag_set_operators=PETSC_FALSE;
        }
    }
    /** \brief constructor with no arguments (set default values) */
    SPIgrid2D::SPIgrid2D(
            SPIVec &y,  ///< [in] grid in wall-normal direction to save
            SPIVec &t,  ///< [in] grid in time dimension to save
            std::string name, ///< [in] name of grid (default to SPIgrid1D)
            gridtype y_gridtype, ///< [in] what type of grid (default finite difference FD)
            gridtype t_gridtype ///< [in] what type of grid (default Fourier Transform FT)
            ){
        // set grid name and type
        this->name = name;
        this->ytype = y_gridtype;
        this->ttype = t_gridtype;
        // set grid 
        this->set_grid(y,t);
        // set respective derivatives
        this->set_derivatives();
        // set respective operators
        this->set_operators();
    }


    /** \brief saves grid to internal grid */
    void SPIgrid2D::print(){
    SPI::printf("---------------- "+this->name+" start --------------------------");
        if(this->flag_set_grid) {
            //PetscInt ny2 = this->ny;
            //SPI::printf("ny = %D",ny2);
            this->y.print();
            this->t.print();
        }
        if(this->flag_set_derivatives){
            this->Dy.print();
            this->Dyy.print();
            this->Dt.print();
        }
        if(this->flag_set_operators){
            this->O.print();
            this->I.print();
            this->avgt.print();
        }
    SPI::printf("---------------- "+this->name+" done ---------------------------");
    }

    /** \brief saves grid to internal grid */
    void SPIgrid2D::set_grid(
            SPIVec &y, ///< [in] grid in wall-normal dimension to save
            SPIVec &t  ///< [in] grid in time dimension to save
            ){
        this->y=y;
        this->y.name=std::string("y");
        this->ny = y.rows;
        this->t=t;
        this->t.name=std::string("t");
        this->nt = t.rows;
        this->grid1Dy.set_grid(y);
        this->grid1Dt.set_grid(t);
        this->flag_set_grid=PETSC_TRUE;
    }

    /** \brief sets derivatives Dy and Dyy using saved grid */
    void SPIgrid2D::set_derivatives(
            PetscInt order      ///< [in] order of accuracy of finite difference derivative (default 4)
            ){
        //std::cout<<" grid3D set derivatives 1"<<std::endl;
        //PetscInt ny=this->ny;
        //PetscInt nt=this->nt;
        // set 2D operators in wall-normal
        grid1Dy.ytype=ytype;
        grid1Dy.set_derivatives(order);
        grid1Dt.ytype=ttype;
        grid1Dt.set_derivatives(order);
        // if(this->ytype==FD){
        //     this->Dy2D=set_D(this->y,1,order);   // default of fourth order nonuniform grid
        //     this->Dy2D.name=std::string("Dy2D");
        //     this->Dyy2D=set_D(this->y,2,order); // default of fourth order nonuniform grid
        //     this->Dyy2D.name=std::string("Dyy2D");
        //     this->Dy2D.real(); // just take only real part
        //     this->Dyy2D.real(); // just take only real part
        //     this->flag_set_derivatives=PETSC_TRUE;
        // }
        // else if(this->ytype==FT){
        //     this->Dy2D=set_D_Fourier(this->y,1);   // default Fourier operator on uniform grid
        //     this->Dy2D.name=std::string("Dy2D");
        //     this->Dyy2D=set_D_Fourier(this->y,2);   // default Chebyshev operator on uniform grid
        //     this->Dyy2D.name=std::string("Dyy2D");
        //     this->Dy2D.real(); // just take only real part
        //     this->Dyy2D.real(); // just take only real part
        //     this->flag_set_derivatives=PETSC_TRUE;
        // }
        // else if(this->ytype==Chebyshev){
        //     this->Dy2D=set_D_Chebyshev(this->y,1,PETSC_TRUE);   // default Chebyshev operator on non-uniform grid
        //     this->Dy2D.name=std::string("Dy2D");
        //     this->Dyy2D=set_D_Chebyshev(this->y,2,PETSC_TRUE);   // default Chebyshev operator on non-uniform grid
        //     this->Dyy2D.name=std::string("Dyy2D");
        //     this->Dy2D.real(); // just take only real part
        //     this->Dyy2D.real(); // just take only real part
        //     this->flag_set_derivatives=PETSC_TRUE;
        // }
        // set 2D operators in time
        // if(this->ttype==FD){
        //     this->Dt2D=set_D(this->t,1);   // default of fourth order nonuniform grid
        //     this->Dt2D.name=std::string("Dt2D");
        //     this->Dt2D.real(); // just take only real part
        //     this->flag_set_derivatives=PETSC_TRUE;
        // }
        // else if(this->ttype==FT){
        //     this->Dt2D=set_D_Fourier(this->t,1);   // default Fourier operator on uniform grid
        //     this->Dt2D.name=std::string("Dt2D");
        //     this->Dt2D.real(); // just take only real part
        //     this->flag_set_derivatives=PETSC_TRUE;
        // }
        // else if(this->ttype==Chebyshev){
        //     this->Dt2D=set_D_Chebyshev(this->y,1,PETSC_TRUE);   // default Chebyshev operator on non-uniform grid
        //     this->Dt2D.name=std::string("Dt2D");
        //     this->Dt2D.real(); // just take only real part
        //     this->flag_set_derivatives=PETSC_TRUE;
        // }
        // set 3D operators
        this->Dy = kron(eye(nt),this->grid1Dy.Dy);
        this->Dy.name = "Dy";
        this->Dyy = kron(eye(nt),this->grid1Dy.Dyy);
        this->Dyy.name = "Dyy";
        this->Dt = kron(this->grid1Dt.Dy,eye(ny));
        this->Dt.name = "Dt";
        // set 3D operators
        if((ytype==Chebyshev) || (ytype==UltraS)){
            this->T = kron(eye(nt),this->grid1Dy.T);
            this->T.name = "T";
            this->That = kron(eye(nt),this->grid1Dy.That);
            this->That.name = "That";
        }
        if(ytype==UltraS){
            this->S1S0That = kron(eye(nt),this->grid1Dy.S1S0That);
            this->S1S0That.name = "S1S0That";
            this->S0invS1inv = kron(eye(nt),this->grid1Dy.S0invS1inv);
            this->S0invS1inv.name = "S0invS1inv";
        }
    }

    /** \brief sets zero and identity operators for grid */
    void SPIgrid2D::set_operators(){
        //PetscInt ny = Dy.rows;
        //PetscInt n = Dy.cols;
        PetscInt n=(this->ny)*(this->nt);
        this->O=zeros(n,n);;    // zero matrix of size ny*nt x ny*nt
        this->O.name="zero";
        this->I=eye(n);   
        this->I.name="eye";     // identity matrix of size ny*nt x ny*nt
        // set average in time operator
        this->avgt=zeros(n,n);
        PetscScalar val = 1.0/(PetscScalar)(this->nt);
        //this->avgt /= (double)(this->nt);
        for(PetscInt ti=0; ti<nt; ++ti){
            for(PetscInt yi=0; yi<ny; ++yi){
                PetscInt ii = ti*ny + yi;
                for(PetscInt jj=yi; jj<n; jj+=this->ny){
                    this->avgt(ii,jj,val);
                }
            }
        }
        this->avgt();
        this->grid1Dy.set_operators(); // in case they are needed
        this->grid1Dt.set_operators(); // in case they are needed
        // set operators for Fourier transform
        std::tie(this->grid1Dt.FT,this->grid1Dt.FTinv,this->grid1Dt.Ihalf,this->grid1Dt.Ihalfn) = dft_dftinv_Ihalf_Ihalfn(this->nt);
        this->FT = kron(this->grid1Dt.FT,eye(ny));
        this->FTinv = kron(this->grid1Dt.FTinv,eye(ny));
        this->Ihalf = kron(this->grid1Dt.Ihalf,eye(ny));
        this->Ihalfn = kron(this->grid1Dt.Ihalfn,eye(ny));
        this->flag_set_operators=PETSC_TRUE;
    }

    /** \brief destructor of saved SPIVec and SPIMat */
    SPIgrid2D::~SPIgrid2D(){
        // destroy grid variables and reset flags
        if (this->flag_set_grid){
            this->y.~SPIVec();
            this->t.~SPIVec();
            this->flag_set_grid=PETSC_FALSE;
        }
        // destroy derivative operators and reset flags
        if (this->flag_set_derivatives){
            //this->Dy2D.~SPIMat();
            //this->Dyy2D.~SPIMat();
            //this->Dt2D.~SPIMat();
            this->Dy.~SPIMat();
            this->Dyy.~SPIMat();
            this->Dt.~SPIMat();
            this->T.~SPIMat();
            this->That.~SPIMat();
            this->S1S0That.~SPIMat();
            this->S0invS1inv.~SPIMat();
            this->flag_set_derivatives=PETSC_FALSE;
        }
        // destroy operators and reset flags
        if (this->flag_set_operators){
            this->O.~SPIMat();
            this->I.~SPIMat();
            this->avgt.~SPIMat();
            this->FTinv.~SPIMat();
            this->FT.~SPIMat();
            this->Ihalf.~SPIMat();
            this->Ihalfn.~SPIMat();
            this->flag_set_operators=PETSC_FALSE;
        }
        grid1Dy.~SPIgrid1D();
        grid1Dt.~SPIgrid1D();
    }

    /** \brief expand a 1D vector to a 2D vector copying data along time dimension */
    SPIVec SPIVec1Dto2D(
            SPIgrid2D &grid2D,  ///< [in] 3D grid containing wall-normal and time dimensions
            SPIVec &u           ///< [in] vector to inflate to match the 3D operator
            ){
        PetscInt nu=u.rows; // should be equal to ny
        PetscInt ny=grid2D.ny;
        if(nu!=ny) exit(1);
        PetscInt nt=grid2D.nt;
        SPIVec U(ny*nt,u.name);
        std::vector<PetscScalar> utmp(ny);
        for(PetscInt j=0; j<ny; ++j){
            utmp[j] = u(j,PETSC_TRUE);
        }
        for(PetscInt i=0; i<nt; ++i){
            for(PetscInt j=0; j<ny; ++j){
                //std::cout<<" i,j,j+i*ny = "<<i<<","<<j<<","<<j+i*ny<<std::endl;
                U(j+i*ny,utmp[j]);
            }
        }
        U();
        return U;
    }
    /** \brief integrate a vector of chebyshev Coefficients on a physical grid */
    PetscScalar integrate(
            const SPIVec &a,      ///< [in] vector to integrate over grid (Chebyshev coefficients or physical values)
            SPIgrid1D &grid       ///< [in] respective grid
            ){
        if(grid.ytype==UltraS){ // if using UltraSpherical, then it is in Chebyshev coefficients
            PetscInt ny=grid.y.rows;
            PetscInt nyx=a.rows;
            PetscInt ni=nyx/ny;
            PetscScalar val=0.0;
            SPIVec atmp(ny,"atmp");
            for(PetscInt i=0; i<ni; ++i){
                for(PetscInt j=0; j<ny; ++j){
                    atmp(j,a(ny*i + j,PETSC_TRUE));
                }
                atmp();
                val += integrate_coeffs(atmp,grid.y);
            }
            return val;
        }
        else if(grid.ytype==Chebyshev){ // if using UltraSpherical, then it is in Chebyshev coefficients
            PetscInt ny=grid.y.rows;
            PetscInt nyx=a.rows;
            PetscInt ni=nyx/ny;
            PetscScalar val=0.0;
            SPIVec atmp(ny,"atmp");
            SPIVec atmp2(ny,"atmp");
            for(PetscInt i=0; i<ni; ++i){
                for(PetscInt j=0; j<ny; ++j){
                    atmp(j,a(ny*i + j,PETSC_TRUE));
                }
                atmp();
                //((grid.That)*atmp).print();
                atmp2 = ((grid.That)*atmp);
                val += integrate_coeffs(atmp2,grid.y);
            }
            return val;
        }
        else{ // otherwise they are physical values, let's integrate using trapezoidal rule
            PetscInt ny=grid.y.rows;
            PetscInt nyx=a.rows;
            PetscInt ni=nyx/ny;
            PetscScalar val=0.0;
            SPIVec atmp(ny,"atmp");
            for(PetscInt i=0; i<ni; ++i){
                for(PetscInt j=0; j<ny; ++j){
                    atmp(j,a(ny*i + j,PETSC_TRUE));
                }
                atmp();
                val += trapz(atmp,grid.y);
            }
            return val;
        }
    }
    /** \brief integrate a vector of chebyshev Coefficients on a physical grid */
    PetscScalar integrate(
            const SPIVec &a,      ///< [in] vector to integrate over grid (Chebyshev coefficients or physical values)
            SPIgrid2D &grid       ///< [in] respective grid
            ){
        if((grid.ytype==UltraS) && (grid.ttype==FD)){ // if using UltraSpherical, then it is in Chebyshev coefficients
            PetscInt ny=grid.ny;
            PetscInt nt=grid.nt;
            PetscInt nytx=a.rows;
            PetscInt ni=nytx/(ny*nt);
            PetscScalar val=0.0;
            PetscScalar val2=0.0;
            SPIVec atmp(ny,"atmp");
            //SPIVec atmp2(ny,"atmp2");
            SPIVec atmpt(nt,"atmpt");
            //SPIVec atmpt2(nt,"atmpt2");
            for(PetscInt i=0; i<ni; ++i){
                for(PetscInt j=0; j<nt; ++j){
                    for(PetscInt k=0; k<ny; ++k){
                        atmp(k,a(ny*nt*i + ny*j + k,PETSC_TRUE));
                    }
                    atmp();
                    //((grid.That)*atmp).print();
                    //atmp2 = ((grid.grid1Dy.That)*atmp);
                    val2 = integrate_coeffs(atmp,grid.y);
                    //val += val2;
                    atmpt(j,val2);
                }
                atmpt();
                val2 = trapz(atmpt,grid.t);
                val += val2;
            }
            return val;
        }
        else if((grid.ytype==UltraS) && (grid.ttype==Chebyshev)){ // if using UltraSpherical, then it is in Chebyshev coefficients
            PetscInt ny=grid.ny;
            PetscInt nt=grid.nt;
            PetscInt nytx=a.rows;
            PetscInt ni=nytx/(ny*nt);
            PetscScalar val=0.0;
            PetscScalar val2=0.0;
            SPIVec atmp(ny,"atmp");
            //SPIVec atmp2(ny,"atmp2");
            SPIVec atmpt(nt,"atmpt");
            SPIVec atmpt2(nt,"atmpt2");
            for(PetscInt i=0; i<ni; ++i){
                for(PetscInt j=0; j<nt; ++j){
                    for(PetscInt k=0; k<ny; ++k){
                        atmp(k,a(ny*nt*i + ny*j + k,PETSC_TRUE));
                    }
                    atmp();
                    //((grid.That)*atmp).print();
                    //atmp2 = ((grid.grid1Dy.That)*atmp);
                    val2 = integrate_coeffs(atmp,grid.y);
                    //val += val2;
                    atmpt(j,val2);
                }
                atmpt();
                atmpt2 = ((grid.grid1Dy.That)*atmpt);
                val2 = integrate_coeffs(atmpt,grid.t);
                val += val2;
            }
            return val;
        }
        else if((grid.ytype==Chebyshev) && (grid.ttype==Chebyshev)){ // if using UltraSpherical, then it is in Chebyshev coefficients
            PetscInt ny=grid.ny;
            PetscInt nt=grid.nt;
            PetscInt nytx=a.rows;
            PetscInt ni=nytx/(ny*nt);
            PetscScalar val=0.0;
            PetscScalar val2=0.0;
            SPIVec atmp(ny,"atmp");
            SPIVec atmp2(ny,"atmp2");
            SPIVec atmpt(nt,"atmpt");
            SPIVec atmpt2(nt,"atmpt2");
            for(PetscInt i=0; i<ni; ++i){
                for(PetscInt j=0; j<nt; ++j){
                    for(PetscInt k=0; k<ny; ++k){
                        atmp(k,a(ny*nt*i + ny*j + k,PETSC_TRUE));
                    }
                    atmp();
                    //((grid.That)*atmp).print();
                    atmp2 = ((grid.grid1Dy.That)*atmp);
                    val2 = integrate_coeffs(atmp2,grid.y);
                    //val += val2;
                    atmpt(j,val2);
                }
                atmpt();
                atmpt2 = ((grid.grid1Dt.That)*atmpt);
                val2 = integrate_coeffs(atmpt2,grid.t);
                val += val2;
            }
                //val2 = integrate_coeffs(atmp2,grid.y);
            return val;
        }
        else if((grid.ytype==UltraS) && (grid.ttype==Fourier)){
            PetscInt ny=grid.ny;
            PetscInt nt=grid.nt;
            PetscInt nytx=a.rows;
            PetscInt ni=nytx/(ny*nt);
            PetscScalar val=0.0;
            PetscScalar val2=0.0;
            SPIVec atmp(ny,"atmp");
            //SPIVec atmp2(ny,"atmp2");
            //SPIVec atmpt(nt,"atmpt");
            //SPIVec atmpt2(nt,"atmpt2");
            SPIVec atmpt(nt+1,"atmpt");
            SPIVec t(nt+1,"t");
            for(PetscInt i=0; i<nt; ++i){
                t(i,grid.t(i,PETSC_TRUE));
            }
            PetscScalar dt = grid.t(1,PETSC_TRUE) - grid.t(0,PETSC_TRUE);
            t(nt,grid.t(nt-1,PETSC_TRUE)+dt);
            t();
            for(PetscInt i=0; i<ni; ++i){
                for(PetscInt j=0; j<nt; ++j){
                    for(PetscInt k=0; k<ny; ++k){
                        atmp(k,a(ny*nt*i + ny*j + k,PETSC_TRUE));
                    }
                    atmp();
                    //((grid.That)*atmp).print();
                    //atmp2 = ((grid.grid1Dy.That)*atmp);
                    val2 = integrate_coeffs(atmp,grid.y);
                    //val += val2;
                    atmpt(j,val2);
                    if(j==0){ atmpt(nt,val2); }
                }
                atmpt();
                //atmpt2 = ((grid.grid1Dt.That)*atmpt);
                //val2 = integrate_coeffs(atmpt2,grid.t);
                val2 = trapz(atmpt,t);
                val += val2;
            }
            std::cout<<"integrate val = "<<val<<std::endl;
            return val;
        }
        else if((grid.ytype==Chebyshev) && (grid.ttype==Fourier)){
            PetscInt ny=grid.ny;
            PetscInt nt=grid.nt;
            PetscInt nytx=a.rows;
            PetscInt ni=nytx/(ny*nt);
            PetscScalar val=0.0;
            PetscScalar val2=0.0;
            SPIVec atmp(ny,"atmp");
            SPIVec atmp2(ny,"atmp2");
            //SPIVec atmpt(nt,"atmpt");
            //SPIVec atmpt2(nt,"atmpt2");
            SPIVec atmpt(nt+1,"atmpt");
            SPIVec t(nt+1,"t");
            for(PetscInt i=0; i<nt; ++i){
                t(i,grid.t(i,PETSC_TRUE));
            }
            PetscScalar dt = grid.t(1,PETSC_TRUE) - grid.t(0,PETSC_TRUE);
            t(nt,grid.t(nt-1,PETSC_TRUE)+dt);
            t();
            for(PetscInt i=0; i<ni; ++i){
                for(PetscInt j=0; j<nt; ++j){
                    for(PetscInt k=0; k<ny; ++k){
                        atmp(k,a(ny*nt*i + ny*j + k,PETSC_TRUE));
                    }
                    atmp();
                    //((grid.That)*atmp).print();
                    atmp2 = ((grid.grid1Dy.That)*atmp);
                    val2 = integrate_coeffs(atmp2,grid.y);
                    //val += val2;
                    atmpt(j,val2);
                    if(j==0){ atmpt(nt,val2); }
                }
                atmpt();
                //atmpt2 = ((grid.grid1Dt.That)*atmpt);
                //val2 = integrate_coeffs(atmpt2,grid.t);
                val2 = trapz(atmpt,t);
                val += val2;
            }
            std::cout<<"integrate val = "<<val<<std::endl;
            return val;
        }
        else if((grid.ytype==FD) && (grid.ttype==FD)){ // otherwise they are physical values, let's integrate using trapezoidal rule
            PetscInt ny=grid.ny;
            PetscInt nt=grid.nt;
            PetscInt nytx=a.rows;
            PetscInt ni=nytx/(ny*nt);
            PetscScalar val=0.0;
            PetscScalar val2=0.0;
            SPIVec atmp(ny,"atmp");
            //SPIVec atmp2(ny,"atmp2");
            SPIVec atmpt(nt,"atmpt");
            //SPIVec atmpt2(nt,"atmpt2");
            for(PetscInt i=0; i<ni; ++i){
                for(PetscInt j=0; j<nt; ++j){
                    for(PetscInt k=0; k<ny; ++k){
                        atmp(k,a(ny*nt*i + ny*j + k,PETSC_TRUE));
                    }
                    atmp();
                    //((grid.That)*atmp).print();
                    //atmp2 = ((grid.grid1Dy.That)*atmp);
                    val2 = trapz(atmp,grid.y);
                    //val += val2;
                    atmpt(j,val2);
                }
                atmpt();
                //atmpt2 = ((grid.grid1Dt.That)*atmpt);
                val2 = trapz(atmpt,grid.t);
                val += val2;
            }
                //val2 = integrate_coeffs(atmp2,grid.y);
            return val;
        }
        else if((grid.ytype==FD) && (grid.ttype==FDperiodic)){ // otherwise they are physical values, let's integrate using trapezoidal rule assuming periodic endpoint
            //std::cout<<" integrating FD and FDperiodic"<<std::endl;
            PetscInt ny=grid.ny;
            PetscInt nt=grid.nt;
            PetscInt nytx=a.rows;
            PetscInt ni=nytx/(ny*nt);
            PetscScalar val=0.0;
            PetscScalar val2=0.0;
            SPIVec atmp(ny,"atmp");
            //SPIVec atmp2(ny,"atmp2");
            SPIVec atmpt(nt+1,"atmpt");
            SPIVec t(nt+1,"t");
            for(PetscInt i=0; i<nt; ++i){
                t(i,grid.t(i,PETSC_TRUE));
            }
            PetscScalar dt = grid.t(1,PETSC_TRUE) - grid.t(0,PETSC_TRUE);
            t(nt,grid.t(nt-1,PETSC_TRUE)+dt);
            t();
            //SPIVec atmpt2(nt,"atmpt2");
            for(PetscInt i=0; i<ni; ++i){
                for(PetscInt j=0; j<nt; ++j){
                    for(PetscInt k=0; k<ny; ++k){
                        atmp(k,a(ny*nt*i + ny*j + k,PETSC_TRUE));
                    }
                    atmp();
                    //((grid.That)*atmp).print();
                    //atmp2 = ((grid.grid1Dy.That)*atmp);
                    val2 = trapz(atmp,grid.y);
                    //val += val2;
                    atmpt(j,val2);
                    if(j==0){ atmpt(nt,val2); }
                }
                atmpt();
                //atmpt2 = ((grid.grid1Dt.That)*atmpt);
                val2 = trapz(atmpt,t);
                val += val2;
            }
                //val2 = integrate_coeffs(atmp2,grid.y);
            return val;
        }
        else if((grid.ytype==FD) && (grid.ttype==Fourier)){ // otherwise they are physical values, let's integrate using trapezoidal rule assuming periodic endpoint
            PetscInt ny=grid.ny;
            PetscInt nt=grid.nt;
            PetscInt nytx=a.rows;
            PetscInt ni=nytx/(ny*nt);
            PetscScalar val=0.0;
            PetscScalar val2=0.0;
            SPIVec atmp(ny,"atmp");
            //SPIVec atmp2(ny,"atmp2");
            SPIVec atmpt(nt+1,"atmpt");
            SPIVec t(nt+1,"t");
            for(PetscInt i=0; i<nt; ++i){
                t(i,grid.t(i,PETSC_TRUE));
            }
            PetscScalar dt = grid.t(1,PETSC_TRUE) - grid.t(0,PETSC_TRUE);
            t(nt,grid.t(nt-1,PETSC_TRUE)+dt);
            t();
            //SPIVec atmpt2(nt,"atmpt2");
            for(PetscInt i=0; i<ni; ++i){
                for(PetscInt j=0; j<nt; ++j){
                    for(PetscInt k=0; k<ny; ++k){
                        atmp(k,a(ny*nt*i + ny*j + k,PETSC_TRUE));
                    }
                    atmp();
                    //((grid.That)*atmp).print();
                    //atmp2 = ((grid.grid1Dy.That)*atmp);
                    val2 = trapz(atmp,grid.y);
                    //val += val2;
                    atmpt(j,val2);
                    if(j==0){ atmpt(nt,val2); }
                }
                atmpt();
                //atmpt2 = ((grid.grid1Dt.That)*atmpt);
                val2 = trapz(atmpt,t);
                val += val2;
            }
                //val2 = integrate_coeffs(atmp2,grid.y);
            return val;
        }
        else{
            std::cout<<"integrate mixed types not available"<<std::endl;
            exit(1);
            return 0;
        }
    }
    /* \brief project using inner product for Gram-Schmidt process */
    SPIVec proj(
            SPIVec &u,      ///< [in] first vector to project
            SPIVec &v,      ///< [in] second vector to project
            SPIgrid1D &grid   ///< [in] respective grid
            ){
        if(grid.ytype==UltraS){
            PetscInt n = u.rows/grid.y.rows;
            SPIVec utmp, vtmp;
            SPIMat T, That;
            if(n==1){
                T = grid.T;
                That = grid.That;
                utmp=T*u;
                vtmp=T*v;
            }else if(n==2){
                T = block({
                        {grid.T,grid.O},
                        {grid.O,grid.T},
                        })();
                That = block({
                        {grid.That,grid.O},
                        {grid.O,grid.That},
                        })();
                utmp=T*u;
                vtmp=T*v;
            }else if(n==4){
                T = block({
                        {grid.T,grid.O,grid.O,grid.O},
                        {grid.O,grid.T,grid.O,grid.O},
                        {grid.O,grid.O,grid.T,grid.O},
                        {grid.O,grid.O,grid.O,grid.T},
                        })();
                That = block({
                        {grid.That,grid.O,grid.O,grid.O},
                        {grid.O,grid.That,grid.O,grid.O},
                        {grid.O,grid.O,grid.That,grid.O},
                        {grid.O,grid.O,grid.O,grid.That},
                        })();
                utmp=T*u;
                vtmp=T*v;
            }
            return (integrate(That*(conj(utmp)*vtmp),grid)/integrate(That*(conj(utmp)*utmp),grid)) * u;
        }else{
            return (integrate(conj(u)*v,grid)/integrate(conj(u)*u,grid)) * u;
        }
    }
    /* \brief project using inner product for Gram-Schmidt process */
    SPIVec proj(
            SPIVec &u,      ///< [in] first vector to project
            SPIVec &v,      ///< [in] second vector to project
            SPIgrid2D &grid   ///< [in] respective grid
            ){
        if(grid.ytype==UltraS){
            std::cout<<"in proj 1"<<std::endl;
            PetscInt n = u.rows/(grid.y.rows*grid.t.rows);
            SPIVec utmp, vtmp;
            SPIMat T, That;
            if(n==1){
                std::cout<<"in proj 2"<<std::endl;
                T = grid.T;
                That = grid.That;
                utmp=T*u;
                vtmp=T*v;
            }else if(n==2){
                std::cout<<"in proj 3"<<std::endl;
                T = block({
                        {grid.T,grid.O},
                        {grid.O,grid.T},
                        })();
                That = block({
                        {grid.That,grid.O},
                        {grid.O,grid.That},
                        })();
                utmp=T*u;
                vtmp=T*v;
            }else if(n==4){
                std::cout<<"in proj 4"<<std::endl;
                T = block({
                        {grid.T,grid.O,grid.O,grid.O},
                        {grid.O,grid.T,grid.O,grid.O},
                        {grid.O,grid.O,grid.T,grid.O},
                        {grid.O,grid.O,grid.O,grid.T},
                        })();
                std::cout<<"in proj 5"<<std::endl;
                That = block({
                        {grid.That,grid.O,grid.O,grid.O},
                        {grid.O,grid.That,grid.O,grid.O},
                        {grid.O,grid.O,grid.That,grid.O},
                        {grid.O,grid.O,grid.O,grid.That},
                        })();
                std::cout<<"in proj 6"<<std::endl;
                utmp=T*u;
                std::cout<<"in proj 7"<<std::endl;
                vtmp=T*v;
                std::cout<<"in proj 8"<<std::endl;
            }
            std::cout<<"in proj 9"<<std::endl;
            return (integrate(That*(conj(utmp)*vtmp),grid)/integrate(That*(conj(utmp)*utmp),grid)) * u;
        }else{
            return (integrate(conj(u)*v,grid)/integrate(conj(u)*u,grid)) * u;
        }
    }
    /* \brief orthogonalize a basis dense matrix from an array of vec using Gram-Schmidt */
    std::vector<SPIVec> orthogonalize(
            std::vector<SPIVec> &x,  ///< [in] array of vectors to orthogonalize 
            SPIgrid1D &grid             ///< [in] respective grid for integration using Gram-Schmidt
            ){
        //PetscInt m=x[0].rows;   // number of rows
        PetscInt n=x.size();    // number of columns
        //SPIMat Q(m,n,"Q");
        std::vector<SPIVec> qi(n);
        // copy x[i]
        for(PetscInt i=0; i<n; ++i){
            qi[i] = x[i];
        }
        // project
        for(PetscInt i=0; i<n; ++i){
            if(i>0){
                for(PetscInt j=1; j<=i; ++j){
                    //qi[i] -= proj(qi[j-1],x[i],grid); // if using Classical Gram-Schmidt orthogonalization
                    qi[i] -= proj(qi[j-1],qi[i],grid); // if using Modified  Gram-Schmidt orthogonalization
                    //std::cout<<" projecting i,j = "<<i<<","<<j<<std::endl;
                }
            }
        }
        // normalize
        if(grid.ytype==UltraS){
            SPIVec qtmp;
            PetscInt ni = qi[0].rows/(grid.y.rows);
            if(ni==1){
                for(PetscInt i=0; i<n; ++i){
                    //std::cout<<" norm(q) = "<<sqrt(integrate(SPI::abs(qi[i])^2,grid));
                    qtmp = SPI::abs(grid.T*qi[i]);
                    qtmp = grid.That*(qtmp*qtmp);
                    qi[i] /= sqrt(integrate(qtmp,grid));
                }
            }
            else if(ni==2){
                SPIMat T(block({
                            {grid.T,grid.O},
                            {grid.O,grid.T},
                            })(),"T");
                SPIMat That(block({
                            {grid.That,grid.O},
                            {grid.O,grid.That},
                            })(),"That");
                for(PetscInt i=0; i<n; ++i){
                    //std::cout<<" norm(q) = "<<sqrt(integrate(SPI::abs(qi[i])^2,grid));
                    qtmp = SPI::abs(T*qi[i]);
                    qtmp = That*(qtmp*qtmp);
                    qi[i] /= sqrt(integrate(qtmp,grid));
                }
            }
            else if(ni==4){
                SPIMat T(block({
                            {grid.T,grid.O,grid.O,grid.O},
                            {grid.O,grid.T,grid.O,grid.O},
                            {grid.O,grid.O,grid.T,grid.O},
                            {grid.O,grid.O,grid.O,grid.T},
                            })(),"T");
                SPIMat That(block({
                            {grid.That,grid.O,grid.O,grid.O},
                            {grid.O,grid.That,grid.O,grid.O},
                            {grid.O,grid.O,grid.That,grid.O},
                            {grid.O,grid.O,grid.O,grid.That},
                            })(),"That");
                for(PetscInt i=0; i<n; ++i){
                    //std::cout<<" norm(q) = "<<sqrt(integrate(SPI::abs(qi[i])^2,grid));
                    qtmp = SPI::abs(T*qi[i]);
                    qtmp = That*(qtmp*qtmp);
                    qi[i] /= sqrt(integrate(qtmp,grid));
                }
            }
            else{
                SPI::printf("orthogonalize not implemented yet");
                exit(1);
            }
        }
        else{
            for(PetscInt i=0; i<n; ++i){
                qi[i] /= sqrt(integrate(SPI::abs(qi[i])^2,grid));
            }
        }
        return qi;
    }
    /* \brief orthogonalize a basis dense matrix from an array of vec using SLEPc BV */
    std::vector<SPIVec> orthogonalize(
            std::vector<SPIVec> &x,  ///< [in] array of vectors to orthogonalize 
            SPIgrid2D &grid             ///< [in] respective grid for integration using Gram-Schmidt
            ){
        //std::cout<<"in orthogonalize 1"<<std::endl;
        //PetscInt m=x[0].rows;   // number of rows
        PetscInt n=x.size();    // number of columns
        //SPIMat Q(m,n,"Q");
        std::vector<SPIVec> qi(n);
        // copy x[i]
        for(PetscInt i=0; i<n; ++i){
            qi[i] = x[i];
        }
        //std::cout<<"in orthogonalize 2"<<std::endl;
        // project
        for(PetscInt i=0; i<n; ++i){
            if(i>0){
                for(PetscInt j=1; j<=i; ++j){
                    //qi[i] -= proj(qi[j-1],x[i],grid);
                    //qi[i] -= proj(qi[j-1],x[i],grid); // if using Classical Gram-Schmidt orthogonalization
                    qi[i] -= proj(qi[j-1],qi[i],grid); // if using Modified  Gram-Schmidt orthogonalization
                    //std::cout<<" projecting i,j = "<<i<<","<<j<<std::endl;
                }
            }
        }
        //std::cout<<"in orthogonalize 3"<<std::endl;
        // normalize
        if(grid.ytype==UltraS){
            SPIVec qtmp;
            PetscInt ni = qi[0].rows/(grid.y.rows*grid.t.rows);
            if(ni==1){
                for(PetscInt i=0; i<n; ++i){
                    //std::cout<<" norm(q) = "<<sqrt(integrate(SPI::abs(qi[i])^2,grid));
                    qtmp = SPI::abs(grid.T*qi[i]);
                    qtmp = grid.That*(qtmp*qtmp);
                    qi[i] /= sqrt(integrate(qtmp,grid));
                }
            }
            else if(ni==2){
                SPIMat T(block({
                            {grid.T,grid.O},
                            {grid.O,grid.T},
                            })(),"T");
                SPIMat That(block({
                            {grid.That,grid.O},
                            {grid.O,grid.That},
                            })(),"That");
                for(PetscInt i=0; i<n; ++i){
                    //std::cout<<" norm(q) = "<<sqrt(integrate(SPI::abs(qi[i])^2,grid));
                    qtmp = SPI::abs(T*qi[i]);
                    qtmp = That*(qtmp*qtmp);
                    qi[i] /= sqrt(integrate(qtmp,grid));
                }
            }
            else if(ni==4){
                //std::cout<<"in orthogonalize 4"<<std::endl;
                SPIMat T(block({
                            {grid.T,grid.O,grid.O,grid.O},
                            {grid.O,grid.T,grid.O,grid.O},
                            {grid.O,grid.O,grid.T,grid.O},
                            {grid.O,grid.O,grid.O,grid.T},
                            })(),"T");
                //std::cout<<"in orthogonalize 5"<<std::endl;
                SPIMat That(block({
                            {grid.That,grid.O,grid.O,grid.O},
                            {grid.O,grid.That,grid.O,grid.O},
                            {grid.O,grid.O,grid.That,grid.O},
                            {grid.O,grid.O,grid.O,grid.That},
                            })(),"That");
                //std::cout<<"in orthogonalize 6"<<std::endl;
                for(PetscInt i=0; i<n; ++i){
                    //std::cout<<" norm(q) = "<<sqrt(integrate(SPI::abs(qi[i])^2,grid));
                    qtmp = SPI::abs(T*qi[i]);
                    qtmp = That*(qtmp*qtmp);
                    qi[i] /= sqrt(integrate(qtmp,grid));
                }
                //std::cout<<"in orthogonalize 7"<<std::endl;
            }
            else{
                SPI::printf("orthogonalize not implemented yet");
                exit(1);
            }
        }
        else{
            for(PetscInt i=0; i<n; ++i){
                //std::cout<<"sqrt(integrate(SPI::abs(qi[i])^2,grid)) = "<<sqrt(integrate(SPI::abs(qi[i])^2,grid))<<std::endl;
                //std::cout<<"sqrt(integrate(SPI::abs(qi[i])*SPI::abs(qi[i]),grid)) = "<<sqrt(integrate(SPI::abs(qi[i])*SPI::abs(qi[i]),grid))<<std::endl;
                //std::cout<<"abs(qi[i])(1),qi[i](1) = "<<SPI::abs(qi[i])(1,PETSC_TRUE)<<", "<<qi[i](1,PETSC_TRUE)<<std::endl;
                //std::cout<<"(abs(qi[i])*abs(qi[i]))(1) = "<<(SPI::abs(qi[i])*SPI::abs(qi[i]))(1,PETSC_TRUE)<<std::endl;
                //std::cout<<"sum(abs(qi[i])*abs(qi[i])) = "<<SPI::sum(SPI::abs(qi[i])*SPI::abs(qi[i]))<<std::endl;
                //std::cout<<"integrate(abs(qi[i])*abs(qi[i])) = "<<SPI::integrate(SPI::abs(qi[i])*SPI::abs(qi[i]),grid)<<std::endl;
                //(SPI::abs(qi[i])*SPI::abs(qi[i])).print();
                qi[i] /= sqrt(integrate(SPI::abs(qi[i])^2,grid));
            }
        }
        //std::cout<<"in orthogonalize 8"<<std::endl;
        //qi[0].print();
        return qi;
    }
    /* \brief create matrix to interpolate from grid1 to grid2 \returns out matrix such that u2(grid2.y) = out*u1(grid1.y) */
    SPIMat interp1D_Mat(
            SPIgrid1D &grid1, ///< [in] grid to interpolate values from
            SPIgrid1D &grid2  ///< [in] grid to interpolate values to
            ){
        PetscInt n1=grid1.ny;
        PetscInt n2=grid2.ny;
        SPIMat I_n2Xn1(n2,n1,"I");
        for(PetscInt i=0; i<n2; i++){
            PetscScalar y2i = grid2.y(i,PETSC_TRUE);
            PetscBool flag=PETSC_TRUE;
            for(PetscInt j=0; j<n1; j++){
                PetscScalar y1j = grid1.y(j,PETSC_TRUE);
                if(y1j.real()>y2i.real()){
                    if(flag){
                        I_n2Xn1(i,j-1,1.0);
                        flag = PETSC_FALSE;
                    }
                }
                else if(y1j==y2i){
                    if(flag){
                        I_n2Xn1(i,j,1.0);
                        flag = PETSC_FALSE;
                    }
                }
            }
        }
        I_n2Xn1();
        SPIMat Deltay(diag(grid2.y - (I_n2Xn1*grid1.y)),"Deltay");
        SPIMat interp_n1_to_n2(I_n2Xn1 + (Deltay*(I_n2Xn1*grid1.Dy)) + 0.5*(Deltay*Deltay*(I_n2Xn1*grid1.Dyy)),"interp");
        return interp_n1_to_n2;
    }
    /* \brief create matrix to interpolate from grid1 to grid2 \returns out matrix such that u2(grid2.y) = out*u1(grid1.y) */
    SPIMat interp1D_Mat(
            SPIVec &y1, ///< [in] grid to interpolate values from
            SPIVec &y2  ///< [in] grid to interpolate values to
            ){
        PetscInt n1=y1.rows;
        PetscInt n2=y2.rows;
        SPIMat Dy1(set_D(y1,1),"Dy");
        SPIMat Dyy1(set_D(y1,2),"Dy");
        SPIMat I_n2Xn1(n2,n1,"I"); // get nearest neighbor below the value (always positive dy)
        // nearest neighbor below the value (TODO improve by finding nearest neighbor)
        for(PetscInt i=0; i<n2; i++){
            PetscScalar y2i = y2(i,PETSC_TRUE);
            PetscBool flag=PETSC_TRUE;
            for(PetscInt j=0; j<n1; j++){
                PetscScalar y1j = y1(j,PETSC_TRUE);
                if(y1j.real()>y2i.real()){
                    if(flag){
                        I_n2Xn1(i,j-1,1.0);
                        flag = PETSC_FALSE;
                    }
                }
                else if(y1j==y2i){
                    if(flag){
                        I_n2Xn1(i,j,1.0);
                        flag = PETSC_FALSE;
                    }
                }
            }
        }
        I_n2Xn1();
        SPIMat Deltay(diag(y2 - (I_n2Xn1*y1)),"Deltay");
        SPIMat interp_n1_to_n2(I_n2Xn1 + (Deltay*(I_n2Xn1*Dy1)) + 0.5*(Deltay*Deltay*(I_n2Xn1*Dyy1)),"interp");
        return interp_n1_to_n2;
    }
    /* \brief create a discrete fourier transform transformation operator \returns DFT matrix*/
    SPIMat dft(
            PetscInt nt         ///< [in] size of matrix to create
            ){
        PetscScalar Nt = (PetscScalar)nt;
        PetscScalar e = (PetscExpScalar(-2.0*PETSC_PI*PETSC_i/Nt));
        SPIMat W(nt,nt,"W");
        for(PetscInt i=0; i<nt; ++i){
            for(PetscInt j=0; j<nt; ++j){
                W(i,j,PetscPowScalar(e,i*j));
            }
        }
        W(); // assemble
        W /= Nt;
        return W;
    }
    /* \brief create a discrete fourier transform transformation operator and it's associated inverse along with positive and negative identity like operators\returns DFT matrix, inv(DFT), and I_half, and I_halfn (Ihalf + Ihalfn = eye(nt))*/
    std::tuple<SPIMat,SPIMat,SPIMat,SPIMat> dft_dftinv_Ihalf_Ihalfn(
            PetscInt nt         ///< [in] size of matrix to create
            ){
        PetscScalar Nt = (PetscScalar)nt;
        SPIMat FT(dft(nt),"FT");
        SPIMat FTinv(nt,nt,"inv(FT)");
        FT.H(FTinv);
        FTinv *= Nt;
        SPIMat Ihalf(nt,nt,"Ihalf");
        SPIMat Ihalfn(nt,nt,"Ihalf");
        PetscInt Nthalf = nt/2;
        if(nt%2 == 0) // nt is even
            Nthalf = nt/2;
        else // odd
            Nthalf = nt/2 + 1;
        for(PetscInt i=0; i<nt; ++i){
            if(i==0){
                Ihalf(i,i,0.5);
                Ihalfn(i,i,0.5);
            }
            else if(i < Nthalf){
                Ihalf(i,i,1.0);
            }
            else if(i >= Nthalf){
                Ihalfn(i,i,1.0);
            }
        }
        // assemble identity operators for splitting positive and negative wavenumbers
        Ihalf();
        Ihalfn();

        return std::make_tuple(FT,FTinv,Ihalf,Ihalfn);
    }

}
