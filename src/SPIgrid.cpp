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
        SPIMat X;
        SPIMat Ns; // X and Ns meshgrid for T and That creation
        SPIVec ns(arange(n));
        std::tie(X,Ns) = meshgrid(xi,ns);
        SPIMat T(cos(Ns%acos(X)),"T");
        SPIMat That;
        T.T(That);
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
            D.ierr = MatDiagonalSet(D.mat,zeros(npts).vec,INSERT_VALUES);CHKERRXX(D.ierr);
            D();
            D.real();
            if(d==2) D = D*D; // then return Dyy
            return D;
        }
        else{
            exit(1);
        }
    }

    /** \brief constructor with no arguments (set default values) */
    SPIgrid::SPIgrid(
            SPIVec &y,  ///< [in] grid to save
            std::string name, ///< [in] name of grid (default to SPIgrid)
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
    void SPIgrid::print(){
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
    void SPIgrid::set_grid(
            SPIVec &y ///< [in] grid to save
            ){
        this->y=y;
        this->y.name=std::string("y");
        this->ny = y.rows;
        this->flag_set_grid=PETSC_TRUE;
    }

    /** \brief sets derivatives Dy and Dyy using saved grid */
    void SPIgrid::set_derivatives(
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
        else if(this->ytype==Chebyshev){
            this->Dy=set_D_Chebyshev(this->y,1,PETSC_TRUE);   // default Chebyshev operator on non-uniform grid
            this->Dy.name=std::string("Dy");
            this->Dyy=set_D_Chebyshev(this->y,2,PETSC_TRUE);   // default Chebyshev operator on non-uniform grid
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
            S1S0That = S1*S0*That;

            this->Dy.name=std::string("Dy");
            this->S0.name=std::string("S0");
            this->S1.name=std::string("S1");
            this->Dyy.name=std::string("Dyy");
            this->T.name=std::string("T");
            this->That.name=std::string("That");
            //this->PS1S0That.name=std::string("PS1S0That");
            this->flag_set_derivatives=PETSC_TRUE;
        }
        else{
            SPI::printf("Warning this type of ytype grid is not implemented in SPIgrid.set_derivatives");
        }
    }

    /** \brief sets zero and identity operators for grid */
    void SPIgrid::set_operators(){
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
    SPIgrid::~SPIgrid(){
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
            this->flag_set_derivatives=PETSC_FALSE;
        }
        // destroy operators and reset flags
        if (this->flag_set_operators){
            this->O.~SPIMat();
            this->I.~SPIMat();
            this->flag_set_operators=PETSC_FALSE;
        }
    }


}
