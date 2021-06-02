#include "SPILST.hpp"

namespace SPI{
    /** \brief solve the local stability theory problem for the linearized Navier-Stokes equations using parallel baseflow with alpha being pure real, and omega the eigenvalue \return tuple of eigenvalue and eigenvector closest to the target value e.g. std::tie(omega,eig_vector) = LST_temporal(params,grid,baseflow).  Will solve for closest eigenvalue to params.omega */
    std::tuple<PetscScalar, SPIVec> LST_temporal(
            SPIparams &params,          ///< [inout] contains parameters including Re, alpha, and omega.  Will overwrite omega with true omega value once solved
            SPIgrid1D &grid,              ///< [in] grid class containing the grid location and respective derivatives
            SPIbaseflow &baseflow,      ///< [in] baseflow for parallel flow
            SPIVec q                    ///< [in] initial guess for temporal problem
            ){
        PetscInt n = grid.ny;
        PetscScalar Re = params.Re;
        PetscScalar alpha = params.alpha;
        PetscScalar omega = params.omega;
        PetscScalar beta = params.beta;
        PetscScalar i=PETSC_i;
        PetscScalar k=alpha*alpha+beta*beta;
        PetscScalar k2=k*k;
        const SPIMat &O = grid.O;
        const SPIMat &I = grid.I;
        SPIMat U = SPI::diag(baseflow.U);
        SPIMat Uy = SPI::diag(baseflow.Uy);
        if(grid.ytype==UltraS){// modify baseflow to be in C^(2) coefficient space
            //SPI::SPIMat S1S0That = grid.S1*grid.S0*grid.That;
            U = grid.S1S0That*U*grid.T;
            Uy = grid.S1S0That*Uy*grid.T;
        }
        SPI::SPIMat d((i*alpha*U)+((k2/Re)*I)-(1./Re)*grid.Dyy,"d");
        SPI::SPIMat L=-i*SPI::block({
                {d,         Uy,         O,          i*alpha*I}, // u-mom
                {O,         d,          O,          grid.Dy  },  // v-mom
                {O,         O,          d,          i*beta*I },  // w-mom
                {i*alpha*I, grid.Dy,    i*beta*I,   O        }   // continuity
                })();//,"L");
        SPI::SPIMat M=SPI::block({
                {I,    O,          O,          O}, // u-mom
                {O,         I,     O,          O}, // v-mom
                {O,         O,          I,     O}, // w-mom
                {O,         O,          O,          O}  // continuity
                })();//,"M");
        // set BCs
        //PetscInt rowBCs[] = {0,n-1,n,2*n-1,2*n,3*n-1}; // u,v,w at wall and freestream
        //PetscInt rowBCs[] = {0,n-1,2*n,3*n-1,3*n,4*n-1}; // u,v,w at wall and freestream
        L();
        M();
        //for(PetscInt rowi : rowBCs){
        ////SPI::printf(std::to_string(rowi));
        //L.eye_row(rowi);
        //M.eye_row(rowi);
        //L(rowi,rowi,1.0);
        //M(rowi,rowi,60.0);
        //M();
        //L();
        //for(PetscInt j=0; j<4*n; ++j){
        //L(rowi,j,0.0,INSERT_VALUES);
        //M(rowi,j,0.0,INSERT_VALUES);
        //}
        //}
        //for(PetscInt rowi : rowBCs){
        //L(rowi,rowi,1.0);
        //M(rowi,rowi,60.0);
        //}
        if(grid.ytype==UltraS){
            std::vector<PetscInt> rowBCs = {n-2,n-1,2*n-2,2*n-1,3*n-2,3*n-1}; // u,v,w at wall and freestream and dv'/dy = 0 at wall // before Permutation matrix reordering
            //std::vector<PetscInt> rowBCs = {0,1,n,n+1,2*n,2*n+1}; // u,v,w at wall and freestream
            L.zero_rows(rowBCs);
            M.zero_rows(rowBCs);
            for(PetscInt j=0; j<n; ++j){
                L(rowBCs[0],j,grid.T(0,j,PETSC_TRUE));      // u at the wall
                L(rowBCs[1],j,grid.T(n-1,j,PETSC_TRUE));    // u at the freestream
                L(rowBCs[2],n+j,grid.T(0,j,PETSC_TRUE));    // v at the wall
                L(rowBCs[3],n+j,grid.T(n-1,j,PETSC_TRUE));  // v at the freestream
                L(rowBCs[4],2*n+j,grid.T(0,j,PETSC_TRUE));  // w at the wall
                L(rowBCs[5],2*n+j,grid.T(n-1,j,PETSC_TRUE));// w at the freestream
                //M(rowBCs[0],j,grid.T(0,j,PETSC_TRUE));      // u at the wall
                //M(rowBCs[1],j,grid.T(n-1,j,PETSC_TRUE));    // u at the freestream
                //M(rowBCs[2],n+j,grid.T(0,j,PETSC_TRUE));    // v at the wall
                //M(rowBCs[3],n+j,grid.T(n-1,j,PETSC_TRUE));  // v at the freestream
                //M(rowBCs[4],2*n+j,grid.T(0,j,PETSC_TRUE));  // w at the wall
                //M(rowBCs[5],2*n+j,grid.T(n-1,j,PETSC_TRUE));// w at the freestream
                //L(rowBCs[6],n+j,(grid.T*inv(grid.S0)*inv(grid.S1)*grid.Dy)(0,j,PETSC_TRUE));// v at the freestream
            }
            L();
            M();
            // reorder matrix rows to reduce LU factorization due to extra numerical pivoting
            // otherwise you need to add -mat_mumps_icntl_14 25 to the simulation command line call to increase factorization space
            //SPIMat o2xnm2 = zeros(2,n-2);
            //SPIMat onm2x2 = zeros(n-2,2);
            //SPIMat o2x2 = zeros(2,2);
            //SPIMat Ptmp = block({
                    //{zeros(2,n-2),zeros(2,2)},
                    //{eye(n-2),zeros(n-2,2)}
                    //});
            //Ptmp(0,n-2,1.0);
            //Ptmp(1,n-1,1.0);
            //Ptmp();
            SPIMat P = block({
                    {grid.P,O,O,O},
                    {O,grid.P,O,O},
                    {O,O,grid.P,O},
                    //{O,O,O,eye(n)},
                    {O,O,O,eye(n)},
                    })();
            //P.print();
            L = P*L;
            M = P*M;

            //L.print();

            //save(L,"L_from_LST_temporal_UltraS.dat");
            //save(M,"M_from_LST_temporal_UltraS.dat");
        }
        else{
            std::vector<PetscInt> rowBCs = {0,n-1,n,2*n-1,2*n,3*n-1}; // u,v,w at wall and freestream
            L.eye_rows(rowBCs);
            M.eye_rows(rowBCs);
        }
        L();
        M();
        //L.print();
        //M.print();
        //L();
        //M();
        SPI::SPIVec eig_vec(grid.y.rows*4,"q");
        SPI::SPIVec eigl_vec(grid.y.rows*4,"q");
        //PetscScalar omega;
        // std::tie(eigenvalue,eigenfunction) = SPI::eig(L,M,0.3-0.0001*i); // doesn't work because M is singular
        if(q.flag_init){
            std::tie(omega,eigl_vec,eig_vec) = SPI::eig_init(M,L,1./(params.omega),q.conj(),q);
            omega = 1./omega; // invert
        }else{
            //std::cout<<"here right before eig"<<std::endl;
            //std::tie(omega,eigl_vec,eig_vec) = SPI::eig(M,L,1./params.omega);
            //std::tie(omega,eig_vec) = SPI::eig_right(M,L,1./params.omega);
            //std::cout<<"here right after eig"<<std::endl;
            //std::tie(omega,eigl_vec,eig_vec) = SPI::eig(L,M,params.omega);
            std::tie(omega,eig_vec) = SPI::eig_right(L,M,params.omega);
            //std::tie(omega,eig_vec) = SPI::eig_right(M,L,1./params.omega);
            //omega = 1./omega; // invert
            //std::cout<<"here right after eig 2"<<std::endl;
        }
        params.omega = omega;
        //SPI::printfc("ω is: %g+%gi",omega);
        return std::make_tuple(omega,eig_vec);
    }
    /** \brief solve the local stability theory problem for the linearized Navier-Stokes equations using parallel baseflow with omega being pure real, and alpha the eigenvalue \return tuple of eigenvalue and eigenvector closest to the target value e.g. std::tie(omega,eig_vector) = LST_spatial(params,grid,baseflow).  Will solve for closest eigenvalue to params.omega */
    std::tuple<PetscScalar, SPIVec> LST_spatial(
            SPIparams &params,          ///< [inout] contains parameters including Re, alpha, and omega.  Will overwrite omega with true omega value once solved
            SPIgrid1D &grid,              ///< [in] grid class containing the grid location and respective derivatives
            SPIbaseflow &baseflow,      ///< [in] baseflow for parallel flow
            SPIVec q                    ///< [in] initial guess for temporal problem
            ){
        PetscInt n = grid.ny;
        PetscScalar Re = params.Re;
        PetscScalar alpha = params.alpha;
        PetscScalar omega = params.omega;
        PetscScalar beta = params.beta;
        PetscScalar i=PETSC_i;
        SPIMat &O = grid.O;
        SPIMat &I = grid.I;
        SPIMat U = SPI::diag(baseflow.U);
        SPIMat Uy = SPI::diag(baseflow.Uy);
        if(grid.ytype==SPI::UltraS){
            SPIMat &S1S0That = grid.S1S0That;
            SPIMat &T = grid.T;
            U = S1S0That*U*T;
            Uy = S1S0That*Uy*T;
        }
        SPIMat &Dy = grid.Dy;
        SPIMat &Dyy = grid.Dyy;
        SPIMat d(Dyy + i*Re*omega*I - beta*beta*I,"d");
        //SPI::SPIMat d("d");
        //d = Dyy + i*Re*omega*I - beta*beta*I;
        if(1){
            SPI::SPIMat L0(SPI::block({
                        {d,         -Re*Uy,     O,          O},
                        {O,         d,          O,          -Re*Dy},
                        {O,         O,          d,          -i*Re*beta*I},
                        {O,         Dy,         i*beta*I,   O} 
                        })(),"L0");
            SPI::SPIMat L1(SPI::block({
                        {-i*Re*U,   O,          O,          -i*Re*I},
                        {O,         -i*Re*U,    O,          O},
                        {O,         O,          -i*Re*U,    O},
                        {i*I,       O,          O,          O} 
                        })(),"L1");
            SPI::SPIMat L2(SPI::block({
                        {-I,        O,          O,          O},
                        {O,         -I,         O,          O},
                        {O,         O,          -I,         O},
                        {O,         O,          O,          O} 
                        })(),"L2");
            // set BCs
            if(grid.ytype==SPI::UltraS){
                //PetscInt rowBCs[] = {0,n-1,n,2*n-1,2*n,3*n-1}; // u,v,w at wall and freestream
                std::vector<PetscInt> rowBCs = {n-2,n-1,2*n-2,2*n-1,3*n-2,3*n-1}; // u,v,w at wall and freestream
                L0.zero_rows(rowBCs);
                L1.zero_rows(rowBCs);
                L2.zero_rows(rowBCs);
                for(PetscInt j=0; j<n; ++j){
                    L0(rowBCs[0],j,grid.T(0,j,PETSC_TRUE));       // u at the wall
                    L0(rowBCs[1],j,grid.T(n-1,j,PETSC_TRUE));    // u at the freestream
                    L0(rowBCs[2],n+j,grid.T(0,j,PETSC_TRUE));       // v at the wall
                    L0(rowBCs[3],n+j,grid.T(n-1,j,PETSC_TRUE));    // v at the freestream
                    L0(rowBCs[4],2*n+j,grid.T(0,j,PETSC_TRUE));       // w at the wall
                    L0(rowBCs[5],2*n+j,grid.T(n-1,j,PETSC_TRUE));    // w at the freestream
                }
                // assemble
                L0();
                L1();
                L2();
                SPIMat P = block({
                        {grid.P,O,O,O},
                        {O,grid.P,O,O},
                        {O,O,grid.P,O},
                        {O,O,O,eye(n)},
                        })();
                // reorder rows for UltraSpherical
                L0 = P*L0;
                L1 = P*L1;
                L2 = P*L2;
                //std::cout<<"Re-ordered rows for UltraS"<<std::endl;
                //save(U,"U_from_LST_spatial_UltraS.dat");
                //save(Uy,"Uy_from_LST_spatial_UltraS.dat");
                //save(baseflow.U,"U_and_Uy_from_LST_spatial_UltraS.h5");
                //save(baseflow.Uy,"U_and_Uy_from_LST_spatial_UltraS.h5");
                //baseflow.Uy.print();
                //save(L0,"L0_from_LST_spatial_UltraS.dat");
                //save(L1,"L1_from_LST_spatial_UltraS.dat");
                //save(L2,"L2_from_LST_spatial_UltraS.dat");
            }else{
                PetscInt rowBCs[] = {0,n-1,n,2*n-1,2*n,3*n-1}; // u,v,w at wall and freestream
                for(PetscInt rowi : rowBCs){
                    //SPI::printf(std::to_string(rowi));
                    L0.eye_row(rowi);
                    L1.zero_row(rowi);
                    L2.zero_row(rowi);
                    //L0(rowi,rowi,1.0);
                    //L2(rowi,rowi,60.0);
                }
            }
            // assemble
            L0();
            L1();
            L2();
            //L0.print();
            //L1.print();
            //L2.print();
            SPI::SPIVec eig_vec(grid.y.rows*4,"q");
            //SPI::SPIVec eigl_vec(grid.y.rows*4,"q");
            //PetscScalar omega;
            // std::tie(eigenvalue,eigenfunction) = SPI::eig(L,M,0.3-0.0001*i); // doesn't work because M is singular
            if(q.flag_init){
                std::tie(alpha,eig_vec) = SPI::polyeig_init({L0,L1,L2},alpha,q);
            }else{
                std::tie(alpha,eig_vec) = SPI::polyeig({L0,L1,L2},alpha);
            }
            //alpha = alpha; // invert
            params.alpha = alpha;
            //SPI::printfc("α is: %g+%gi",alpha);
            return std::make_tuple(alpha,eig_vec);
        }
        else if(0){
            SPI::SPIMat L(SPI::block({
                        {O,         O,          O,          O,          I,          O,          O,          O},
                        {O,         O,          O,          O,          O,          I,          O,          O},
                        {O,         O,          O,          O,          O,          O,          I,          O},
                        {O,         O,          O,          O,          O,          O,          O,          I},
                        {d,         -Re*Uy,     O,          O,          -i*Re*U,    O,          O,          -i*Re*I},
                        {O,         d,          O,          -Re*Dy,     O,          -i*Re*U,    O,          O},
                        {O,         O,          d,         -i*Re*beta*I,O,          O,          -i*Re*U,    O},
                        {O,         Dy,         i*beta*I,   O,          i*I,        O,          O,          O},
                        }),"L");
            SPI::SPIMat M(SPI::block({
                        {I,         O,          O,          O,          O,          O,          O,          O},
                        {O,         I,          O,          O,          O,          O,          O,          O},
                        {O,         O,          I,          O,          O,          O,          O,          O},
                        {O,         O,          O,          I,          O,          O,          O,          O},
                        {O,         O,          O,          O,          I,          O,          O,          O},
                        {O,         O,          O,          O,          O,          I,          O,          O},
                        {O,         O,          O,          O,          O,          O,          I,          O},
                        {O,         O,          O,          O,          O,          O,          O,          O},
                        }),"M");
            // set BCs
            //PetscInt rowBCs[] = {0,n-1,n,2*n-1,2*n,3*n-1,4*n,5*n-1,5*n,6*n-1,6*n,7*n-1}; // u,v,w at wall and freestream
            PetscInt rowBCs[] = {4*n,5*n-1,5*n,6*n-1,6*n,7*n-1}; // u,v,w at wall and freestream
            for(PetscInt rowi : rowBCs){
                //SPI::printf(std::to_string(rowi));
                L.zero_row(rowi);
                M.zero_row(rowi);
                L(rowi,rowi,1.0);
                M(rowi,rowi,60.0*i);
            }
            SPI::SPIVec eig_vec(grid.y.rows*8,"q");
            SPI::SPIVec eigl_vec(grid.y.rows*8,"q");
            if(q.flag_init){
                std::tie(alpha,eigl_vec,eig_vec) = SPI::eig_init(L,M,alpha,q.conj(),q);
            }else{
                std::tie(alpha,eigl_vec,eig_vec) = SPI::eig(M,L,1./alpha);
            }
            alpha = alpha; // invert
            params.alpha = alpha;
            //SPI::printfc("α is: %g+%gi",alpha);
            return std::make_tuple(alpha,eig_vec);
        }
        else{
            d = 1./Re*(Dyy - beta*beta*I) + i*omega*I;
            SPI::SPIMat L(SPI::block({
                        //u         v           w           p           vx          wx
                        {O,         i*Dy,       -beta*I,    O,          O,          O}, // continuity
                        {O,         O,          O,          O,          I,          O}, // v-sub
                        {O,         O,          O,          O,          O,          I}, // w-sub
                        {-i*d,      i*(Uy-U*Dy),beta*U,     O,          -1./Re*Dy,  -i*beta/Re*I}, // u-mom
                        {O,         i*Re*d,     O,          -i*Re*Dy,   Re*U,       O}, // v-mom
                        {O,         O,          i*Re*d,     beta*Re*I,  O,          Re*U}, // w-mom
                        }),"L");
            SPI::SPIMat M(SPI::block({
                        {I,         O,          O,          O,          O,          O},
                        {O,         I,          O,          O,          O,          O},
                        {O,         O,          I,          O,          O,          O},
                        {O,         O,          O,          I,          O,          O},
                        {O,         O,          O,          O,          I,          O},
                        {O,         O,          O,          O,          O,          I}
                        }),"M");
            // set BCs
            //PetscInt rowBCs[] = {0,n-1,n,2*n-1,2*n,3*n-1,4*n,5*n-1,5*n,6*n-1,6*n,7*n-1}; // u,v,w at wall and freestream
            PetscInt rowBCs[] = {0*n,1*n-1,1*n,2*n-1,2*n,3*n-1}; // u,v,w at wall and freestream
            for(PetscInt rowi : rowBCs){
                //SPI::printf(std::to_string(rowi));
                L.zero_row(rowi);
                M.zero_row(rowi);
                L(rowi,rowi,1.0);
                M(rowi,rowi,60.0*i);
            }
            SPI::SPIVec eig_vec(grid.y.rows*8,"q");
            SPI::SPIVec eigl_vec(grid.y.rows*8,"q");
            if(q.flag_init){
                std::tie(alpha,eigl_vec,eig_vec) = SPI::eig_init(L,M,alpha,q.conj(),q,1e-16,20000);
            }else{
                std::tie(alpha,eigl_vec,eig_vec) = SPI::eig(L,M,alpha,1e-12,20000);
            }
            //alpha = 1./alpha; // invert
            params.alpha = alpha;
            //SPI::printfc("α is: %g+%gi",alpha);
            return std::make_tuple(alpha,eig_vec);
        }
    }
    /** \brief solve the local stability theory problem for the linearized Navier-Stokes equations using parallel baseflow with omega being pure real, and alpha the eigenvalue \return tuple of eigenvalue and eigenvector closest to the target value e.g. std::tie(omega,eig_vector) = LST_spatial(params,grid,baseflow).  Will solve for closest eigenvalue to params.omega */
    std::tuple<PetscScalar, PetscScalar, SPIVec, SPIVec> LST_spatial_cg(
            SPIparams &params,          ///< [inout] contains parameters including Re, alpha, and omega.  Will overwrite omega with true omega value once solved
            SPIgrid1D &grid,              ///< [in] grid class containing the grid location and respective derivatives
            SPIbaseflow &baseflow       ///< [in] baseflow for parallel flow
            ){
        PetscInt n = grid.ny;
        PetscScalar Re = params.Re;
        PetscScalar alpha = params.alpha;
        PetscScalar omega = params.omega;
        PetscScalar beta = params.beta;
        PetscScalar i=PETSC_i;
        SPIMat &O = grid.O;
        SPIMat &I = grid.I;
        SPIMat U = SPI::diag(baseflow.U);
        SPIMat Uy = SPI::diag(baseflow.Uy);
        if(grid.ytype==SPI::UltraS){
            SPIMat &S1S0That = grid.S1S0That;
            SPIMat &T = grid.T;
            U = S1S0That*U*T;
            Uy = S1S0That*Uy*T;
        }
        SPIMat &Dy = grid.Dy;
        SPIMat &Dyy = grid.Dyy;
        SPIMat d(Dyy + i*Re*omega*I - beta*beta*I,"d");
        //SPI::SPIMat d("d");
        //d = Dyy + i*Re*omega*I - beta*beta*I;
        if(1){
            SPI::SPIMat L0(SPI::block({
                        {d,         -Re*Uy,     O,          O},
                        {O,         d,          O,          -Re*Dy},
                        {O,         O,          d,          -i*Re*beta*I},
                        {O,         Dy,         i*beta*I,   O} 
                        })(),"L0");
            SPI::SPIMat L1(SPI::block({
                        {-i*Re*U,   O,          O,          -i*Re*I},
                        {O,         -i*Re*U,    O,          O},
                        {O,         O,          -i*Re*U,    O},
                        {i*I,       O,          O,          O} 
                        })(),"L1");
            SPI::SPIMat L2(SPI::block({
                        {-I,        O,          O,          O},
                        {O,         -I,         O,          O},
                        {O,         O,          -I,         O},
                        {O,         O,          O,          O} 
                        })(),"L2");
            SPI::SPIMat L2_noBCs(SPI::block({
                        {-I,        O,          O,          O},
                        {O,         -I,         O,          O},
                        {O,         O,          -I,         O},
                        {O,         O,          O,          O} 
                        })(),"L2");
            SPI::SPIMat L2_noBCs_physical(SPI::block({
                        {-eye(n),        O,     O,          O},
                        {O,         -eye(n),    O,          O},
                        {O,         O,          -eye(n),    O},
                        {O,         O,          O,          O} 
                        })(),"L2");
            //std::vector<PetscInt> rowBCs_physical = {0,n-1,n,2*n-1,2*n,3*n-1}; // u,v,w at wall and freestream
            //L2_noBCs_physical.zero_rows(rowBCs_physical);
            // set BCs
            if(grid.ytype==SPI::UltraS){
                //SPIMat S1S0T(grid.S1*grid.S0*grid.T,"S1*S0*T");
                //PetscInt rowBCs[] = {0,n-1,n,2*n-1,2*n,3*n-1}; // u,v,w at wall and freestream
                std::vector<PetscInt> rowBCs = {n-2,n-1,2*n-2,2*n-1,3*n-2,3*n-1}; // u,v,w at wall and freestream
                L0.zero_rows(rowBCs);
                L1.zero_rows(rowBCs);
                L2.zero_rows(rowBCs);
                for(PetscInt j=0; j<n; ++j){
                    L0(rowBCs[0],j,grid.T(0,j,PETSC_TRUE));       // u at the wall
                    L0(rowBCs[1],j,grid.T(n-1,j,PETSC_TRUE));    // u at the freestream
                    L0(rowBCs[2],n+j,grid.T(0,j,PETSC_TRUE));       // v at the wall
                    L0(rowBCs[3],n+j,grid.T(n-1,j,PETSC_TRUE));    // v at the freestream
                    L0(rowBCs[4],2*n+j,grid.T(0,j,PETSC_TRUE));       // w at the wall
                    L0(rowBCs[5],2*n+j,grid.T(n-1,j,PETSC_TRUE));    // w at the freestream
                }
                // assemble
                L0();
                L1();
                L2();
                SPIMat P = block({
                        {grid.P,O,O,O},
                        {O,grid.P,O,O},
                        {O,O,grid.P,O},
                        {O,O,O,eye(n)},
                        })();
                // reorder rows for UltraSpherical
                L0 = P*L0;
                L1 = P*L1;
                L2 = P*L2;
                //std::cout<<"Re-ordered rows for UltraS"<<std::endl;
                //save(U,"U_from_LST_spatial_UltraS.dat");
                //save(Uy,"Uy_from_LST_spatial_UltraS.dat");
                //save(baseflow.U,"U_and_Uy_from_LST_spatial_UltraS.h5");
                //save(baseflow.Uy,"U_and_Uy_from_LST_spatial_UltraS.h5");
                //baseflow.Uy.print();
                //save(L0,"L0_from_LST_spatial_UltraS.dat");
                //save(L1,"L1_from_LST_spatial_UltraS.dat");
                //save(L2,"L2_from_LST_spatial_UltraS.dat");
            }else{
                PetscInt rowBCs[] = {0,n-1,n,2*n-1,2*n,3*n-1}; // u,v,w at wall and freestream
                for(PetscInt rowi : rowBCs){
                    //SPI::printf(std::to_string(rowi));
                    L0.eye_row(rowi);
                    L1.zero_row(rowi);
                    L2.zero_row(rowi);
                    //L0(rowi,rowi,1.0);
                    //L2(rowi,rowi,60.0);
                }
            }
            // assemble
            L0();
            L1();
            L2();
            //L0.print();
            //L1.print();
            //L2.print();
            SPI::SPIVec eig_vec(n*8,"q");
            SPI::SPIVec eigl_vec(n*8,"ql");
            SPIMat O4(zeros(4*n,4*n));
            //SPIMat I4(eye(4*n));
            SPIMat I4(block({
                        {I,O,O,O},
                        {O,I,O,O},
                        {O,O,I,O},
                        {O,O,O,I}} 
                        )(),"I4");
            SPIMat L(block({
                        {O4,    I4},
                        {L0,    L1}})(),"L");
            SPIMat M(block({
                        {I4,    O4},
                        {O4,    -L2}})(),"M");
            //SPIMat M_noBCs(block({
                        //{I4,    O4},
                        //{O4,    -L2_noBCs}})(),"M noBCs");
            //SPI::SPIVec eigl_vec(grid.y.rows*4,"q");
            //PetscScalar omega;
            // std::tie(eigenvalue,eigenfunction) = SPI::eig(L,M,0.3-0.0001*i); // doesn't work because M is singular
            std::tie(alpha,eigl_vec,eig_vec) = eig(L,M,alpha);
            //alpha = alpha; // invert
            params.alpha = alpha;
            // get cg
            SPIMat dLdomega4(block({
                        {i*Re*I,O,      O,      O},
                        {O,     i*Re*I, O,      O},
                        {O,     O,      i*Re*I, O},
                        {O,     O,      O,      O},
                        })(),"dLdomega4");
            // set BCs for physical values
            //dLdomega4_physical.zero_rows(rowBCs_physical);
            SPIMat dLdomega(block({
                        {O4,        O4},
                        {dLdomega4, O4}})(),"dLdomega");
            //SPIVec eigl_vecconj(eigl_vec.conj());
            //eigl_vec.conj();
            PetscScalar cg;
            if(grid.ytype==UltraS){
                SPIMat M_noBCs_physical(block({
                            {eye(4*n),    O4},
                            {O4,    -L2_noBCs_physical}})(),"M noBCs physical");
                SPIMat dLdomega4_physical(block({
                            {i*Re*eye(n),   O,          O,          O},
                            {O,             i*Re*eye(n),O,          O},
                            {O,             O,          i*Re*eye(n),O},
                            {O,             O,          O,          O},
                            })(),"dLdomega4");
                SPIMat dLdomega_physical(block({
                            {O4,        O4},
                            {dLdomega4_physical, O4}})(),"dLdomega");
                //cg = ((M*eig_vec).dot(eigl_vec)) / ((dLdomega*eig_vec).dot(eigl_vec));
                //SPIVec eigl_vecconj(eigl_vec.conj());
                //eigl_vec.conj();
                //SPIMat S0invS1inv(block({
                            //{grid.S0invS1inv,O,O,O,O,O,O,O},
                            //{O,grid.S0invS1inv,O,O,O,O,O,O},
                            //{O,O,grid.S0invS1inv,O,O,O,O,O},
                            //{O,O,O,grid.S0invS1inv,O,O,O,O},
                            //{O,O,O,O,grid.S0invS1inv,O,O,O},
                            //{O,O,O,O,O,grid.S0invS1inv,O,O},
                            //{O,O,O,O,O,O,grid.S0invS1inv,O},
                            //{O,O,O,O,O,O,O,grid.S0invS1inv},
                            //})());
                SPIMat T(block({
                            {grid.T,O,O,O,O,O,O,O},
                            {O,grid.T,O,O,O,O,O,O},
                            {O,O,grid.T,O,O,O,O,O},
                            {O,O,O,grid.T,O,O,O,O},
                            {O,O,O,O,grid.T,O,O,O},
                            {O,O,O,O,O,grid.T,O,O},
                            {O,O,O,O,O,O,grid.T,O},
                            {O,O,O,O,O,O,O,grid.T},
                            })());
                //SPIMat That(block({
                            //{grid.That,O,O,O,O,O,O,O},
                            //{O,grid.That,O,O,O,O,O,O},
                            //{O,O,grid.That,O,O,O,O,O},
                            //{O,O,O,grid.That,O,O,O,O},
                            //{O,O,O,O,grid.That,O,O,O},
                            //{O,O,O,O,O,grid.That,O,O},
                            //{O,O,O,O,O,O,grid.That,O},
                            //{O,O,O,O,O,O,O,grid.That},
                            //})());
                //cg = integrate(((eig_vec))*conj(S0invS1inv*M*eigl_vec),grid) / integrate((S0invS1inv*(dLdomega*eig_vec))*conj(eigl_vec),grid);
                //SPI::printfc("in LST_spatial_cg UltraS cg = %.10f + %.10fi",cg);
                //cg = integrate(That*((T*((eig_vec)))*conj(T*(S0invS1inv*(M*eigl_vec)))),grid) / integrate(That*((T*(S0invS1inv*(dLdomega*eig_vec)))*conj(T*eigl_vec)),grid);
                //SPI::printfc("in LST_spatial_cg UltraS cg = %.10f + %.10fi",cg);
                //SPIgrid1D grid2(grid.y,"grid",SPI::Chebyshev);
                //cg = integrate(((T*(S0invS1inv*(M*eig_vec)))*conj(T*eigl_vec)),grid2) / integrate(((T*(S0invS1inv*(dLdomega*eig_vec)))*conj(T*eigl_vec)),grid2);
                //SPI::printfc("in LST_spatial_cg UltraSp cg = %.10f + %.10fi",cg);
                //cg = ((T*((eig_vec))).dot(T*(S0invS1inv*(M*eigl_vec)))) / ((T*(S0invS1inv*(dLdomega*eig_vec))).dot(T*eigl_vec));
                //SPI::printfc("in LST_spatial_cg UltraS to physical dot cg = %.10f + %.10fi",cg);
                //SPI::printfc("in LST_spatial_cg vecdot cg = %.10f + %.10fi",eig_vec.dot(eigl_vec));
                //SPI::printfc("in LST_spatial_cg vecdot cg = %.10f + %.10fi",eig_vec.dot(eigl_vec));
                //SPI::printfc("in LST_spatial_cg vecdot cg = %.10f + %.10fi",eig_vec.dot(eig_vec));
                //SPI::printfc("in LST_spatial_cg vecdot cg = %.10f + %.10fi",(T*eig_vec).dot(T*eig_vec));
                //SPI::printfc("in LST_spatial_cg vecdot cg = %.10f + %.10fi",(T*eig_vec).dot(T*eigl_vec));
                //SPI::printfc("in LST_spatial_cg vecdot cg = %.10f + %.10fi",(T*eig_vec).dot(T*eigl_vec));
                //cg = integrate((S0invS1inv*(M_noBCs*eig_vec))*conj(eigl_vec),grid) / integrate((S0invS1inv*(dLdomega*eig_vec))*conj(eigl_vec),grid);
                //SPI::printfc("in LST_spatial_cg noBCs cg = %.10f + %.10fi",cg);
                cg = (M_noBCs_physical*(T*eig_vec)).dot(T*eigl_vec) / ((dLdomega_physical*(T*eig_vec))).dot(T*eigl_vec);
                SPI::printfc("in LST_spatial_cg physical noBCs cg = %.10f + %.10fi",cg);
                //cg = (T*eig_vec).dot(M_noBCs_physical*(T*eigl_vec)) / ((dLdomega_physical*(T*eig_vec))).dot(T*eigl_vec);
                //SPI::printfc("in LST_spatial_cg physical noBCs cg = %.10f + %.10fi",cg);
                //SPIVec eig_vec2(eig_vec);
                //SPIVec eigl_vec2(eigl_vec);
                //eig_vec2 /= (T*eig_vec).dot(M_noBCs_physical*(T*eigl_vec));
                //cg = 1. / ((dLdomega_physical*(T*eig_vec2))).dot(T*eigl_vec);
                //SPI::printfc("in LST_spatial_cg physical noBCs normed cg = %.10f + %.10fi",cg);
                //SPIVec eig_vec_p(T*eig_vec);
                //SPIVec eigl_vec_p(T*eigl_vec);
                //eig_vec_p /= SPI::L2(eig_vec_p);
                //eigl_vec_p /= SPI::L2(eigl_vec_p);
                //cg = (((eig_vec_p)).dot(M_noBCs_physical*eigl_vec_p)) / ((dLdomega_physical*(eig_vec_p)).dot(eigl_vec_p));
                //SPI::printfc("in LST_spatial_cg physical noBCs normed2 cg = %.10f + %.10fi",cg);
                //SPIVec eig_vec_2H(conj(eig_vec));
                //SPIVec eig_vec_2(eig_vec);
                //SPIVec eigl_vec_2H(conj(eigl_vec));
                //SPIVec eigl_vec_2(eigl_vec);
                //std::cout<<"in LST_spatial_cg UltraS normed2 to norm value = "<<(integrate(eig_vec_2*conj(eig_vec_2),grid))<<std::endl;
                //std::cout<<"in LST_spatial_cg UltraSp normed2 to norm value = "<<(integrate(That*((T*eig_vec_2)*conj(T*eig_vec_2)),grid))<<std::endl;
                //std::cout<<"in LST_spatial_cg UltraS normed2 to sqrt(norm value) = "<<sqrt(integrate(eig_vec_2*conj(eig_vec_2),grid))<<std::endl;
                //PetscScalar normval = integrate(eig_vec_2*conj(eig_vec_2),grid);
                //if(PetscRealPart(normval)<=0.){
                    //eig_vec_2 /= sqrt(normval);
                //}else{
                    //eig_vec_2 /= sqrt(normval);
                //}
                //eig_vec_2H = conj(eig_vec_2);
                //eig_vec_2.conj();
                //eigl_vec_2 = ((1./integrate(eigl_vec_2*conj(eigl_vec_2),grid))*eigl_vec_2);
                //eigl_vec_2H = conj(eigl_vec_2);
                //eigl_vec_2.conj();
                //SPI::printfc("in LST_spatial_cg UltraS normed2 norm = %.10f + %.10fi",integrate(eig_vec_2*conj(eig_vec_2),grid));
                //SPI::printfc("in LST_spatial_cg UltraS normed2 norm = %.10f + %.10fi",integrate(eigl_vec_2*conj(eigl_vec_2),grid));
                //cg = integrate(((eig_vec_2))*((M*eigl_vec_2).conj()),grid) / integrate((dLdomega*(eig_vec_2))*conj(eigl_vec_2),grid);
                //SPI::printfc("in LST_spatial_cg UltraS normed2 cg = %.10f + %.10fi",cg);
                //M.H();
                //eig_vec2 /= integrate(eig_vec2*(S0invS1inv*(M*conj(eigl_vec))),grid);
                //SPI::printfc("in LST_spatial_cg physical noBCs normed normval = %.10f + %.10fi",integrate(eig_vec2*(S0invS1inv*(M*conj(eigl_vec))),grid));
                //cg = 1. / integrate((S0invS1inv*(dLdomega*(eig_vec2)))*conj(eigl_vec),grid);
                //M.H();
                //SPI::printfc("in LST_spatial_cg physical noBCs normed cg = %.10f + %.10fi",cg);
                //grid.ytype=SPI::Chebyshev;
                //cg = integrate((M_noBCs_physical*(T*eig_vec))*conj(T*eigl_vec),grid) / integrate((dLdomega_physical*(T*eig_vec))*conj(T*eigl_vec),grid);
                //SPI::printfc("in LST_spatial_cg physical integrate cg = %.10f + %.10fi",cg);
                //grid.ytype=SPI::UltraS;
            }else{
                //cg = ((M*eig_vec).dot(eigl_vec)) / ((dLdomega*eig_vec).dot(eigl_vec));
                //SPIVec eigl_vecconj(eigl_vec.conj());
                //eigl_vec.conj();
                cg = integrate((M*eig_vec)*conj(eigl_vec),grid) / integrate((dLdomega*eig_vec)*conj(eigl_vec),grid);
                SPI::printfc("in LST_spatial_cg physical integrate cg = %.10f + %.10fi",cg);
                cg = (((eig_vec)).dot(M*eigl_vec)) / ((dLdomega*(eig_vec)).dot(eigl_vec));
                SPI::printfc("in LST_spatial_cg physical dot cg = %.10f + %.10fi",cg);
            }
            //SPI::printfc("α is: %g+%gi",alpha);
            return std::make_tuple(alpha,cg,eigl_vec,eig_vec);
        }
        else if(0){
            SPI::SPIMat L(SPI::block({
                        {O,         O,          O,          O,          I,          O,          O,          O},
                        {O,         O,          O,          O,          O,          I,          O,          O},
                        {O,         O,          O,          O,          O,          O,          I,          O},
                        {O,         O,          O,          O,          O,          O,          O,          I},
                        {d,         -Re*Uy,     O,          O,          -i*Re*U,    O,          O,          -i*Re*I},
                        {O,         d,          O,          -Re*Dy,     O,          -i*Re*U,    O,          O},
                        {O,         O,          d,         -i*Re*beta*I,O,          O,          -i*Re*U,    O},
                        {O,         Dy,         i*beta*I,   O,          i*I,        O,          O,          O},
                        }),"L");
            SPI::SPIMat M(SPI::block({
                        {I,         O,          O,          O,          O,          O,          O,          O},
                        {O,         I,          O,          O,          O,          O,          O,          O},
                        {O,         O,          I,          O,          O,          O,          O,          O},
                        {O,         O,          O,          I,          O,          O,          O,          O},
                        {O,         O,          O,          O,          I,          O,          O,          O},
                        {O,         O,          O,          O,          O,          I,          O,          O},
                        {O,         O,          O,          O,          O,          O,          I,          O},
                        {O,         O,          O,          O,          O,          O,          O,          O},
                        }),"M");
            // set BCs
            //PetscInt rowBCs[] = {0,n-1,n,2*n-1,2*n,3*n-1,4*n,5*n-1,5*n,6*n-1,6*n,7*n-1}; // u,v,w at wall and freestream
            PetscInt rowBCs[] = {4*n,5*n-1,5*n,6*n-1,6*n,7*n-1}; // u,v,w at wall and freestream
            for(PetscInt rowi : rowBCs){
                //SPI::printf(std::to_string(rowi));
                L.zero_row(rowi);
                M.zero_row(rowi);
                L(rowi,rowi,1.0);
                M(rowi,rowi,60.0*i);
            }
            SPI::SPIVec eig_vec(grid.y.rows*8,"q");
            SPI::SPIVec eigl_vec(grid.y.rows*8,"q");
            //if(q.flag_init){
                //std::tie(alpha,eigl_vec,eig_vec) = SPI::eig_init(L,M,alpha,q.conj(),q);
            //}else{
            std::tie(alpha,eigl_vec,eig_vec) = SPI::eig(M,L,1./alpha);
            //}
            alpha = alpha; // invert
            params.alpha = alpha;
            //SPI::printfc("α is: %g+%gi",alpha);
            //return std::make_tuple(alpha,eig_vec);
            return std::make_tuple(alpha,alpha,eig_vec,eig_vec);
        }
        else{
            d = 1./Re*(Dyy - beta*beta*I) + i*omega*I;
            SPI::SPIMat L(SPI::block({
                        //u         v           w           p           vx          wx
                        {O,         i*Dy,       -beta*I,    O,          O,          O}, // continuity
                        {O,         O,          O,          O,          I,          O}, // v-sub
                        {O,         O,          O,          O,          O,          I}, // w-sub
                        {-i*d,      i*(Uy-U*Dy),beta*U,     O,          -1./Re*Dy,  -i*beta/Re*I}, // u-mom
                        {O,         i*Re*d,     O,          -i*Re*Dy,   Re*U,       O}, // v-mom
                        {O,         O,          i*Re*d,     beta*Re*I,  O,          Re*U}, // w-mom
                        }),"L");
            SPI::SPIMat M(SPI::block({
                        {I,         O,          O,          O,          O,          O},
                        {O,         I,          O,          O,          O,          O},
                        {O,         O,          I,          O,          O,          O},
                        {O,         O,          O,          I,          O,          O},
                        {O,         O,          O,          O,          I,          O},
                        {O,         O,          O,          O,          O,          I}
                        }),"M");
            // set BCs
            //PetscInt rowBCs[] = {0,n-1,n,2*n-1,2*n,3*n-1,4*n,5*n-1,5*n,6*n-1,6*n,7*n-1}; // u,v,w at wall and freestream
            PetscInt rowBCs[] = {0*n,1*n-1,1*n,2*n-1,2*n,3*n-1}; // u,v,w at wall and freestream
            for(PetscInt rowi : rowBCs){
                //SPI::printf(std::to_string(rowi));
                L.zero_row(rowi);
                M.zero_row(rowi);
                L(rowi,rowi,1.0);
                M(rowi,rowi,60.0*i);
            }
            SPI::SPIVec eig_vec(grid.y.rows*8,"q");
            SPI::SPIVec eigl_vec(grid.y.rows*8,"q");
            //if(q.flag_init){
                //std::tie(alpha,eigl_vec,eig_vec) = SPI::eig_init(L,M,alpha,q.conj(),q,1e-16,20000);
            //}else{
                std::tie(alpha,eigl_vec,eig_vec) = SPI::eig(L,M,alpha,1e-12,20000);
            //}
            //alpha = 1./alpha; // invert
            params.alpha = alpha;
            //SPI::printfc("α is: %g+%gi",alpha);
            //return std::make_tuple(alpha,eig_vec);
            return std::make_tuple(alpha,alpha,eig_vec,eig_vec);
        }
    }
    /** \brief solve the local stability theory problem for the linearized Navier-Stokes equations using parallel baseflow with omega being pure real, and alpha the eigenvalue \return tuple of eigenvalue and eigenvector closest to the target value e.g. std::tie(alpha,group_velocity,left_eig_vector,right_eig_vector) = LSTNP_spatial(params,grid,baseflow).  Will solve for closest eigenvalue to params.alpha */
    std::tuple<PetscScalar, PetscScalar, SPIVec, SPIVec> LSTNP_spatial(
            SPIparams &params,          ///< [inout] contains parameters including Re, alpha, and omega.  Will overwrite omega with true omega value once solved
            SPIgrid1D &grid,              ///< [in] grid class containing the grid location and respective derivatives
            SPIbaseflow &baseflow,      ///< [in] baseflow for parallel flow
            SPIVec ql,                  ///< [in] initial guess for spatial problem (adjoint) for left eigenfunction
            SPIVec qr                   ///< [in] initial guess for spatial problem
            ){
        PetscInt ny = grid.ny;
        PetscScalar Re = params.Re;
        PetscScalar Reinv = 1./Re;
        PetscScalar alpha = params.alpha;
        PetscScalar omega = params.omega;
        PetscScalar beta = params.beta;
        PetscScalar i=PETSC_i;
        SPIMat &O = grid.O;
        SPIMat &I = grid.I;
        SPIMat U = SPI::diag(baseflow.U);
        SPIMat V = SPI::diag(baseflow.V);
        SPIMat Ux = SPI::diag(baseflow.Ux);
        SPIMat Uy = SPI::diag(baseflow.Uy);
        SPIMat Uxy = SPI::diag(baseflow.Uxy);
        SPIMat Vy = SPI::diag(baseflow.Vy);
        SPIMat W = SPI::diag(baseflow.W);
        SPIMat Wx = SPI::diag(baseflow.Wx);
        SPIMat Wy = SPI::diag(baseflow.Wy);
        SPIMat Wxy = SPI::diag(baseflow.Wxy);
        SPIMat P = SPI::diag(baseflow.P);
        if(grid.ytype==UltraS){ // map baseflow to C^(2) coefficient space using S0, S1, T, That operators
            //SPIMat S1S0That = grid.S1*grid.S0*grid.That;
            SPIMat &S1S0That = grid.S1S0That;
            //SPIMat &S1 = grid.S1;
            SPIMat &T = grid.T;
            //SPIMat &That = grid.That;
            U = S1S0That*U*T;
            V = S1S0That*V*T;
            Ux = S1S0That*Ux*T;
            Uy = S1S0That*Uy*T;
            Uxy = S1S0That*Uxy*T;
            Vy = S1S0That*Vy*T;
            W = S1S0That*W*T;
            Wx = S1S0That*Wx*T;
            Wy = S1S0That*Wy*T;
            Wxy = S1S0That*Wxy*T;
            P = S1S0That*P*T;
            //std::cout<<"Altered the baseflow"<<std::endl;
        }
        SPIMat &Dy = grid.Dy;
        SPIMat &Dyy = grid.Dyy;
        SPIMat VDy(V*Dy,"V*Dy");
        if(grid.ytype==UltraS){
            //SPIMat Vp = SPI::diag(baseflow.V); // physical to physical baseflow
            //SPIMat &S0invS1inv = grid.S0invS1inv;
            //SPIMat Dyp = grid.T*((S0invS1inv*Dy)*grid.That); // from physical->Chebyshev->C^(2)->Chebyshev->physical
            //VDy = grid.S1S0That*((grid.T*(S0invS1inv*V))*(grid.T*(S0invS1inv*Dy)));
            //VDy = grid.S1S0That*((Vp*Dyp)*grid.T); // from Chebyshev to C^(2)
            //VDy = O;
            //VDy = V*S0invS1inv*Dy;
            VDy = grid.S1S0That*(diag(baseflow.V) * (grid.T*grid.S0invS1inv*Dy*grid.That)) * grid.T;
        }
        SPIMat d(VDy - Reinv*Dyy + i*beta*W + (-i*omega + beta*beta*Reinv)*I,"d");
        if(1){
            SPIMat ReinvI = Reinv*I;
            SPIMat ABC2(SPI::block({
                        {ReinvI,    O,      O,      O},
                        {O,         ReinvI, O,      O},
                        {O,         O,      ReinvI, O},
                        {O,         O,      O,      O}
                        })(),"ABC2");
            SPIMat O4(zeros(4*ny,4*ny));
            SPIMat D2(O4,"D2");
            SPIMat dA2(O4,"dA2");
            SPIMat iU = i*U;
            SPIMat iI = i*I;
            SPIMat ABC1(SPI::block({
                        {iU,    O,      O,      iI},
                        {O,     iU,     O,      O  },
                        {O,     O,      iU,     O  },
                        {iI,    O,      O,      O  }
                        })(),"ABC1");
            SPIMat D1(O4,"D1");
            SPIMat iUx = i*Ux;
            SPIMat dA1(SPI::block({
                        {iUx,   O,      O,      O  },
                        {O,     iUx,    O,      O  },
                        {O,     O,      iUx,    O  },
                        {O,     O,      O,      O  }
                        })(),"dA1");
            SPIMat ibetaI = i*beta*I;
            SPIMat ABC0(SPI::block({
                        {d+Ux,  Uy,     O,      O  },
                        {O,     d+Vy,   O,      Dy },
                        {Wx,    Wy,     d,      ibetaI},
                        {O,     Dy,     ibetaI, O  }
                        })(),"ABC0");
            SPIMat D0(SPI::block({
                        {U,     O,      O,      I  },
                        {O,     U,      O,      O  },
                        {O,     O,      U,      O  },
                        {I,     O,      O,      O  }
                        })(),"D0");
            SPIMat ibetaWx = i*beta*Wx;
            SPIMat dA0(SPI::block({
                        {ibetaWx,   Uxy,        O,          O  },
                        {O,         ibetaWx,    O,          O  },
                        {O,         Wxy,        ibetaWx,    O  },
                        {O,         O,          O,          O  }
                        })(),"dA0");
            // set BCs
            //PetscInt rowBCs[] = {0,n-1,n,2*n-1,2*n,3*n-1,4*n,5*n-1,5*n,6*n-1,6*n,7*n-1}; // u,v,w at wall and freestream
            //PetscInt rowBCs[] = {0*ny,1*ny-1,1*ny,2*ny-1,2*ny,3*ny-1}; // u,v,w at wall and freestream
            //for(PetscInt rowi : rowBCs){
            //    //SPI::printf(std::to_string(rowi));
            //    ABC2.zero_row(rowi);
            //    ABC1.zero_row(rowi);
            //    dA1.zero_row(rowi);
            //    ABC0.zero_row(rowi);
            //    D0.zero_row(rowi);
            //    dA0.zero_row(rowi);

            //    ABC2(rowi,rowi,1.0);
            //    ABC1(rowi,rowi,1.0);
            //    dA1(rowi,rowi,1.0);
            //    ABC0(rowi,rowi,1.0);
            //    D0(rowi,rowi,1.0);
            //    dA0(rowi,rowi,1.0);
            //}
            // alternative way to set the BCs
            if(grid.ytype==UltraS){
                std::vector<PetscInt> rowBCs = {1*ny-2,1*ny-1,2*ny-2,2*ny-1,3*ny-2,3*ny-1}; // u,v,w at wall and freestream
                ABC2.zero_rows(rowBCs);
                ABC1.zero_rows(rowBCs);
                dA1.zero_rows(rowBCs);
                ABC0.zero_rows(rowBCs);
                D0.zero_rows(rowBCs);
                dA0.zero_rows(rowBCs);
                for(PetscInt j=0; j<ny; ++j){
                    ABC0(rowBCs[0],j,grid.T(0,j,PETSC_TRUE));       // u at the wall
                    ABC0(rowBCs[1],j,grid.T(ny-1,j,PETSC_TRUE));    // u at the freestream
                    ABC0(rowBCs[2],ny+j,grid.T(0,j,PETSC_TRUE));       // v at the wall
                    ABC0(rowBCs[3],ny+j,grid.T(ny-1,j,PETSC_TRUE));    // v at the freestream
                    ABC0(rowBCs[4],2*ny+j,grid.T(0,j,PETSC_TRUE));       // w at the wall
                    ABC0(rowBCs[5],2*ny+j,grid.T(ny-1,j,PETSC_TRUE));    // w at the freestream
                }
                ABC2();
                ABC1();
                dA1();
                ABC0();
                D0();
                dA0();
                //std::cout<<"set BCs"<<std::endl;
                //ABC0.print();
                //grid.T.print();
            } 
            else{
                std::vector<PetscInt> rowBCs = {0*ny,1*ny-1,1*ny,2*ny-1,2*ny,3*ny-1}; // u,v,w at wall and freestream
                ABC2.zero_rows(rowBCs); // is eye_rows in python?  but isn't is supposed to be zero?
                ABC1.zero_rows(rowBCs); // is eye_rows in python... but isn't it supposed to be zero?
                dA1.zero_rows(rowBCs); // is eye_rows in python... but isn't it supposed to be zero?
                ABC0.eye_rows(rowBCs);
                D0.zero_rows(rowBCs); // is eye_rows in python... but isn't it supposed to be zero?
                dA0.zero_rows(rowBCs); // is eye_rows in python... but isn't it supposed to be zero?
            }

            // inflate for derivatives in streamwise direction
            SPIMat L0(block({
                        {ABC0,  D0  },
                        {dA0,   ABC0}})(),"L0");
            SPIMat L1(block({
                        {ABC1,  D1  },
                        {dA1,   ABC1}})(),"L1");
            SPIMat L2(block({
                        {ABC2,  D2  },
                        {dA2,   ABC2}})(),"L2");
            // inflate due to polynomial eigenvalue problem
            SPIMat O8(zeros(8*ny,8*ny));
            //SPIMat I8(eye(8*ny));
            SPIMat I8(block({
                        {I,O,O,O,O,O,O,O},
                        {O,I,O,O,O,O,O,O},
                        {O,O,I,O,O,O,O,O},
                        {O,O,O,I,O,O,O,O},
                        {O,O,O,O,I,O,O,O},
                        {O,O,O,O,O,I,O,O},
                        {O,O,O,O,O,O,I,O},
                        {O,O,O,O,O,O,O,I},
                        })(),"I8");
            SPIMat L(block({
                        {O8,    I8},
                        {L0,    L1}})(),"L");
            SPIMat M(block({
                        {I8,    O8},
                        {O8,    -L2}})(),"M");
            //L.print();
            //M.print();
            //SPIMat L(block({
            //{O8,    I8},
            //{-L0,    -L1}}),"L");
            //SPIMat M(block({
            //{I8,    O8},
            //{O8,    L2}}),"M");
            //SPIMat L(block({
            //{-L0,    O8},
            //{O8,     I8}}),"L");
            //SPIMat M(block({
            //{L1,    L2},
            //{I8,    O8}}),"M");
            //PetscScalar a=10.5,b=20.5;
            //SPIMat L(block({
            //{-b*L0,     a*I8},
            //{-a*L0,     -a*L1+b*I8}}),"L");
            //SPIMat M(block({
            //{a*I8+b*L1, b*L2},
            //{b*I8,      a*L2}}),"M");


            SPI::SPIVec eig_vec(grid.y.rows*16,"q");
            SPI::SPIVec eigl_vec(grid.y.rows*16,"q");
            if(ql.flag_init){
                std::tie(alpha,eigl_vec,eig_vec) = SPI::eig_init(L,M,alpha,ql,qr);
            }else{
                std::tie(alpha,eigl_vec,eig_vec) = SPI::eig(L,M,alpha);
                //std::tie(alpha,eig_vec) = SPI::eig_right(L,M,alpha);
            }
            //SPIMat L(block(
            //alpha = alpha; // invert
            params.alpha = alpha;
            //SPI::printfc("α is: %g+%gi",alpha);
            SPIMat dLdomega4(block({
                        {-i*I,  O,      O,      O},
                        {O,     -i*I,   O,      O},
                        {O,     O,      -i*I,   O},
                        {O,     O,      O,      O},
                        })(),"dLdomega 4nx4n");
            // inflate for non-parallel
            SPIMat dLdomega8(block({
                        {dLdomega4, O4},
                        {O4,        dLdomega4}
                        })(),"dLdomega 8nx8n");
            // inflate for polynomial eigenvalue problem
            SPIMat dLdomega(block({
                        {O8,        O8},
                        {dLdomega8, O8}
                        })(),"dLdomega 16nx16n");
            PetscScalar cg;
            if(grid.ytype==UltraS){
                // TODO this group velocity is incorrect for the test case... need to fix!
                //SPIMat &s = grid.S0invS1inv;
                SPIMat &t = grid.T;
                //SPIMat S(block({
                //            {s,O,O,O,O,O,O,O,O,O,O,O,O,O,O,O},
                //            {O,s,O,O,O,O,O,O,O,O,O,O,O,O,O,O},
                //            {O,O,s,O,O,O,O,O,O,O,O,O,O,O,O,O},
                //            {O,O,O,s,O,O,O,O,O,O,O,O,O,O,O,O},
                //            {O,O,O,O,s,O,O,O,O,O,O,O,O,O,O,O},
                //            {O,O,O,O,O,s,O,O,O,O,O,O,O,O,O,O},
                //            {O,O,O,O,O,O,s,O,O,O,O,O,O,O,O,O},
                //            {O,O,O,O,O,O,O,s,O,O,O,O,O,O,O,O},
                //            {O,O,O,O,O,O,O,O,s,O,O,O,O,O,O,O},
                //            {O,O,O,O,O,O,O,O,O,s,O,O,O,O,O,O},
                //            {O,O,O,O,O,O,O,O,O,O,s,O,O,O,O,O},
                //            {O,O,O,O,O,O,O,O,O,O,O,s,O,O,O,O},
                //            {O,O,O,O,O,O,O,O,O,O,O,O,s,O,O,O},
                //            {O,O,O,O,O,O,O,O,O,O,O,O,O,s,O,O},
                //            {O,O,O,O,O,O,O,O,O,O,O,O,O,O,s,O},
                //            {O,O,O,O,O,O,O,O,O,O,O,O,O,O,O,s}
                //            })(),"S0invS1inv");
                SPIMat T(block({
                            {t,O,O,O,O,O,O,O,O,O,O,O,O,O,O,O},
                            {O,t,O,O,O,O,O,O,O,O,O,O,O,O,O,O},
                            {O,O,t,O,O,O,O,O,O,O,O,O,O,O,O,O},
                            {O,O,O,t,O,O,O,O,O,O,O,O,O,O,O,O},
                            {O,O,O,O,t,O,O,O,O,O,O,O,O,O,O,O},
                            {O,O,O,O,O,t,O,O,O,O,O,O,O,O,O,O},
                            {O,O,O,O,O,O,t,O,O,O,O,O,O,O,O,O},
                            {O,O,O,O,O,O,O,t,O,O,O,O,O,O,O,O},
                            {O,O,O,O,O,O,O,O,t,O,O,O,O,O,O,O},
                            {O,O,O,O,O,O,O,O,O,t,O,O,O,O,O,O},
                            {O,O,O,O,O,O,O,O,O,O,t,O,O,O,O,O},
                            {O,O,O,O,O,O,O,O,O,O,O,t,O,O,O,O},
                            {O,O,O,O,O,O,O,O,O,O,O,O,t,O,O,O},
                            {O,O,O,O,O,O,O,O,O,O,O,O,O,t,O,O},
                            {O,O,O,O,O,O,O,O,O,O,O,O,O,O,t,O},
                            {O,O,O,O,O,O,O,O,O,O,O,O,O,O,O,t}
                            })(),"T");
                SPIMat dLdomega4_physical(block({
                            {-i*eye(ny),     O,         O,          O},
                            {O,             -i*eye(ny), O,          O},
                            {O,             O,          -i*eye(ny), O},
                            {O,             O,          O,          O},
                            })(),"dLdomega 4nx4n");
                // inflate for non-parallel
                SPIMat dLdomega8_physical(block({
                            {dLdomega4_physical,O4},
                            {O4,                dLdomega4_physical}
                            })(),"dLdomega 8nx8n");
                // inflate for polynomial eigenvalue problem
                SPIMat dLdomega_physical(block({
                            {O8,                O8},
                            {dLdomega8_physical,O8}
                            })(),"dLdomega 16nx16n");
                SPIMat ABC2_physical(SPI::block({
                            {Reinv*eye(ny), O,              O,              O},
                            {O,             Reinv*eye(ny),  O,              O},
                            {O,             O,              Reinv*eye(ny),  O},
                            {O,             O,              O,              O}
                            })(),"ABC2");
                SPIMat L2_physical(block({
                            {ABC2_physical,  D2  },
                            {dA2,   ABC2_physical}})(),"L2");
                SPIMat M_physical(block({
                            {eye(8*ny),     O8},
                            {O8,            -L2_physical}})(),"M");
                //SPIVec eigl_vecconj(eigl_vec.conj());
                //eigl_vec.conj();
                //SPIVec tmp((S16*(M*eig_vec))*eigl_vecconj);
                //PetscScalar numerator = integrate((S*(M*eig_vec))*(eigl_vecconj),grid);
                //PetscScalar denominator = integrate((S*(dLdomega*eig_vec))*(eigl_vecconj),grid);
                //eigl_vec.conj();
                //PetscScalar numerator = (T*S*(M*eig_vec)).dot(T*eigl_vec);
                //PetscScalar denominator = (T*S*(dLdomega*eig_vec)).dot(T*eigl_vec);
                PetscScalar numerator = (((M_physical*(T*eig_vec))).dot(T*eigl_vec));
                PetscScalar denominator = (((dLdomega_physical*(T*eig_vec))).dot(T*eigl_vec));
                cg = numerator/denominator;
            }
            else{
                cg = ((M*eig_vec).dot(eigl_vec)) / ((dLdomega*eig_vec).dot(eigl_vec));
            }
            return std::make_tuple(alpha,cg,eigl_vec,eig_vec);
        }
    }
    /** \brief solve the local stability theory problem for the linearized Navier-Stokes equations using parallel baseflow with omega being pure real, and alpha the eigenvalue \return tuple of eigenvalue and eigenvector closest to the target value e.g. std::tie(alpha,right_eig_vector) = LSTNP_spatial(params,grid,baseflow).  Will solve for closest eigenvalue to params.alpha */
    std::tuple<PetscScalar, SPIVec> LSTNP_spatial_right(
            SPIparams &params,          ///< [inout] contains parameters including Re, alpha, and omega.  Will overwrite omega with true omega value once solved
            SPIgrid1D &grid,              ///< [in] grid class containing the grid location and respective derivatives
            SPIbaseflow &baseflow,      ///< [in] baseflow for parallel flow
            SPIVec qr                   ///< [in] initial guess for spatial problem
            ){
        PetscInt ny = grid.ny;
        PetscScalar Re = params.Re;
        PetscScalar Reinv = 1.0/Re;
        PetscScalar alpha = params.alpha;
        PetscScalar omega = params.omega;
        PetscScalar beta = params.beta;
        PetscScalar i=PETSC_i;
        SPIMat &O = grid.O;
        SPIMat &I = grid.I;
        SPIMat U = SPI::diag(baseflow.U);
        SPIMat V = SPI::diag(baseflow.V);
        SPIMat Ux = SPI::diag(baseflow.Ux);
        SPIMat Uy = SPI::diag(baseflow.Uy);
        SPIMat Uxy = SPI::diag(baseflow.Uxy);
        SPIMat Vy = SPI::diag(baseflow.Vy);
        SPIMat W = SPI::diag(baseflow.W);
        SPIMat Wx = SPI::diag(baseflow.Wx);
        SPIMat Wy = SPI::diag(baseflow.Wy);
        SPIMat Wxy = SPI::diag(baseflow.Wxy);
        SPIMat P = SPI::diag(baseflow.P);
        if(grid.ytype==SPI::UltraS){
            SPIMat &S1S0That = grid.S1S0That;
            SPIMat &T = grid.T;
            U = S1S0That*U*T;
            V = S1S0That*V*T;
            Ux = S1S0That*Ux*T;
            Uy = S1S0That*Uy*T;
            Uxy = S1S0That*Uxy*T;
            Vy = S1S0That*Vy*T;
            W = S1S0That*W*T;
            Wx = S1S0That*Wx*T;
            Wy = S1S0That*Wy*T;
            Wxy = S1S0That*Wxy*T;
            P = S1S0That*P*T;
        }
        SPIMat &Dy = grid.Dy;
        SPIMat &Dyy = grid.Dyy;
        SPIMat VDy(V*Dy,"V*Dy");
        if(grid.ytype==UltraS){
            //SPIMat &S0invS1inv = grid.S0invS1inv;
            //VDy = V*S0invS1inv*Dy;
            //VDy = grid.S1S0That*((grid.T*(S0invS1inv*V))*(grid.T*(S0invS1inv*Dy)));
            VDy = grid.S1S0That*(diag(baseflow.V) * (grid.T*grid.S0invS1inv*Dy*grid.That)) * grid.T;
            //VDy = grid.O;
        }
        SPIMat d(VDy - Reinv*Dyy + i*beta*W + (-i*omega + beta*beta*Reinv)*I,"d");
        if(1){
            SPIMat ReinvI(Reinv*I);
            SPIMat ABC2(SPI::block({
                        {ReinvI,    O,      O,      O},
                        {O,         ReinvI, O,      O},
                        {O,         O,      ReinvI, O},
                        {O,         O,      O,      O}
                        })(),"ABC2");
            SPIMat O4(zeros(4*ny,4*ny));
            SPIMat D2(O4,"D2");
            SPIMat dA2(O4,"dA2");
            SPIMat iU(i*U);
            SPIMat iI(i*I);
            SPIMat ABC1(SPI::block({
                        {iU,    O,      O,      iI },
                        {O,     iU,     O,      O  },
                        {O,     O,      iU,     O  },
                        {iI,    O,      O,      O  }
                        })(),"ABC1");
            SPIMat D1(O4,"D1");
            SPIMat iUx(i*Ux);
            SPIMat dA1(SPI::block({
                        {iUx,   O,      O,      O  },
                        {O,     iUx,    O,      O  },
                        {O,     O,      iUx,    O  },
                        {O,     O,      O,      O  }
                        })(),"dA1");
            SPIMat ibetaI(i*beta*I);
            SPIMat ABC0(SPI::block({
                        {d+Ux,  Uy,     O,      O  },
                        {O,     d+Vy,   O,      Dy },
                        {Wx,    Wy,     d,      ibetaI},
                        {O,     Dy,    ibetaI,O  }
                        })(),"ABC0");
            SPIMat D0(SPI::block({
                        {U,     O,      O,      I  },
                        {O,     U,      O,      O  },
                        {O,     O,      U,      O  },
                        {I,     O,      O,      O  }
                        })(),"D0");
            SPIMat ibetaWx(i*beta*Wx);
            SPIMat dA0(SPI::block({
                        {ibetaWx,   Uxy,        O,          O  },
                        {O,         ibetaWx,    O,          O  },
                        {O,         Wxy,        ibetaWx,    O  },
                        {O,         O,          O,          O  }
                        })(),"dA0");
            // set BCs
            //PetscInt rowBCs[] = {0,n-1,n,2*n-1,2*n,3*n-1,4*n,5*n-1,5*n,6*n-1,6*n,7*n-1}; // u,v,w at wall and freestream
            //PetscInt rowBCs[] = {0*ny,1*ny-1,1*ny,2*ny-1,2*ny,3*ny-1}; // u,v,w at wall and freestream
            //for(PetscInt rowi : rowBCs){
            //    //SPI::printf(std::to_string(rowi));
            //    ABC2.zero_row(rowi);
            //    ABC1.zero_row(rowi);
            //    dA1.zero_row(rowi);
            //    ABC0.zero_row(rowi);
            //    D0.zero_row(rowi);
            //    dA0.zero_row(rowi);

            //    ABC2(rowi,rowi,1.0);
            //    ABC1(rowi,rowi,1.0);
            //    dA1(rowi,rowi,1.0);
            //    ABC0(rowi,rowi,1.0);
            //    D0(rowi,rowi,1.0);
            //    dA0(rowi,rowi,1.0);
            //}
            // alternative way to set the BCs
            if(grid.ytype==SPI::UltraS){
                std::vector<PetscInt> rowBCs = {1*ny-2,1*ny-1,2*ny-2,2*ny-1,3*ny-2,3*ny-1}; // u,v,w at wall and freestream
                ABC2.zero_rows(rowBCs);
                ABC1.zero_rows(rowBCs);
                dA1.zero_rows(rowBCs);
                ABC0.zero_rows(rowBCs);
                D0.zero_rows(rowBCs);
                dA0.zero_rows(rowBCs);
                for(PetscInt j=0; j<ny; ++j){
                    ABC0(rowBCs[0],j,grid.T(0,j,PETSC_TRUE));       // u at the wall
                    ABC0(rowBCs[1],j,grid.T(ny-1,j,PETSC_TRUE));    // u at the freestream
                    ABC0(rowBCs[2],ny+j,grid.T(0,j,PETSC_TRUE));       // v at the wall
                    ABC0(rowBCs[3],ny+j,grid.T(ny-1,j,PETSC_TRUE));    // v at the freestream
                    ABC0(rowBCs[4],2*ny+j,grid.T(0,j,PETSC_TRUE));       // w at the wall
                    ABC0(rowBCs[5],2*ny+j,grid.T(ny-1,j,PETSC_TRUE));    // w at the freestream
                }
                // assemble
                ABC0();
                // SPIMat P = block({
                //         {grid.P,O,O,O},
                //         {O,grid.P,O,O},
                //         {O,O,grid.P,O},
                //         {O,O,O,eye(ny)},
                //         })();
                // // reorder rows for UltraSpherical
                // ABC2 = P*ABC2;
                // ABC1 = P*ABC1;
                // dA1 = P*dA1;
                // ABC0 = P*ABC0;
                // D0 = P*D0;
                // dA0 = P*dA0;
            }else{
                std::vector<PetscInt> rowBCs = {0*ny,1*ny-1,1*ny,2*ny-1,2*ny,3*ny-1}; // u,v,w at wall and freestream
                ABC2.zero_rows(rowBCs);// is eye_rows in python...
                ABC1.zero_rows(rowBCs);// is eye_rows in python...
                dA1.zero_rows(rowBCs);// is eye_rows in python...
                ABC0.eye_rows(rowBCs);
                D0.zero_rows(rowBCs);// is eye_rows in python...
                dA0.zero_rows(rowBCs);// is eye_rows in python...
            }

            // inflate for derivatives in streamwise direction
            SPIMat L0(block({
                        {ABC0,  D0  },
                        {dA0,   ABC0}})(),"L0");
            SPIMat L1(block({
                        {ABC1,  D1  },
                        {dA1,   ABC1}})(),"L1");
            SPIMat L2(block({
                        {ABC2,  D2  },
                        {dA2,   ABC2}})(),"L2");
            // inflate due to polynomial eigenvalue problem
            //if(grid.ytype==SPI::UltraS){
                //SPI::SPIVec eig_vec_tmp(grid.y.rows*8,"q");
                //std::tie(alpha,eig_vec_tmp) = SPI::polyeig({L0,L1,L2},alpha);
                //return std::make_tuple(alpha,eig_vec_tmp);
            //}
            SPIMat O8(zeros(8*ny,8*ny));
            //SPIMat I8(eye(8*ny));
            SPIMat I8(block({
                        {I,O,O,O,O,O,O,O},
                        {O,I,O,O,O,O,O,O},
                        {O,O,I,O,O,O,O,O},
                        {O,O,O,I,O,O,O,O},
                        {O,O,O,O,I,O,O,O},
                        {O,O,O,O,O,I,O,O},
                        {O,O,O,O,O,O,I,O},
                        {O,O,O,O,O,O,O,I},
                        })(),"I8");
            SPIMat L(block({
                        {O8,    I8},
                        {L0,    L1}})(),"L");
            SPIMat M(block({
                        {I8,    O8},
                        {O8,    -L2}})(),"M");
            //SPIMat L(block({
            //{O8,    I8},
            //{-L0,    -L1}}),"L");
            //SPIMat M(block({
            //{I8,    O8},
            //{O8,    L2}}),"M");
            //SPIMat L(block({
            //{-L0,    O8},
            //{O8,     I8}}),"L");
            //SPIMat M(block({
            //{L1,    L2},
            //{I8,    O8}}),"M");
            //PetscScalar a=10.5,b=20.5;
            //SPIMat L(block({
            //{-b*L0,     a*I8},
            //{-a*L0,     -a*L1+b*I8}}),"L");
            //SPIMat M(block({
            //{a*I8+b*L1, b*L2},
            //{b*I8,      a*L2}}),"M");


            SPI::SPIVec eig_vec(grid.y.rows*16,"q");
            if(qr.flag_init){
                std::tie(alpha,eig_vec) = SPI::eig_init_right(L,M,alpha,qr);
            }else{
                std::tie(alpha,eig_vec) = SPI::eig_right(L,M,alpha);
            }
            eig_vec.rows=4*ny;
            //SPIMat L(block(
            //alpha = alpha; // invert
            params.alpha = alpha;
            //SPI::printfc("α is: %g+%gi",alpha);
            return std::make_tuple(alpha,eig_vec);
        }
    }
    /** \brief solve the local stability theory problem for the linearized Navier-Stokes equations using parallel baseflow with omega being pure real, and alpha the eigenvalue \return tuple of eigenvalue and eigenvector closest to the target value e.g. std::tie(alpha,right_eig_vector) = LSTNP_spatial(params,grid,baseflow).  Will solve for closest eigenvalue to params.alpha */
    std::tuple<PetscScalar, SPIVec> LSTNP_spatial_right2(
            SPIparams &params,          ///< [inout] contains parameters including Re, alpha, and omega.  Will overwrite omega with true omega value once solved
            SPIgrid1D &grid,              ///< [in] grid class containing the grid location and respective derivatives
            SPIbaseflow &baseflow,      ///< [in] baseflow for parallel flow
            SPIVec qr                   ///< [in] initial guess for spatial problem
            ){
        PetscInt ny = grid.ny;
        PetscScalar Re = params.Re;
        PetscScalar Reinv = 1.0/Re;
        PetscScalar alpha = params.alpha;
        PetscScalar omega = params.omega;
        PetscScalar beta = params.beta;
        PetscScalar i=PETSC_i;
        SPIMat &O = grid.O;
        SPIMat &I = grid.I;
        SPIMat U = SPI::diag(baseflow.U);
        SPIMat V = SPI::diag(baseflow.V);
        SPIMat Ux = SPI::diag(baseflow.Ux);
        SPIMat Uy = SPI::diag(baseflow.Uy);
        SPIMat Uxy = SPI::diag(baseflow.Uxy);
        SPIMat Vy = SPI::diag(baseflow.Vy);
        SPIMat W = SPI::diag(baseflow.W);
        SPIMat Wx = SPI::diag(baseflow.Wx);
        SPIMat Wy = SPI::diag(baseflow.Wy);
        SPIMat Wxy = SPI::diag(baseflow.Wxy);
        SPIMat P = SPI::diag(baseflow.P);
        if(grid.ytype==SPI::UltraS){
            SPIMat &S1S0That = grid.S1S0That;
            SPIMat &T = grid.T;
            U = S1S0That*U*T;
            V = S1S0That*V*T;
            Ux = S1S0That*Ux*T;
            Uy = S1S0That*Uy*T;
            Uxy = S1S0That*Uxy*T;
            Vy = S1S0That*Vy*T;
            W = S1S0That*W*T;
            Wx = S1S0That*Wx*T;
            Wy = S1S0That*Wy*T;
            Wxy = S1S0That*Wxy*T;
            P = S1S0That*P*T;
        }
        SPIMat &Dy = grid.Dy;
        SPIMat &Dyy = grid.Dyy;
        SPIMat VDy(V*Dy,"V*Dy");
        if(grid.ytype==UltraS){
            VDy = grid.S1S0That*(diag(baseflow.V) * (grid.T*grid.S0invS1inv*Dy*grid.That)) * grid.T;
        }
        SPIMat d(VDy - Reinv*Dyy + i*beta*W + (-i*omega + beta*beta*Reinv)*I,"d");
        if(1){
            SPIMat ReinvI(Reinv*I);
            SPIMat ABC2(SPI::block({
                        {ReinvI,    O,      O,      O},
                        {O,         ReinvI, O,      O},
                        {O,         O,      ReinvI, O},
                        {O,         O,      O,      O}
                        })(),"ABC2");
            SPIMat O4(zeros(4*ny,4*ny));
            SPIMat D2(O4,"D2");
            SPIMat dA2(O4,"dA2");
            SPIMat iU(i*U);
            SPIMat iI(i*I);
            SPIMat ABC1(SPI::block({
                        {iU,    O,      O,      iI },
                        {O,     iU,     O,      O  },
                        {O,     O,      iU,     O  },
                        {iI,    O,      O,      O  }
                        })(),"ABC1");
            SPIMat D1(O4,"D1");
            SPIMat iUx(i*Ux);
            SPIMat dA1(SPI::block({
                        {iUx,   O,      O,      O  },
                        {O,     iUx,    O,      O  },
                        {O,     O,      iUx,    O  },
                        {O,     O,      O,      O  }
                        })(),"dA1");
            SPIMat ibetaI(i*beta*I);
            SPIMat ABC0(SPI::block({
                        {d+Ux,  Uy,     O,      O  },
                        {O,     d+Vy,   O,      Dy },
                        {Wx,    Wy,     d,      ibetaI},
                        {O,     Dy,    ibetaI,O  }
                        })(),"ABC0");
            SPIMat D0(SPI::block({
                        {U,     O,      O,      I  },
                        {O,     U,      O,      O  },
                        {O,     O,      U,      O  },
                        {I,     O,      O,      O  }
                        })(),"D0");
            SPIMat ibetaWx(i*beta*Wx);
            SPIMat dA0(SPI::block({
                        {ibetaWx,   Uxy,        O,          O  },
                        {O,         ibetaWx,    O,          O  },
                        {O,         Wxy,        ibetaWx,    O  },
                        {O,         O,          O,          O  }
                        })(),"dA0");
            // alternative way to set the BCs
            if(grid.ytype==SPI::UltraS){
                std::vector<PetscInt> rowBCs = {1*ny-2,1*ny-1,2*ny-2,2*ny-1,3*ny-2,3*ny-1}; // u,v,w at wall and freestream
                ABC2.zero_rows(rowBCs);
                ABC1.zero_rows(rowBCs);
                dA1.zero_rows(rowBCs);
                ABC0.zero_rows(rowBCs);
                D0.zero_rows(rowBCs);
                dA0.zero_rows(rowBCs);
                for(PetscInt j=0; j<ny; ++j){
                    ABC0(rowBCs[0],j,grid.T(0,j,PETSC_TRUE));       // u at the wall
                    ABC0(rowBCs[1],j,grid.T(ny-1,j,PETSC_TRUE));    // u at the freestream
                    ABC0(rowBCs[2],ny+j,grid.T(0,j,PETSC_TRUE));       // v at the wall
                    ABC0(rowBCs[3],ny+j,grid.T(ny-1,j,PETSC_TRUE));    // v at the freestream
                    ABC0(rowBCs[4],2*ny+j,grid.T(0,j,PETSC_TRUE));       // w at the wall
                    ABC0(rowBCs[5],2*ny+j,grid.T(ny-1,j,PETSC_TRUE));    // w at the freestream
                }
                // assemble
                ABC0();
                SPIMat P = block({
                        {grid.P,O,O,O},
                        {O,grid.P,O,O},
                        {O,O,grid.P,O},
                        {O,O,O,eye(ny)},
                        })();
                // reorder rows for UltraSpherical
                ABC2 = P*ABC2;
                ABC1 = P*ABC1;
                dA1 = P*dA1;
                ABC0 = P*ABC0;
                D0 = P*D0;
                dA0 = P*dA0;
            }else{
                std::vector<PetscInt> rowBCs = {0*ny,1*ny-1,1*ny,2*ny-1,2*ny,3*ny-1}; // u,v,w at wall and freestream
                ABC2.zero_rows(rowBCs);// is eye_rows in python...
                ABC1.zero_rows(rowBCs);// is eye_rows in python...
                dA1.zero_rows(rowBCs);// is eye_rows in python...
                ABC0.eye_rows(rowBCs);
                D0.zero_rows(rowBCs);// is eye_rows in python...
                dA0.zero_rows(rowBCs);// is eye_rows in python...
            }

            // inflate for derivatives in streamwise direction
            SPIMat L0(block({
                        {ABC0,  D0  },
                        {dA0,   ABC0}})(),"L0");
            SPIMat L1(block({
                        {ABC1,  D1  },
                        {dA1,   ABC1}})(),"L1");
            SPIMat L2(block({
                        {ABC2,  D2  },
                        {dA2,   ABC2}})(),"L2");
            // inflate due to polynomial eigenvalue problem
            //if(grid.ytype==SPI::UltraS){
                //SPI::SPIVec eig_vec_tmp(grid.y.rows*8,"q");
                //std::tie(alpha,eig_vec_tmp) = SPI::polyeig({L0,L1,L2},alpha);
                //return std::make_tuple(alpha,eig_vec_tmp);
            //}
            SPIMat O8(zeros(8*ny,8*ny));
            //SPIMat I8(eye(8*ny));
            SPIMat I8(block({
                        {I,O,O,O,O,O,O,O},
                        {O,I,O,O,O,O,O,O},
                        {O,O,I,O,O,O,O,O},
                        {O,O,O,I,O,O,O,O},
                        {O,O,O,O,I,O,O,O},
                        {O,O,O,O,O,I,O,O},
                        {O,O,O,O,O,O,I,O},
                        {O,O,O,O,O,O,O,I},
                        })(),"I8");
            SPIMat L(block({
                        {O8,    I8},
                        {L0,    L1}})(),"L");
            SPIMat M(block({
                        {I8,    O8},
                        {O8,    -L2}})(),"M");
            //SPIMat L(block({
                        //{O8,    I8},
                        //{-L0,    -L1}})(),"L");
            //SPIMat M(block({
                        //{I8,    O8},
                        //{O8,    L2}})(),"M");
            //SPIMat L(block({
                        //{-L0,    O8},
                        //{O8,     I8}})(),"L");
            //SPIMat M(block({
                        //{L1,    L2},
                        //{I8,    O8}})(),"M");
            //PetscScalar a=10.5,b=20.5;
            //SPIMat L(block({
                        //{-b*L0,     a*I8},
                        //{-a*L0,     -a*L1+b*I8}})(),"L");
            //SPIMat M(block({
                        //{a*I8+b*L1, b*L2},
                        //{b*I8,      a*L2}})(),"M");


            SPI::SPIVec eig_vec(grid.y.rows*16,"q");
            if(qr.flag_init){
                std::tie(alpha,eig_vec) = SPI::eig_init_right(L,M,alpha,qr);
            }else{
                std::tie(alpha,eig_vec) = SPI::eig_right(L,M,alpha);
            }
            //SPIMat L(block(
            //alpha = alpha; // invert
            params.alpha = alpha;
            //SPI::printfc("α is: %g+%gi",alpha);
            return std::make_tuple(alpha,eig_vec);
        }
    }
    /** \brief solve the local stability theory problem for the linearized Navier-Stokes equations using non-parallel baseflow with omega being pure real, and alpha the eigenvalue \return tuple of eigenvalue and eigenvector closest to the target value e.g. std::tie(alphas,right_eig_vectors) = LSTNP_spatials_right(params,grid,baseflow).  Will solve for closest eigenvalue to params.alpha */
    std::tuple<std::vector<PetscScalar>, std::vector<SPIVec>> LSTNP_spatials_right(
            SPIparams &params,                  ///< [inout] contains parameters including Re and omega.
            SPIgrid1D &grid,                      ///< [in] grid class containing the grid location and respective derivatives
            SPIbaseflow &baseflow,              ///< [in] baseflow for parallel flow
            std::vector<PetscScalar> &alphas,    ///< [in] vector of alpha guesses
            std::vector<SPIVec> &qrs             ///< [in] initial guesses for spatial problem
            ){
        PetscInt ny = grid.ny;
        PetscScalar Re = params.Re;
        PetscScalar Reinv = 1./Re;
        //PetscScalar alpha = alphas[0];
        std::vector<PetscScalar> _alphas;
        std::vector<SPIVec> eig_vecs;
        PetscScalar omega = params.omega;
        PetscScalar beta = params.beta;
        PetscScalar i=PETSC_i;
        SPIMat &O = grid.O;
        SPIMat &I = grid.I;
        SPIMat U = SPI::diag(baseflow.U);
        SPIMat V = SPI::diag(baseflow.V);
        SPIMat Ux = SPI::diag(baseflow.Ux);
        SPIMat Uy = SPI::diag(baseflow.Uy);
        SPIMat Uxy = SPI::diag(baseflow.Uxy);
        SPIMat Vy = SPI::diag(baseflow.Vy);
        SPIMat W = SPI::diag(baseflow.W);
        SPIMat Wx = SPI::diag(baseflow.Wx);
        SPIMat Wy = SPI::diag(baseflow.Wy);
        SPIMat Wxy = SPI::diag(baseflow.Wxy);
        SPIMat P = SPI::diag(baseflow.P);
        if(grid.ytype==SPI::UltraS){
            SPIMat &S1S0That = grid.S1S0That;
            SPIMat &T = grid.T;
            U = S1S0That*U*T;
            V = S1S0That*V*T;
            Ux = S1S0That*Ux*T;
            Uy = S1S0That*Uy*T;
            Uxy = S1S0That*Uxy*T;
            Vy = S1S0That*Vy*T;
            W = S1S0That*W*T;
            Wx = S1S0That*Wx*T;
            Wy = S1S0That*Wy*T;
            Wxy = S1S0That*Wxy*T;
            P = S1S0That*P*T;
        }
        SPIMat &Dy = grid.Dy;
        SPIMat &Dyy = grid.Dyy;
        SPIMat VDy(V*Dy,"V*Dy");
        if(grid.ytype==UltraS){
            //SPIMat &S0invS1inv = grid.S0invS1inv;
            //VDy = V*S0invS1inv*Dy;
            //VDy = grid.S1S0That*((grid.T*(S0invS1inv*V))*(grid.T*(S0invS1inv*Dy)));
            //VDy = grid.O;
            VDy = grid.S1S0That*(diag(baseflow.V) * (grid.T*grid.S0invS1inv*Dy*grid.That)) * grid.T;
        }
        SPIMat d(VDy - Reinv*Dyy + i*beta*W + (-i*omega + beta*beta*Reinv)*I,"d");
        if(1){
            SPIMat ABC2(SPI::block({
                        {Reinv*I,   O,      O,      O},
                        {O,         Reinv*I,O,      O},
                        {O,         O,      Reinv*I,O},
                        {O,         O,      O,      O}
                        })(),"ABC2");
            SPIMat O4(zeros(4*ny,4*ny));
            SPIMat D2(O4,"D2");
            SPIMat dA2(O4,"dA2");
            SPIMat ABC1(SPI::block({
                        {i*U,   O,      O,      i*I},
                        {O,     i*U,    O,      O  },
                        {O,     O,      i*U,    O  },
                        {i*I,   O,      O,      O  }
                        })(),"ABC1");
            SPIMat D1(O4,"D1");
            SPIMat dA1(SPI::block({
                        {i*Ux,  O,      O,      O  },
                        {O,     i*Ux,   O,      O  },
                        {O,     O,      i*Ux,   O  },
                        {O,     O,      O,      O  }
                        })(),"dA1");
            SPIMat ABC0(SPI::block({
                        {d+Ux,  Uy,     O,      O  },
                        {O,     d+Vy,   O,      Dy },
                        {Wx,    Wy,     d,      i*beta*I},
                        {O,     Dy,    i*beta*I,O  }
                        })(),"ABC0");
            SPIMat D0(SPI::block({
                        {U,     O,      O,      I  },
                        {O,     U,      O,      O  },
                        {O,     O,      U,      O  },
                        {I,     O,      O,      O  }
                        })(),"D0");
            SPIMat dA0(SPI::block({
                        {i*beta*Wx, Uxy,        O,          O  },
                        {O,         i*beta*Wx,  O,          O  },
                        {O,         Wxy,        i*beta*Wx,  O  },
                        {O,         O,          O,          O  }
                        })(),"dA0");
            // set BCs
            //PetscInt rowBCs[] = {0,n-1,n,2*n-1,2*n,3*n-1,4*n,5*n-1,5*n,6*n-1,6*n,7*n-1}; // u,v,w at wall and freestream
            //PetscInt rowBCs[] = {0*ny,1*ny-1,1*ny,2*ny-1,2*ny,3*ny-1}; // u,v,w at wall and freestream
            //for(PetscInt rowi : rowBCs){
            //    //SPI::printf(std::to_string(rowi));
            //    ABC2.zero_row(rowi);
            //    ABC1.zero_row(rowi);
            //    dA1.zero_row(rowi);
            //    ABC0.zero_row(rowi);
            //    D0.zero_row(rowi);
            //    dA0.zero_row(rowi);

            //    ABC2(rowi,rowi,1.0);
            //    ABC1(rowi,rowi,1.0);
            //    dA1(rowi,rowi,1.0);
            //    ABC0(rowi,rowi,1.0);
            //    D0(rowi,rowi,1.0);
            //    dA0(rowi,rowi,1.0);
            //}
            // alternative way to set the BCs
            if(grid.ytype==SPI::UltraS){
                std::vector<PetscInt> rowBCs = {1*ny-2,1*ny-1,2*ny-2,2*ny-1,3*ny-2,3*ny-1}; // u,v,w at wall and freestream
                ABC2.zero_rows(rowBCs);
                ABC1.zero_rows(rowBCs);
                dA1.zero_rows(rowBCs);
                ABC0.zero_rows(rowBCs);
                D0.zero_rows(rowBCs);
                dA0.zero_rows(rowBCs);
                for(PetscInt j=0; j<ny; ++j){
                    ABC0(rowBCs[0],j,grid.T(0,j,PETSC_TRUE));       // u at the wall
                    ABC0(rowBCs[1],j,grid.T(ny-1,j,PETSC_TRUE));    // u at the freestream
                    ABC0(rowBCs[2],ny+j,grid.T(0,j,PETSC_TRUE));       // v at the wall
                    ABC0(rowBCs[3],ny+j,grid.T(ny-1,j,PETSC_TRUE));    // v at the freestream
                    ABC0(rowBCs[4],2*ny+j,grid.T(0,j,PETSC_TRUE));       // w at the wall
                    ABC0(rowBCs[5],2*ny+j,grid.T(ny-1,j,PETSC_TRUE));    // w at the freestream
                }
                // assemble
                ABC0();
                // SPIMat P = block({
                //         {grid.P,O,O,O},
                //         {O,grid.P,O,O},
                //         {O,O,grid.P,O},
                //         {O,O,O,eye(ny)},
                //         })();
                // // reorder rows for UltraSpherical
                // ABC2 = P*ABC2;
                // ABC1 = P*ABC1;
                // dA1 = P*dA1;
                // ABC0 = P*ABC0;
                // D0 = P*D0;
                // dA0 = P*dA0;
            }else{
                std::vector<PetscInt> rowBCs = {0*ny,1*ny-1,1*ny,2*ny-1,2*ny,3*ny-1}; // u,v,w at wall and freestream
                ABC2.zero_rows(rowBCs);// is eye_rows in python...
                ABC1.zero_rows(rowBCs);// is eye_rows in python...
                dA1.zero_rows(rowBCs);// is eye_rows in python...
                ABC0.eye_rows(rowBCs);
                D0.zero_rows(rowBCs);// is eye_rows in python...
                dA0.zero_rows(rowBCs);// is eye_rows in python...
            }

            // inflate for derivatives in streamwise direction
            SPIMat L0(block({
                        {ABC0,  D0  },
                        {dA0,   ABC0}})(),"L0");
            SPIMat L1(block({
                        {ABC1,  D1  },
                        {dA1,   ABC1}})(),"L1");
            SPIMat L2(block({
                        {ABC2,  D2  },
                        {dA2,   ABC2}})(),"L2");
            // inflate due to polynomial eigenvalue problem
            //if(grid.ytype==SPI::UltraS){
                //SPI::SPIVec eig_vec_tmp(grid.y.rows*8,"q");
                //std::tie(alpha,eig_vec_tmp) = SPI::polyeig({L0,L1,L2},alpha);
                //return std::make_tuple(alpha,eig_vec_tmp);
            //}
            SPIMat O8(zeros(8*ny,8*ny));
            //SPIMat I8(eye(8*ny));
            SPIMat I8(block({
                        {I,O,O,O,O,O,O,O},
                        {O,I,O,O,O,O,O,O},
                        {O,O,I,O,O,O,O,O},
                        {O,O,O,I,O,O,O,O},
                        {O,O,O,O,I,O,O,O},
                        {O,O,O,O,O,I,O,O},
                        {O,O,O,O,O,O,I,O},
                        {O,O,O,O,O,O,O,I},
                        })(),"I8");
            SPIMat L(block({
                        {O8,    I8},
                        {L0,    L1}})(),"L");
            SPIMat M(block({
                        {I8,    O8},
                        {O8,    -L2}})(),"M");
            //SPIMat L(block({
            //{O8,    I8},
            //{-L0,    -L1}}),"L");
            //SPIMat M(block({
            //{I8,    O8},
            //{O8,    L2}}),"M");
            //SPIMat L(block({
            //{-L0,    O8},
            //{O8,     I8}}),"L");
            //SPIMat M(block({
            //{L1,    L2},
            //{I8,    O8}}),"M");
            //PetscScalar a=10.5,b=20.5;
            //SPIMat L(block({
            //{-b*L0,     a*I8},
            //{-a*L0,     -a*L1+b*I8}}),"L");
            //SPIMat M(block({
            //{a*I8+b*L1, b*L2},
            //{b*I8,      a*L2}}),"M");


            //SPI::SPIVec eig_vec(grid.y.rows*16,"q");
            //if(qr.flag_init){
            std::tie(_alphas,eig_vecs) = SPI::eig_init_rights(L,M,alphas,qrs);
            //}else{
                //std::tie(alphas[0],eig_vecs[0]) = SPI::eig_right(L,M,alpha);
            //}
            //SPIMat L(block(
            //alpha = alpha; // invert
            //params.alpha = alpha;
            //SPI::printfc("α is: %g+%gi",alpha);
            return std::make_tuple(_alphas,eig_vecs);
        }
    }

}
