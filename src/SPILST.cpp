#include "SPILST.hpp"

namespace SPI{
    /** \brief solve the local stability theory problem for the linearized Navier-Stokes equations using parallel baseflow with alpha being pure real, and omega the eigenvalue \return tuple of eigenvalue and eigenvector closest to the target value e.g. std::tie(omega,eig_vector) = LST_temporal(params,grid,baseflow).  Will solve for closest eigenvalue to params.omega */
    std::tuple<PetscScalar, SPIVec> LST_temporal(
            SPIparams &params,          ///< [inout] contains parameters including Re, alpha, and omega.  Will overwrite omega with true omega value once solved
            SPIgrid &grid,              ///< [in] grid class containing the grid location and respective derivatives
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
        const SPIMat U = SPI::diag(baseflow.U);
        const SPIMat Uy = SPI::diag(baseflow.Uy);
        SPI::SPIMat d((i*alpha*Re*U)+(k2*I)-grid.Dyy,"d");
        SPI::SPIMat L=SPI::block({
                    {d,         Re*Uy,      O,          i*Re*alpha*I}, // u-mom
                    {O,         d,          O,          Re*grid.Dy  },  // v-mom
                    {O,         O,          d,          i*Re*beta*I },  // w-mom
                    {i*alpha*I, grid.Dy,    i*beta*I,   O           }   // continuity
                        });//,"L");
        SPI::SPIMat M=SPI::block({
                    {i*Re*I,    O,          O,          O}, // u-mom
                    {O,         i*Re*I,     O,          O}, // v-mom
                    {O,         O,          i*Re*I,     O}, // w-mom
                    {O,         O,          O,          O}  // continuity
                        });//,"M");
        // set BCs
        std::vector<PetscInt> rowBCs = {0,n-1,n,2*n-1,2*n,3*n-1}; // u,v,w at wall and freestream
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
        L.eye_rows(rowBCs);
        M.eye_rows(rowBCs);
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
        }else{
            //std::tie(omega,eigl_vec,eig_vec) = SPI::eig(M,L,1./params.omega);
            std::tie(omega,eigl_vec,eig_vec) = SPI::eig(L,M,params.omega);
            omega = 1./omega; // invert
        }
        omega = 1./omega; // invert
        params.omega = omega;
        //SPI::printfc("ω is: %g+%gi",omega);
        return std::make_tuple(omega,eig_vec);
    }
    /** \brief solve the local stability theory problem for the linearized Navier-Stokes equations using parallel baseflow with omega being pure real, and alpha the eigenvalue \return tuple of eigenvalue and eigenvector closest to the target value e.g. std::tie(omega,eig_vector) = LST_spatial(params,grid,baseflow).  Will solve for closest eigenvalue to params.omega */
    std::tuple<PetscScalar, SPIVec> LST_spatial(
            SPIparams &params,          ///< [inout] contains parameters including Re, alpha, and omega.  Will overwrite omega with true omega value once solved
            SPIgrid &grid,              ///< [in] grid class containing the grid location and respective derivatives
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
            PetscInt rowBCs[] = {0,n-1,n,2*n-1,2*n,3*n-1}; // u,v,w at wall and freestream
            for(PetscInt rowi : rowBCs){
                //SPI::printf(std::to_string(rowi));
                L0.eye_row(rowi);
                L1.zero_row(rowi);
                L2.zero_row(rowi);
                //L0(rowi,rowi,1.0);
                //L2(rowi,rowi,60.0);
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
            alpha = alpha; // invert
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
    std::tuple<PetscScalar, PetscScalar, SPIVec, SPIVec> LSTNP_spatial(
            SPIparams &params,          ///< [inout] contains parameters including Re, alpha, and omega.  Will overwrite omega with true omega value once solved
            SPIgrid &grid,              ///< [in] grid class containing the grid location and respective derivatives
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
        SPIMat &Dy = grid.Dy;
        SPIMat &Dyy = grid.Dyy;
        SPIMat d(V*Dy - Reinv*Dyy + i*beta*W + (-i*omega + beta*beta*Reinv)*I,"d");
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
                        {O,         Wxy,        i*beta*Wx,  O },
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
            std::vector<PetscInt> rowBCs = {0*ny,1*ny-1,1*ny,2*ny-1,2*ny,3*ny-1}; // u,v,w at wall and freestream
            ABC2.eye_rows(rowBCs);
            ABC1.eye_rows(rowBCs);
            dA1.eye_rows(rowBCs);
            ABC0.eye_rows(rowBCs);
            D0.eye_rows(rowBCs);
            dA0.eye_rows(rowBCs);

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
            SPIMat I8(eye(8*ny));
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


            SPI::SPIVec eig_vec(grid.y.rows*8,"q");
            SPI::SPIVec eigl_vec(grid.y.rows*8,"q");
            if(ql.flag_init){
                std::tie(alpha,eigl_vec,eig_vec) = SPI::eig_init(L,M,alpha,ql,qr);
            }else{
                std::tie(alpha,eigl_vec,eig_vec) = SPI::eig(L,M,alpha);
            }
            //SPIMat L(block({
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
                        {O4,        O4}
                        })(),"dLdomega 8nx8n");
            // inflate for polynomial eigenvalue problem
            SPIMat dLdomega(block({
                        {O8,        O8},
                        {dLdomega8, O8}
                        })(),"dLdomega 16nx16n");
            PetscScalar cg = ((M*eig_vec).dot(eigl_vec)) / ((dLdomega*eig_vec).dot(eigl_vec));
            return std::make_tuple(alpha,cg,eigl_vec,eig_vec);
        }
    }

}
