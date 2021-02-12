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
        SPI::SPIMat d(i*alpha*Re*U+k2*I-grid.Dyy,"d");
        SPI::SPIMat L(SPI::block({
                    {d,         Re*Uy,      O,          i*Re*alpha*I},
                    {O,         d,          O,          Re*grid.Dy},
                    {O,         O,          d,          i*Re*beta*I},
                    {i*alpha*I, grid.Dy,    i*beta*I,   O} 
                        }),"L");
        SPI::SPIMat M(SPI::block({
                    {i*Re*I,    O,          O,          O},
                    {O,         i*Re*I,     O,          O},
                    {O,         O,          i*Re*I,     O},
                    {O,         O,          O,          O} 
                        }),"M");
        // set BCs
        PetscInt rowBCs[] = {0,n-1,n,2*n-1,2*n,3*n-1}; // u,v,w at wall and freestream
        for(PetscInt rowi : rowBCs){
            //SPI::printf(std::to_string(rowi));
            L.zero_row(rowi);
            M.zero_row(rowi);
            L(rowi,rowi,1.0);
            M(rowi,rowi,60.0);
        }
        SPI::SPIVec eig_vec(grid.y.rows*4,"q");
        SPI::SPIVec eigl_vec(grid.y.rows*4,"q");
        //PetscScalar omega;
        // std::tie(eigenvalue,eigenfunction) = SPI::eig(L,M,0.3-0.0001*i); // doesn't work because M is singular
        if(q.flag_init){
            std::tie(omega,eigl_vec,eig_vec) = SPI::eig_init(M,L,1./(0.3-0.0001*i),q.conj(),q);
        }else{
            std::tie(omega,eigl_vec,eig_vec) = SPI::eig(M,L,1./(0.3-0.0001*i));
        }
        omega = 1./omega; // invert
        params.omega = omega;
        //SPI::printfc("ω is: %g+%gi",omega);
        return std::make_tuple(omega,eig_vec);
    }
    /** \brief solve the local stability theory problem for the linearized Navier-Stokes equations using parallel baseflow with omega being pure real, and alpha the eigenvalue \return tuple of eigenvalue and eigenvector closest to the target value e.g. std::tie(omega,eig_vector) = LST_temporal(params,grid,baseflow).  Will solve for closest eigenvalue to params.omega */
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
        SPI::SPIMat d(Dyy + i*Re*omega*I - beta*beta*I,"d");
        //SPI::SPIMat d("d");
        //d = Dyy + i*Re*omega*I - beta*beta*I;
        if(1){
            SPI::SPIMat L0(SPI::block({
                        {d,         -Re*Uy,     O,          O},
                        {O,         d,          O,          -Re*Dy},
                        {O,         O,          d,          -i*Re*beta*I},
                        {O,         Dy,         i*beta*I,   O} 
                        }),"L0");
            SPI::SPIMat L1(SPI::block({
                        {-i*Re*U,   O,          O,          -i*Re*I},
                        {O,         -i*Re*U,    O,          O},
                        {O,         O,          -i*Re*U,    O},
                        {i*I,       O,          O,          O} 
                        }),"L1");
            SPI::SPIMat L2(SPI::block({
                        {-I,        O,          O,          O},
                        {O,         -I,         O,          O},
                        {O,         O,          -I,         O},
                        {O,         O,          O,          O} 
                        }),"L2");
            // set BCs
            PetscInt rowBCs[] = {0,n-1,n,2*n-1,2*n,3*n-1}; // u,v,w at wall and freestream
            for(PetscInt rowi : rowBCs){
                //SPI::printf(std::to_string(rowi));
                L0.zero_row(rowi);
                L1.zero_row(rowi);
                L2.zero_row(rowi);
                L0(rowi,rowi,1.0);
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

}
