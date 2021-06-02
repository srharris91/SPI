#include "SPImain.hpp"
// include tests.hpp if running the tests function that tests all the functionality of the SPI classes
#include "tests.hpp"

static char help[] = "SPI class to wrap PETSc Mat variables \n\n";


int main(int argc, char **args){
    PetscErrorCode ierr;
    std::cout<<"-----------------Petsc Slepc Init Starting---------------"<<std::endl;
    ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;
    ierr = SlepcInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;
    SPI::printf("\n\n\n");
    SPI::printf("-----------------Petsc Slepc Init Complete---------------");

    if(1){// set to 1 if wanting to test SPI Mat and Vec and other operations found in tests.cpp
        tests();
    }
    else{// started working on derivative operators, but it didn't quite work yet
        // get_D_Coeffs
        //SPI::SPIVec s(SPI::arange(-2,3,1.),"s");
        //get_D_Coeffs(s,d).print();
        // set_D
        PetscInt n=64;
        //PetscScalar y_max=21.;
        //SPI::SPIVec ytmp(SPI::linspace(0.,y_max,n),"y");
        //PetscScalar delta=2.0001;
        //SPI::SPIVec y(y_max*(1.+(SPI::tanh(delta*((ytmp/y_max) - 1.))/tanh(delta))),"y");
        //y.print();
        //SPI::SPIVec ytmp;
        //ytmp.Init(4,"tmp");
        //ytmp.print();
        //SPI::SPIVec y(SPI::set_FD_stretched_y(y_max,n),"y");
        //SPI::SPIVec y(SPI::linspace(-1.,1.,n),"y"); // for finite difference grid
        //SPI::SPIVec y(SPI::set_Cheby_mapped_y(0.,61.,n),"yCheby");
        //SPI::SPIVec y(SPI::set_Cheby_mapped_y(-1,1,n),"yCheby");
        SPI::SPIVec y(SPI::set_Cheby_y(n),"yCheby");
        //SPI::SPIMat D(SPI::set_D(y,d,order));
        //D.print();
        //(D*(y^2)-2*y).print(); // near zero
        SPI::SPIgrid1D grid(y,"grid",SPI::Chebyshev);
        //grid.Dy.print();

        // channel flow Orr-Sommerfeld solution
        SPI::SPIMat U(SPI::diag(1.0-((grid.y)^2)),"U");
        SPI::SPIMat Uy(SPI::diag(-2.*grid.y),"Uy");
        //SPI::SPIMat Uyy(SPI::diag(-2.*SPI::ones(grid.y.rows)),"Uyy");

        //PetscScalar Re=2000.0;
        //PetscScalar omega=0.3;
        //PetscScalar beta=0.0;
        SPI::SPIparams params("channel parameter");
        params.Re = 2000.0;
        params.omega = 0.3;
        params.alpha = 0.97875+0.044394*PETSC_i;
        params.beta = 0.0;

        SPI::SPIVec eigenfunction(grid.y.rows*6,"q");
        PetscScalar eigenvalue;
        // std::tie(eigenvalue,eigenfunction) = SPI::eig(L,M,0.3-0.0001*i); // doesn't work because M is singular
        //std::tie(eigenvalue,eigenfunction) = SPI::eig(M,L,1./(0.3-0.0001*i));
        //SPI::printfc("eigenvalue is: %g+%gi",1./eigenvalue);
        //(1./eigenvalue*SPI::ones(1)).print();

        eigenfunction.name = "eigenfunction";
        //SPI::load(eigenfunction,"eigenfunction_saved.hdf5");

        //std::tie(eigenvalue,eigenfunction) = SPI::eig_init(M,L,1.0/(0.3121-0.0197987*i),eigenfunction);
        //std::tie(eigenvalue,eigenfunction) = SPI::eig(M,L,1.0/(0.3121-0.0197987*i));
        //std::cout<<1./eigenvalue<<std::endl;
        //SPI::printfc("eigenvalue is: %.10f+%.10fi",1./eigenvalue);

        SPI::SPIVec o(U.diag()*0.0,"o");
        SPI::SPIbaseflow channel(U.diag(),o,o,Uy.diag(),o,o,o,o,o,o,o);
        //channel.print();

        //params.print();

        //std::tie(eigenvalue,eigenfunction) = SPI::LST_temporal(params,grid,channel);
        //SPI::printfc("eigenvalue is: %.10f+%.10fi",eigenvalue);
        //SPI::printfc("eigenvalue is: %.10f+%.10fi",params.omega);
        //params.omega += 0.00001;

        SPI::printfc("alpha is: %.10f+%.10fi",params.alpha);
        std::tie(eigenvalue,eigenfunction) = SPI::LST_spatial(params,grid,channel);
        SPI::printfc("eigenvalue is: %.10f+%.10fi",eigenvalue);
        SPI::printfc("eigenvalue is: %.10f+%.10fi",params.alpha);
        std::tie(eigenvalue,eigenfunction) = SPI::LST_spatial(params,grid,channel,eigenfunction);
        SPI::printfc(" eigenvalue is: %.10f+%.10fi",params.alpha);
        params.alpha = 0.34312+0.049677*PETSC_i;
        std::tie(eigenvalue,eigenfunction) = SPI::LST_spatial(params,grid,channel,eigenfunction);
        SPI::printfc(" eigenvalue is: %.10f+%.10fi",params.alpha);
        params.alpha = 0.61+0.1*PETSC_i;
        std::tie(eigenvalue,eigenfunction) = SPI::LST_spatial(params,grid,channel,eigenfunction);
        SPI::printfc(" eigenvalue is: %.10f+%.10fi",params.alpha);
        //SPI::save(eigenfunction,"eigenfunction_saved.hdf5");
        //SPI::draw(eigenfunction);

    }

    ierr = PetscFinalize();CHKERRQ(ierr);
    ierr = SlepcFinalize();CHKERRQ(ierr);

    return 0;
}
