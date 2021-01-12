#include "main.hpp"
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

    if(0){// set to 1 if wanting to test SPI Mat and Vec and other operations found in tests.cpp
        tests();
    }
    else{// started working on derivative operators, but it didn't quite work yet
        // get_D_Coeffs
        //SPI::SPIVec s(SPI::arange(-2,3,1.),"s");
        //get_D_Coeffs(s,d).print();
        // set_D
        PetscInt n=31;
        //PetscScalar y_max=21.;
        //SPI::SPIVec ytmp(SPI::linspace(0.,y_max,n),"y");
        //PetscScalar delta=2.0001;
        //SPI::SPIVec y(y_max*(1.+(SPI::tanh(delta*((ytmp/y_max) - 1.))/tanh(delta))),"y");
        //y.print();
        SPI::SPIVec ytmp;
        //ytmp.Init(4,"tmp");
        //ytmp.print();
        //SPI::SPIVec y(SPI::set_FD_stretched_y(y_max,n),"y");
        //SPI::SPIVec y(SPI::linspace(-1.,1.,n),"y"); // for finite difference grid
        //SPI::SPIVec y(SPI::set_Cheby_mapped_y(0.,61.,n),"yCheby");
        SPI::SPIVec y(SPI::set_Cheby_y(n),"yCheby");
        //SPI::SPIMat D(SPI::set_D(y,d,order));
        //D.print();
        //(D*(y^2)-2*y).print(); // near zero
        SPI::SPIgrid grid(y,"grid",SPI::Chebyshev);

        // channel flow Orr-Sommerfeld solution
        SPI::SPIMat U(SPI::diag(1.0-((grid.y)^2)),"U");
        SPI::SPIMat Uy(SPI::diag(-2.*grid.y),"Uy");
        SPI::SPIMat Uyy(SPI::diag(-2.*SPI::ones(grid.y.rows)),"Uyy");

        PetscScalar Re=2000.0;
        PetscScalar alpha=1.0;
        PetscScalar beta=0.0;
        PetscScalar k=alpha*alpha+beta*beta;
        PetscScalar k2=k*k;
        PetscScalar i=PETSC_i;
        SPI::SPIMat O(SPI::zeros(grid.y.rows,grid.y.rows));
        SPI::SPIMat I(SPI::eye(grid.y.rows));
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

        SPI::SPIVec eigenfunction(grid.y.rows*4,"q");
        PetscScalar eigenvalue;
        // std::tie(eigenvalue,eigenfunction) = SPI::eig(L,M,0.3-0.0001*i); // doesn't work because M is singular
        std::tie(eigenvalue,eigenfunction) = SPI::eig(M,L,1./(0.3-0.0001*i));
        std::cout<<1./eigenvalue<<std::endl;
        (1./eigenvalue*SPI::ones(1)).print();



    }

    ierr = PetscFinalize();CHKERRQ(ierr);
    ierr = SlepcFinalize();CHKERRQ(ierr);

    return 0;
}
