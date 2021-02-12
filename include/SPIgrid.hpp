#ifndef SPIGRID_H
#define SPIGRID_H
#include "SPIVec.hpp"
#include "SPIMat.hpp"
#include "SPIprint.hpp"
#include <string>

namespace SPI{
    PetscInt factorial(PetscInt n);                 // compute and return the factorial of n
    SPIVec get_D_Coeffs( SPIVec &s, PetscInt d );   // get the coefficients of the given stencil s
    SPIMat map_D(SPIMat D, SPIVec y, PetscInt d, PetscInt order=4); // map the derivative operator to the proper y grid
    SPIMat set_D(SPIVec &y, PetscInt d, PetscInt order=4, PetscBool uniform=PETSC_FALSE); // get derivative operator using finite difference stencils
    SPIVec set_FD_stretched_y(PetscScalar y_max,PetscInt ny,PetscScalar delta=2.0001); // set stretched grid from [0,y_max] using tanh stretching
    SPIMat set_D_Chebyshev(SPIVec &x, PetscInt d=1, PetscBool need_map=PETSC_FALSE); // set a chebyshev operator acting on the collocated grid
    SPIMat map_D_Chebyshev(SPIVec &x, PetscInt d=1); // map the chebyshev operator to the proper x grid
    SPIVec set_Cheby_stretched_y(PetscScalar y_max, PetscInt ny, PetscScalar yi=10.); // create chebyshev stretched grid from [0, y_max]
    SPIVec set_Cheby_mapped_y(PetscScalar a, PetscScalar b, PetscInt ny);   // create a mapped Chebyshev grid on domain from [a,b]
    SPIVec set_Cheby_y(PetscInt ny); // create Chebyshev collocated grid on [-1,1]
    /** \brief enumeration of grid types */
    enum gridtype {
        FD,         ///< finite difference grid
        Chebyshev   ///< Chebyshev collocated grid
    };
    /** 
     * \brief Class to contain various grid parameters
     */
    struct SPIgrid {
        SPIgrid(SPIVec &y, std::string name="SPIgrid", gridtype _gridtype=FD);            // constructor with set_grid arguments (set default values and derivatives)
        ~SPIgrid();                    // destructor
        PetscInt ny;                // number of points in wall-normal coordinate
        void print();               // print all members of the class
        void set_grid( SPIVec &y ); // function to save grid to internal grid 
        void set_derivatives(PetscInt order=4);     // function to create derivatives on internal grid
        void set_operators();     // function to create zero and identity operators on internal grid
        // vars
        std::string name;   ///< name of grid
        // grid
        SPIVec y;       ///< grid
        gridtype ytype;  ///< type of grid

        // derivatives
        SPIMat Dy,      ///< 1st derivative operator with respect to y
               Dyy;     ///< 2nd derivative operator with respect to y
        SPIMat O,       ///< zero matrix same size as derivative operators
               I;       ///< identity matrix same size as derivative operators
        // flags
        PetscBool flag_set_grid=PETSC_FALSE,        ///< flag if set_grid has been executed
                  flag_set_derivatives=PETSC_FALSE, ///< flag if set_derivatives has been executed
                  flag_set_operators=PETSC_FALSE; ///< flag if set_operators has been executed

    };

}
#endif // SPIGRID_H
