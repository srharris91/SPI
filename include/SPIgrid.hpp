#ifndef SPIGRID_H
#define SPIGRID_H
#include "SPIVec.hpp"
#include "SPIMat.hpp"

namespace SPI{
    PetscInt factorial(PetscInt n);                 // compute and return the factorial of n
    SPIVec get_D_Coeffs( SPIVec &s, PetscInt d );   // get the coefficients of the given stencil s
    SPIMat map_D(SPIMat D, SPIVec y, PetscInt d, PetscInt order=4); // map the derivative operator to the proper y grid
    SPIMat set_D(SPIVec &y, PetscInt d, PetscInt order=4, PetscBool uniform=PETSC_FALSE); // get derivative operator using finite difference stencils

}
#endif // SPIGRID_H
