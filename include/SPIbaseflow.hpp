#ifndef SPIBASEFLOW_H
#define SPIBASEFLOW_H
#include <iostream>
#include <vector>
#include <petscksp.h>
#include <slepceps.h>
#include <slepcpep.h>
#include <string>
#include <tuple>
#include "SPIgrid.hpp"
#include "SPIparams.hpp"
#include "SPIMat.hpp"
#include "SPIVec.hpp"
#include "SPIprint.hpp"

namespace SPI{
    struct SPIbaseflow{
        SPIbaseflow(std::string _name="baseflow");                 // constructor with no arguments
        SPIbaseflow(
                SPIVec U,
                SPIVec V,
                SPIVec Ux,
                SPIVec Uy,
                SPIVec Uxy,
                SPIVec Vy,
                SPIVec W,
                SPIVec Wx,
                SPIVec Wy,
                SPIVec Wxy, 
                SPIVec P,
                std::string _name="baseflow");                  // constructor with baseflow values
        std::string name;                                       ///< baseflow name
        PetscErrorCode ierr;                                    ///< ierr for various routines and operations
        PetscBool flag_init=PETSC_FALSE;                        ///< flag if it has been initialized
        PetscInt print();                                       // print baseflow to screen
        SPIVec U,       // streamwise baseflow
               V,       // wall-normal baseflow
               Ux,      // streamwise baseflow derivative with respect to streamwise
               Uy,      // streamwise baseflow derivative with respect to wall-normal
               Uxy,     // streamwise baseflow mixed derivative
               Vy,      // wall-normal baseflow derivative with respect to wall-normal
               W,       // spanwise baseflow
               Wx,      // spanwise baseflow derivative with respect to streamwise
               Wy,      // spanwise baseflow derivative with respect to wall-normal
               Wxy,     // spanwise baseflow mixed derivative
               P;       //  pressure baseflow
        ~SPIbaseflow(); // destructor to delete memory
    };
    SPIbaseflow blasius(SPIparams &params, SPIgrid1D &grid);
    int _bblf( const PetscScalar input[3], PetscScalar output[3]); // Blasius boundary layer flow ODE
    SPIbaseflow channel(SPIparams &params, SPIgrid1D &grid);
}
#endif // SPIBASEFLOW_H
