#ifndef SPIPARAMS_H
#define SPIPARAMS_H
#include <petscksp.h>
#include <string>
#include "SPIprint.hpp"
namespace SPI{

    struct SPIparams{
        SPIparams(std::string _name="parameters"); // constructor with no arguments
        PetscInt print();// print all parameters
        std::string name;   ///< name of parameter class
        PetscScalar Re,     ///< Reynolds number
                    beta,   ///< beta spanwise wavenumber
                    alpha,   ///< alpha streamwise wavenumber
                    omega,  ///< omega, temporal frequency (rad/s)
                    x_start,///< streamwise starting location
                    x,      ///< current streamwise position
                    x_prev, ///< previous streamwise position
                    h,      ///< streamwise step size h=(x-x_prev)
                    nu;     ///< kinematic viscosity (typically 1/Re)
                    
    };

}

#endif
