#ifndef SPILST_H
#define SPILST_H
#include <iostream>
#include <vector>
#include <petscksp.h>
#include <slepceps.h>
#include <slepcpep.h>
#include <string>
#include <tuple>
#include "SPIMat.hpp"
#include "SPIVec.hpp"
#include "SPIbaseflow.hpp"
#include "SPIgrid.hpp"
#include "SPIparams.hpp"

namespace SPI{
    std::tuple<PetscScalar, SPIVec> LST_temporal(SPIparams &params, SPIgrid &grid, SPIbaseflow &baseflow,SPIVec q=SPIVec()); // temporal LST solution
    std::tuple<PetscScalar, SPIVec> LST_spatial(SPIparams &params, SPIgrid &grid, SPIbaseflow &baseflow, SPIVec q=SPIVec()); // spatial LST solution
}
#endif // SPILST_H
