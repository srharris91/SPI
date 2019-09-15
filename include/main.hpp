#ifndef MAIN_H
#define MAIN_H
#include <iostream>
#include <petscksp.h>
#include <tuple>
#include "SPEMat.hpp"
#include "SPEVec.hpp"
#include "SPEprint.hpp"
/** \mainpage SPE Solver
 *
 * \author Shaun Harris (<A HREF="https://srharris91.github.io/" TARGET="_top">https://srharris91.github.io/</A>)\n
 * Copyright (C) 2019\n
 * \section Info Information
 * This code is being implemented to make the PETSc Mat and Vec easier to work with and use.  It is intended for simplifying many of the MatAXPY calls and things to make them more intuitive and easier to parallelize.  It's main use is to make use of operator overloading.
 *
 * \section PETSc PETSc configuration
 * Use the following code to configure your PETSc Submodule.   This will capture all the necessary packages for MPI and C++ items. Then follow the directions generated to compile and test. Everything should work.
 * \code{.sh}
python2 './configure' '--with-scalar-type=complex' '--with-precision=double' 'with-clanguage=c++' '--download-mumps' '--download-hdf5' '--download-scalapack' '--download-parmetis' '--download-metis' '--download-ptscotch' '--with-cc=mpicc' '--with-cxx=mpicxx' '--with-fc=mpif90' '--with-debugging=0' 'COPTFLAGS='-O3 -march=native -mtune=native'' 'CXXOPTFLAGS='-O3 -march=native -mtune=native'' 'FOPTFLAGS='-O3 -march=native -mtune=native''
\endcode
 * \section SLEPc SLEPc configuration
 * use the following code to configure your SLEPc Submodule.  It should capture all of the commands when you supply the PETSC_DIR and PETSC_ARCH variables to the environment. 
 *  \code{.sh}
     python2 './configure' 
\endcode
 */

#endif // MAIN_H
