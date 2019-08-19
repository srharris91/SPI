#ifndef SPEPRINT_H
#define SPEPRINT_H
#include <iostream>
#include <string>
#include <petscksp.h>
// #include <errno.h>
// #include <../src/sys/fileio/mprint.h>

namespace SPE{
    PetscInt printf(std::string msg,...); // print a message to string using PetscPrintf (also adds newline at end)


}



#endif
