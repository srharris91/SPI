![SPI](spi2_black_logo.png "SLEPc PETSc Interface")
# SPI (SLEPc PETSc Interface)

Library to work with SLEPc and PETSc Mat and Vec in C++

This uses many overloaded operators and objects to aid in the rapid development of solvers

# PETSc configuration
Use the following code to configure your PETSc Submodule.   This will capture all the necessary packages for MPI and C++ items. Then follow the directions generated to compile and test. Everything should work.
```bash
python2 './configure' '--with-scalar-type=complex' '--with-precision=double' 'with-clanguage=c++' '--download-mumps' '--download-hdf5' '--download-scalapack' '--download-parmetis' '--download-metis' '--download-ptscotch' '--with-cc=mpicc' '--with-cxx=mpicxx' '--with-fc=mpif90' '--with-debugging=0' 'COPTFLAGS='-O3 -march=native -mtune=native'' 'CXXOPTFLAGS='-O3 -march=native -mtune=native'' 'FOPTFLAGS='-O3 -march=native -mtune=native''
```

# SLEPc configuration

Use the following code to configure your SLEPc Submodule.  It should capture all of the commands when you supply the PETSC_DIR and PETSC_ARCH variables to the environment. 
```bash
     python2 './configure' 
```
