# PETSC_LIB
Library to work with PETSc Mat and Vec in C++

compile petsc with the following 

```bash
python2 './configure' '--with-scalar-type=complex' '--with-precision=double' 'with-clanguage=c++' '--download-mumps' '--download-scalapack' '--download-parmetis' '--download-metis' '--download-ptscotch' '--with-cc=mpicc' '--with-cxx=mpicxx' '--with-fc=mpif90' '--with-debugging=0' 'COPTFLAGS='-O3 -march=native -mtune=native'' 'CXXOPTFLAGS='-O3 -march=native -mtune=native'' 'FOPTFLAGS='-O3 -march=native -mtune=native''
``` 

and everything should work just fine
