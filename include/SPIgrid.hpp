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
    SPIVec set_Fourier_t(PetscScalar T, PetscInt ny); // create Fourier collocated grid on [0,T]
    SPIMat set_D_Fourier(SPIVec t, PetscInt d=1); // create Fourier collocated grid derivative operator acting on t
    std::tuple<SPIMat,SPIMat> set_D_UltraS(SPIVec &x, PetscInt d=1); // set a UltraSpherical operators S_(d-1) and D_d
    std::tuple<SPIMat,SPIMat> set_T_That(PetscInt n); // set a Chebyshev operators T and That
    /** \brief enumeration of grid types */
    enum gridtype {
        FD,         ///< finite difference grid
        FT,         ///< Fourier transform collocated grid
        Chebyshev,  ///< Chebyshev collocated grid
        UltraS      ///< UltraSpherical grid and derivatives
    };
    /** 
     * \brief Class to contain various grid parameters
     */
    struct SPIgrid1D {
        SPIgrid1D();            // constructor with no arguments
        SPIgrid1D(SPIVec &y, std::string name="SPIgrid1D", gridtype _gridtype=FD);            // constructor with set_grid arguments (set default values and derivatives)
        ~SPIgrid1D();                    // destructor
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
        SPIMat S0,      ///< UltraSpherical helper matrix S_0 takes chebyshev coefficients and outputs C^(1) coefficients
               S1,      ///< UltraSpherical helper matrices S_1 takes C^(1) coefficients and outputs C^(2) coefficients
               S1S0That;///< UltraSpherical helper matrix S1*S0*That for baseflow
        SPIMat S0invS1inv;  ///< [in] inverse of S0^-1 * S1^-1
        SPIMat P;       ///< row permutation matrix for UltraSpherical operators to shift rows from bottom to top to reduce LU factorization pivoting
        SPIMat T,       ///< Chebyshev operator taking it from Chebyshev coefficients to physical space
               That;    ///< Chebyshev operator taking it from physical space to Chebyshev coefficients
        SPIMat O,       ///< zero matrix same size as derivative operators
               I;       ///< identity matrix same size as derivative operators
        // flags
        PetscBool flag_set_grid=PETSC_FALSE,        ///< flag if set_grid has been executed
                  flag_set_derivatives=PETSC_FALSE, ///< flag if set_derivatives has been executed
                  flag_set_operators=PETSC_FALSE; ///< flag if set_operators has been executed

    };
    struct SPIgrid2D{
        SPIgrid2D(SPIVec &y, SPIVec &t, std::string name="SPIgrid2D", gridtype y_gridtype=FD, gridtype t_gridtype=FT);            // constructor with set_grid arguments (set default values and derivatives)
        ~SPIgrid2D();                    // destructor
        PetscInt ny,nt;                // number of points in wall-normal coordinate
        void print();               // print all members of the class
        void set_grid( SPIVec &y, SPIVec &t ); // function to save grid to internal grid 
        void set_derivatives(PetscInt order=4);     // function to create derivatives on internal grid
        void set_operators();     // function to create zero and identity operators on internal grid
        // vars
        std::string name;   ///< name of grid
        // grid
        SPIVec y,       ///< grid for wall-normal dimension
               t;       ///< grid for time dimension
        SPIMat T,       ///< transform from chebyshev to physical
               That,    ///< transform from physical to chebyshev
               S1S0That,///< UltraSpherical helper matrix S1*S0*That for baseflow
               S0invS1inv;  ///< inverse of S0^-1 * S1^-1
        gridtype ytype,     ///< type of grid for wall-normal dimension
                 ttype;  ///< type of grid for time dimension

        // derivatives
        // 2D derivatives
        //SPIMat Dy2D,    ///< 1st derivative operator with respect to y of size nyxny
               //Dyy2D,   ///< 2nd derivative operator with respect to y of size nyxny
               //Dt2D;    ///< 1st derivative operator with respect to t of size ntxnt
        SPIgrid1D grid1Dy,grid1Dt;
        // 3D derivatives
        SPIMat Dy,      ///< 1st derivative operator with respect to y of size ny*nt x ny*nt
               Dyy,     ///< 2nd derivative operator with respect to y of size ny*nt x ny*nt
               Dt;      ///< 1st derivative operator with respect to t of size ny*nt x ny*nt
        // other helper operators
        SPIMat O,       ///< zero matrix same size as derivative operators of size ny*nt x ny*nt
               I,       ///< identity matrix same size as derivative operators of size ny*nt x ny*nt
               avgt;    ///< average in time operator
        // flags
        PetscBool flag_set_grid=PETSC_FALSE,        ///< flag if set_grid has been executed
                  flag_set_derivatives=PETSC_FALSE, ///< flag if set_derivatives has been executed
                  flag_set_operators=PETSC_FALSE; ///< flag if set_operators has been executed

    };
    SPIVec SPIVec1Dto2D(SPIgrid2D &grid2D, SPIVec &u); // function to inflate a vector u from a 2D grid to a 3D grid
    PetscScalar integrate(const SPIVec &a, SPIgrid1D &grid); // integrate a vector of Chebyshev Coefficients on physical grid
    PetscScalar integrate(const SPIVec &a, SPIgrid2D &grid); // integrate a vector of Chebyshev Coefficients on physical grid
    SPIVec proj(SPIVec &u, SPIVec &v, SPIgrid1D &grid);   // project using inner product for Gram-Schmidt process
    SPIVec proj(SPIVec &u, SPIVec &v, SPIgrid2D &grid);   // project using inner product for Gram-Schmidt process
    std::vector<SPIVec> orthogonalize(std::vector<SPIVec> &x,SPIgrid1D &grid); // create orthonormal basis from array of vectors 
    std::vector<SPIVec> orthogonalize(std::vector<SPIVec> &x,SPIgrid2D &grid); // create orthonormal basis from array of vectors 
    SPIMat interp1D_Mat(const SPIgrid1D &grid1, const SPIgrid1D &grid2); // create an interpolation routine to interpolate grid1.y -> grid2.y (i.e. create out matrix such that u(grid2.y) = out*u(grid1.y)

}
#endif // SPIGRID_H
