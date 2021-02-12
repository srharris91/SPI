#include "SPIparams.hpp"

namespace SPI{

    /** \brief constructor with no arguments */
    SPIparams::SPIparams(
            std::string _name ///< [in] name of parameters (default parameters)
            ){
        this->name = _name;
    }
    /** \brief print all variables in SPIparams */
    PetscInt SPIparams::print(){
        PetscPrintf(PETSC_COMM_WORLD,("\n---------------- "+name+"---start------\n").c_str());
        SPI::printfc("Re      = %g+%gi",Re);
        SPI::printfc("β       = %g+%gi",beta);
        SPI::printfc("α       = %g+%gi",alpha);
        SPI::printfc("ω       = %g+%gi",omega);
        SPI::printfc("x_start = %g+%gi",x_start);
        SPI::printfc("x       = %g+%gi",x);
        SPI::printfc("x_prev  = %g+%gi",x_prev);
        SPI::printfc("h       = %g+%gi",h);
        PetscPrintf(PETSC_COMM_WORLD,("---------------- "+name+"---done-------\n\n").c_str());
        return 0;
    }
    

}
