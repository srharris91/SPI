#include "SPIprint.hpp"

namespace SPI{
    /** print a message to string using PetscPrintf (also adds a newline at end) (note: only prints on rank 0 processor) \return 0 if successful */
    PetscInt printf(
            std::string msg, ///< [in] message to print with formatting such as \%g
            ...             ///< [in] scalar or double or int to match formatting string to be output on processor with rank 0
            ){
        MPI_Comm comm=PETSC_COMM_WORLD;
        msg+="\n";
        const char *format = msg.c_str();
        // the rest of this is copied from PetscPrintf routine
        PetscErrorCode ierr;
        PetscMPIInt    rank;

        PetscFunctionBeginUser;
        ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
        if (!rank) {
            va_list Argp;
            //va_start(Argp,format,1);
            va_start(Argp,msg);
            va_arg(Argp,const char*);
            ierr = (*PetscVFPrintf)(PETSC_STDOUT,format,Argp);CHKERRQ(ierr);
            va_end(Argp);
        }
        PetscFunctionReturn(0);
    }
    /** print a message to string using PetscPrintf (also adds a newline at end) (note: only prints on rank 0 processor) with PetscScalars as input and two formats per argument \return 0 if successful */
    PetscInt printfc(
            std::string msg, ///< [in] message to print with formatting such as \%g
            PetscScalar val ///< [in] scalar or double or int to match formatting string to be output on processor with rank 0
            ){
        return printf(msg,PetscRealPart(val),PetscImaginaryPart(val));
    }
}
