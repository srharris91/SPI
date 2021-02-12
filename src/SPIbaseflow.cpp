#include "SPIbaseflow.hpp"

namespace SPI{
    // constructors
    /** \brief constructor with no arguments */
    SPIbaseflow::SPIbaseflow(
            std::string _name    ///< [in] baseflow name (default baseflow)
            ){ this->name = _name; }
    /** \brief constructor with baseflow arguments */
    SPIbaseflow::SPIbaseflow(
            SPIVec U,       ///< [in] streamwise baseflow
            SPIVec V,       ///< [in] wall-normal baseflow
            SPIVec Ux,      ///< [in] streamwise baseflow derivative with respect to streamwise
            SPIVec Uy,      ///< [in] streamwise baseflow derivative with respect to wall-normal
            SPIVec Uxy,     ///< [in] streamwise baseflow mixed derivative
            SPIVec Vy,      ///< [in] wall-normal baseflow derivative with respect to wall-normal
            SPIVec W,       ///< [in] spanwise baseflow
            SPIVec Wx,      ///< [in] spanwise baseflow derivative with respect to streamwise
            SPIVec Wy,      ///< [in] spanwise baseflow derivative with respect to wall-normal
            SPIVec Wxy,         ///< [in] spanwise baseflow mixed derivative
            SPIVec P,       ///< [in] pressure baseflow
            std::string _name       ///< [in] baseflow name (default baseflow)
            ){
        this->name = _name;
        this->U = U; this->U.name = "U";
        this->V = V; this->V.name = "V";
        this->Ux = Ux; this->Ux.name = "Ux";
        this->Uy = Uy; this->Uy.name = "Uy";
        this->Uxy = Uxy; this->Uxy.name = "Uxy";
        this->Vy = Vy; this->Vy.name = "Vy";
        this->W = W; this->W.name = "W";
        this->Wx = Wx; this->Wx.name = "Wx";
        this->Wy = Wy; this->Wy.name = "Wy";
        this->Wxy = Wxy; this->Wxy.name = "Wxy";
        this->P = P; this->P.name = "P";
        this->flag_init=PETSC_TRUE;
    }
    /** \brief print baseflow values */
    PetscInt SPIbaseflow::print(){
        // print baseflow values if initialized
        PetscPrintf(PETSC_COMM_WORLD,("---------------- "+name+"---done-------\n\n").c_str());
        if(this->flag_init){
            this->U.print();
            this->V.print();
            this->Ux.print();
            this->Uy.print();
            this->Uxy.print();
            this->Vy.print();
            this->W.print();
            this->Wx.print();
            this->Wy.print();
            this->Wxy.print();
            this->P.print();
        }else{
            SPI::printf(this->name+std::string(" not initialized"));
        }
        PetscPrintf(PETSC_COMM_WORLD,("---------------- "+name+"---done-------\n\n").c_str());
        return 0;
    }

    /** \brief destructor of saved SPIVec of baseflow */
    SPIbaseflow::~SPIbaseflow(){
        if(this->flag_init){
            this->U.~SPIVec();
            this->V.~SPIVec();
            this->Ux.~SPIVec();
            this->Uy.~SPIVec();
            this->Uxy.~SPIVec();
            this->Vy.~SPIVec();
            this->W.~SPIVec();
            this->Wx.~SPIVec();
            this->Wy.~SPIVec();
            this->Wxy.~SPIVec();
            this->P.~SPIVec();
        }
    }

}
