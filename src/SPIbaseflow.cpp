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
    /* \brief set Blasius boundary layer flow \return SPIbaseflow at the current conditions */
    SPIbaseflow blasius(
            SPIparams &params,  ///< [in] parameters such as x and nu (freestream Uinf=1)
            SPIgrid &grid        ///< [in] grid containing wall-normal points
            ){ // if base flow is Blasius Flat-Plate
                PetscInt multiply_nypts = 8; //data.multiply_nypts_for_bblf;
                PetscScalar dy,jfloat;
                PetscScalar multiply_nypts_float=multiply_nypts;
                PetscScalar eta[multiply_nypts*grid.ny];
                PetscScalar deta[multiply_nypts*grid.ny-1];
                // TODO: fairly inefficient way to calculate on all cores, then save it in parallel
                for(int i=0; i<grid.ny; ++i){ 
                    dy = grid.y(i+1,PETSC_TRUE)-grid.y(i,PETSC_TRUE);
                    //std::cout<<"y["<<i<<"] = "<<data.y[i]<<std::endl;
                    for(int j=0; j<multiply_nypts; j++){
                        PetscInt ij = i*multiply_nypts+j;
                        jfloat=(PetscScalar)j;
                        eta[ij] = (grid.y(i,PETSC_TRUE)+dy*jfloat/multiply_nypts_float)*PetscSqrtScalar(1./(2.*params.nu*params.x)); // asume U_inf = 1
                    }
                }
                for(int i=0; i<grid.ny*multiply_nypts-1; ++i){ // set spacing of eta
                    deta[i] = eta[i+1] - eta[i];
                }
                // initialize variable
                PetscScalar **fs = new PetscScalar*[grid.ny*multiply_nypts];// ny pts to solve
                for(int i=0; i<grid.ny*multiply_nypts; ++i) fs[i] = new PetscScalar[3]; // 3 variables
                PetscScalar k1[3],k2[3],k3[3],k4[3];// RK4 intermediate variables
                PetscScalar temp[3],temp2[3];// temporary array to hold values
                // set initial conditions (ICs)
                //fs[0][0] = 0.469600;    // f''[0] = 0.469600
                //fs[0][0] = 0.33205733621519630*sqrt(2.);    // f''[0] = 0.469600
                fs[0][0] = 0.332057336215195*std::sqrt(2.);
                fs[0][1] = 0.;          // f'[0] = 0
                fs[0][2] = 0.;          // f[0] = 0
                
                // march through each eta value using _bblf function evaluation
                for(int i=0; i<grid.ny*multiply_nypts-1; ++i){ // for each eta value
                    _bblf(fs[i],temp); // calculate f(i)
                    for(int j=0; j<3; ++j){ 
                        k1[j] = deta[i]*temp[j];
                        temp2[j] = fs[i][j] + k1[j]/2.;
                    }
                    _bblf(temp2,temp); // calculate f(i+k1/2);
                    for(int j=0; j<3; ++j){ 
                        k2[j] = deta[i]*temp[j];
                        temp2[j] = fs[i][j] + k2[j]/2.;
                    }
                    _bblf(temp2,temp); // calculate f(i+k2/2);
                    for(int j=0; j<3; ++j){ 
                        k3[j] = deta[i]*temp[j];
                        temp2[j] = fs[i][j] + k3[j];
                    }
                    _bblf(temp2,temp); // calculate f(i+k3);
                    for(int j=0; j<3; ++j){ 
                        k4[j] = deta[i]*temp[j];
                    }
                    for(int j=0; j<3; ++j){ // f[i+1] = f[i] + (k1+2k2 + 2k3 + k4)/6
                        fs[i+1][j] = fs[i][j] + (k1[j] + (k2[j]*2.) + (k3[j]*2.) + k4[j])/6.;
                        //std::cout<<"fs["<<i+1<<"]["<<j<<"] = "<<fs[i+1][j]<<std::endl;
                        //std::cout<<"  k1,k2,k3,k4 = "<<k1[j]<<", "<<k2[j]<<", "<<k3[j]<<", "<<k4[j]<<std::endl;
                    }
                }
                // print eta and fp, fpp to compare to textbook values
                //for(int i=0; i<grid.ny; ++i){ 
                    //PetscInt j=0;
                    //PetscInt ij = i*multiply_nypts+j;
                    //printf("eta = %g and f = %g and f' = %g and f'' = %g",PetscRealPart(eta[ij]),PetscRealPart(fs[ij][2]),PetscRealPart(fs[ij][1]),PetscRealPart(fs[ij][0]));
                //}

                // save values
                PetscScalar *fpp = new PetscScalar[grid.ny*multiply_nypts];
                PetscScalar *fp  = new PetscScalar[grid.ny*multiply_nypts];
                PetscScalar *f   = new PetscScalar[grid.ny*multiply_nypts];
                PetscScalar *Utmp   = new PetscScalar[grid.ny*multiply_nypts];
                PetscScalar *Uxtmp  = new PetscScalar[grid.ny*multiply_nypts];
                PetscScalar *Uytmp  = new PetscScalar[grid.ny*multiply_nypts];
                PetscScalar *Vtmp   = new PetscScalar[grid.ny*multiply_nypts];
                PetscScalar *Vxtmp  = new PetscScalar[grid.ny*multiply_nypts];
                PetscScalar *Vytmp  = new PetscScalar[grid.ny*multiply_nypts];
                for (int i=0; i<grid.ny*multiply_nypts; ++i){
                    fpp[i] = fs[i][0];
                    fp [i] = fs[i][1];
                    f  [i] = fs[i][2];

                    // save U
                    Utmp[i] = fp[i];                                           // U  =f'  assuming Uinf=1
                    Uxtmp[i]= fpp[i]*(-eta[i]/(2.*params.x));               // Ux =f''(-eta/(2x))
                    Uytmp[i]= fpp[i]*PetscSqrtScalar(1./(2.*params.nu*params.x));  // Uy =f''sqrt(1./(2 nu x))
                    // save V
                    Vtmp[i] = PetscSqrtScalar(params.nu/(2.*params.x))*(eta[i]*fp[i] - f[i]); // V  = sqrt(nu/2x)(eta f' - f)
                    Vxtmp[i]= PetscSqrtScalar(params.nu/(8.*PetscPowScalar(params.x,3)))*
                        (
                         - eta[i]*fp[i] 
                         + f[i] 
                         - PetscPowScalar(eta[i],2)*fpp[i]
                        ); // Vx = sqrt(nu/8x^3)(-eta fp + f - eta^2 f'')
                    Vytmp[i]= (1./(2.*params.x))*eta[i]*fpp[i];             // Vy = (1/2x) eta f''


                }

                // Checks
                // Check divergence
                PetscScalar *divergence = new PetscScalar[grid.ny*multiply_nypts];
                for(int i=0; i<grid.ny*multiply_nypts; ++i) divergence[i] = Uxtmp[i] + Vytmp[i];
                PetscScalar sum_divergence = 0.;
                for(int i=0; i<grid.ny*multiply_nypts; ++i) sum_divergence += PetscAbsScalar(divergence[i]);
                printfc("the sum of the divergence of Blasius boundary flow is %g+%gi",sum_divergence);

                // save values for SPE in grid structure
                SPIVec U(grid.ny), Uy(grid.ny), Ux(grid.ny);
                SPIVec V(grid.ny), Vy(grid.ny), Vx(grid.ny);
                SPIVec O(zeros(grid.ny));;
                for(int i=0; i<grid.ny; i++){
                    U(i,Utmp[i*multiply_nypts]);
                    Uy(i,Uytmp[i*multiply_nypts]);
                    Ux(i,Uxtmp[i*multiply_nypts]);
                    V(i,Vtmp[i*multiply_nypts]);
                    Vy(i,Vytmp[i*multiply_nypts]);
                    Vx(i,Vxtmp[i*multiply_nypts]);
                    // set freestream velocities
                }
                U();
                Ux();
                Uy();
                V();
                Vx();
                Vy();

                //SPIbaseflow baseflow(U, V, Ux, Uy, grid.Dy*Ux, Vy, O, O, O, O, O); // doesn't project correctly
                SPIbaseflow baseflow(U, V, Ux, grid.Dy*U, grid.Dy*Ux, grid.Dy*V, O, O, O, O, O); // bad for UltraS grid
                if(grid.ytype==SPI::UltraS){ // fix for UltraS grid
                    SPIgrid gridCheby(grid.y,"gridCheby",SPI::Chebyshev);
                    baseflow.Uy = gridCheby.Dy*U;
                    baseflow.Vy = gridCheby.Dy*V;
                }

                // clear memory
                for(int i=0; i<grid.ny*multiply_nypts; ++i) delete[] fs[i];
                delete[] fs;
                delete[] fpp;
                delete[] fp;
                delete[] f;
                delete[] Utmp;
                delete[] Uxtmp;
                delete[] Uytmp;
                delete[] Vtmp;
                delete[] Vxtmp;
                delete[] Vytmp;

                return baseflow;
    }
    int _bblf(
        const PetscScalar input[3], 
        PetscScalar output[3]){
      output[0] = -input[2]*input[0]; // f''= \int -f*f'' deta
      output[1] = input[0];           // f' = \int f'' deta
      output[2] = input[1];           // f  = \int f' deta
      return 0;
    }
    
    /* \brief calculate baseflow for Plane Poiseuille flow \return SPIbaseflow of the Plane Poiseuille flow */
    SPIbaseflow channel(
            SPIparams &params,  ///< [in] parameters such as x and nu (freestream Uinf=1)
            SPIgrid &grid        ///< [in] grid containing wall-normal points
            ){ 
        SPI::SPIVec U((1.0-((grid.y)^2)),"U");
        SPI::SPIVec Uy((-2.*grid.y),"Uy");
        SPI::SPIVec o(U*0.0,"o"); // zero vector
        SPI::SPIbaseflow channel_flow(U,o,o,Uy,o,o,o,o,o,o,o);
        return channel_flow;
    }

}
