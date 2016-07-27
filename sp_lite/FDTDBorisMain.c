/*
 * BorisYeeMain.c
 *
 *  Created on: 2016年6月15日
 *      Author: salmon
 */
#include <assert.h>
#include <stdio.h>
#include <math.h>

#include "sp_lite_def.h"
#include "spParallel.h"

#include "spMesh.h"
#include "spField.h"
#include "spParticle.h"
#include "spMisc.h"

#include "FDTDBoris.h"


int main(int argc, char **argv)
{
    SP_CHECK_RETURN(spParallelInitialize(argc, argv));

    ShowSimPlaLogo();

    char out_file[2048] = "untitled.h5";

    int num_of_steps = argc < 2 ? 100 : atoi(argv[1]);
    int check_point = argc < 3 ? 10 : atoi(argv[2]);
    int PIC = 256;
    Real n0 = 1.0e18;
    Real T0 = 0.026;
    size_type dims[3] = {0x100, 0x100, 0x1};
    size_type gw[3] = {0x2, 0x2, 0x2};
    Real lower[3] = {0, 0, 0};
    Real upper[3] = {1, 1, 1};

    Real dt = nan("");

    Real amp[3] = {0.0, 0.0, 1.0};
    Real k[3] = {TWOPI / (upper[0] - lower[0]), TWOPI / (upper[0] - lower[0]), 0};

    /*****************************************************************************************************************/

    spIOStream *os = NULL;
    SP_CHECK_RETURN(spIOStreamCreate(&os));
    SP_CHECK_RETURN(spIOStreamOpen(os, out_file));

    /*****************************************************************************************************************/

    spMesh *mesh;

    SP_CHECK_RETURN(spMeshCreate(&mesh));
    SP_CHECK_RETURN(spMeshSetDims(mesh, dims));
    SP_CHECK_RETURN(spMeshSetGhostWidth(mesh, gw));
    SP_CHECK_RETURN(spMeshSetBox(mesh, lower, upper));
    SP_CHECK_RETURN(spMeshDeploy(mesh));

    if (isnan(dt)) { dt = spMeshCFLDt(mesh, 299792458.0/* speed_of_light*/); }
    /*****************************************************************************************************************/

    spField *fE = NULL;
    spField *fB = NULL;
    spField *fRho = NULL;
    spField *fJ = NULL;

    SP_CHECK_RETURN(spFieldCreate(&fE, mesh, EDGE, SP_TYPE_Real));
    SP_CHECK_RETURN(spFieldCreate(&fB, mesh, FACE, SP_TYPE_Real));
    SP_CHECK_RETURN(spFieldCreate(&fJ, mesh, EDGE, SP_TYPE_Real));
    SP_CHECK_RETURN(spFieldCreate(&fRho, mesh, VERTEX, SP_TYPE_Real));


    SP_CHECK_RETURN(spFieldClear(fE));
    SP_CHECK_RETURN(spFieldClear(fB));
    SP_CHECK_RETURN(spFieldClear(fJ));
    SP_CHECK_RETURN(spFieldClear(fRho));


    SP_CHECK_RETURN(spFieldAssignValueSin(fE, k, amp));
    /*****************************************************************************************************************/

    spParticle *sp = NULL;
    Real SI_elementary_charge = 1.60217656e-19;
    Real SI_electron_mass = 9.10938291e-31;
    Real SI_proton_mass = 1.672621777e-27;
    SP_CHECK_RETURN(spBorisYeeParticleCreate(&sp, mesh));
    SP_CHECK_RETURN(spParticleSetMass(&sp, SI_electron_mass));
    SP_CHECK_RETURN(spParticleSetCharge(&sp, SI_elementary_charge));
    SP_CHECK_RETURN(spBorisYeeParticleInitialize(sp, PIC, n0, T0));

    /*****************************************************************************************************************/

    SP_CHECK_RETURN(spIOStreamOpen(os, "/start/"));

    SP_CHECK_RETURN(spFieldWrite(fE, os, "E", SP_FILE_NEW));
    SP_CHECK_RETURN(spFieldWrite(fB, os, "B", SP_FILE_NEW));
    SP_CHECK_RETURN(spFieldWrite(fJ, os, "J", SP_FILE_NEW));
    SP_CHECK_RETURN(spFieldWrite(fRho, os, "rho", SP_FILE_NEW));

    SP_CHECK_RETURN(spIOStreamOpen(os, "/checkpoint/"));

    for (int count = 0; count < num_of_steps; ++count)
    {
        spParallelDeviceSync();

        SP_CHECK_RETURN(spFieldClear(fJ));
        SP_CHECK_RETURN(spBorisYeeParticleUpdate(sp, dt, fE, fB, fRho, fJ));
        SP_CHECK_RETURN(spUpdateFieldYee(mesh, dt, fRho, fJ, fE, fB));

        spParallelDeviceSync();


        if (count % check_point == 0)
        {
            if (spMPIRank() == 0) { printf("====== STEP = %i ======\n", count); }

            SP_CHECK_RETURN(spFieldWrite(fE, os, "E", SP_FILE_RECORD));
            SP_CHECK_RETURN(spFieldWrite(fB, os, "B", SP_FILE_RECORD));
            SP_CHECK_RETURN(spFieldWrite(fJ, os, "J", SP_FILE_RECORD));
            SP_CHECK_RETURN(spFieldWrite(fRho, os, "rho", SP_FILE_RECORD));
        }

    }


    spParallelDeviceSync();

    SP_CHECK_RETURN(spIOStreamOpen(os, "/dump/"));

    SP_CHECK_RETURN(spFieldWrite(fE, os, "E", SP_FILE_NEW));
    SP_CHECK_RETURN(spFieldWrite(fB, os, "B", SP_FILE_NEW));
    SP_CHECK_RETURN(spFieldWrite(fJ, os, "J", SP_FILE_NEW));
    SP_CHECK_RETURN(spFieldWrite(fRho, os, "rho", SP_FILE_NEW));
    SP_CHECK_RETURN(spParticleWrite(sp, os, "H", SP_FILE_NEW));

    SP_CHECK_RETURN(spFieldDestroy(&fE));
    SP_CHECK_RETURN(spFieldDestroy(&fB));
    SP_CHECK_RETURN(spFieldDestroy(&fJ));
    SP_CHECK_RETURN(spFieldDestroy(&fRho));

    SP_CHECK_RETURN(spParticleDestroy(&sp));

    SP_CHECK_RETURN(spMeshDestroy(&mesh));

    DONE


    SP_CHECK_RETURN(spIOStreamDestroy(&os));
    SP_CHECK_RETURN(spParallelFinalize());

}
