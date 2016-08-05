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
#include "spPhysicalConstants.h"

#include "spParallel.h"

#include "spMesh.h"
#include "spField.h"
#include "spParticle.h"

#include "spPICBoris.h"
#include "spFDTD.h"

#include "spMisc.h"

int main(int argc, char **argv)
{
    SP_CALL(spParallelInitialize(argc, argv));

    ShowSimPlaLogo();

    char out_file[2048] = "untitled.h5";

    int num_of_steps = argc < 2 ? 100 : atoi(argv[1]);
    int check_point = argc < 3 ? 10 : atoi(argv[2]);
    size_type PIC = 256;
    Real n0 = 1.0e18;
    Real T0 = 0.026;
    size_type dims[3] = {0x10, 0x10, 0x1};
    size_type gw[3] = {0x2, 0x2, 0x2};
    Real lower[3] = {0, 0, 0};
    Real upper[3] = {1, 1, 1};

    Real dt = nan("");

    Real amp[3] = {0.0, 0.0, 1.0};
    Real k[3] = {TWOPI / (upper[0] - lower[0]), TWOPI / (upper[0] - lower[0]), 0};

    /*****************************************************************************************************************/

    spIOStream *os = NULL;
    SP_CALL(spIOStreamCreate(&os));
    SP_CALL(spIOStreamOpen(os, out_file));

    /*****************************************************************************************************************/

    spMesh *mesh;

    SP_CALL(spMeshCreate(&mesh));
    SP_CALL(spMeshSetDims(mesh, dims));
    SP_CALL(spMeshSetGhostWidth(mesh, gw));
    SP_CALL(spMeshSetBox(mesh, lower, upper));
    SP_CALL(spMeshDeploy(mesh));

    if (isnan(dt)) { dt = spMeshCFLDt(mesh, 299792458.0/* speed_of_light*/); }
    /*****************************************************************************************************************/

    spField *fE = NULL;
    spField *fB = NULL;
    spField *fRho = NULL;
    spField *fJ = NULL;

    SP_CALL(spFieldCreate(&fE, mesh, EDGE, SP_TYPE_Real));
    SP_CALL(spFieldCreate(&fB, mesh, FACE, SP_TYPE_Real));
    SP_CALL(spFieldCreate(&fJ, mesh, EDGE, SP_TYPE_Real));
    SP_CALL(spFieldCreate(&fRho, mesh, VERTEX, SP_TYPE_Real));


    SP_CALL(spFieldClear(fE));
    SP_CALL(spFieldClear(fB));
    SP_CALL(spFieldClear(fJ));
    SP_CALL(spFieldClear(fRho));


    SP_CALL(spFieldAssignValueSin(fE, k, amp));
    /*****************************************************************************************************************/

    spParticle *sp = NULL;

    SP_CALL(spParticleCreateBorisYee(&sp, mesh));

    SP_CALL(spParticleSetMass(sp, SI_electron_mass));
    SP_CALL(spParticleSetCharge(sp, SI_elementary_charge));
    SP_CALL(spParticleSetPIC(sp, PIC));

    SP_CALL(spParticleInitializeBorisYee(sp, n0, T0));

    /*****************************************************************************************************************/

    SP_CALL(spIOStreamOpen(os, "/start/"));

    SP_CALL(spFieldWrite(fE, os, "E", SP_FILE_NEW));
    SP_CALL(spFieldWrite(fB, os, "B", SP_FILE_NEW));
    SP_CALL(spFieldWrite(fJ, os, "J", SP_FILE_NEW));
    SP_CALL(spFieldWrite(fRho, os, "rho", SP_FILE_NEW));

    SP_CALL(spParticleWrite(sp, os, "H", SP_FILE_NEW));

    SP_CALL(spIOStreamOpen(os, "/checkpoint/"));

    for (int count = 0; count < num_of_steps; ++count)
    {

        SP_CALL(spFieldClear(fJ));
        SP_CALL(spParticleUpdateBorisYee(sp, dt, fE, fB, fRho, fJ));
        SP_CALL(spUpdateFieldFDTD(mesh, dt, fRho, fJ, fE, fB));


        if (count % check_point == 0)
        {
            if (spMPIRank() == 0) { printf("====== STEP = %i ======\n", count); }

            SP_CALL(spFieldWrite(fE, os, "E", SP_FILE_RECORD));
            SP_CALL(spFieldWrite(fB, os, "B", SP_FILE_RECORD));
            SP_CALL(spFieldWrite(fJ, os, "J", SP_FILE_RECORD));
            SP_CALL(spFieldWrite(fRho, os, "rho", SP_FILE_RECORD));
        }

    }

    SP_CALL(spIOStreamOpen(os, "/dump/"));

    SP_CALL(spFieldWrite(fE, os, "E", SP_FILE_NEW));
    SP_CALL(spFieldWrite(fB, os, "B", SP_FILE_NEW));
    SP_CALL(spFieldWrite(fJ, os, "J", SP_FILE_NEW));
    SP_CALL(spFieldWrite(fRho, os, "rho", SP_FILE_NEW));

    SP_CALL(spFieldDestroy(&fE));
    SP_CALL(spFieldDestroy(&fB));
    SP_CALL(spFieldDestroy(&fJ));
    SP_CALL(spFieldDestroy(&fRho));

    SP_CALL(spParticleWrite(sp, os, "H", SP_FILE_NEW));
    SP_CALL(spParticleDestroy(&sp));

    SP_CALL(spMeshDestroy(&mesh));

    SP_CALL(spIOStreamDestroy(&os));

    DONE

    SP_CALL(spIOStreamDestroy(&os));
    SP_CALL(spParallelFinalize());

}
