/*
 * BorisYeeMain.c
 *
 *  Created on: 2016年6月15日
 *      Author: salmon
 */
#include <assert.h>
#include <stdio.h>

#include "sp_lite_def.h"
#include "spParallel.h"

#include "spMesh.h"
#include "spField.h"
#include "spParticle.h"
#include "spMisc.h"

#include "FDTDBoris.h"

#define TWOPI 3.141592653589793*2.0

int main(int argc, char **argv)
{
    SP_CHECK_RETURN(spParallelInitialize(argc, argv));

    spIOStream *os = NULL;
    SP_CHECK_RETURN(spIOStreamCreate(&os));
    SP_CHECK_RETURN(spIOStreamOpen(os, "untitled.h5"));

    spMesh *mesh;
    SP_CHECK_RETURN(spMeshCreate(&mesh));

    size_type dims[3] = {0x100, 0x100, 0x1};
    size_type gw[3] = {0x2, 0x2, 0x2};
    Real lower[3] = {0, 0, 0};
    Real upper[3] = {1, 1, 1};

    SP_CHECK_RETURN(spMeshSetDims(mesh, dims));
    SP_CHECK_RETURN(spMeshSetGhostWidth(mesh, gw));
    SP_CHECK_RETURN(spMeshSetBox(mesh, lower, upper));
    SP_CHECK_RETURN(spMeshDeploy(mesh));

    spField *fE = NULL;
    spField *fB = NULL;
    spField *fRho = NULL;
    spField *fJ = NULL;

    SP_CHECK_RETURN(spFieldCreate(&fE, mesh, 1, SP_TYPE_Real));
    SP_CHECK_RETURN(spFieldCreate(&fB, mesh, 2, SP_TYPE_Real));
    SP_CHECK_RETURN(spFieldCreate(&fJ, mesh, 1, SP_TYPE_Real));
    SP_CHECK_RETURN(spFieldCreate(&fRho, mesh, 0, SP_TYPE_Real));

    Real amp[3] = {1.0, 2.0, 3.0};
    Real k[3] = {TWOPI / (upper[0] - lower[0]), TWOPI / (upper[0] - lower[0]), 0};


    SP_CHECK_RETURN(spFieldClear(fE));
    SP_CHECK_RETURN(spFieldClear(fB));
    SP_CHECK_RETURN(spFieldClear(fJ));
    SP_CHECK_RETURN(spFieldClear(fRho));


    SP_CHECK_RETURN(spFieldAssignValueSin(fE, k, amp));

    spParticle *sp = NULL;

//    SP_CHECK_RETURN(spBorisYeeParticleCreate(&sp, mesh));
//    SP_CHECK_RETURN(spParticleDeploy(sp));

    int count = 5;

    Real dt = 1.0;

    SP_CHECK_RETURN(spIOStreamOpen(os, "/start/"));

    SP_CHECK_RETURN(spFieldWrite(fE, os, "E", SP_FILE_NEW));
    SP_CHECK_RETURN(spFieldWrite(fB, os, "B", SP_FILE_NEW));
    SP_CHECK_RETURN(spFieldWrite(fJ, os, "J", SP_FILE_NEW));
    SP_CHECK_RETURN(spFieldWrite(fRho, os, "rho", SP_FILE_NEW));

    SP_CHECK_RETURN(spIOStreamOpen(os, "/checkpoint/"));

    while (count > 0)
    {
//		printf("====== REMINED STEP= %i ======\n", count);
//        SP_CHECK_RETURN(spBorisYeeParticleUpdate(sp, dt, fE, fB, fRho, fJ));
        SP_CHECK_RETURN(spUpdateFieldYee(mesh, dt, fRho, fJ, fE, fB));

        SP_CHECK_RETURN(spFieldWrite(fE, os, "E", SP_FILE_RECORD));
        SP_CHECK_RETURN(spFieldWrite(fB, os, "B", SP_FILE_RECORD));
        SP_CHECK_RETURN(spFieldWrite(fJ, os, "J", SP_FILE_RECORD));
        SP_CHECK_RETURN(spFieldWrite(fRho, os, "rho", SP_FILE_RECORD));

        --count;
    }

    printf("======  The End ======\n");

    spParallelDeviceSync();

    SP_CHECK_RETURN(spIOStreamOpen(os, "/dump/"));

    SP_CHECK_RETURN(spFieldWrite(fE, os, "E", SP_FILE_NEW));
    SP_CHECK_RETURN(spFieldWrite(fB, os, "B", SP_FILE_NEW));
    SP_CHECK_RETURN(spFieldWrite(fJ, os, "J", SP_FILE_NEW));
    SP_CHECK_RETURN(spFieldWrite(fRho, os, "rho", SP_FILE_NEW));
//    SP_CHECK_RETURN(spParticleWrite(sp, os, "H", SP_FILE_NEW));

    SP_CHECK_RETURN(spFieldDestroy(&fE));
    SP_CHECK_RETURN(spFieldDestroy(&fB));
    SP_CHECK_RETURN(spFieldDestroy(&fJ));
    SP_CHECK_RETURN(spFieldDestroy(&fRho));

//    SP_CHECK_RETURN(spParticleDestroy(&sp));

    SP_CHECK_RETURN(spMeshDestroy(&mesh));

    SP_CHECK_RETURN(spIOStreamDestroy(&os));

    SP_CHECK_RETURN(spParallelFinalize());

    DONE
}
