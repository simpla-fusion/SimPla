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

#include "BorisYee.h"

int main(int argc, char **argv)
{
    spParallelInitialize(argc, argv);

    spIOStream *os = NULL;
    spIOStreamCreate(&os);
    spIOStreamOpen(os, "untitled.h5");

    spMesh *mesh;
    spMeshCreate(&mesh);

    size_type dims[3] = {0x8, 0x8, 0x1};
    size_type gw[3] = {0x2, 0x2, 0x2};
    Real lower[3] = {0, 0, 0};
    Real upper[3] = {1, 1, 1};

    spMeshSetDims(mesh, dims);
    spMeshSetGhostWidth(mesh, gw);
    spMeshSetBox(mesh, lower, upper);
    spMeshDeploy(mesh);

    spField *fE = NULL;
    spField *fB = NULL;
    spField *fRho = NULL;
    spField *fJ = NULL;

    spFieldCreate(mesh, &fE, 1);
    spFieldCreate(mesh, &fB, 2);
    spFieldCreate(mesh, &fJ, 1);
    spFieldCreate(mesh, &fRho, 0);

    spFieldClear(fE);
    spFieldClear(fB);
    spFieldClear(fJ);
    spFieldClear(fRho);


    spParticle *sp = NULL;

    spBorisYeeParticleCreate(mesh, &sp);

    spParticleDeploy(sp, 256/* number of PIC */);

    int count = 5;

    Real dt = 1.0;

    spIOStreamOpen(os, "/start/");

    spFieldWrite(fE, os, "E", SP_FILE_NEW);
    spFieldWrite(fB, os, "B", SP_FILE_NEW);
    spFieldWrite(fJ, os, "J", SP_FILE_NEW);
    spFieldWrite(fRho, os, "rho", SP_FILE_NEW);

    spIOStreamOpen(os, "/checkpoint/");

    while (count > 0)
    {
//		printf("====== REMINED STEP= %i ======\n", count);
        spBorisYeeParticleUpdate(sp, dt, fE, fB, fRho, fJ);
        spUpdateField_Yee(mesh, dt, fRho, fJ, fE, fB);

        spFieldWrite(fE, os, "E", SP_FILE_RECORD);
        spFieldWrite(fB, os, "B", SP_FILE_RECORD);
        spFieldWrite(fJ, os, "J", SP_FILE_RECORD);
        spFieldWrite(fRho, os, "rho", SP_FILE_RECORD);

        --count;
    }
    printf("======  The End ======\n");
    spParallelDeviceSync();
    spIOStreamOpen(os, "/dump/");

    spFieldWrite(fE, os, "E", SP_FILE_NEW);
    spFieldWrite(fB, os, "B", SP_FILE_NEW);
    spFieldWrite(fJ, os, "J", SP_FILE_NEW);
    spFieldWrite(fRho, os, "rho", SP_FILE_NEW);

    spParticleWrite(sp, os, "H", SP_FILE_NEW);

    spFieldDestroy(&fE);
    spFieldDestroy(&fB);
    spFieldDestroy(&fJ);
    spFieldDestroy(&fRho);

    spParticleDestroy(&sp);

    spMeshDestroy(&mesh);

    spIOStreamDestroy(&os);

    spParallelFinalize();

    DONE
}
