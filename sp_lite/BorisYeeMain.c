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
#include "spPage.h"




int main(int argc, char **argv)
{
    spParallelInitialize();

    spMesh *mesh;
    spParticle *ps = 0x0;
    spField *fE = 0x0;
    spField *fB = 0x0;
    spField *fRho = 0x0;
    spField *fJ = 0x0;

    spMeshCreate(&mesh);
    mesh->dims.x = 0x8;
    mesh->dims.y = 0x8;
    mesh->dims.z = 0x8;
    mesh->dx.x = 1;
    mesh->dx.y = 1;
    mesh->dx.z = 1;
    spMeshDeploy(mesh);

    spFieldCreate(mesh, &fE, 1);
    spFieldCreate(mesh, &fB, 2);
    spFieldCreate(mesh, &fJ, 1);
    spFieldCreate(mesh, &fRho, 0);

    spFieldClear(fE);
    spFieldClear(fB);
    spFieldClear(fJ);
    spFieldClear(fRho);

    int NUMBER_OF_PIC = 256;
    spParticleCreate(mesh, &ps);

    spBorisYeeInitializeParticle(ps, NUMBER_OF_PIC);

    int count = 5;
    Real dt = 1.0;

    spIOStream *os = NULL;

    spIOStreamCreate(&os);

    spIOStreamOpen(os, "untitled.h5");

    spFieldWrite(fE, os, "/start/E", SP_FILE_NEW);
    spFieldWrite(fB, os, "/start/B", SP_FILE_NEW);
    spFieldWrite(fJ, os, "/start/J", SP_FILE_NEW);
    spFieldWrite(fRho, os, "/start/rho", SP_FILE_NEW);

    while (count > 0)
    {
        printf("====== REMINED STEP= %i ======\n", count);
        spBorisYeeUpdateParticle(ps, dt, fE, fB, fRho, fJ);
////		spUpdateField_Yee( dt, fRho, fJ, fE, fB);
////
////		spFieldWrite( fE, "/checkpoint/E", SP_FILE_RECORD);
////		spFieldWrite( fB, "/checkpoint/B", SP_FILE_RECORD);
////		spFieldWrite( fJ, "/checkpoint/J", SP_FILE_RECORD);
////		spFieldWrite( fRho, "/checkpoint/rho", SP_FILE_RECORD);
//
        --count;
    }
    printf("======  The End ======\n");
    spParallelThreadSync();

//	spFieldWrite( fE, "/dump/E", SP_FILE_NEW);
//	spFieldWrite( fB, "/dump/B", SP_FILE_NEW);
//	spFieldWrite( fJ, "/dump/J", SP_FILE_NEW);
//	spFieldWrite( fRho, "/dump/rho", SP_FILE_NEW);
//	spParticleWrite( ps, "/dump/H", SP_FILE_NEW);

    spFieldDestroy(&fE);
    spFieldDestroy(&fB);
    spFieldDestroy(&fJ);
    spFieldDestroy(&fRho);
    spParticleDestroy(&ps);
    spMeshDestroy(&mesh);

    spIOStreamDestroy(&os);

    spParallelFinalize();

//	DONE
    ;
}
