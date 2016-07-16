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
    spParallelInitialize(0, nullptr);

    spMesh *mesh;
    spParticle *ps = 0x0;
    spField *fE = 0x0;
    spField *fB = 0x0;
    spField *fRho = 0x0;
    spField *fJ = 0x0;

    spMeshCreate(&mesh);

    dim3 dims;
    dim3 gw;
    dims.x = 0x8;
    dims.y = 0x8;
    dims.z = 0x8;
    gw.x = 0x2;
    gw.y = 0x2;
    gw.z = 0x2;

    spMeshSetDims(mesh, dims);

    spMeshSetGhostWidth(mesh, gw);

    Real3 lower, upper;
    lower.x = 0;
    lower.y = 0;
    lower.z = 0;
    upper.x = 0;
    upper.y = 0;
    upper.z = 0;

    spMeshSetBox(mesh, lower, upper);

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
    spIOStreamOpen(os, "/start/");

    spFieldWrite(fE, os, "E", SP_FILE_NEW);
    spFieldWrite(fB, os, "B", SP_FILE_NEW);
    spFieldWrite(fJ, os, "J", SP_FILE_NEW);
    spFieldWrite(fRho, os, "rho", SP_FILE_NEW);
    spIOStreamOpen(os, "/checkpoint/");
//	while (count > 0)
//	{
//		printf("====== REMINED STEP= %i ======\n", count);
//		spBorisYeeUpdateParticle(ps, dt, fE, fB, fRho, fJ);
//////		spUpdateField_Yee( dt, fRho, fJ, fE, fB);
//////
//	spFieldWrite(fE, os, "E", SP_FILE_RECORD);
//	spFieldWrite(fB, os, "B", SP_FILE_RECORD);
//	spFieldWrite(fJ, os, "J", SP_FILE_RECORD);
//	spFieldWrite(fRho, os, "rho", SP_FILE_RECORD);
////
//		--count;
//	}
    printf("======  The End ======\n");
    spParallelDeviceSync();
    spIOStreamOpen(os, "/dump/");

    spFieldWrite(fE, os, "E", SP_FILE_NEW);
    spFieldWrite(fB, os, "B", SP_FILE_NEW);
    spFieldWrite(fJ, os, "J", SP_FILE_NEW);
    spFieldWrite(fRho, os, "rho", SP_FILE_NEW);
    spParticleWrite(ps, os, "H", SP_FILE_NEW);

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
