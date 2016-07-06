/*
 * BorisYeeMain.c
 *
 *  Created on: 2016年6月15日
 *      Author: salmon
 */

#include <stdio.h>

#include "sp_def.h"

#include "spMesh.h"
#include "spField.h"
#include "spParticle.h"

#include "Boris.h"
#include "BorisYee.h"

int main(int argc, char **argv)
{
	CUDA_CHECK_RETURN(cudaThreadSynchronize()); // Wait for the GPU launched work to complete
	CUDA_CHECK_RETURN(cudaGetLastError());
	spMesh *mesh;
	sp_particle_type *ps = 0x0;
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

	spFieldClear(mesh, fE);
	spFieldClear(mesh, fB);
	spFieldClear(mesh, fJ);
	spFieldClear(mesh, fRho);

	int NUMBER_OF_PIC = 256;
	spParticleCreate(mesh, &ps);

	spBorisYeeInitializeParticle(mesh, ps, NUMBER_OF_PIC);

	int count = 5;
	Real dt = 1.0;

	spIOStream *os = NULL;

	spIOStreamCreate(&os);

	spIOStreamOpen(os, "untitled.h5", SP_FILE_NEW);

	spFieldWrite(mesh, fE, os, "/start/E", SP_FILE_NEW);
	spFieldWrite(mesh, fB, os, "/start/B", SP_FILE_NEW);
	spFieldWrite(mesh, fJ, os, "/start/J", SP_FILE_NEW);
	spFieldWrite(mesh, fRho, os, "/start/rho", SP_FILE_NEW);

	while (count > 0)
	{
		printf("====== REMINED STEP= %i ======\n", count);
		spBorisYeeUpdateParticle(mesh, dt, ps, fE, fB, fRho, fJ);
////		spUpdateField_Yee(mesh, dt, fRho, fJ, fE, fB);
////
////		spFieldWrite(mesh, fE, "/checkpoint/E", SP_FILE_RECORD);
////		spFieldWrite(mesh, fB, "/checkpoint/B", SP_FILE_RECORD);
////		spFieldWrite(mesh, fJ, "/checkpoint/J", SP_FILE_RECORD);
////		spFieldWrite(mesh, fRho, "/checkpoint/rho", SP_FILE_RECORD);
//
		--count;
	}
	printf("======  The End ======\n");
	CUDA_CHECK_RETURN(cudaDeviceSynchronize()); // Wait for the GPU launched work to complete

//	spFieldWrite(mesh, fE, "/dump/E", SP_FILE_NEW);
//	spFieldWrite(mesh, fB, "/dump/B", SP_FILE_NEW);
//	spFieldWrite(mesh, fJ, "/dump/J", SP_FILE_NEW);
//	spFieldWrite(mesh, fRho, "/dump/rho", SP_FILE_NEW);
//	spParticleWrite(mesh, ps, "/dump/H", SP_FILE_NEW);

	spFieldDestroy(&fE);
	spFieldDestroy(&fB);
	spFieldDestroy(&fJ);
	spFieldDestroy(&fRho);
	spParticleDestroy(&ps);
	spMeshDestroy(&mesh);
	spIOStreamDestroy(&os);

	CUDA_CHECK_RETURN(cudaDeviceReset());
	DONE
	return 0;
}
