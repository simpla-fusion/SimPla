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
	sp_field_type *fE = 0x0;
	sp_field_type *fB = 0x0;
	sp_field_type *fRho = 0x0;
	sp_field_type *fJ = 0x0;

	spCreateMesh(&mesh);
	mesh->dims.x = 256;
	mesh->dims.y = 256;
	mesh->dims.z = 256;
	mesh->dx.x = 1;
	mesh->dx.y = 1;
	mesh->dx.z = 1;
	spInitializeMesh(mesh);
	spCreateField(mesh, &fE, 1);
	spCreateField(mesh, &fB, 2);
	spCreateField(mesh, &fJ, 1);
	spCreateField(mesh, &fRho, 0);

	spClearField(mesh, fE);
	spClearField(mesh, fB);
	spClearField(mesh, fJ);
	spClearField(mesh, fRho);

	int NUMBER_OF_PIC = 256;
	spCreateParticle(mesh, &ps, (sizeof(struct boris_page_s) - SP_PAGE_HEAD_SIZE) / SP_NUMBER_OF_ENTITIES_IN_PAGE,
			NUMBER_OF_PIC);
	spInitializeParticle_BorisYee(mesh, ps, NUMBER_OF_PIC);

	int count = 5;
	Real dt = 1.0;

	spWriteField(mesh, fE, "/start/E", SP_NEW);
	spWriteField(mesh, fB, "/start/B", SP_NEW);
	spWriteField(mesh, fJ, "/start/J", SP_NEW);
	spWriteField(mesh, fRho, "/start/rho", SP_NEW);
#ifdef ENABLE_SOA
	printf("====== Enable SOA ======\n");
#else
	printf("====== Enable AOS ======\n");
#endif
	while (count > 0)
	{
		printf("====== REMINED STEP= %i ======\n", count);
		spUpdateParticle_BorisYee(mesh, dt, ps, fE, fB, fRho, fJ);
//		spUpdateField_Yee(mesh, dt, fRho, fJ, fE, fB);

//		spWriteField(mesh, fE, "/checkpoint/E", SP_RECORD);
//		spWriteField(mesh, fB, "/checkpoint/B", SP_RECORD);
//		spWriteField(mesh, fJ, "/checkpoint/J", SP_RECORD);
//		spWriteField(mesh, fRho, "/checkpoint/rho", SP_RECORD);

		--count;
	}
	printf("======  The End ======\n");

	spWriteField(mesh, fE, "/dump/E", SP_NEW);
	spWriteField(mesh, fB, "/dump/B", SP_NEW);
	spWriteField(mesh, fJ, "/dump/J", SP_NEW);
	spWriteField(mesh, fRho, "/dump/rho", SP_NEW);
	spWriteParticle(mesh, ps, "/dump/H", SP_NEW);

	CUDA_CHECK_RETURN(cudaDeviceSynchronize()); // Wait for the GPU launched work to complete

	spDestroyField(&fE);
	spDestroyField(&fB);
	spDestroyField(&fJ);
	spDestroyField(&fRho);
	spDestroyParticle(&ps);
	spDestroyMesh(&mesh);

	CUDA_CHECK_RETURN(cudaDeviceReset());
	DONE
	return 0;
}
