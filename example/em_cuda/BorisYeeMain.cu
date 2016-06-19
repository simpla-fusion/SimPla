/*
 * BorisYeeMain.c
 *
 *  Created on: 2016年6月15日
 *      Author: salmon
 */

#include <stdio.h>
#include "../../sp_lite/sp_def.h"
#include "../../sp_lite/spMesh.h"
#include "../../sp_lite/spField.h"
#include "../../sp_lite/spParticle.h"
#include "Boris.h"
#include "BorisYee.h"

int main(int argc, char **argv)
{
	CUDA_CHECK_RETURN(cudaThreadSynchronize()); // Wait for the GPU launched work to complete
	CUDA_CHECK_RETURN(cudaGetLastError());
	spMesh *mesh;
	sp_particle_type *pg = 0x0;
	sp_field_type *fE = 0x0;
	sp_field_type *fB = 0x0;
	sp_field_type *fRho = 0x0;
	sp_field_type *fJ = 0x0;

	spCreateMesh(&mesh);
	mesh->dims[0] = 8;
	mesh->dims[1] = 8;
	mesh->dims[2] = 8;
	mesh->dx[0] = 1;
	mesh->dx[1] = 1;
	mesh->dx[2] = 1;
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
	spCreateParticle(mesh, &pg, sizeof(struct boris_point_s), NUMBER_OF_PIC);
	spInitializeParticle_BorisYee(mesh, pg, NUMBER_OF_PIC);

	int count = 1;
	Real dt = 1.0;
	while (count > 0)
	{
		printf("====== REMINED STEP= %d ======\n", count);
		spUpdateParticle_BorisYee(mesh, dt, pg, fE, fB, fRho, fJ);

		spUpdateField_Yee(mesh, dt, fRho, fJ, fE, fB);

//		spWriteField(mesh, fE, "/checkpoint/E", SP_RECORD);
//		spWriteField(mesh, fB, "/checkpoint/B", SP_RECORD);
//		spWriteField(mesh, fJ, "/checkpoint/J", SP_RECORD);
//		spWriteField(mesh, fRho, "/checkpoint/rho", SP_RECORD);

		--count;
	}

	spWriteField(mesh, fE, "/dump/E", SP_NEW);
	spWriteField(mesh, fB, "/dump/B", SP_NEW);
	spWriteField(mesh, fJ, "/dump/J", SP_NEW);
	spWriteField(mesh, fRho, "/dump/rho", SP_NEW);
//	spWriteParticle(mesh, pg, "/dump/H", SP_NEW);
	spDestroyField(&fE);
	spDestroyField(&fB);
	spDestroyField(&fJ);
	spDestroyField(&fRho);
	spDestroyParticle(&pg);
	spDestroyMesh(&mesh);

	CUDA_CHECK_RETURN(cudaDeviceReset());
	DONE

	return 0;
}
