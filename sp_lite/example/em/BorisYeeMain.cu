/*
 * BorisYeeMain.c
 *
 *  Created on: 2016年6月15日
 *      Author: salmon
 */

#include <stdio.h>
#include "../../src/sp_def.h"
#include "../../src/spMesh.h"
#include "../../src/spField.h"
#include "../../src/spParticle.h"
#include "Boris.h"
#include "BorisYee.h"

int main(int argc, char **argv)
{
#if defined(__CUDA_ARCH__)
	CUDA_CHECK_RETURN(cudaThreadSynchronize()); // Wait for the GPU launched work to complete
	CUDA_CHECK_RETURN(cudaGetLastError());
#endif
	spMesh *mesh;
	sp_particle_type *pg = 0x0;
	sp_field_type *fE = 0x0;
	sp_field_type *fB = 0x0;
	sp_field_type *fRho = 0x0;
	sp_field_type *fJ = 0x0;

	spCreateMesh(&mesh);
	mesh->dims[0] = 10;
	mesh->dims[1] = 10;
	mesh->dims[2] = 10;
	mesh->dx[0] = 10;
	mesh->dx[1] = 10;
	mesh->dx[2] = 10;
	spInitializeMesh(mesh);
	spCreateField(mesh, &fE, 1);
	spCreateField(mesh, &fB, 2);
	spCreateField(mesh, &fJ, 1);
	spCreateField(mesh, &fRho, 0);

	int NUMBER_OF_PIC = 256;

	spCreateParticle(mesh, &pg, sizeof(struct boris_point_s), NUMBER_OF_PIC);
	spInitializeParticle_BorisYee(mesh, pg, NUMBER_OF_PIC);

	int count = 10;
	Real dt = 1.0;
	while (count > 0)
	{
		spUpdateParticle_BorisYee(mesh, dt, pg, fE, fB, fRho, fJ);

		spUpdateField_Yee(mesh, dt, fRho, fJ, fE, fB);
////        spSyncParticle(mesh, pg, MPI_COMMON_GLOBAL);
////        spSyncField(mesh,  fJ, MPI_COMMON_GLOBAL);
////        spSyncField(mesh, fRho, MPI_COMMON_GLOBAL);
////
////        spWriteField(mesh,  fRho, "/checkpoint/rho", SP_RECORD);
////
////        spSyncField(mesh,  fE, MPI_COMMON_GLOBAL);
////        spSyncField(mesh,  fB, MPI_COMMON_GLOBAL);
//
		--count;
	}
//
////    spWriteField(mesh, h_fE, "/dump/rho", SP_NEW);
////    spWriteField(mesh, h_fB, "/dump/rho", SP_NEW);
////    spWriteField(mesh, h_fJ, "/dump/rho", SP_NEW);
////    spWriteField(mesh, h_fRho, "/dump/rho", SP_NEW);
////    spWriteParticle(mesh, pg, "/dump/H", SP_NEW);
//
//	CUDA_CHECK_RETURN(cudaFree(d_mesh));
//	CUDA_CHECK_RETURN(cudaFree(d_pg));
//	CUDA_CHECK_RETURN(cudaFree(d_fE));
//	CUDA_CHECK_RETURN(cudaFree(d_fB));
//	CUDA_CHECK_RETURN(cudaFree(d_fRho));
//	CUDA_CHECK_RETURN(cudaFree(d_fJ));

	spDestroyField(&fE);
	spDestroyField(&fB);
	spDestroyField(&fJ);
	spDestroyField(&fRho);
	spDestroyParticle(&pg);
	spDestroyMesh(&mesh);

#if defined(__CUDA_ARCH__)
	CUDA_CHECK_RETURN(cudaDeviceReset());
#endif
	CUDA_CHECK(0);
	return 0;
}
