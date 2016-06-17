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
	spMesh *h_mesh;
	sp_particle_type *h_pg = 0x0;
	sp_field_type *h_fE = 0x0;
	sp_field_type *h_fB = 0x0;
	sp_field_type *h_fRho = 0x0;
	sp_field_type *h_fJ = 0x0;

	spCreateMesh(&h_mesh);
	h_mesh->dims[0] = 10;
	h_mesh->dims[1] = 10;
	h_mesh->dims[2] = 10;
	h_mesh->dx[0] = 10;
	h_mesh->dx[1] = 10;
	h_mesh->dx[2] = 10;
	spInitializeMesh(h_mesh);
	spCreateField(h_mesh, &h_fE, 1);
	spCreateField(h_mesh, &h_fB, 1);
	spCreateField(h_mesh, &h_fJ, 1);
	spCreateField(h_mesh, &h_fRho, 1);

	int NUMBER_OF_PIC = 256;

	spCreateParticle(h_mesh, &h_pg, sizeof(struct boris_point_s),
			NUMBER_OF_PIC);
	spInitializeParticle_BorisYee(h_mesh, h_pg, NUMBER_OF_PIC);

	spMesh *d_mesh;
	sp_particle_type *d_pg = 0x0;
	sp_field_type *d_fE = 0x0;
	sp_field_type *d_fB = 0x0;
	sp_field_type *d_fRho = 0x0;
	sp_field_type *d_fJ = 0x0;

	CUDA_CHECK_RETURN(cudaMalloc(&d_mesh, sizeof(spMesh)));
	CUDA_CHECK_RETURN(cudaMalloc(&d_pg, sizeof(sp_particle_type)));
	CUDA_CHECK_RETURN(cudaMalloc(&d_fE, sizeof(sp_field_type)));
	CUDA_CHECK_RETURN(cudaMalloc(&d_fB, sizeof(sp_field_type)));
	CUDA_CHECK_RETURN(cudaMalloc(&d_fRho, sizeof(sp_field_type)));
	CUDA_CHECK_RETURN(cudaMalloc(&d_fJ, sizeof(sp_field_type)));

	CUDA_CHECK_RETURN(
			cudaMemcpy(d_mesh, h_mesh, sizeof(spMesh), cudaMemcpyDefault));
	CUDA_CHECK_RETURN(
			cudaMemcpy(d_pg, h_pg, sizeof(sp_particle_type), cudaMemcpyDefault));
	CUDA_CHECK_RETURN(
			cudaMemcpy(d_fE, h_fE, sizeof(sp_field_type), cudaMemcpyDefault));
	CUDA_CHECK_RETURN(
			cudaMemcpy(d_fB, h_fB, sizeof(sp_field_type), cudaMemcpyDefault));
	CUDA_CHECK_RETURN(
			cudaMemcpy(d_fRho, h_fRho, sizeof(sp_field_type),
					cudaMemcpyDefault));
	CUDA_CHECK_RETURN(
			cudaMemcpy(d_fJ, h_fJ, sizeof(sp_field_type), cudaMemcpyDefault));

	int count = 10;
	Real dt = 1.0;
	while (count > 0)
	{
		spUpdateParticle_BorisYee(d_mesh, d_pg, dt, d_fE, d_fB, d_fRho, d_fJ);
//
		spUpdateField_Yee(d_mesh, dt, d_fRho, d_fJ, d_fE, d_fB);
////        spSyncParticle(mesh, pg, MPI_COMMON_GLOBAL);
////        spSyncField(mesh, h_fJ, MPI_COMMON_GLOBAL);
////        spSyncField(mesh, h_fRho, MPI_COMMON_GLOBAL);
////
////        spWriteField(mesh, h_fRho, "/checkpoint/rho", SP_RECORD);
////
////        spSyncField(mesh, h_fE, MPI_COMMON_GLOBAL);
////        spSyncField(mesh, h_fB, MPI_COMMON_GLOBAL);
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
	CUDA_CHECK_RETURN(cudaFree(d_mesh));
	CUDA_CHECK_RETURN(cudaFree(d_pg));
	CUDA_CHECK_RETURN(cudaFree(d_fE));
	CUDA_CHECK_RETURN(cudaFree(d_fB));
	CUDA_CHECK_RETURN(cudaFree(d_fRho));
	CUDA_CHECK_RETURN(cudaFree(d_fJ));

	spDestroyField(&h_fE);

	spDestroyField(&h_fB);
	spDestroyField(&h_fJ);
	spDestroyField(&h_fRho);

	spDestroyParticle(&h_pg);

	spDestroyMesh(&h_mesh);

#if defined(__CUDA_ARCH__)
	CUDA_CHECK_RETURN(cudaDeviceReset());
#endif
	CUDA_CHECK(0);
	return 0;
}
