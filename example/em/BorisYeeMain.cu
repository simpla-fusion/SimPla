/*
 * BorisYeeMain.c
 *
 *  Created on: 2016年6月15日
 *      Author: salmon
 */

#include <stdio.h>
#include "../../src/sp_config.h"
#include "../../src/capi/sp_cuda_common.h"
#include "../../src/capi/spMesh.h"
#include "../../src/capi/spField.h"
#include "../../src/capi/spParticle.h"
#include "Boris.h"
#include "BorisYee.h"

int main(int argc, char **argv)
{
	Real dt = 1.0;

	spMesh t_m;
	t_m.dims[0] = 10;
	t_m.dims[1] = 10;
	t_m.dims[2] = 10;
	t_m.dx[0] = 10;
	t_m.dx[1] = 10;
	t_m.dx[2] = 10;

#if defined(__CUDA_ARCH__)
	CUDA_CHECK_RETURN(cudaThreadSynchronize()); // Wait for the GPU launched work to complete
	CUDA_CHECK_RETURN(cudaGetLastError());
#endif

	spMesh *mesh = 0x0;
	sp_particle_type *pg = 0x0;
	sp_field_type *d_fE = 0x0;
	sp_field_type *d_fB = 0x0;
	sp_field_type *d_fRho = 0x0;
	sp_field_type *d_fJ = 0x0;

	spCreateMesh(&mesh);

	sp_memcpy(mesh, &t_m, sizeof(spMesh));
	spInitializeMesh(mesh);

	spCreateField(&t_m, &d_fE, 1);
	spCreateField(&t_m, &d_fB, 1);
	spCreateField(&t_m, &d_fJ, 1);
	spCreateField(&t_m, &d_fRho, 1);

	int NUMBER_OF_PIC = 256;
	spCreateParticle(&t_m, &pg, sizeof(struct boris_point_s), 1.0, 1.0);

//	spInitializeParticle_BorisYee(mesh, pg, NUMBER_OF_PIC);
	CUDA_CHECK(NUMBER_OF_PIC);

	int count = 10;

//	while (count > 0)
//	{
//		spUpdateParticle_BorisYee(mesh, pg, dt, d_fE, d_fB, d_fRho, d_fJ);
//
//		spUpdateField_Yee(mesh, dt, d_fRho, d_fJ, d_fE, d_fB);
////        spSyncParticle(mesh, pg, MPI_COMMON_GLOBAL);
////        spSyncField(mesh, d_fJ, MPI_COMMON_GLOBAL);
////        spSyncField(mesh, d_fRho, MPI_COMMON_GLOBAL);
////
////        spWriteField(mesh, d_fRho, "/checkpoint/rho", SP_RECORD);
////
////        spSyncField(mesh, d_fE, MPI_COMMON_GLOBAL);
////        spSyncField(mesh, d_fB, MPI_COMMON_GLOBAL);
//
//		--count;
//	}
//
////    spWriteField(mesh, d_fE, "/dump/rho", SP_NEW);
////    spWriteField(mesh, d_fB, "/dump/rho", SP_NEW);
////    spWriteField(mesh, d_fJ, "/dump/rho", SP_NEW);
////    spWriteField(mesh, d_fRho, "/dump/rho", SP_NEW);
////    spWriteParticle(mesh, pg, "/dump/H", SP_NEW);
//

	spDestroyField(mesh, &d_fE);
	spDestroyField(mesh, &d_fB);
	spDestroyField(mesh, &d_fJ);
	spDestroyField(mesh, &d_fRho);
	CUDA_CHECK(0);
	spDestroyParticle(mesh, &pg);
	CUDA_CHECK(0);

	spDestroyMesh(&mesh);
#if defined(__CUDA_ARCH__)
	CUDA_CHECK_RETURN(cudaDeviceReset());
#endif
	CUDA_CHECK(0);
	return 0;
}
