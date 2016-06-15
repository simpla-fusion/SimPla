/*
 * BorisYeeMain.c
 *
 *  Created on: 2016年6月15日
 *      Author: salmon
 */

#include "../../src/sp_config.h"
#include "../../src/capi/sp_cuda_common.h"
#include "../../src/capi/spMesh.h"
#include "../../src/capi/spField.h"
#include "../../src/capi/spParticle.h"
#include "BorisYee.h"

int
main (int argc, char **argv)
{
  Real dt = 1.0;

  spMesh * mesh = 0x0;
  sp_particle_type *pg = 0x0;
  sp_field_type *d_fE = 0x0;
  sp_field_type *d_fB = 0x0;
  sp_field_type *d_fRho = 0x0;
  sp_field_type *d_fJ = 0x0;

#if defined(__CUDA_ARCH__)
  CUDA_CHECK_RETURN(cudaThreadSynchronize ());// Wait for the GPU launched work to complete
  CUDA_CHECK_RETURN(cudaGetLastError ());
#endif
  spCreateMesh (&mesh);

  spCreateField (mesh, &d_fE, 1);
  spCreateField (mesh, &d_fB, 1);
  spCreateField (mesh, &d_fJ, 1);
  spCreateField (mesh, &d_fRho, 1);

  int NUMBER_OF_PIC = 256;
  spCreateParticle (mesh, &pg, NUMBER_OF_PIC);
  spInitializeParticle_BorisYee (mesh, pg);

  int count = 10;

  while (count > 0)
	{
	  spUpdateParticle_BorisYee (mesh, pg, dt, d_fE, d_fB, d_fRho, d_fJ);

	  spUpdateField_Yee (mesh, dt, d_fRho, d_fJ, d_fE, d_fB);

	  // check_point();
	  --count;
	}

  spDestroyField (mesh, &d_fE);
  spDestroyField (mesh, &d_fB);
  spDestroyField (mesh, &d_fJ);
  spDestroyField (mesh, &d_fRho);

  spDestroyParticle (mesh, &pg);

  spDestroyMesh (&mesh);
#if defined(__CUDA_ARCH__)
  CUDA_CHECK_RETURN(cudaDeviceReset ());
#endif
  return 0;
}
