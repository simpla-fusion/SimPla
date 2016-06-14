//
// Created by salmon on 16-6-11.
//

#ifndef SIMPLA_BORISYEE_H
#define SIMPLA_BORISYEE_H

#include "../../src/sp_config.h"
#include "../../src/particle/ParticleInterface.h"

#ifdef __cplusplus
extern "C"
{
#endif
typedef struct BorisYeeUpdateArgs_s
{
	Real cmr;
	Real inv_dx[3];
	size_type i_lower[3];
	size_type i_upper[3];
	size_type i_dims[3];
	size_type number_of_idx;
	size_type *cell_idx;
	size_type ele_size_in_type;

} BorisYeeUpdateArgs;

void spBorisYeeUpdate(BorisYeeUpdateArgs const *args, Real dt, spPage **first,
		spPage **second, spPagePool *pool, const Real *fE, const Real *fB,
		Real *fRho, Real *fJ);

#ifdef __cplusplus
}
;
#endif

#endif //SIMPLA_BORISYEE_H
