/*
 * BorisYee.c
 *
 *  Created on: 2016年7月9日
 *      Author: salmon
 */
#include <assert.h>
#include <math.h>
#include "sp_lite_def.h"
#include "spField.h"
#include "spMesh.h"
#include "spParticle.h"
#include "spPage.h"
#include "BorisYee.h"

#define ADD_PARTICLE_ATTRIBUTE(_SP_, _S_, _T_, _N_) spParticleAddAttribute(_SP_, __STRING(_N_), SP_TYPE_##_T_, sizeof(_T_), offsetof(_S_,_N_));

void spBorisYeeInitializeParticle(spParticle *sp, int NUM_OF_PIC)
{

	ADD_PARTICLE_ATTRIBUTE(sp, struct boris_s, Real, vx);
	ADD_PARTICLE_ATTRIBUTE(sp, struct boris_s, Real, vy);
	ADD_PARTICLE_ATTRIBUTE(sp, struct boris_s, Real, vz);
	ADD_PARTICLE_ATTRIBUTE(sp, struct boris_s, Real, f);
	ADD_PARTICLE_ATTRIBUTE(sp, struct boris_s, Real, w);

	spParticleDeploy(sp, NUM_OF_PIC);
}
