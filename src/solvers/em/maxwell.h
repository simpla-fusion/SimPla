/*Copyright (C) 2007-2011 YU Zhi. All rights reserved.
 * $Id$
 * Maxwell/Maxwell.h
 *
 *  Created on: 2010-11-16
 *      Author: salmon
 */

#ifndef SRC_EMFIELD_MAXWELL_H_
#define SRC_EMFIELD_MAXWELL_H_

#include "fetl/fetl.h"
#include "fetl/vector_calculus.h"
#include "engine/solver.h"
#include "engine/context.h"

namespace simpla
{
namespace em
{

template<typename TV, typename TG>
void Maxwell(TR1::shared_ptr<Context> ctx)
{

	using namespace fetl;
	using namespace vector_calculus;

	DEFINE_FIELDS(TV,TG);

	double dt;
	double mu0;
	double epsilon0;

	OneForm & E1 = *ctx->template FindObject<OneForm>("E1");
	TwoForm & B1 = *ctx->template FindObject<TwoForm>("B1");
	OneForm & J1 = *ctx->template FindObject<OneForm>("J1");

	B1 -= Curl(E1) * dt;
	E1 += (Curl(B1 / mu0) - J1) / epsilon0 * dt;
}

} // namespace em_field
} // namespace simpla
#endif  // SRC_EMFIELD_MAXWELL_H_
