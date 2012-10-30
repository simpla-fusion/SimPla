/*
 * field_fun.cpp
 *
 *  Created on: 2012-10-30
 *      Author: salmon
 */
#include "field_fun.h"

namespace simpla
{

namespace field_fun
{

template<>
TR1::function<void(void)> Create<UniformRectGrid, RampWave>(
		Context<UniformRectGrid>* ctx, ptree const & pt);

}  // namespace field_fun

}  // namespace simpla

