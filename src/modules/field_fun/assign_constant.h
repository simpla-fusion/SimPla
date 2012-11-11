/*
 * assign_constant.h
 *
 *  Created on: 2012-11-1
 *      Author: salmon
 */

#ifndef ASSIGN_CONSTANT_H_
#define ASSIGN_CONSTANT_H_

#include "include/simpla_defs.h"
#include "utilities/properties.h"
#include "fetl/grid.h"
#include "fetl/fetl.h"
#include "engine/context.h"
namespace simpla
{

namespace field_fun
{

template<typename TV>
struct AssignConstant
{
	TV value;

	AssignConstant(ptree const pt) :
			value(pt.get<TV>("Arguments.value"))
	{
	}
	~AssignConstant()
	{
	}
	template<typename TE> inline TV operator()(nTuple<THREE, TE>, Real)
	{
		return value;
	}

};

}  // namespace field_fun

}  // namespace simpla

#endif /* ASSIGN_CONSTANT_H_ */
