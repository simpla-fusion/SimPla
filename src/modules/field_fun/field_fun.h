/*
 * set_value.h
 *
 *  Created on: 2012-10-29
 *      Author: salmon
 */

#ifndef FIELD_FUN_H_
#define FIELD_FUN_H_
#include "include/simpla_defs.h"
#include "engine/context.h"
#include "utilities/properties.h"
#include "detail/field_fun_impl.h"
#include "ramp_wave.h"
#include "assign_constant.h"
#include "smooth.h"
#include "damping.h"

namespace simpla
{
namespace field_fun
{

template<typename TG, template<typename > class TFun>
TR1::function<void(void)> Create(Context<TG>* ctx, ptree const & pt);

template<typename TG>
void RegisterModules(Context<TG> * ctx)

{
	DEFINE_FIELDS(typename TG::ValueType, TG)

	ctx->moduleFactory_["RampWave"] = TR1::bind(field_fun::Create<TG, RampWave>,
			ctx, TR1::placeholders::_1);

	ctx->moduleFactory_["AssignConstant"] = TR1::bind(
			field_fun::Create<TG, AssignConstant>, ctx, TR1::placeholders::_1);

//	moduleFactory_["Smooth"] = TR1::bind(
//			&field_fun::Create<TG, field_fun::Smooth>, ctx,
//			TR1::placeholders::_1);
//
//	moduleFactory_["Damping"] = TR1::bind(
//			&field_fun::Create<TG, field_fun::Damping>, ctx,
//			TR1::placeholders::_1);

}

}  // namespace field_op
}  // namespace simpla

#endif /* FIELD_FUN_H_ */
