/*
 * set_value.h
 *
 *  Created on: 2012-10-29
 *      Author: salmon
 */

#ifndef SET_VALUE_H_
#define SET_VALUE_H_
#include "include/simpla_defs.h"
#include "engine/context.h"
#include "utilities/properties.h"

namespace simpla
{
namespace field_fun
{

template<typename TG, template<typename > class TFun>
TR1::function<void(void)> Create(Context<TG>* ctx, ptree const & pt);

}  // namespace field_op
}  // namespace simpla
#include "detail/field_fun_impl.h"
#include "ramp_wave.h"
#include "smooth.h"
#include "damping.h"
#endif /* SET_VALUE_H_ */
