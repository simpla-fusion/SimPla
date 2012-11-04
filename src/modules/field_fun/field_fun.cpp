/*
 * field_fun.cpp
 *
 *  Created on: 2012-10-30
 *      Author: salmon
 */
#include "field_fun.h"
#include "fetl/grid/uniform_rect.h"
namespace simpla
{
namespace field_fun
{


#define DEFINE_FIELD_FUNCTION(TFUN)                                                        \
template<>                                                                                 \
struct FieldFunction<Field<UniformRectGrid, ITwoForm, Real>,                               \
	TFUN<nTuple<THREE, Real> > > ;                                                 \
template<>                                                                                 \
struct FieldFunction<Field<UniformRectGrid, IOneForm, Real>,                               \
	TFUN<nTuple<THREE, Real> > > ;                                                 \
template<>                                                                                 \
struct FieldFunction<Field<UniformRectGrid, IZeroForm, nTuple<THREE, Real> >,              \
	TFUN<nTuple<THREE, Real> > > ;                                                 \


DEFINE_FIELD_FUNCTION(RampWave)
DEFINE_FIELD_FUNCTION(AssignConstant)
}  // namespace field_fun

}  // namespace simpla

