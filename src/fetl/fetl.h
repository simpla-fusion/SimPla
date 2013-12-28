/*Copyright (C) 2007-2011 YU Zhi. All rights reserved.
 *
 * $Id: FETL.h 1009 2011-02-07 23:20:45Z salmon $
 * FETL.h
 *
 *  Created on: 2009-3-31
 *      Author: salmon
 */
#ifndef FETL_H_
#define FETL_H_

#include "primitives.h"
#include "geometry.h"

#include "ntuple.h"
#include "field_ops.h"
#include "ntuple_ops.h"
#include "constant_ops.h"
#include "proxycache.h"

#ifndef NO_FIELD_IO_CACHE
#  include "field_rw_cache.h"
#endif

namespace simpla
{

#define DEFINE_FIELDS(TG)                                                                                          \
typedef TG Mesh;                                                                                                   \
template<int IFORM> using Form = Field<Geometry<Mesh,IFORM>,typename Mesh::scalar_type >;                               \
template<int IFORM> using VectorForm = Field<Geometry<Mesh,IFORM>,nTuple<3,typename Mesh::scalar_type> >;               \
template<int IFORM> using TensorForm = Field<Geometry<Mesh,IFORM>,nTuple<3,nTuple<3,typename Mesh::scalar_type> > >;    \
template<int IFORM> using RForm = Field<Geometry<Mesh,IFORM>,Real >;                                               \
template<int IFORM> using RVectorForm = Field<Geometry<Mesh,IFORM>,nTuple<3,Real> >;                               \
template<int IFORM> using RTensorForm = Field<Geometry<Mesh,IFORM>,nTuple<3,nTuple<3,Real> > >;                    \
template<int IFORM> using CForm = Field<Geometry<Mesh,IFORM>,Complex >;                                            \
template<int IFORM> using CVectorForm = Field<Geometry<Mesh,IFORM>,nTuple<3,Complex> >;                            \
template<int IFORM> using CTensorForm = Field<Geometry<Mesh,IFORM>,nTuple<3,nTuple<3,Complex> > >;

} // namespace simpla
#endif  // FETL_H_
