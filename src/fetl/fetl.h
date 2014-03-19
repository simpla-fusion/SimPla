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

#include "ntuple.h"
#include "field.h"
#include "field_ops.h"
#include "field_vector_ops.h"
#include "field_constant.h"
#include "complex_ops.h"
#include "cache.h"


namespace simpla
{
#define DEFINE_FIELDS(TM)                                                                              \
typedef TM Mesh;                                                                                       \
typedef typename Mesh::index_type index_type ;                                                         \
typedef typename Mesh::scalar_type scalar_type;                                                        \
typedef typename Mesh::coordinates_type coordinates_type;                                              \
template<int IFORM> using Form = Field<Mesh,IFORM,scalar_type >;              \
template<int IFORM> using VectorForm = Field<Mesh,IFORM,nTuple<3,scalar_type> >;    \
template<int IFORM> using TensorForm = Field<Mesh,IFORM,nTuple<3,nTuple<3,scalar_type> > >;    \
template<int IFORM> using RForm = Field<Mesh,IFORM,Real >;                                               \
template<int IFORM> using RVectorForm = Field<Mesh,IFORM,nTuple<3,Real> >;                               \
template<int IFORM> using RTensorForm = Field<Mesh,IFORM,nTuple<3,nTuple<3,Real> > >;                    \
template<int IFORM> using CForm = Field<Mesh,IFORM,Complex >;                                            \
template<int IFORM> using CVectorForm = Field<Mesh,IFORM,nTuple<3,Complex> >;                            \
template<int IFORM> using CTensorForm = Field<Mesh,IFORM,nTuple<3,nTuple<3,Complex> > >;

} // namespace simpla
#endif  // FETL_H_
