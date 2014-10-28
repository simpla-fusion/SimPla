/*Copyright (C) 2007-2011 YU Zhi. All rights reserved.
 *
 * $Id: FETL.h 1009 2011-02-07 23:20:45Z salmon $
 * FETL.h
 *
 *  created on: 2009-3-31
 *      Author: salmon
 */
#ifndef FETL_H_
#define FETL_H_

#include "../utilities/primitives.h"
#include "../utilities/ntuple.h"
#include "../utilities/sp_complex.h"

#include "../field/field.h"
#include "calculus.h"
#include "domain.h"
#include "manifold.h"

//#include "field_vector_ops.h"
//#include "field_constant.h"
//#include "cache.h"

/**
 * \defgroup  FETL Field expression template library
 *
 *  \brief Field expression template library
 *
 */
//namespace simpla
//{
//#define DEFINE_FIELDS(TM)                                                                              \
//typedef TM Mesh;                                                                                       \
//typedef typename Mesh::iterator iterator ;                                                         \
//typedef typename Mesh::scalar_type scalar_type;                                                        \
//typedef typename Mesh::coordinates_type coordinates_type;                                              \
//template<unsigned int IFORM> using Form = Field<Mesh,IFORM,scalar_type >;              \
//template<unsigned int IFORM> using VectorForm = Field<Mesh,IFORM,nTuple<3,scalar_type> >;    \
//template<unsigned int IFORM> using TensorForm = Field<Mesh,IFORM,nTuple<3,nTuple<3,scalar_type> > >;    \
//template<unsigned int IFORM> using RForm = Field<Mesh,IFORM,Real >;                                               \
//template<unsigned int IFORM> using RVectorForm = Field<Mesh,IFORM,nTuple<3,Real> >;                               \
//template<unsigned int IFORM> using RTensorForm = Field<Mesh,IFORM,nTuple<3,nTuple<3,Real> > >;                    \
//template<unsigned int IFORM> using CForm = Field<Mesh,IFORM,Complex >;                                            \
//template<unsigned int IFORM> using CVectorForm = Field<Mesh,IFORM,nTuple<3,Complex> >;                            \
//template<unsigned int IFORM> using CTensorForm = Field<Mesh,IFORM,nTuple<3,nTuple<3,Complex> > >;
//
//} // namespace simpla
#endif  // FETL_H_
