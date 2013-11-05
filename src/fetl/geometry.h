/*
 * geometry.h
 *
 *  Created on: 2013-7-20
 *      Author: salmon
 */

#ifndef GEOMETRY_H_
#define GEOMETRY_H_

#include <fetl/ntuple.h>
#include <fetl/primitives.h>

namespace simpla
{

template<typename, int> struct Geometry;

template<typename TM>
struct Geometry<TM, 0>
{
	static const int IForm = 0;
	typedef TM mesh_type;
	template<typename T> using field_value_type = T;
	typedef Real weight_type;
	typedef Real scatter_weight_type;
	typedef Real gather_weight_type;
};
template<typename TM>
struct Geometry<TM, 1>
{
	static const int IForm = 1;
	typedef TM mesh_type;
	template<typename T> using field_value_type = nTuple<3,T>;
	typedef nTuple<3, Real> weight_type;
	typedef nTuple<3, Real> scatter_weight_type;
	typedef nTuple<3, Real> gather_weight_type;

};
template<typename TM>
struct Geometry<TM, 2>
{

	static const int IForm = 2;
	typedef TM mesh_type;
	template<typename T> using field_value_type = nTuple<3,T>;
	typedef nTuple<3, Real> weight_type;
	typedef nTuple<3, Real> scatter_weight_type;
	typedef nTuple<3, Real> gather_weight_type;

};
template<typename TM>
struct Geometry<TM, 3>
{

	static const int IForm = 3;
	typedef TM mesh_type;
	template<typename T> using field_value_type = T;
	typedef Real weight_type;
	typedef Real scatter_weight_type;
	typedef Real gather_weight_type;

};

}  // namespace simpla

#endif /* GEOMETRY_H_ */
