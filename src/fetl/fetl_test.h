/*
 * fetl_test.h
 *
 *  Created on: 2014年2月20日
 *      Author: salmon
 */

#ifndef FETL_TEST_H_
#define FETL_TEST_H_

#include "fetl.h"
#include "../utilities/log.h"
#include "../utilities/pretty_stream.h"

#include "../mesh/octree_forest.h"
#include "../mesh/mesh.h"
#include "../mesh/geometry_cylindrical.h"
#include "../mesh/geometry_euclidean.h"
using namespace simpla;

template<typename TM, typename TV = double, int ICase = 0>
struct TestFETLParam
{
	typedef TM mesh_type;
	typedef TV value_type;
	static constexpr int IForm = ICase / 100;

	static void SetUpMesh(mesh_type * mesh)
	{
	}

	static void SetDefaultValue(value_type * v)
	{
	}
};

template<typename T>
void SetDefaultValue(T* v)
{
	*v = 1;
}
template<typename T>
void SetDefaultValue(std::complex<T>* v)
{
	T r;
	SetDefaultValue(&r);
	*v = std::complex<T>();
}

template<int N, typename T>
void SetDefaultValue(nTuple<N, T>* v)
{
	for (int i = 0; i < N; ++i)
	{
		(*v)[i] = i;
	}
}
#endif /* FETL_TEST_H_ */
