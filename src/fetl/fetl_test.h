/*
 * fetl_test.h
 *
 *  Created on: 2014年2月20日
 *      Author: salmon
 */

#ifndef FETL_TEST_H_
#define FETL_TEST_H_
#include <gtest/gtest.h>

#include "fetl.h"
#include "../utilities/log.h"
#include "../utilities/pretty_stream.h"

using namespace simpla;

#ifndef TMESH
#include "../mesh/octree_forest.h"
#include "../mesh/geometry_cartesian.h"
#include "../mesh/mesh_rectangle.h"

typedef Mesh<CartesianGeometry<OcForest<Real>>> TMesh;
#else
typedef TMESH TMesh;
#endif

class TestFETL: public testing::TestWithParam<
        std::tuple<typename TMesh::coordinates_type, typename TMesh::coordinates_type, nTuple<TMesh::NDIMS, size_t> > >
{

protected:
	virtual void SetUp()
	{
		auto param = GetParam();

		xmin = std::get<0>(param);

		xmax = std::get<1>(param);

		dims = std::get<2>(param);

		mesh.SetExtents(xmin, xmax, dims);

		SetDefaultValue(&default_value);

		for (int i = 0; i < NDIMS; ++i)
		{
			if (xmax[i] - xmin[i] < EPSILON || dims[i] <= 1)
				K[i] = 0.0;
		}
	}
public:

	typedef TMesh mesh_type;
	typedef Real value_type;
	typedef mesh_type::scalar_type scalar_type;
	typedef mesh_type::iterator iterator;
	typedef mesh_type::coordinates_type coordinates_type;

	mesh_type mesh;

	static constexpr unsigned int NDIMS = mesh_type::NDIMS;

	nTuple<NDIMS, Real> xmin;

	nTuple<NDIMS, Real> xmax;

	nTuple<NDIMS, size_t> dims;

	static constexpr double PI = 3.141592653589793;

	nTuple<3, Real> K = { 2.0 * PI, 3.0 * PI, 4.0 * PI }; // @NOTE must   k = n TWOPI, period condition

	value_type default_value;

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

};



#endif /* FETL_TEST_H_ */
