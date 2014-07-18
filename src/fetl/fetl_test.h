/*
 * fetl_test.h
 *
 *  created on: 2014-2-20
 *      Author: salmon
 */

#ifndef FETL_TEST_H_
#define FETL_TEST_H_
#include <gtest/gtest.h>

#include "fetl.h"
#include "../utilities/log.h"
#include "../utilities/pretty_stream.h"
#include "../io/data_stream.h"
using namespace simpla;

#ifndef TMESH
#include "../mesh/uniform_array.h"
#include "../mesh/geometry_cartesian.h"
#include "../mesh/mesh_rectangle.h"

typedef Mesh<CartesianGeometry<UniformArray>, false> TMesh;
#else
typedef TMESH TMesh;
#endif

class TestFETL: public testing::TestWithParam<
        std::tuple<typename TMesh::coordinates_type, typename TMesh::coordinates_type, nTuple<TMesh::NDIMS, size_t>,
                nTuple<TMesh::NDIMS, Real> > >
{

protected:
	void SetUp()
	{
		LOG_STREAM.set_stdout_visable_level(LOG_INFORM);
		auto param = GetParam();

		xmin = std::get<0>(param);

		xmax = std::get<1>(param);

		dims = std::get<2>(param);

		K_real = std::get<3>(param);

		SetDefaultValue(&default_value);

		for (int i = 0; i < NDIMS; ++i)
		{
			if (dims[i] <= 1 || (xmax[i] <= xmin[i]))
			{
				dims[i] = 1;
				K_real[i] = 0.0;
				xmax[i] = xmin[i];
			}
		}

		mesh.set_dimensions( dims);
		mesh.set_extents(xmin, xmax);

		mesh.Update();

		K_imag = mesh.k_imag;

		GLOBAL_DATA_STREAM.cd("MeshTest.h5:/");

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

	nTuple<3, Real> K_real; // @NOTE must   k = n TWOPI, period condition

	nTuple<3, scalar_type> K_imag;

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

	template<unsigned int N, typename T>
	void SetDefaultValue(nTuple<N, T>* v)
	{
		for (int i = 0; i < N; ++i)
		{
			(*v)[i] = i;
		}
	}

	virtual ~TestFETL()
	{

	}

};

#endif /* FETL_TEST_H_ */
