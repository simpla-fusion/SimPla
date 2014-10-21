/*
 * fetl_test.h
 *
 *  created on: 2014-2-20
 *      Author: salmon
 */

#ifndef FETL_TEST_H_
#define FETL_TEST_H_
#include <gtest/gtest.h>
#include <tuple>

using namespace simpla;

#include "../utilities/log.h"

#include "../manifold/manifold.h"
#include "../manifold/geometry/cartesian.h"
#include "../manifold/topology/structured.h"
#include "../manifold/diff_scheme/fdm.h"
#include "../manifold/interpolator/interpolator.h"

typedef Manifold<CartesianCoordinates<StructuredMesh, CARTESIAN_ZAXIS>,
		FiniteDiffMehtod, InterpolatorLinear> TMesh;

typedef nTuple<3, Real> coordiantes_type;

class TestFETL: public testing::TestWithParam<
		std::tuple<coordiantes_type, coordiantes_type,
				nTuple<TMesh::NDIMS, size_t>, nTuple<TMesh::NDIMS, Real> > >
{

protected:
	void SetUp()
	{
		LOGGER.set_stdout_visable_level(LOG_INFORM);
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

		mesh.dimensions(dims);
		mesh.extents(xmin, xmax);

		mesh.update();

	}
public:

	typedef TMesh manifold_type;
	typedef Real value_type;
	typedef typename manifold_type::scalar_type scalar_type;
	typedef typename manifold_type::iterator iterator;
	typedef typename manifold_type::coordinates_type coordinates_type;

	manifold_type mesh;

	static constexpr unsigned int NDIMS = manifold_type::NDIMS;

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

	template<unsigned int IFORM, typename TV>
	Field<Domain<manifold_type, IFORM>, TV> make_field() const
	{
		return std::move(Field<Domain<manifold_type, IFORM>, TV>(mesh));
	}

};

#endif /* FETL_TEST_H_ */
