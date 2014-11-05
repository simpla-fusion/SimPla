/*
 * field_test_common.h
 *
 *  Created on: 2014年11月5日
 *      Author: salmon
 */

#ifndef CORE_FIELD_FIELD_TEST_COMMON_H_
#define CORE_FIELD_FIELD_TEST_COMMON_H_

#include <gtest/gtest.h>
#include <tuple>

#include "../utilities/log.h"
#include "../utilities/ntuple.h"
#include "../utilities/pretty_stream.h"
#include "../io/data_stream.h"
#include "field.h"
#include "save_field.h"
#include "update_ghosts_field.h"

using namespace simpla;

#ifndef TMESH
#include "../manifold/manifold.h"
#include "../manifold/geometry/cartesian.h"
#include "../manifold/topology/structured.h"
#include "../manifold/diff_scheme/fdm.h"
#include "../manifold/interpolator/interpolator.h"

typedef Manifold<CartesianCoordinates<StructuredMesh, CARTESIAN_ZAXIS>,
		FiniteDiffMethod, InterpolatorLinear> TManifold;

#else
typedef TMESH TManifold;
#endif

class FETLTest: public testing::TestWithParam<
		std::tuple<nTuple<Real, 3>, nTuple<Real, 3>, nTuple<size_t, 3>,
				nTuple<Real, 3>> >
{

protected:
	void SetUp()
	{
		LOGGER.set_stdout_visable_level(LOG_VERBOSE);

		std::tie(xmin, xmax, dims, K_real) = GetParam();

		K_imag = 0;

		SetDefaultValue(&one);

		for (int i = 0; i < ndims; ++i)
		{
			if (dims[i] <= 1 || (xmax[i] <= xmin[i]))
			{
				dims[i] = 1;
				K_real[i] = 0.0;
				xmax[i] = xmin[i];
			}
		}

		manifold = make_manifold<manifold_type>();

		manifold->dimensions(dims);
		manifold->extents(xmin, xmax);
		manifold->update();

		Vec3 dx = manifold->dx();

		error = 2.0 * std::pow(inner_product(K_real, dx), 2.0);

	}

	void TearDown()
	{
		std::shared_ptr<manifold_type>(nullptr).swap(manifold);
	}
public:

	typedef TManifold manifold_type;
#ifndef VALUE_TYPE
	typedef Real value_type;
#else
	typedef VALUE_TYPE value_type;
#endif

	typedef typename manifold_type::scalar_type scalar_type;
	typedef typename manifold_type::iterator iterator;
	typedef typename manifold_type::coordinates_type coordinates_type;

	std::shared_ptr<manifold_type> manifold;

	static constexpr size_t ndims = manifold_type::ndims;

	nTuple<Real, 3> xmin;

	nTuple<Real, 3> xmax;

	nTuple<size_t, 3> dims;

	nTuple<Real, 3> K_real;	// @NOTE must   k = n TWOPI, period condition

	nTuple<scalar_type, 3> K_imag;

	value_type one;

	Real error;

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

	template<size_t N, typename T>
	void SetDefaultValue(nTuple<T, N>* v)
	{
		for (int i = 0; i < N; ++i)
		{
			(*v)[i] = i;
		}
	}

	virtual ~FETLTest()
	{

	}

};

#endif /* CORE_FIELD_FIELD_TEST_COMMON_H_ */
