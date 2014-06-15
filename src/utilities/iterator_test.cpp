/*
 * iterator_test.cpp
 *
 *  Created on: 2014年6月15日
 *      Author: salmon
 */

#include <gtest/gtest.h>
using namespace simpla;

class TestIterator: public testing::TestWithParam<nTuple<3, size_t> >
{
protected:
	virtual void SetUp()
	{
		GLOBAL_COMM.Init(0, nullptr);
		LOG_STREAM.SetStdOutVisableLevel(12);

		auto param = GetParam();

		xmin = std::get<0>(param);
		xmax = std::get<1>(param);
		dims = std::get<2>(param);

		mesh.SetExtents(xmin, xmax, dims);

		mesh.Decompose();

		cfg_str = "n0=function(x,y,z)"
				"  return (x-0.5)*(x-0.5)+(y-0.5)*(y-0.5)+(z-0.5)*(z-0.5) "
				" end "
				"ion={ Name=\"H\",Mass=1.0e-31,Charge=1.6021892E-19 ,PIC=500,Temperature=300 ,Density=n0"
				"}";

	}
public:

	typedef typename pool_type::mesh_type mesh_type;

	typedef typename mesh_type::scalar_type scalar_type;

	typedef typename mesh_type::iterator iterator;

	typedef typename mesh_type::coordinates_type coordinates_type;

	mesh_type mesh;

	nTuple<3, Real> xmin, xmax;

	nTuple<3, size_t> dims;

	std::string cfg_str;

	bool enable_sorting;

};
