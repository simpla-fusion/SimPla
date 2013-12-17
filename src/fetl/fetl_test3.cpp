/*
 * fetl_test3.cpp
 *
 *  Created on: 2013年12月17日
 *      Author: salmon
 */

#include <gtest/gtest.h>
#include <random>

#include "fetl.h"
#include "../utilities/log.h"
#include "../mesh/co_rect_mesh.h"
#include "../utilities/pretty_stream.h"

using namespace simpla;

DEFINE_FIELDS(CoRectMesh<>)

template<typename TP>
class TestFETLDiffCalcuate: public testing::Test
{

protected:
	virtual void SetUp()
	{
		mesh.dt_ = 1.0;
		mesh.xmin_[0] = 0;
		mesh.xmin_[1] = 0;
		mesh.xmin_[2] = 0;
		mesh.xmax_[0] = 1.0;
		mesh.xmax_[1] = 1.0;
		mesh.xmax_[2] = 1.0;
		mesh.dims_[0] = 20;
		mesh.dims_[1] = 30;
		mesh.dims_[2] = 40;

		mesh.Update();

	}
public:

	Mesh mesh;

	typedef TP value_type;
	typedef Field<Geometry<Mesh, 0>, value_type> TZeroForm;
	typedef Field<Geometry<Mesh, 1>, value_type> TOneForm;
	typedef Field<Geometry<Mesh, 2>, value_type> TTwoForm;

	double RelativeError(double a, double b)
	{
		return (2.0 * fabs((a - b) / (a + b)));
	}

	void SetValue(double *v)
	{
		*v = 1.0;
	}

	void SetValue(Complex *v)
	{
		*v = Complex(1.0, 2.0);
	}

	template<int N, typename TV>
	void SetValue(nTuple<N, TV> *v)
	{
		for (size_t i = 0; i < N; ++i)
		{
			SetValue(&((*v)[i]));
		}
	}
};

typedef testing::Types<double, Complex, nTuple<3, double>, nTuple<3, nTuple<3, double>> > PrimitiveTypes;

TYPED_TEST_CASE(TestFETLDiffCalcuate, PrimitiveTypes);

TYPED_TEST(TestFETLDiffCalcuate, curl_grad_eq_0){
{
	Mesh const & mesh = TestFixture::mesh;

	typename TestFixture::value_type v;

	TestFixture::SetValue(&v);

	typename TestFixture::TZeroForm sf(mesh,v);
	typename TestFixture::TTwoForm vf2(mesh,v);

	std::mt19937 gen;
	std::uniform_real_distribution<Real> uniform_dist(0, 1.0);

	Real m=0.0;

	for(auto & p:sf)
	{
		p = uniform_dist(gen);
		m+= abs(p);
	}

	m/=sf.size();

	vf2 = Curl(Grad(sf));

	size_t count=0;
	Real relative_error=0;
	mesh.ForEach(
			[&](typename TestFixture::TTwoForm::value_type const & u)
			{	relative_error+=abs(u);
				count+=( abs(u)>1.0e-10)?1:0;
			},
			vf2
	);
	relative_error=relative_error/m;
	EXPECT_GT(1.0e-8,relative_error);
	ASSERT_EQ(0,count)<< "number of non-zero points =" << count;

}
}

TYPED_TEST(TestFETLDiffCalcuate, div_curl_eq_0){
{

	Mesh const & mesh = TestFixture::mesh;

	typename TestFixture::value_type v;

	v=1.0;

	typename TestFixture::TZeroForm sf(mesh);
	typename TestFixture::TOneForm vf1(mesh);
	typename TestFixture::TTwoForm vf2(mesh,v);

	std::mt19937 gen;
	std::uniform_real_distribution<Real> uniform_dist(0, 1.0);

	for(auto &p:vf2)
	{
		p*= uniform_dist(gen);
	}

	vf1 = Curl(vf2);
	sf = Diverge( Curl(vf2));

	size_t count=0;

	Real m=0.0;

	for(auto const &p:vf2)
	{
		m+=abs(p);
	}
	m/=vf2.size();

	Real relative_error=0;
	size_t num=0;
	mesh.ForEach(
			[&](typename TestFixture::TZeroForm::value_type const &s)
			{
				relative_error+=abs(s);
				count+=( abs(s)>1.0e-10*m)?1:0;
			},sf
	);

	relative_error=relative_error/m;
	EXPECT_GT(1.0e-8,relative_error);
	ASSERT_DOUBLE_EQ(0,count)<< "number of non-zero points =" << count;

}
}

