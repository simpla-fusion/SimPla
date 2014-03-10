/*
 * fetl_test3.cpp
 *
 *  Created on: 2013年12月17日
 *      Author: salmon
 */

#include "fetl_test.h"
using namespace simpla;
DEFINE_FIELDS(DEF_MESH)

template<typename TP>
class TestFETLDiffCalcuate: public testing::Test
{

protected:
	virtual void SetUp()
	{
		nTuple<3, Real> xmin = { 0, 0, 0 };
		nTuple<3, Real> xmax = { 1, 1, 1 };
		mesh.SetExtent(xmin, xmax);

		nTuple<3, size_t> dims = { 20, 0, 0 };
		mesh.SetDimensions(dims);

		mesh.Update();

	}
public:

	Mesh mesh;

	typedef TP value_type;
	typedef Field<Mesh, VERTEX, value_type> TZeroForm;
	typedef Field<Mesh, EDGE, value_type> TOneForm;
	typedef Field<Mesh, FACE, value_type> TTwoForm;

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

typedef testing::Types<double
//		, Complex, nTuple<3, double>, nTuple<3, nTuple<3, double>>
> PrimitiveTypes;

TYPED_TEST_CASE(TestFETLDiffCalcuate, PrimitiveTypes);

TYPED_TEST(TestFETLDiffCalcuate, curl_grad_eq_0){
{
	Mesh const & mesh = TestFixture::mesh;

	typename TestFixture::value_type v;

	TestFixture::SetValue(&v);

	typename TestFixture::TZeroForm sf(mesh,v);
	typename TestFixture::TOneForm vf1(mesh,v);
	typename TestFixture::TTwoForm vf2(mesh,v);
	typename TestFixture::TTwoForm vf2b(mesh,v);

	std::mt19937 gen;
	std::uniform_real_distribution<Real> uniform_dist(0, 1.0);

	Real m=0.0;

	for(auto & p:sf)
	{
		p = uniform_dist(gen);
		m+= abs(p);
	}

	m/=sf.size();

	LOG_CMD(vf1=Grad(sf));
	LOG_CMD(vf2 = Curl(vf1));
	LOG_CMD(vf2b = Curl(Grad(sf)));

	size_t count=0;
	Real relative_error=0;

	mesh.Traversal<FACE>(
			[&](typename TestFixture::TTwoForm::index_type s)
			{	relative_error+=abs(vf2[s]);

				if(abs(vf2[s])>1.0e-10)
				{
					CHECK(vf2[s]);
					++count;
				}
			}

	);
	relative_error=relative_error/m;

	CHECK(relative_error);

	EXPECT_GT(1.0e-8,relative_error);
	ASSERT_EQ(0,count)<< "number of non-zero points =" << count;

	count =0;
	relative_error=0.0;

	mesh.Traversal<FACE>(
			[&](typename TestFixture::TTwoForm::index_type s)
			{	relative_error+=abs(vf2b[s]);

				if(abs(vf2b[s])>1.0e-10)
				{
					CHECK(vf2b[s]);
					++count;
				}
			}
	);
	relative_error=relative_error/m;

	CHECK(relative_error);

	EXPECT_GT(1.0e-8,relative_error);
	ASSERT_EQ(0,count)<< "number of non-zero points =" << count;

}
}

TYPED_TEST(TestFETLDiffCalcuate, div_curl_eq_0){
{

	Mesh const & mesh = TestFixture::mesh;

	typename TestFixture::value_type v;

	v=1.0;

	typename TestFixture::TZeroForm sf1(mesh);
	typename TestFixture::TZeroForm sf2(mesh);
	typename TestFixture::TOneForm vf1(mesh);
	typename TestFixture::TTwoForm vf2(mesh,v);

	std::mt19937 gen;
	std::uniform_real_distribution<Real> uniform_dist(0, 1.0);

	for(auto &p:vf2)
	{
		p*= uniform_dist(gen);
	}

	LOG_CMD(vf1 = Curl(vf2));
	LOG_CMD(sf1 = Diverge( vf1));
	LOG_CMD(sf2 = Diverge( Curl(vf2)));

	size_t count=0;

	Real m=0.0;

	for(auto const &p:vf2)
	{
		m+=abs(p);
	}
	m/=vf2.size();

	Real relative_error=0;
	size_t num=0;
	mesh.Traversal<VERTEX>(
			[&](typename TestFixture::TZeroForm::index_type s)
			{
				relative_error+=abs(sf1[s]);
				if(abs(sf1[s])>1.0e-10*m)
				{
					CHECK(sf1[s]);
					++count;
				}
			}
	);

	relative_error=relative_error/m;
	CHECK(relative_error);
	EXPECT_GT(1.0e-8,relative_error);
	ASSERT_DOUBLE_EQ(0,count)<< "number of non-zero points =" << count;

	count =0;
	relative_error=0.0;
	mesh.Traversal<VERTEX>(
			[&](typename TestFixture::TZeroForm::index_type s)
			{
				relative_error+=abs(sf2[s]);
				if(abs(sf2[s])>1.0e-10*m)
				{
					CHECK(sf2[s]);
					++count;
				}
			}
	);

	relative_error=relative_error/m;
	CHECK(relative_error);
	EXPECT_GT(1.0e-8,relative_error);
	ASSERT_DOUBLE_EQ(0,count)<< "number of non-zero points =" << count;

}
}

