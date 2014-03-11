/*
 * fetl_test2.cpp
 *
 *  Created on: 2013年12月17日
 *      Author: salmon
 */

#include "fetl_test.h"
using namespace simpla;
DEFINE_FIELDS(DEF_MESH)

template<typename T>
class TestFETLVecAlgegbra: public testing::Test
{
protected:
	virtual void SetUp()
	{
		nTuple<3, Real> xmin = { 0, 0, 0 };
		nTuple<3, Real> xmax = { 1, 1, 1 };
		mesh.SetExtent(xmin, xmax);

		nTuple<3, size_t> dims = { 20, 1, 1 };
		mesh.SetDimensions(dims);

		mesh.Update();

	}
public:
	Mesh mesh;
	typedef T value_type;
	typedef nTuple<3, value_type> Vec3;
	typedef Field<Mesh, VERTEX, T> ScalarField;
	typedef Field<Mesh, VERTEX, nTuple<3, T> > VectorField;
};

typedef testing::Types<double
//		, Complex
> VecFieldTypes;

TYPED_TEST_CASE(TestFETLVecAlgegbra, VecFieldTypes);

TYPED_TEST(TestFETLVecAlgegbra, vector_arithmetic){
{
	//FIXME  should test with non-uniform field

	typedef typename TestFixture::value_type value_type;
	Mesh const & mesh=TestFixture::mesh;

	Field<Mesh,VERTEX,value_type> f0( mesh);
	Field<Mesh,EDGE,value_type> f1a( mesh),f1b(mesh);
	Field<Mesh,FACE,value_type> f2a( mesh),f2b(mesh);
	Field<Mesh,VOLUME,value_type> f3( mesh);

	Real ra=1.0,rb=10.0,rc=100.0;
	value_type va,vb,vc;

	va=ra;
	vb=rb;
	vc=rc;

	f0.Init();
	f1a.Init();

	f1b.Init();
	f2a.Init();
	f2b.Init();
	f3.Init();

	size_t count=0;

	std::mt19937 gen;
	std::uniform_real_distribution<Real> uniform_dist(0, 1.0);

	for(auto & v:f1a)
	{
		v=va *uniform_dist(gen);
	}
	for(auto & v:f2a)
	{
		v=vb *uniform_dist(gen);
	}

	for(auto & v:f3)
	{
		v=vc *uniform_dist(gen);
	}

	LOG_CMD(f2b=Cross(f1a,f1b));
	LOG_CMD(f3=Dot(f1a,f2a));
	LOG_CMD(f3=Dot(f2a,f1a));
	LOG_CMD(f3=InnerProduct(f2a,f2a));
	LOG_CMD(f3=InnerProduct(f1a,f1a));

	LOG_CMD(f0=Wedge(f0,f0));
	LOG_CMD(f1b=Wedge(f0,f1a));
	LOG_CMD(f1b=Wedge(f1a,f0));
	LOG_CMD(f2b=Wedge(f0,f2a));
	LOG_CMD(f2b=Wedge(f2a,f0));
	LOG_CMD(f3=Wedge(f0,f3));
	LOG_CMD(f3=Wedge(f3,f0));

	LOG_CMD(f2a=Wedge(f1a,f1b));
	LOG_CMD(f3=Wedge(f1a,f2b));
	LOG_CMD(f3=Wedge(f2a,f1b));

}
}

