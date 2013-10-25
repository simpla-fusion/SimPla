/*
 * test_field.cpp
 *
 *  Created on: 2012-1-13
 *      Author: salmon
 */

#include <gtest/gtest.h>
#include "utilities/log.h"

#include "mesh/uniform_rect.h"
#include "fetl.h"
#include "fetl/primitives.h"
#include "fetl/ntuple.h"
#include "fetl/expression.h"
#include "fetl/vector_calculus.h"
#include "fetl/geometry.h"
#include "physics/constants.h"
using namespace simpla;

DEFINE_FIELDS(UniformRectMesh)

template<typename TF>
class TestFETLBasicArithmetic: public testing::Test
{
protected:
	virtual void SetUp()
	{
		Log::Verbose(10);

		mesh.dt = 1.0;
		mesh.xmin[0] = 0;
		mesh.xmin[1] = 0;
		mesh.xmin[2] = 0;
		mesh.xmax[0] = 1.0;
		mesh.xmax[1] = 1.0;
		mesh.xmax[2] = 1.0;
		mesh.dims[0] = 20;
		mesh.dims[1] = 30;
		mesh.dims[2] = 40;
		mesh.gw[0] = 2;
		mesh.gw[1] = 2;
		mesh.gw[2] = 2;

		mesh.Init();

	}
public:
	typedef TF FieldType;

	Mesh mesh;

};

typedef testing::Types<RZeroForm, ROneForm, RTwoForm, RThreeForm, CZeroForm,
		COneForm, CTwoForm, CThreeForm, VecZeroForm, VecOneForm, VecTwoForm
//		,VecThreeForm
//		,CVecZeroForm, CVecOneForm, CVecTwoForm,CVecThreeForm
> AllFieldTypes;

//, VecThreeForm

// test arithmetic.h
TYPED_TEST_CASE(TestFETLBasicArithmetic, AllFieldTypes);

TYPED_TEST(TestFETLBasicArithmetic,create_write_read){
{

	Log::Verbose(10);

	Mesh const & mesh = TestFixture::mesh;

	typename TestFixture::FieldType f( mesh );

	typename TestFixture::FieldType::value_type a; a= 1.0;

	size_t num= f.get_num_of_elements();

	for (size_t s = 0; s < num; ++s)
	{
		f[s] = a*static_cast<double>(s);
	}

	for (size_t s = 0; s<num; ++s)
	{
		typename TestFixture::FieldType::value_type res;
		res=a*static_cast<Real>(s);
		ASSERT_EQ(res,f[s])<<"idx=" << s;
	}

}
}

TYPED_TEST(TestFETLBasicArithmetic,assign){
{
	typename TestFixture::FieldType::geometry_type geometry(TestFixture::mesh);

	typename TestFixture::FieldType f1(TestFixture::mesh),f2(TestFixture::mesh);

	typedef typename TestFixture::FieldType::value_type value_type;

	size_t num= f1.get_num_of_elements();

	value_type a; a = 3.0;

	std::fill(f2.begin(),f2.end(), a);

	for (size_t s = 0; s<num; ++s)
	{
		ASSERT_EQ(a,f2[s])<<"idx="<< s;
	}

	for (size_t s = 0; s<num; ++s)
	{
		f1[s]=a*static_cast<Real>(s);
	}

	f2 = f1;

	for (auto s = geometry.get_center_elements_begin( );
			s!=geometry.get_center_elements_end( ); ++s)
	{
		typename TestFixture::FieldType::value_type res;
		res=a*static_cast<Real>(*s);
		ASSERT_EQ( res,f2[*s])<<"idx="<< *s;
	}
}
}
TYPED_TEST(TestFETLBasicArithmetic, constant_real){
{
	typename TestFixture::FieldType::geometry_type geometry(TestFixture::mesh);

	typename TestFixture::FieldType f1( TestFixture::mesh),f2(TestFixture::mesh),f3(TestFixture::mesh);

	size_t num=geometry.get_num_of_elements( );

	Real a,b,c;
	a=1.0,b=-2.0,c=3.0;

	typedef typename TestFixture::FieldType::value_type value_type;

	value_type va,vb;

	va=2.0;vb=3.0;

	std::fill(f1.begin(),f1.end(), va);
	std::fill(f2.begin(),f2.end(), vb);

	f3 = - f1*2.0 + f2 * c -f1/b;

	for (auto s = geometry.get_center_elements_begin( );
			s!=geometry.get_center_elements_end( ); ++s)
	{
		value_type res;
		res= - f1[*s]*2.0 + f2[*s] *c -f1[*s]/b
		;
		ASSERT_EQ( res, f3[*s]) << *s;
	}
}
}

TYPED_TEST(TestFETLBasicArithmetic, scalar_field){
{
	//FIXME  should test with non-uniform field

	typename TestFixture::FieldType f1( TestFixture::mesh),f2(TestFixture::mesh),
	f3(TestFixture::mesh),f4(TestFixture::mesh);

	RScalarField a(TestFixture::mesh),b(TestFixture::mesh),c(TestFixture::mesh);

	std::fill(a.begin(),a.end(), 1.0);
	std::fill(b.begin(),b.end(), 3.0);
	std::fill(c.begin(),c.end(), 5.0);

	size_t count=0;

	for (auto s = a.get_center_elements_begin( );
			s!=a.get_center_elements_end( ); ++s)
	{

		if( 1.0 == a[*s] && 3.0== b[*s] && 5.0==c[*s])
		{
			++count;
		}
	}
	EXPECT_EQ(a.get_num_of_center_elements(),count);

	typename TestFixture::FieldType::value_type va,vb,vc;

	va=2.0;
	vb=3.0;
	vc=5.0;

	f4= -(f1^a)-f2/b +f3*c;
//	Plus( Minus(Negate(Wedge(f1,a)),Divides(f2,b)),Multiplies(f3,c) )
	;
	/**           (+)
	 *           /   \
	 *         (-)    (*)
	 *        /   \    | \
	 *      (^)    (/) f1 c
	 *     /  \   /  \
	 *-f1      a f2   b
	 *
	 * */
	count =0;

	size_t num_of_comp=f3.get_num_of_comp();

	for (auto s = f3.get_center_elements_begin( );
			s!=f3.get_center_elements_end( ); ++s)
	{
		typename TestFixture::FieldType::value_type res;
		res=
		-f1[*s]*a[*s/num_of_comp]
		-f2[*s]/b[*s/num_of_comp]
		+f3[*s]*c[*s/num_of_comp]
		;

		if(res==f4[*s])
		{
			++count;
		}

//		EXPECT_EQ(res,f4[*s])<<*s
//		<<" "<<num_of_comp
//		<<" "<<f1[*s]
//		<<" "<<f2[*s]
//		<<" "<<f3[*s]
//		<<" "<<a[*s/num_of_comp]
//		<<" "<<b[*s/num_of_comp]
//		<<" "<<c[*s/num_of_comp]
//		;

	}
	EXPECT_EQ(f3.get_num_of_center_elements(),count);

}
}
//// test vector_calculus.h
////template<typename T>
////class TestFETLVecAlgegbra: public testing::Test
////{
////protected:
////	virtual void SetUp()
////	{
////		mesh.dt = 1.0;
////		mesh.xmin[0] = 0;
////		mesh.xmin[1] = 0;
////		mesh.xmin[2] = 0;
////		mesh.xmax[0] = 1.0;
////		mesh.xmax[1] = 1.0;
////		mesh.xmax[2] = 1.0;
////		mesh.dims[0] = 20;
////		mesh.dims[1] = 30;
////		mesh.dims[2] = 40;
////		mesh.gw[0] = 2;
////		mesh.gw[1] = 2;
////		mesh.gw[2] = 2;
////
////		mesh.Init();
////	}
////public:
////	mesh mesh;
////	typedef Field<Geometry<mesh, 0>, Array<T> > ScalarField;
////	typedef Field<Geometry<mesh, 0>, Array<nTuple<3, T> > > VectorField;
////};
////
////typedef testing::Types<double, Complex> VecFieldTypes;
////
////TYPED_TEST_CASE(TestFETLVecAlgegbra, VecFieldTypes);
////
////TYPED_TEST(TestFETLVecAlgegbra,constant_vector){
////{
////	const mesh& mesh = TestFixture::mesh;
////
////	Geometry<mesh, 0> geometry(TestFixture::mesh);
////
////	Vec3 vc1 =
////	{	1.0, 2.0, 3.0};
////	Vec3 vc2 =
////	{	-1.0, 4.0, 2.0};
////
////	Vec3 res_vec;
////
////	res_vec = Cross(vc1, vc2);
////
////	Real res_scalar;
////	res_scalar = Dot(vc1, vc2);
////
////	typename TestFixture::ScalarField res_scalar_field(mesh);
////
////	typename TestFixture::VectorField va(mesh), vb(mesh), res_vector_field(
////			mesh);
////
////	va = vc2;
////
////	res_scalar_field = Dot(vc1, va);
////
////	res_vector_field = Cross(vc1, va);
////
////	size_t num_of_comp = geometry.get_num_of_comp( );
////
////	for (auto s = geometry.get_center_elements_begin( );
////			s != geometry.get_center_elements_end( ); ++s)
////	{
////		EXPECT_EQ(res_scalar, res_scalar_field[(*s)] )<< "idx=" <<(*s)<< " | "
////		<<va[(*s)] <<" | "<< vc1 << " | "<< res_scalar_field[(*s)]
////		;
////
////		EXPECT_EQ(res_vec, (res_vector_field[(*s)])) << "idx=" <<(*s)<< " | "
////		<<va[(*s)] <<" | "<< vc1 << " | "<< res_vector_field[(*s)]
////		;
////	}
////
////}
////}
////TYPED_TEST(TestFETLVecAlgegbra,vector_field){
////{
////	using namespace space3;
////	//FIXME  should test with non-uniform field
////
////	mesh const & mesh = TestFixture::mesh;
////
////	Geometry<mesh, 0> geometry(mesh);
////
////	Vec3 vc1 =
////	{	1.0, 2.0, 3.0};
////	Vec3 vc2 =
////	{	-1.0, 4.0, 2.0};
////
////	Vec3 res_vec;
////
////	res_vec=Cross(vc1,vc2);
////
////	Real res_scalar= Dot(vc1,vc2);
////
////	typename TestFixture::VectorField va(mesh), vb(mesh);
////
////	typename TestFixture::VectorField res_vector_field(mesh);
////	typename TestFixture::ScalarField res_scalar_field(mesh);
////
////	va = vc1;
//////
//////	vb = vc2;
//////
//////	res_scalar_field = Dot( vb ,va);
//////
//////	res_vector_field = Cross(va , vb);
//////
//////	size_t num_of_comp =geometry.get_num_of_comp( );
//////
//////	for (auto s = geometry.get_center_elements_begin( );
//////			s!=geometry.get_center_elements_end( ); ++s)
//////	{
//////		ASSERT_EQ(res_scalar, res_scalar_field[(*s)] ) << "idx=" <<(*s);
//////
//////		ASSERT_EQ(res_vec, (res_vector_field[(*s)])) << "idx=" <<(*s);
//////
//////	}
////
////}
////}
////TYPED_TEST(TestFETLVecAlgegbra,complex_vector_field){
////{
////	//FIXME  should test with non-uniform field
////
////	mesh const & mesh = TestFixture::mesh;
////
////	Vec3 vc1 =
////	{	1.0,2.0,3.0};
////
////	CVec3 vc2 =
////	{
////		Complex( 0.0,0.0) ,
////		Complex( -0.2,0.2) ,
////		Complex( 3.0,1.3)};
////
////	Complex res_scalar= Dot(vc2,vc1);
////
////	CVec3 res_vec;
////
////	res_vec=Cross(vc1,vc2);
////
////	typename TestFixture::VectorField va(mesh);
////
////	typename TestFixture::CVectorField vb(mesh);
////
////	va = vc1;
////
////	vb = vc2;
////
////	typename TestFixture::CVectorField res_vector_field(mesh);
////	typename TestFixture::CScalarField res_scalar_field(mesh);
////
////	res_scalar_field = Dot(vb, va);
////
////	res_vector_field = Cross(va, vb);
////
////	size_t num_of_comp =mesh.get_num_of_comp(TestFixture::VectorField::IForm);
////
////	for (typename mesh::const_iterator s = mesh.get_center_elements_begin(TestFixture::VectorField::IForm);
////			s!=mesh.get_center_elements_end(TestFixture::VectorField::IForm); ++s)
////	{
////		ASSERT_EQ(res_scalar, res_scalar_field[(*s)] ) << "idx=" <<(*s);
////
////		ASSERT_EQ(res_vec, (res_vector_field[(*s)])) << "idx=" <<(*s);
////
////	}
////
////}
////}

template<typename TP>
class TestFETLDiffCalcuate: public testing::Test
{

protected:
	virtual void SetUp()
	{
		mesh.dt = 1.0;
		mesh.xmin[0] = 0;
		mesh.xmin[1] = 0;
		mesh.xmin[2] = 0;
		mesh.xmax[0] = 1.0;
		mesh.xmax[1] = 1.0;
		mesh.xmax[2] = 1.0;
		mesh.dims[0] = 20;
		mesh.dims[1] = 30;
		mesh.dims[2] = 40;
		mesh.gw[0] = 2;
		mesh.gw[1] = 2;
		mesh.gw[2] = 2;

		mesh.Init();

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

	void Setvalue_type(double *v)
	{
		*v = 1.0;
	}

	void Setvalue_type(Complex *v)
	{
		*v = Complex(1.0, 2.0);
	}

	template<int N, typename TV>
	void Setvalue_type(nTuple<N, TV> *v)
	{
		for (size_t i = 0; i < N; ++i)
		{
			Setvalue_type(&((*v)[i]));
		}
	}
};

typedef testing::Types<double, Complex, nTuple<3, double>,
		nTuple<3, nTuple<3, double> > > PrimitiveTypes;

TYPED_TEST_CASE(TestFETLDiffCalcuate, PrimitiveTypes);

TYPED_TEST(TestFETLDiffCalcuate, curl_grad_eq_0){
{
	Mesh const & mesh = TestFixture::mesh;

	typename TestFixture::value_type v;

	TestFixture::Setvalue_type(&v);

	typename TestFixture::TZeroForm sf(mesh);
	typename TestFixture::TOneForm vf1(mesh);
	typename TestFixture::TTwoForm vf2(mesh);

	size_t num = mesh.get_num_of_vertex() * mesh.get_num_of_comp(0);

	for (size_t i = 0; i < mesh.dims[0]; ++i)
	for (size_t j = 0; j < mesh.dims[1]; ++j)
	for (size_t k = 0; k < mesh.dims[2]; ++k)
	{
		size_t s = mesh.get_cell_num(i, j, k);
		sf[s] = static_cast<double>(s)*v;
	}

	vf1 = Grad(sf);

	vf2 = Curl(Grad(sf));

	Geometry<Mesh,1> geometry1(mesh);

//	for (auto s = geometry1.get_center_elements_begin(); s != geometry1.get_center_elements_end( ); ++s)
//	{
//
//		ASSERT_NE(0.0,abs(vf1[(*s)])) << "idx=" << *s;
//	}
	Geometry<Mesh,2> geometry2(mesh);
	for (auto s = geometry2.get_center_elements_begin( ); s != geometry2.get_center_elements_end( ); ++s)
	{
		ASSERT_DOUBLE_EQ(0.0,abs(vf2[(*s)])) << "idx=" << *s;
	}
}
}

TYPED_TEST(TestFETLDiffCalcuate, div_curl_eq_0){
{

	Mesh const & mesh = TestFixture::mesh;

	typename TestFixture::TZeroForm sf(mesh);
	typename TestFixture::TOneForm vf1(mesh);
	typename TestFixture::TTwoForm vf2(mesh);

	typename TestFixture::value_type v;

	TestFixture::Setvalue_type(&v);

	for (size_t s = 0; s < vf2.get_num_of_elements(); ++s)
	{
		vf2[s] = v * (s + 1.0);
	}

	vf1 = Curl(vf2);
	sf = Diverge( Curl(vf2));

	for (auto s = sf.get_center_elements_begin();
			s != sf.get_center_elements_end(); ++s)
	{
//		ASSERT_NE(0.0,abs(vf1[(*s)])) << "idx=" << *s;
		ASSERT_DOUBLE_EQ(0.0,abs(sf[(*s)])) << "idx=" << *s;
	}
}
}

//////class TestFETLPerformance: public testing::TestWithParam<size_t>
//////{
//////protected:
//////	virtual void SetUp()
//////	{
//////		size_t ratio = GetParam();
//////
//////		mesh.dt = 1.0;
//////		mesh.xmin[0] = 0;
//////		mesh.xmin[1] = 0;
//////		mesh.xmin[2] = 0;
//////		mesh.xmax[0] = 1.0;
//////		mesh.xmax[1] = 1.0;
//////		mesh.xmax[2] = 1.0;
//////		mesh.dims[0] = 2 * ratio;
//////		mesh.dims[1] = 3 * ratio;
//////		mesh.dims[2] = 4 * ratio;
//////		mesh.gw[0] = 2;
//////		mesh.gw[1] = 2;
//////		mesh.gw[2] = 2;
//////
//////		mesh.Init();
//////
//////	}
//////public:
//////	mesh mesh;
//////
//////	static const int NDIMS = 3;
//////
//////	double RelativeError2(double a, double b)
//////	{
//////		return pow(2.0 * (a - b) / (a + b), 2.0);
//////	}
//////};
//////
//////INSTANTIATE_TEST_CASE_P(TestPerformance, TestFETLPerformance,
//////		testing::value_types(10, 20, 50));
//////
//////TEST_P(TestFETLPerformance, error_analyze)
//////{
//////
//////	static const double epsilon = 0.01;
//////
//////	ZeroForm sf(mesh), res_sf(mesh);
//////
//////	OneForm res_vf(mesh);
//////
//////	size_t size = mesh.get_num_of_vertex() * mesh.get_num_of_comp(IZeroForm);
//////
//////	Real k0 = 1.0 * TWOPI, k1 = 2.0 * TWOPI, k2 = 3.0 * TWOPI;
//////
//////#pragma omp parallel for
//////	for (size_t i = 0; i < mesh.dims[0]; ++i)
//////	{
//////		for (size_t j = 0; j < mesh.dims[1]; ++j)
//////			for (size_t k = 0; k < mesh.dims[2]; ++k)
//////			{
//////				size_t s = mesh.get_cell_num(i, j, k);
//////				sf[s] = sin(
//////						k0 * i * mesh.dx[0] + k1 * j * mesh.dx[1]
//////								+ k2 * k * mesh.dx[2]);
//////			}
//////	}
//////	res_vf = Grad(sf);
//////
//////	res_sf = Diverge(Grad(sf));
//////
//////	Real rex = 0.0;
//////	Real rey = 0.0;
//////	Real rez = 0.0;
//////	Real re2 = 0.0;
//////
//////#pragma omp parallel for  reduction(+:rex,rey,rez,re2)
//////	for (size_t i = mesh.gw[0]; i < mesh.dims[0] - mesh.gw[0]; ++i)
//////		for (size_t j = mesh.gw[1]; j < mesh.dims[1] - mesh.gw[1]; ++j)
//////			for (size_t k = mesh.gw[2]; k < mesh.dims[2] - mesh.gw[2]; ++k)
//////			{
//////				size_t s = mesh.get_cell_num(i, j, k);
//////
//////				Real alpha = k0 * i * mesh.dx[0] + k1 * j * mesh.dx[1]
//////						+ k2 * k * mesh.dx[2];
//////
//////				rex += RelativeError2(k0 * cos(alpha + k0 * 0.5 * mesh.dx[0]),
//////						res_vf[s * 3 + 0]);
//////				rey += RelativeError2(k1 * cos(alpha + k1 * 0.5 * mesh.dx[1]),
//////						res_vf[s * 3 + 1]);
//////				rez += RelativeError2(k2 * cos(alpha + k2 * 0.5 * mesh.dx[2]),
//////						res_vf[s * 3 + 2]);
//////
//////				re2 += RelativeError2(
//////						-(k0 * k0 + k1 * k1 + k2 * k2) * sin(alpha), res_sf[s]);
//////			}
//////
//////	rex /= static_cast<Real>(size);
//////	rey /= static_cast<Real>(size);
//////	rez /= static_cast<Real>(size);
//////	re2 /= static_cast<Real>(size);
//////
//////	EXPECT_LE(rex, pow(k0*mesh.dx[0],2));
//////	EXPECT_LE(rey, pow(k1*mesh.dx[1],2));
//////	EXPECT_LE(rez, pow(k2*mesh.dx[2],2));
//////	EXPECT_LE( re2,
//////			(pow(k0*mesh.dx[0],2)+pow(k1*mesh.dx[1],2)+pow(k2*mesh.dx[2],2))/3.0);
//////
//////}
//
