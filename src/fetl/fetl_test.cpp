/*
 * test_field.cpp
 *
 *  Created on: 2012-1-13
 *      Author: salmon
 */

#include <gtest/gtest.h>
//#include "include/simpla_defs.h"
#include "grid/uniform_rect.h"
#include "fetl.h"
#include "fetl/primitives.h"
#include "fetl/ntuple.h"
#include "fetl/expression.h"
#include "fetl/vector_calculus.h"
#include "physics/constants.h"
using namespace simpla;

DEFINE_FIELDS( UniformRectGrid)

template<typename TF>
class TestFETLBasicArithmetic: public testing::Test
{
protected:
	virtual void SetUp()
	{
		Log::Verbose(10);

		grid.dt = 1.0;
		grid.xmin[0] = 0;
		grid.xmin[1] = 0;
		grid.xmin[2] = 0;
		grid.xmax[0] = 1.0;
		grid.xmax[1] = 1.0;
		grid.xmax[2] = 1.0;
		grid.dims[0] = 20;
		grid.dims[1] = 30;
		grid.dims[2] = 40;
		grid.gw[0] = 2;
		grid.gw[1] = 2;
		grid.gw[2] = 2;

		grid.Init();

	}
public:
	typedef TF FieldType;

	Grid grid;

};

typedef testing::Types<
//		RZeroForm, ROneForm, RTwoForm, RThreeForm, CZeroForm,
//		COneForm, CTwoForm, CThreeForm, VecZeroForm,
		VecOneForm
//, VecTwoForm
//		,VecThreeForm
//		,CVecZeroForm, CVecOneForm, CVecTwoForm,CVecThreeForm
> AllFieldTypes;

//, VecThreeForm

// test arithmetic.h
TYPED_TEST_CASE(TestFETLBasicArithmetic, AllFieldTypes);

TYPED_TEST(TestFETLBasicArithmetic,create_write_read){
{

	Log::Verbose(10);

	Grid const & grid = TestFixture::grid;

	typename TestFixture::FieldType f( grid );

	typename TestFixture::FieldType::ValueType a; a= 1.0;

	size_t num= f.get_num_of_elements();

	for (size_t s = 0; s < num; ++s)
	{
		f[s] = a*static_cast<double>(s);
	}

	for (size_t s = 0; s<num; ++s)
	{
		typename TestFixture::FieldType::ValueType res;
		res=a*static_cast<Real>(s);
		ASSERT_EQ(res,f[s])<<"idx=" << s;
	}

}
}

TYPED_TEST(TestFETLBasicArithmetic,assign){
{
	typename TestFixture::FieldType::GeometryType geometry(TestFixture::grid);

	typename TestFixture::FieldType f1(TestFixture::grid),f2(TestFixture::grid);

	typedef typename TestFixture::FieldType::ValueType ValueType;

	size_t num= f1.get_num_of_elements();

	ValueType a; a = 3.0;

	f2.fill( a);

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
		typename TestFixture::FieldType::ValueType res;
		res=a*static_cast<Real>(*s);
		ASSERT_EQ( res,f2[*s])<<"idx="<< *s;
	}
}
}
TYPED_TEST(TestFETLBasicArithmetic, constant_real){
{
	typename TestFixture::FieldType::GeometryType geometry(TestFixture::grid);

	typename TestFixture::FieldType f1( TestFixture::grid),f2(TestFixture::grid),f3(TestFixture::grid);

	size_t num=geometry.get_num_of_elements( );

	Real a,b,c;
	a=1.0,b=-2.0,c=3.0;

	typedef typename TestFixture::FieldType::ValueType ValueType;

	ValueType va,vb;

	va=2.0;vb=3.0;

	f1 .fill( va);
	f2 .fill( vb);
	f3 = - f1*2.0 + f2 * c -f1/b;

	for (auto s = geometry.get_center_elements_begin( );
			s!=geometry.get_center_elements_end( ); ++s)
	{
		ValueType res;
		res= - f1[*s]*2.0 + f2[*s] *c -f1[*s]/b
		;
		ASSERT_EQ( res, f3[*s]) << *s;
	}
}
}

TYPED_TEST(TestFETLBasicArithmetic, scalar_field){
{
	//FIXME  should test with non-uniform field

	typename TestFixture::FieldType f1( TestFixture::grid),f2(TestFixture::grid),
	f3(TestFixture::grid),f4(TestFixture::grid);

	RScalarField a(TestFixture::grid),b(TestFixture::grid),c(TestFixture::grid);

	a.fill(1.0);
	b.fill(3.0);
	c.fill(5.0);

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

	typename TestFixture::FieldType::ValueType va,vb,vc;

	va=2.0;
	vb=3.0;
	vc=5.0;

	f1.fill(va);
	f2.fill(vb);
	f3.fill(vc);

	f4= -(f1^a)-f2/b+f3*c
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
		typename TestFixture::FieldType::ValueType res;
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
// test vector_calculus.h
//template<typename T>
//class TestFETLVecAlgegbra: public testing::Test
//{
//protected:
//	virtual void SetUp()
//	{
//		grid.dt = 1.0;
//		grid.xmin[0] = 0;
//		grid.xmin[1] = 0;
//		grid.xmin[2] = 0;
//		grid.xmax[0] = 1.0;
//		grid.xmax[1] = 1.0;
//		grid.xmax[2] = 1.0;
//		grid.dims[0] = 20;
//		grid.dims[1] = 30;
//		grid.dims[2] = 40;
//		grid.gw[0] = 2;
//		grid.gw[1] = 2;
//		grid.gw[2] = 2;
//
//		grid.Init();
//	}
//public:
//	Grid grid;
//	typedef Field<Geometry<Grid, 0>, Array<T> > ScalarField;
//	typedef Field<Geometry<Grid, 0>, Array<nTuple<3, T> > > VectorField;
//};
//
//typedef testing::Types<double, Complex> VecFieldTypes;
//
//TYPED_TEST_CASE(TestFETLVecAlgegbra, VecFieldTypes);
//
//TYPED_TEST(TestFETLVecAlgegbra,constant_vector){
//{
//	const Grid& grid = TestFixture::grid;
//
//	Geometry<Grid, 0> geometry(TestFixture::grid);
//
//	Vec3 vc1 =
//	{	1.0, 2.0, 3.0};
//	Vec3 vc2 =
//	{	-1.0, 4.0, 2.0};
//
//	Vec3 res_vec;
//
//	res_vec = Cross(vc1, vc2);
//
//	Real res_scalar;
//	res_scalar = Dot(vc1, vc2);
//
//	typename TestFixture::ScalarField res_scalar_field(grid);
//
//	typename TestFixture::VectorField va(grid), vb(grid), res_vector_field(
//			grid);
//
//	va = vc2;
//
//	res_scalar_field = Dot(vc1, va);
//
//	res_vector_field = Cross(vc1, va);
//
//	size_t num_of_comp = geometry.get_num_of_comp( );
//
//	for (auto s = geometry.get_center_elements_begin( );
//			s != geometry.get_center_elements_end( ); ++s)
//	{
//		EXPECT_EQ(res_scalar, res_scalar_field[(*s)] )<< "idx=" <<(*s)<< " | "
//		<<va[(*s)] <<" | "<< vc1 << " | "<< res_scalar_field[(*s)]
//		;
//
//		EXPECT_EQ(res_vec, (res_vector_field[(*s)])) << "idx=" <<(*s)<< " | "
//		<<va[(*s)] <<" | "<< vc1 << " | "<< res_vector_field[(*s)]
//		;
//	}
//
//}
//}
//TYPED_TEST(TestFETLVecAlgegbra,vector_field){
//{
//	using namespace space3;
//	//FIXME  should test with non-uniform field
//
//	Grid const & grid = TestFixture::grid;
//
//	Geometry<Grid, 0> geometry(grid);
//
//	Vec3 vc1 =
//	{	1.0, 2.0, 3.0};
//	Vec3 vc2 =
//	{	-1.0, 4.0, 2.0};
//
//	Vec3 res_vec;
//
//	res_vec=Cross(vc1,vc2);
//
//	Real res_scalar= Dot(vc1,vc2);
//
//	typename TestFixture::VectorField va(grid), vb(grid);
//
//	typename TestFixture::VectorField res_vector_field(grid);
//	typename TestFixture::ScalarField res_scalar_field(grid);
//
//	va = vc1;
////
////	vb = vc2;
////
////	res_scalar_field = Dot( vb ,va);
////
////	res_vector_field = Cross(va , vb);
////
////	size_t num_of_comp =geometry.get_num_of_comp( );
////
////	for (auto s = geometry.get_center_elements_begin( );
////			s!=geometry.get_center_elements_end( ); ++s)
////	{
////		ASSERT_EQ(res_scalar, res_scalar_field[(*s)] ) << "idx=" <<(*s);
////
////		ASSERT_EQ(res_vec, (res_vector_field[(*s)])) << "idx=" <<(*s);
////
////	}
//
//}
//}
//TYPED_TEST(TestFETLVecAlgegbra,complex_vector_field){
//{
//	//FIXME  should test with non-uniform field
//
//	Grid const & grid = TestFixture::grid;
//
//	Vec3 vc1 =
//	{	1.0,2.0,3.0};
//
//	CVec3 vc2 =
//	{
//		Complex( 0.0,0.0) ,
//		Complex( -0.2,0.2) ,
//		Complex( 3.0,1.3)};
//
//	Complex res_scalar= Dot(vc2,vc1);
//
//	CVec3 res_vec;
//
//	res_vec=Cross(vc1,vc2);
//
//	typename TestFixture::VectorField va(grid);
//
//	typename TestFixture::CVectorField vb(grid);
//
//	va = vc1;
//
//	vb = vc2;
//
//	typename TestFixture::CVectorField res_vector_field(grid);
//	typename TestFixture::CScalarField res_scalar_field(grid);
//
//	res_scalar_field = Dot(vb, va);
//
//	res_vector_field = Cross(va, vb);
//
//	size_t num_of_comp =grid.get_num_of_comp(TestFixture::VectorField::IForm);
//
//	for (typename Grid::const_iterator s = grid.get_center_elements_begin(TestFixture::VectorField::IForm);
//			s!=grid.get_center_elements_end(TestFixture::VectorField::IForm); ++s)
//	{
//		ASSERT_EQ(res_scalar, res_scalar_field[(*s)] ) << "idx=" <<(*s);
//
//		ASSERT_EQ(res_vec, (res_vector_field[(*s)])) << "idx=" <<(*s);
//
//	}
//
//}
//}

template<typename TP>
class TestFETLDiffCalcuate: public testing::Test
{

protected:
	virtual void SetUp()
	{
		grid.dt = 1.0;
		grid.xmin[0] = 0;
		grid.xmin[1] = 0;
		grid.xmin[2] = 0;
		grid.xmax[0] = 1.0;
		grid.xmax[1] = 1.0;
		grid.xmax[2] = 1.0;
		grid.dims[0] = 20;
		grid.dims[1] = 30;
		grid.dims[2] = 40;
		grid.gw[0] = 2;
		grid.gw[1] = 2;
		grid.gw[2] = 2;

		grid.Init();

	}
public:

	Grid grid;

	typedef TP ValueType;
	typedef Field<Geometry<Grid, 0>, Array<ValueType> > TZeroForm;
	typedef Field<Geometry<Grid, 1>, Array<ValueType> > TOneForm;
	typedef Field<Geometry<Grid, 2>, Array<ValueType> > TTwoForm;

	double RelativeError(double a, double b)
	{
		return (2.0 * fabs((a - b) / (a + b)));
	}

	void SetValueType(double *v)
	{
		*v = 1.0;
	}

	void SetValueType(Complex *v)
	{
		*v = Complex(1.0, 2.0);
	}

	template<int N, typename TV>
	void SetValueType(nTuple<N, TV> *v)
	{
		for (size_t i = 0; i < N; ++i)
		{
			SetValueType(&((*v)[i]));
		}
	}
};

typedef testing::Types<
		double, Complex, nTuple<3, double>,
		nTuple<3, nTuple<3, double> > > PrimitiveTypes;

TYPED_TEST_CASE(TestFETLDiffCalcuate, PrimitiveTypes);

TYPED_TEST(TestFETLDiffCalcuate, curl_grad_eq_0){
{
	Grid const & grid = TestFixture::grid;

	typename TestFixture::ValueType v;

	TestFixture::SetValueType(&v);

	typename TestFixture::TZeroForm sf(grid);
	typename TestFixture::TOneForm vf1(grid);
	typename TestFixture::TTwoForm vf2(grid);

	size_t num = grid.get_num_of_vertex() * grid.get_num_of_comp(0);

	for (size_t i = 0; i < grid.dims[0]; ++i)
	for (size_t j = 0; j < grid.dims[1]; ++j)
	for (size_t k = 0; k < grid.dims[2]; ++k)
	{
		size_t s = grid.get_cell_num(i, j, k);
		sf[s] = static_cast<double>(s)*v;
	}

	vf1 = Grad(sf);

	vf2 = Curl(Grad(sf));

	Geometry<Grid,1> geometry1(grid);

//	for (auto s = geometry1.get_center_elements_begin(); s != geometry1.get_center_elements_end( ); ++s)
//	{
//
//		ASSERT_NE(0.0,abs(vf1[(*s)])) << "idx=" << *s;
//	}
	Geometry<Grid,2> geometry2(grid);
	for (auto s = geometry2.get_center_elements_begin( ); s != geometry2.get_center_elements_end( ); ++s)
	{
		ASSERT_DOUBLE_EQ(0.0,abs(vf2[(*s)])) << "idx=" << *s;
	}
}
}

TYPED_TEST(TestFETLDiffCalcuate, div_curl_eq_0){
{

	Grid const & grid = TestFixture::grid;

	typename TestFixture::TZeroForm sf(grid);
	typename TestFixture::TOneForm vf1(grid);
	typename TestFixture::TTwoForm vf2(grid);

	typename TestFixture::ValueType v;

	TestFixture::SetValueType(&v);

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

////class TestFETLPerformance: public testing::TestWithParam<size_t>
////{
////protected:
////	virtual void SetUp()
////	{
////		size_t ratio = GetParam();
////
////		grid.dt = 1.0;
////		grid.xmin[0] = 0;
////		grid.xmin[1] = 0;
////		grid.xmin[2] = 0;
////		grid.xmax[0] = 1.0;
////		grid.xmax[1] = 1.0;
////		grid.xmax[2] = 1.0;
////		grid.dims[0] = 2 * ratio;
////		grid.dims[1] = 3 * ratio;
////		grid.dims[2] = 4 * ratio;
////		grid.gw[0] = 2;
////		grid.gw[1] = 2;
////		grid.gw[2] = 2;
////
////		grid.Init();
////
////	}
////public:
////	Grid grid;
////
////	static const int NDIMS = 3;
////
////	double RelativeError2(double a, double b)
////	{
////		return pow(2.0 * (a - b) / (a + b), 2.0);
////	}
////};
////
////INSTANTIATE_TEST_CASE_P(TestPerformance, TestFETLPerformance,
////		testing::ValueTypes(10, 20, 50));
////
////TEST_P(TestFETLPerformance, error_analyze)
////{
////
////	static const double epsilon = 0.01;
////
////	ZeroForm sf(grid), res_sf(grid);
////
////	OneForm res_vf(grid);
////
////	size_t size = grid.get_num_of_vertex() * grid.get_num_of_comp(IZeroForm);
////
////	Real k0 = 1.0 * TWOPI, k1 = 2.0 * TWOPI, k2 = 3.0 * TWOPI;
////
////#pragma omp parallel for
////	for (size_t i = 0; i < grid.dims[0]; ++i)
////	{
////		for (size_t j = 0; j < grid.dims[1]; ++j)
////			for (size_t k = 0; k < grid.dims[2]; ++k)
////			{
////				size_t s = grid.get_cell_num(i, j, k);
////				sf[s] = sin(
////						k0 * i * grid.dx[0] + k1 * j * grid.dx[1]
////								+ k2 * k * grid.dx[2]);
////			}
////	}
////	res_vf = Grad(sf);
////
////	res_sf = Diverge(Grad(sf));
////
////	Real rex = 0.0;
////	Real rey = 0.0;
////	Real rez = 0.0;
////	Real re2 = 0.0;
////
////#pragma omp parallel for  reduction(+:rex,rey,rez,re2)
////	for (size_t i = grid.gw[0]; i < grid.dims[0] - grid.gw[0]; ++i)
////		for (size_t j = grid.gw[1]; j < grid.dims[1] - grid.gw[1]; ++j)
////			for (size_t k = grid.gw[2]; k < grid.dims[2] - grid.gw[2]; ++k)
////			{
////				size_t s = grid.get_cell_num(i, j, k);
////
////				Real alpha = k0 * i * grid.dx[0] + k1 * j * grid.dx[1]
////						+ k2 * k * grid.dx[2];
////
////				rex += RelativeError2(k0 * cos(alpha + k0 * 0.5 * grid.dx[0]),
////						res_vf[s * 3 + 0]);
////				rey += RelativeError2(k1 * cos(alpha + k1 * 0.5 * grid.dx[1]),
////						res_vf[s * 3 + 1]);
////				rez += RelativeError2(k2 * cos(alpha + k2 * 0.5 * grid.dx[2]),
////						res_vf[s * 3 + 2]);
////
////				re2 += RelativeError2(
////						-(k0 * k0 + k1 * k1 + k2 * k2) * sin(alpha), res_sf[s]);
////			}
////
////	rex /= static_cast<Real>(size);
////	rey /= static_cast<Real>(size);
////	rez /= static_cast<Real>(size);
////	re2 /= static_cast<Real>(size);
////
////	EXPECT_LE(rex, pow(k0*grid.dx[0],2));
////	EXPECT_LE(rey, pow(k1*grid.dx[1],2));
////	EXPECT_LE(rez, pow(k2*grid.dx[2],2));
////	EXPECT_LE( re2,
////			(pow(k0*grid.dx[0],2)+pow(k1*grid.dx[1],2)+pow(k2*grid.dx[2],2))/3.0);
////
////}

