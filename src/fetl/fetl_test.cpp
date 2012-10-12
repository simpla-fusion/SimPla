/*
 * test_field.cpp
 *
 *  Created on: 2012-1-13
 *      Author: salmon
 */

#include <gtest/gtest.h>
#include "include/simpla_defs.h"

#include "physics/constants.h"
#include "grid/uniform_rect.h"
#include "fetl.h"
#include "vector_calculus.h"
#include "primitives/properties.h"

using namespace simpla;
using namespace simpla::fetl;

DEFINE_FIELDS(Real, UniformRectGrid)

template<typename TF>
class TestFETLBasicArithmetic: public testing::Test
{
protected:
	virtual void SetUp()
	{
		Log::Verbose(10);
		ptree pt;
		pt.push_back(ptree::value_type("type", ptree("UniformRect")));
		pt.push_back(ptree::value_type("dt", ptree("1.0")));
		pt.push_back(ptree::value_type("xmin", ptree("0 0 0 ")));
		pt.push_back(ptree::value_type("xmax", ptree("1.0 1.0 1.0 ")));
		pt.push_back(ptree::value_type("dims", ptree("20  30  40 ")));
		pt.push_back(ptree::value_type("ghostwidth", ptree("2 2 2")));

		grid = TR1::shared_ptr<Grid>(new Grid(pt));
	}
public:
	typedef TF FieldType;
	typedef typename FieldType::ValueType ValueType;
	typedef typename _fetl_impl::ComplexTraits<ValueType>::ValueType CValueType;
	typedef Field<Grid, TF::IForm, CValueType> CFieldType;
	TR1::shared_ptr<Grid> grid;

};

typedef testing::Types<RZeroForm
		, CZeroForm, VecZeroForm
		, ROneForm, COneForm, VecOneForm
		, RTwoForm, CTwoForm, VecTwoForm
> AllFieldTypes;

//, VecThreeForm

// test arithmetic.h
TYPED_TEST_CASE(TestFETLBasicArithmetic, AllFieldTypes);

TYPED_TEST(TestFETLBasicArithmetic,create_write_read){
{	Log::Verbose(10);
	Grid const & grid = *TestFixture::grid;
	typename TestFixture::FieldType f(grid);
	size_t size =grid.get_num_of_elements(TestFixture::FieldType::IForm);

	typename TestFixture::FieldType::ValueType a; a= 1.0;

	for (size_t s = 0; s < size; ++s)
	{
		f[s] = a*static_cast<double>(s);
	}

	for (size_t s = 0; s<size; ++s)
	{
		typename TestFixture::ValueType res;
		res=a*static_cast<Real>(s);
		ASSERT_EQ(res,f[s])<<"idx=" << s;
	}

}
}

TYPED_TEST(TestFETLBasicArithmetic,assign){
{
	Grid const & grid = *TestFixture::grid;

	typename TestFixture::FieldType f1(grid),f2(grid);

	typedef typename TestFixture::FieldType::ValueType ValueType;

	size_t size = grid.get_num_of_elements(TestFixture::FieldType::IForm);

	ValueType a; a = 3.0;

	f2 = a;

	for (size_t s = 0; s<size; ++s)
	{
		ASSERT_EQ(a,f2[s])<<"idx="<< s;
	}

	for (size_t s = 0; s<size; ++s)
	{
		f1[s]=a*static_cast<Real>(s);
	}

	f2 = f1;

	for (typename Grid::const_iterator s = grid.get_center_elements_begin(TestFixture::FieldType::IForm);
			s!=grid.get_center_elements_end(TestFixture::FieldType::IForm); ++s)
	{
		typename TestFixture::ValueType res;
		res=a*static_cast<Real>(*s);
		ASSERT_EQ( res,f2[*s])<<"idx="<< *s;
	}
}
}
TYPED_TEST(TestFETLBasicArithmetic, constant_real){
{
	Grid const & grid= *TestFixture::grid;

	typename TestFixture::FieldType f1( grid),f2(grid),f3(grid);

	size_t size=grid.get_num_of_elements(TestFixture::FieldType::IForm);

	Real a,b,c;
	a=1.0,b=-2.0,c=3.0;

	typedef typename TestFixture::ValueType ValueType;

	ValueType va,vb;

	va=2.0;vb=3.0;

	f1 = va;
	f2 = vb;
	f3 = -f1+f2 * c-f1/b;

	for (typename Grid::const_iterator s = grid.get_center_elements_begin(TestFixture::FieldType::IForm);
			s!=grid.get_center_elements_end(TestFixture::FieldType::IForm); ++s)
	{
		ValueType res;
		res=-f1[*s] + f2[*s] *c - f1[*s]/b;
		ASSERT_EQ( res, f3[*s]) << *s;
	}
}
}

TYPED_TEST(TestFETLBasicArithmetic, constant_complex){
{
	Grid const & grid = *TestFixture::grid;

	typename TestFixture::FieldType f1( grid),f2(grid);
	typename TestFixture::CFieldType f3(grid);

	size_t size=grid.get_num_of_elements(TestFixture::FieldType::IForm);

	Complex a(1.0,-1.0);
	Complex b(-2.0,3.0);
	Complex c(4.0,3.0);
	typename TestFixture::ValueType va,vb;

	va=2.0;vb=3.0;

	f1 = va;
	f2 = vb;
	f3 = -f1 + c* f2-f1/b;

	for (typename Grid::const_iterator s = grid.get_center_elements_begin(TestFixture::FieldType::IForm);
			s!=grid.get_center_elements_end(TestFixture::FieldType::IForm); ++s)
	{
		typename TestFixture::CValueType res;
		res=-f1[*s] + f2[*s] *c - f1[*s]/b;
		ASSERT_EQ( res, f3[*s]) << *s;
	}

}
}
TYPED_TEST(TestFETLBasicArithmetic, scalar_field){
{
	//FIXME  should test with non-uniform field

	Grid const & grid = * TestFixture::grid;

	typename TestFixture::FieldType f1( grid),f2(grid),f3(grid);

	size_t num_of_comp=grid.get_num_of_comp(TestFixture::FieldType::IForm);

	size_t size=grid.get_num_of_elements(TestFixture::FieldType::IForm);

	RScalarField a(grid),b(grid),c(grid);

	for(size_t s=0;s<size/num_of_comp;++s)
	{
		a[s]= (1.0); //+s;
		b[s]= (3.0);//*s;
		c[s]= (5.0);//(s+1.0);
	}
	typename TestFixture::ValueType va,vb;

	va=2.0;vb=3.0;

	f1 = va;

	f2 = vb;

	f3 = - f1/a- b*f2 +f1*c;

	for (typename Grid::const_iterator s = grid.get_center_elements_begin(TestFixture::FieldType::IForm);
			s!=grid.get_center_elements_end(TestFixture::FieldType::IForm); ++s)
	{
		typename TestFixture::ValueType res;
		res=-f1[*s] /a[*s/num_of_comp] -b[*s/num_of_comp]*f2[*s] +f1[*s]*c[*s/num_of_comp];
		ASSERT_EQ( res, f3[*s])
		<< "idx=" <<*s;
	}
}

}
// test vector_calculus.h
template<typename T>
class TestFETLVecAlgegbra: public testing::Test
{
protected:
	virtual void SetUp()
	{
		ptree pt;
		pt.push_back(ptree::value_type("type", ptree("UniformRect")));
		pt.push_back(ptree::value_type("dt", ptree("1.0")));
		pt.push_back(ptree::value_type("xmin", ptree("0 0 0 ")));
		pt.push_back(ptree::value_type("xmax", ptree("1.0 1.0 1.0 ")));
		pt.push_back(ptree::value_type("dims", ptree("20  30  40 ")));
		pt.push_back(ptree::value_type("ghostwidth", ptree("2 2 2")));

		grid = TR1::shared_ptr<Grid>(new Grid(pt));

	}
public:
	TR1::shared_ptr<Grid> grid;
	typedef Field<Grid, IZeroForm, T> ScalarField;
	typedef Field<Grid, IZeroForm, nTuple<THREE, T> > VectorField;

	typedef Field<Grid, IZeroForm,
			typename _fetl_impl::ComplexTraits<T>::ValueType> CScalarField;
	typedef Field<Grid, IZeroForm,
			nTuple<THREE, typename _fetl_impl::ComplexTraits<T>::ValueType> > CVectorField;

};

typedef testing::Types<double, Complex> VecFieldTypes;

TYPED_TEST_CASE(TestFETLVecAlgegbra, VecFieldTypes);

TYPED_TEST(TestFETLVecAlgegbra,constant_vector){
{
	const Grid& grid = *TestFixture::grid;

	Vec3 vc1 =
	{	1.0, 2.0, 3.0};
	Vec3 vc2 =
	{	-1.0, 4.0, 2.0};

	Vec3 res_vec;

	res_vec = Cross(vc1, vc2);

	Real res_scalar;
	res_scalar = Dot(vc1, vc2);

	typename TestFixture::ScalarField res_scalar_field(grid);

	typename TestFixture::VectorField va(grid), vb(grid), res_vector_field(
			grid);

	va = vc2;

	res_scalar_field = Dot(vc1, va);

	res_vector_field = Cross(vc1, va);

	size_t num_of_comp = grid.get_num_of_comp(
			TestFixture::VectorField::IForm);

	for (typename Grid::const_iterator s = grid.get_center_elements_begin(
					TestFixture::VectorField::IForm);
			s
			!= grid.get_center_elements_end(
					TestFixture::VectorField::IForm); ++s)
	{
		ASSERT_EQ(res_scalar, res_scalar_field[(*s)] )<< "idx=" <<(*s);

		ASSERT_EQ(res_vec, (res_vector_field[(*s)])) << "idx=" <<(*s);
	}

}
}
TYPED_TEST(TestFETLVecAlgegbra,vector_field){
{
	//FIXME  should test with non-uniform field

	Grid const & grid = *TestFixture::grid;

	Vec3 vc1 =
	{	1.0, 2.0, 3.0};
	Vec3 vc2 =
	{	-1.0, 4.0, 2.0};

	Vec3 res_vec;

	res_vec=Cross(vc1,vc2);

	Real res_scalar= Dot(vc1,vc2);

	typename TestFixture::VectorField va(grid), vb(grid);

	typename TestFixture::VectorField res_vector_field(grid);
	typename TestFixture::ScalarField res_scalar_field(grid);

	va = vc1;

	vb = vc2;

	res_scalar_field = Dot(vb, va);

	res_vector_field = Cross(va, vb);

	size_t num_of_comp =grid.get_num_of_comp(TestFixture::VectorField::IForm);

	for (typename Grid::const_iterator s = grid.get_center_elements_begin(TestFixture::VectorField::IForm);
			s!=grid.get_center_elements_end(TestFixture::VectorField::IForm); ++s)
	{
		ASSERT_EQ(res_scalar, res_scalar_field[(*s)] ) << "idx=" <<(*s);

		ASSERT_EQ(res_vec, (res_vector_field[(*s)])) << "idx=" <<(*s);

	}

}
}
TYPED_TEST(TestFETLVecAlgegbra,complex_vector_field){
{
	//FIXME  should test with non-uniform field

	Grid const & grid = *TestFixture::grid;

	Vec3 vc1 =
	{	1.0,2.0,3.0};

	CVec3 vc2 =
	{
		Complex( 0.0,0.0) ,
		Complex( -0.2,0.2) ,
		Complex( 3.0,1.3)};

	Complex res_scalar= Dot(vc2,vc1);

	CVec3 res_vec;

	res_vec=Cross(vc1,vc2);

	typename TestFixture::VectorField va(grid);

	typename TestFixture::CVectorField vb(grid);

	va = vc1;

	vb = vc2;

	typename TestFixture::CVectorField res_vector_field(grid);
	typename TestFixture::CScalarField res_scalar_field(grid);

	res_scalar_field = Dot(vb, va);

	res_vector_field = Cross(va, vb);

	size_t num_of_comp =grid.get_num_of_comp(TestFixture::VectorField::IForm);

	for (typename Grid::const_iterator s = grid.get_center_elements_begin(TestFixture::VectorField::IForm);
			s!=grid.get_center_elements_end(TestFixture::VectorField::IForm); ++s)
	{
		ASSERT_EQ(res_scalar, res_scalar_field[(*s)] ) << "idx=" <<(*s);

		ASSERT_EQ(res_vec, (res_vector_field[(*s)])) << "idx=" <<(*s);

	}

}
}

class TestFETLDiffCalcuate: public testing::Test
{

protected:
	virtual void SetUp()
	{
		ptree pt;
		pt.push_back(ptree::value_type("type", ptree("UniformRect")));
		pt.push_back(ptree::value_type("dt", ptree("1.0")));
		pt.push_back(ptree::value_type("xmin", ptree("0 0 0 ")));
		pt.push_back(ptree::value_type("xmax", ptree("1.0 1.0 1.0 ")));
		pt.push_back(ptree::value_type("dims", ptree("20  30  40 ")));
		pt.push_back(ptree::value_type("ghostwidth", ptree("2 2 2")));

		pgrid = TR1::shared_ptr<Grid>(new Grid(pt));

	}
public:

	TR1::shared_ptr<Grid> pgrid;

	double RelativeError(double a, double b)
	{
		return 2.0 * fabs((a - b) / (a + b));
	}
};

TEST_F(TestFETLDiffCalcuate, curl_grad_eq_0)
{
	Grid const & grid = *pgrid;

	ZeroForm sf(grid);
	OneForm vf1(grid);
	TwoForm vf2(grid);
	size_t size = grid.get_num_of_vertex() * grid.get_num_of_comp(IZeroForm);

	for (size_t i = 0; i < grid.dims[0]; ++i)
		for (size_t j = 0; j < grid.dims[1]; ++j)
			for (size_t k = 0; k < grid.dims[2]; ++k)
			{
				size_t s = grid.get_cell_num(i, j, k);
				sf[s] = s;
			}

	vf1 = Grad(sf);

	vf2 = Curl(Grad(sf));

	size_t num_of_comp = grid.get_num_of_comp(IOneForm);

	for (typename Grid::const_iterator s = grid.get_center_elements_begin(
			IOneForm); s != grid.get_center_elements_end(IOneForm); ++s)
	{

		EXPECT_NE(0.0, vf1[(*s)]) << "idx=" << *s;
	}

	for (typename Grid::const_iterator s = grid.get_center_elements_begin(
			ITwoForm); s != grid.get_center_elements_end(ITwoForm); ++s)
	{
		EXPECT_DOUBLE_EQ(0.0, vf2[(*s)]) << "idx=" << *s;
	}
}

TEST_F(TestFETLDiffCalcuate, div_curl_eq_0)
{
	Grid const & grid = *pgrid;

	ZeroForm sf(grid);
	OneForm vf1(grid);
	TwoForm vf2(grid);
	size_t num_of_ele = grid.get_num_of_elements(IZeroForm);

	for (size_t s = 0; s < num_of_ele; ++s)
	{

		vf2[s * 3 + 0] = s + 1.0;
		vf2[s * 3 + 1] = s * 2.0;
		vf2[s * 3 + 2] = s + 10.0;
	}

	vf1 = Curl(vf2);

	sf = Diverge(Curl(vf2));

	for (typename Grid::const_iterator s = grid.get_center_elements_begin(
			ITwoForm); s != grid.get_center_elements_end(ITwoForm); ++s)
	{

		EXPECT_NE(0.0, vf1[(*s)]) << "idx=" << *s;
	}

	for (typename Grid::const_iterator s = grid.get_center_elements_begin(
			IZeroForm); s != grid.get_center_elements_end(IZeroForm); ++s)
	{
		EXPECT_DOUBLE_EQ(0.0, sf[(*s)]) << "idx=" << *s;
	}
}

class TestFETLPerformance: public testing::TestWithParam<size_t>
{
protected:
	virtual void SetUp()
	{
		size_t ratio = GetParam();

		IVec3 dims =
		{ 2 * ratio, 3 * ratio, 4 * ratio };

		ptree pt;
		pt.push_back(ptree::value_type("type", ptree("UniformRect")));
		pt.push_back(ptree::value_type("dt", ptree("1.0")));
		pt.push_back(ptree::value_type("xmin", ptree("0 0 0 ")));
		pt.push_back(ptree::value_type("xmax", ptree("1.0 1.0 1.0 ")));
		pt.push_back(ptree::value_type("dims", ptree("20  30  40 ")));
		pt.push_back(ptree::value_type("ghostwidth", ptree("2 2 2")));
		pgrid = TR1::shared_ptr<Grid>(new Grid(pt));

	}
public:
	TR1::shared_ptr<Grid> pgrid;

	static const int NDIMS = 3;

	double RelativeError2(double a, double b)
	{
		return pow(2.0 * (a - b) / (a + b), 2.0);
	}
};

INSTANTIATE_TEST_CASE_P(TestPerformance, TestFETLPerformance,
		testing::Values(10, 50, 100,1000,2000));

TEST_P(TestFETLPerformance, error_analyze)
{
	Grid const & grid = *pgrid;

	static const double epsilon = 0.01;

	ZeroForm sf(grid), res_sf(grid);

	OneForm res_vf(grid);

	size_t size = grid.get_num_of_vertex() * grid.get_num_of_comp(IZeroForm);

	Real k0 = 1.0 * TWOPI, k1 = 2.0 * TWOPI, k2 = 3.0 * TWOPI;

#pragma omp parallel for
	for (size_t i = 0; i < grid.dims[0]; ++i)
	{
		for (size_t j = 0; j < grid.dims[1]; ++j)
			for (size_t k = 0; k < grid.dims[2]; ++k)
			{
				size_t s = grid.get_cell_num(i, j, k);
				sf[s] = sin(
						k0 * i * grid.dx[0] + k1 * j * grid.dx[1]
								+ k2 * k * grid.dx[2]);
			}
	}
	res_vf = Grad(sf);

	res_sf = Diverge(Grad(sf));

	Real rex = 0.0;
	Real rey = 0.0;
	Real rez = 0.0;
	Real re2 = 0.0;

#pragma omp parallel for  reduction(+:rex,rey,rez,re2)
	for (size_t i = grid.gw[0]; i < grid.dims[0] - grid.gw[0]; ++i)
		for (size_t j = grid.gw[1]; j < grid.dims[1] - grid.gw[1]; ++j)
			for (size_t k = grid.gw[2]; k < grid.dims[2] - grid.gw[2]; ++k)
			{
				size_t s = grid.get_cell_num(i, j, k);

				Real alpha = k0 * i * grid.dx[0] + k1 * j * grid.dx[1]
						+ k2 * k * grid.dx[2];

				rex += RelativeError2(k0 * cos(alpha + k0 * 0.5 * grid.dx[0]),
						res_vf[s * 3 + 0]);
				rey += RelativeError2(k1 * cos(alpha + k1 * 0.5 * grid.dx[1]),
						res_vf[s * 3 + 1]);
				rez += RelativeError2(k2 * cos(alpha + k2 * 0.5 * grid.dx[2]),
						res_vf[s * 3 + 2]);

				re2 += RelativeError2(
						-(k0 * k0 + k1 * k1 + k2 * k2) * sin(alpha), res_sf[s]);
			}

	rex /= static_cast<Real>(size);
	rey /= static_cast<Real>(size);
	rez /= static_cast<Real>(size);
	re2 /= static_cast<Real>(size);

	EXPECT_LE(rex, pow(k0*grid.dx[0],2));
	EXPECT_LE(rey, pow(k1*grid.dx[1],2));
	EXPECT_LE(rez, pow(k2*grid.dx[2],2));
	EXPECT_LE( re2,
			(pow(k0*grid.dx[0],2)+pow(k1*grid.dx[1],2)+pow(k2*grid.dx[2],2))/3.0);

}

