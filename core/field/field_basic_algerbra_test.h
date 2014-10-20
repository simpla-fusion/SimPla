/*
 * field_basic_algerbra_test.h
 *
 *  created on: 2014-2-20
 *      Author: salmon
 */

#ifndef FIELD_BASIC_ALGEBRA_TEST_H_
#define FIELD_BASIC_ALGEBRA_TEST_H_

#include <gtest/gtest.h>
#include <random>
#include "field.h"

using namespace simpla;

template<typename TField>
class TestField: public testing::Test
{
protected:
	virtual void SetUp()
	{
		LOGGER.set_stdout_visable_level(10);

		domain_type(12, 20).swap(domain);

	}
public:

	typedef typename TField::domain_type domain_type;
	typedef typename TField::value_type value_type;

	domain_type domain;
	value_type default_value;

	typedef Field<domain_type, value_type> field_type;

	Field<domain_type, value_type> make_field() const
	{
		return std::move(Field<domain_type, value_type>(domain));
	}

	Field<domain_type, Real> make_scalar_field() const
	{
		return std::move(Field<domain_type, Real>(domain));
	}

};

TYPED_TEST_CASE_P(TestField);

TYPED_TEST_P(TestField, constant_real){
{

	typedef typename TestFixture::value_type value_type;
	typedef typename TestFixture::field_type field_type;

	auto f1 = TestFixture::make_field();
	auto f2 = TestFixture::make_field();
	auto f3 = TestFixture::make_field();

	Real a,b,c;
	a=1.0,b=-2.0,c=3.0;

	value_type va,vb;

	va=2.0;vb=3.0;

	f1=va;
	f2=vb;

	LOG_CMD(f3 = -f1*a +f2*c - f1/b -f1 );

	for(auto s : TestFixture::domain)
	{
		value_type res;
		res= - f1[s]*a + f2[s] *c -f1[s]/b-f1[s];

		EXPECT_LE(abs( res- f3[s]),EPSILON);
	}
}
}

TYPED_TEST_P(TestField, scalar_field){
{

	typedef typename TestFixture::value_type value_type;

	auto f1 = TestFixture::make_field();
	auto f2 = TestFixture::make_field();
	auto f3 = TestFixture::make_field();
	auto f4 = TestFixture::make_field();

	auto a=TestFixture::make_scalar_field();
	auto b=TestFixture::make_scalar_field();
	auto c=TestFixture::make_scalar_field();

	Real ra=1.0,rb=10.0,rc=100.0;

	value_type va,vb,vc;

	va=ra;
	vb=rb;
	vc=rc;

	a=ra;
	b=rb;
	c=rc;

	f1.allocate();
	f2.allocate();
	f3.allocate();
	f4.allocate();

	size_t count=0;

	std::mt19937 gen;
	std::uniform_real_distribution<Real> uniform_dist(0, 1.0);

	for(auto s:f1.domain())
	{
		f1[s]=va *uniform_dist(gen);
	}
	for(auto s:f2.domain())
	{
		f2[s]=vb *uniform_dist(gen);
	}

	for(auto s:f3.domain())
	{
		f3[s]=vc *uniform_dist(gen);
	}

	LOG_CMD(f4= -f1*a +f2*b -f3/c -f1 );

//	Plus( Minus(Negate(Wedge(f1,a)),Divides(f2,b)),Multiplies(f3,c) )

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

	for(auto s :TestFixture::domain )
	{
		value_type res= - f1[s]*ra +f2[s]* rb -f3[s]/ rc -f1[s];

		EXPECT_LE( abs(res-f4[s]) ,EPSILON )<< "s= "<<(TestFixture::domain.hash(s));
	}

	EXPECT_EQ(0,count)<< "number of error points =" << count;
}
}

REGISTER_TYPED_TEST_CASE_P(TestField, constant_real, scalar_field);
//#include <gtest/gtest.h>
//
//#include "field.h"
//#include "../manifold/domain_traits.h"
//using namespace simpla;
//
////#include "../utilities/log.h"
////#include "../utilities/pretty_stream.h"
////
////using namespace simpla;
////
//class Domain;
//class Container;
//
//class TestFIELD: public testing::TestWithParam<
//		std::tuple<typename domain_traits<Domain>::coordinates_type,
//				typename Domain::coordinates_type,
//				nTuple<Domain::NDIMS, size_t>, nTuple<Domain::NDIMS, Real> > >
//{
//
//protected:
//	void SetUp()
//	{
//		LOGGER.set_stdout_visable_level(LOG_INFORM);
//		auto param = GetParam();
//
//		xmin = std::get<0>(param);
//
//		xmax = std::get<1>(param);
//
//		dims = std::get<2>(param);
//
//		K_real = std::get<3>(param);
//
//		SetDefaultValue(&default_value);
//
//		for (int i = 0; i < NDIMS; ++i)
//		{
//			if (dims[i] <= 1 || (xmax[i] <= xmin[i]))
//			{
//				dims[i] = 1;
//				K_real[i] = 0.0;
//				xmax[i] = xmin[i];
//			}
//		}
//
//		mesh.set_dimensions(dims);
//		mesh.set_extents(xmin, xmax);
//
//		mesh.update();
//
//	}
//public:
//
//	typedef Domain domain_type;
//	typedef Real value_type;
//	typedef domain_type::scalar_type scalar_type;
//	typedef domain_type::iterator iterator;
//	typedef domain_type::coordinates_type coordinates_type;
//
//	domain_type mesh;
//
//	static constexpr unsigned int NDIMS = domain_type::NDIMS;
//
//	nTuple<NDIMS, Real> xmin;
//
//	nTuple<NDIMS, Real> xmax;
//
//	nTuple<NDIMS, size_t> dims;
//
//	nTuple<3, Real> K_real; // @NOTE must   k = n TWOPI, period condition
//
//	nTuple<3, scalar_type> K_imag;
//
//	value_type default_value;
//
//	template<typename T>
//	void SetDefaultValue(T* v)
//	{
//		*v = 1;
//	}
//	template<typename T>
//	void SetDefaultValue(std::complex<T>* v)
//	{
//		T r;
//		SetDefaultValue(&r);
//		*v = std::complex<T>();
//	}
//
//	template<unsigned int N, typename T>
//	void SetDefaultValue(nTuple<N, T>* v)
//	{
//		for (int i = 0; i < N; ++i)
//		{
//			(*v)[i] = i;
//		}
//	}
//
//	virtual ~TestFIELD()
//	{
//
//	}
//
//};

#endif /* FIELD_BASIC_ALGEBRA_TEST_H_ */
