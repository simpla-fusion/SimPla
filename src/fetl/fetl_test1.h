/*
 * fetl_test1.h
 *
 *  created on: 2014-3-24
 *      Author: salmon
 */

#ifndef FETL_TEST1_H_
#define FETL_TEST1_H_

#include <gtest/gtest.h>
#include <random>
#include "fetl.h"
#include "fetl_test.h"
#include "../utilities/log.h"
#include "../utilities/pretty_stream.h"
#include "field_update_ghosts.h"
using namespace simpla;

template<unsigned int IFORM, typename TV>
struct TestFETLParam1
{
	typedef TMesh manifold_type;
	typedef TV value_type;
	static constexpr unsigned int iform = IFORM;

//	static void SetUpMesh(manifold_type * manifold)
//	{
//
//	}
//
//	static void SetDefaultValue(value_type * v)
//	{
//		SetDefaultValue(v);
//	}
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
};

template<typename TParam>
class TestFETLBase: public testing::Test
{
protected:
	virtual void SetUp()
	{
		LOGGER.set_stdout_visable_level(10);

		nTuple<3, Real> xmin =
		{ 1.0, 1.0, 0.0 };

		nTuple<3, Real> xmax =
		{ 2.0, 3.0, TWOPI };

		nTuple<3, size_t> dims =
		{ 16, 32, 67 };

		manifold.set_dimensions(dims);
		manifold.set_extents(xmin, xmax);
		manifold.update();

	}
public:

	typedef typename TParam::manifold_type manifold_type;
	typedef typename TParam::value_type value_type;
	static constexpr unsigned int iform = TParam::iform;

	typedef typename manifold_type::template field<VERTEX, Real> scalar_field_type;
	typedef typename manifold_type::template field<VERTEX, value_type> field_type;

	manifold_type manifold;
	value_type default_value;

};

TYPED_TEST_CASE_P(TestFETLBase);

TYPED_TEST_P(TestFETLBase, constant_real){
{
	typename TestFixture::manifold_type const & manifold= TestFixture::manifold;

	if (!manifold.is_valid()) return;

	typedef typename TestFixture::value_type value_type;
	typedef typename TestFixture::field_type field_type;

	auto f1 = manifold.template make_field<field_type>();
	auto f2 = manifold.template make_field<field_type>();
	auto f3 = manifold.template make_field<field_type>();

	Real a,b,c;
	a=1.0,b=-2.0,c=3.0;

	value_type va,vb;

	va=2.0;vb=3.0;

	f1=va;
	f2=vb;

	LOG_CMD(f3 = -f1*a +f2*c - f1/b -f1 );

	for(auto s :manifold.select( field_type::iform))
	{
		value_type res;
		res= - f1[s]*a + f2[s] *c -f1[s]/b-f1[s];
		ASSERT_EQ( res, f3[s]);
	}
}
}

TYPED_TEST_P(TestFETLBase, scalar_field){
{

	typename TestFixture::manifold_type const & manifold= TestFixture::manifold;
	if (!manifold.is_valid()) return;
	typedef typename TestFixture::value_type value_type;
	typedef typename TestFixture::field_type field_type;
	typedef typename TestFixture::scalar_field_type scalar_field_type;

	auto f1 = manifold.template make_field<field_type>();
	auto f2 = manifold.template make_field<field_type>();
	auto f3 = manifold.template make_field<field_type>();
	auto f4 = manifold.template make_field<field_type>();

	auto a=manifold.template make_field<scalar_field_type>();
	auto b=manifold.template make_field<scalar_field_type>();
	auto c=manifold.template make_field<scalar_field_type>();

	Real ra=1.0,rb=10.0,rc=100.0;

	value_type va,vb,vc;

	va=ra;
	vb=rb;
	vc=rc;

	a.fill(ra);
	b.fill(rb);
	c.fill(rc);

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
//	update_ghosts(&f1);
//	update_ghosts(&f2);
//	update_ghosts(&f3);

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

	auto hash=manifold.make_hash(manifold.select( field_type::iform ));

	for(auto s :manifold.select( field_type::iform ) )
	{
		value_type res= - f1[s]*ra +f2[s]* rb -f3[s]/ rc -f1[s];

		EXPECT_LE( abs(res-f4[s]) ,1.0e-10 )<< "s= "<<(hash(s));
	}

	EXPECT_EQ(0,count)<< "number of error points =" << count;
}
}

REGISTER_TYPED_TEST_CASE_P(TestFETLBase, constant_real, scalar_field);

typedef testing::Types<

TestFETLParam1<VERTEX, Real>	//
		, TestFETLParam1<EDGE, Real>	//
		, TestFETLParam1<FACE, Real>	//
		, TestFETLParam1<VOLUME, Real>	//

		, TestFETLParam1<VERTEX, Complex>	//
		, TestFETLParam1<EDGE, Complex>	//
		, TestFETLParam1<FACE, Complex>	//
		, TestFETLParam1<VOLUME, Complex>	//

		, TestFETLParam1<VERTEX, nTuple<3, Real> >	//
		, TestFETLParam1<EDGE, nTuple<3, Real> >	//
		, TestFETLParam1<FACE, nTuple<3, Real> >	//
		, TestFETLParam1<VOLUME, nTuple<3, Real> >	//

		, TestFETLParam1<VERTEX, nTuple<3, Complex> >	//
		, TestFETLParam1<EDGE, nTuple<3, Complex> >	//
		, TestFETLParam1<FACE, nTuple<3, Complex> >	//
		, TestFETLParam1<VOLUME, nTuple<3, Complex> >	//

		, TestFETLParam1<VERTEX, nTuple<3, nTuple<3, Real>> >	//
		, TestFETLParam1<EDGE, nTuple<3, nTuple<3, Real>> >	//
		, TestFETLParam1<FACE, nTuple<3, nTuple<3, Real>> >	//
		, TestFETLParam1<VOLUME, nTuple<3, nTuple<3, Real>> >	//

		, TestFETLParam1<VERTEX, nTuple<3, nTuple<3, Complex>> >	//
		, TestFETLParam1<EDGE, nTuple<3, nTuple<3, Complex>> >	//
		, TestFETLParam1<FACE, nTuple<3, nTuple<3, Complex>> >	//
		, TestFETLParam1<VOLUME, nTuple<3, nTuple<3, Complex>> >	//

> TypeParamList;

INSTANTIATE_TYPED_TEST_CASE_P(FETL, TestFETLBase, TypeParamList);

#endif /* FETL_TEST1_H_ */
