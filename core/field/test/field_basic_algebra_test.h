/*
 * field_basic_algerbra_test.h
 *
 *  created on: 2014-2-20
 *      Author: salmon
 */

#ifndef FIELD_BASIC_ALGEBRA_TEST_H_
#define FIELD_BASIC_ALGEBRA_TEST_H_

#include <stddef.h>
#include <memory>
#include <random>

#include <gtest/gtest.h>

#include "../../gtl/macro.h"
#include "../../gtl/primitives.h"
#include "../../gtl/type_traits.h"
#include "../../gtl/utilities/log.h"
#include "../../manifold/ManifoldTraits.h"
#include "../../field/FieldDense.h"
#include "../../field/FieldTraits.h"
#include "../../field/FieldExpression.h"

using namespace simpla;

template<typename TField>
class TestField : public testing::Test
{
protected:
    virtual void SetUp()
    {
        logger::set_stdout_level(10);

        mesh = std::make_shared<mesh_type>();

        nTuple<size_t, 3> dims = {10, 1, 1};

        mesh->dimensions(dims);
//		geometry->extents(xmin, xmax);
        mesh->deploy();
    }

public:

    typedef TField field_type;

    typedef traits::manifold_type_t<field_type> mesh_type;

    typedef traits::value_type_t<field_type> value_type;

    typedef Real scalar_type;

//    typedef  traits::scalar_type_t<mesh_type> scalar_type;

    static constexpr int iform = traits::iform<TField>::value;

    static std::shared_ptr<mesh_type> mesh;

    value_type default_value;

    static auto domain()
    DECL_RET_TYPE((mesh->template range<iform>()))


    auto make_field() const
    DECL_RET_TYPE((traits::make_field<value_type, iform>(*mesh)))

    auto make_scalarField() const
    DECL_RET_TYPE((traits::make_field<scalar_type, iform>(*mesh)))

    auto make_vectorField() const
    DECL_RET_TYPE((traits::make_field<nTuple<value_type, 3>, iform>(*mesh)))


};

TYPED_TEST_CASE_P(TestField);

TYPED_TEST_P(TestField, assign)
{


    typedef typename TestFixture::value_type value_type;
    typedef typename TestFixture::field_type field_type;

    auto f1 = TestFixture::make_field();

    value_type va;

    va = 2.0;

    f1 = va;

    size_t count = 0;

    for (auto s : TestFixture::domain())
    {
        ++count;
        EXPECT_LE(mod(va - f1[s]), EPSILON);
    }
    EXPECT_EQ(count, TestFixture::domain().size());
}

TYPED_TEST_P(TestField, index)
{


    typedef typename TestFixture::value_type value_type;
    typedef typename TestFixture::field_type field_type;

    auto f1 = TestFixture::make_field();

    f1.clear();

    value_type va;

    va = 2.0;

    for (auto s : TestFixture::domain())
    {
        f1[s] = va * TestFixture::mesh->hash(s);
    }

    for (auto s : TestFixture::domain())
    {
        EXPECT_LE(mod(va * TestFixture::mesh->hash(s) - f1[s]), EPSILON);
    }

}

TYPED_TEST_P(TestField, constant_real)
{


    typedef typename TestFixture::value_type value_type;
    typedef typename TestFixture::field_type field_type;

    auto f1 = TestFixture::make_field();
    auto f2 = TestFixture::make_field();
    auto f3 = TestFixture::make_field();

    f3 = 1;
    Real a, b, c;
    a = 1.0, b = -2.0, c = 3.0;

    value_type va, vb;

    va = 2.0;
    vb = 3.0;

    f1 = va;
    f2 = vb;

    LOG_CMD(f3 = -f1 * a + f2 * c - f1 / b - f1);

    for (auto s : TestFixture::domain())
    {
        value_type res;
        res = -f1[s] * a + f2[s] * c - f1[s] / b - f1[s];

        EXPECT_LE(mod(res - f3[s]), EPSILON)
//			<<res << " " << f1[s] << " " << f2[s] << " " << f3[s];
                            ;
    }

}

TYPED_TEST_P(TestField, scalarField)
{


    typedef typename TestFixture::value_type value_type;

    auto f1 = TestFixture::make_field();
    auto f2 = TestFixture::make_field();
    auto f3 = TestFixture::make_field();
    auto f4 = TestFixture::make_field();

    auto a = TestFixture::make_scalarField();
    auto b = TestFixture::make_scalarField();
    auto c = TestFixture::make_scalarField();

    Real ra = 1.0, rb = 10.0, rc = 100.0;

    value_type va, vb, vc;

    va = ra;
    vb = rb;
    vc = rc;

    a = ra;
    b = rb;
    c = rc;

    f1.deploy();
    f2.deploy();
    f3.deploy();
    f4.deploy();

    size_t count = 0;

    std::mt19937 gen;
    std::uniform_real_distribution<Real> uniform_dist(0, 1.0);

    for (auto s: f1.range())
    {
        f1[s] = va * uniform_dist(gen);
    }
    for (auto s: f2.range())
    {
        f2[s] = vb * uniform_dist(gen);
    }

    for (auto s:   f3.range())
    {
        f3[s] = vc * uniform_dist(gen);
    }

    LOG_CMD(f4 = -f1 * a + f2 * b - f3 / c - f1);

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
    count = 0;

    for (auto s : TestFixture::domain())
    {
        value_type res = -f1[s] * ra + f2[s] * rb - f3[s] / rc - f1[s];

        EXPECT_LE(mod(res - f4[s]), EPSILON) << "s= " << (TestFixture::mesh->hash(s));
    }

    EXPECT_EQ(0, count) << "number of error points =" << count;

}

REGISTER_TYPED_TEST_CASE_P(TestField, assign, index, constant_real, scalarField);

//#include <gtest/gtest.h>
//
//#include "field.h"
//#include "../geometry/domain_traits.h"
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
//		std::tuple<typename domain_traits<Domain>::coordinate_tuple,
//				typename Domain::coordinate_tuple,
//				nTuple<Domain::NDIMS, size_t>, nTuple<Domain::NDIMS, Real> > >
//{
//
//protected:
//	void SetUp()
//	{
//		LOGGER.set_stdout_level(LOG_INFORM);
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
//		geometry.set_dimensions(dims);
//		geometry.set_extents(xmin, xmax);
//
//		geometry.update();
//
//	}
//public:
//
//	typedef Domain domain_type;
//	typedef Real value_type;
//	typedef domain_type::scalar_type scalar_type;
//	typedef domain_type::iterator iterator;
//	typedef domain_type::coordinate_tuple coordinate_tuple;
//
//	domain_type geometry;
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
//	void SetDefaultValue(nTuple<T,N>* v)
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
