//
// Created by salmon on 17-1-12.
//
#include <gtest/gtest.h"
#include "simpla/concept/CheckConcept.h"

struct Foo {
    typedef double value_type;
    typedef std::true_type is_foo;
    static constexpr int iform = 2;
    int data;
    static int foo(double);

    double operator+(double) const;
    double const& operator[](int) const;
    double& operator[](int);

    double operator()(int, double) const;

    template <typename... Args>
    double operator()(Args&&...);
};
struct Goo {
    typedef int value_type;
    typedef std::false_type is_foo;
    static constexpr double iform = 2.1;
    double data;
    void foo(int);
    int foo(double);
};
struct Koo {};

int foo(double, double);

int goo(double, double);

CHECK_MEMBER_TYPE(value_type, value_type)
TEST(CheckConceptTest, CheckTypeMember) {
    EXPECT_TRUE(has_value_type<Foo>::value);
    EXPECT_TRUE((std::is_same<value_type_t<Foo>, double>::value));
    EXPECT_FALSE((std::is_same<value_type_t<Foo>, int>::value));
    EXPECT_FALSE((std::is_same<value_type_t<Foo>, void>::value));

    EXPECT_TRUE(has_value_type<Goo>::value);
    EXPECT_TRUE((std::is_same<value_type_t<Goo>, int>::value));

    EXPECT_FALSE(has_value_type<Koo>::value);
    EXPECT_TRUE((std::is_same<value_type_t<Koo>, void>::value));
}

CHECK_BOOLEAN_TYPE_MEMBER(is_foo, is_foo)
TEST(CheckConceptTest, CheckMemberTypeBoolean) {
    EXPECT_TRUE(is_foo<Foo>::value);
    EXPECT_FALSE(is_foo<Goo>::value);
    EXPECT_FALSE(is_foo<Koo>::value);
}

CHECK_STATIC_CONSTEXPR_DATA_MEMBER(has_iform, iform)
TEST(CheckConceptTest, CheckMemberStaticConstexprData) {
    EXPECT_TRUE(has_iform<Foo>::value);
    EXPECT_TRUE(has_iform<Goo>::value);
    EXPECT_FALSE(has_iform<Koo>::value);
}
CHECK_VALUE_OF_STATIC_CONSTEXPR_DATA_MEMBER(iform_value, iform, 12)
TEST(CheckConceptTest, CheckMemberStaticConstexprDataValue) {
    EXPECT_EQ(iform_value<Foo>::value, 2);
    EXPECT_DOUBLE_EQ(iform_value<Goo>::value, 2.1);
    EXPECT_NE(iform_value<Goo>::value, 1);
    EXPECT_EQ(iform_value<Koo>::value, 12);
}
CHECK_DATA_MEMBER(has_data, data)
TEST(CheckConceptTest, CheckDataMember) {
    EXPECT_TRUE((has_data<Foo>::value));
    EXPECT_TRUE((has_data<Foo, int>::value));
    EXPECT_FALSE((has_data<Foo, double>::value));
    EXPECT_FALSE((has_data<Koo>::value));
    EXPECT_FALSE((has_data<Koo, double>::value));
}

CHECK_MEMBER_FUNCTION(has_member_function_foo, foo)
TEST(CheckConceptTest, CheckFunctionMember) {
    EXPECT_TRUE((has_member_function_foo<Foo, int(double)>::value));
    EXPECT_FALSE((has_member_function_foo<Foo, void(int)>::value));
    EXPECT_FALSE((has_member_function_foo<Goo, void(double)>::value));
    EXPECT_TRUE((has_member_function_foo<Goo, void(int)>::value));
    EXPECT_TRUE((has_member_function_foo<Goo, int(double)>::value));
    EXPECT_FALSE((has_member_function_foo<Koo, void(double)>::value));
}
CHECK_STATIC_FUNCTION_MEMBER(has_static_function_foo, foo)
TEST(CheckConceptTest, CheckStaticFunctionMember) {
    EXPECT_TRUE((has_static_function_foo<Foo, int(double)>::value));
    EXPECT_FALSE((has_static_function_foo<Foo, void(int)>::value));
    EXPECT_FALSE((has_static_function_foo<Goo, void(int)>::value));
    EXPECT_FALSE((has_static_function_foo<Goo, int(double)>::value));
    EXPECT_FALSE((has_static_function_foo<Koo, void(double)>::value));
}

CHECK_OPERATOR(is_plus_able, +)
TEST(CheckConceptTest, CheckOperator) {
    EXPECT_TRUE((is_plus_able<Foo, double(double)>::value));
    EXPECT_TRUE((is_plus_able<const Foo, double(double)>::value));
    EXPECT_TRUE((is_plus_able<Foo, double(int)>::value));
    EXPECT_FALSE((is_plus_able<Foo, int>::value));
    EXPECT_FALSE((is_plus_able<Goo, double(double)>::value));
}
using namespace simpla::concept;

TEST(CheckConceptTest, CheckIsIndexable) {
    EXPECT_FALSE((is_indexable<Foo, double const&(int)>::value));
    EXPECT_TRUE((is_indexable<const Foo, double const&(int)>::value));
    EXPECT_TRUE((is_indexable<Foo, double&(int)>::value));
    EXPECT_FALSE((is_indexable<Goo, double&(int)>::value));
    EXPECT_FALSE((is_indexable<Koo, double&(int)>::value));

    EXPECT_TRUE((is_indexable<int[]>::value));
    EXPECT_TRUE((is_indexable<int[][10]>::value));
    EXPECT_TRUE((is_indexable<int*>::value));
}
TEST(CheckConceptTest, CheckIsCallable) {
    EXPECT_FALSE((is_callable<Foo, double const&(int)>::value));
    EXPECT_TRUE((is_callable<const Foo, double(int, double)>::value));
    EXPECT_TRUE((is_callable<Foo, double(double, double)>::value));
    EXPECT_FALSE((is_callable<Goo, double&(int)>::value));
    EXPECT_FALSE((is_callable<Koo, double&(int)>::value));
}