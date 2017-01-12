//
// Created by salmon on 17-1-12.
//
#include <gtest/gtest.h>
#include <simpla/concept/CheckConcept.h>
#include <iostream>

struct Foo {
    typedef double value_type;
    typedef std::true_type is_foo;
    static constexpr int iform = 2;
    int data;
};
struct Goo {
    typedef int value_type;
    typedef std::false_type is_foo;
    static constexpr double iform = 2.1;
    double data;
};
struct Koo {};

CHECK_MEMBER_TYPE(value_type, value_type)
TEST(CheckConceptTest, CheckMemberType) {
    EXPECT_TRUE(has_value_type<Foo>::value);
    EXPECT_TRUE((std::is_same<value_type_t<Foo>, double>::value));
    EXPECT_FALSE((std::is_same<value_type_t<Foo>, int>::value));
    EXPECT_FALSE((std::is_same<value_type_t<Foo>, void>::value));

    EXPECT_TRUE(has_value_type<Goo>::value);
    EXPECT_TRUE((std::is_same<value_type_t<Goo>, int>::value));

    EXPECT_FALSE(has_value_type<Koo>::value);
    EXPECT_TRUE((std::is_same<value_type_t<Koo>, void>::value));
}

CHECK_MEMBER_TYPE_BOOLEAN(is_foo, is_foo)
TEST(CheckConceptTest, CheckMemberTypeBoolean) {
    EXPECT_TRUE(is_foo<Foo>::value);
    EXPECT_FALSE(is_foo<Goo>::value);
    EXPECT_FALSE(is_foo<Koo>::value);
}

CHECK_MEMBER_STATIC_CONSTEXPR_DATA(iform, iform)
TEST(CheckConceptTest, CheckMemberStaticConstexprData) {
    EXPECT_TRUE(has_iform<Foo>::value);
    EXPECT_TRUE(has_iform<Goo>::value);
    EXPECT_FALSE(has_iform<Koo>::value);
}
CHECK_MEMBER_STATIC_CONSTEXPR_DATA_VALUE(iform, iform, 12)
TEST(CheckConceptTest, CheckMemberStaticConstexprDataValue) {
    EXPECT_EQ(iform<Foo>::value, 2);
    EXPECT_DOUBLE_EQ(iform<Goo>::value, 2.1);
    EXPECT_NE(iform<Goo>::value, 1);
    EXPECT_EQ(iform<Koo>::value, 12);
}

CHECK_MEMBER_DATA(data, data)
TEST(CheckConceptTest, CheckMemberData) {
    EXPECT_TRUE((has_data<Foo>::value));
    EXPECT_TRUE((has_data<Foo, int>::value));
    EXPECT_FALSE((has_data<Foo, double>::value));
    EXPECT_FALSE((has_data<Koo>::value));
    EXPECT_FALSE((has_data<Koo, double>::value));
}