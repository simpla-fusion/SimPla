/** 
 * @file integer_sequence_test.cpp
 * @author salmon
 * @date 16-5-27 - 上午8:21
 *  */
#include <gtest/gtest.h"
//#include <iostream>
#include "../integer_sequence.h"
//#include "../PrettyStream.h"

using namespace simpla;

TEST(integer_sequence, foo)
{

    EXPECT_TRUE((std::is_base_of<index_sequence<1, 2, 3, 4, 1, 2, 3, 4>,
            traits::seq_concat<index_sequence<1, 2, 3, 4>, index_sequence<1, 2, 3, 4> > >::value));


}