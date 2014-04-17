/*
 * test_bitop.cpp
 *
 *  Created on: 2013年12月28日
 *      Author: salmon
 */

#include <limits>
#include <type_traits>
#include <iostream>
#include <iomanip>

#define CHECK(_MSG_)  std::cout<<std::setfill(' ')<<std::setw(10) <<__STRING(_MSG_)<<" = 0x"<< std::hex<<std::setw(DIGITS_FULL/4)<<std::setfill('0')<< _MSG_ << std::endl;

static constexpr int DIGITS_FULL = std::numeric_limits<unsigned long>::digits;

static constexpr int DIGITS_INDEX = std::numeric_limits<unsigned short>::digits; //(std::numeric_limits<unsigned long>::digits) / 3; //!< signed long is 63bit, unsigned long is 64 bit, add a sign bit

static constexpr int DIGITS_HEAD = std::numeric_limits<unsigned long>::digits - DIGITS_INDEX * 3;

int main(int argc, char **argv)
{
	typedef unsigned long size_type;

	typedef unsigned long compact_index_type;
	struct IndexIJK
	{
		size_type H :DIGITS_HEAD;
		size_type I[3] :DIGITS_INDEX;

	};

	union index_type
	{
		IndexIJK ijk;
		compact_index_type s;
	};

}
