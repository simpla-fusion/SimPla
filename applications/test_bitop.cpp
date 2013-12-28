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

#define CHECK(_MSG_)  std::cout<<__STRING(_MSG_)<<" = "<< std::hex <<"0x"<< _MSG_ << std::endl;

int main(int argc, char **argv)
{
	typedef signed long shift_type;

	static constexpr int DIGITS_S_LONG = std::numeric_limits<unsigned long>::digits;

	static constexpr int DIGITS_S_SHORT = std::numeric_limits<unsigned short>::digits;

	std::cout << "DIGITS_S_LONG= " << DIGITS_S_LONG << std::endl;
	std::cout << "DIGITS_S_SHORT= " << DIGITS_S_SHORT << std::endl;

	unsigned long d = -1;

//	std::cout <<;
//
//	<< "signed short  \t d=" << std::endl
//
//	<< "unsigned short \t d= " << static_cast<unsigned short>(d) << std::endl
//
//	<< "unsigned short \t d= " << static_cast<unsigned short>(d) << std::endl;

	CHECK(d);
	CHECK(static_cast<unsigned short>(d));
	CHECK((static_cast<signed long>(d) << (DIGITS_S_LONG - DIGITS_S_SHORT)));
	CHECK(
	        static_cast<signed long>(static_cast<unsigned long>((-1L) << (DIGITS_S_LONG - DIGITS_S_SHORT))
	                >> (DIGITS_S_LONG - DIGITS_S_SHORT * 2)));

//	for (signed long d = 0; d < 65535 * 2; ++d)
//	{
//		auto r0 = (d << (DIGITS_S_LONG - DIGITS_S_SHORT * 1)) >> (DIGITS_S_LONG - DIGITS_S_SHORT);
//		auto r1 = (d << (DIGITS_S_LONG - DIGITS_S_SHORT * 2)) >> (DIGITS_S_LONG - DIGITS_S_SHORT);
//		auto r2 = (d << (DIGITS_S_LONG - DIGITS_S_SHORT * 3)) >> (DIGITS_S_LONG - DIGITS_S_SHORT);
//		std::cout << " d = " << std::setfill('0') << std::setw(8) << std::hex << d << "\t"
//
//		<< std::setfill(' ') << std::setw(8) << std::dec << r0 << "\t"
//
//		<< std::setfill(' ') << std::setw(8) << std::dec << r1 << "\t"
//
//		<< std::setfill(' ') << std::setw(4) << std::dec << r2
//
//		<< std::endl;
//	}

}
