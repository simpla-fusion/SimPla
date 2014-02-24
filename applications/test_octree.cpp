/*
 * test_octree.cpp
 *
 *  Created on: 2014年2月24日
 *      Author: salmon
 */

#include <bitset>
#include <iomanip>
#include <iostream>
#include <string>

#include "../src/mesh/octree_forest.h"

using namespace simpla;

std::string ShowBit(unsigned long s)
{
	return std::bitset<64>(s).to_string();
}
#define CHECK_BIT(_MSG_)  std::cout<<std::setfill(' ')<<std::setw(30) <<__STRING(_MSG_)<<" = 0b"<<ShowBit( _MSG_) << std::endl

#define CHECK_HEX(_MSG_)  std::cout<<std::setfill(' ')<<std::setw(30) <<__STRING(_MSG_)<<" = 0x"<<std::setw(20)<<std::setfill('0')<< std::hex<< ( _MSG_) << std::dec<< std::endl

//<< std::hex<<std::setw(20)<<std::setfill('0')
#define CHECK_DEC(_MSG_)  std::cout<<std::setfill(' ')<<std::setw(30) <<__STRING(_MSG_)<<" =  " << std::dec<<_MSG_ << std::endl

int main(int argc, char **argv)
{
#define TDEPTH 1
	OcForest<TDEPTH> foo;
//	foo.SetTreeDepth(4);

	std::cout << OcForest<TDEPTH>::dh << std::endl;

	CHECK_DEC(1.0 / OcForest<TDEPTH>::dh);

	CHECK_DEC(OcForest<TDEPTH>::INDEX_MAX);

	CHECK_DEC(OcForest<TDEPTH>::INDEX_MIN);

	CHECK_DEC(OcForest<TDEPTH>::DIGITS_HEAD);

	CHECK_DEC(OcForest<TDEPTH>::DIGITS_INDEX);

	CHECK_BIT(foo.MASK_A);
	CHECK_BIT(foo.MASK_I);
	CHECK_BIT(foo.MASK_J);
	CHECK_BIT(foo.MASK_K);

	size_t r[3] = { 10000, 100, 1 };

	nTuple<3, Real> x = { 1.0 - 0.125, 1.0, 0.125 };

	OcForest<TDEPTH>::index_type s = foo.GetIndex(x);

//	CHECK_HEX(s.s);
//
//	CHECK_BIT(s.s);
//
//	CHECK_BIT(s.ijk.I);
//	CHECK_BIT(s.ijk.J);
//	CHECK_BIT(s.ijk.K);
//
//	CHECK_DEC(foo.HashRootIndex(s, r, 3));
//
	OcForest<TDEPTH>::index_type v[4];

	s.I = (1L << 4);
	s.J = (1L << 4) + 1;
	s.K = (1L << 4);
	foo.GetAdjacentCells(Int2Type<EDGE>(), Int2Type<VERTEX>(), s, v);
	CHECK_BIT(OcForest<TDEPTH>::_C(s));
	CHECK_BIT(OcForest<TDEPTH>::_C(v[0]));
	CHECK_BIT(OcForest<TDEPTH>::_C(v[1]));

	s.I = (0L << 4);
	s.J = (1L << 4) + 1;
	s.K = (1L << 4) + 1;
	foo.GetAdjacentCells(Int2Type<FACE>(), Int2Type<VERTEX>(), s, v);
	CHECK_BIT(OcForest<TDEPTH>::_C(s));
	CHECK_BIT(OcForest<TDEPTH>::_C(v[0]));
	CHECK_BIT(OcForest<TDEPTH>::_C(v[1]));
	CHECK_BIT(OcForest<TDEPTH>::_C(v[2]));
	CHECK_BIT(OcForest<TDEPTH>::_C(v[3]));
}

