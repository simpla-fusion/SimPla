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
#include "../src/utilities/pretty_stream.h"

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
	nTuple<3, unsigned int> d = { 1, 2, 4 };
	OcForest ocf(d);
	CHECK(ocf.index_digits_);CHECK(ocf.GetDimensions());

//	CHECK( OcForest::dh );

	CHECK_DEC(1.0 / OcForest::dh);

	CHECK_DEC(OcForest::INDEX_MAX);

	CHECK_DEC(OcForest::INDEX_MIN);

	CHECK_DEC(OcForest::DIGITS_HEAD);

	CHECK_DEC(OcForest::INDEX_DIGITS);

	size_t r[3] = { 10000, 100, 1 };

	nTuple<3, Real> x = { 1.0 - 0.125, 1.0, 0.125 };

	OcForest::index_type s = ocf.GetIndex(x);

	int TDEPTH = OcForest::MAX_TREE_HEIGHT;

	OcForest::index_type v[OcForest::MAX_NUM_VERTEX_PER_CEL];

	s = ocf.GetIndex(x);

	CHECK_BIT(OcForest::_C(s));

	CHECK(ocf.GetCoordinates(s));

	ocf.GetAdjacentCells(Int2Type<VERTEX>(), Int2Type<EDGE>(), s, v);

	CHECK_BIT(OcForest::_C(v[0]));CHECK(ocf.GetCoordinates(v[0]));

	CHECK_BIT(OcForest::_C(v[1]));CHECK(ocf.GetCoordinates(v[1]));

	CHECK_BIT(OcForest::_C(v[2]));CHECK(ocf.GetCoordinates(v[2]));

	CHECK_BIT(OcForest::_C(v[3]));CHECK(ocf.GetCoordinates(v[3]));

	CHECK_BIT(OcForest::_C(v[4]));CHECK(ocf.GetCoordinates(v[4]));

	CHECK_BIT(OcForest::_C(v[5]));CHECK(ocf.GetCoordinates(v[5]));

	ocf.GetAdjacentCells(Int2Type<EDGE>(), Int2Type<VERTEX>(), s, v);

	CHECK_BIT(OcForest::_C(v[0]));

	CHECK(ocf.GetCoordinates(v[0]));

	CHECK_BIT(OcForest::_C(v[1]));

	CHECK(ocf.GetCoordinates(v[1]));

	ocf.GetAdjacentCells(Int2Type<FACE>(), Int2Type<VERTEX>(), s, v);
	CHECK_BIT(OcForest::_C(s));
	CHECK_BIT(OcForest::_C(v[0]));
	CHECK_BIT(OcForest::_C(v[1]));
	CHECK_BIT(OcForest::_C(v[2]));
	CHECK_BIT(OcForest::_C(v[3]));

	CHECK(ocf.GetCoordinates(s));

	CHECK(ocf.GetCoordinates(v[0]));

	CHECK(ocf.GetCoordinates(v[1]));

	CHECK(ocf.GetCoordinates(v[2]));

	CHECK(ocf.GetCoordinates(v[3]));

}

