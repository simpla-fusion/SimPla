/*
 * pic_gague.h
 *
 *  Created on: 2013年10月15日
 *      Author: salmon
 */

#ifndef PIC_GAGUE_H_
#define PIC_GAGUE_H_

namespace simpla
{

struct PStr_GGauge
{
	nTuple<3, Real> x;
	nTuple<3, Real> v;
	Real w[];
};

template<typename T> struct PICTraits;

template<>
struct PICTraits<PStr_GGauge>
{
	inline static size_t size_in_bytes(int num_of_mates)
	{
		return (sizeof(PStr_GGauge) + sizeof(Real) * num_of_mates);
	}
};

template<typename TE, typename TB> inline
void push(PStr_GGauge & p, TE const & E, TB const &B)
{
}
template<typename TF> inline
void scatterJ(PStr_GGauge const& p, TF *J)
{
}
template<typename TF> inline
void scatterN(PStr_GGauge const& p, TF *n)
{
}

}  // namespace simpla

#endif /* PIC_GAGUE_H_ */
