/*
 * pic_delta_f.h
 *
 *  Created on: 2013年10月15日
 *      Author: salmon
 */

#ifndef PIC_DELTA_F_H_
#define PIC_DELTA_F_H_
namespace simpla
{
struct PStr_DeltaF
{
	nTuple<3, Real> x;
	nTuple<3, Real> v;
	Real f;
	Real w;
};
template<typename TE, typename TB> inline
void push(PStr_DeltaF & p, TE const & E, TB const &B)
{
}
template<typename TF> inline
void scatterJ(PStr_DeltaF const& p, TF *J)
{
}
template<typename TF> inline
void scatterN(PStr_DeltaF const& p, TF *n)
{
}

}  // namespace simpla

#endif /* PIC_DELTA_F_H_ */
