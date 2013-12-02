/*
 * inverse_function.h
 *
 *  Created on: 2013年12月2日
 *      Author: salmon
 */

#ifndef INVERSE_FUNCTION_H_
#define INVERSE_FUNCTION_H_

namespace simpla
{
extern struct LinearInterpolation;
template<typename TK, typename TY, typename TInterpolator = LinearInterpolation>
extern struct Interpolation;

template<typename TX, typename TY, typename TInterpolation = Interpolation<TX,
		TY, LinearInterpolation> >
class InverseFunction: public TInterpolation
{
public:
	typedef InverseFunction<TX, TY, TInterpolation> this_type;
	typedef TInterpolation base_type;
	typedef std::map<TX, TY> container_type;
	typedef typename container_type::iterator iterator;
	typedef typename container_type::key_type key_type;
	typedef typename container_type::mapped_type value_type;

	InverseFunction(container_type const &xy)
	{
		inverse(xy);
		base_type::update();
	}

	~InverseFunction()
	{
	}

	void swap(this_type & r)
	{
		base_type::swap(r);
	}

	void inverse(container_type const & xy)
	{
		for (auto const &p : xy)
		{
			base_type::emplace(std::make_pair(p.second, p.first));
		}

	}
};

}  // namespace simpla

#endif /* INVERSE_FUNCTION_H_ */
