/*
 * function.h
 *
 *  Created on: 2014年12月2日
 *      Author: salmon
 */

#ifndef CORE_MODEL_FUNCTION_H_
#define CORE_MODEL_FUNCTION_H_

namespace simpla
{
template<typename TD>
struct Function
{
	typedef TD domain_type;
	typedef typename domain_type::index_type index_type;

	domain_type def_domain_;
	std::function<void(index_type)> fun_;

	void eval()
	{
		parallel_for(def_domain_, fun_);
	}
};

}  // namespace simpla

#endif /* CORE_MODEL_FUNCTION_H_ */
