/*
 * properties_impl.h
 *
 *  Created on: 2012-10-28
 *      Author: salmon
 */

#ifndef PROPERTIES_IMPL_H_
#define PROPERTIES_IMPL_H_

namespace simpla
{

class ptree::iterator
{

public:
	typedef typename baset::reference reference;
	iterator()
	{
	}
	explicit iterator(typename iterator::base_type b) :
			iterator::iterator_adaptor_(b)
	{
	}
	reference dereference() const
	{
		// multi_index doesn't allow modification of its values, because
		// indexes could sort by anything, and modification screws that up.
		// However, we only sort by the key, and it's protected against
		// modification in the value_type, so this const_cast is safe.
		return const_cast<reference>(*this->base_reference());
	}
};

}  // namespace simpla

#endif /* PROPERTIES_IMPL_H_ */
