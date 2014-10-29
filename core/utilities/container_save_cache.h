/*
 * container_save_cache.h
 *
 *  Created on: 2014年7月12日
 *      Author: salmon
 */

#ifndef CONTAINER_SAVE_CACHE_H_
#define CONTAINER_SAVE_CACHE_H_
#include "../io/data_stream.h"
namespace simpla
{

template<typename TV>
class ContainerSaveCache: public std::vector<TV>
{
	typedef std::vector<TV> container_type;

	typedef TV value_type;

	std::vector<value_type> cache_;

	size_t tail_;

public:
	template<typename ...Args>
	ContainerSaveCache(Args&& ... args) :
			container_type(std::forward<Args>(args)...), tail_(0), cache_(
					container_type::size())
	{
	}
	~ContainerSaveCache()
	{

	}

	void set_depth(size_t d)
	{
		cache_.resize(container_type::size() * d);
	}

	std::string save(std::string const & name)
	{

		std::copy(container_type::begin(), container_type::end(),
				&cache_[tail_]);

		tail_ += container_type::size();

		if (tail_ >= cache_.size())
		{
			size_t dims[2] = { cache_.size() / container_type::size(),
					container_type::size() };
			return GLOBAL_DATA_STREAM.write(name, &cache_[0],DataType::create<TV>(),2,nullptr,dims,nullptr,nullptr,nullptr,nullptr, true);
		}
		return "";
	}
};
}  // namespace simpla

#endif /* CONTAINER_SAVE_CACHE_H_ */
