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
	size_t cache_depth_ = 10;
	std::string path_;
public:
	template<typename ...Args>
	ContainerSaveCache(Args&& ... args) :
			container_type(std::forward<Args>(args)...), tail_(0), cache_()
	{
	}
	~ContainerSaveCache()
	{

	}

	std::string flush()
	{
		return save(path_, true);
	}
	size_t cache_depth() const
	{
		return cache_depth_;

	}
	void cache_depth(size_t d)
	{
		cache_depth_ = d;

	}

	std::string save(std::string const & path, bool is_forced = false)
	{
		path_ = path;

		if (!is_forced)
		{
			if (tail_ + container_type::size() > cache_.size())
			{
				cache_.resize(tail_ + container_type::size());
			}

			std::copy(container_type::begin(), container_type::end(),
					&cache_[tail_]);

			tail_ += container_type::size();
		}

		if (tail_ / container_type::size() >= cache_depth_ || is_forced)
		{
			size_t dims[2] =
			{ tail_ / container_type::size(), container_type::size() };
			tail_ = 0;
			return GLOBAL_DATA_STREAM.write(path, &cache_[0],make_datatype<TV>(),
					2,nullptr,dims,nullptr,nullptr,nullptr,nullptr, DataStream::SP_CACHE |DataStream::SP_APPEND);
		}
		return "";
	}
};
}  // namespace simpla

#endif /* CONTAINER_SAVE_CACHE_H_ */
