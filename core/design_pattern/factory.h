/*
 * factory.h
 *
 *  created on: 2014-6-13
 *      Author: salmon
 */

#ifndef FACTORY_H_
#define FACTORY_H_

#include <map>

namespace simpla
{

/**
 *  @ingroup design_pattern
 * @addtogroup factory Factory
 * @{
 *  \note  Modern C++ Design, Andrei Alexandrescu , Addison Wesley 2001  Charpt 8
 */
template<typename TId, typename TProduct, typename ...Args>
struct Factory
{
	typedef TId identifier_type;

	typedef typename std::conditional<std::is_fundamental<TProduct>::value,
			TProduct, std::shared_ptr<TProduct>>::type product_type;
	typedef std::function<product_type(Args ...)> create_fun_callback;

	typedef std::map<identifier_type, create_fun_callback> CallbackMap;

	typedef typename CallbackMap::iterator iterator;

private:
	CallbackMap callbacks_;
public:

	Factory()
	{
	}

	~Factory()
	{
	}
	template<typename OS>
	OS & print(OS & os) const
	{
		for (auto const& item : callbacks_)
		{
			os << "\t" << item.first << std::endl;
		}
		return os;
	}
	size_t size() const
	{
		return callbacks_.size();
	}

	product_type create(identifier_type const &id, Args ... args) const
	{
		auto it = callbacks_.find(id);

		if (it == callbacks_.end())
		{
			RUNTIME_ERROR("Can not find id " + ToString(id));
		}
		return (it->second)(std::forward<Args>(args)...);
	}

	template<typename ... Others>
	auto Register(Others && ... args) DECL_RET_TYPE((callbacks_.insert(std::forward<Others>(args)...)))

	int Unregister(identifier_type const & id)
	{
		return callbacks_.erase(id);
	}

}
;

/** @} */
}
// namespace simpla

#endif /* FACTORY_H_ */
