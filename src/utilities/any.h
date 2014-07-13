/**
 * \file any.h
 *
 * \date    2014年7月13日  上午7:18:19 
 * \author salmon
 */

#ifndef ANY_H_
#define ANY_H_

#include <memory>
#include <typeindex>
#include <iostream>
#include "log.h"
namespace simpla
{
/**
 *   change from http://www.cnblogs.com/qicosmos/p/3420095.html
 */
struct Any
{
	template<typename U> Any(U && value)
			: m_ptr(new Derived<typename std::decay<U>::type>(std::forward<U>(value))), m_tpIndex(
			        std::type_index(typeid(typename std::remove_pointer<U>::type)))
	{
	}
	Any(void)
			: m_tpIndex(std::type_index(typeid(void)))
	{
	}
	Any(Any& that)
			: m_ptr(that.clone()), m_tpIndex(that.m_tpIndex)
	{
	}
	Any(Any const& that)
			: m_ptr(that.clone()), m_tpIndex(that.m_tpIndex)
	{
	}
	Any(Any && that)
			: m_ptr(std::move(that.m_ptr)), m_tpIndex(that.m_tpIndex)
	{
	}
	void swap(Any & other)
	{
		std::swap(m_ptr, other.m_ptr);
		std::swap(m_tpIndex, other.m_tpIndex);
	}
	bool empty() const
	{
		return !bool(m_ptr);
	}
	inline bool IsNull() const
	{
		return empty();
	}
	operator bool() const
	{
		return !empty();
	}

	template<class U> bool is() const
	{
		return m_tpIndex == std::type_index(typeid(U));
	}
	template<class U>
	U& as()
	{
		if (!is<U>())
		{
			WARNING << "can not cast " << typeid(U).name() << " to " << m_tpIndex.name() << std::endl;
			throw std::bad_cast();
		}
		auto derived = dynamic_cast<Derived<U>*>(m_ptr.get());
		return derived->m_value;
	}

	template<class U>
	U const& as() const
	{
		if (!is<U>())
		{
			WARNING << "Can not cast " << typeid(U).name() << " to " << m_tpIndex.name() << std::endl;
			throw std::bad_cast();
		}
		auto derived = dynamic_cast<Derived<U> const*>(m_ptr.get());
		return derived->m_value;
	}
	Any& operator=(const Any& a)
	{
		if (m_ptr == a.m_ptr)
			return *this;
		m_ptr = a.clone();
		m_tpIndex = a.m_tpIndex;
		return *this;
	}
	template<typename T>
	Any& operator=(T const & v)
	{
		if (is<T>())
		{
			as<T>() = v;
		}
		else
		{
			Any(v).swap(*this);
		}
		return *this;
	}
private:
	struct Base;
	typedef std::unique_ptr<Base> BasePtr;
	struct Base
	{
		virtual ~Base()
		{
		}
		virtual BasePtr clone() const = 0;
	};
	template<typename T>
	struct Derived: Base
	{
		template<typename U>
		Derived(U && value)
				: m_value(std::forward<U>(value))
		{
		}
		BasePtr clone() const
		{
			return BasePtr(new Derived<T>(m_value));
		}
		T m_value;
	};
	BasePtr clone() const
	{
		if (m_ptr != nullptr)
			return m_ptr->clone();
		return nullptr;
	}
	BasePtr m_ptr;
	std::type_index m_tpIndex;
};
}

#endif /* ANY_H_ */
