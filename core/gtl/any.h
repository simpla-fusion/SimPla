/**
 * @file any.h
 *
 * \date    2014年7月13日  上午7:18:19 
 * \author salmon
 */

#ifndef ANY_H_
#define ANY_H_

#include <algorithm>
#include <cstdbool>
#include <iostream>
#include <memory>
#include <string>
#include <type_traits>
#include <typeindex>
#include <typeinfo>

#ifdef simpla
#include "../dataset/datatype.h"
#include "../utilities/log.h"
#endif

namespace simpla
{

/**
 *  @ingroup gtl
 *   base on http://www.cnblogs.com/qicosmos/p/3420095.html
 */
struct Any
{
	template<typename U>
	Any(U && value) :
			ptr_(
					new Derived<typename std::decay<U>::type>(
							std::forward<U>(value)))
	{
	}
	Any(void)
	{
	}
	Any(Any& that) :
			ptr_(that.clone())
	{
	}
	Any(Any const& that) :
			ptr_(that.clone())
	{
	}
	Any(Any && that) :
			ptr_(std::move(that.ptr_))
	{
	}
	void swap(Any & other)
	{
		std::swap(ptr_, other.ptr_);
	}
	bool empty() const
	{
		return !bool(ptr_);
	}
	inline bool IsNull() const
	{
		return empty();
	}
	operator bool() const
	{
		return !empty();
	}

	void const * data() const
	{
		return ptr_->data();
	}
	void * data()
	{
		return ptr_->data();
	}

	template<class U>
	bool is_same() const
	{
		return ptr_->is_same<U>();
	}

	template<class U>
	bool as(U* v) const
	{
		bool is_found = false;
		if (is_same<U>())
		{
			*v = dynamic_cast<Derived<U>*>(ptr_.get())->m_value;
			is_found = true;
		}
		return is_found;
	}

	template<class U>
	U & as() const
	{

		if (!is_same<U>())
		{
#ifdef simpla
			LOGGER << "Can not cast " << typeid(U).name() << " to "
			<< ptr_->type_name() << runtime_error_endl;
#endif
			throw std::bad_cast();
		}
		return dynamic_cast<Derived<U> *>(ptr_.get())->m_value;
	}

	Any& operator=(const Any& a)
	{
		if (ptr_ == a.ptr_)
			return *this;
		ptr_ = a.clone();

		return *this;
	}
	template<typename T>
	Any& operator=(T const & v)
	{

		if (is_same<T>())
		{
			as<T>() = v;
		}
		else
		{
			Any(v).swap(*this);
		}
		return *this;
	}

	template<typename OS>
	OS & print(OS & os) const
	{
		return ptr_->print(os);
	}
	DataType datatype() const
	{
		return ptr_->datatype();
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

		virtual void const * data() const=0;
		virtual void * data()=0;
		virtual std::ostream & print(std::ostream & os) const=0;
		virtual bool is_same(std::type_index const &) const=0;
		virtual std::string type_name() const=0;

		template<typename T>
		bool is_same() const
		{
			return is_same(std::type_index(typeid(T)));
		}

		virtual DataType datatype() const=0;

	};
	template<typename T>
	struct Derived: Base
	{
		template<typename U>
		Derived(U && value) :
				m_value(std::forward<U>(value))
		{
		}
		BasePtr clone() const
		{
			return BasePtr(new Derived<T>(m_value));
		}

		void const * data() const
		{
			return reinterpret_cast<void const *>(&m_value);
		}
		void * data()
		{
			return reinterpret_cast<void *>(&m_value);
		}
		std::ostream & print(std::ostream & os) const
		{
			os << m_value;
			return os;
		}
		bool is_same(std::type_index const &t_idx) const
		{
			return std::type_index(typeid(T)) == t_idx;
		}
		std::string type_name() const
		{
			return typeid(T).name();
		}

		T m_value;

		DataType datatype() const
		{
			return traits::datatype<T>::create();
		}

	};
	BasePtr clone() const
	{
		if (ptr_ != nullptr)
			return ptr_->clone();
		return nullptr;
	}

	BasePtr ptr_;
};

} // namespace simpla

#endif /* ANY_H_ */
