/**
 * @file any.h
 *
 * \date    2014年7月13日  上午7:18:19 
 * \author salmon
 */

#ifndef ANY_H_
#define ANY_H_

#include <algorithm>
#include <iostream>
#include <memory>
#include <string>
#include <type_traits>
#include <typeindex>
#include <typeinfo>

#include "../data_interface/data_type.h"
#include "../utilities/log.h"
#include "ntuple.h"

namespace simpla
{
/**
 *  @ingroup gtl
 *   change from http://www.cnblogs.com/qicosmos/p/3420095.html
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

	DataType datatype() const
	{
		return std::move(ptr_->datatype());
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
	typename array_to_ntuple_convert<U>::type& as()
	{
		typedef typename array_to_ntuple_convert<U>::type U2;

		if (!is_same<U2>())
		{
			Logger(LOG_ERROR) << "Can not cast " << typeid(U).name() << " to "
					<< ptr_->type_name() << std::endl;
			throw std::bad_cast();
		}
		return dynamic_cast<Derived<U2>*>(ptr_.get())->m_value;
	}

	template<class U>
	typename array_to_ntuple_convert<U>::type const& as() const
	{
		typedef typename array_to_ntuple_convert<U>::type U2;

		if (!is_same<U2>())
		{
			Logger(LOG_ERROR) << "Can not cast " << typeid(U).name() << " to "
					<< ptr_->type_name() << std::endl;
			throw std::bad_cast();
		}

		return dynamic_cast<Derived<U2> const*>(ptr_.get())->m_value;
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
		typedef typename array_to_ntuple_convert<T>::type U2;

		if (is_same<U2>())
		{
			as<U2>() = v;
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
private:
	struct Base;
	typedef std::unique_ptr<Base> BasePtr;
	struct Base
	{
		virtual ~Base()
		{
		}
		virtual BasePtr clone() const = 0;
		virtual DataType datatype() const=0;
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
		DataType datatype() const
		{
			return make_datatype<T>();
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
	};
	BasePtr clone() const
	{
		if (ptr_ != nullptr)
			return ptr_->clone();
		return nullptr;
	}

	BasePtr ptr_;
};

}

#endif /* ANY_H_ */
