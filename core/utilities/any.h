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
namespace simpla
{
/**
 *   change from http://www.cnblogs.com/qicosmos/p/3420095.html
 */
struct Any
{
	template<typename U> Any(U && value) :
			ptr_(new Derived<typename std::decay<U>::type>(std::forward<U>(value))), t_index_(
			        std::type_index(typeid(typename std::remove_pointer<U>::type)))
	{
	}
	Any(void) :
			t_index_(std::type_index(typeid(void)))
	{
	}
	Any(Any& that) :
			ptr_(that.clone()), t_index_(that.t_index_)
	{
	}
	Any(Any const& that) :
			ptr_(that.clone()), t_index_(that.t_index_)
	{
	}
	Any(Any && that) :
			ptr_(std::move(that.ptr_)), t_index_(that.t_index_)
	{
	}
	void swap(Any & other)
	{
		std::swap(ptr_, other.ptr_);
		std::swap(t_index_, other.t_index_);
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

	template<class U> bool is() const
	{
		return t_index_ == std::type_index(typeid(U));
	}
	template<class U>
	U& as()
	{
		if (!is<U>())
		{
//			WARNING << "can not cast " << typeid(U).name() << " to " << t_index_.name() << std::endl;
			throw std::bad_cast();
		}
		auto derived = dynamic_cast<Derived<U>*>(ptr_.get());
		return derived->m_value;
	}

	template<class U>
	U const& as() const
	{
		if (!is<U>())
		{
//			WARNING << "Can not cast " << typeid(U).name() << " to " << t_index_.name() << std::endl;
			throw std::bad_cast();
		}
		auto derived = dynamic_cast<Derived<U> const*>(ptr_.get());
		return derived->m_value;
	}

	Any& operator=(const Any& a)
	{
		if (ptr_ == a.ptr_) return *this;
		ptr_ = a.clone();
		t_index_ = a.t_index_;
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

	std::ostream & print(std::ostream & os) const
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
		virtual std::ostream & print(std::ostream & os) const=0;
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

		std::ostream & print(std::ostream & os) const
		{
			os << m_value;
			return os;
		}

		T m_value;
	};
	BasePtr clone() const
	{
		if (ptr_ != nullptr) return ptr_->clone();
		return nullptr;
	}

	BasePtr ptr_;
	std::type_index t_index_;
};

inline std::ostream & operator<<(std::ostream & os, Any const& v)
{
	return v.print(os);

}
}

#endif /* ANY_H_ */
