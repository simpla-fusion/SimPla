/*
 * @file field.h
 *
 * @date  2013-7-19
 * @author  salmon
 */

#ifndef FIELD_H_
#define FIELD_H_

#include <string>
#include <type_traits>
#include <vector>
#include <utility> //for move

#include "../utilities/log.h"
#include "../parallel/parallel.h"
namespace simpla
{
template<typename ...> struct Field;
template<typename, unsigned int> struct Domain;
/**
 * \ingroup FETL
 * @class Field
 * \brief Field object
 *
 */
template<typename TM, unsigned int IFORM, typename TV>
struct Field<Domain<TM, IFORM>, TV>
{

public:

	typedef Domain<TM, IFORM> domain_type;

	typedef TV storage_type;

	typedef typename storage_type::value_type value_type;

	typedef Field<domain_type, storage_type> this_type;

	typedef typename domain_type::coordinates_type coordinates_type;

	typedef typename domain_type::index_type index_type;

	static constexpr unsigned int iform = domain_type::iform;

private:
	domain_type domain_;

	std::shared_ptr<storage_type> data_;
public:

	Field() = default; // default constructor

	/**
	 *
	 *  \brief Copy Constructer
	 *  Copy/clone Construct only copy mesh reference, but do not copy/move data, which is designed to
	 *  initializie stl containers, such as std::vector
	 *    \code
	 *       Field<...> default_value(mesh);
	 *       std::vector<Field<...> > v(4,default_value);
	 *    \endcode
	 *  the element in v have same mesh reference.
	 *
	 * @param rhs
	 */
	Field(this_type const & rhs) = default;

	Field(this_type && rhs) = default; //< move constructor

	~Field() = default;

	friend void swap(this_type &l, this_type & r);

	/// @defgroup Capacity Capacity
	/// @{

	bool empty() const
	{
		return !data_;
	}
	void allocate()
	{
		if (empty())
			std::shared_ptr<storage_type>(new storage_type(domain_.max_hash())).swap(
					data_);
	}
	void clear()
	{
		std::shared_ptr<storage_type>(nullptr).swap(data_);
	}

	storage_type & data()
	{
		return *(data_->get());
	}
	storage_type const & data() const
	{
		return *(data_->get());
	}
	///@}

	/// @defgroup DomainSplit Domain and Split
	/// @{

	Field(domain_type const &d) :
			domain_(d)
	{
	}
	Field(std::shared_ptr<storage_type> data, domain_type const &d) :
			domain_(d), data_(data)
	{
	}

	this_type split(domain_type const &d)
	{
		return std::move(this_type(data_, domain_ & d));
	}
	this_type boundary()
	{
		return std::move(this_type(data_, domain_.boundary()));
	}

	const domain_type & domain() const
	{
		return domain_;
	}
	void domain(domain_type d)
	{
		clear();
		swap(domain_, d);
	}
	/// @}

	/// @defgroup  Element access
	/// @{
	inline value_type & at(index_type s)
	{
		return data_->at(domain_.hash(s));
	}

	inline value_type const & at(index_type s) const
	{
		return data_->at(domain_.hash(s));
	}

	inline value_type & operator[](index_type s)
	{
		return data_->operator[](domain_.hash(s));
	}

	inline value_type const & operator[](index_type s) const
	{
		return data_->operator[](domain_.hash(s));
	}

	template<typename ... Args>
	inline void scatter(Args && ... args)
	{
		domain_.scatter(*this, std::forward<Args>(args)...);
	}
	inline auto gather(coordinates_type const &x) const
	DECL_RET_TYPE( (this->domain_.gather( *this, x)))

	inline auto operator()(coordinates_type const &x) const
	DECL_RET_TYPE( (this->domain_.gather( *this, x)))
	/// @}

	/// @defgroup Assignment
	/// @{

	template<typename T>
	void fill(T const &v)
	{
		*this = v;
	}

	this_type & operator=(this_type rhs)//< copy and swap assignment operator
	{
		swap(*this, rhs);
		return *this;
	}

	template<typename TR>
	this_type & operator=(TR const & rhs)
	{
		assign(rhs);
		return *this;
	}

	template<typename TR>
	void assign(TR const & rhs)
	{
		allocate();

		domain_.assign(*this, rhs);

	}

	template<typename TR> inline this_type &
	operator +=(TR const & rhs)
	{
		*this = *this + rhs;
		return (*this);
	}
	template<typename TR> inline this_type &
	operator -=(TR const & rhs)
	{
		*this = *this - rhs;
		return (*this);
	}
	template<typename TR> inline this_type &
	operator *=(TR const & rhs)
	{
		*this = *this * rhs;
		return (*this);
	}
	template<typename TR> inline this_type &
	operator /=(TR const & rhs)
	{
		*this = *this / rhs;
		return (*this);
	}

//	template<typename TR> inline bool operator ==(TR const & rhs)
//	{
//		return (reduce(Field<_impl::equal_to, this_type, TR>(*this, rhs),
//				_impl::logical_and(), true));
//	}

	/// @}

};

template<typename TDomain, typename TV>
void swap(Field<TDomain, TV> &l, Field<TDomain, TV> &r)
{
	swap(l.domain_, r.domain_);
	swap(l.data_, r.data_);
}

template<typename TD, typename TExpr, typename TI>
auto get_value(Field<TD, TExpr> const & f, TI const & s)
DECL_RET_TYPE ((f[s]))

}
// namespace simpla

#endif /* FIELD_H_ */
