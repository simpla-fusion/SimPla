/*
 * field.h
 *
 *  Created on: 2013-7-19
 *      Author: salmon
 */

#ifndef FIELD_H_
#define FIELD_H_

#include <iostream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>
#include <utility>
#include <mutex>

#include "primitives.h"
#include "../parallel/parallel.h"
#include "../utilities/log.h"
#include "../utilities/range.h"
namespace simpla
{
template<typename TG, int IFORM, typename TValue> struct Field;

/***
 *
 * @brief Field
 *
 * @ingroup Field Expression
 *
 */

template<typename TM, int IFORM, typename TValue>
struct Field
{
	std::mutex write_lock_;
public:

	typedef TM mesh_type;

	static constexpr unsigned int IForm = IFORM;

	typedef TValue value_type;

	typedef Field<mesh_type, IForm, value_type> this_type;

	static const int NDIMS = mesh_type::NDIMS;

	typedef typename mesh_type::coordinates_type coordinates_type;

	typedef typename mesh_type::iterator index_type;

	typedef std::shared_ptr<value_type> container_type;

	typedef typename std::conditional<(IForm == VERTEX || IForm == VOLUME),  //
	        value_type, nTuple<NDIMS, value_type> >::type field_value_type;

	container_type data_;

	mesh_type const &mesh;

	Field(mesh_type const &pmesh)
			: mesh(pmesh), data_(nullptr)
	{
	}

	/**
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

	Field(this_type const & rhs)
			: mesh(rhs.mesh), data_(nullptr)
	{
	}

	/// Move Construct copy mesh, and move data,
	Field(this_type &&rhs)
			: mesh(rhs.mesh), data_(rhs.data_)
	{
	}

	~Field()
	{
	}

	template<typename TVistor>
	void Accept(TVistor const & visitor)
	{
		visitor.Visit(this);
	}

	void swap(this_type & rhs)
	{
		ASSERT(mesh == rhs.mesh);

		std::swap(data_, rhs.data_);
	}

	void Init()
	{
		AllocMemory_();
	}

	void AllocMemory_()
	{
		if (data_ == nullptr)
		{
			data_ = mesh.template MakeContainer<IForm, value_type>();
		}

	}

	template<typename ...Args>
	int GetDataSetShape(Args ...others) const
	{
		return mesh.GetDataSetShape(IForm, std::forward<Args >(others)...);
	}

	container_type & data()
	{
		return data_;
	}

	const container_type & data() const
	{
		return data_;
	}
	size_t size() const
	{
		return mesh.GetNumOfElements(IForm);
	}
	bool empty() const
	{
		return data_ == nullptr;
	}

	void lock()
	{
		write_lock_.lock();
	}
	void unlock()
	{
		write_lock_.unlock();
	}

	inline value_type & at(index_type s)
	{
		if (!mesh.CheckLocalMemoryBounds(IForm, s))
			OUT_RANGE_ERROR << ((mesh.Decompact(s.self_) >> mesh.D_FP_POS) - mesh.local_outer_start_);

		return get(s);
	}

	inline value_type const & at(index_type s) const
	{
		if (!mesh.CheckLocalMemoryBounds(IForm, s))
			OUT_RANGE_ERROR << ((mesh.Decompact(s.self_) >> mesh.D_FP_POS) - mesh.local_outer_start_);

		return get(s);
	}

	inline value_type & get(index_type s)
	{

		return *(data_.get() + mesh.Hash(s));
	}

	inline value_type const & get(index_type s) const
	{
		return *(data_.get() + mesh.Hash(s));
	}

	inline value_type & operator[](index_type s)
	{
		return get(s);
	}

	inline value_type const & operator[](index_type s) const
	{
		return get(s);
	}
	template<typename ... Args>
	auto Select()
	DECL_RET_TYPE((make_mapped_range( *this, mesh.Select(IForm ))))
	template<typename ... Args>
	auto Select() const
	DECL_RET_TYPE((make_mapped_range( *this, mesh.Select(IForm ))))

	template<typename ... Args>
	auto Select(Args const & ... args)
	DECL_RET_TYPE((make_mapped_range( *this, mesh.Select(IForm,std::forward<Args const &>(args)...))))
	template<typename ... Args>
	auto Select(Args const & ... args) const
	DECL_RET_TYPE((make_mapped_range( *this, mesh.Select(IForm,std::forward<Args const &>(args)...))))

	auto begin() DECL_RET_TYPE(this->Select().begin())
	auto begin() const DECL_RET_TYPE(this->Select().begin())
	auto end() DECL_RET_TYPE(this->Select().end())
	auto end() const DECL_RET_TYPE(this->Select().end())

	template<typename TD>
	void Fill(TD default_value)
	{
		AllocMemory_();

		ParallelForEach(mesh.Select(IForm),

		[this,default_value](index_type s)
		{
			this->get(s) = default_value;
		}

		);
	}

	void Clear()
	{
		Fill(0);
	}

	this_type & operator =(value_type rhs)
	{
		Fill(rhs);
		return (*this);
	}
	this_type & operator =(this_type const & rhs)
	{
		AllocMemory_();

		ParallelForEach(mesh.Select(IForm),

		[this,&rhs](index_type s)
		{
			this->get(s) = rhs.get(s);
		}

		);

		return (*this);
	}
	template<typename TR>
	this_type & operator =(Field<mesh_type, IForm, TR> const & rhs)
	{
		AllocMemory_();

		ParallelForEach(mesh.Select(IForm),

		[this,&rhs](index_type s)
		{
			this->get(s) = rhs.get(s);
		}

		);

		UpdateGhosts(this);

		return (*this);
	}

#define DECL_SELF_ASSIGN( _OP_ )                                                                   \
		template<typename TR> inline this_type &                                                   \
		operator _OP_##= (TR const & rhs)                                                          \
		{	AllocMemory_(); *this = *this _OP_ rhs;                                                      \
			return (*this) ;                                                                        \
		}                                                                                          \


	DECL_SELF_ASSIGN(+ )

DECL_SELF_ASSIGN	(- )

	DECL_SELF_ASSIGN(* )

	DECL_SELF_ASSIGN(/ )
#undef DECL_SELF_ASSIGN

	inline field_value_type operator()(coordinates_type const &x) const
	{
		return mesh.Gather(*this,x);
	}

}
;

//	template<typename TC>
//	struct iterator_
//	{
//		/// One of the @link iterator_tags tag types@endlink.
//		typedef std::output_iterator_tag iterator_category;
//		/// The type "pointed to" by the iterator.
//		typedef typename std::remove_reference<decltype(*std::declval<TC>() )>::type value_type;
//		/// Distance between iterators is represented as this type.
//		typedef size_t difference_type;
//		/// This type represents a pointer-to-value_type.
//		typedef value_type* pointer;
//		/// This type represents a reference-to-value_type.
//		typedef value_type& reference;
//
//		TC data_;
//
//		mesh_type const & mesh;
//
//		typename mesh_type::iterator it_;
//
//		typedef iterator_<TC> this_type;
//
//		iterator_(TC d, mesh_type const & m, typename mesh_type::iterator s) :
//				data_(d), mesh(m), it_(s)
//		{
//
//		}
//		iterator_(this_type &rhs)
//				: data_(rhs.data_), it_(rhs.it_)
//		{
//		}
//		iterator_(this_type const&rhs)
//				: data_(rhs.data_), it_(rhs.it_)
//		{
//		}
//		~iterator_()
//		{
//		}
//
//		value_type & operator*()
//		{
//			return *(data_.get() + mesh.Hash(it_));
//		}
//		value_type const& operator*() const
//		{
//			return *(data_.get() + mesh.Hash(it_));
//		}
//
//		pointer operator ->()
//		{
//			return (data_.get() + make_hash(it_));
//		}
//		pointer operator ->() const
//		{
//			return (data_.get() + make_hash(it_));
//		}
//
//		this_type & operator++()
//		{
//			++it_;
//
//			return *this;
//		}
//
//		this_type operator++(int)
//		{
//			this_type res(*this);
//			++res;
//			return std::move(res);
//		}
//
//		bool operator==(this_type const &rhs) const
//		{
//			return it_ == rhs.it_ && data_ == rhs.data_;
//		}
//
//		bool operator!=(this_type const &rhs) const
//		{
//			return it_ != rhs.it_ || data_ != rhs.data_;
//		}
//	};
//
//	typedef iterator_<container_type> iterator;
//
//	typedef iterator_<const container_type> const_iterator;

//	iterator begin()
//	{
//		AllocMemory_();
//		return iterator_<container_type>(data_, mesh, mesh.Select(IForm).begin());
//	}
//
//	iterator end()
//	{
//		AllocMemory_();
//		return iterator_<container_type>(data_, mesh, mesh.Select(IForm).end());
//	}
//
//	const_iterator begin() const
//	{
//		return iterator_<const container_type>(data_, mesh, mesh.Select(IForm).begin());
//	}
//
//	const_iterator end() const
//	{
//		return iterator_<const container_type>(data_, mesh, mesh.Select(IForm).end());
//	}

}
// namespace simpla

#endif /* FIELD_H_ */
