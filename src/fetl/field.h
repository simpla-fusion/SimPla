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

#include "../utilities/log.h"
#include "../utilities/primitives.h"
#include "../utilities/range.h"

#include "../parallel/parallel.h"
#include "../utilities/iterator_mapped.h"
#include "field_update_ghosts.h"

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

	typedef typename mesh_type::index_type index_type;

	typedef typename mesh_type::iterator mesh_iterator;

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
			OUT_RANGE_ERROR(mesh.Decompact(s));
		return get(s);
	}

	inline value_type const & at(index_type s) const
	{
		if (!mesh.CheckLocalMemoryBounds(IForm, s))
			OUT_RANGE_ERROR(mesh.Decompact(s));

		return get(s);
	}

	inline value_type & operator[](index_type s)
	{
		return get(s);
	}

	inline value_type const & operator[](index_type s) const
	{
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

		[this,default_value](mesh_iterator const & s)
		{
			this->get(*s) = default_value;
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

		[this,&rhs](mesh_iterator const & s)
		{
			this->get(*s) = rhs.get(*s);
		}

		);

		return (*this);
	}
	template<typename TR>
	this_type & operator =(Field<mesh_type, IForm, TR> const & rhs)
	{
		AllocMemory_();

		ParallelForEach(mesh.Select(IForm),

		[this,&rhs](mesh_iterator const & s)
		{
			this->get(*s) = rhs.get(*s);
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

template<typename TL>
struct is_field
{
	static const bool value = false;
};

template<typename TG, int IF, typename TL>
struct is_field<Field<TG, IF, TL>>
{
	static const bool value = true;
};

template<typename T>
struct is_field_expression
{
	static constexpr bool value = false;
};

template<typename TG, int IF, int TOP, typename TL, typename TR>
struct is_field_expression<Field<TG, IF, BiOp<TOP, TL, TR> > >
{
	static constexpr bool value = true;
};

template<typename TG, int IF, int TOP, typename TL>
struct is_field_expression<Field<TG, IF, UniOp<TOP, TL> > >
{
	static constexpr bool value = true;
};

template<typename TG, int IF, int TOP, typename TL, typename TR>
struct is_expression<Field<TG, IF, BiOp<TOP, TL, TR> > >
{
	static constexpr bool value = true;
};

template<typename TG, int IF, int TOP, typename TL>
struct is_expression<Field<TG, IF, UniOp<TOP, TL> > >
{
	static constexpr bool value = true;
};

}
// namespace simpla

#endif /* FIELD_H_ */
