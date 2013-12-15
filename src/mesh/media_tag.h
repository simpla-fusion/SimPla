/*
 * media_tag.h
 *
 *  Created on: 2013年12月15日
 *      Author: salmon
 */

#ifndef MEDIA_TAG_H_
#define MEDIA_TAG_H_

#include <algorithm>
#include <bitset>
#include <cstddef>
#include <functional>

#include "../fetl/primitives.h"
#include "../utilities/log.h"

namespace simpla
{
template<typename TM>
class MediaTag
{

public:
	static constexpr int MAX_NUM_OF_MEIDA_TYPE = 64;
	typedef TM mesh_type;

	typedef std::bitset<MAX_NUM_OF_MEIDA_TYPE> tag_type;
	typedef typename mesh_type::index_type index_type;
	typedef typename mesh_type::coordinates_type coordinates_type;
private:
	mesh_type const &mesh;
	typename mesh_type::template Container<tag_type> tags_[mesh_type::NUM_OF_COMPONENT_TYPE];
public:

	enum
	{
		NULL_TAG = 0, VACUUM = 1, PLASMA, CORE, BOUNDARY, PLATEAU,
		// @NOTE: add tags for different physical area or media
		CUSTOM = 20
	};

	MediaTag(mesh_type const & m)
			: mesh(m), tags_(mesh.MakeContainer<0, tag_type>(0))
	{
	}
	~MediaTag()
	{
	}

	tag_type GetTag(unsigned int tag_pos) const
	{
		tag_type res;
		res.set(tag_pos);
		return std::move(res);
	}

	void ClearAll()
	{
		for (auto &v : tags_[0])
		{
			v.reset();
		}

		Update();
	}

	/**
	 * Set media tag on vertics
	 * @param tag media tag is  set to 1<<tag
	 * @param args args are trans-forward to
	 *      SelectVerticsInRegion(<lambda function>,*this,args)
	 */
	template<typename ...Args>
	void Set(unsigned int media_tag, Args const & ... args)
	{

		tag_type m_tag;

		m_tag.set(media_tag);

		_ForEachVertics(

		[&](bool isSelected,tag_type &v)
		{	if(isSelected) v&=m_tag;},

		*this, std::forward<Args>(args)...);
	}

	template<typename ...Args>
	void InverseSet(unsigned int media_tag, Args const & ... args)
	{

		tag_type m_tag;

		m_tag.set(media_tag);

		_ForEachVertics(

		[&](bool isSelected,tag_type &v)
		{	if(! isSelected) v&=m_tag;},

		*this, std::forward<Args>(args)...);
	}

	template<typename ...Args>
	void SetInterface(unsigned int in_tag_pos, unsigned int out_tag_pos, Args const & ... args)
	{

		tag_type in_tag;
		in_tag.set(in_tag_pos);

		tag_type out_tag;
		out_tag.set(out_tag_pos);

		_ForEachVertics(

		[&](bool isSelected,tag_type &v)
		{
			if( isSelected)
			{
				v&=in_tag;
			}
			else
			{
				v&=out_tag_pos;
			}

		},

		*this, std::forward<Args>(args)...);

	}

	template<typename ...Args>
	void Add(unsigned int media_tag, Args const & ... args)
	{

		tag_type m_tag;

		m_tag.set(media_tag);

		_ForEachVertics(

		[&](bool isSelected,tag_type &v)
		{	if( isSelected) v.set(media_tag);},

		*this, std::forward<Args>(args)...);
	}

	template<typename ...Args>
	void Remove(unsigned int media_tag, Args const & ... args)
	{

		tag_type m_tag;

		m_tag.set(media_tag);

		_ForEachVertics(

		[&](bool isSelected,tag_type &v)
		{	if(isSelected) v.reset(media_tag);},

		*this, std::forward<Args>(args)...);
	}

	/**
	 *  Update media tag on edge ,face and cell, base on media tag on vertics
	 */
	void Update()
	{
		_UpdateTags<1>();
		_UpdateTags<2>();
		_UpdateTags<3>();
	}

	/**
	 *  Choice elements that most close to and out of the interface,
	 *  No element cross interface.
	 * @param
	 * @param fun
	 * @param in_tag
	 * @param out_tag
	 * @param flag
	 */
	template<int IFORM> inline
	void ForEachElementOnInterface(std::function<void(int, index_type)> const &fun, unsigned int in, unsigned int out,
	        unsigned int flag = 0) const
	{
		_ForEachElementOnInterface(Int2Type<IFORM>(), fun, in, out, flag);
	}

	/**
	 *   Choice elements which cross interface.
	 * @param fun
	 * @param in
	 * @param out
	 * @param flag
	 */
	template<int IFORM> inline
	void ForEachElementCrossInterface(std::function<void(int, index_type)> const &fun, unsigned int in,
	        unsigned int out, unsigned int flag = 0) const
	{
		_ForEachElementCrossInterface(Int2Type<IFORM>(), fun, in, out, flag);
	}

private:

	/**
	 * Set media tag on vertics
	 * @param tag media tag is  set to 1<<tag
	 * @param args args are trans-forward to
	 *      SelectVerticsInRegion(<lambda function>,*this,args)
	 */
	template<typename ...Args>
	void _ForEachVertics(std::function<void(bool, tag_type&)> fun, Args const & ... args)
	{
		if (tags_[0].empty())
			tags_[0].resize(mesh.GetNumOfElements(0), GetTag(VACUUM));

		SelectVericsInRegion(

		[&](bool in,index_type const &s)
		{
			fun(in,tags_[0][s]);

		}, *this, std::forward<Args>(args)...);
	}

	template<int I>
	void _UpdateTags()
	{
		if (tags_[I].empty())
			tags_[I].resize(mesh.GetNumOfElements(I), 0);

		mesh.TraversalIndex(I,

		[&](int m, index_type const & s )
		{
			index_type v[mesh_type::MAX_NUM_VERTEX_PER_CEL];

			int n=GetVerticesOfCell<I>(v,m,s);
			tag_type flag = 0;
			for(int i=0;i<n;++i)
			{
				flag!=tags_[0][v[i]];
			}
			tags_[I][mesh.GetComponentIndex<I>(m,s)]=flag;

		}, mesh_type::DO_PARALLEL);
	}

	template<int IFORM>
	void _ForEachElementOnInterface(Int2Type<IFORM>, std::function<void(int, index_type)> const &fun,
	        unsigned int in_tag, unsigned int out_tag, int flag) const
	{
		mesh.TraversalIndex(IFORM,

		[&](int m,index_type s)
		{

			size_t idx = mesh.GetComponentIndex(IFORM,m,s);
			auto tag=tags_[IFORM][idx];

			if( (tag[out_tag] && !tag[in_tag]) )
			{
				index_type cells[mesh_type::MAX_NUM_NEIGHBOUR_ELEMENT];
				int num=mesh.GetConnectedElement<3>(IFORM,idx,cells);
				for(int i=0;i<num;++i)
				{
					auto t=tags_[3][cells[i]];
					if(t[in_tag]!=t[out_tag])
					{
						fun(m,s);
					}
				}

			}

		},

		flag)

	}

	template<int IFORM>
	void _ForEachElementCrossInterface(Int2Type<IFORM>, std::function<void(int, index_type)> const &fun,
	        unsigned int in, unsigned int out, unsigned int flag) const
	{

		mesh.TraversalIndex(IFORM,

		[&](int m,index_type s)
		{
			size_t idx = mesh.GetComponentIndex(IFORM,m,s);
			auto tag=tags_[IFORM][idx];

			if( (tag[in] && tag[out]) )
			{
				fun(m,s);
			}
		}, mesh_type::DO_PARALLEL)
	}

	void _ForEachElementCrossInterface(Int2Type<0>, std::function<void(int, index_type)> const &fun, unsigned int in,
	        unsigned int out, unsigned int) const
	{
		DEADEND << "the volume of point is zero, which can not cross interface!";
	}

};

}  // namespace simpla

#endif /* MEDIA_TAG_H_ */
