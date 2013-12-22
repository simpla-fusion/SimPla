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
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "../fetl/primitives.h"
#include "../utilities/log.h"
#include "../utilities/pretty_stream.h"
#include "mesh_algorithm.h"
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
	std::vector<tag_type> tags_[mesh_type::NUM_OF_COMPONENT_TYPE];
	std::map<std::string, tag_type> register_tag_;
	unsigned int max_tag_;
public:

	const tag_type none;

	enum
	{
		NONE = 0, VACUUM = 1, PLASMA, CORE, BOUNDARY, PLATEAU,
		// @NOTE: add tags for different physical area or media
		CUSTOM = 20
	};

	MediaTag(mesh_type const & m)
			: mesh(m), max_tag_(CUSTOM + 1), none(1 << NONE)
	{
		register_tag_.emplace("NONE", none);

		register_tag_.emplace("Vacuum", tag_type(1 << VACUUM));
		register_tag_.emplace("Plasma", tag_type(1 << PLASMA));
		register_tag_.emplace("Core", tag_type(1 << CORE));
		register_tag_.emplace("Boundary", tag_type(1 << BOUNDARY));
		register_tag_.emplace("Plateau", tag_type(1 << PLATEAU));
		register_tag_.emplace("Limter", tag_type(1 << PLATEAU));

	}
	~MediaTag()
	{
	}

	bool empty() const
	{
		return tags_[0].empty();
	}

	tag_type RegisterTag(std::string const & name)
	{
		tag_type res;
		if (register_tag_.find(name) != register_tag_.end())
		{
			res = register_tag_[name];
		}
		else if (max_tag_ < MAX_NUM_OF_MEIDA_TYPE)
		{
			res.set(max_tag_);
			++max_tag_;
		}
		else
		{
			ERROR << "Too much media Type";
		}
		return res;
	}

	unsigned int GetNumMediaType() const
	{
		return max_tag_;
	}
	tag_type GetTagFromNumber(unsigned int tag_pos) const
	{
		tag_type res;
		res.set(tag_pos);
		return std::move(res);
	}
	tag_type GetTagFromString(std::string const &name) const
	{
		return std::move(register_tag_.at(name));
	}
	tag_type GetTagFromString(std::string const &name)
	{
		return std::move(RegisterTag(name));
	}
	tag_type operator[](std::string const &name)
	{
		auto it = register_tag_.find(name);
		if (it != register_tag_.end())
		{
			RegisterTag(name);
		}

		return std::move(register_tag_.at(name));
	}

	void ClearAll()
	{
		for (auto &v : tags_[0])
		{
			v.reset();
		}

		Update();
	}

	template<typename TCfg>
	void Deserialize(TCfg const & cfg)
	{
		if (cfg.empty())
			return;
		for (auto const & p : cfg)
		{
			Modify(p.second);
		}
		Update();

	}
	std::ostream & Serialize(std::ostream &os) const
	{
//		std::vector<unsigned long> tmp[4];
//
//		for (auto const & v : tags_[0])
//		{
//			tmp[0].emplace_back(v.to_ulong());
//		}
//
//		for (auto const & v : tags_[1])
//		{
//			tmp[1].emplace_back(v.to_ulong());
//		}
//
//		for (auto const & v : tags_[2])
//		{
//			tmp[2].emplace_back(v.to_ulong());
//		}
//
//		for (auto const & v : tags_[3])
//		{
//			tmp[3].emplace_back(v.to_ulong());
//		}

		os << "Media={ \n" << "\t -- register media type\n";

		for (auto const& p : register_tag_)
		{
			os << "\t" << p.first << " = " << p.second.to_ulong() << ", \n";
		}

//		<< Data(&tmp[0][0], "tag0", mesh.GetShape(0)) << ","
//
//		<< Data(&tmp[1][0], "tag1", mesh.GetShape(1)) << ","
//
//		<< Data(&tmp[2][0], "tag2", mesh.GetShape(2)) << ","
//
//		<< Data(&tmp[3][0], "tag3", mesh.GetShape(3)) << ",";

		os << " }\n"

		;
		return os;
	}

	template<typename TCmd>
	void Modify(TCmd const& cmd)
	{
		std::string op = "";
		std::string type = "";

		cmd["Op"].template as<std::string>(&op);
		cmd["Type"].template as<std::string>(&type);

		if (type == "")
		{
			WARNING << "Illegal input! [ undefine type ]";
			return;
		}

		auto select = cmd["Select"];
		if (select.empty())
		{
			std::vector<coordinates_type> region;

			cmd["Region"].as(&region);

			if (op == "Set")
			{
				Set(type, region);
			}
			else if (op == "Remove")
			{
				Remove(type, region);
			}
			else if (op == "Add")
			{
				Add(type, region);
			}
		}
		else
		{
			if (op == "Set")
			{
				Set(type, select);
			}
			else if (op == "Remove")
			{
				Remove(type, select);
			}
			else if (op == "Add")
			{
				Add(type, select);
			}
		}

		LOGGER << op << " media " << type << DONE;
	}

	template<typename ...Args> inline
	void Set(std::string media_tag, Args const & ... args)
	{
		Set(GetTagFromString(media_tag), std::forward<Args const &>(args)...);
	}
	template<typename ...Args> inline
	void Set(unsigned int media_tag, Args const & ... args)
	{
		Set(GetTagFromNumber(media_tag), std::forward<Args const &>(args)...);
	}

	template<typename ...Args> inline
	void Add(std::string media_tag, Args const & ... args)
	{
		Add(GetTagFromString(media_tag), std::forward<Args const &>(args)...);
	}
	template<typename ...Args> inline
	void Add(unsigned int media_tag, Args const & ... args)
	{
		Add(GetTagFromNumber(media_tag), std::forward<Args const &>(args)...);
	}

	template<typename ...Args> inline
	void Remove(std::string media_tag, Args const & ... args)
	{
		Set(GetTagFromString(media_tag), std::forward<Args const &>(args)...);
	}
	template<typename ...Args> inline
	void Remove(unsigned int media_tag, Args const & ... args)
	{
		Set(GetTagFromNumber(media_tag), std::forward<Args const &>(args)...);
	}

	/**
	 * Set media tag on vertics
	 * @param tag media tag is  set to 1<<tag
	 * @param args args are trans-forward to
	 *      SelectVerticsInRegion(<lambda function>,mesh,args)
	 */
	template<typename ...Args>
	void Set(tag_type media_tag, Args const & ... args)
	{
		_ForEachVertics(

		[&](bool isSelected,tag_type &v)
		{	if(isSelected) v=media_tag;},

		std::forward<Args const &>(args)...);
	}

	template<typename ...Args>
	void InverseSet(tag_type media_tag, Args const & ... args)
	{

		_ForEachVertics(

		[&](bool isSelected,tag_type &v)
		{	if(! isSelected) v =media_tag;},

		std::forward<Args const &>(args)...);
	}

	template<typename ...Args>
	void Add(tag_type media_tag, Args const & ... args)
	{

		_ForEachVertics(

		[&](bool isSelected,tag_type &v)
		{	if( isSelected) v|=media_tag;},

		std::forward<Args const &>(args)...);
	}

	template<typename ...Args>
	void Remove(tag_type media_tag, Args const & ... args)
	{

		_ForEachVertics(

		[&](bool isSelected,tag_type &v)
		{	if(isSelected) v^=media_tag;},

		std::forward<Args const &>(args)...);
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

	enum
	{
		CROSS_BOUNDAR, ON_BOUNDARY
	};

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
	void SelectBoundaryCell(std::function<void(index_type)> const &fun, tag_type in, tag_type out, unsigned int flag =
	        ON_BOUNDARY, int flag2 = 0) const
	{
		_SelectBoundaryCell(Int2Type<IFORM>(), fun, in, out, flag, flag2);
	}

private:

	/**
	 * Set media tag on vertics
	 * @param tag media tag is  set to 1<<tag
	 * @param args args are trans-forward to
	 *      SelectVerticsInRegion(<lambda function>,mesh,args)
	 */
	template<typename ...Args>
	void _ForEachVertics(std::function<void(bool, tag_type&)> fun, Args const & ... args)
	{
		if (tags_[0].empty())
			tags_[0].resize(mesh.GetNumOfElements(0), none);

		SelectVericsInRegion(mesh,

		[&](bool is_selected,index_type const &s)
		{
			fun(is_selected,tags_[0][s]);
		}, std::forward<Args const&>(args)...);
	}

	template<int I>
	void _UpdateTags()
	{
		if (tags_[I].empty())
			tags_[I].resize(mesh.GetNumOfElements(I), none);

		mesh.ParallelTraversal(I,

		[&](index_type const & s )
		{
			index_type v[mesh_type::MAX_NUM_VERTEX_PER_CEL];

			int n=mesh.template GetNeighbourCell(Int2Type<I>(),Int2Type<0>(),v,s);
			tag_type flag = 0;
			for(int i=0;i<n;++i)
			{
				flag|=tags_[0][v[i]];
			}
			tags_[I][s]=flag;

		});
	}

	template<int IFORM>
	void _SelectBoundaryCell(Int2Type<IFORM>, std::function<void(index_type)> const &fun, tag_type A, tag_type B,
	        int flag = ON_BOUNDARY, int parallel_traversal = 0) const
	{

		if ((B & (~A)).any())
		{
			/**
			 *   +----------#----------+
			 *   |          #          |
			 *   |    A     #-> B   C  |
			 *   |          #          |
			 *   +----------#----------+
			 *
			 *   +--------------------+
			 *   |         ^          |
			 *   |       B |     C    |
			 *   |     ########       |
			 *   |     #      #       |
			 *   |     #  A   #       |
			 *   |     #      #       |
			 *   |     ########       |
			 *   +--------------------+
			 *
			 *   			+----------+
			 *              |      C    |
			 *   +----------######     |
			 *   |          | A  #     |
			 *   |    A     | &  #  B  |
			 *   |          | B  #->   |
			 *   +----------######     |
			 *              |          |
			 *              +----------+
			 *
			 *   			+----------+
			 *         C     |          |
			 *   +----------#----+     |
			 *   |          # A  |     |
			 *   |    B   <-# &  |  A  |
			 *   |          # B  |     |
			 *   +----------#----+     |
			 *              |          |
			 *              +----------+
			 */

			B &= (~A);
		}
		else
		{
			/**
			 *   +--------------------+
			 *   |                    |
			 *   |        A           |
			 *   |     ########       |
			 *   |     #      #       |
			 *   |     #->B C #       |
			 *   |     #      #       |
			 *   |     ########       |
			 *   +--------------------+
			 *
			 */

			A &= (~B);
		}

		/**
		 * 	            +----------+
		 *              |          |
		 *   +-------+  |          |
		 *   |       |  |          |
		 *   |   B   |  |    A     |
		 *   |       |  |          |
		 *   +-------+  |          |
		 *              |          |
		 *              +----------+
		 */

		tag_type AB = A | B;

		if (!AB.none())
		{

			mesh.SerialTraversal(IFORM,

			[&](int m, index_type x,index_type y,index_type z)
			{
				index_type s=mesh.GetComponentIndex(IFORM,m,x,y,z);

				if((tags_[IFORM][s]&(B)).none()) return;

				index_type neighbours[mesh_type::MAX_NUM_NEIGHBOUR_ELEMENT];

				if(flag==ON_BOUNDARY)
				{

					int num=mesh.GetNeighbourCell(Int2Type<IFORM>(),Int2Type<3>(),neighbours,m,x,y,z);

					for(int i=0;i<num;++i)
					{
						if((tags_[3].at(neighbours[i])&A).any())
						{
							fun(s);
							break;
						}
					}

				}
				else
				{
					int num=mesh.GetNeighbourCell(Int2Type<IFORM>(),Int2Type<0>(),neighbours,m,x,y,z);

					for(int i=0;i<num;++i)
					{
						if((tags_[0].at(neighbours[i])&A).any())
						{
							fun(s);
							break;
						}
					}
				}
			},

			0);
		}
	}
};
template<typename TM>
inline std::ostream & operator<<(std::ostream & os, MediaTag<TM> const &self)
{
	return self.Serialize(os);
}
}  // namespace simpla

#endif /* MEDIA_TAG_H_ */
