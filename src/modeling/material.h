/*
 * material.h
 *
 *  Created on: 2013年12月15日
 *      Author: salmon
 */

#ifndef MATERIAL_H_
#define MATERIAL_H_

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
#include "pointinpolygen.h"
#include "select.h"
namespace simpla
{

template<typename TM>
class Material
{

public:
	static constexpr int MAX_NUM_OF_MEIDA_TYPE = 64;
	typedef TM mesh_type;

	typedef std::bitset<MAX_NUM_OF_MEIDA_TYPE> material_type;
	typedef typename mesh_type::index_type index_type;
	typedef typename mesh_type::coordinates_type coordinates_type;

	const material_type null_material;

	Field<mesh_type, VERTEX, material_type> vertex_;
	std::vector<material_type> material_[mesh_type::NUM_OF_COMPONENT_TYPE];
	std::map<std::string, material_type> register_material_;
	unsigned int max_material_;
public:

	enum
	{
		NONE = 0, VACUUM = 1, PLASMA, CORE, BOUNDARY, PLATEAU, LIMTER,
		// @NOTE: add materials for different physical area or media
		CUSTOM = 20
	};

	mesh_type const &mesh;

	Material(mesh_type const & m)
			: null_material(1 << NONE), mesh(m), max_material_(CUSTOM + 1), vertex_(mesh)
	{
		register_material_.emplace("NONE", null_material);

		register_material_.emplace("Vacuum", material_type(1 << VACUUM));
		register_material_.emplace("Plasma", material_type(1 << PLASMA));
		register_material_.emplace("Core", material_type(1 << CORE));
		register_material_.emplace("Boundary", material_type(1 << BOUNDARY));
		register_material_.emplace("Plateau", material_type(1 << PLATEAU));
		register_material_.emplace("Limter", material_type(1 << LIMTER));

	}
	~Material()
	{
	}

	bool empty() const
	{
		return material_[VERTEX].empty();
	}

	operator bool() const
	{
		return material_[VERTEX].empty();
	}

	material_type RegisterMaterial(std::string const & name)
	{
		material_type res;
		if (register_material_.find(name) != register_material_.end())
		{
			res = register_material_[name];
		}
		else if (max_material_ < MAX_NUM_OF_MEIDA_TYPE)
		{
			res.set(max_material_);
			++max_material_;
		}
		else
		{
			ERROR << "Too much media Type";
		}
		return res;
	}

	unsigned int GetNumMaterialType() const
	{
		return max_material_;
	}
	material_type GetMaterialFromNumber(unsigned int material_pos) const
	{
		material_type res;
		res.set(material_pos);
		return std::move(res);
	}
	material_type GetMaterialFromString(std::string const &name) const
	{
		return std::move(register_material_.at(name));
	}
	material_type GetMaterialFromString(std::string const &name)
	{
		return std::move(RegisterMaterial(name));
	}

	std::vector<material_type> & operator[](unsigned int n)
	{
		return material_[n];
//		auto it = register_material_.find(name);
//		if (it != register_material_.end())
//		{
//			RegisterMaterial(name);
//		}
//
//		return std::move(register_material_.at(name));
	}

	std::vector<material_type> const& operator[](unsigned int n) const
	{
		return material_[n];
	}

	void ClearAll()
	{
		for (auto &v : material_[0])
		{
			v.reset();
		}

		Update();
	}

	template<typename TDict>
	void Load(TDict const & dict)
	{
		if (dict)
		{
			for (auto const & p : dict)
			{
				Modify(p.second);
			}
		}

	}
	std::ostream & Save(std::ostream &os) const
	{

//		os << "{ \n" << "\t -- register media type\n";
//
//		for (auto const& p : register_material_)
//		{
//			os << std::setw(10) << p.first << " = 0x" << std::hex << p.second.to_ulong() << std::dec << ", \n";
//		}
//
//		os << " }\n"
//
//		;
		return os;
	}

	void Init(int I = VERTEX)
	{
		if (material_[I].empty())
		{
			material_[I].resize(mesh.GetNumOfElements(I), null_material);
		}
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

		LOGGER << op << " material " << type << DONE;
	}

	template<typename ...Args> inline
	void Set(std::string material, Args const & ... args)
	{
		Set(GetMaterialFromString(material), std::forward<Args const &>(args)...);
	}
	template<typename ...Args> inline
	void Set(unsigned int material, Args const & ... args)
	{
		Set(GetMaterialFromNumber(material), std::forward<Args const &>(args)...);
	}

	template<typename ...Args> inline
	void Add(std::string material, Args const & ... args)
	{
		Add(GetMaterialFromString(material), std::forward<Args const &>(args)...);
	}
	template<typename ...Args> inline
	void Add(unsigned int material, Args const & ... args)
	{
		Add(GetMaterialFromNumber(material), std::forward<Args const &>(args)...);
	}

	template<typename ...Args> inline
	void Remove(std::string material, Args const & ... args)
	{
		Set(~GetMaterialFromString(material), std::forward<Args const &>(args)...);
	}
	template<typename ...Args> inline
	void Remove(unsigned int material, Args const & ... args)
	{
		Set(~GetMaterialFromNumber(material), std::forward<Args const &>(args)...);
	}

	/**
	 * Set material on vertics
	 * @param material is  set to 1<<material
	 * @param args args are trans-forward to
	 *      SelectVerticsInRegion(<lambda function>,mesh,args)
	 */
	template<typename ...Args>
	void Set(material_type material, Args const & ... args)
	{
		_ForEachVertics([&]( material_type &v)
		{	v=material;},

		std::forward<Args const &>(args)...);
	}

	template<typename ...Args>
	void Add(material_type material, Args const & ... args)
	{

		_ForEachVertics([&]( material_type &v)
		{	v|=material;},

		std::forward<Args const &>(args)...);
	}

	template<typename ...Args>
	void Remove(material_type material, Args const & ... args)
	{

		_ForEachVertics([&]( material_type &v)
		{	v^=material;},

		std::forward<Args const &>(args)...);
	}

	/**
	 *  Update material on edge ,face and cell, base on material on vertics
	 */
	void Update()
	{
		_UpdateMaterials<EDGE>();
		_UpdateMaterials<FACE>();
		_UpdateMaterials<VOLUME>();
	}

	/**
	 *  Choice elements that most close to and out of the interface,
	 *  No element cross interface.
	 * @param
	 * @param fun
	 * @param in_material
	 * @param out_material
	 * @param flag
	 */
	template<int IFORM>
	void SelectBoundary(std::function<void(index_type)> const &fun, material_type in, material_type out) const;

	template<int IFORM>
	void SelectBoundary(std::function<void(index_type)> const &fun, std::string const & in,
	        std::string const & out) const
	{
		SelectBoundary<IFORM>(fun, GetMaterialFromString(in), GetMaterialFromString(out));
	}

	template<int IFORM>
	void SelectCell(std::function<void(index_type)> const &fun, material_type) const;

	template<int IFORM>
	void SelectCell(std::function<void(index_type)> const &fun, std::string const & m) const
	{
		SelectCell<IFORM>(fun, GetMaterialFromString(m));
	}

	template<int IFORM, typename TDict>
	void Select(std::function<void(index_type)> const &fun, TDict const & dict) const;

private:

	/**
	 * Set material on vertics
	 * @param material is  set to 1<<material
	 * @param args args are trans-forward to
	 *      SelectVerticsInRegion(<lambda function>,mesh,args)
	 */
	template<typename ...Args>
	void _ForEachVertics(std::function<void(material_type&)> fun, Args const & ... args)
	{
		Init();

		SelectFromMesh<VERTEX>(mesh, [&]( index_type s )
		{	fun( material_[VERTEX][mesh.Hash(s)]);}, std::forward<Args const&>(args)...);
	}

	template<int I>
	void _UpdateMaterials()
	{
		Init(I);

		mesh.template Traversal<I>(

		[&](index_type s )
		{
			index_type v[mesh_type::MAX_NUM_VERTEX_PER_CEL];

			int n=mesh.template GetAdjacentCells(Int2Type<I>(),Int2Type<VERTEX>(),s,v);
			material_type flag = null_material;
			for(int i=0;i<n;++i)
			{
				flag|=material_[VERTEX][mesh.Hash(v[i])];
			}
			material_[I][mesh.Hash(s)]=flag;

		});
	}
};
template<typename TM>
inline std::ostream & operator<<(std::ostream & os, Material<TM> const &self)
{
	return self.Save(os);
}

template<typename TM> template<int IFORM>
void Material<TM>::SelectBoundary(std::function<void(index_type)> const &fun, material_type in, material_type out) const
{

	// Good
	//  +----------#----------+
	//  |          #          |
	//  |    A     #-> B   C  |
	//  |          #          |
	//  +----------#----------+
	//
	//  +--------------------+
	//  |         ^          |
	//  |       B |     C    |
	//  |     ########       |
	//  |     #      #       |
	//  |     #  A   #       |
	//  |     #      #       |
	//  |     ########       |
	//  +--------------------+
	//
	//             +----------+
	//             |      C   |
	//  +----------######     |
	//  |          | A  #     |
	//  |    A     | &  #  B  |
	//  |          | B  #->   |
	//  +----------######     |
	//             |          |
	//             +----------+
	//
	//     	       +----------+
	//       C     |          |
	//  +----------#----+     |
	//  |          # A  |     |
	//  |    B   <-# &  |  A  |
	//  |          # B  |     |
	//  +----------#----+     |
	//             |          |
	//             +----------+
	//
	//
	// 	 Bad
	//
	//  +--------------------+
	//  |                    |
	//  |        A           |
	//  |     ########       |
	//  |     #      #       |
	//  |     #->B C #       |
	//  |     #      #       |
	//  |     ########       |
	//  +--------------------+
	//
	// 	            +----------+
	//              |          |
	//   +-------+  |          |
	//   |       |  |          |
	//   |   B   |  |    A     |
	//   |       |  |          |
	//   +-------+  |          |
	//              |          |
	//              +----------+

	mesh.template Traversal<IFORM>(

	[&]( index_type s )
	{
		if((this->material_[IFORM].at(mesh.Hash(s))&in).none() &&
				(this->material_[IFORM].at(mesh.Hash(s))&out).any() )
		{
			index_type neighbours[mesh_type::MAX_NUM_NEIGHBOUR_ELEMENT];

			int num=this->mesh.GetAdjacentCells(Int2Type<IFORM>(),Int2Type<VOLUME>(),s,neighbours );

			for(int i=0;i<num;++i)
			{

				if(((this->material_[VOLUME].at(mesh.Hash(neighbours[i]))&in) ).any())
				{
					fun( s );
					break;
				}
			}
		}

	});

}

template<typename TM>
template<int IFORM>
void Material<TM>::SelectCell(std::function<void(index_type)> const &fun, material_type material) const
{

	auto const & materials = material_[IFORM];
	mesh.template Traversal<IFORM>(

	[&]( index_type s )
	{
		if(((this->material_[IFORM].at(mesh.Hash(s))&material) ).any())
		{	fun( s );}
	});
}

template<typename TM>
template<int IFORM, typename TDict>
void Material<TM>::Select(std::function<void(index_type)> const &fun, TDict const & dict) const
{

	if (dict["Type"])
	{
		auto type = dict["Type"].template as<std::string>("");

		if (type == "Boundary")
		{
			auto material = GetMaterialFromString(dict["Material"].template as<std::string>());
			SelectBoundary<IFORM>(fun, material, null_material);

		}
		else if (type == "Interface")
		{
			auto in = GetMaterialFromString(dict["In"].template as<std::string>());
			auto out = GetMaterialFromString(dict["Out"].template as<std::string>());
			SelectBoundary<IFORM>(fun, in, out);
		}
		else if (type == "Element")
		{
			auto material = GetMaterialFromString(dict["Material"].template as<std::string>());
			SelectCell<IFORM>(fun, material);
		}
	}

}

}
// namespace simpla

#endif /* MATERIAL_H_ */
