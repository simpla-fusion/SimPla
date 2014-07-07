/*
 * model.h
 *
 *  Created on: 2013-12-15
 *      Author: salmon
 */

#ifndef MODEL_H_
#define MODEL_H_

#include <algorithm>
#include <bitset>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "../utilities/primitives.h"
#include "../utilities/log.h"
#include "../utilities/pretty_stream.h"
#include "../utilities/utilities.h"
#include "../utilities/sp_iterator_filter.h"
#include "../utilities/sp_type_traits.h"

#include "../numeric/pointinpolygon.h"

namespace simpla
{

/**
 *  \defgroup  Model Model
 *   \brief Geometry modeling
 */

/**
 *  \ingroup Model
 *  \brief Model
 */
template<typename TM>
class Model
{

public:
	static constexpr unsigned int MAX_NUM_OF_MEIDA_TYPE = std::numeric_limits<unsigned long>::digits;
	typedef TM mesh_type;
	static constexpr unsigned int NDIMS = mesh_type::NDIMS;
	typedef std::bitset<MAX_NUM_OF_MEIDA_TYPE> material_type;
	typedef typename mesh_type::iterator iterator;
	typedef typename mesh_type::coordinates_type coordinates_type;
	typedef typename mesh_type::compact_index_type compact_index_type;

	const material_type null_material;

	std::map<compact_index_type, material_type> material_;
	std::map<std::string, material_type> registered_material_;

	unsigned int max_material_;
public:

	enum
	{
		NONE = 0, VACUUM = 1, PLASMA, CORE, BOUNDARY, LIMITER,
		// @NOTE: add materials for different physical area or media
		CUSTOM = 20
	};

	mesh_type mesh;

	Model() :
			null_material(), max_material_(CUSTOM + 1)
	{
		registered_material_.emplace("NONE", null_material);

		registered_material_.emplace("Vacuum", material_type(1 << VACUUM));
		registered_material_.emplace("Plasma", material_type(1 << PLASMA));
		registered_material_.emplace("Core", material_type(1 << CORE));
		registered_material_.emplace("Boundary", material_type(1 << BOUNDARY));
		registered_material_.emplace("Limiter", material_type(1 << LIMITER));

	}
	~Model()
	{
	}

	bool empty() const
	{
		return material_.empty();
	}

	operator bool() const
	{
		return material_.empty();
	}

	material_type RegisterMaterial(std::string const & name)
	{
		material_type res;
		if (registered_material_.find(name) != registered_material_.end())
		{
			res = registered_material_[name];
		}
		else if (max_material_ < MAX_NUM_OF_MEIDA_TYPE)
		{
			res.set(max_material_);
			++max_material_;
		}
		else
		{
			RUNTIME_ERROR("Too much media Type");
		}
		return res;
	}

	unsigned int get_num_of_material_type() const
	{
		return max_material_;
	}

	material_type get_material(material_type const & m) const
	{
		return m;
	}

	material_type get_material(unsigned int material_pos) const
	{
		material_type res;
		res.set(material_pos);
		return std::move(res);
	}
	material_type get_material(std::string const &name) const
	{

		material_type res;

		try
		{
			res = registered_material_.at(name);

		}
		catch (...)
		{
			RUNTIME_ERROR("Unknown material name : " + name);
		}
		return std::move(res);
	}

	material_type get(compact_index_type s) const;

	material_type operator[](compact_index_type s) const
	{
		return get(s);
	}

	void clear()
	{
		material_.clear();
	}

	template<typename TDict>
	void Load(TDict const & dict)
	{
		UNIMPLEMENT;
	}
	std::string Save(std::string const & path, bool is_verbose = false) const
	{
		UNIMPLEMENT;
		return "UNIMPLEMENT!";
	}
	template<typename OS>
	OS & Print(OS &os) const
	{
		return mesh.Print(os);
	}

	typedef std::function<bool(compact_index_type const &)> pred_fun_type;
	template<typename TR> using filter_iterator_type =
	Iterator<typename std::remove_reference<decltype(std::get<0>(std::declval<TR>()))>::type, pred_fun_type ,_iterator_policy_filter,true >;

	template<typename TR> using filter_range_type = std::pair<filter_iterator_type<TR>,filter_iterator_type<TR> >;

	typedef filter_range_type<typename mesh_type::range_type> filter_mesh_range;

	template<typename TDict>
	void Modify(TDict const& dict)
	{

		std::function<material_type(material_type const &)> fun;

		auto range = SelectByConfig(dict["Select"]);

		auto material = get_material(dict["Value"].template as<std::string>(""));

		std::string op = dict["Op"].template as<std::string>("");

		if (op == "Set")
		{
			Set(range, material);
		}
		else if (op == "Remove")
		{
			Remove(range, material);
		}
		else if (op == "Add")
		{
			Add(range, material);
		}

		LOGGER << op << " material " << DONE;

	}

	template<typename TR>
	void Modify(TR const & r, std::function<material_type(material_type const &)> const &fun)
	{
		for (auto s : r)
		{
			material_[s] = fun(material_[s]);
			if (material_[s] == null_material) material_.erase(s);
		}
	}

	template<typename TR>
	void Erase(TR const & r)
	{
		for (auto s : r)
		{
			material_.erase(s);
		}
	}

	template<typename TR, typename M>
	void Set(TR const & r, M const& material)
	{
		auto t = get_material(material);
		Modify(r, [=](material_type const & m)->material_type
		{	return m|t;});
	}

	template<typename TR, typename M>
	void Unset(TR const & r, M const& material)
	{
		auto t = get_material(material);
		Modify(r, [=](material_type const & m)->material_type
		{	return m&(~t);});
	}

	template<typename TR, typename TDict>
	filter_range_type<TR> SelectByConfig(TR const& range, TDict const& dict) const;

	template<typename TR>
	filter_range_type<TR> SelectByFunction(TR const& range, std::function<bool(coordinates_type)> fun) const;

	template<typename TR, typename ...Args>
	filter_range_type<TR> SelectByMaterial(TR const& range, Args &&...args) const;

	template<typename TR, typename T1, typename T2>
	filter_range_type<TR> SelectInterface(TR const& range, T1 in, T2 out) const;

	template<typename TR>
	filter_range_type<TR> SelectByRectangle(TR const& range, coordinates_type v0, coordinates_type v1) const;

	template<typename TR>
	filter_range_type<TR> SelectByPolylines(TR const& range, PointInPolygon checkPointsInPolygen) const;

	template<typename TR>
	filter_range_type<TR> SelectByPoints(TR const& range, std::vector<coordinates_type>const & points) const;

	auto Select(unsigned int iform) const
	DECL_RET_TYPE( (mesh.Select(iform)))

	template<typename ...Args>
	auto SelectInterface(int iform, Args &&...args) const
	DECL_RET_TYPE((SelectInterface(std::move(mesh.Select(iform)), std::forward<Args>(args)...)))

	template<typename ...Args>
	auto SelectByConfig(int iform, Args &&...args) const
	DECL_RET_TYPE( (SelectByConfig(std::move(mesh.Select(iform)), std::forward<Args>(args)...)))

	template<typename ...Args>
	auto SelectByMaterial(int iform, Args &&...args) const
	DECL_RET_TYPE( (SelectByMaterial(std::move(mesh.Select(iform)), std::forward<Args>(args)...)))

	template<typename ...Args>
	auto SelectByPoints(int iform, Args &&...args) const
	DECL_RET_TYPE( ( SelectByPoints(std::move(mesh.Select(iform)), std::forward<Args>(args)...)))

	template<typename ...Args>
	auto SelectByRectangle(int iform, Args &&...args) const
	DECL_RET_TYPE( ( SelectByRectangle(std::move(mesh.Select(iform)), std::forward<Args>(args)...)))

	template<typename ...Args>
	auto SelectByPolylines(int iform, Args &&...args) const
	DECL_RET_TYPE( ( SelectByPolylines(std::move(mesh.Select(iform)), std::forward<Args>(args)...)))

	auto SelectByFunction(int iform, std::function<bool(coordinates_type)> fun) const
	DECL_RET_TYPE( (SelectByFunction(std::move(mesh.Select(iform)), fun)))

	template<typename TR, typename T1, typename T2>
	filter_range_type<TR> SelectOnSurface(TR const& range, T1 in, T2 out) const;
	template<typename TR, typename T1, typename T2>
	filter_range_type<TR> SelectCrossSurface(TR const& range, T1 in, T2 out) const;

}
;
template<typename TM>
std::ostream & operator<<(std::ostream & os, Model<TM> const & model)
{
	return model.Print(os);
}
template<typename TM>
template<typename TR, typename TDict>
typename Model<TM>::template filter_range_type<TR> Model<TM>::SelectByConfig(TR const& range, TDict const& dict) const
{

	auto type = dict["Type"].template as<std::string>("");

	if (type == "Boundary")
	{
		return std::move(SelectInterface(range, dict["In"].template as<std::string>("NONE"), "NONE"));

	}
	else if (type == "Interface")
	{
		return std::move(
		        SelectInterface(range, dict["In"].template as<std::string>("NONE"),
		                dict["Out"].template as<std::string>("NONE")));
	}
	else if (type == "Range" && dict["Points"].is_table())
	{
		std::vector<coordinates_type> points;

		dict["Points"].as(&points);

		if (points.size() == 2) //select points in a rectangle with diagonal  (x0,y0,z0)~(x1,y1,z1）,
		{
			return std::move(SelectByRectangle(range, points[0], points[1]));
		}
		else if (points.size() > 2) //select points in polylines
		{
			return std::move(SelectByPolylines(range, PointInPolygon(points, dict["Z-Axis"].template as<int>(2))));
		}
		else
		{
			PARSER_ERROR("Number of points  [" + ToString(points.size()) + "]<2");
		}

	}
	else if (!dict["Material"])
	{
		return std::move(SelectByMaterial(range, dict["Material"].template as<std::string>()));
	}
	else if (dict.is_function())
	{
		pred_fun_type pred = [=]( compact_index_type const & s )->bool
		{
			return (dict( this->mesh.get_coordinates( s)).template as<bool>());
		};

		return std::move(make_range_filter(range, std::move(pred)));
	}
	else
	{
		PARSER_ERROR("Unknown 'Seltect' options");
	}
	return filter_range_type<TR>();
}
template<typename TM> template<typename TR>
typename Model<TM>::template filter_range_type<TR> Model<TM>::SelectByPoints(TR const& range,
        std::vector<coordinates_type>const & points) const
{
	if (points.size() == 2)
	{
		return std::move(SelectByRectangle(range, points[0], points[1]));
	}
	else
	{
		return std::move(SelectByPolylines(range, PointInPolygon(points)));
	}
}

template<typename TM> template<typename TR>
typename Model<TM>::template filter_range_type<TR> Model<TM>::SelectByFunction(TR const& range,
        std::function<bool(coordinates_type)> fun) const
{
	pred_fun_type pred = [fun,this]( compact_index_type const & s )->bool
	{
		return fun( this->mesh.get_coordinates( s));
	};

	return std::move(make_range_filter(range, std::move(pred)));
}

template<typename TM>
template<typename TR, typename T1, typename T2>
typename Model<TM>::template filter_range_type<TR> Model<TM>::SelectInterface(TR const& range, T1 pin, T2 pout) const
{
	/** \note
	 * Good
	 *  +----------#----------+
	 *  |          #          |
	 *  |    A     #-> B   C  |
	 *  |          #          |
	 *  +----------#----------+
	 *
	 *  +--------------------+
	 *  |         ^          |
	 *  |       B |     C    |
	 *  |     ########       |
	 *  |     #      #       |
	 *  |     #  A   #       |
	 *  |     #      #       |
	 *  |     ########       |
	 *  +--------------------+
	 *
	 *             +----------+
	 *             |      C   |
	 *  +----------######     |
	 *  |          | A  #     |
	 *  |    A     | &  #  B  |
	 *  |          | B  #->   |
	 *  +----------######     |
	 *             |          |
	 *             +----------+
	 *
	 *     	       +----------+
	 *       C     |          |
	 *  +----------#----+     |
	 *  |          # A  |     |
	 *  |    B   <-# &  |  A  |
	 *  |          # B  |     |
	 *  +----------#----+     |
	 *             |          |
	 *             +----------+
	 *
	 *
	 * 	 Bad
	 *
	 *  +--------------------+
	 *  |                    |
	 *  |        A           |
	 *  |     ########       |
	 *  |     #      #       |
	 *  |     #->B C #       |
	 *  |     #      #       |
	 *  |     ########       |
	 *  +--------------------+
	 *
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
	material_type in = get_material(pin);
	material_type out = get_material(pout);

	if (in == out) out = null_material;

	pred_fun_type pred =

	[=]( compact_index_type const & s )->bool
	{

		material_type res;

		auto iform = this->mesh.IForm(s);

		auto self=this->get(s);

		if (( self & in).none() && ( (self & out).any() || (out == null_material) ))
		{
			compact_index_type neighbours[mesh_type::MAX_NUM_NEIGHBOUR_ELEMENT];

			int num=0;
			switch(iform)
			{	case VERTEX:
				num= this->mesh.GetAdjacentCells(Int2Type<VERTEX>(), Int2Type<VOLUME>(), s, neighbours);
				break;
				case EDGE:
				num= this->mesh.GetAdjacentCells(Int2Type<EDGE>(), Int2Type<VOLUME>(), s, neighbours);
				break;
				case FACE:
				num= this->mesh.GetAdjacentCells(Int2Type<FACE>(), Int2Type<VOLUME>(), s, neighbours);
				break;
				case VOLUME:
				num= this->mesh.GetAdjacentCells(Int2Type<VOLUME>(), Int2Type<VOLUME>(), s, neighbours);
				break;
			}

			for (int i = 0; i < num; ++i)
			{
				res |=this->get(neighbours[i]);
			}
		}

		return (res & in).any();
	};

	return std::move(make_range_filter(range, std::move(pred)));

}
template<typename TM>
typename Model<TM>::material_type Model<TM>::get(compact_index_type s) const
{

	material_type res;

	if (this->mesh.IForm(s) == VERTEX)
	{
		auto it = material_.find(s);
		if (it != material_.end())
		{
			res = it->second;
		}
	}
	else
	{
		compact_index_type neighbours[mesh_type::MAX_NUM_NEIGHBOUR_ELEMENT];

		int num = this->mesh.GetVertices(s, neighbours);

		for (int i = 0; i < num; ++i)
		{
			res |= this->get(neighbours[i]);
		}
	}
	return std::move(res);
}

template<typename TM> template<typename TR, typename ...Args>
typename Model<TM>::template filter_range_type<TR> Model<TM>::SelectByMaterial(TR const& range, Args && ... args) const
{
	auto material = get_material(std::forward<Args>(args)...);

	pred_fun_type pred = [=]( compact_index_type const & s )->bool
	{
		return (this->get(s) & material).any();
	};
	return std::move(make_range_filter(range, std::move(pred)));

}

template<typename TM> template<typename TR>
typename Model<TM>::template filter_range_type<TR> Model<TM>::SelectByRectangle(TR const& range, coordinates_type v0,
        coordinates_type v1) const
{
	pred_fun_type pred =
	        [v0,v1,this]( compact_index_type const & s )->bool
	        {

		        auto x = this->mesh.get_coordinates(s);
		        return ((((v0[0] - x[0]) * (x[0] - v1[0])) >= 0) && (((v0[1] - x[1]) * (x[1] - v1[1])) >= 0)
				        && (((v0[2] - x[2]) * (x[2] - v1[2])) >= 0));
	        };
	return std::move(make_range_filter(range, std::move(pred)));
}

template<typename TM> template<typename TR>
typename Model<TM>::template filter_range_type<TR> Model<TM>::SelectByPolylines(TR const& range,
        PointInPolygon checkPointsInPolygen) const
{
	pred_fun_type pred = [=](compact_index_type s )->bool
	{	return (checkPointsInPolygen(this->mesh.get_coordinates(s) ));};

	return std::move(make_range_filter(range, std::move(pred)));

}

}
// namespace simpla

#endif /* MODEL_H_ */
