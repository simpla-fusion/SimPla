/*
 * model.h
 *
 *  created on: 2013-12-15
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
#include "../utilities/sp_iterator.h"
#include "../utilities/sp_iterator_filter.h"
#include "../utilities/sp_type_traits.h"

#include "../numeric/pointinpolygon.h"
#include "../numeric/geometric_algorithm.h"
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
class Model: public TM
{
private:
	bool is_ready_ = false;
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

	Model() :
			mesh_type(), null_material(0), max_material_(CUSTOM + 1)
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

	bool is_ready() const
	{
		return is_ready_ && mesh_type::is_ready();
	}
	template<typename TDict>
	bool load(TDict const & dict)
	{
		mesh_type::load(dict["Mesh"]);

		mesh_type::update();

		if (dict["Material"].is_table())
		{
			for (auto const & item : dict["Material"])
			{
				Modify(item.second);
			}
		}

		return true;
	}
	std::string save(std::string const & path) const
	{
		return mesh_type::save(path);
	}
	template<typename OS>
	OS & print(OS &os) const
	{
		os << std::endl

		<< " Type= \"" << mesh_type::get_type_as_string() << "\"," << std::endl;

		mesh_type::print(os);

		return os;
	}

	void update()
	{
		mesh_type::update();
		is_ready_ = mesh_type::is_ready();
	}

	operator bool() const
	{
		return !material_.empty() && is_ready();
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

		if (name == "" || name == "NONE")
		{
			return null_material;
		}
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

	typedef typename mesh_type::iterator mesh_iterator;

	typedef typename mesh_type::range_type mesh_range;

	typedef std::function<bool(compact_index_type const &)> pred_fun_type;

	typedef Iterator<typename mesh_type::iterator, pred_fun_type, _iterator_policy_filter, true> filter_mesh_iterator;

	typedef std::pair<filter_mesh_iterator, filter_mesh_iterator> filter_mesh_range;

	filter_mesh_range make_mesh_range_filter(mesh_range const & range, pred_fun_type const& pred)const
	{
		return std::make_pair(filter_mesh_iterator(std::get<0>(range), std::get<1>(range), pred),
		        filter_mesh_iterator(std::get<1>(range), std::get<1>(range), pred));
	}

	template<typename TDict>
	void Modify(TDict const& dict)
	{

		std::function<material_type(material_type const &)> fun;

		auto range = SelectByConfig(VERTEX, dict["Select"]);

		auto material_name = dict["Value"].template as<std::string>("");
		auto material = get_material(material_name);

		std::string op = dict["Op"].template as<std::string>("");

		if (op == "Set")
		{
			Set(range, material);
		}
		else if (op == "Unset")
		{
			Unset(range, material);
		}
		else if (op == "Erase")
		{
			Erase(range);
		}
		LOGGER << op << " material [" << material_name << "]" << DONE;

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

	template<typename TDict>
	filter_mesh_range SelectByConfig(mesh_range const& range, TDict const& dict) const;

	filter_mesh_range SelectByFunction(mesh_range const& range, std::function<bool(coordinates_type const&)> fun) const;

	filter_mesh_range SelectByMaterial(mesh_range const& range, material_type const&) const;

	template<typename T1, typename T2>
	filter_mesh_range SelectInterface(mesh_range const& range, T1 in, T2 out) const;

	filter_mesh_range SelectByRectangle(mesh_range const& range, coordinates_type v0, coordinates_type v1) const;

	filter_mesh_range SelectByPolylines(mesh_range const& range, PointInPolygon checkPointsInPolygen) const;

	filter_mesh_range SelectByPoints(mesh_range const& range, std::vector<coordinates_type>const & points) const;

	filter_mesh_range SelectByNGP(mesh_range const& range, coordinates_type const & points) const;

	template<typename T1, typename T2>
	filter_mesh_range SelectOnSurface(mesh_range const& range, T1 in, T2 out) const;
	template<typename T1, typename T2>
	filter_mesh_range SelectCrossSurface(mesh_range const& range, T1 in, T2 out) const;

	mesh_range Select(unsigned int iform) const
	{
		return std::move(mesh_type::Select(iform));
	}

	template<typename T1, typename T2>
	filter_mesh_range SelectInterface(unsigned int iform, T1 in, T2 out) const
	{
		return SelectInterface(Select(iform), in, out);
	}

	template<typename TDict>
	filter_mesh_range SelectByConfig(unsigned int iform, TDict const& dict) const
	{
		return ((SelectByConfig(Select(iform), dict)));
	}

	template<typename ...Args>
	filter_mesh_range SelectByMaterial(unsigned int iform, Args &&...args) const
	{
		return ((SelectByMaterial(Select(iform), get_material(std::forward<Args>(args)...))));
	}

	filter_mesh_range SelectByPoints(unsigned int iform, std::vector<coordinates_type>const & points) const
	{
		return ((SelectByPoints(Select(iform), points)));
	}

	filter_mesh_range SelectByRectangle(unsigned int iform, coordinates_type v0, coordinates_type v1) const
	{
		mesh_range range = Select(iform);
		return std::move(SelectByRectangle(range, v0, v1));
	}

	filter_mesh_range SelectByPolylines(unsigned int iform, PointInPolygon checkPointsInPolygen) const
	{
		return ((SelectByPolylines(Select(iform), checkPointsInPolygen)));
	}

	filter_mesh_range SelectByFunction(unsigned int iform, std::function<bool(coordinates_type const&)> fun) const
	{
		return ((SelectByFunction(Select(iform), fun)));
	}

	filter_mesh_range SelectByNGP(unsigned int iform, coordinates_type const & points) const
	{
		return ((SelectByNGP(Select(iform), points)));
	}

}
;
template<typename TM>
std::ostream & operator<<(std::ostream & os, Model<TM> const & model)
{
	return model.print(os);
}
template<typename TM>
template<typename TDict>
typename Model<TM>::filter_mesh_range Model<TM>::SelectByConfig(mesh_range const& range, TDict const& dict) const
{
	if (!dict)
	{
		pred_fun_type pred = [=]( compact_index_type const & s )->bool
		{
			return true;
		};

		return std::move(make_mesh_range_filter(range, pred));
	}
	else if (dict.is_function())
	{
		pred_fun_type pred = [=]( compact_index_type const & s )->bool
		{
			return (dict( this->mesh_type::get_coordinates( s)).template as<bool>());
		};

		return std::move(make_mesh_range_filter(range, pred));
	}
	else if (dict["Material"])
	{
		return std::move(SelectByMaterial(range, dict["Material"].template as<std::string>()));
	}
	else if (dict["Type"])
	{
		auto type = dict["Type"].template as<std::string>("");

		if (type == "NGP")
		{
			return std::move(SelectByNGP(range, dict["Points"].template as<coordinates_type>()));

		}
		else if (type == "Boundary")
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

	}

	else
	{
		PARSER_ERROR("Unknown 'Select' options");
	}
	return filter_mesh_range();
}
template<typename TM>
typename Model<TM>::filter_mesh_range Model<TM>::SelectByPoints(mesh_range const& range,
        std::vector<coordinates_type>const & points) const
{
	if (points.size() == 1)
	{
		return std::move(SelectByNGP(range, points[0]));
	}
	else if (points.size() == 2)
	{
		return std::move(SelectByRectangle(range, points[0], points[1]));
	}
	else
	{
		return std::move(SelectByPolylines(range, PointInPolygon(points)));
	}
}

template<typename TM>
typename Model<TM>::filter_mesh_range Model<TM>::SelectByNGP(mesh_range const& range, coordinates_type const & x) const
{
	compact_index_type dest;

	std::tie(dest, std::ignore) = mesh_type::CoordinatesGlobalToLocal(x);

	if (mesh_type::InLocalRange(dest))
	{

		pred_fun_type pred = [dest]( compact_index_type const & s )->bool
		{
			return mesh_type::GetCellIndex(s)==mesh_type::GetCellIndex(dest);
		};

		return std::move(make_mesh_range_filter(range, pred));
	}
	else
	{
		pred_fun_type pred = []( compact_index_type const & )->bool
		{
			return false;
		};

		return std::move(make_mesh_range_filter(range, pred));
	}

}

template<typename TM>
typename Model<TM>::filter_mesh_range Model<TM>::SelectByFunction(mesh_range const& range,
        std::function<bool(coordinates_type const&)> fun) const
{
	pred_fun_type pred = [fun,this]( compact_index_type const & s )->bool
	{
		return fun( this->mesh_type::get_coordinates( s));
	};

	return std::move(make_mesh_range_filter(range, pred));
}

template<typename TM>
template<typename T1, typename T2>
typename Model<TM>::filter_mesh_range Model<TM>::SelectInterface(mesh_range const& range, T1 pin, T2 pout) const
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

		        auto iform = this->mesh_type::IForm(s);

		        auto self=this->get(s);

		        if (( self & in).none() && ( (self & out).any() || (out == null_material) ))
		        {
			        compact_index_type neighbours[mesh_type::MAX_NUM_NEIGHBOUR_ELEMENT];

			        int num=0;
			        switch(iform)
			        {	case VERTEX:
				        num= this->mesh_type::GetAdjacentCells(std::integral_constant<unsigned int ,VERTEX>(), std::integral_constant<unsigned int ,VOLUME>(), s, neighbours);
				        break;
				        case EDGE:
				        num= this->mesh_type::GetAdjacentCells(std::integral_constant<unsigned int ,EDGE>(), std::integral_constant<unsigned int ,VOLUME>(), s, neighbours);
				        break;
				        case FACE:
				        num= this->mesh_type::GetAdjacentCells(std::integral_constant<unsigned int ,FACE>(), std::integral_constant<unsigned int ,VOLUME>(), s, neighbours);
				        break;
				        case VOLUME:
				        num= this->mesh_type::GetAdjacentCells(std::integral_constant<unsigned int ,VOLUME>(), std::integral_constant<unsigned int ,VOLUME>(), s, neighbours);
				        break;
			        }

			        for (int i = 0; i < num; ++i)
			        {
				        res |=this->get(neighbours[i]);
			        }
		        }

		        return (res & in).any();
	        };

	return std::move(make_mesh_range_filter(range, pred));

}
template<typename TM>
typename Model<TM>::material_type Model<TM>::get(compact_index_type s) const
{

	material_type res = null_material;

	if (this->mesh_type::IForm(s) == VERTEX)
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

		int num = this->mesh_type::GetVertices(s, neighbours);

		for (int i = 0; i < num; ++i)
		{
			res |= this->get(neighbours[i]);
		}
	}
	return std::move(res);
}

template<typename TM>
typename Model<TM>::filter_mesh_range Model<TM>::SelectByMaterial(mesh_range const& range,
        material_type const & material) const
{

	if (material != null_material)
	{
		pred_fun_type pred = [=]( compact_index_type const & s )->bool
		{
			return (this->get(s) & material).any();
		};
		return std::move(make_mesh_range_filter(range, pred));
	}

	else
	{
		pred_fun_type pred = [=]( compact_index_type const & s )->bool
		{
			return (this->get(s) == null_material);
		};
		return std::move(make_mesh_range_filter(range, std::move(pred)));
	}

}

template<typename TM>
typename Model<TM>::filter_mesh_range Model<TM>::SelectByRectangle(mesh_range const& range, coordinates_type v0,
        coordinates_type v1) const
{
	pred_fun_type pred =
	        [v0,v1,this]( compact_index_type const & s )->bool
	        {

		        auto x = this->mesh_type::get_coordinates(s);
		        return ((((v0[0] - x[0]) * (x[0] - v1[0])) >= 0) && (((v0[1] - x[1]) * (x[1] - v1[1])) >= 0)
				        && (((v0[2] - x[2]) * (x[2] - v1[2])) >= 0));
	        };

	return std::make_pair(filter_mesh_iterator(std::get<0>(range), std::get<1>(range), pred),
	        filter_mesh_iterator(std::get<1>(range), std::get<1>(range), pred));

//	return std::move(make_range<_iterator_policy_filter,mesh_range_type,pred_fun_type>(range, std::move(pred)));
}

template<typename TM>
typename Model<TM>::filter_mesh_range Model<TM>::SelectByPolylines(mesh_range const& range,
        PointInPolygon checkPointsInPolygen) const
{
	pred_fun_type pred = [=](compact_index_type s )->bool
	{	return (checkPointsInPolygen(this->mesh_type::get_coordinates(s) ));};

	return std::move(make_mesh_range_filter(range, pred));

}

}
// namespace simpla

#endif /* MODEL_H_ */
