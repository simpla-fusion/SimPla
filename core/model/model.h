/**
 * @file model.h
 *
 *  created on: 2013-12-15
 *      Author: salmon
 */

#ifndef MODEL_H_
#define MODEL_H_

#include <stddef.h>
#include <algorithm>
#include <bitset>
#include <limits>
#include <map>
#include <string>

#include "../gtl/primitives.h"
#include "../gtl/type_traits.h"
#include "../gtl/utilities/utilities.h"
#include "../mesh/mesh_ids.h"

namespace simpla {

/**
 *  @defgroup  Model Model
 *  @brief Geometry modeling
 */

/**
 *  @ingroup Model
 *   @brief Model
 */
template<size_t NDIMS = 3>
class Model_ : public MeshIDs_<NDIMS, 0>
{

public:

    typedef MeshIDs_<NDIMS, AXIS_FLAG> mesh_ids;

    using typename mesh_ids::id_type;

    typedef std::uint64_t tag_type;

    static constexpr size_t MAX_NUM_OF_MEIDA_TYPE =
            std::numeric_limits<tag_type>::digits;

    static constexpr size_t ndims = 3;

    static constexpr tag_type null_material = 0UL;

    std::map<id_type, tag_type> m_data_;

    std::map<std::string, tag_type> registered_material_;

    size_t max_material_;
public:

    enum
    {
        NONE = 0,
        VACUUM = 1UL << 1,
        PLASMA = 1UL << 2,
        CORE = 1UL << 3,
        BOUNDARY = 1UL << 4,
        LIMITER = 1UL << 5,
        // @NOTE: add materials for different physical area or media
                CUSTOM = 1UL << 20
    };

    Model_()
            : max_material_(CUSTOM << 1)
    {
        registered_material_.emplace("NONE", null_material);

        registered_material_.emplace("Vacuum", (VACUUM));
        registered_material_.emplace("Plasma", (PLASMA));
        registered_material_.emplace("Core", (CORE));
        registered_material_.emplace("Boundary", (BOUNDARY));
        registered_material_.emplace("Limiter", (LIMITER));

    }

    ~Model_()
    {
    }

    bool empty() const
    {
        return m_data_.empty();
    }

    size_t register_material(std::string const &name)
    {
        size_t res;
        if (registered_material_.find(name) != registered_material_.end())
        {
            res = registered_material_[name];
        }
        else if (max_material_ < MAX_NUM_OF_MEIDA_TYPE)
        {
            max_material_ = max_material_ << 1;

            res = (max_material_);

        }
        else
        {
            RUNTIME_ERROR("Too much media Type");
        }
        return res;
    }

    size_t get_material(std::string const &name) const
    {

        if (name == "" || name == "NONE")
        {
            return null_material;
        }
        size_t res;

        try
        {
            res = registered_material_.at(name);

        } catch (...)
        {
            RUNTIME_ERROR("Unknown material name : " + name);
        }
        return std::move(res);
    }

    tag_type at(id_type s) const
    {
        auto it = m_data_.find(s & ID_MASK);

        if (it != m_data_.end())
        {
            return it->second;
        }
        else
        {
            return null_material;
        }

    }

    tag_type operator[](id_type s) const
    {
        return get(s);
    }

    void clear()
    {
        m_data_.clear();
    }

    template<typename TR>
    void erase(TR const &r)
    {
        for (auto s : r)
        {
            m_data_.erase(s);
        }
    }

    template<typename TR>
    void set(TR const &r, size_t const &tag)
    {
        for (auto s : r)
        {
            m_data_[s] |= tag;
        }
    }

    template<typename TR>
    void unset(TR const &r, size_t const &tag)
    {
        for (auto s : r)
        {
            m_data_[s] &= ~tag;
        }
    }

    /**
     *
     * @param s is a FACE or EDGE
     * @param in
     * @param out
     * @return
     */
    bool check_boundary_surface(id_type const &s, size_t in)
    {
        id_type d = (~s) & (_DA) & ID_MASK;

        return ((in & get(s - d)) == 0UL) ^ ((in & get(s + d)) == 0UL);
    }

};

template<size_t N> constexpr size_t Model_<N>::null_material;

typedef Model_<3> Model;

//typename Model::size_t Model::get(id_type s) const
//{
//
//	size_t res = null_material;
//
//	if (this->geometry_type::IForm(s) == VERTEX)
//	{
//		auto it = m_data_.find(s);
//		if (it != m_data_.end())
//		{
//			res = it->second;
//		}
//	}
//	else
//	{
//		id_type neighbours[geometry_type::MAX_NUM_NEIGHBOUR_ELEMENT];
//
//		int num = this->geometry_type::get_vertices(s, neighbours);
//
//		for (int i = 0; i < num; ++i)
//		{
//			res |= this->get(neighbours[i]);
//		}
//	}
//	return std::move(res);
//}

//template<typename TM>
//template<typename TR, typename TDict>
//FilterRange<TR> Model<TM>::select_by_config(TR const& range,
//		TDict const& dict) const
//{
//	if (!dict)
//	{
//		pred_fun_type pred = [=]( id_type const & s )->bool
//		{
//			return true;
//		};
//
//		return std::move(FilterRange < TR > (range, std::move(pred)));
//	}
//	else if (dict.is_function())
//	{
//		pred_fun_type pred =
//				[=]( id_type const & s )->bool
//				{
//					return (dict( this->geometry_type::get_coordinates( s)).template as<bool>());
//				};
//
//		return std::move(FilterRange < TR > (range, std::move(pred)));
//	}
//	else if (dict["Material"])
//	{
//		return std::move(
//				SelectByMaterial(range,
//						dict["Material"].template as<std::string>()));
//	}
//	else if (dict["Type"])
//	{
//		auto type = dict["Type"].template as < std::string > ("");
//
//		if (type == "NGP")
//		{
//			return std::move(
//					SelectByNGP(range,
//							dict["Points"].template as<coordinates_type>()));
//
//		}
//		else if (type == "Boundary")
//		{
//			return std::move(
//					SelectInterface(range,
//							dict["In"].template as < std::string > ("NONE"),
//							"NONE"));
//
//		}
//		else if (type == "Interface")
//		{
//			return std::move(
//					SelectInterface(range,
//							dict["In"].template as < std::string > ("NONE"),
//							dict["Out"].template as < std::string > ("NONE")));
//		}
//		else if (type == "Range" && dict["Points"].is_table())
//		{
//			std::vector < coordinates_type > points;
//
//			dict["Points"].as(&points);
//
//			if (points.size() == 2) //select points in a rectangle with diagonal  (x0,y0,z0)~(x1,y1,z1）,
//			{
//				return std::move(SelectByRectangle(range, points[0], points[1]));
//			}
//			else if (points.size() > 2) //select points in polylines
//			{
//				return std::move(
//						SelectByPolylines(range,
//								PointInPolygon(points,
//										dict["Z-Axis"].template as<int>(2))));
//			}
//			else
//			{
//				PARSER_ERROR(
//						"Number of points  [" + value_to_string(points.size())
//								+ "]<2");
//			}
//
//		}
//
//	}
//
//	else
//	{
//		PARSER_ERROR("Unknown 'Select' options");
//	}
//	return FilterRange < TR > (range, [=](id_type const &)
//	{	return true;});
//}
//
//template<typename TM>
//template<typename TR, typename T1, typename T2>
//FilterRange<TR> Model<TM>::SelectInterface(TR const& range, T1 pin,
//		T2 pout) const
//{
//	/** \note
//	 * Good
//	 *  +----------#----------+
//	 *  |          #          |
//	 *  |    A     #-> B   C  |
//	 *  |          #          |
//	 *  +----------#----------+
//	 *
//	 *  +--------------------+
//	 *  |         ^          |
//	 *  |       B |     C    |
//	 *  |     ########       |
//	 *  |     #      #       |
//	 *  |     #  A   #       |
//	 *  |     #      #       |
//	 *  |     ########       |
//	 *  +--------------------+
//	 *
//	 *             +----------+
//	 *             |      C   |
//	 *  +----------######     |
//	 *  |          | A  #     |
//	 *  |    A     | &  #  B  |
//	 *  |          | B  #->   |
//	 *  +----------######     |
//	 *             |          |
//	 *             +----------+
//	 *
//	 *     	       +----------+
//	 *       C     |          |
//	 *  +----------#----+     |
//	 *  |          # A  |     |
//	 *  |    B   <-# &  |  A  |
//	 *  |          # B  |     |
//	 *  +----------#----+     |
//	 *             |          |
//	 *             +----------+
//	 *
//	 *
//	 * 	 Bad
//	 *
//	 *  +--------------------+
//	 *  |                    |
//	 *  |        A           |
//	 *  |     ########       |
//	 *  |     #      #       |
//	 *  |     #->B C #       |
//	 *  |     #      #       |
//	 *  |     ########       |
//	 *  +--------------------+
//	 *
//	 * 	            +----------+
//	 *              |          |
//	 *   +-------+  |          |
//	 *   |       |  |          |
//	 *   |   B   |  |    A     |
//	 *   |       |  |          |
//	 *   +-------+  |          |
//	 *              |          |
//	 *              +----------+
//	 */
//	size_t in = get_material(pin);
//	size_t out = get_material(pout);
//
//	if (in == out)
//		out = null_material;
//
//	pred_fun_type pred =
//
//			[=]( id_type const & s )->bool
//			{
//
//				size_t res;
//
//				auto iform = this->geometry_type::IForm(s);
//
//				auto self=this->get(s);
//
//				if (( self & in).none() && ( (self & out).any() || (out == null_material) ))
//				{
//					id_type neighbours[geometry_type::MAX_NUM_NEIGHBOUR_ELEMENT];
//
//					int num=0;
//					switch(iform)
//					{	case VERTEX:
//						num= this->geometry_type::get_adjacent_cells(std::integral_constant<size_t ,VERTEX>(), std::integral_constant<size_t ,VOLUME>(), s, neighbours);
//						break;
//						case EDGE:
//						num= this->geometry_type::get_adjacent_cells(std::integral_constant<size_t ,EDGE>(), std::integral_constant<size_t ,VOLUME>(), s, neighbours);
//						break;
//						case FACE:
//						num= this->geometry_type::get_adjacent_cells(std::integral_constant<size_t ,FACE>(), std::integral_constant<size_t ,VOLUME>(), s, neighbours);
//						break;
//						case VOLUME:
//						num= this->geometry_type::get_adjacent_cells(std::integral_constant<size_t ,VOLUME>(), std::integral_constant<size_t ,VOLUME>(), s, neighbours);
//						break;
//					}
//
//					for (int i = 0; i < num; ++i)
//					{
//						res |=this->get(neighbours[i]);
//					}
//				}
//
//				return (res & in).any();
//			};
//
//	return std::move(FilterRange < TR > (range, (pred)));
//
//}
//
//template<typename TM> template<typename TR, typename ...Args>
//FilterRange<TR> Model<TM>::SelectByMaterial(TR const& range,
//		Args && ... args) const
//{
//	auto material = get_material(std::forward<Args>(args)...);
//
//	if (material != null_material)
//	{
//		pred_fun_type pred = [=]( id_type const & s )->bool
//		{
//			return (this->get(s) & material).any();
//		};
//		return std::move(FilterRange < TR > (range, std::move(pred)));
//	}
//
//	else
//	{
//		pred_fun_type pred = [=]( id_type const & s )->bool
//		{
//			return (this->get(s) == null_material);
//		};
//		return std::move(FilterRange < TR > (range, std::move(pred)));
//	}
//
//}

}
// namespace simpla

#endif /* MODEL_H_ */
