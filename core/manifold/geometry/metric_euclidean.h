/**
 * @file metric_euclidean.h
 * @author salmon
 * @date 2015-10-28.
 */

#ifndef SIMPLA_METRIC_EUCLIDEAN_H
#define SIMPLA_METRIC_EUCLIDEAN_H
namespace simpla
{
namespace tags
{
struct Cartesian;
}
template<>
struct Metric<tags::Cartesian, tags::uniform>
{
private:

    typedef Metric<tags::Cartesian, tags::uniform> this_type;

    typedef nTuple<Real, 3> point_type;

    point_type m_inv_map_orig_ = {0, 0, 0};

    point_type m_map_orig_ = {0, 0, 0};

    point_type m_map_scale_ = {1, 1, 1};

    point_type m_inv_map_scale_ = {1, 1, 1};

public:

    Metric() { }

    ~Metric() { }

    void swap(this_type &other)
    {
        std::swap(m_inv_map_orig_, other.m_inv_map_orig_);
        std::swap(m_map_orig_, other.m_map_orig_);
        std::swap(m_map_scale_, other.m_map_scale_);
        std::swap(m_inv_map_scale_, other.m_inv_map_scale_);
    }

    template<typename TB0, typename TB1>
    void set(TB0 const &src_box, TB1 const &dest_box)
    {

        point_type src_min_, src_max_;

        point_type dest_min, dest_max;

        std::tie(src_min_, src_max_) = src_box;

        std::tie(dest_min, dest_max) = dest_box;

        for (int i = 0; i < 3; ++i)
        {
            m_map_scale_[i] = (dest_max[i] - dest_min[i]) / (src_max_[i] - src_min_[i]);

            m_inv_map_scale_[i] = (src_max_[i] - src_min_[i]) / (dest_max[i] - dest_min[i]);


            m_map_orig_[i] = dest_min[i] - src_min_[i] * m_map_scale_[i];

            m_inv_map_orig_[i] = src_min_[i] - dest_min[i] * m_inv_map_scale_[i];
        }
    }

    void set_zeor_axe(int n)
    {


        m_to_topology_orig_[i] = i_min[i] - m_coords_min_[i] * m_to_topology_scale_[i];

        m_from_topology_orig_[i] = m_coords_min_[i] - i_min[i] * m_from_topology_scale_[i];

        m_map_scale_[n] = 0;
        m_inv_map_scale_[n] = 0;
    }

    point_type inv_map(point_type const &x) const
    {

        point_type res;


        res[0] = std::fma(x[0], m_inv_map_scale_[0], m_inv_map_orig_[0]);

        res[1] = std::fma(x[1], m_inv_map_scale_[1], m_inv_map_orig_[1]);

        res[2] = std::fma(x[2], m_inv_map_scale_[2], m_inv_map_orig_[2]);


        return std::move(res);
    }

    point_type map(point_type const &y) const
    {

        point_type res;


        res[0] = std::fma(y[0], m_map_scale_[0], m_map_orig_[0]);

        res[1] = std::fma(y[1], m_map_scale_[1], m_map_orig_[1]);

        res[2] = std::fma(y[2], m_map_scale_[2], m_map_orig_[2]);

        return std::move(res);
    }


    Real volume(int n, point_type const x0[], bool is_box = false) const
    {
        Real res = 0.0;
        switch (n)
        {
            case 2: // line or box
                if (is_box)
                {

                }
                else
                {

                }
                break;
            case 4: // rectangle
                break;
            case 8: // Hexahedron
                break;
            default:
                ERROR("illegal point number!");
                break;
        }

        return res;
    }

    template<typename TB>
    Real box_volume(TB const &b) const
    {
        point_type x0, x1;
        std::tie(x0, x1) = b;
    }


};

}//namespace simpla
#endif //SIMPLA_METRIC_EUCLIDEAN_H
