//
// Created by salmon on 16-12-1.
//

#ifndef SIMPLA_REVOLVE_H
#define SIMPLA_REVOLVE_H

#include "GeoObject.h"
#include "Polygon.h"

namespace simpla { namespace geometry
{

template<typename TObj>
class Revolve : public GeoObject
{
public:
    Revolve(TObj const &obj, int PhiAxis = 2)
            : base_obj(obj),
              r_axis((PhiAxis + 2) % 3),
              phi_axis(PhiAxis),
              z_axis((PhiAxis + 1) % 3)
    {

    }

    virtual ~Revolve() {}

    virtual int check_inside(const Real *x) const { return base_obj.check_inside(x[r_axis], x[z_axis]); };

    int r_axis;
    int phi_axis;
    int z_axis;
    TObj const &base_obj;
};

template<typename TObj>
std::shared_ptr<GeoObject>
revolve(TObj const &obj, int phi_axis = 2)
{
    return std::dynamic_pointer_cast<GeoObject>(std::make_shared<Revolve<TObj>>(obj, phi_axis));
}
}}
#endif //SIMPLA_REVOLVE_H
