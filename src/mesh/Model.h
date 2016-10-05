/** 
 * @file MeshUtility.h
 * @author salmon
 * @date 16-6-2 - 上午7:16
 *  */

#ifndef SIMPLA_MESHUTILITIY_H
#define SIMPLA_MESHUTILITIY_H

#include "../sp_def.h"
#include <functional>
#include "MeshCommon.h"
#include "Chart.h"

namespace simpla { namespace mesh
{


struct Model
{

    typedef std::function<Real(point_type const &)> distance_fun_t;
    enum
    {
        INSIDE, OUTSIDE, ON_SURFACE
    };

    Chart const *m;

    Model(Chart const *);

    virtual ~Model();

    virtual void deploy();

    virtual void add(EntityRange const &, distance_fun_t const dist_fun);
    virtual void remove(EntityRange const &, distance_fun_t const dist_fun);

    template<typename ...Args>
    void add(box_type const &b, Args &&...args)
    {
        add(m->range(b, VERTEX), std::forward<Args>(args)...);
    };
    template<typename ...Args>
    void remove(box_type const &b, Args &&...args)
    {
        remove(m->range(b, VERTEX), std::forward<Args>(args)...);
    };
    int check(MeshEntityId const &s);

    virtual EntityRange surface(MeshEntityType iform, int flag = OUTSIDE);
    virtual EntityRange inside(MeshEntityType iform);
    virtual EntityRange outside(MeshEntityType iform);

private:
    struct pimpl_s;
    std::unique_ptr<pimpl_s> m_pimpl_;
};

}}//namespace simpla {namespace get_mesh
#endif //SIMPLA_MESHUTILITIY_H
