/** 
 * @file MeshUtility.h
 * @author salmon
 * @date 16-6-2 - 上午7:16
 *  */

#ifndef SIMPLA_MESHUTILITIY_H
#define SIMPLA_MESHUTILITIY_H

#include <simpla/SIMPLA_config.h>
#include <simpla/mesh/MeshCommon.h>
#include <simpla/mesh/MeshBlock.h>
#include <functional>
namespace simpla { namespace mesh
{


struct Model
{

    typedef std::function<Real(point_type const &)> distance_fun_t;

    enum
    {
        INSIDE, OUTSIDE, ON_SURFACE
    };

    MeshBlock const *m;

    Model(MeshBlock const *);

    virtual ~Model();

    virtual void deploy();

    virtual void add(EntityIdRange const &, distance_fun_t const dist_fun);

    virtual void remove(EntityIdRange const &, distance_fun_t const dist_fun);

    template<typename ...Args>
    void add(box_type const &b, Args &&...args)
    {
        add(m->range(VERTEX, b, 0), std::forward<Args>(args)...);
    };

    template<typename ...Args>
    void remove(box_type const &b, Args &&...args)
    {
//        remove(m->range(b, VERTEX), std::forward<Args>(args)...);
    };

    int check(MeshEntityId const &s);

    virtual EntityIdRange surface(MeshEntityType iform, int flag = OUTSIDE);

    virtual EntityIdRange inside(MeshEntityType iform);

    virtual EntityIdRange outside(MeshEntityType iform);

private:
    struct pimpl_s;
    std::unique_ptr<pimpl_s> m_pimpl_;
};

}}//namespace simpla {namespace get_mesh
#endif //SIMPLA_MESHUTILITIY_H
