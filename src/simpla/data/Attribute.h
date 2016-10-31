//
// Created by salmon on 16-10-20.
//

#ifndef SIMPLA_ATTRIBUTE_H
#define SIMPLA_ATTRIBUTE_H

#include <simpla/mesh/MeshBlock.h>
#include <simpla/mesh/EntityId.h>
#include <simpla/mesh/EntityRange.h>
#include <simpla/mesh/Atlas.h>

#include "Patch.h"

namespace simpla { namespace data
{


class AttrVisitorBase
{
    template<typename ... T> void visit(Attribute<T...> *) {};
};

/**
 *  Attribute IS-A container of patchs
 */
class AttributeBase :
        public toolbox::Object,
        public DataBase,
        public std::enable_shared_from_this<AttributeBase>
{
public:
    SP_OBJECT_HEAD(AttributeBase, toolbox::Object)

    typedef typename mesh::Atlas::id_type mesh_id_type;

    AttributeBase();

    AttributeBase(std::shared_ptr<mesh::MeshBlock> const &);

    AttributeBase(std::shared_ptr<mesh::Atlas> const &);

    AttributeBase(AttributeBase const &) = delete;

    AttributeBase(AttributeBase &&);

    virtual ~AttributeBase();


    virtual std::ostream &print(std::ostream &os, int indent = 1) const;

    virtual void load(DataBase const &, std::string const & = "") {};

    virtual void save(DataBase *, std::string const & = "") const {};

    virtual void apply(AttrVisitorBase *)=0;

    virtual std::type_info const &value_type_info()=0;

    virtual size_type entity_type() const =0;

    virtual void deploy();

    virtual std::shared_ptr<PatchBase> create(mesh_id_type id) const =0;

    virtual void update()=0;

    virtual void move_to(mesh_id_type t_id);

    virtual mesh::MeshBlock const *mesh(mesh_id_type t_id = 0) const;

    virtual PatchBase *patch(mesh_id_type t_id = 0);

    virtual PatchBase const *patch(mesh_id_type t_id = 0) const;

    std::shared_ptr<mesh::Atlas> const &atlas() const;

    bool has(const mesh_id_type &t_id);

    mesh_id_type mesh_id() const;

private:
    struct pimpl_s;
    std::unique_ptr<pimpl_s> m_pimpl_;


};

template<typename ...> class Attribute;

template<typename P, typename M, size_type IFORM = 0>
class Attribute<P, M, index_const<IFORM>> : public AttributeBase
{
public:

    typedef P patch_type;
    typedef M mesh_type;
    typedef typename patch_type::value_type value_type;
    typedef typename mesh_type::id_type mesh_id_type;

    typedef Attribute<P, M, index_const<IFORM>> this_type;

    patch_type *m_patch_ = nullptr;

    mesh_type const *m_mesh_ = nullptr;

    virtual std::type_info const &value_type_info() { return typeid(value_type); };

    virtual size_type entity_type() const { return IFORM; }

    virtual bool is_a(std::type_info const &t_info) const
    {
        return t_info == typeid(this_type) || AttributeBase::is_a(t_info);
    };


    template<typename ...Args>
    Attribute(Args &&...args) : AttributeBase(std::forward<Args>(args)...) {};

    virtual ~Attribute() {}

    virtual void apply(AttrVisitorBase *visitor) { visitor->visit(this); }

    virtual std::shared_ptr<PatchBase> create(mesh_id_type id) const
    {
        return std::dynamic_pointer_cast<PatchBase>(std::make_shared<patch_type>(mesh(id)));
    };


    virtual void update()
    {
        m_mesh_ = mesh();
        m_patch_ = patch();
    }


    inline value_type &get(mesh::MeshEntityId const &s) { return m_patch_->get(s.x, s.y, s.z, s.w); }

    inline value_type const &get(mesh::MeshEntityId const &s) const { return m_patch_->get(s.x, s.y, s.z, s.w); }

    inline value_type &operator[](mesh::MeshEntityId const &s) { return m_patch_->get(s.x, s.y, s.z, s.w); }

    inline value_type const &operator[](mesh::MeshEntityId const &s) const { return m_patch_->get(s.x, s.y, s.z, s.w); }

    struct expression_tag {};
    struct function_tag {};
    struct field_function_tag {};

    template<typename TOP, typename ...Args> void
    apply(TOP const &op, mesh::MeshZoneTag tag, Args &&...args)
    {
        deploy();
        apply(op, m_mesh_->range(IFORM, tag), std::forward<Args>(args)...);
    }

    template<typename TOP, typename TRange> void
    apply(TOP const &op, TRange const &r0, value_type const &v)
    {
        deploy();
        r0.foreach([&](mesh::MeshEntityId const &s) { op(get(s), v); });
    }

    template<typename TOP> void
    apply(TOP const &op, mesh::EntityRange const &r0, this_type const &other)
    {
        deploy();
        r0.foreach([&](mesh::MeshEntityId const &s) { op(get(s), other.get(s)); });
    }


    template<typename TOP, typename TFun> void
    apply(TOP const &op, mesh::EntityRange const &r0, TFun const &fun)
    {
        deploy();
        r0.foreach([&](mesh::MeshEntityId const &s) { op(get(s), fun(s)); });
    }


    template<typename TOP, typename TFun, typename ...Args> void
    apply(TOP const &op, mesh::EntityRange const r0, function_tag const *, TFun const &fun, Args &&...args)
    {
        deploy();
        r0.foreach(
                [&](mesh::MeshEntityId const &s)
                {
                    op(get(s), fun(m_mesh_->point(s), std::forward<Args>(args)...));
                });
    }

    template<typename TOP, typename ...TExpr> void
    apply(TOP const &op, mesh::EntityRange const &r0, expression_tag const *, TExpr &&...fexpr)
    {
        deploy();
        r0.foreach([&](mesh::MeshEntityId const &s)
                   {
                       op(get(s), m_mesh_->eval(std::forward<TExpr>(fexpr), s)...);
                   });
    }

    template<typename TOP, typename TFun, typename ...Args> void
    apply(TOP const &op, mesh::EntityRange const r0, field_function_tag const *, TFun const &fun, Args &&...args)
    {
        deploy();
        r0.foreach(
                [&](mesh::MeshEntityId const &s)
                {
                    op(get(s), m_mesh_->template sample<IFORM>(s, fun(m_mesh_->point(s), std::forward<Args>(args)...)));
                });
    }



//    template<typename TOP, typename TFun> void
//    apply_function_with_define_domain(TOP const &op, mesh::EntityRange const r0,
//                                      std::function<Real(point_type const &)> const &geo,
//                                      TFun const &fun)
//    {
//        deploy();
//        r0.foreach([&](mesh::MeshEntityId const &s)
//                   {
//                       auto x = m_mesh_->point(s);
//                       if (geo(x) < 0)
//                       {
//                           op(m_patch_->get(s), m_mesh_->template sample<IFORM>(s, fun(x)));
//                       }
//                   });
//    }

public:
    void copy(mesh::EntityRange const &r0, this_type const &g)
    {
        r0.foreach([&](mesh::MeshEntityId const &s) { get(s) = g.get(s); });
    }


    virtual void copy(mesh::EntityRange const &r0, PatchBase const &other)
    {
        assert(other.is_a(typeid(this_type)));

        this_type const &g = static_cast<this_type const & >(other);

        copy(r0, static_cast<this_type const & >(other));

    }


private:

};
}} //namespace data
#endif //SIMPLA_ATTRIBUTE_H
