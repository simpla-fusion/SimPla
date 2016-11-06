//
// Created by salmon on 16-10-20.
//

#ifndef SIMPLA_ATTRIBUTE_H
#define SIMPLA_ATTRIBUTE_H

#include <simpla/SIMPLA_config.h>
#include <simpla/toolbox/Serializable.h>
#include <simpla/toolbox/Printable.h>

#include "MeshBlock.h"
#include "DataBlock.h"
#include "Worker.h"

namespace simpla { namespace mesh
{

/**
 *  AttributeBase IS-A container of datablock
 */
class Attribute :
        public toolbox::Object,
        public toolbox::Printable,
        public toolbox::Serializable,
        public std::enable_shared_from_this<Attribute>
{


public:

    SP_OBJECT_HEAD(Attribute, toolbox::Object)

    Attribute(std::string const &s);

    Attribute(Attribute const &) = delete;

    Attribute(Attribute &&) = delete;

    virtual ~Attribute();

    virtual std::string name() const { return toolbox::Object::name(); };

    virtual std::ostream &print(std::ostream &os, int indent = 1) const;

    virtual void load(const data::DataBase &);

    virtual void save(data::DataBase *) const;

    void insert(MeshBlock const *m, const std::shared_ptr<DataBlock> &);

    virtual bool has(MeshBlock const *) const;

    virtual void erase(MeshBlock const *);

    virtual void deploy(MeshBlock const * = nullptr);

    virtual void clear(MeshBlock const * = nullptr);

    virtual void update(MeshBlock const *, MeshBlock const * = nullptr);

    virtual DataBlock const *at(MeshBlock const *m = nullptr) const;

    virtual DataBlock *at(const MeshBlock *, const MeshBlock *hint = nullptr);

    template<typename TB>
    TB *as(MeshBlock const *m)
    {
        if (!has(m))
        {
            auto res = std::make_shared<TB>(m);
            insert(m, std::dynamic_pointer_cast<DataBlock>(res));
            return res.get();

        } else
        {
            return static_cast<TB *>(at(m));
        }
    };

private:
    struct pimpl_s;
    std::unique_ptr<pimpl_s> m_pimpl_;
};

template<typename TV, typename TM, MeshEntityType IFORM>
class AttributeView :
        public Worker::Observer,
        public toolbox::Printable
{
public:
    typedef TM mesh_type;
    typedef TV value_type;

protected:
    struct scalar_value_tag {};
    struct expression_tag {};
    struct function_tag {};
    struct field_function_tag {};

    typedef AttributeView<value_type, mesh_type, IFORM> this_type;

    typedef typename mesh_type::template data_block_type<TV, IFORM> data_block_type;
    std::shared_ptr<Attribute> m_attr_;
    EntityIdRange m_range_;
protected:
    mesh_type const *m_mesh_ = nullptr;
    data_block_type *m_data_ = nullptr;
public:
    AttributeView(mesh_type *m = nullptr, std::string const &s = "", Worker *w = nullptr) :
            Worker::Observer(w), m_attr_(new Attribute(s)), m_mesh_(m) {};

    AttributeView(std::shared_ptr<mesh_type> const &m, std::string const &s = "", Worker *w = nullptr) :
            Worker::Observer(w), m_attr_(new Attribute(s)), m_mesh_(m.get()) {};

    AttributeView(std::string const &s, Worker *w = nullptr) :
            Worker::Observer(w), m_attr_(new Attribute(s)) {};

    AttributeView(std::shared_ptr<Attribute> const &attr, Worker *w = nullptr) :
            Worker::Observer(w), m_attr_(attr) {};

    virtual ~AttributeView() {}

    AttributeView(AttributeView const &other) = delete;

    AttributeView(AttributeView &&other) = delete;

    virtual bool is_a(std::type_info const &t_info) const { return t_info == typeid(this_type); }

    virtual MeshEntityType entity_type() const { return IFORM; };

    virtual std::type_info const &value_type_info() const { return typeid(value_type); };

    mesh_type const *mesh() const { return m_mesh_; }

    data_block_type *data() const { return m_data_; }

    std::shared_ptr<Attribute> &attribute() { return m_attr_; }

    virtual std::string name() const { return m_attr_->name(); };

    virtual std::ostream &print(std::ostream &os, int indent) const { return m_attr_->print(os, indent); }

    virtual void clear() { m_data_->clear(); };

    virtual void destroy() { UNIMPLEMENTED; };

    virtual void
    create(MeshBlock const *m, bool is_scratch = false)
    {
        auto res = std::dynamic_pointer_cast<DataBlock>(std::make_shared<data_block_type>(m));
        if (!is_scratch) { attribute()->insert(m, res); };
    };

    void deploy(MeshBlock const *m = nullptr)
    {
        move_to(m);
        m_data_ = static_cast<data_block_type *>(m_attr_->at(m_mesh_));
        m_data_->deploy();
    }

    virtual void move_to(MeshBlock const *m) { if (m != nullptr) { m_mesh_ = static_cast<mesh_type const *>(m); }};

    virtual void erase(MeshBlock const *m = nullptr) { UNIMPLEMENTED; };

    virtual void update(MeshBlock const *m = nullptr, bool only_ghost = false) { UNIMPLEMENTED; };


    inline value_type &get(MeshEntityId const &s) { return m_data_->get(s.x, s.y, s.z, s.w); }

    inline value_type const &get(MeshEntityId const &s) const { return m_data_->get(s.x, s.y, s.z, s.w); }

    inline value_type &operator[](MeshEntityId const &s) { return m_data_->get(s.x, s.y, s.z, s.w); }

    inline value_type const &operator[](MeshEntityId const &s) const { return m_data_->get(s.x, s.y, s.z, s.w); }


    template<typename TOP> void
    apply(TOP const &op, EntityIdRange const &r0, this_type const &other)
    {
        deploy();
        r0.foreach([&](MeshEntityId const &s) { op(get(s), other.get(s)); });
    }


    template<typename TOP> void
    apply(TOP const &op, EntityIdRange const &r0, scalar_value_tag *, value_type const &v)
    {
        deploy();
        r0.foreach([&](MeshEntityId const &s) { op(get(s), v); });
    }


    template<typename TOP, typename TFun, typename ...Args> void
    apply(TOP const &op, EntityIdRange const r0, function_tag const *, TFun const &fun, Args &&...args)
    {
        deploy();
        r0.foreach(
                [&](MeshEntityId const &s)
                {
                    op(get(s), fun(m_mesh_->point(s), std::forward<Args>(args)...));
                });
    }

    template<typename TOP, typename ...TExpr> void
    apply(TOP const &op, EntityIdRange const &r0, expression_tag const *, TExpr &&...fexpr)
    {
        deploy();
        r0.foreach([&](MeshEntityId const &s) { op(get(s), m_mesh_->eval(std::forward<TExpr>(fexpr), s)...); });
    }


    template<typename TOP, typename ...Args> void
    apply(TOP const &op, MeshZoneTag const &tag, Args &&...args)
    {
//        apply(op, m_mesh_->range(entity_type(), tag), std::forward<Args>(args)...);
    }

    template<typename TOP, typename TRange, typename ...Args> void
    apply_function(TOP const &op, TRange r0, Args &&...args)
    {
        apply(op, r0, static_cast< function_tag *>(nullptr), std::forward<Args>(args)...);
    }

    template<typename ...Args> void
    assign_function(Args &&...args) { apply_function(_impl::_assign(), std::forward<Args>(args)...); }

};

}} //namespace data
#endif //SIMPLA_ATTRIBUTE_H

