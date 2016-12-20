//
// Created by salmon on 16-12-18.
//

#ifndef SIMPLA_PHYSICALQUANTITY_H
#define SIMPLA_PHYSICALQUANTITY_H

#include <simpla/SIMPLA_config.h>
#include <simpla/mesh/Attribute.h>

namespace simpla { namespace manifold
{

template<typename TV, typename TM, mesh::MeshEntityType I, size_type DOF>
class PhysicalQuantity : public mesh::Attribute
{
    typedef PhysicalQuantity<TV, TM, I, DOF> field_type;

SP_OBJECT_HEAD(field_type, mesh::Attribute);

private:
    static constexpr mesh::MeshEntityType IFORM = I;

public:
    typedef TV value_type;
    typedef TM mesh_type;

private:
    typedef typename mesh_type::template data_block_type<TV, IFORM, DOF> data_block_type;

    data_block_type *m_data_;
    mesh_type const *m_mesh_;

public:


    template<typename ...Args>
    explicit PhysicalQuantity(Args &&...args):
            base_type(nullptr, std::make_shared<mesh::AttributeDescTemp<TV, I, DOF>>(std::forward<Args>(args)  ...)),
            m_mesh_(nullptr),
            m_data_(nullptr) {};


    virtual ~PhysicalQuantity() {}

    PhysicalQuantity(this_type const &other) = delete;

    PhysicalQuantity(this_type &&other) = delete;

    virtual std::shared_ptr<mesh::Attribute> clone() const
    {
        return std::dynamic_pointer_cast<mesh::Attribute>(std::make_shared<this_type>());
    };

    bool empty() const { return m_data_ == nullptr || m_data_->empty() || m_mesh_ == nullptr; };

    virtual mesh::MeshEntityType entity_type() const { return IFORM; };

    virtual std::type_info const &value_type_info() const { return typeid(typename traits::value_type<TV>::type); };

    virtual size_type dof() const { return DOF; };

    virtual std::shared_ptr<mesh::DataBlock>
    create_data_block(std::shared_ptr<mesh::MeshBlock> const &m, value_type *p = nullptr) const
    {
        return data_block_type::create(m, p);
    };

    virtual void pre_process()
    {
        if (base_type::is_valid()) { return; } else { base_type::pre_process(); }

//        m_mesh_ = base_type::mesh_as<mesh_type>();
//        m_data_ = base_type::data_as<data_block_type>();
        ASSERT(m_data_ != nullptr);
        ASSERT(m_mesh_ != nullptr);

    }

    virtual void post_process()
    {
        if (!base_type::is_valid()) { return; } else { base_type::post_process(); }

        m_mesh_ = nullptr;
        m_data_ = nullptr;

    }

/** @name as_function  @{*/

/** @name as_array   @{*/

    virtual value_type &
    get(mesh::MeshEntityId s) { return m_data_->get(mesh::MeshEntityIdCoder::unpack_index4(s, DOF)); }

    virtual value_type const &
    get(mesh::MeshEntityId s) const { return m_data_->get(mesh::MeshEntityIdCoder::unpack_index4(s, DOF)); }


    virtual value_type &
    get(index_type i, index_type j, index_type k = 0, index_type l = 0) { return m_data_->get(i, j, k, l); }

    virtual value_type const &
    get(index_type i, index_type j, index_type k = 0, index_type l = 0) const { return m_data_->get(i, j, k, l); }

    template<typename TI>
    inline value_type &operator[](TI const &s) { return get(s); }

    template<typename TI>
    inline value_type const &operator[](TI const &s) const { return get(s); }

    template<typename Other> inline this_type &
    operator=(Other const &other)
    {
        assign(other);
        return *this;
    }

    template<typename Other> inline this_type &
    operator+=(Other const &other)
    {
        *this = *this + other;
        return *this;
    }

    template<typename Other> inline this_type &
    operator-=(Other const &other)
    {
        *this = *this - other;
        return *this;
    }

    template<typename Other> inline this_type &
    operator*=(Other const &other)
    {
        *this = *this * other;
        return *this;
    }

    template<typename Other> inline this_type &
    operator/=(Other const &other)
    {
        *this = *this / other;
        return *this;
    }

    inline this_type &
    operator=(this_type const &other)
    {
        assign(other);
        return *this;
    }
/* @}*/
private:

public:


    typedef mesh::MeshEntityIdCoder M;

    void assign(this_type const &other, mesh::EntityIdRange const &r0)
    {
        pre_process();

        r0.foreach([&](mesh::MeshEntityId const &s)
                   {
                       for (int i = 0; i < DOF; ++i) { get(M::sw(s, i)) = other.get(M::sw(s, i)); }
                   });

    }


    template<typename Other> void
    assign(Other const &other, mesh::MeshZoneTag const &tag = mesh::SP_ES_ALL)
    {
        pre_process();
        if (tag == mesh::SP_ES_ALL)
        {
            assign(other, m_data_->range());
        } else
        {
            assign(other, m_mesh_->mesh_block()->range(entity_type(), tag));
        }
    }

    void copy(mesh::EntityIdRange const &r0, this_type const &g)
    {
        UNIMPLEMENTED;
//        r0.assign([&](mesh::MeshEntityId const &s) { get(s) = g.get(s); });
    }


    virtual void copy(mesh::EntityIdRange const &r0, mesh::DataBlock const &other)
    {
        UNIMPLEMENTED;
//        assert(other.is_a(typeid(this_type)));
//
//        this_type const &g = static_cast<this_type const & >(other);
//
//        copy(r0, static_cast<this_type const & >(other));

    }


};
}}//namespace simpla { namespace manifold

#endif //SIMPLA_PHYSICALQUANTITY_H
