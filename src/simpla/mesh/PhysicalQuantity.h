//
// Created by salmon on 16-12-18.
//

#ifndef SIMPLA_PHYSICALQUANTITY_H
#define SIMPLA_PHYSICALQUANTITY_H

#include <simpla/SIMPLA_config.h>
#include <simpla/mesh/AttributeView.h>

namespace simpla
{


template<typename TV, typename TM, size_type I, size_type DOF>
class PhysicalQuantity : public mesh::AttributeView
{
    typedef PhysicalQuantity<TV, TM, I, DOF> field_type;

SP_OBJECT_HEAD(field_type, mesh::Attribute);


public:
    typedef TV value_type;
    typedef TM mesh_type;
    static constexpr mesh::MeshEntityType IFORM = I;
private:
    typedef typename mesh_type::template data_block_type<TV, IFORM, DOF> data_block_type;

    data_block_type *m_data_;
    mesh_type const *m_mesh_;

public:


    template<typename ...Args>
    explicit PhysicalQuantity(Args &&...args):
            m_mesh_(nullptr),
            m_data_(nullptr), AttributeView(<#initializer#>, <#initializer#>),
            AttributeView(<#initializer#>, nullptr, <#initializer#>) {};


    virtual ~PhysicalQuantity() {}

    PhysicalQuantity(this_type const &other) = delete;

    PhysicalQuantity(this_type &&other) = delete;

    virtual std::shared_ptr<mesh::AttributeView> clone() const
    {
        return std::dynamic_pointer_cast<mesh::AttributeView>(std::make_shared<this_type>());
    };

    bool empty() const { return m_data_ == nullptr || m_data_->empty() || m_mesh_ == nullptr; };

    virtual mesh::MeshEntityType entity_type() const { return IFORM; };

    virtual std::type_info const &value_type_info() const { return typeid(TV); };

    virtual size_type dof() const { return DOF; };

    virtual std::shared_ptr<mesh::DataBlock>
    create_data_block(std::shared_ptr<mesh::MeshBlock> const &m, value_type *p = nullptr) const
    {
        return data_block_type::create(m, p);
    };

    virtual void PreProcess()
    {
        if (base_type::isValid()) { return; } else { base_type::PreProcess(); }

//        m_chart_ = self_type::mesh_as<mesh_type>();
//        m_value_ = self_type::data_as<data_block_type>();
        ASSERT(m_data_ != nullptr);
        ASSERT(m_mesh_ != nullptr);

    }

    virtual void PostProcess()
    {
        if (!base_type::isValid()) { return; } else { base_type::PostProcess(); }

        m_mesh_ = nullptr;
        m_data_ = nullptr;

    }

/** @name as_function  @{*/

/** @name as_array   @{*/

    virtual value_type &
    get(EntityId s) { return m_data_->get(EntityIdCoder::unpack_index4(s, DOF)); }

    virtual value_type const &
    get(EntityId s) const { return m_data_->get(EntityIdCoder::unpack_index4(s, DOF)); }


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


    typedef EntityIdCoder M;

    void assign(this_type const &other, EntityIdRange const &r0)
    {
        PreProcess();

        r0.foreach([&](EntityId const &s)
                   {
                       for (int i = 0; i < DOF; ++i) { get(M::sw(s, i)) = other.get(M::sw(s, i)); }
                   });

    }


    template<typename Other> void
    assign(Other const &other, mesh::MeshZoneTag const &tag = mesh::SP_ES_ALL)
    {
        PreProcess();
        if (tag == mesh::SP_ES_ALL)
        {
            assign(other, m_data_->range());
        } else
        {
            assign(other, m_mesh_->mesh_block()->range(entity_type(), tag));
        }
    }

    void copy(EntityIdRange const &r0, this_type const &g)
    {
        UNIMPLEMENTED;
//        r0.Assign([&](EntityId const &s) { get(s) = g.Get(s); });
    }


    virtual void copy(EntityIdRange const &r0, mesh::DataBlock const &other)
    {
        UNIMPLEMENTED;
//        assert(other.is_a(typeid(this_type)));
//
//        this_type const &g = dynamic_cast<this_type const & >(other);
//
//        Duplicate(r0, dynamic_cast<this_type const & >(other));

    }


};

namespace traits
{



template<typename TV, typename TM, mesh::MeshEntityType I, size_type DOF>
struct iform<PhysicalQuantity<TV, TM, I, DOF> > : public std::integral_constant<size_t, I> {};

template<typename TV, typename TM, mesh::MeshEntityType I, size_type DOF>
struct value_type<PhysicalQuantity<TV, TM, I, DOF> > { typedef TV value_type; };

}

}//namespace simpla

#endif //SIMPLA_PHYSICALQUANTITY_H
