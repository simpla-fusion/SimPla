//
// Created by salmon on 17-7-16.
//

#ifndef SIMPLA_MESH_H
#define SIMPLA_MESH_H

#include "Attribute.h"
#include "SPObject.h"
#include "simpla/data/EnableCreateFromDataTable.h"
#include "simpla/geometry/GeoObject.h"

namespace simpla {
namespace geometry {
struct Chart;
}
namespace engine {
class MeshBlock;
class Patch;

struct MeshBase : public SPObject, public AttributeGroup, public data::EnableCreateFromDataTable<MeshBase> {
    SP_OBJECT_HEAD(MeshBase, SPObject)
    DECLARE_REGISTER_NAME(MeshBase)
   public:
    using AttributeGroup::attribute_type;

    MeshBase();
    ~MeshBase() override;

    MeshBase(MeshBase const &other) = delete;
    MeshBase(MeshBase &&other) noexcept = delete;
    void swap(MeshBase &other) = delete;
    MeshBase &operator=(this_type const &other) = delete;
    MeshBase &operator=(this_type &&other) noexcept = delete;

    virtual const geometry::Chart &GetChart() const = 0;
    virtual geometry::Chart &GetChart() = 0;

    virtual this_type *GetMesh() { return this; }
    virtual this_type const *GetMesh() const { return this; }

    virtual void AddGeoObject(std::string const &prefix, geometry::GeoObject const *g){};

    void SetBlock(const MeshBlock &blk);
    virtual const MeshBlock &GetBlock() const;
    virtual id_type GetBlockId() const;

    std::shared_ptr<data::DataTable> Serialize() const override;
    void Deserialize(std::shared_ptr<data::DataTable> const &t) override;

    void DoInitialize() override;
    void DoFinalize() override;
    void DoUpdate() override;
    void DoTearDown() override;

    virtual void DoInitialCondition(Real time_now) {}
    virtual void DoBoundaryCondition(Real time_now, Real dt) {}
    virtual void DoAdvance(Real time_now, Real dt) {}

    void InitialCondition(Real time_now);
    void BoundaryCondition(Real time_now, Real dt);
    void Advance(Real time_now, Real dt);

    void Pull(Patch *p) override;
    void Push(Patch *p) override;

    void InitialCondition(Patch *patch, Real time_now);
    void BoundaryCondition(Patch *patch, Real time_now, Real dt);
    void Advance(Patch *patch, Real time_now, Real dt);

    void SetRange(std::string const &, Range<EntityId> const &);
    Range<EntityId> &GetRange(std::string const &k);
    Range<EntityId> GetRange(std::string const &k) const;

   private:
    MeshBlock m_mesh_block_;
    std::shared_ptr<geometry::Chart> m_chart_ = nullptr;

    struct pimpl_s;
    std::unique_ptr<pimpl_s> m_pimpl_;
};

template <typename TChart, template <typename> class... Policies>
class Mesh : public MeshBase, public Policies<Mesh<TChart, Policies...>>... {
   public:
    DECLARE_REGISTER_NAME(Mesh)

    typedef Mesh<TChart, Policies...> this_type;
    Mesh() : Policies<this_type>(this)... {};
    ~Mesh() override = default;

    TChart &GetChart() override { return m_chart_; };
    const TChart &GetChart() const override { return m_chart_; };

    const engine::MeshBlock &GetBlock() const override { return MeshBase::GetBlock(); }

    this_type *GetMesh() override { return this; }
    this_type const *GetMesh() const override { return this; }

    void DoInitialCondition(Real time_now) override;
    void DoBoundaryCondition(Real time_now, Real dt) override;
    void DoAdvance(Real time_now, Real dt) override;

    void Deserialize(std::shared_ptr<data::DataTable> const &cfg) override;
    std::shared_ptr<data::DataTable> Serialize() const override;

    void AddGeoObject(std::string const &prefix, geometry::GeoObject const *g) override;

    template <typename TL, typename TR>
    void Fill(TL &lhs, TR &&rhs) const {
        FillRange(lhs, std::forward<TR>(rhs), Range<EntityId>{}, true);
    };

    template <typename TL, typename TR>
    void FillRange(TL &lhs, TR &&rhs, Range<EntityId> r = Range<EntityId>{},
                   bool full_fill_if_range_is_null = false) const;

    template <typename TL, typename TR>
    void FillRange(TL &lhs, TR &&rhs, std::string const &k = "", bool full_fill_if_range_is_null = false) const {
        FillRange(lhs, std::forward<TR>(rhs), GetRange(k), full_fill_if_range_is_null);
    };

    template <typename TL, typename TR>
    void FillBody(TL &lhs, TR &&rhs, std::string prefix = "") const {
        FillRange(lhs, std::forward<TR>(rhs), prefix + "_BODY_" + std::to_string(TL::iform), true);
    };

    template <typename TL, typename TR>
    void FillBoundary(TL &lhs, TR &&rhs, std::string prefix = "") const {
        FillRange(lhs, std::forward<TR>(rhs), prefix + "_BOUNDARY_" + std::to_string(TL::iform), false);
    };

   private:
    TChart m_chart_;
};

// template <typename TM, template <typename> class... Policies>
// bool Domain<TM, Policies...>::is_registered = DomainBase::RegisterCreator<Domain<TM, Policies...>>();
namespace mesh_detail {

#define DEFINE_INVOKE_HELPER(_FUN_NAME_)                                                                           \
    CHECK_MEMBER_FUNCTION(has_mem_fun_##_FUN_NAME_, _FUN_NAME_)                                                    \
    template <typename this_type, typename... Args>                                                                \
    int _invoke_##_FUN_NAME_(std::true_type const &has_function, this_type *self, Args &&... args) {               \
        self->_FUN_NAME_(std::forward<Args>(args)...);                                                             \
        return 1;                                                                                                  \
    }                                                                                                              \
    template <typename this_type, typename... Args>                                                                \
    int _invoke_##_FUN_NAME_(std::false_type const &has_not_function, this_type *self, Args &&... args) {          \
        return 0;                                                                                                  \
    }                                                                                                              \
    template <template <typename> class _T0, typename this_type, typename... Args>                                 \
    int _try_invoke_##_FUN_NAME_(this_type const *self, Args &&... args) {                                         \
        return _invoke_##_FUN_NAME_(has_mem_fun_##_FUN_NAME_<_T0<this_type> const, void, Args...>(),               \
                                    dynamic_cast<_T0<this_type> const *>(self), std::forward<Args>(args)...);      \
    }                                                                                                              \
    template <template <typename> class _T0, typename this_type, typename... Args>                                 \
    int _try_invoke_##_FUN_NAME_(this_type *self, Args &&... args) {                                               \
        return _invoke_##_FUN_NAME_(has_mem_fun_##_FUN_NAME_<_T0<this_type>, void, Args...>(),                     \
                                    dynamic_cast<_T0<this_type> *>(self), std::forward<Args>(args)...);            \
    }                                                                                                              \
    template <template <typename> class _T0, template <typename> class _T1, template <typename> class... _TOthers, \
              typename this_type, typename... Args>                                                                \
    int _try_invoke_##_FUN_NAME_(this_type *self, Args &&... args) {                                               \
        return _try_invoke_##_FUN_NAME_<_T0>(self, std::forward<Args>(args)...) +                                  \
               _try_invoke_##_FUN_NAME_<_T1, _TOthers...>(self, std::forward<Args>(args)...);                      \
    }                                                                                                              \
    template <template <typename> class _T0, template <typename> class _T1, template <typename> class... _TOthers, \
              typename this_type, typename... Args>                                                                \
    int _try_invoke_once_##_FUN_NAME_(this_type *self, Args &&... args) {                                          \
        if (_try_invoke_##_FUN_NAME_<_T0>(self, std::forward<Args>(args)...) == 0) {                               \
            return _try_invoke_##_FUN_NAME_<_T1, _TOthers...>(self, std::forward<Args>(args)...);                  \
        } else {                                                                                                   \
            return 1;                                                                                              \
        }                                                                                                          \
    }

DEFINE_INVOKE_HELPER(InitialCondition)
DEFINE_INVOKE_HELPER(BoundaryCondition)
DEFINE_INVOKE_HELPER(Advance)
DEFINE_INVOKE_HELPER(Deserialize)
DEFINE_INVOKE_HELPER(Serialize)
DEFINE_INVOKE_HELPER(SetGeoObject)
DEFINE_INVOKE_HELPER(Calculate)

#undef DEFINE_INVOKE_HELPER
}  // namespace mesh_detail
template <typename TM, template <typename> class... Policies>
void Mesh<TM, Policies...>::DoInitialCondition(Real time_now) {
    mesh_detail::_try_invoke_InitialCondition<Policies...>(this, time_now);
}
template <typename TM, template <typename> class... Policies>
void Mesh<TM, Policies...>::DoBoundaryCondition(Real time_now, Real dt) {
    mesh_detail::_try_invoke_BoundaryCondition<Policies...>(this, time_now, dt);
}
template <typename TM, template <typename> class... Policies>
void Mesh<TM, Policies...>::DoAdvance(Real time_now, Real dt) {
    mesh_detail::_try_invoke_Advance<Policies...>(this, time_now, dt);
}
template <typename TM, template <typename> class... Policies>
std::shared_ptr<data::DataTable> Mesh<TM, Policies...>::Serialize() const {
    auto res = MeshBase::Serialize();
    mesh_detail::_try_invoke_Serialize<Policies...>(this, res.get());
    return res;
};
template <typename TM, template <typename> class... Policies>
void Mesh<TM, Policies...>::Deserialize(std::shared_ptr<data::DataTable> const &cfg) {
    mesh_detail::_try_invoke_Deserialize<Policies...>(this, cfg);
    MeshBase::Deserialize(cfg);
};
template <typename TM, template <typename> class... Policies>
void Mesh<TM, Policies...>::AddGeoObject(std::string const &prefix, geometry::GeoObject const *g) {
    mesh_detail::_try_invoke_SetGeoObject<Policies...>(this, prefix, g);
};

template <typename TM, template <typename> class... Policies>
template <typename LHS, typename RHS>
void Mesh<TM, Policies...>::FillRange(LHS &lhs, RHS &&rhs, Range<EntityId> r, bool full_fill_if_range_is_null) const {
    if (r.isNull() && full_fill_if_range_is_null) {
        mesh_detail::_try_invoke_once_Calculate<Policies...>(this, lhs, std::forward<RHS>(rhs));
    } else {
        mesh_detail::_try_invoke_once_Calculate<Policies...>(this, lhs, std::forward<RHS>(rhs), r);
    }
};

#define MESH_POLICY_HEAD(_NAME_)                     \
   private:                                          \
    typedef THost host_type;                         \
    typedef _NAME_<THost> this_type;                 \
                                                     \
   public:                                           \
    host_type *m_host_ = nullptr;                    \
    _NAME_(host_type *h) noexcept : m_host_(h) {}    \
    virtual ~_NAME_() = default;                     \
    _NAME_(_NAME_ const &other) = delete;            \
    _NAME_(_NAME_ &&other) = delete;                 \
    _NAME_ &operator=(_NAME_ const &other) = delete; \
    _NAME_ &operator=(_NAME_ &&other) = delete;      \
    static std::string RegisterName() { return __STRING(_NAME_); }

}  // namespace mesh

}  // namespace simpla{

#endif  // SIMPLA_MESH_H
