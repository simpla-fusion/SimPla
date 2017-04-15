//
// Created by salmon on 17-4-11.
//

#ifndef SIMPLA_PEC_H
#define SIMPLA_PEC_H

#include <simpla/SIMPLA_config.h>
#include <simpla/algebra/all.h>
#include <simpla/engine/all.h>
#include <simpla/physics/PhysicalConstants.h>
#include <simpla/toolbox/Log.h>
namespace simpla {
using namespace engine;

/**
 *  @ingroup
 *  @brief   PEC
 */
template <typename TChart>
class PEC : public engine::Worker {
    SP_OBJECT_HEAD(PEC<TChart>, engine::Worker);

   public:
    static const bool is_register;

    typedef engine::MeshView<TChart> mesh_type;
    typedef algebra::traits::scalar_type_t<mesh_type> scalar_type;
    mesh_type m_mesh_;
    template <int IFORM, int DOF = 1>
    using field_type = Field<mesh_type, scalar_type, IFORM, DOF>;

    template <typename... Args>
    explicit PEC(Args&&... args) : engine::Worker(std::forward<Args>(args)...){};
    virtual ~PEC(){};

    virtual std::shared_ptr<data::DataTable> Serialize() const {
        auto res = std::make_shared<data::DataTable>();
        res->SetValue<std::string>("Type", "PEC");
        return res;
    };
    virtual void Deserialize(std::shared_ptr<data::DataTable> const& t) { UNIMPLEMENTED; }

    void Initialize();
    void Run(Real time, Real dt);
    field_type<EDGE> E{&m_mesh_, "name"_ = "E"};
    field_type<FACE> B{&m_mesh_, "name"_ = "B"};

   private:
};
template <typename TM>
const bool PEC<TM>::is_register = engine::Worker::RegisterCreator<PEC<TM>>(std::string("PEC<") + TM::ClassName() + ">");

template <typename TM>
void PEC<TM>::Initialize() {}

template <typename TM>
void PEC<TM>::Run(Real time, Real dt) {
    E = 0.0;
    B = 0.0;
}

}  // namespace simpla

#endif  // SIMPLA_PEC_H
