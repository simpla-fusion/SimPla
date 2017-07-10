//
// Created by salmon on 17-7-9.
//

#ifndef SIMPLA_TOKAMAK_H
#define SIMPLA_TOKAMAK_H

#include "GEqdsk.h"
#include "simpla/model/Model.h"
namespace simpla {
namespace model {

class Tokamak : public Model {
    SP_OBJECT_HEAD(Tokamak, Model)
    DECLARE_REGISTER_NAME(Tokamak);

   public:
    Tokamak();
    ~Tokamak() override = default;

    Tokamak(Tokamak const &) = delete;
    Tokamak(Tokamak &&) = delete;
    Tokamak &operator=(Tokamak const &) = delete;
    Tokamak &operator=(Tokamak &&) = delete;

    std::shared_ptr<data::DataTable> Serialize() const override;
    void Deserialize(const std::shared_ptr<DataTable> &cfg) override;
    void DoUpdate() override;

   private:
    struct pimpl_s;
    std::unique_ptr<pimpl_s> m_pimpl_;
};

}  // namespace model{
}  // namespace simpla
#endif  // SIMPLA_TOKAMAK_H
