//
// Created by salmon on 17-7-9.
//

#ifndef SIMPLA_TOKAMAK_H
#define SIMPLA_TOKAMAK_H

#include "GEqdsk.h"
#include "simpla/engine/Model.h"
#include "simpla/engine/SPObject.h"
namespace simpla {

class Tokamak : public engine::Model {
    SP_OBJECT_HEAD(Tokamak, engine::Model)

   public:
    Tokamak();
    ~Tokamak() override = default;

    Tokamak(Tokamak const &) = delete;
    Tokamak(Tokamak &&) = delete;
    Tokamak &operator=(Tokamak const &) = delete;
    Tokamak &operator=(Tokamak &&) = delete;

    void DoUpdate() override;

    void LoadGFile(std::string const &);

    std::shared_ptr<data::DataTable> Serialize() const override;
    void Deserialize(const std::shared_ptr<data::DataTable> &cfg) override;

    engine::Model::attr_fun GetAttribute(std::string const &attr_name) const override;
    engine::Model::vec_attr_fun GetAttributeVector(std::string const &attr_name) const override;

   private:
    struct pimpl_s;
    std::unique_ptr<pimpl_s> m_pimpl_;
};

}  // namespace simpla
#endif  // SIMPLA_TOKAMAK_H
