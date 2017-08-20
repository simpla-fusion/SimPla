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
    void DoUpdate() override;
    void LoadGFile(std::string const &);
    engine::Model::attr_fun GetAttribute(std::string const &attr_name) const override;
    engine::Model::vec_attr_fun GetAttributeVector(std::string const &attr_name) const override;
};

}  // namespace simpla
#endif  // SIMPLA_TOKAMAK_H
