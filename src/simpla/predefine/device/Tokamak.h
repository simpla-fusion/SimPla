//
// Created by salmon on 17-7-9.
//

#ifndef SIMPLA_TOKAMAK_H
#define SIMPLA_TOKAMAK_H

#include "GEqdsk.h"
#include "simpla/engine/EngineObject.h"
#include "simpla/engine/Model.h"
namespace simpla {
class Tokamak {
   public:
    Tokamak();
    ~Tokamak();
    void DoUpdate();
    void LoadGFile(std::string const &);
    typedef std::function<Real(point_type const &)> attr_fun;
    typedef std::function<Vec3(point_type const &)> vec_attr_fun;
    engine::Model::attr_fun GetAttribute(std::string const &attr_name) const;
    engine::Model::vec_attr_fun GetAttributeVector(std::string const &attr_name) const;

   private:
    std::shared_ptr<engine::Model> m_self_;
    struct pimpl_s;
    pimpl_s *m_pimpl_ = nullptr;
};

}  // namespace simpla
#endif  // SIMPLA_TOKAMAK_H
