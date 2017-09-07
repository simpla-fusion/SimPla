//
// Created by salmon on 17-7-9.
//

#ifndef SIMPLA_TOKAMAK_H
#define SIMPLA_TOKAMAK_H

#include "GEqdsk.h"
#include "simpla/engine/EngineObject.h"
namespace simpla {
class Tokamak {
   protected:
    explicit Tokamak(std::string const &url = "");

   public:
    ~Tokamak();
    template <typename... Args>
    static std::shared_ptr<Tokamak> New(Args &&... args) {
        return std::shared_ptr<Tokamak>(new Tokamak(std::forward<Args>(args)...));
    };
    void LoadGFile(std::string const &);
    std::shared_ptr<geometry::GeoObject> Limiter() const;
    std::shared_ptr<geometry::GeoObject> Boundary() const;

    std::function<Vec3(point_type const &)> B0() const;
    std::function<Real(point_type const &)> profile(std::string const &k) const;

   private:
    struct pimpl_s;
    pimpl_s *m_pimpl_ = nullptr;
};

}  // namespace simpla
#endif  // SIMPLA_TOKAMAK_H
