//
// Created by salmon on 17-11-1.
//

#ifndef SIMPLA_BOUNDEDCURVE_H
#define SIMPLA_BOUNDEDCURVE_H

#include "Curve.h"
namespace simpla {
namespace geometry {
class BoundedCurve : public Curve {
    SP_SERIALIZABLE_HEAD(Curve, BoundedCurve);

   public:
    virtual void Open() = 0;
    virtual void Close() = 0;
    virtual size_type size() const = 0;
    virtual void AddPoint(point_type const &) = 0;
    virtual point_type GetPoint(index_type s) const = 0;
};
class BoundedCurve2D : public BoundedCurve {
    SP_SERIALIZABLE_HEAD(BoundedCurve, BoundedCurve2D);

   public:
    point_type xyz(Real u) const override;

    void Open() override;
    void Close() override;
    bool IsClosed() const override;

    size_type size() const override;
    void AddPoint(point_type const &) override;
    point_type GetPoint(index_type s) const override;
    virtual void AddPoint(Real x, Real y);
    virtual point2d_type GetPoint2d(index_type s) const;
    std::vector<point2d_type> &data();
    std::vector<point2d_type> const &data() const;

   private:
    struct pimpl_s;
    pimpl_s *m_pimpl_ = nullptr;
};
class BoundedCurve3D : public BoundedCurve {
    SP_SERIALIZABLE_HEAD(BoundedCurve, BoundedCurve3D);

   public:
    point_type xyz(Real u) const override;

    void Open() override;
    void Close() override;
    bool IsClosed() const override;

    size_type size() const override;
    void AddPoint(Real x, Real y, Real z);
    void AddPoint(point_type const &) override;
    point_type GetPoint(index_type s) const override;
    std::vector<point_type> &data();
    std::vector<point_type> const &data() const;

   private:
    struct pimpl_s;
    pimpl_s *m_pimpl_ = nullptr;
};
}  // namespace geometry
}  // namespace simpla

#endif  // SIMPLA_BOUNDEDCURVE_H
