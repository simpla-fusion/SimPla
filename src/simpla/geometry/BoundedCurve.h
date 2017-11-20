//
// Created by salmon on 17-11-1.
//

#ifndef SIMPLA_BOUNDEDCURVE_H
#define SIMPLA_BOUNDEDCURVE_H

#include "gCurve.h"
namespace simpla {
namespace geometry {
class BoundedCurve : public gCurve {
    SP_GEO_ENTITY_ABS_HEAD(gCurve, BoundedCurve);

   public:
    virtual void Open() = 0;
    virtual void Close() = 0;
    virtual size_type size() const = 0;
    virtual void AddPoint(point_type const &) = 0;
    virtual point_type GetPoint(index_type s) const = 0;
};
class BoundedCurve2D : public BoundedCurve {
    SP_GEO_ENTITY_ABS_HEAD(BoundedCurve, BoundedCurve2D);

   public:
    void Deserialize(std::shared_ptr<const simpla::data::DataEntry> const &cfg) override;
    std::shared_ptr<simpla::data::DataEntry> Serialize() const override;

    void Open() override;
    void Close() override;
    bool IsClosed() const override;

    size_type size() const override;
    virtual void AddPoint2D(Real x, Real y);
    virtual point2d_type GetPoint2D(index_type s) const;

    void AddPoint(point_type const &p) override { AddPoint2D(p[0], p[1]); }
    point_type GetPoint(index_type s) const override {
        auto p = GetPoint2D(s);
        return point_type{p[0], p[1], 0};
    }

    std::vector<point2d_type> &data() { return m_data_; }
    std::vector<point2d_type> const &data() const { return m_data_; }

   private:
    std::vector<point2d_type> m_data_;
};
class BoundedCurve3D : public BoundedCurve {
    SP_GEO_ENTITY_ABS_HEAD(BoundedCurve, BoundedCurve3D);

   public:
    void Deserialize(std::shared_ptr<const simpla::data::DataEntry> const &cfg) override;
    std::shared_ptr<simpla::data::DataEntry> Serialize() const override;

    void Open() override;
    void Close() override;
    bool IsClosed() const override;

    size_type size() const override { return m_data_.size(); }
    virtual void AddPoint(Real x, Real y, Real z) { AddPoint(point_type{x, y, z}); }
    void AddPoint(point_type const &) override;
    point_type GetPoint(index_type s) const override;
    std::vector<point_type> &data() { return m_data_; }
    std::vector<point_type> const &data() const { return m_data_; }

   private:
    std::vector<point_type> m_data_;
};
}  // namespace geometry
}  // namespace simpla

#endif  // SIMPLA_BOUNDEDCURVE_H
