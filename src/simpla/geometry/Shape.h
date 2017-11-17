//
// Created by salmon on 17-11-6.
//

#ifndef SIMPLA_SHAPE_H
#define SIMPLA_SHAPE_H

#include <memory>
#include "GeoObject.h"
namespace simpla {
namespace geometry {
struct Shape : public data::Serializable {
   private:
    typedef Shape this_type;

   public:
    std::string FancyTypeName() const override { return "Shape"; }

   protected:
    Shape();
    Shape(Shape const &other);

   public:
    virtual ~Shape();
    static std::shared_ptr<Shape> Create(std::string const &key);
    virtual std::shared_ptr<Shape> Copy() const = 0;
};

}  // namespace geometry{
}  // namespace simpla{
#endif  // SIMPLA_SHAPE_H
