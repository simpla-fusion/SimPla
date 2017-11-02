//
// Created by salmon on 17-7-27.
//

#include "GeoObjectOCE.h"

#include <BRepBndLib.hxx>
#include <BRepBuilderAPI_MakeVertex.hxx>
#include <BRepBuilderAPI_Transform.hxx>
#include <BRepExtrema_DistShapeShape.hxx>
#include <Bnd_Box.hxx>
#include <Interface_Static.hxx>
#include <STEPControl_Reader.hxx>
#include <StlAPI_Reader.hxx>
#include <TColStd_HSequenceOfTransient.hxx>
#include <TopoDS_Shape.hxx>
#include <gp_Quaternion.hxx>

#include "simpla/utilities/SPDefines.h"
namespace simpla {
namespace geometry {

}
}  // namespace simpla