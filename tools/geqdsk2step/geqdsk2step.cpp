//
// Created by salmon on 16-11-27.
//
#include <simpla/toolbox/Log.h>
#include <simpla/algebra/nTupleExt.h>

#include <oce/BRep_Tool.hxx>

#include <oce/BRepAlgoAPI_Fuse.hxx>

#include <oce/BRepBuilderAPI_MakeEdge.hxx>
#include <oce/BRepBuilderAPI_MakeFace.hxx>
#include <oce/BRepBuilderAPI_MakeWire.hxx>
#include <oce/BRepBuilderAPI_Transform.hxx>

#include <oce/BRepFilletAPI_MakeFillet.hxx>

#include <oce/BRepLib.hxx>

#include <oce/BRepOffsetAPI_MakeThickSolid.hxx>
#include <oce/BRepOffsetAPI_ThruSections.hxx>

#include <oce/BRepPrimAPI_MakeCylinder.hxx>
#include <oce/BRepPrimAPI_MakePrism.hxx>

#include <oce/GC_MakeArcOfCircle.hxx>
#include <oce/GC_MakeSegment.hxx>

#include <oce/GCE2d_MakeSegment.hxx>

#include <oce/gp.hxx>
#include <oce/gp_Ax1.hxx>
#include <oce/gp_Ax2.hxx>
#include <oce/gp_Ax2d.hxx>
#include <oce/gp_Dir.hxx>
#include <oce/gp_Dir2d.hxx>
#include <oce/gp_Pnt.hxx>
#include <oce/gp_Pnt2d.hxx>
#include <oce/gp_Trsf.hxx>
#include <oce/gp_Vec.hxx>

#include <oce/Geom_CylindricalSurface.hxx>
#include <oce/Geom_Plane.hxx>
#include <oce/Geom_Surface.hxx>
#include <oce/Geom_TrimmedCurve.hxx>

#include <oce/Geom2d_Ellipse.hxx>
#include <oce/Geom2d_TrimmedCurve.hxx>

#include <oce/TopExp_Explorer.hxx>

#include <oce/TopoDS.hxx>
#include <oce/TopoDS_Edge.hxx>
#include <oce/TopoDS_Face.hxx>
#include <oce/TopoDS_Wire.hxx>
#include <oce/TopoDS_Shape.hxx>
#include <oce/TopoDS_Compound.hxx>

#include <oce/TopTools_ListOfShape.hxx>

#include <oce/TopoDS_Shape.hxx>
#include <oce/BRepPrimAPI_MakeSphere.hxx>
#include <oce/TDocStd_Document.hxx>
#include <oce/Handle_TDocStd_Document.hxx>
#include <oce/XCAFApp_Application.hxx>
#include <oce/Handle_XCAFApp_Application.hxx>
#include <oce/XCAFDoc_ShapeTool.hxx>
#include <oce/Handle_XCAFDoc_ShapeTool.hxx>
#include <oce/XCAFDoc_DocumentTool.hxx>
#include <oce/STEPCAFControl_Writer.hxx>
#include <oce/STEPCAFControl_Writer.hxx>
#include <oce/XCAFDoc_DocumentTool.hxx>
#include <oce/XCAFApp_Application.hxx>
#include <oce/GeomAPI_PointsToBSpline.hxx>
#include <oce/TColgp_Array1OfPnt.hxx>
#include <oce/Geom_BSplineCurve.hxx>
#include <oce/GeomAPI_Interpolate.hxx>
#include <oce/TColgp_HArray1OfPnt.hxx>

#include <oce/AIS.hxx>
#include <oce/AIS_Shape.hxx>
#include <oce/AIS_InteractiveContext.hxx>
#include <oce/Graphic3d_AspectLine3d.hxx>
#include <oce/Graphic3d_AspectMarker3d.hxx>
#include <oce/Graphic3d_AspectFillArea3d.hxx>
#include <oce/Graphic3d_AspectText3d.hxx>
#include <oce/Graphic3d_GraphicDriver.hxx>
#include <oce/OpenGl_GraphicDriver.hxx>
#include <oce/V3d.hxx>
#include <oce/V3d_View.hxx>
#include <oce/V3d_Viewer.hxx>


#include <oce/BRepPrimAPI_MakeBox.hxx>
#include <oce/Standard_Real.hxx>


#include <simpla/model/GEqdsk.h>
#include <BRepBuilderAPI_MakePolygon.hxx>
#include <BRepPrimAPI_MakeRevol.hxx>
#include <oce/BRepPrimAPI_MakeRevol.hxx>
#include <oce/TDataStd_Name.hxx>
#include <oce/BRepBuilderAPI_MakePolygon.hxx>

namespace simpla
{
void convert_geqdsk2step(GEqdsk const &geqdsk, std::string const &filename)
{


    {
        BRepBuilderAPI_MakeWire wireMaker;

        Handle(TColgp_HArray1OfPnt) gp_array = new TColgp_HArray1OfPnt(1, geqdsk.boundary().data().size() - 1);

        for (size_type i = 0, ie = geqdsk.boundary().data().size() - 1; i < ie; ++i)
        {
            gp_array->setValue(i + 1, gp_Pnt(geqdsk.boundary().data()[i][0], 0, geqdsk.boundary().data()[i][1]));
        }

        GeomAPI_Interpolate sp(gp_array, true, 1.0e-3);
        sp.Perform();

        wireMaker.Add(BRepBuilderAPI_MakeEdge(sp.Curve()));


        gp_Ax1 axis(gp_Pnt(0, 0, 0), gp_Dir(0, 0, 1));

        TopoDS_Face myBoundaryFaceProfile = BRepBuilderAPI_MakeFace(wireMaker.Wire());
        TopoDS_Shape myBoundary = BRepPrimAPI_MakeRevol(myBoundaryFaceProfile, axis);

        // Create document
        Handle(TDocStd_Document) aDoc;
        Handle(XCAFApp_Application) anApp = XCAFApp_Application::GetApplication();
        anApp->NewDocument("MDTV-XCAF", aDoc);

        // Create label and add our m_global_dims_
        Handle(XCAFDoc_ShapeTool) myShapeTool = XCAFDoc_DocumentTool::ShapeTool(aDoc->Main());


        TDF_Label aLabel0 = myShapeTool->NewShape();
        Handle(TDataStd_Name) NameAttrib1 = new TDataStd_Name();
        NameAttrib1->Set("boundary");
        aLabel0.AddAttribute(NameAttrib1);
        myShapeTool->SetShape(aLabel0, myBoundary);
        STEPCAFControl_Writer().Perform(aDoc, (filename + "_boundary.stp").c_str());

    }
    {
        BRepBuilderAPI_MakePolygon polygonMaker;
        for (size_type i = 0, ie = geqdsk.limiter().data().size(); i < ie; ++i)
        {
            polygonMaker.Add(gp_Pnt(geqdsk.limiter().data()[i][0], 0, geqdsk.limiter().data()[i][1]));
        }
        gp_Ax1 axis(gp_Pnt(0, 0, 0), gp_Dir(0, 0, 1));
        TopoDS_Face myLimterFaceProfile = BRepBuilderAPI_MakeFace(polygonMaker.Wire());
        TopoDS_Shape myLimiter = BRepPrimAPI_MakeRevol(myLimterFaceProfile, axis);


        Handle(TDocStd_Document) aDoc;
        Handle(XCAFApp_Application) anApp = XCAFApp_Application::GetApplication();
        anApp->NewDocument("MDTV-XCAF", aDoc);

        Handle(XCAFDoc_ShapeTool) myShapeTool = XCAFDoc_DocumentTool::ShapeTool(aDoc->Main());

        TDF_Label aLabel1 = myShapeTool->NewShape();
        Handle(TDataStd_Name) NameAttrib1 = new TDataStd_Name();
        NameAttrib1->Set("limiter");
        aLabel1.AddAttribute(NameAttrib1);
        myShapeTool->SetShape(aLabel1, myLimiter);
        STEPCAFControl_Writer().Perform(aDoc, (filename + "_limiter.stp").c_str());

    }

}

}//namespace simpla
