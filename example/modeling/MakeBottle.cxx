#include <QApplication>
#include <QWidget>

#include <BRep_Tool.hxx>

#include <BRepAlgoAPI_Fuse.hxx>

#include <BRepBuilderAPI_MakeEdge.hxx>
#include <BRepBuilderAPI_MakeFace.hxx>
#include <BRepBuilderAPI_MakeWire.hxx>
#include <BRepBuilderAPI_Transform.hxx>

#include <BRepFilletAPI_MakeFillet.hxx>

#include <BRepLib.hxx>

#include <BRepOffsetAPI_MakeThickSolid.hxx>
#include <BRepOffsetAPI_ThruSections.hxx>

#include <BRepPrimAPI_MakeCylinder.hxx>
#include <BRepPrimAPI_MakePrism.hxx>

#include <GC_MakeArcOfCircle.hxx>
#include <GC_MakeSegment.hxx>

#include <GCE2d_MakeSegment.hxx>

#include <gp.hxx>
#include <gp_Ax1.hxx>
#include <gp_Ax2.hxx>
#include <gp_Ax2d.hxx>
#include <gp_Dir.hxx>
#include <gp_Dir2d.hxx>
#include <gp_Pnt.hxx>
#include <gp_Pnt2d.hxx>
#include <gp_Trsf.hxx>
#include <gp_Vec.hxx>

#include <Geom_CylindricalSurface.hxx>
#include <Geom_Plane.hxx>
#include <Geom_Surface.hxx>
#include <Geom_TrimmedCurve.hxx>

#include <Geom2d_Ellipse.hxx>
#include <Geom2d_TrimmedCurve.hxx>

#include <TopExp_Explorer.hxx>

#include <TopoDS.hxx>
#include <TopoDS_Compound.hxx>
#include <TopoDS_Edge.hxx>
#include <TopoDS_Face.hxx>
#include <TopoDS_Shape.hxx>
#include <TopoDS_Wire.hxx>

#include <TopTools_ListOfShape.hxx>

#include <BRepPrimAPI_MakeSphere.hxx>
#include <GeomAPI_Interpolate.hxx>
#include <GeomAPI_PointsToBSpline.hxx>
#include <Geom_BSplineCurve.hxx>
#include <Handle_TDocStd_Document.hxx>
#include <Handle_XCAFApp_Application.hxx>
#include <Handle_XCAFDoc_ShapeTool.hxx>
#include <STEPCAFControl_Writer.hxx>
#include <STEPCAFControl_Writer.hxx>
#include <TColgp_Array1OfPnt.hxx>
#include <TColgp_HArray1OfPnt.hxx>
#include <TDocStd_Document.hxx>
#include <TopoDS_Shape.hxx>
#include <XCAFApp_Application.hxx>
#include <XCAFApp_Application.hxx>
#include <XCAFDoc_DocumentTool.hxx>
#include <XCAFDoc_DocumentTool.hxx>
#include <XCAFDoc_ShapeTool.hxx>

#include <AIS.hxx>
#include <AIS_InteractiveContext.hxx>
#include <AIS_Shape.hxx>
#include <Graphic3d_AspectFillArea3d.hxx>
#include <Graphic3d_AspectLine3d.hxx>
#include <Graphic3d_AspectMarker3d.hxx>
#include <Graphic3d_AspectText3d.hxx>
#include <Graphic3d_GraphicDriver.hxx>
#include <OpenGl_GraphicDriver.hxx>
#include <V3d.hxx>
#include <V3d_View.hxx>
#include <V3d_Viewer.hxx>

#include <BRepPrimAPI_MakeBox.hxx>
#include <Xw_Window.hxx>

TopoDS_Shape MakeBottle(const Standard_Real myWidth, const Standard_Real myHeight, const Standard_Real myThickness) {
    BRepBuilderAPI_MakeWire wireMaker;

    // Profile : Define Support Points
    {
        Handle(TColgp_HArray1OfPnt) gp_array = new TColgp_HArray1OfPnt(1, 4);
        gp_Pnt p0(-myWidth / 2., 0, 0);
        gp_Pnt p4(myWidth / 2., 0, 0);

        //    gp_array->SetValue(0, p0);
        gp_array->SetValue(1, gp_Pnt(-myWidth / 2., -myThickness / 4., 0));
        gp_array->SetValue(2, gp_Pnt(0, -myThickness / 2., 0));
        gp_array->SetValue(3, gp_Pnt(myWidth / 2., -myThickness / 4., 0));
        gp_array->SetValue(4, p4);

        GeomAPI_Interpolate sp(gp_array, true, 1.0e-3);
        sp.Perform();

        wireMaker.Add(BRepBuilderAPI_MakeEdge(sp.Curve()));
    }
    // Body : Prism the Profile
    TopoDS_Face myFaceProfile = BRepBuilderAPI_MakeFace(wireMaker.Wire());
    gp_Vec aPrismVec(0, 0, myHeight);
    TopoDS_Shape myBody = BRepPrimAPI_MakePrism(myFaceProfile, aPrismVec);

    // Building the Resulting Compound
    TopoDS_Compound aRes;
    BRep_Builder aBuilder;
    aBuilder.MakeCompound(aRes);
    aBuilder.Add(aRes, myBody);
    //    aBuilder.Add(aRes, myThreading);

    return aRes;
}

int main(int argc, char **argv) {
    TopoDS_Shape shape = MakeBottle(50, 70, 30);
    Handle(Aspect_DisplayConnection) aDisplayConnection = new Aspect_DisplayConnection();
    Handle(OpenGl_GraphicDriver) aGraphicDriver = new OpenGl_GraphicDriver(aDisplayConnection);
    Handle(V3d_Viewer) aViewer = new V3d_Viewer(aGraphicDriver, Standard_ExtString("Test"), "");

    // Space size                400.,
    // Default projection               V3d_Xpos,
    // Default  background              Quantity_NOC_DARKVIOLET,
    // Type of  visualization            V3d_ZBUFFER,
    // Shading  geometry            V3d_GOURAUD,
    // UpdateDataOnPatch mode          V3d_WAIT
    // SetValue parameters for V3d_Viewer
    // defines default lights -
    //   positional-light 0.3 0.0 0.0
    //   directional-light V3d_XnegYposZpos
    //   directional-light V3d_XnegYneg
    //   ambient-light
    //    aViewer->SetDefaultLights();
    //// activates all the lights defined in this viewer
    //    aViewer->SetLightOn();
    //// set background color to black
    //    aViewer->SetDefaultBackgroundColor(Quantity_NOC_BLACK);

    Handle(AIS_InteractiveContext) aContext = new AIS_InteractiveContext(aViewer);
    Handle(AIS_Shape) anAis = new AIS_Shape(shape);
    aContext->Display(anAis);
    QApplication qApplication(argc, argv);
    QWidget qWidget;
    qWidget.show();
    Handle(Xw_Window) aWindow = new Xw_Window(aDisplayConnection, qWidget.winId());
    Handle(V3d_View) aView = aViewer->CreateView();
    aView->SetWindow(aWindow);
    /////////////////////////////////////////////////////////
    // CreateNew document
    Handle(TDocStd_Document) aDoc;
    Handle(XCAFApp_Application) anApp = XCAFApp_Application::GetApplication();
    anApp->NewDocument("MDTV-XCAF", aDoc);

    // CreateNew label and add our shape
    Handle(XCAFDoc_ShapeTool) myShapeTool = XCAFDoc_DocumentTool::ShapeTool(aDoc->Main());
    TDF_Label aLabel = myShapeTool->NewShape();
    myShapeTool->SetShape(aLabel, shape);

    // Write as STEP file
    STEPCAFControl_Writer *myWriter = new STEPCAFControl_Writer();
    myWriter->Perform(aDoc, "demo.stp");

    return qApplication.exec();
}