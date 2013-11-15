#include <Xdmf.h>
#include <iostream>
int main(int argc, char ** argv)
{

	XdmfDOM DOM;
	DOM.SetInputFileName(argv[1]);
	DOM.Parse();

	std::cout << DOM.GetNumberOfChildren() << std::endl;
	std::cout << DOM.Serialize() << std::endl;
	XdmfGrid Grid;
//
//	XdmfAttribute *XPos;
	XdmfXmlNode GridNode = DOM.FindElementByPath("/Xdmf/Domain/Grid[@Name=\"Test\"]");

	Grid.SetDOM(&DOM);
	Grid.SetElement(GridNode);
	Grid.UpdateInformation();
	std::cout << "First Grid has " << Grid.GetNumberOfAttributes()
			<< " Attributes" << std::endl;

	auto geo = Grid.GetGeometry();

	std::cout << geo->GetGeometryTypeAsString() << std::endl;
}
