
/*Code for extracting 3D shape features using LabelGeometry andn LabelShape ITK filters with customized implementation for EllipsoidDiameter and Flatness parameters*/
/*Code written by Jhimli Mitra*/

#include "itkImage.h"
#include "itkImageFileWriter.h"
#include "itkImageRegionIterator.h"
#include "itkImageFileReader.h"
#include "itkLabelGeometryImageFilter.h"
//#include "itkLabelToRGBImageFilter.h"
#include "itkLabelImageToShapeLabelMapFilter.h"
#include <itkVector.h>
#include "itkMath.h"
#include <sstream>
#include <fstream>

typedef itk::Image<unsigned int, 3>  ImageType;
//typedef itk::RGBPixel<unsigned char> RGBPixelType;
//typedef itk::Image<RGBPixelType, 3>  RGBImageType;
typedef unsigned short LabelType;
typedef itk::ShapeLabelObject< LabelType, 3 > ShapeLabelObjectType;
typedef itk::LabelMap< ShapeLabelObjectType > LabelMapType;
typedef itk::Vector<double, 3> VectorType;
unsigned int ImageDimension=3;
int main(int argc, char *argv[])
{

  if ( argc < 4)
  {
    std::cout<<"Usage: <ProgramName=ShapeFeatures3D> <LabelImage> <IntensityImage> <OutputFeatureFile>"<<std::endl;
  } 
  ImageType::Pointer labelImage = ImageType::New();
  ImageType::Pointer intensityImage = ImageType::New();
  //int label = 1;
    //Read Label Image
    typedef itk::ImageFileReader< ImageType  > ImageReaderType;
    ImageReaderType::Pointer labelReader =
      ImageReaderType::New();
    labelReader->SetFileName(argv[1]);
    labelReader->Update();

    labelImage = labelReader->GetOutput();
    //Read Intensity Image
    ImageReaderType::Pointer intensityReader =
      ImageReaderType::New();
    intensityReader->SetFileName(argv[2]);
    intensityReader->Update();

    intensityImage = intensityReader->GetOutput();

      
  // NOTE: As of April 8, 2015 the filter does not work with non-zero
  // origins
  double origin[3] = {0.0, 0.0, 0.0};
  labelImage->SetOrigin(origin);
  intensityImage->SetOrigin(origin);

  typedef itk::LabelGeometryImageFilter< ImageType > LabelGeometryImageFilterType;
  LabelGeometryImageFilterType::Pointer labelGeometryImageFilter = LabelGeometryImageFilterType::New();
  labelGeometryImageFilter->SetInput( labelImage );
  labelGeometryImageFilter->SetIntensityInput( intensityImage );

  // These generate optional outputs.
  labelGeometryImageFilter->CalculatePixelIndicesOn();
  labelGeometryImageFilter->CalculateOrientedBoundingBoxOn();
  labelGeometryImageFilter->CalculateOrientedLabelRegionsOn();
  labelGeometryImageFilter->CalculateOrientedIntensityRegionsOn();
  
  labelGeometryImageFilter->Update();
  LabelGeometryImageFilterType::LabelsType allLabels = labelGeometryImageFilter->GetLabels();
  
  /*typedef itk::LabelToRGBImageFilter<ImageType, RGBImageType> RGBFilterType;
  RGBFilterType::Pointer rgbLabelImage =
    RGBFilterType::New();
  rgbLabelImage->SetInput(labelImage);

  typedef itk::LabelToRGBImageFilter<ImageType, RGBImageType> RGBFilterType;
  RGBFilterType::Pointer rgbOrientedImage =
    RGBFilterType::New();
  rgbOrientedImage->SetInput(labelGeometryImageFilter->GetOrientedLabelImage(allLabels[label]));*/


//Features using LabelShape filter
typedef itk::LabelImageToShapeLabelMapFilter< ImageType, LabelMapType> I2LType;
I2LType::Pointer i2l = I2LType::New();
i2l->SetInput( labelImage);
i2l->SetComputePerimeter(true);
i2l->Update();
LabelMapType *labelMap = i2l->GetOutput();
VectorType principalMoments;
VectorType ellipsoidDiameter;
double edet = 1.0;
double equivalentRadius;

double elongation = 0;
double flatness = 0;
double compactness=0;
  std::ofstream   fixedFile;
  fixedFile.open( argv[3],std::ios::out);
  
  LabelGeometryImageFilterType::LabelsType::iterator allLabelsIt;
  std::cout << "Number of labels: " << labelGeometryImageFilter->GetNumberOfLabels()-1 << std::endl;
  std::cout << std::endl;
//Calculate Label Geometry features for all labels except background
  for( allLabelsIt = allLabels.begin(); allLabelsIt != allLabels.end(); allLabelsIt++ )
 {
    LabelGeometryImageFilterType::LabelPixelType labelValue = *allLabelsIt;
    if (labelValue != 0 )
    {
    
    std::cout << "\tLabel: "
              << (int)labelValue << std::endl;
    std::cout << "\tVolume: "
              << labelGeometryImageFilter->GetVolume(labelValue) << std::endl;
    std::cout << "\tIntegrated Intensity (sum within label): "
              << labelGeometryImageFilter->GetIntegratedIntensity(labelValue) << std::endl;
    std::cout << "\tCentroid (normalized by voxel-count): "
              << labelGeometryImageFilter->GetCentroid(labelValue) << std::endl;
    std::cout << "\tWeighted Centroid (normalized by sum of intensities): "
              << labelGeometryImageFilter->GetWeightedCentroid(labelValue) << std::endl;
    std::cout << "\tAxes Length (all axes): "
              << labelGeometryImageFilter->GetAxesLength(labelValue) << std::endl;
    std::cout << "\tMajorAxisLength: "
              << labelGeometryImageFilter->GetMajorAxisLength(labelValue) << std::endl;
    std::cout << "\tMinorAxisLength: "
              << labelGeometryImageFilter->GetMinorAxisLength(labelValue) << std::endl;
    std::cout << "\tEccentricity: "
              << labelGeometryImageFilter->GetEccentricity(labelValue) << std::endl;
     std::cout << "\tElongation: "
              << labelGeometryImageFilter->GetElongation(labelValue) << std::endl;//Fraction of Major and Minor Axes
    std::cout << "\tOrientation (in radians): "
              << labelGeometryImageFilter->GetOrientation(labelValue) << std::endl;
    std::cout << "\tBounding box: [xmin xmax ymin ymax zmin zmax]:"
              << labelGeometryImageFilter->GetBoundingBox(labelValue) << std::endl;


    ShapeLabelObjectType *labelObject = labelMap->GetNthLabelObject(labelValue-1);
    std::cout << "\tNumberOfPixelsOnBorder (only if label touches border): "
              << labelObject->GetNumberOfPixelsOnBorder() << std::endl;
    std::cout << "\tPerimeterOnBorder (only if label touches border): "
              << labelObject->GetPerimeterOnBorder() << std::endl;
    std::cout << "\tPerimeterOnBorderRatio (only if label touches border): "
              << labelObject->GetPerimeterOnBorderRatio() << std::endl;
    //std::cout << "    FeretDiameter: "
             // << labelObject->GetFeretDiameter() << std::endl;
    std::cout << "\tPrincipalMoments (Eigen values): "
              << labelObject->GetPrincipalMoments() << std::endl;
   principalMoments =labelObject->GetPrincipalMoments();
    std::cout << "\tPrincipalAxes (Eigen-vectors/Matrix): "
              << labelObject->GetPrincipalAxes() << std::endl;
    std::cout << "\tPerimeter (surface for 3D): "
              << labelObject->GetPerimeter() << std::endl;
    std::cout << "\tRoundness: "
              << labelObject->GetRoundness() << std::endl;
    std::cout << "\tEquivalentSphericalRadius: "
              << labelObject->GetEquivalentSphericalRadius() << std::endl;
   equivalentRadius=labelObject->GetEquivalentSphericalRadius();
    std::cout << "\tEquivalentSphericalPerimeter (surface): "
              << labelObject->GetEquivalentSphericalPerimeter() << std::endl;
  //custom implementation to prevent NaN generated by built-in ITK function
    for (unsigned int i=0;i<ImageDimension;i++)
    {
          edet *= std::abs(principalMoments[i]);
    }
  	edet = std::pow(edet, 1.0 / ImageDimension);
       
  	for ( unsigned int i = 0; i < ImageDimension; i++ )
    	{
    		if ( edet != 0.0 )
      		{
      			ellipsoidDiameter[i] = 2.0*equivalentRadius*std::sqrt(std::abs(principalMoments[i]) / edet);
      		}
    		else
      		{
      			ellipsoidDiameter[i] = 0;
      		}
	}

    std::cout << "\tEquivalentEllipsoidDiameter: " <<ellipsoidDiameter<<std::endl;
      //        << labelObject->GetEquivalentEllipsoidDiameter() << std::endl;
//custom implementation to prevent NaN generated by built-in ITK function
	if (itk::Math::NotAlmostEquals( principalMoments[0], itk::NumericTraits<VectorType::ValueType >::ZeroValue() ) )
	{
	        //(elongation shape factor square-root ratio of two second moments)
		elongation = std::sqrt(std::abs(principalMoments[ImageDimension - 1] )/ std::abs(principalMoments[ImageDimension - 2]));
		flatness = std::sqrt(std::abs(principalMoments[1]) / std::abs(principalMoments[0]));
	}
    std::cout << "\tFlatness (ratio of first and second moments): "<<flatness<<std::endl;
    std::cout <<"\tElongation Shape Factor (ratio two second moments) "<<elongation<<std::endl;
     //Compute Compactness as Volume/perimeter
    compactness=labelGeometryImageFilter->GetVolume(labelValue)/labelObject->GetPerimeter();
    std::cout<<"\tTumor Compactness(Volume/Surface area): "<<compactness<<std::endl;
             // << labelObject->GetFlatness() << std::endl;
 fixedFile<<(int)labelValue<<std::endl<<labelGeometryImageFilter->GetVolume(labelValue)<<std::endl<<labelGeometryImageFilter->GetIntegratedIntensity(labelValue)<<std::endl<<labelGeometryImageFilter->GetCentroid(labelValue)<<std::endl<<labelGeometryImageFilter->GetWeightedCentroid(labelValue)<<std::endl<<labelGeometryImageFilter->GetAxesLength(labelValue)<<std::endl<<labelGeometryImageFilter->GetMajorAxisLength(labelValue)<<std::endl<<labelGeometryImageFilter->GetMinorAxisLength(labelValue)<<std::endl<<labelGeometryImageFilter->GetEccentricity(labelValue)<<std::endl<<labelGeometryImageFilter->GetElongation(labelValue)<<std::endl<<labelGeometryImageFilter->GetOrientation(labelValue)<<std::endl<<labelGeometryImageFilter->GetBoundingBox(labelValue)<<std::endl<<labelObject->GetPrincipalMoments()<<std::endl<<labelObject->GetPerimeter()<<std::endl<<labelObject->GetRoundness()<<std::endl<<labelObject->GetEquivalentSphericalRadius()<<std::endl<<labelObject->GetEquivalentSphericalPerimeter()<<std::endl<<ellipsoidDiameter<<std::endl<<flatness<<std::endl<<elongation<<std::endl<<labelObject->GetNumberOfPixelsOnBorder()<<std::endl<<labelObject->GetPerimeterOnBorder()<<std::endl<<labelObject->GetPerimeterOnBorderRatio()<<std::endl<<compactness<<std::endl;
        }
  
    }
fixedFile.close();

/*
//Extract features for all labels except background
for(unsigned int n = 0; n < labelMap->GetNumberOfLabelObjects(); n++ )
{
ShapeLabelObjectType *labelObject = labelMap->GetNthLabelObject(n);
std::cout << "Label: "
              << itk::NumericTraits<LabelMapType::LabelType>::PrintType(labelObject->GetLabel()) << std::endl;
    //std::cout << "    BoundingBox: "
              //<< labelObject->GetBoundingBox() << std::endl;
    //std::cout << "    NumberOfPixels: "
              //<< labelObject->GetNumberOfPixels() << std::endl;
    //std::cout << "    PhysicalSize: "
             // << labelObject->GetPhysicalSize() << std::endl;
    //std::cout << "    Centroid: "
              //<< labelObject->GetCentroid() << std::endl;
    std::cout << "    NumberOfPixelsOnBorder (only if label touches border): "
              << labelObject->GetNumberOfPixelsOnBorder() << std::endl;
    std::cout << "    PerimeterOnBorder (only if label touches border): "
              << labelObject->GetPerimeterOnBorder() << std::endl;
    std::cout << "    PerimeterOnBorderRatio (only if label touches border): "
              << labelObject->GetPerimeterOnBorderRatio() << std::endl;
    //std::cout << "    FeretDiameter: "
             // << labelObject->GetFeretDiameter() << std::endl;
    std::cout << "    PrincipalMoments (Eigen values): "
              << labelObject->GetPrincipalMoments() << std::endl;
   principalMoments =labelObject->GetPrincipalMoments();
    std::cout << "    PrincipalAxes (Eigen-vectors/Matrix): "
              << labelObject->GetPrincipalAxes() << std::endl;
    //std::cout << "    Elongation: "
             // << labelObject->GetElongation() << std::endl;
    std::cout << "    Perimeter (no. of surface voxels for 3D): "
              << labelObject->GetPerimeter() << std::endl;
    std::cout << "    Roundness: "
              << labelObject->GetRoundness() << std::endl;
    std::cout << "    EquivalentSphericalRadius: "
              << labelObject->GetEquivalentSphericalRadius() << std::endl;
   equivalentRadius=labelObject->GetEquivalentSphericalRadius();
    std::cout << "    EquivalentSphericalPerimeter (no. of voxels for 3D): "
              << labelObject->GetEquivalentSphericalPerimeter() << std::endl;
  //custom implementation to prevent NaN generated by built-in ITK function
    for (unsigned int i=0;i<ImageDimension;i++)
	{
          edet *= std::abs(principalMoments[i]);
   	 }
  	edet = std::pow(edet, 1.0 / ImageDimension);
       
  	for ( unsigned int i = 0; i < ImageDimension; i++ )
    	{
    		if ( edet != 0.0 )
      		{
      			ellipsoidDiameter[i] = 2.0*equivalentRadius*std::sqrt(std::abs(principalMoments[i]) / edet);
      		}
    		else
      		{
      			ellipsoidDiameter[i] = 0;
      		}
	}

    std::cout << "    EquivalentEllipsoidDiameter: " <<ellipsoidDiameter<<std::endl;
      //        << labelObject->GetEquivalentEllipsoidDiameter() << std::endl;
//custom implementation to prevent NaN generated by built-in ITK function
	if (itk::Math::NotAlmostEquals( principalMoments[0], itk::NumericTraits< typename VectorType::ValueType >::ZeroValue() ) )
	{
	        //(elongation shape factor square-root ratio of two second moments)
		elongation = std::sqrt(std::abs(principalMoments[ImageDimension - 1] )/ std::abs(principalMoments[ImageDimension - 2]));
		flatness = std::sqrt(std::abs(principalMoments[1]) / std::abs(principalMoments[0]));
	}
    std::cout << "    Flatness (ratio of first and second moments): "<<flatness<<std::endl;
    std::cout <<"     Elongation Shape Factor (ratio two second moments) "<<elongation<<std::endl;
             // << labelObject->GetFlatness() << std::endl;
    
}*/
  return EXIT_SUCCESS;
}


