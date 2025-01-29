
Run script as <Executable> <LabelImage> <IntensityImage> <FeatureValueFile.txt> 

Output feature file has values in the following order:
-----------------------------------------------------
Label: 
Volume:           
Integrated Intensity (sum within label): 
Centroid (normalized by voxel-count): 
Weighted Centroid (normalized by sum of intensities): 
Axes Length (all axes): 
MajorAxisLength: 
MinorAxisLength: 
Eccentricity:
Elongation (Fraction of Major and Minor Axes):           
Orientation (in radians): 
Bounding box: [xmin xmax ymin ymax zmin zmax]:
PrincipalMoments (Eigen values):
Perimeter (surface for 3D):
Roundness:
EquivalentSphericalRadius:
EquivalentSphericalPerimeter (surface):
EquivalentEllipsoidDiameter:
Flatness (ratio of first and second moments):
Elongation Shape Factor (ratio two second moments):
NumberOfPixelsOnBorder (only if label touches border):
PerimeterOnBorder (only if label touches border):
PerimeterOnBorderRatio (only if label touches border):
Tumor Compactness (Volume/Surface Area):
------------------------------------------------------
