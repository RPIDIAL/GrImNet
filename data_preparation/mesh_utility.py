import numpy as np
import SimpleITK as sitk
import vtk
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk, numpy_to_vtkIdTypeArray
import os, sys
import torch
sys.path.append('{}/../data_preparation'.format(os.path.dirname(os.path.realpath(__file__))))
from jet_color_table import jet_colormap
    
device_id = 0

def generate_distance_matrix_torch(points):
    point_num = points.shape[0]
    point_tensor = torch.tensor(points).to(device=torch.device('cuda', device_id)).float()
    dist_mat = torch.zeros(point_num, point_num).to(device=torch.device('cuda', device_id)).float()
    dist_vectors = torch.zeros(point_num, point_num, 3).to(device=torch.device('cuda', device_id)).float()
    for i in range(3):
        a = torch.tile(point_tensor[:,i].view(point_num, 1), (1, point_num))
        b = torch.tile(point_tensor[:,i].view(1, point_num), (point_num, 1))

        dist_vectors[:,:,i] = a - b

        dist_mat += dist_vectors[:,:,i]**2

    dist_mat = torch.sqrt(dist_mat)

    for i in range(3):
        dist_vectors[:,:,i] = dist_vectors[:,:,i] / dist_mat

    output_dist_mat = dist_mat.detach().cpu().numpy()
    output_dist_vectors = dist_vectors.detach().cpu().numpy()

    del dist_mat, dist_vectors

    return output_dist_mat, output_dist_vectors

def generate_distance_matrix(points):
    point_num = points.shape[0]
    dist_mat = np.zeros((point_num, point_num), dtype=float)
    dist_vectors = np.zeros((point_num, point_num, 3), dtype=float)
    for i in range(3):
        a = np.reshape(points[:,i], (point_num, 1)).copy()
        b = np.reshape(points[:,i], (1, point_num)).copy()

        a = np.tile(a, (1, point_num))
        b = np.tile(b, (point_num, 1))

        dist_vectors[:,:,i] = a - b

        dist_mat += dist_vectors[:,:,i]**2

    dist_mat = np.sqrt(dist_mat)

    for i in range(3):
        dist_vectors[:,:,i] = dist_vectors[:,:,i] / dist_mat

    return dist_mat, dist_vectors

def resample_image(image, size, spacing, origin, interpolation = 'linear'):
    resampler = sitk.ResampleImageFilter()
    resampler.SetSize((int(size[0]), int(size[1]), int(size[2])))
    resampler.SetOutputSpacing((float(spacing[0]), float(spacing[1]), float(spacing[2])))
    resampler.SetOutputOrigin((float(origin[0]), float(origin[1]), float(origin[2])))
    resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
    if interpolation == 'linear':
        resampler.SetInterpolator(sitk.sitkLinear)
    else:
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    resampler.SetDefaultPixelValue(float(sitk.GetArrayFromImage(image).min()))
    resampled_image = resampler.Execute(image)
    return resampled_image

def get_mandible_landmark_list():
    valid_lmk_list = []
    with open('{}/mandible_lmk_list.txt'.format(os.path.dirname(os.path.realpath(__file__))), 'r') as fp:
        for line in fp:
            lmk_name = line[:-1]
            valid_lmk_list.append(lmk_name)
    return valid_lmk_list

def get_midface_landmark_list():
    valid_lmk_list = []
    with open('{}/midface_lmk_list.txt'.format(os.path.dirname(os.path.realpath(__file__))), 'r') as fp:
        for line in fp:
            lmk_name = line[:-1]
            valid_lmk_list.append(lmk_name)
    return valid_lmk_list

def get_lmk_id_by_name(lmk_names, lmk_list):
    lmk_ids = []
    for lmk_name in lmk_names:
        for lmk_id, lmk_name2 in enumerate(lmk_list):
            if lmk_name == lmk_name2:
                lmk_ids.append(lmk_id)
    return lmk_ids

def get_landmark_coordinate(lmk_df, lmk_name):
    lmk_pos = np.array([0.0, 0.0, 0.0])
    temp_df = lmk_df.loc[lmk_df['Landmark'] == lmk_name]
    lmk_pos[0] = temp_df['X'].to_list()[0]
    lmk_pos[1] = temp_df['Y'].to_list()[0]
    lmk_pos[2] = temp_df['Z'].to_list()[0]
    return lmk_pos

def generate_mesh_coordinate_grid(origin, size, spacing):
    x = np.linspace(origin[0], origin[0]+size[0]*spacing[0], size[0], endpoint=False)
    y = np.linspace(origin[1], origin[1]+size[1]*spacing[1], size[1], endpoint=False)
    z = np.linspace(origin[2], origin[2]+size[2]*spacing[2], size[2], endpoint=False)
    zv, yv, xv = np.meshgrid(z, y, x, indexing='ij')
    xv = xv.reshape(size[2], size[1], size[0], 1)
    yv = yv.reshape(size[2], size[1], size[0], 1)
    zv = zv.reshape(size[2], size[1], size[0], 1)
    coord_grid = np.concatenate((xv, yv, zv), axis=3)
    return coord_grid

# this function fills the holes in the original segmentation mask
## input param: 
#### label_fn: filename of original segmentation mask file (*.nii.gz)
#### output_fn: filename of solidified segmentation mask file (*.nii.gz) 
def generate_solid_label(label_fn, output_fn):
    src_label = sitk.ReadImage(label_fn)
    label_array = sitk.GetArrayFromImage(src_label)
    output_array = np.zeros_like(label_array)
    for c in [1,2]:
        temp_array = np.zeros_like(label_array)
        temp_array[label_array == c] = 1
        #print(temp_array.sum())
        label = sitk.GetImageFromArray(temp_array)
        label.CopyInformation(src_label)
        filter1 = sitk.BinaryMorphologicalClosingImageFilter()
        filter1.SetKernelRadius(5)
        label = filter1.Execute(label)
        filter = sitk.BinaryFillholeImageFilter()
        label = filter.Execute(label)
        temp_array = sitk.GetArrayFromImage(label)
        #print(temp_array.sum())
        output_array[temp_array > 0] = c
    output_label = sitk.GetImageFromArray(output_array)
    output_label.CopyInformation(src_label)
    sitk.WriteImage(output_label, output_fn)

def flip_solid_label(label_fn, output_fn):
    src_label = sitk.ReadImage(label_fn)
    label_array = sitk.GetArrayFromImage(src_label)
    output_array = np.flip(label_array, axis=2)
    output_label = sitk.GetImageFromArray(output_array)
    output_label.CopyInformation(src_label)
    sitk.WriteImage(output_label, output_fn)

def read_nifti_image(image_fn):
    image_reader = vtk.vtkNIFTIImageReader()
    image_reader.SetFileName(image_fn)
    image_reader.Update()
    image = image_reader.GetOutput()
    return image

def extract_roi_polydata(label_image, roi_label, kept_point_rate=0.1, max_point_num=43000, preserve_topo=True):
    # thresholding for volume extraction
    threshold = vtk.vtkImageThreshold()
    threshold.SetInputData(label_image)
    threshold.ThresholdBetween(roi_label, roi_label)
    threshold.SetInValue(1.0)
    threshold.SetOutValue(0.0)

    # marching cube algorithm for mesh generation
    iso=vtk.vtkMarchingCubes()
    iso.SetInputConnection(threshold.GetOutputPort())
    iso.SetValue(0, 0.5)
    iso.ComputeGradientsOn()
    iso.ComputeNormalsOff()
    iso.ComputeScalarsOff()
    
    # mesh smoothing
    smooth = vtk.vtkWindowedSincPolyDataFilter()
    smooth.SetInputConnection(iso.GetOutputPort())
    smooth.SetNumberOfIterations(20)
    smooth.BoundarySmoothingOff()
    smooth.FeatureEdgeSmoothingOff()
    smooth.SetFeatureAngle(120)
    smooth.SetPassBand(0.01)
    smooth.NonManifoldSmoothingOn()
    smooth.NormalizeCoordinatesOn()
    smooth.Update()
    smooth_polydata = smooth.GetOutput()
    p_num = smooth_polydata.GetNumberOfPoints()
    print('num of points before downsampling:', p_num)
    

    # mesh downsampling
    decimate = vtk.vtkDecimatePro()
    decimate.SetInputData(smooth_polydata)
    #decimate.SetTargetReduction(1.0 - 1e-6) # this is a parameter in range of [0.0, 1.0] controling the dropping rate of the downsampling, we assign a high rate (nearly 1.0) to make the filter downsample the mesh to a minimum number of points preserving topology
    if p_num * kept_point_rate > max_point_num:
    #if p_num * kept_point_rate > 60000:
        decimate.SetTargetReduction(1.0 - max_point_num / p_num)
    else:
        decimate.SetTargetReduction(1.0 - kept_point_rate)
    if preserve_topo:
        decimate.PreserveTopologyOn()
    else:
        decimate.PreserveTopologyOff()
    decimate.Update()

    polydata = decimate.GetOutput()
    print('num of points after downsampling:', polydata.GetNumberOfPoints())
    return polydata

# this function extracts a mesh from a given segmentation mask file ("label_fn") with specified label ("roi_label")
## input param:
#### label_fn: filename of segmentation mask file (*.nii.gz)
#### roi_label: label of target volume in the segmentation mask
#### color: display color of the extract mesh (tuple like (R, G, B), e.g., (1.0, 0.0, 0.0))
#### opacity: display opacity of the extract mesh (range [0.0, 1.0])
#### represent: display mode of the the extract mesh ("surface", "wire", or "point")
def extract_roi(label_fn, roi_label, color, opacity=1, represent='surface', downsample_rate=0.2, max_sample=43000, preserve_topo=True):
    label_image = read_nifti_image(label_fn)
    roi_polydata = extract_roi_polydata(label_image, roi_label, kept_point_rate=downsample_rate, max_point_num=max_sample, preserve_topo=preserve_topo)

    # calculate the center point of the extracted mesh
    # this step is irrelevant to the mesh generation 
    # we just need a center point to locate the camera focus for displaying purpose
    cm = vtk.vtkCenterOfMass()
    cm.SetInputData(roi_polydata)
    cm.SetUseScalarsAsWeights(False)
    cm.Update()
    roi_center = cm.GetCenter()
    
    # get the polydata instance of the extracted mesh
    # all vertex and egde information are stored in the polydata instance
    # so we need this polydata to calculate the adjacent matrix in the following process

    # generate polydata mapper (a specific data structure required by vtk to display 3D objects)
    roi_mapper = vtk.vtkPolyDataMapper()
    roi_mapper.SetInputData(roi_polydata)

    # generate actor (a specific data structure required by vtk to display 3D objects)
    roi_actor = vtk.vtkActor()
    roi_actor.SetMapper(roi_mapper)
    roi_actor.GetProperty().SetColor(color[0],color[1],color[2]) # (R, G, B) color
    roi_actor.GetProperty().SetOpacity(opacity)
    if represent == 'point':
        roi_actor.GetProperty().SetRepresentationToPoints()
    elif represent == 'wire':
        roi_actor.GetProperty().SetRepresentationToWireframe()
    elif represent == 'surface':
        roi_actor.GetProperty().SetRepresentationToSurface()
    roi_actor.GetProperty().SetLineWidth(1)

    return roi_actor, roi_center, roi_polydata

# this function generates point list and adjacent matrix given a polydata
## input param:
#### polydata: target polydata
## output param:
#### points: a list of point coordinates in a shape of (N, 3). N is the number of mesh points
#### adjacent_matrix: adjacent matrix in a shape of (N, N)
def generate_adjacent_matrix(polydata, curvature_types = ['gaussian', 'mean', 'min', 'max']):
    point_num = polydata.GetNumberOfPoints()
    cell_num = polydata.GetNumberOfCells()
    points = vtk_to_numpy(polydata.GetPoints().GetData())
    cells = vtk_to_numpy(polydata.GetPolys().GetData()).reshape((cell_num, 4))

    #curvatures = vtk.vtkCurvatures()
    #curvatures.SetInputData(polydata)
    #curvatures.SetCurvatureTypeToGaussian()
    #curvatures.Update()
    #gauss_curv_arr = vtk_to_numpy(curvatures.GetOutput().GetPointData().GetScalars()).reshape((-1, 1))

    #curvatures = vtk.vtkCurvatures()
    #curvatures.SetInputData(polydata)
    #curvatures.SetCurvatureTypeToMean()
    #curvatures.Update()
    #mean_curv_arr = vtk_to_numpy(curvatures.GetOutput().GetPointData().GetScalars()).reshape((-1, 1))

    #curvatures = vtk.vtkCurvatures()
    #curvatures.SetInputData(polydata)
    #curvatures.SetCurvatureTypeToMinimum()
    #curvatures.Update()
    #min_curv_arr = vtk_to_numpy(curvatures.GetOutput().GetPointData().GetScalars()).reshape((-1, 1))

    #curvatures = vtk.vtkCurvatures()
    #curvatures.SetInputData(polydata)
    #curvatures.SetCurvatureTypeToMaximum()
    #curvatures.Update()
    #max_curv_arr = vtk_to_numpy(curvatures.GetOutput().GetPointData().GetScalars()).reshape((-1, 1))

    #curv_arr = np.concatenate((gauss_curv_arr, mean_curv_arr, min_curv_arr, max_curv_arr), axis=1)
    #curv_arr = mean_curv_arr

    normals = vtk.vtkTriangleMeshPointNormals()
    normals.SetInputData(polydata)
    normals.Update()
    norm_arr = vtk_to_numpy(normals.GetOutput().GetPointData().GetNormals())

    adjacent_matrix = np.zeros((point_num, point_num), dtype=bool)
    rows = cells[:,1]
    cols = cells[:,2]
    adjacent_matrix[rows, cols] = True
    rows = cells[:,1]
    cols = cells[:,3]
    adjacent_matrix[rows, cols] = True
    rows = cells[:,2]
    cols = cells[:,3]
    adjacent_matrix[rows, cols] = True
    adjacent_matrix |= adjacent_matrix.transpose()
    diag = np.diag(np.ones(point_num, dtype=adjacent_matrix.dtype))
    adjacent_matrix |= diag
    #return points, cells, adjacent_matrix, curv_arr, norm_arr
    return points, cells, adjacent_matrix, norm_arr

def calculate_multi_step_adjacent_matrix(adj_mat, step_num):
    #os.environ['CUDA_VISIBLE_DEVICES'] = "1"

    #adj_mat = torch.from_numpy(np.array([[0,1,0,0,0],[1,0,1,0,1],[0,1,0,1,1],[0,0,1,0,0],[0,1,1,0,0]])) # for debug only
    single_step_adj_mat = torch.from_numpy(adj_mat).to(device=torch.device('cuda', device_id)).float()
    dist_mat = torch.zeros_like(single_step_adj_mat)
    #mandible_step_mat = torch.ones_like(mandible_mat) * point_num - torch.diag(torch.ones(point_num, dtype=int)) * (point_num - 1)
    #mandible_step_mat = (torch.ones_like(mandible_mat) - torch.diag(torch.ones(point_num, dtype=int))) * point_num
    #mandible_mat = mandible_mat.cuda().float()
    #mandible_step_mat = mandible_step_mat.cuda().float()
    multi_step_adj_mat = single_step_adj_mat.detach()
    for i in range(step_num):
        if i > 0:
            multi_step_adj_mat = torch.matmul(multi_step_adj_mat, single_step_adj_mat)
        dist_mat[(multi_step_adj_mat > 0) & (dist_mat == 0)] = i + 1
        #tmp_mat = (step_mat > 0)*(i+2)
        #tmp_mat[tmp_mat == 0] = point_num
        #mandible_step_mat = torch.minimum(mandible_step_mat, tmp_mat)
    #mandible_step_mat = 1.0/mandible_step_mat
    #mandible_step_mat[mandible_step_mat == point_num] = 0
    #mandible_step_mat = torch.exp(-mandible_step_mat)
    #mandible_step_mat[mandible_step_mat <= 1/point_num] = 0
    #print(mandible_step_mat)
    output = dist_mat.detach().cpu().numpy().astype(np.uint8)
    del single_step_adj_mat, multi_step_adj_mat, dist_mat
    return output

def calculate_curvature(adj_mat, norm):
    #os.environ['CUDA_VISIBLE_DEVICES'] = "1"

    #adj_mat = torch.from_numpy(np.array([[0,1,0,0,0],[1,0,1,0,1],[0,1,0,1,1],[0,0,1,0,0],[0,1,1,0,0]])) # for debug only
    single_step_adj_mat = torch.from_numpy(adj_mat).to(device=torch.device('cuda', device_id)).float()
    unit_norm = torch.from_numpy(norm).to(device=torch.device('cuda', device_id)).float()
    cos_sim = torch.matmul(unit_norm, unit_norm.transpose(0,1))
    single_step_adj_mat[single_step_adj_mat != 0] = 1
    N = single_step_adj_mat.sum(dim=0)
    cos_sim = cos_sim * single_step_adj_mat
    E = cos_sim.sum(dim=0)
    E2 = (cos_sim**2).sum(dim=0)
    E = E / N
    E2 = E2 / N
    V = torch.sqrt(torch.clamp(E2 - E**2, min=0))
    V = V.detach().cpu().numpy().reshape(-1, 1)
    cos_sim = cos_sim.detach().cpu().numpy()
    return V, E, N, cos_sim

def calculate_rel_features(adj_mat, dist, norm):
    #os.environ['CUDA_VISIBLE_DEVICES'] = "1"

    #adj_mat = torch.from_numpy(np.array([[0,1,0,0,0],[1,0,1,0,1],[0,1,0,1,1],[0,0,1,0,0],[0,1,1,0,0]])) # for debug only
    edge_adj_mat = torch.from_numpy(adj_mat).to(device=torch.device('cuda', device_id)).float()
    unit_norm = torch.from_numpy(norm).to(device=torch.device('cuda', device_id)).float()
    edge_lent = torch.from_numpy(dist).to(device=torch.device('cuda', device_id)).float()
    cos_sim = torch.matmul(unit_norm, unit_norm.transpose(0,1))
    edge_adj_mat[edge_adj_mat != 0] = 1
    N = edge_adj_mat.sum(dim=0)
    cos_sim = cos_sim * edge_adj_mat
    edge_lent = edge_lent * edge_adj_mat
    E_norm = cos_sim.sum(dim=0) / N
    E2_norm = (cos_sim**2).sum(dim=0) / N
    V_norm = torch.sqrt(torch.clamp(E2_norm - E_norm**2, min=0))
    E_norm = E_norm.detach().cpu().numpy().reshape(-1, 1)
    V_norm = V_norm.detach().cpu().numpy().reshape(-1, 1)
    E_edge = edge_lent.sum(dim=0) / N
    E2_edge = (edge_lent**2).sum(dim=0) / N
    V_edge = torch.sqrt(torch.clamp(E2_edge - E_edge**2, min=0))
    E_edge = E_edge.detach().cpu().numpy().reshape(-1, 1)
    V_edge = V_edge.detach().cpu().numpy().reshape(-1, 1)
    N = N.detach().cpu().numpy().reshape(-1, 1)
    #cos_sim = cos_sim.detach().cpu().numpy()
    return np.concatenate((E_norm, V_norm, E_edge, V_edge, N), axis=1)

# this function generates actor of a given landmark for 3D displaying purpose
## input param:
#### pos: coordinates of landmark (x, y, z)
#### color: display color of the extract mesh (tuple like (R, G, B), e.g., (1.0, 0.0, 0.0))
#### label: name of landmark
def create_point_actor(pos, color, label, radius):
    source = vtk.vtkSphereSource()
    source.SetCenter(pos[0], pos[1], pos[2])
    source.SetRadius(radius)

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(source.GetOutputPort())
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(color[0],color[1],color[2])

    text_actor = vtk.vtkBillboardTextActor3D()
    text_actor.SetInput(label)
    text_actor.SetPosition(pos[0], pos[1], pos[2])
    text_actor.SetDisplayOffset(0, 5)
    text_actor.SetForceOpaque(True)
    text_actor.GetTextProperty().SetFontSize(12)
    text_actor.GetTextProperty().SetBold(True)
    text_actor.GetTextProperty().SetColor(color[0],color[1],color[2])
    text_actor.GetTextProperty().SetJustificationToCentered()
    return actor, text_actor

def create_line_actor(pos1, pos2, color):
    source = vtk.vtkLineSource()
    source.SetPoint1(pos1[0], pos1[1], pos1[2])
    source.SetPoint2(pos2[0], pos2[1], pos2[2])
    source.SetResolution(1)
    source.Update()

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(source.GetOutputPort())
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(color[0],color[1],color[2])

    return actor

def create_cube_actor(bounds, color):
    source = vtk.vtkCubeSource()
    source.SetBounds(bounds)
    source.Update()

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(source.GetOutputPort())
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(color[0],color[1],color[2])
    actor.GetProperty().SetOpacity(0.2)

    return actor

# this function finds a closest point in a polydata to a specified point "p"
## input param:
#### polydata: target polydata
#### p: coordinates of the point (x, y, z)
## output param:
#### c: coordinates of the closest point (x, y, z)
#### int(ptIds): point index of the closest point in the input polydata
#### dist: distance from the closest point to the input point
def find_closest_point(polydata, p):    
    point_locator = vtk.vtkPointLocator()
    point_locator.SetDataSet(polydata)  # reverse.GetOutput() --> vtkPolyData
    point_locator.BuildLocator()
    ptIds = point_locator.FindClosestPoint(p)
    coord = polydata.GetPoint(ptIds)
    c = np.array(coord)
    dist = np.sqrt(np.sum((p-c)**2))
    
    return c, int(ptIds), dist

def generate_roi_from_points(pts_arr, cell_arr, color, opacity=1, represent='surface'):
    
    roi_polydata = vtk.vtkPolyData()
    roi_pts = vtk.vtkPoints()
    roi_pts.SetData(numpy_to_vtk(pts_arr))
    roi_cells = vtk.vtkCellArray()
    roi_cells.SetNumberOfCells(cell_arr.shape[0])
    roi_cells.SetCells(cell_arr.shape[0], numpy_to_vtkIdTypeArray(cell_arr))
    roi_polydata.SetPoints(roi_pts)
    roi_polydata.SetPolys(roi_cells)

    cm = vtk.vtkCenterOfMass()
    cm.SetInputData(roi_polydata)
    cm.SetUseScalarsAsWeights(False)
    cm.Update()
    roi_center = cm.GetCenter()

    # generate polydata mapper (a specific data structure required by vtk to display 3D objects)
    roi_mapper = vtk.vtkPolyDataMapper()
    roi_mapper.SetInputData(roi_polydata)

    # generate actor (a specific data structure required by vtk to display 3D objects)
    roi_actor = vtk.vtkActor()
    roi_actor.SetMapper(roi_mapper)
    roi_actor.GetProperty().SetColor(color[0],color[1],color[2]) # (R, G, B) color
    roi_actor.GetProperty().SetOpacity(opacity)
    if represent == 'point':
        roi_actor.GetProperty().SetRepresentationToPoints()
    elif represent == 'wire':
        roi_actor.GetProperty().SetRepresentationToWireframe()
    elif represent == 'surface':
        roi_actor.GetProperty().SetRepresentationToSurface()
    #roi_actor.GetProperty().SetLineWidth(1)
    roi_actor.GetProperty().SetLineWidth(0)

    return roi_actor, roi_center, roi_polydata

def generate_distance_map_actor_to_point(polydata, dist_map):
    scalar_data = numpy_to_vtk(num_array=dist_map, deep=True, array_type=vtk.VTK_FLOAT)
    polydata.GetPointData().SetScalars(scalar_data)

    lut = vtk.vtkLookupTable()
    jet = jet_colormap()
    lut.SetNumberOfColors(len(jet))
    for i in range(len(jet)):
        lut.SetTableValue(i, jet[i][0], jet[i][1], jet[i][2], 1.0)

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(polydata)
    mapper.SetLookupTable(lut)
    mapper.SetScalarRange(0, 1)
    mapper.Update()

    dist_actor = vtk.vtkActor()
    dist_actor.SetMapper(mapper)
    #dist_actor.GetProperty().SetColor(color[0],color[1],color[2])
    dist_actor.GetProperty().SetOpacity(1.0)
    #dist_actor.GetProperty().SetRepresentationToPoints()
    #dist_actor.GetProperty().SetRepresentationToWireframe()
    dist_actor.GetProperty().SetRepresentationToSurface()
    #dist_actor.GetProperty().SetLineWidth(0.1)
    dist_actor.GetProperty().SetLineWidth(0)

    return dist_actor
