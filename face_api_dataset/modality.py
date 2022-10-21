from enum import Enum, auto


class Modality(Enum):
    """
    Different modalities of Synthesis AI dataset.
    All image modalities are in `[y][x][channel]` format, with axis going as follows::

        ┌-----> x
        |
        |
        v
        y
    """

    SCENE_ID = auto()
    """
    Scene ID (rendered scene number). 
    
    **Type**: `int`.
    """
    RGB = auto()
    """
    RGB image modality. 
    
    **Type**: `ndarray[uint8]`. **Channels**: `3`.
    """
    NORMALS = auto()
    """
    Normals image. All values are in [-1,1] range.
    
    **Type**: `ndarray[float16]`. **Channels**: 3.
    """
    DEPTH = auto()
    """
    Depth Image. All values are positive floats. Background has depth=0.
    
    **Type**: `ndarray[float16]`. **Channels**: 1.
    """
    ALPHA = auto()
    """
    Alpha Image. 0 - means complete transparency, 255 - solid object.
    
    **Type**: `ndarray[uint8]`. **Channels**: 1.
    """
    BODY_SEGMENTATION = auto()
    """
    Semantic segmentation map of various body parts.
    
    **Type**: `ndarray[uint16]`. **Channels**: 1.
    """
    CLOTHING_SEGMENTATION = auto()
    """
    Semantic segmentation map of different types of clothing.
    
    **Type**: `ndarray[uint16]`. **Channels**: 1.
    """
    INSTANCE_SEGMENTATION = auto()
    """
    Semantic segmentation map defining the various characters in the image.
    
    **Type**: `ndarray[uint16]`. **Channels**: 1.
    """
    LANDMARKS_IBUG68 = auto()
    """
    iBUG-68 landmarks. Each landmark is given by name and two coordinates (x,y) in pixels.
    Each keypoint is a 2D projection of a 3D landmark.
    
    **Type**: `Dict[InstanceId, Dict[LandmarkId, Landmark2d]`. Should have no more than 68 points.
    """
    LANDMARKS_CONTOUR_IBUG68 = auto()
    """
    iBUG-68 contour landmarks. Each landmark is given by two coordinates (name, x,y) in pixels.
    Each keypoint is defined in a similar manner to human labelers marking 2D face kepoints.
    
    **Type**: `Dict[InstanceId, Dict[LandmarkId, Landmark2d]]`. Should have no more than 68 points.
    """
    LANDMARKS_KINECT_V2 = auto()
    """
    Kinect v2 landmarks. Each landmark by name and two coordinates (x,y) in pixels.

    **Type**: `Dict[InstanceId, Dict[LandmarkId, Landmark2d]]`. Should have no more than 32 points.
    """
    LANDMARKS_MEDIAPIPE = auto()
    """
    MediaPipe pose landmarks. Each landmark is given by name and two coordinates (x,y) in pixels.

    **Type**: `Dict[InstanceId, Dict[LandmarkId, Landmark2d]]`. Should have no more than 33 points.
    """
    LANDMARKS_COCO = auto()
    """
    COCO whole body landmarks. Each landmark is given by name and two coordinates (x,y) in pixels.

    **Type**: `Dict[InstanceId, Dict[LandmarkId, Landmark2d]]`. Should have no more than 133 points.
    """
    LANDMARKS_MPEG4 = auto()
    """
    MPEG4 landmarks. Each landmark is given by name and two coordinates (x,y) in pixels.

    **Type**: `Dict[InstanceId, Dict[LandmarkId, Landmark2d]]`.
    """
    LANDMARKS_3D_IBUG68 = auto()
    """
    iBUG-68 landmarks in 3D. Each landmark is given by name and three coordinates (x,y,z) in camera space.

    **Type**: `Dict[InstanceId, Dict[LandmarkId, Landmark3d]]`. Should have no more than 68 points.
    """
    LANDMARKS_3D_KINECT_V2 = auto()
    """
    Kinect v2 landmarks in 3D. Each landmark is given by name and three coordinates (x,y,z) in camera space.

    **Type**: `Dict[InstanceId, Dict[LandmarkId, Landmark3d]]`. Should have no more than 32 points.
    """
    LANDMARKS_3D_MEDIAPIPE = auto()
    """
    MediaPipe pose landmarks in 3D. Each landmark is given by name and three coordinates (x,y,z) in camera space.

    **Type**: `Dict[InstanceId, Dict[LandmarkId, Landmark3d]]`. hould have no more than 33 points.
    """
    LANDMARKS_3D_COCO = auto()
    """
    COCO whole body landmarks in 3D. Each landmark is given by name and three coordinates (x,y,z) in camera space.

    **Type**: `Dict[InstanceId, Dict[LandmarkId, Landmark3d]]`. Should have no more than 133 points.
    """
    LANDMARKS_3D_MPEG4 = auto()
    """
    MPEG4 landmarks in 3D. Each landmark is given by name and three coordinates (x,y,z) in camera space.

    **Type**: `Dict[InstanceId, Dict[LandmarkId, Landmark3d]]`.
    """
    PUPILS = auto()
    """
    Coordinates of pupils. Each pupil is given by name and two coordinates (x,y) in pixels.
    
    **Type**: `Dict[InstanceId, Dict[str, Landmark2d]]`.
    """
    PUPILS_3D = auto()
    """
    Coordinates of pupils in 3D. Each pupil is given by name and three coordinates (x,y,z) in camera space.
 
    **Type**: `Dict[InstanceId, Dict[str, Landmark3d]]`.
    """
    IDENTITY = auto()
    """
    Mapping from instance ID (ranging from 0 to the number of humans in the image) to human ID (corresponding to the ID provided in the job config)
    
    **Type**: `Dict[InstanceId, int]`.
    """
    IDENTITY_METADATA = auto()
    """
    Additional metadata about the people in the image.
    
    **Format**::
    
        {
            instance_id: {
                'gender': 'female'|'male',
                'age': int,
                'weight_kg': int,
                'height_cm': int,
                'id': int,
                'ethnicity': 'arab'|'asian'|'black'|'hisp'|'white'
            },
            ...
        }
    """
    HAIR = auto()
    """
    Hair metadata. If no hair are present `None` is returned.
    
    **Format**::
    
        {
            instance_id: {
                'relative_length': float64,
                'relative_density': float64,
                'style': str,
                'color_seed': float64,
                'color': str
            },
            ...
        }
    """
    FACIAL_HAIR = auto()
    """
    Facial hair metadata. If no facial hair is present for a human, `None` is provided.

    **Format**::
    
        {
            instance_id: {
                'relative_length': float64,
                'relative_density': float64,
                'style': str,
                'color_seed': float64,
                'color': str
            },
            ...
        }
    """
    EXPRESSION = auto()
    """
    Expression and its intensity.
    
    **Format**::
    
        {
            instance_id: {
                'intensity': float64, 
                'name': str
            },
            ...
        }
    """
    GAZE = auto()
    """
    Gaze direction in camera space.
    
    **Format**::
        {
            instance_id: {
                'horizontal_angle': ndarray[float64] **Shape**: `(3,)`.
                'vertical_angle': ndarray[float64] **Shape**: `(3,)`.
            },
            ...
        }
    """
    FACE_BBOX = auto()
    """
    Face bounding box in the format (left, top, right, bottom) in pixels.
    
    **Type**: `Dict[InstanceId, Tuple[int, int, int, int]]`.
    """
    HEAD_TO_CAM = auto()
    """
    Transformation matrix from the head to the camera coordinate system.
    
    **Type**: `Dict[InstanceId, ndarray[float32]]`. **Shape**: `(4, 4)`.
    """
    CAM_TO_HEAD = auto()
    """  
    Transformation matrix from the camera to the head coordinate system.
    
    **Type**: `Dict[InstanceId, ndarray[float32]]`. **Shape**: `(4, 4)`.
    """
    HEAD_TO_WORLD = auto()
    """
    Transformation matrix from the head to the world coordinate system.
    
    **Type**: `Dict[InstanceId, ndarray[float32]]`. **Shape**: `(4, 4)`.
    """
    WORLD_TO_HEAD = auto()
    """
    Transformation matrix from the world to the head coordinate system.
    
    **Type**: `Dict[InstanceId, ndarray[float32]]`. **Shape**: `(4, 4)`.
    """
    CAM_TO_WORLD = auto()
    """
    Transformation matrix from the camera to the world coordinate system.
    
    **Type**: `ndarray[float32]`. **Shape**: `(4, 4)`.
    """
    WORLD_TO_CAM = auto()
    """
    Transformation matrix from the world to the camera coordinate system.
    
    **Type**: `ndarray[float32]`. **Shape**: `(4, 4)`.
    """
    CAM_INTRINSICS = auto()
    """
    Camera intrinsics matrix in OpenCV format: https://docs.opencv.org/3.4.15/dc/dbb/tutorial_py_calibration.html.

    **Type**: `ndarray[float32]`. **Shape**: `(4, 4)`.
    """
    CAMERA_NAME = auto()
    """
    Camera name consisting of lowercase alphanumeric characters. Usually used when more than one are defined in a scene.
    Default is "cam_default".
    
    **Type**: `str`. 
    """
    FRAME_NUM = auto()
    """
    Frame number used for consecutive animation frames. Used for animation.
    
    **Type**: `int`.
    """
    LANDMARKS_3D_MEDIAPIPE_FACE = auto()
    """
    MediaPipe dense face landmarks in 3D. Each landmark is given by three coordinates (x,y,z) in camera space.

    **Type**: `Dict[InstanceId, ndarray[float32]]`. **Shape**: `(468, 3)`.
    """
    LANDMARKS_3D_SAI = auto()
    """
    SAI dense face landmarks in 3D. Each landmark is given by three coordinates (x,y,z) in camera space.

    **Type**: `Dict[InstanceId, ndarray[float32]]`. **Shape**: `(4840, 3)`.
    """
    LANDMARKS_MEDIAPIPE_FACE = auto()
    """
    MediaPipe dense face landmarks. Each landmark is given by two coordinates (x,y) in pixels.
    
    **Type**: `Dict[InstanceId, ndarray[float32]]`. **Shape**: `(468, 2)`.
    """
    LANDMARKS_SAI = auto()
    """
    SAI dense face landmarks. Each landmark is given by two coordinates (x,y) in pixels.
    
    **Type**: `Dict[InstanceId, ndarray[float32]]`. **Shape**: `(4840, 2)`.
    """
