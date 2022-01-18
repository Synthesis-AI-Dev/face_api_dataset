from face_api_dataset import FaceApiDataset, Grouping


def test_smoke():
    data_root = "test_dataset"
    dataset = FaceApiDataset(data_root)
    item = dataset[2]
    assert item


def test_grouping_scene():
    data_root = "test_dataset"
    dataset = FaceApiDataset(data_root, grouping=Grouping.SCENE)
    item = dataset[10]
    assert item


def test_grouping_camera():
    data_root = "test_dataset"
    dataset = FaceApiDataset(data_root, grouping=Grouping.CAMERA)
    item = dataset[0]
    assert item


def test_grouping_camera_scene():
    data_root = "test_dataset"
    dataset = FaceApiDataset(data_root, grouping=Grouping.SCENE_CAMERA)
    item = dataset[0]
    assert item


def test_get_group_index():
    data_root = "test_dataset"
    dataset = FaceApiDataset(data_root, grouping=Grouping.SCENE_CAMERA)
    index = dataset.get_group_index()
