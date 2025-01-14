import os
import shutil
import tempfile
from pathlib import Path
import pickle
from tempfile import TemporaryDirectory

from cgmml.common.data_utilities.mlpipeline_utils import create_layers, ArtifactProcessor, create_layers_rgbd

DATA_UTILITIES_DIR = Path(__file__).parents[1].absolute()
COMMON_DIR = DATA_UTILITIES_DIR.parent
TEST_DATA_DIR = COMMON_DIR / 'depthmap_toolkit/tests/huawei_p40pro'
ARTIFACT_ZIP_PATH = 'be1faf54-69c7-11eb-984b-a3ffd42e7b5a/depth/bd67cd9e-69c7-11eb-984b-77ac9d2b4986'
DEPTHMAP_PATH = 'depth/depth_dog_1622182020448_100_282.depth'
RGB_PATH = 'rgb/rgb_dog_1622182020448_100_282.jpg'
RGB_ROTATED_PATH = 'rgb/rgb_dog_1622182020448_100_282_rotated.jpg'
PICKLE_PATH = (DATA_UTILITIES_DIR / 'tests' / 'pickle_files' / 'scans' / '09495c00-ff19-11eb-95bd-6fef000028be'
               / '102' / 'pc_c571de02-a723-11eb-8845-bb6589a1fbe8_2021-04-22_13-34-33-302557_102_3.p')
ARTIFACT_DICT = {
    'file_path': DEPTHMAP_PATH,
    'timestamp': '2021-04-22_13-34-33-302557',
    'scan_id': 'c571de02-a723-11eb-8845-bb6589a1fbe8',
    'scan_step': 102,
    'height': 50.0,
    'weight': 20.555,
    'muac': 10.0,
    'order_number': 3,
    'person_id': '09495c00-ff19-11eb-95bd-6fef000028be'
}


def test_artifact_processor_depthmap():
    with TemporaryDirectory() as output_dir:
        artifact_processor = ArtifactProcessor(TEST_DATA_DIR, output_dir, dataset_type='depthmap')
        processed_fname = artifact_processor.create_and_save_pickle(ARTIFACT_DICT)

        depthmap, targets = pickle.load(open(processed_fname, 'rb'))
        assert depthmap.shape == (240, 180, 1), depthmap.shape
        assert 'height' in targets

        pickle_path_expected = str(PICKLE_PATH)
        assert pickle_path_expected.split('/')[-4:] == processed_fname.split('/')[-4:]


def test_create_layers():
    depthmap_fpath = TEST_DATA_DIR / DEPTHMAP_PATH
    layers, metadata = create_layers(depthmap_fpath)
    assert layers.shape == (240, 180, 1), layers.shape

    assert isinstance(metadata['raw_header'], str), metadata['raw_header']
    assert metadata['raw_header'].startswith('240x180')


def test_artifact_processor_rgbd():
    with TemporaryDirectory() as output_dir:
        artifact_processor = ArtifactProcessor(TEST_DATA_DIR, output_dir, dataset_type='rgbd', should_rotate_rgb=True)

        ARTIFACT_DICT['file_path_rgb'] = RGB_PATH
        ARTIFACT_DICT['file_path'] = DEPTHMAP_PATH

        processed_fname = artifact_processor.create_and_save_pickle(ARTIFACT_DICT)

        layers, targets = pickle.load(open(processed_fname, 'rb'))
        assert layers.shape == (240, 180, 4), layers.shape
        assert 'height' in targets
        assert 'raw_header' in targets
        assert -90 < targets['angle'] < 0

        pickle_path_expected = str(PICKLE_PATH)
        assert pickle_path_expected.split('/')[-4:] == processed_fname.split('/')[-4:]


def test_create_layers_rgbd():
    zip_input_full_path = TEST_DATA_DIR / DEPTHMAP_PATH
    rgb_input_full_path = TEST_DATA_DIR / RGB_PATH
    layers, metadata = create_layers_rgbd(zip_input_full_path, rgb_input_full_path, should_rotate_rgb=True)
    assert layers.shape == (240, 180, 4), layers.shape

    assert isinstance(metadata['raw_header'], str), metadata['raw_header']
    expected_header = '240x180_0.001_7_-0.15386176_0.6911723_0.6840933_-0.17500913_0.024252899_-0.09748171_0.07438941'
    assert metadata['raw_header'] == expected_header
    assert -90 < metadata['angle'] < 0


def test_create_layers_rgbd_should_not_rotate():
    zip_input_full_path = TEST_DATA_DIR / DEPTHMAP_PATH
    rgb_input_full_path = TEST_DATA_DIR / RGB_ROTATED_PATH
    temp_dir = tempfile.gettempdir()
    temp_path = os.path.join(temp_dir, 'f4f8bc5c-cb25-42d1-87ce-0bfea108ddb8')
    shutil.copy2(rgb_input_full_path, temp_path)
    rgb_input_full_path = Path(temp_path)
    layers, metadata = create_layers_rgbd(zip_input_full_path, rgb_input_full_path, should_rotate_rgb=False)
    assert layers.shape == (240, 180, 4), layers.shape
