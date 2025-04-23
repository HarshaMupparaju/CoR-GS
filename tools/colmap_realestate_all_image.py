import os
import numpy as np
import sys
import sqlite3
from scipy.spatial.transform import Rotation

IS_PYTHON3 = sys.version_info[0] >= 3
MAX_IMAGE_ID = 2**31 - 1

CREATE_CAMERAS_TABLE = """CREATE TABLE IF NOT EXISTS cameras (
    camera_id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
    model INTEGER NOT NULL,
    width INTEGER NOT NULL,
    height INTEGER NOT NULL,
    params BLOB,
    prior_focal_length INTEGER NOT NULL)"""

CREATE_DESCRIPTORS_TABLE = """CREATE TABLE IF NOT EXISTS descriptors (
    image_id INTEGER PRIMARY KEY NOT NULL,
    rows INTEGER NOT NULL,
    cols INTEGER NOT NULL,
    data BLOB,
    FOREIGN KEY(image_id) REFERENCES images(image_id) ON DELETE CASCADE)"""

CREATE_IMAGES_TABLE = """CREATE TABLE IF NOT EXISTS images (
    image_id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
    name TEXT NOT NULL UNIQUE,
    camera_id INTEGER NOT NULL,
    prior_qw REAL,
    prior_qx REAL,
    prior_qy REAL,
    prior_qz REAL,
    prior_tx REAL,
    prior_ty REAL,
    prior_tz REAL,
    CONSTRAINT image_id_check CHECK(image_id >= 0 and image_id < {}),
    FOREIGN KEY(camera_id) REFERENCES cameras(camera_id))
""".format(MAX_IMAGE_ID)

CREATE_TWO_VIEW_GEOMETRIES_TABLE = """
CREATE TABLE IF NOT EXISTS two_view_geometries (
    pair_id INTEGER PRIMARY KEY NOT NULL,
    rows INTEGER NOT NULL,
    cols INTEGER NOT NULL,
    data BLOB,
    config INTEGER NOT NULL,
    F BLOB,
    E BLOB,
    H BLOB,
    qvec BLOB,
    tvec BLOB)
"""

CREATE_KEYPOINTS_TABLE = """CREATE TABLE IF NOT EXISTS keypoints (
    image_id INTEGER PRIMARY KEY NOT NULL,
    rows INTEGER NOT NULL,
    cols INTEGER NOT NULL,
    data BLOB,
    FOREIGN KEY(image_id) REFERENCES images(image_id) ON DELETE CASCADE)
"""

CREATE_MATCHES_TABLE = """CREATE TABLE IF NOT EXISTS matches (
    pair_id INTEGER PRIMARY KEY NOT NULL,
    rows INTEGER NOT NULL,
    cols INTEGER NOT NULL,
    data BLOB)"""

CREATE_NAME_INDEX = \
    "CREATE UNIQUE INDEX IF NOT EXISTS index_name ON images(name)"

CREATE_ALL = "; ".join([
    CREATE_CAMERAS_TABLE,
    CREATE_IMAGES_TABLE,
    CREATE_KEYPOINTS_TABLE,
    CREATE_DESCRIPTORS_TABLE,
    CREATE_MATCHES_TABLE,
    CREATE_TWO_VIEW_GEOMETRIES_TABLE,
    CREATE_NAME_INDEX
])


def array_to_blob(array):
    if IS_PYTHON3:
        return array.tostring()
    else:
        return np.getbuffer(array)

def blob_to_array(blob, dtype, shape=(-1,)):
    if IS_PYTHON3:
        return np.fromstring(blob, dtype=dtype).reshape(*shape)
    else:
        return np.frombuffer(blob, dtype=dtype).reshape(*shape)

class COLMAPDatabase(sqlite3.Connection):

    @staticmethod
    def connect(database_path):
        return sqlite3.connect(database_path, factory=COLMAPDatabase)

    def __init__(self, *args, **kwargs):
        super(COLMAPDatabase, self).__init__(*args, **kwargs)

        self.create_tables = lambda: self.executescript(CREATE_ALL)
        self.create_cameras_table = \
            lambda: self.executescript(CREATE_CAMERAS_TABLE)
        self.create_descriptors_table = \
            lambda: self.executescript(CREATE_DESCRIPTORS_TABLE)
        self.create_images_table = \
            lambda: self.executescript(CREATE_IMAGES_TABLE)
        self.create_two_view_geometries_table = \
            lambda: self.executescript(CREATE_TWO_VIEW_GEOMETRIES_TABLE)
        self.create_keypoints_table = \
            lambda: self.executescript(CREATE_KEYPOINTS_TABLE)
        self.create_matches_table = \
            lambda: self.executescript(CREATE_MATCHES_TABLE)
        self.create_name_index = lambda: self.executescript(CREATE_NAME_INDEX)

    def update_camera(self, model, width, height, params, camera_id):
        params = np.asarray(params, np.float64)
        cursor = self.execute(
            "UPDATE cameras SET model=?, width=?, height=?, params=?, prior_focal_length=1 WHERE camera_id=?",
            (model, width, height, array_to_blob(params),camera_id))
        return cursor.lastrowid

def round_python3(number):
    rounded = round(number)
    if abs(number - rounded) == 0.5:
        return 2.0 * round(number / 2.0)
    return rounded

def get_quaternions_and_translations(trans_mat: np.ndarray):
    rot_mat = trans_mat[:3, :3]
    rotation = Rotation.from_matrix(rot_mat)
    assert type(rotation) == Rotation
    quaternions = rotation.as_quat()
    quaternions = np.roll(quaternions, 1)
    quaternions_str = ' '.join(quaternions.astype('str'))
    translations = trans_mat[:3, 3]
    translations_str = ' '.join(translations.astype('str'))
    return quaternions_str, translations_str

def pipeline(scene, base_path, n_views, r):
    llffhold = 8
    view_path = str(n_views) + '_views'
    os.chdir(base_path + scene)
    os.system('rm -r ' + view_path)
    os.mkdir(view_path)
    os.chdir(view_path)
    os.mkdir('created')
    os.mkdir('triangulated')
    os.mkdir('images')
    # os.system('colmap model_converter  --input_path ../sparse/0/ --output_path ../sparse/0/  --output_type TXT')

    intrinsics_filepath = f'{base_path}{scene}/CameraIntrinsics.csv'
    extrinsics_filepath = f'{base_path}{scene}/CameraExtrinsics.csv'
    


    # images = {}
    # with open('../sparse/0/images.txt', "r") as fid:
    #     while True:
    #         line = fid.readline()
    #         if not line:
    #             break
    #         line = line.strip()
    #         if len(line) > 0 and line[0] != "#":
    #             elems = line.split()
    #             image_id = int(elems[0])
    #             qvec = np.array(tuple(map(float, elems[1:5])))
    #             tvec = np.array(tuple(map(float, elems[5:8])))
    #             camera_id = int(elems[8])
    #             image_name = elems[9]
    #             fid.readline().split()
    #             images[image_name] = elems[1:]

    # img_list = sorted(images.keys(), key=lambda x: x)
    # frame_nums = list(range(0, len(img_list)))
    # test_frame_nums = list(range(0, len(img_list), 8))
    # train_frame_nums = list(set(frame_nums) - set(test_frame_nums))
    # train_img_list = [c for idx, c in enumerate(img_list) if idx in train_frame_nums]
    # train_frame_nums = list(range(5, 50, 10))
    # test_frame_nums = list(set(range(50)) - set(train_frame_nums))
    train_frame_nums = list(range(0, 50))

    # train_img_list = [c for idx, c in enumerate(img_list) if idx % llffhold != 0]
    if n_views > 0:
        # idx_sub = [round_python3(i) for i in np.linspace(0, len(train_img_list)-1, n_views)]
        # idx_sub = np.round(np.linspace(-1, len(train_img_list), n_views+2)).astype('int')[1:-1]
        # final_idx_sub = [train_frame_nums[i] for i in idx_sub]
        # train_img_list = [c for idx, c in enumerate(img_list) if idx in final_idx_sub]
        num_extrapolation_frames = 5
        train_frame_nums = [10, 20, 30, 0, 40]
        test_frame_nums = list(set(range(50)) - set(train_frame_nums))
        train_frame_nums = sorted(train_frame_nums[:n_views])
        test_frame_nums = [f for f in test_frame_nums if
                           ((f > min(train_frame_nums)) and (f < max(train_frame_nums))) or
                           ((abs(min(train_frame_nums) - f) <= num_extrapolation_frames) or (abs(f - max(train_frame_nums)) <= num_extrapolation_frames))
                           ]

    intrinsics = np.loadtxt(intrinsics_filepath, delimiter=',').reshape((-1, 3, 3))[train_frame_nums]
    extrinsics = np.loadtxt(extrinsics_filepath, delimiter=',').reshape((-1, 4, 4))[train_frame_nums]


    full_res_img_names = []
    for idx, img_name in enumerate(train_frame_nums):
        if(r != 1):
            full_res_img_names.append(img_name)
            img_name = img_name.replace('.jpg', '.png')
            if(scene != 'horns' and scene != 'room' and scene != 'trex'):
                img_name = f'image{final_idx_sub[idx]:03d}.png'
            if(scene == 'room'):
                img_name = img_name.replace('.JPG', '.png')
            os.system(f'cp ../images_{r}/' + img_name + '  images/' + img_name)
        else:
            img_name = f'{img_name:04d}.png'
            os.system(f'cp ../images/' + img_name + '  images/' + img_name)

    # os.system('cp ../sparse/0/cameras.txt created/.')
    # if(r != 1):
    #     with open('created/cameras.txt', "r") as fid:
    #         lines = fid.readlines()
    #     with open('created/cameras.txt', "w") as fid:
    #         for line in lines:
    #             if len(line) > 0 and line[0] != "#":
    #                 elems = line.split()
    #                 elems[2] = str(int(int(elems[2])/r))
    #                 elems[3] = str(int(int(elems[3])/r))
    #                 elems[4] = str(float(float(elems[4])/r))
    #                 elems[5] = str(int(int(elems[5])/r))
    #                 elems[6] = str(int(int(elems[6])/r))

    #                 fid.write(' '.join(elems) + '\n')
    #             elif(len(line) > 0 and line[0] == '#'):
    #                 fid.write(line)

    with open('created/cameras.txt', "w") as fid:
        # Write intrinsics
        for idx, intr in enumerate(intrinsics):
            fid.write(f'{idx + 1} PINHOLE {intr[0, 2] * 2} {intr[1, 2] * 2} {intr[0, 0]} {intr[1, 1]} {intr[0, 2]} {intr[1, 2]}\n')

    with open('created/points3D.txt', "w") as fid:
        pass

    res = os.popen( 'colmap feature_extractor --database_path database.db --image_path images  --SiftExtraction.max_image_size 4032 --SiftExtraction.max_num_features 32768 --SiftExtraction.estimate_affine_shape 1 --SiftExtraction.domain_size_pooling 1').read()
    os.system( 'colmap exhaustive_matcher --database_path database.db --SiftMatching.guided_matching 1 --SiftMatching.max_num_matches 32768')
    db = COLMAPDatabase.connect('database.db')
    db_images = db.execute("SELECT * FROM images")
    img_rank = [db_image[1] for db_image in db_images]
    with open('created/images.txt', "w") as fid:
        for idx, img_name in enumerate(img_rank):
            if(r != 1):
                img_name = full_res_img_names[idx]
            extrinsic = extrinsics[idx]
            quarternions, translations = get_quaternions_and_translations(extrinsic)
            db_image_data = db.execute(f"SELECT * FROM images WHERE name = '{img_name}'").fetchone()
            camera_id = db_image_data[2]
            data = [f'{idx + 1} {quarternions} {translations} {camera_id} {img_name} \n\n'] 

            if(r != 1):
                if (scene != 'horns' and scene != 'room' and scene != 'trex'):
                    data[9] = f' image{final_idx_sub[idx]:03d}.png'
                else:
                    if(scene == 'room'):
                        data[9] = data[9].replace('.JPG', '.png')
                    else:
                        data[9] = data[9].replace('.jpg', '.png')
            fid.writelines(data)

    os.system('colmap point_triangulator --database_path database.db --image_path images --input_path created  --output_path triangulated  --Mapper.ba_local_max_num_iterations 40 --Mapper.ba_local_max_refinements 3 --Mapper.ba_global_max_num_iterations 100')
    os.system('colmap model_converter  --input_path triangulated --output_path triangulated  --output_type TXT')
    os.system('colmap image_undistorter --image_path images --input_path triangulated --output_path dense')
    os.system('colmap patch_match_stereo --workspace_path dense')
    os.system('colmap stereo_fusion --workspace_path dense --output_path dense/fused.ply')


for scene in ['00000', '00001', '00003', '00004', '00006']:
    # pipeline(scene, base_path = '/mnt/2tb-hdd/Harsha/CoR-GS/data/nerf_llff_data/', n_views = 2, r = 1)  # please use absolute path!
    # pipeline(scene, base_path = '/mnt/2tb-hdd/Harsha/CoR-GS/data/nerf_llff_data/', n_views = 3, r = 1)
    # pipeline(scene, base_path = '/mnt/2tb-hdd/Harsha/CoR-GS/data/nerf_llff_data/', n_views = 4, r = 1)
    # pipeline(scene, base_path = '/mnt/2tb-hdd/Harsha/CoR-GS/data/RealEstate10K/', n_views = -1, r = 1)

pipeline('00000', base_path = '/mnt/2tb-hdd/Harsha/CoR-GS/data/RealEstate10K/', n_views = -1, r = 1)