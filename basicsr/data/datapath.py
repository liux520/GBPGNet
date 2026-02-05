


def train_dataset_dict(task, dataset):
    data_dict = {
        'Derain': {
            'Rain13K': ['Derain/Rain13K/train/Rain100H/input',
                        'Derain/Rain13K/train/Rain100H/target'],
            'Lin': ['Derain/Lin/transdata/input', 'Derain/Lin/transdata/target'],
            'RealRain1k_L': ['Derain/RealRain1k/RealRain-1k-L/train/input',
                             'Derain/RealRain1k/RealRain-1k-L/train/target'],
            'RealRain1k_H': ['Derain/RealRain1k/RealRain-1k-H/train/input',
                             'Derain/RealRain1k/RealRain-1k-H/train/target'],
            'Cityscapes100': ['Derain/Cityscapes/Cityscapes_syn_100mm/trainval/input',
                              'Derain/Cityscapes/Cityscapes_syn_100mm/trainval/target'],
            'Cityscapes200': ['Derain/Cityscapes/Cityscapes_syn_200mm/trainval/input',
                              'Derain/Cityscapes/Cityscapes_syn_200mm/trainval/target'],
        },
        'Dehaze': {
            'SOTS': ['Dehaze/RESIDE/OTS_BETA/input',
                     'Dehaze/RESIDE/OTS_BETA/target']
        },
        'Desnow': {
            'CSD': ['Desnow/CSD/train/input',
                    'Desnow/CSD/train/target'],
            'Snow100K': ['Desnow/Snow100K/train/input',
                         'Desnow/Snow100K/train/target'],
            'SnowCity': ['Desnow/SnowCity/train/input',
                         'Desnow/SnowCity/train/target'],
            'SnowKitti': ['Desnow/Kitti/train/input',
                          'Desnow/Kitti/train/target'],
            'RealSnow': ['Desnow/RealSnow/train/input',
                         'Desnow/RealSnow/train/target']
        },
        'LowLight': {
            'LOL': ['Lowlight/LOL/LOLdataset/our485/low',
                    'Lowlight/LOL/LOLdataset/our485/high']
        },
        'Deblur': {
            'GoPro': ['Deblur/GoPro/train_patch/input',
                      'Deblur/GoPro/train_patch/target']
        }
    }
    return data_dict[task][dataset]


def test_dataset_dict(task, dataset):
    data_dict = {
        'Derain': {
            'Rain100H': ['Derain/Rain13K/test/Rain100H/input',
                         'Derain/Rain13K/test/Rain100H/target'],
            'Rain100L': ['Derain/Rain13K/test/Rain100L/input',
                         'Derain/Rain13K/test/Rain100L/target'],
            'Test100': ['Derain/Rain13K/test/Test100/input',
                        'Derain/Rain13K/test/Test100/target'],
            'Test1200': ['Derain/Rain13K/test/Test1200/input',
                         'Derain/Rain13K/test/Test1200/target'],
            'Test2800': ['Derain/Rain13K/test/Test2800/input',
                         'Derain/Rain13K/test/Test2800/target'],
            'Lin': ['Derain/Lin/transdata_test/input', 'Derain/Lin/transdata_test/target'],
            'RealRain1k_L': ['Derain/RealRain1k/RealRain-1k-L/test/input',
                             'Derain/RealRain1k/RealRain-1k-L/test/target'],
            'RealRain1k_H': ['Derain/RealRain1k/RealRain-1k-H/test/input',
                             'Derain/RealRain1k/RealRain-1k-H/test/target'],
            'Cityscapes100': ['Derain/Cityscapes/Cityscapes_syn_100mm/test/input',
                              'Derain/Cityscapes/Cityscapes_syn_100mm/test/target'],
            'Cityscapes200': ['Derain/Cityscapes/Cityscapes_syn_200mm/test/input',
                              'Derain/Cityscapes/Cityscapes_syn_200mm/test/target'],
        },
        'Dehaze': {
            'SOTS': ['Dehaze/RESIDE/SOTS/outdoor/input_all',
                     'Dehaze/RESIDE/SOTS/outdoor/target']
        },
        'Desnow': {
            'CSD': ['Desnow/CSD/test/input',
                    'Desnow/CSD/test/target'],
            'Snow100K': ['Desnow/Snow100K/test/Snow100K-L/input',
                         'Desnow/Snow100K/test/Snow100K-L/target'],
            'SnowCity': ['Desnow/SnowCity/test/input_L',
                         'Desnow/SnowCity/test/target'],
            'SnowKitti': ['Desnow/Kitti/test/input_L',
                          'Desnow/Kitti/test/target'],
            'RealSnow': ['Desnow/RealSnow/test/input',
                         'Desnow/RealSnow/test/target']
        },
        'LowLight': {
            'LOL': ['Lowlight/LOL/LOLdataset/eval15/low',
                    'Lowlight/LOL/LOLdataset/eval15/high']
        },
        'Deblur': {
            'GoPro': ['Deblur/GoPro/test',
                      'Deblur/GoPro/test'],
            # 'GoPro': ['Deblur/GoPro/test/GOPR0384_11_00/blur',
            #           'Deblur/GoPro/test/GOPR0384_11_00/sharp'],
            'HIDE': ['Deblur/HIDE/input',
                     'Deblur/HIDE/target'],
            'RealBlur': ['',
                         '']
        }
    }
    return data_dict[task][dataset]
