from torch.utils import data as data
from torchvision.transforms.functional import normalize
from tqdm import tqdm
import os
from pathlib import Path
import random
import numpy as np
import torch

from basicsr.data.data_util import (paired_paths_from_folder,
                                    paired_paths_from_lmdb,
                                    paired_paths_from_meta_info_file,
                                    recursive_glob)
from basicsr.data.event_util import events_to_voxel_grid, events_to_voxel_grid_pytorch, voxel_norm
from basicsr.data.transforms import augment, triple_random_crop, random_augmentation
from basicsr.utils import FileClient, imfrombytes, img2tensor, voxel2voxeltensor, padding, get_root_logger
from torch.utils.data.dataloader import default_collate


class RuisiSingleImageEventDataset(data.Dataset):
    """GoPro dataset for training recurrent networks for blurry image interpolation.

    Args:
        opt (dict): Config for train dataset. It contains the following keys:
            dataroot (str): Data root path.
            io_backend (dict): IO backend type and other kwarg.
            num_end_interpolation (int): Number of sharp frames to reconstruct in each blurry image.
            num_inter_interpolation (int): Number of sharp frames to interpolate between two blurry images.
            phase (str): 'train' or 'test'

            gt_size (int): Cropped patched size for gt patches.
            random_reverse (bool): Random reverse input frames.
            use_hflip (bool): Use horizontal flips.
            use_rot (bool): Use rotation (use vertical flip and transposing h
                and w for implementation).

            scale (bool): Scale, which will be added automatically.
    """

    def __init__(self, opt):
        super(RuisiSingleImageEventDataset, self).__init__()
        self.opt = opt
        self.dataroot = Path(opt['dataroot'])
        # self.m = opt['num_end_interpolation']
        # self.n = opt['num_inter_interpolation']
        self.num_avg_blur = opt['num_avg_blur']
        self.num_offset = (self.num_avg_blur - 1) // 2
        # self.num_input_blur = 2
        # self.num_input_gt = 2*self.m + self.n
        self.num_bins = opt['num_bins']
        self.split = 'train' if opt['phase']=='train' else 'test' # train or test
        self.norm_voxel = opt.get('norm_voxel', True)
        self.one_voxel_flg = opt.get('one_voxel_flag', True)
        self.return_deblur_voxel = opt.get('return_deblur_voxel', False)
        self.return_deblur_voxel = self.return_deblur_voxel and self.one_voxel_flg
        self.random_reduction_rate = opt.get('random_reduction_rate', 1)


        ## the sequence names
        # train_video_list = [
        #     'GOPR0372_07_00', 'GOPR0374_11_01', 'GOPR0378_13_00', 'GOPR0384_11_01', 'GOPR0384_11_04', 'GOPR0477_11_00', 'GOPR0868_11_02', 'GOPR0884_11_00', 
        #     'GOPR0372_07_01', 'GOPR0374_11_02', 'GOPR0379_11_00', 'GOPR0384_11_02', 'GOPR0385_11_00', 'GOPR0857_11_00', 'GOPR0871_11_01', 'GOPR0374_11_00', 
        #     'GOPR0374_11_03', 'GOPR0380_11_00', 'GOPR0384_11_03', 'GOPR0386_11_00', 'GOPR0868_11_01', 'GOPR0881_11_00']
        
        # test_video_list = [
        #     'GOPR0384_11_00', 'GOPR0385_11_01', 'GOPR0410_11_00', 'GOPR0862_11_00', 'GOPR0869_11_00', 'GOPR0881_11_01', 'GOPR0384_11_05', 'GOPR0396_11_00', 
        #     'GOPR0854_11_00', 'GOPR0868_11_00', 'GOPR0871_11_00']
        
        train_video_list = os.listdir(os.path.join(self.dataroot, 'train'))
        test_video_list = os.listdir(os.path.join(self.dataroot, 'test'))

        video_list = train_video_list if self.split=='train' else test_video_list

        self.allBlurPath = [] # list of blur images
        self.allSharpPath = [] # list of sharp images
        self.alleventSeqsPath = [] # list of lists of event frames

        ### Formate file lists
        for video in video_list:
            
            blur_frames   = sorted(recursive_glob(rootdir=os.path.join(self.dataroot, self.split, video, 'blur'), suffix='.png'))  # blur frames in one video
            gt_frames  = sorted(recursive_glob(rootdir=os.path.join(self.dataroot, self.split, video, 'gt'), suffix='.png'))  # all sharp frames in one video
            event_frames = sorted(recursive_glob(rootdir=os.path.join(self.dataroot, self.split, video, 'event'), suffix='.npz'))  # all sharp frames in one video
            
            # blurInputs = [[os.path.join(self.dataroot, self.split, video, 'blur',f) for f in group] for group in blurInputs] # GOPR0372_07_00/xxx.png ...

            self.allBlurPath.extend(os.path.join(self.dataroot, self.split, video, 'blur', f) for f in blur_frames)
            # print('debug: {}'.format(self.allBlurPath))
            self.allBlurPath.sort()

            # self.allBlurPath.extend(blur_frames)

        ### event
        for blur_path in self.allBlurPath:
            num_name = os.path.basename(blur_path)
            int_num_name = int(num_name.split('.')[0].replace('_0', ''))
            # print('DEBUG int_num_name:{}'.format(int_num_name))

            int_num_start = int_num_name - self.num_offset
            int_num_end = int_num_name + self.num_offset
            # print('DEBUG int_num_start:{}'.format(int_num_start))
            # print('DEBUG int_num_end:{}'.format(int_num_end))

            npz_files = []
            for i in range(int_num_start, int_num_end):
                basename = '%.8d' % i + '.npz'
                npz_path = blur_path.replace('/blur/', '/event/')
                npz_path = os.path.dirname(npz_path)
                npz_path = os.path.join(npz_path, basename)
                # npz_path = os.path.join( self.dataroot, self.split+'_event', video, basename)
                npz_files.append(npz_path)

            self.alleventSeqsPath.append(npz_files) ### list of lists [[1,2],[1,2]]


        for blur_path in self.allBlurPath:
            sharp_path = blur_path.replace('blur/', 'gt/').replace('_0', '')
            self.allSharpPath.append(sharp_path)

        # print('DEBUG allBlurPath:{}'.format(self.allBlurPath))
        # print('DEBUG allSharpPath:{}'.format(self.allSharpPath))
        # print('DEBUG alleventSeqsPath:{}'.format(self.alleventSeqsPath))

        assert len(self.allBlurPath)==len(self.allSharpPath)==len(self.alleventSeqsPath), 'The number of blur/sharp/event: {}/{}/{} does not match.'.format(len(self.allBlurPath),len(self.allSharpPath),len(self.alleventSeqsPath))
        
        # print('DEBUG: length of blurPairsPath:{}'.format(len(self.blurPairsPath)))

        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']

        # temporal augmentation configs
        self.random_reverse = opt.get('random_reverse', False)
        logger = get_root_logger()
        logger.info(f'Temporal augmentation: random reverse is {self.random_reverse}.')

        # self.allBlurPath:      # list of blur images
        # self.allSharpPath:     # list of sharp images
        # self.alleventSeqsPath: # list of lists of event npz files 

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)
        scale = self.opt['scale']
        gt_size = self.opt['gt_size']

        image_path = self.allBlurPath[index] # single image
        gt_path = self.allSharpPath[index]#    single image
        event_paths = self.alleventSeqsPath[index] # events

        # print('DEBUG image paths:{}'.format(image_path))
        # print('DEBUG gt_paths :{}'.format(gt_path))
        # print('DEBUG event_paths:{}'.format(event_paths))

        img_byte = self.file_client.get(image_path)  # 'lq'
        img_lq = imfrombytes(img_byte, float32=True)

        # for gt_path in gt_paths:
            # get GT
        img_byte_gt = self.file_client.get(gt_path)    # 'gt'
        img_gt = imfrombytes(img_byte_gt, float32=True)

        h_lq, w_lq, _ = img_lq.shape
        ## Read event and convert to voxel grid:
        events = [np.load(event_path) for event_path in event_paths]
        # npz -> ndarray
        all_quad_event_array = np.zeros((0,4)).astype(np.float32)
        for event in events:
            ### x and y are switched because of the mistake in the dataset
            y = event['x'].astype(np.float32)             
            x = event['y'].astype(np.float32)            
            t = event['timestamp'].astype(np.float32)
            p = event['polarity'].astype(np.float32)

            this_quad_event_array = np.concatenate((t,x,y,p),axis=1) # N,4
            ### random sample events to reduce the number of events
            if self.random_reduction_rate <= 1:
                this_quad_event_array = this_quad_event_array[np.random.choice \
                    (this_quad_event_array.shape[0], int(this_quad_event_array.shape[0]*self.random_reduction_rate), replace=False)]

            all_quad_event_array = np.concatenate((all_quad_event_array, this_quad_event_array), axis=0)
        voxel = events_to_voxel_grid(all_quad_event_array, num_bins=self.num_bins, width=w_lq, height=h_lq, return_format='HWC')
        # Voxel Norm
        # if self.norm_voxel:
        #     voxel = voxel_norm(voxel)

        if gt_size is not None:
            img_gt, img_lq, voxel = triple_random_crop(img_gt, img_lq, voxel, gt_size, scale, gt_path)
        all_imgs = [img_gt, img_lq, voxel]
        
        img_results = augment(all_imgs, self.opt['use_hflip'], self.opt['use_rot'])

        # print('debug: len of img_results: {}'.format(len(img_results)))
        img_results = img2tensor(img_results) # hwc -> chw
        img_gt = img_results[0]
        img_lq = img_results[1]
        voxel = img_results[2]
        ## Norm voxel
        voxel = voxel_norm(voxel)

        # print('DEBUG: lq.shape:{}'.format(img_lqs.shape))
        # print('DEBUG: gt.shape:{}'.format(img_gts.shape))
        # print('DEBUG: voxel.shape:{}'.format(voxels.shape))

        # img_lqs: (t, c, h, w)
        # img_lqs: (3*2+(m-1)*2, h, w) if return_deblur_voxel

        # print('blur0_path:{}'.format(blur0_path))
        seq = image_path.split(f'{self.split}/')[1].split('/')[0]
        origin_index = os.path.basename(image_path).split('.')[0]


        return {'lq': img_lq, 'gt': img_gt, 'voxel': voxel, 'seq': seq, 'origin_index': origin_index}

    def __len__(self):
        return len(self.allBlurPath)


class GoProBidirEventRecurrentDataset(data.Dataset):
    """For reveresed voxel 
    
    Args:
        opt (dict): Config for train dataset. It contains the following keys:
            dataroot (str): Data root path.
            io_backend (dict): IO backend type and other kwarg.
            num_end_interpolation (int): Number of sharp frames to reconstruct in each blurry image.
            num_inter_interpolation (int): Number of sharp frames to interpolate between two blurry images.
            phase (str): 'train' or 'test'

            gt_size (int): Cropped patched size for gt patches.
            random_reverse (bool): Random reverse input frames.
            use_hflip (bool): Use horizontal flips.
            use_rot (bool): Use rotation (use vertical flip and transposing h
                and w for implementation).

            scale (bool): Scale, which will be added automatically.
    """

    def __init__(self, opt):
        super(GoProBidirEventRecurrentDataset, self).__init__()
        self.opt = opt
        self.dataroot = Path(opt['dataroot'])
        self.m = opt['num_end_interpolation']
        self.n = opt['num_inter_interpolation']
        self.num_input_blur = 2
        self.num_input_gt = 2*self.m + self.n
        self.num_bins = 2*self.m + self.n + 1
        self.split = 'train' if opt['phase']=='train' else 'test' # train or test
        self.norm_voxel = opt.get('norm_voxel', True)
        self.one_voxel_flg = opt.get('one_voxel_flag', True)
        self.return_deblur_voxel = opt.get('return_deblur_voxel', False)
        self.return_deblur_voxel = self.return_deblur_voxel and self.one_voxel_flg


        ## the sequence names
        train_video_list = [
            'GOPR0372_07_00', 'GOPR0374_11_01', 'GOPR0378_13_00', 'GOPR0384_11_01', 'GOPR0384_11_04', 'GOPR0477_11_00', 'GOPR0868_11_02', 'GOPR0884_11_00', 
            'GOPR0372_07_01', 'GOPR0374_11_02', 'GOPR0379_11_00', 'GOPR0384_11_02', 'GOPR0385_11_00', 'GOPR0857_11_00', 'GOPR0871_11_01', 'GOPR0374_11_00', 
            'GOPR0374_11_03', 'GOPR0380_11_00', 'GOPR0384_11_03', 'GOPR0386_11_00', 'GOPR0868_11_01', 'GOPR0881_11_00']
        test_video_list = [
            'GOPR0384_11_00', 'GOPR0385_11_01', 'GOPR0410_11_00', 'GOPR0862_11_00', 'GOPR0869_11_00', 'GOPR0881_11_01', 'GOPR0384_11_05', 'GOPR0396_11_00', 
            'GOPR0854_11_00', 'GOPR0868_11_00', 'GOPR0871_11_00']
        video_list = train_video_list if self.split=='train' else test_video_list

        self.blurPairsPath = [] 
        self.gtSeqsPath = [] # list of lists of sequences of gt sharp frames
        self.eventSeqsPath = [] # list of lists of event frames

        ### Formate file lists
        for video in video_list:
            blur_frames   = sorted(recursive_glob(rootdir=os.path.join(self.dataroot, self.split, video, 'blur'), suffix='.png'))  # blur frames in one video
            gt_frames  = sorted(recursive_glob(rootdir=os.path.join(self.dataroot, self.split, video, 'gt'), suffix='.png'))  # all sharp frames in one video
            event_frames = sorted(recursive_glob(rootdir=os.path.join(self.dataroot, self.split+'_event', video), suffix='.npz'))  # all sharp frames in one video
            n_sets = len(blur_frames)-1 # number of all the blur image pairs
            
            # Blur
            blurInputs = [blur_frames[i:i+2] for i in range(1,n_sets)] # abandon the firt and last image
            blurInputs = [[os.path.join(self.dataroot, self.split, video, 'blur',f) for f in group] for group in blurInputs] # GOPR0372_07_00/xxx.png ...
            # print("[DEBUG]: videoInputs:{}".format(videoInputs))
            gtInputs = [gt_frames[i*(self.m+self.n): (i+1)*(self.m+self.n) + self.m] for i in range(1,n_sets)]
            gtInputs = [[os.path.join(self.dataroot, self.split, video, 'gt', f) for f in group] for group in gtInputs] # GOPR0372_07_00/xxx.png ...
            # events
            eventInputs = [event_frames[i*(self.m+self.n) -1: (i+2)*(self.m+self.n) - 1] for i in range(1,n_sets)]
            eventInputs = [[os.path.join(self.dataroot, self.split+'_event', video, f) for f in group] for group in eventInputs] # GOPR0372_07_00/xxx.png ...

            self.blurPairsPath.extend(blurInputs)# list of lists of paired blur input, e.g.:
            # [['GOPR0372_07_00/blur/000328.png', 'GOPR0372_07_00/blur/000342.png'],
            #  ['GOPR0372_07_00/blur/000342.png', 'GOPR0372_07_00/blur/000356.png']]
            self.gtSeqsPath.extend(gtInputs)
            self.eventSeqsPath.extend(eventInputs)

            assert len(self.blurPairsPath)==len(self.gtSeqsPath)==len(self.eventSeqsPath), 'The number of blur/sharp/interpo: {}/{}/{} does not match.'
        
        # print('DEBUG: length of blurPairsPath:{}'.format(len(self.blurPairsPath)))

        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']

        # temporal augmentation configs
        self.random_reverse = opt.get('random_reverse', False)
        logger = get_root_logger()
        logger.info(f'Temporal augmentation: random reverse is {self.random_reverse}.')


    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)
        scale = self.opt['scale']
        gt_size = self.opt['gt_size']

        image_paths = self.blurPairsPath[index]
        gt_paths = self.gtSeqsPath[index]
        event_paths = self.eventSeqsPath[index]

        # print('DEBUG image paths:{}'.format(image_paths))
        # print('DEBUG gt_paths :{}'.format(gt_paths))
        # print('DEBUG event_paths:{}'.format(event_paths))

        # random reverse
        if self.random_reverse and random.random() < 0.5:
            image_paths.reverse()
            gt_paths.reverse()
            # TODO: reverse event

        ## Read blur and gt sharps
        img_lqs = []
        img_gts = []
        for image_path in image_paths:
            # get LQ
            img_bytes = self.file_client.get(image_path)  # 'lq'
            img_lq = imfrombytes(img_bytes, float32=True)
            img_lqs.append(img_lq)

        for gt_path in gt_paths:
            # get GT
            img_bytes = self.file_client.get(gt_path)    # 'gt'
            img_gt = imfrombytes(img_bytes, float32=True)
            img_gts.append(img_gt)

        h_lq, w_lq, _ = img_lqs[0].shape
        ## Read event and convert to voxel grid:
        events = [np.load(event_path) for event_path in event_paths]
        # npz -> ndarray
        voxels = []
        # reversed_voxels = []

        if self.one_voxel_flg:
            all_quad_event_array = np.zeros((0,4)).astype(np.float32)

            for event in events:
                x = event['x'].astype(np.float32)[:,np.newaxis]
                y = event['y'].astype(np.float32)[:,np.newaxis]
                t = event['timestamp'].astype(np.float32)[:,np.newaxis]
                p = event['polarity'].astype(np.float32)[:,np.newaxis]
                this_quad_event_array = np.concatenate((t,x,y,p),axis=1) # N,4
                all_quad_event_array = np.concatenate((all_quad_event_array, this_quad_event_array), axis=0)
            reversed_event_array = event_reverse(all_quad_event_array)
            voxel = events_to_voxel_grid(all_quad_event_array, num_bins=self.num_bins, width=w_lq, height=h_lq, return_format='HWC')
            reversed_voxel = events_to_voxel_grid(reversed_event_array, num_bins=self.num_bins, width=w_lq, height=h_lq, return_format='HWC')
            # Voxel Norm
            # if self.norm_voxel:
            #     voxel = voxel_norm(voxel)
            voxels.append(voxel) # len=2, shape:h,w,num_bins
            voxels.append(reversed_voxel)  
            # num_bins,h,w

        else:
            for i in range(len(events)):
                x = events[i]['x'].astype(np.float32)[:,np.newaxis]
                y = events[i]['y'].astype(np.float32)[:,np.newaxis]
                t = events[i]['timestamp'].astype(np.float32)[:,np.newaxis]
                p = events[i]['polarity'].astype(np.float32)[:,np.newaxis]
                this_quad_event_array = np.concatenate((t,x,y,p),axis=1) # N,4
                if i == 0:
                    last_quad_event_array = this_quad_event_array
                elif i >=1:
                    two_quad_event_array = np.concatenate((last_quad_event_array, this_quad_event_array), axis=0)
                    sub_voxel = events_to_voxel_grid(two_quad_event_array, num_bins=2,width=w_lq, height=h_lq, return_format='HWC')
                    voxels.append(sub_voxel)
                    last_quad_event_array = this_quad_event_array
                # len=2m+n+1, each with shape: h,w,2
        # Voxel: list of chw or hwc
        # randomly crop
        # voxel shape: h,w,c
        if gt_size is not None:
            img_gts, img_lqs, voxels = triple_random_crop(img_gts, img_lqs, voxels, gt_size, scale, gt_paths[0])

        # augmentation - flip, rotate
        num_lq = len(img_lqs) if isinstance(img_lqs, list) else 1
        num_gt = len(img_gts) if isinstance(img_gts, list) else 1
        num_voxel = len(voxels) if isinstance(voxels, list) else 1

        img_lqs.extend(img_gts)
        img_lqs.extend(voxels) if isinstance(voxels,list) else img_lqs.append(voxels) # [img_lqs, img_gts, voxels]

        img_results = augment(img_lqs, self.opt['use_hflip'], self.opt['use_rot'])

        img_results = img2tensor(img_results) # hwc -> chw
        img_lqs = torch.stack(img_results[:num_lq], dim=0) # t,c,h,w
        img_gts = torch.stack(img_results[num_lq:num_lq+num_gt], dim=0)
        voxels_list = img_results[num_lq+num_gt:]
        ## Norm voxel
        if self.norm_voxel:
            for voxel in voxels_list: # 2: voxel and reversed voxel
                voxel = voxel_norm(voxel)

        # make deblur voxel
        if self.return_deblur_voxel:
            left_deblur_voxel = voxels_list[0][1:self.m, :, :]
            right_deblur_voxel = voxels_list[0][self.m+2 + self.n:, :, :]
            # print('DEBUG: Total_voxel.shape:{}'.format(voxels_list[0].shape))
            # print('DEBUG: left_deblur_voxel.shape:{}'.format(left_deblur_voxel.shape))
            # print('DEBUG: right_deblur_voxel.shape:{}'.format(right_deblur_voxel.shape))
            # left_deblur_voxel = left_deblur_voxel.unsqueeze(0) # 1,c,h,w
            # right_deblur_voxel = right_deblur_voxel.unsqueeze(0)
            left_lq = img_lqs[0,:,:,:]
            right_lq = img_lqs[1,:,:,:]
            img_lqs = torch.cat((left_lq, left_deblur_voxel, right_lq, right_deblur_voxel), dim=0) # c,h,w

        voxels = torch.stack([voxels_list[0]], dim=0) # t,c,h,w
        reversed_voxels = torch.stack([voxels_list[1]], dim=0) # t,c,h,w


        # reshape of the voxel tensor   1, num_bins, h, w -> t, 2, h, w
        if self.one_voxel_flg:
            voxels = voxels.squeeze(0)
            all_voxel = []
            for i in range(voxels.shape[0]-1):
                sub_voxel = voxels[i:i+2, :, :]
                all_voxel.append(sub_voxel)
            voxels = torch.stack(all_voxel, dim=0)

            reversed_voxels = reversed_voxels.squeeze(0)
            all_reversed_voxel = []
            for i in range(reversed_voxels.shape[0]-1):
                sub_voxel = reversed_voxels[i:i+2, :, :]
                all_reversed_voxel.append(sub_voxel)
            reversed_voxels = torch.stack(all_reversed_voxel, dim=0)

        voxels = torch.cat((voxels, reversed_voxels), dim=0)# t, 2, h, w
        # print('DEBUG: lq.shape:{}'.format(img_lqs.shape))
        # print('DEBUG: gt.shape:{}'.format(img_gts.shape))
        # print('DEBUG: voxel.shape:{}'.format(voxels.shape))

        # img_lqs: (t, c, h, w)
        # img_lqs: (3*2+(m-1)*2, h, w) if return_deblur_voxel

        # img_gts: (t, c, h, w)
        # voxels: (t, num_bins (2), h, w)
        if self.split == 'train':
            return {'lq': img_lqs, 'gt': img_gts, 'voxel': voxels}
        else:
            blur0_path = image_paths[0]
            # print('blur0_path:{}'.format(blur0_path))
            seq = blur0_path.split('test/')[1].split('/')[0]
            origin_index = os.path.basename(blur0_path).split('.')[0]
            # print("DEBUG:0 seq:" + f'{seq}')
            # seq = seq[0]
            # print("DEBUG:1 seq:" + f'{seq}')
            return {'lq': img_lqs, 'gt': img_gts, 'voxel': voxels, 'seq': seq, 'origin_index': origin_index}

    def __len__(self):
        return len(self.blurPairsPath)


# def event_reverse(xs,ys,ts,ps):
#     """
#     Reverse temporal direction of the event stream.
#     Polarities of the events reversed.
#                       (-)       (+)
#     --------|----------|---------|------------|----> time
#        t_start        t_1       t_2        t_end

#                       (+)       (-)
#     --------|----------|---------|------------|----> time
#             0    (t_end-t_2) (t_end-t_1) (t_end-t_start)
#     """
#     xs = xs[::-1] # reverse x
#     ys = ys[::-1] # reverse y
#     t_max = ts[-1]
#     ts = t_max-ts
#     ts = ts[::-1] # reverse t
#     ps = -ps
#     ps = ps[::-1] # reverse p

#     return xs, ys, ts, ps

def event_reverse(input_event):
    """
    event: t,x,y,p

    Reverse temporal direction of the event stream.
    Polarities of the events reversed.
                      (-)       (+)
    --------|----------|---------|------------|----> time
       t_start        t_1       t_2        t_end

                      (+)       (-)
    --------|----------|---------|------------|----> time
            0    (t_end-t_2) (t_end-t_1) (t_end-t_start)
    """
    event = input_event
    event[:,1] = event[:,1][::-1] # reverse x
    event[:,2] = event[:,2][::-1] # reverse y
    t_max = event[:,0][-1]
    event[:,0] = t_max-event[:,0]
    event[:,0] = event[:,0][::-1] # reverse t
    event[:,3] = -event[:,3]
    event[:,3] = event[:,3][::-1] # reverse p

    return event