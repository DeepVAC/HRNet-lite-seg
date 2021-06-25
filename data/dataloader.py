import os
import pickle
import numpy as np
from deepvac import LOG
from deepvac.datasets import FileLineCvSegDataset

class FileLineCvSegWithMetaInfoDataset(FileLineCvSegDataset):

    def __init__(self, deepvac_config, fileline_path, sample_path_prefix):
        super(FileLineCvSegWithMetaInfoDataset, self).__init__(deepvac_config,fileline_path, ',', sample_path_prefix)
        self.classes = self.config.classes
        self.classWeights = np.ones(self.classes, dtype=np.float32)
        self.normVal = self.config.norm_val
        self.mean = np.zeros(3, dtype=np.float32)
        self.std = np.zeros(3, dtype=np.float32)
        self.cached_data_file = self.config.cached_data_file

    def _compute_class_weights(self, histogram):
        normHist = histogram / np.sum(histogram)
        for i in range(self.classes):
            self.classWeights[i] = 1 / (np.log(self.normVal + normHist[i]))

    def _accumulateMeanStd(self, image_path, label_path):
        img_file = os.path.join(self.sample_path_prefix, image_path.strip())
        label_file = os.path.join(self.sample_path_prefix, label_path.strip())
        label_img = self._buildLabelFromPath(label_file)
        unique_values = np.unique(label_img)

        max_val = max(unique_values)
        min_val = min(unique_values)

        self.max_val_al = max(max_val, self.max_val_al)
        self.min_val_al = min(min_val, self.min_val_al)

        hist = np.histogram(label_img, self.classes)
        self.global_hist += hist[0]

        rgb_img = self._buildSampleFromPath(img_file)
        self.mean[0] += np.mean(rgb_img[:,:,0])
        self.mean[1] += np.mean(rgb_img[:, :, 1])
        self.mean[2] += np.mean(rgb_img[:, :, 2])

        self.std[0] += np.std(rgb_img[:, :, 0])
        self.std[1] += np.std(rgb_img[:, :, 1])
        self.std[2] += np.std(rgb_img[:, :, 2])

        if max_val > (self.classes - 1) or min_val < 0:
            LOG.logE('Some problem with labels. Please check image file: {}. Labels can take value between 0 and number of classes {}.'.format(label_file, self.classes-1), exit=True)


    def _readFile(self):
        self.global_hist = np.zeros(self.classes, dtype=np.float32)

        no_files = 0
        self.min_val_al = 0
        self.max_val_al = 0

        for image_path, label_path in self.samples:
            if no_files % 100 == 0:
                LOG.logI('accumulateMeanStd: {}'.format(no_files))
            self._accumulateMeanStd(image_path, label_path)
            no_files += 1

        self.mean /= no_files
        self.std /= no_files

        self._compute_class_weights(self.global_hist)

    def processData(self):

        print('Processing training data')
        self._readFile()

        print('Pickling data')
        data_dict = dict()

        data_dict['mean'] = self.mean
        data_dict['std'] = self.std
        data_dict['classWeights'] = self.classWeights

        pickle.dump(data_dict, open(self.cached_data_file, "wb"))
        return data_dict

    def __call__(self):
        if os.path.isfile(self.cached_data_file):
            return pickle.load(open(self.cached_data_file, "rb"))

        data = self.processData()
        if data is None:
            LOG.logE('Error while pickling data. Please check.', exit=True)
        LOG.logI('Process train dataset finished.')
        LOG.logI('Your train dataset mean: {}'.format(data['mean']))
        LOG.logI('Your train dataset std: {}'.format(data['std']))
        LOG.logI('Your train dataset classWeights: {}'.format(data['classWeights']))
        return data