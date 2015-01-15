# Authors: Alan Peixinho <alan-peixinho@hotmail.com>
# Giovani Chiachia <giovani.chiachia@gmail.com>
#
# License: BSD

import os
from glob import glob
import numpy as np

import sklearn
import sklearn.cross_validation
import balancedshufflesplit

from base import Dataset
from simplehp.util.util import (get_folders_recursively, load_imgs)

def _get_label_file(file):
    f = os.path.basename(file)
    f = os.path.splitext(f)[0]
    return int(f[:f.find('_')])


class IFTSplitDataset(Dataset):
    """
    Interface for datasets whose evaluation protocol consists on randomly
    splitting the samples.
    """

    def __init__(self, path, img_type, img_shape,
                 hp_nsplits, hp_ntrain, hp_neval, protocol_ntrain, protocol_ntest,
                 bkg_categories, color = False, seed=42):

        self.flatten = not color
        self.path = path
        self.img_type = img_type
        self.img_shape = img_shape
        self.hp_nsplits = hp_nsplits
        self.hp_ntrain = hp_ntrain
        self.protocol_ntrain = protocol_ntrain
        self.protocol_ntest = protocol_ntest
        self.hp_neval = hp_neval
        self.bkg_categories = bkg_categories
        self.rng = np.random.RandomState(seed)
        self.seed = seed

    	assert(self.hp_neval+self.hp_ntrain<1.0)
        assert(self.protocol_ntrain >= self.hp_ntrain + self.hp_neval)
        assert(self.protocol_ntrain + self.protocol_ntest<=1.0)

    	self.data = None
    	self.label = None

    	self.__build_meta()


    def __build_meta(self):
        """
        Retrieve dataset metadata, which, in this case, consists of image paths
        and labels. The latter assumed as the directory name where the image
        files are located.
        """

        folders = np.array(sorted(get_folders_recursively(
                           self.path, self.img_type)))

        all_fnames = glob(os.path.join(self.path, '*'+self.img_type))
        all_labels = map(_get_label_file, all_fnames)


        all_fnames = np.array(all_fnames)
        all_labels = np.array(all_labels)
        all_idxs = np.arange(all_labels.size)
	
	self.all_fnames = all_fnames
	self.all_labels = all_labels
	self.all_idxs = all_idxs
	
	#split out the test set
	splitter = balancedshufflesplit.BalancedShuffleSplit(self.all_labels, 1, train_size = self.hp_ntrain+self.hp_neval, random_state = np.random.RandomState(self.seed))
	self.learn_idxs, self.test_idxs = [(x[0],x[1]) for x in splitter][0]

	print self.learn_idxs
	print self.test_idxs

	#import pdb; pdb.set_trace()

    def __get_imgs(self):

        try:
            return self._imgs
        except AttributeError:
            # -- load all images in memory because dataset is not large
            self._imgs = load_imgs(self.all_fnames,
                                   out_shape=self.img_shape,
                                   dtype='uint8', flatten=self.flatten)

            self._imgs = np.rollaxis(self._imgs, 3, 1)
            self._imgs = np.ascontiguousarray(self._imgs)

	    #import pdb; pdb.set_trace()

            return self._imgs
    imgs = property(__get_imgs)


    def hp_imgs(self):

        return self.imgs[self.learn_idxs]


    def __build_hp_splits(self):
        """
        Randomly split hyperoptimization samples according to the given number
        of train and test samples for the task.
        """
        hp_labels = self.meta['all_labels'][self.meta['hp_idxs']]

        categories = np.unique(hp_labels)
        hp_samples_cat = self.hp_ntrain + self.hp_ntest
        hp_splits = []

        for s in xrange(self.hp_nsplits):

            hp_train_idxs = []
            hp_test_idxs = []

            for cat in categories:
                cat_idxs = np.argwhere(hp_labels==cat)[:,0]
                assert cat_idxs.size == hp_samples_cat
                shuffle = self.rng.permutation(cat_idxs.size)
                hp_train_idxs += sorted(cat_idxs[shuffle[:self.hp_ntrain]])

                if cat not in self.bkg_categories:
                    hp_test_idxs += sorted(cat_idxs[shuffle[self.hp_ntrain:]])

            hp_splits += [{'train': hp_train_idxs, 'test': hp_test_idxs}]

        return hp_splits


    def __get_hp_splits(self):
        try:
            return self._hp_splits
        except AttributeError:
            self._hp_splits = self.__build_hp_splits()
            return self._hp_splits
    hp_splits = property(__get_hp_splits)


    def hp_eval(self, algo, feat_set):

        #import pdb; pdb.set_trace()

        hp_labels = self.all_labels[self.learn_idxs]

        # -- normalize features
        f_mean = feat_set.mean(axis=0)
        f_std = feat_set.std(axis=0, ddof=1)
        f_std[f_std==0.] = 1.

        feat_set -= f_mean
        feat_set /= f_std

	splitter = balancedshufflesplit.BalancedShuffleSplit(self.all_labels[self.learn_idxs], self.hp_nsplits, train_size = self.hp_ntrain, random_state = np.random.RandomState(self.seed))

	splits = [{'train':x[0], 'test':x[1]} for x in splitter]

	#import pdb; pdb.set_trace()

        acc, r_dict = algo(feat_set, hp_labels, splits,
                           bkg_categories=self.bkg_categories)

        return {'loss': 1. - acc}


    def protocol_imgs(self):

        return self.imgs

    def protocol_labels(self):
	return self.all_labels


    def protocol_eval(self, algo, feat_set):

	print 'IFT Protocol eval ...'

        all_labels = self.all_labels

	splitter = sklearn.cross_validation.StratifiedShuffleSplit(self.all_labels[self.test_idxs], self.hp_nsplits, train_size = self.protocol_ntrain - (self.hp_ntrain+self.hp_neval), random_state = np.random.RandomState(self.seed))

        splits = [{'train':np.hstack((self.learn_idxs, self.test_idxs[x[0]])), 'test':self.test_idxs[x[1]]} for x in splitter]

        acc, r_dict = algo(feat_set, all_labels, splits,
                           bkg_categories=self.bkg_categories)

        accs = [r_dict[k]['acc'] for k in r_dict.keys()]

        print 'Acc: ', np.mean(accs), ' +/- ', np.std(accs)

        return {'loss': 1. - acc}


def IFTDataset(path, img_type='pgm', img_shape=None):
	return IFTSplitDataset(path, img_type, img_shape, hp_nsplits=10, hp_ntrain=0.025, hp_neval=0.025, protocol_ntrain = 0.1, protocol_ntest=0.9, bkg_categories=[None,])

def IFTColorDataset(path, img_type='ppm', img_shape=None):
    return IFTSplitDataset(path, img_type, img_shape, hp_nsplits=10, hp_ntrain=0.025, hp_neval=0.025, protocol_ntrain = 0.1, protocol_ntest=0.9, bkg_categories=[None,], color=True)


