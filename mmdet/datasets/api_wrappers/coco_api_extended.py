# Copyright (c) OpenMMLab. All rights reserved.
# This file add snake case alias for coco api

import warnings
import numpy as np

import pycocotools
from pycocotools.coco import COCO as _COCO
from pycocotools.cocoeval import COCOeval as _COCOeval


class COCO(_COCO):
    """This class is almost the same as official pycocotools package.

    It implements some snake case function aliases. So that the COCO class has
    the same interface as LVIS class.
    """

    def __init__(self, annotation_file=None):
        if getattr(pycocotools, '__version__', '0') >= '12.0.2':
            warnings.warn(
                'mmpycocotools is deprecated. Please install official pycocotools by "pip install pycocotools"',  # noqa: E501
                UserWarning)
        super().__init__(annotation_file=annotation_file)
        self.img_ann_map = self.imgToAnns
        self.cat_img_map = self.catToImgs

    def get_ann_ids(self, img_ids=[], cat_ids=[], area_rng=[], iscrowd=None):
        return self.getAnnIds(img_ids, cat_ids, area_rng, iscrowd)

    def get_cat_ids(self, cat_names=[], sup_names=[], cat_ids=[]):
        return self.getCatIds(cat_names, sup_names, cat_ids)

    def get_img_ids(self, img_ids=[], cat_ids=[]):
        return self.getImgIds(img_ids, cat_ids)

    def load_anns(self, ids):
        return self.loadAnns(ids)

    def load_cats(self, ids):
        return self.loadCats(ids)

    def load_imgs(self, ids):
        return self.loadImgs(ids)


# just for the ease of import
class COCOExtendEval(_COCOeval):
    def __init__(self, cocoGt=None, cocoDt=None, iouType='segm'):
        super().__init__(cocoGt, cocoDt, iouType)
    
    def summarize_2(self):
            '''
            Compute and display summary metrics for evaluation results.
            Note this functin can *only* be applied on the default parameter setting
            '''

            print("In method")
            def _summarize(ap=1, iouThr=None, areaRng='all', maxDets=2000):
                p = self.params
                iStr = ' {:<18} {} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} ] = {:0.3f}'
                titleStr = 'Average Precision' if ap == 1 else 'Average Recall'
                typeStr = '(AP)' if ap == 1 else '(AR)'
                iouStr = '{:0.2f}:{:0.2f}'.format(p.iouThrs[0], p.iouThrs[-1]) \
                    if iouThr is None else '{:0.2f}'.format(iouThr)

                aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng]
                mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]
                if ap == 1:
                    # dimension of precision: [TxRxKxAxM]
                    s = self.eval['precision']
                    # IoU
                    if iouThr is not None:
                        t = np.where(iouThr == p.iouThrs)[0]
                        s = s[t]
                    s = s[:, :, :, aind, mind]
                else:
                    # dimension of recall: [TxKxAxM]
                    s = self.eval['recall']
                    if iouThr is not None:
                        t = np.where(iouThr == p.iouThrs)[0]
                        s = s[t]
                    s = s[:, :, aind, mind]
                if len(s[s > -1]) == 0:
                    mean_s = -1
                else:
                    mean_s = np.mean(s[s > -1])
                print(iStr.format(titleStr, typeStr, iouStr, areaRng, maxDets, mean_s))
                return mean_s

            def _summarizeDets():

                stats = np.zeros((12,))
                stats[0] = _summarize(1, maxDets=self.params.maxDets[2])
                stats[1] = _summarize(1, iouThr=.5, maxDets=self.params.maxDets[2])
                stats[2] = _summarize(1, iouThr=.75, maxDets=self.params.maxDets[2])
                stats[3] = _summarize(1, areaRng='small', maxDets=self.params.maxDets[2])
                stats[4] = _summarize(1, areaRng='medium', maxDets=self.params.maxDets[2])
                stats[5] = _summarize(1, areaRng='large', maxDets=self.params.maxDets[2])
                stats[6] = _summarize(0, maxDets=self.params.maxDets[0])
                stats[7] = _summarize(0, maxDets=self.params.maxDets[1])
                stats[8] = _summarize(0, maxDets=self.params.maxDets[2])
                stats[9] = _summarize(0, areaRng='small', maxDets=self.params.maxDets[2])
                stats[10] = _summarize(0, areaRng='medium', maxDets=self.params.maxDets[2])
                stats[11] = _summarize(0, areaRng='large', maxDets=self.params.maxDets[2])
                return stats


            def _summarizeKps():
                stats = np.zeros((10,))
                stats[0] = _summarize(1, maxDets=self.params.maxDets[2])
                stats[1] = _summarize(1, maxDets=self.params.maxDets[2], iouThr=.5)
                stats[2] = _summarize(1, maxDets=self.params.maxDets[2], iouThr=.75)
                stats[3] = _summarize(1, maxDets=self.params.maxDets[2], areaRng='medium')
                stats[4] = _summarize(1, maxDets=self.params.maxDets[2], areaRng='large')
                stats[5] = _summarize(0, maxDets=self.params.maxDets[2])
                stats[6] = _summarize(0, maxDets=self.params.maxDets[2], iouThr=.5)
                stats[7] = _summarize(0, maxDets=self.params.maxDets[2], iouThr=.75)
                stats[8] = _summarize(0, maxDets=self.params.maxDets[2], areaRng='medium')
                stats[9] = _summarize(0, maxDets=self.params.maxDets[2], areaRng='large')
                return stats

            if not self.eval:
                raise Exception('Please run accumulate() first')
            iouType = self.params.iouType
            if iouType == 'segm' or iouType == 'bbox':
                summarize = _summarizeDets
            elif iouType == 'keypoints':
                summarize = _summarizeKps
            self.stats = summarize()

