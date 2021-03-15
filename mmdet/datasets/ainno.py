
import os
import os.path as osp
import numpy as np
import xml.etree.ElementTree as ET
from collections import OrderedDict

from mmcv.utils import print_log

from mmdet.core import eval_map, eval_recalls, eval_map_single_proc
from .builder import DATASETS
from .xml_style import XMLDataset

# xml_dir = '/root/public02/manuag/zhangshuai/data/drink/Annotations'
# uuids = {}
# max_id = 0
# for xml_name in os.listdir(xml_dir):
#     xml_path = osp.join(xml_dir, xml_name)
#     tree = ET.parse(xml_path)
#     root = tree.getroot()
#     bboxes = []
#     labels = []
#     bboxes_ignore = []
#     labels_ignore = []
#
#     for obj in root.findall('object'):
#         name = obj.find('name').text
#         if name not in uuids:
#             uuids[name] = 'class_{}'.format(max_id)
#             max_id += 1
# print(uuids)
# print()

@DATASETS.register_module()
class AinnoDataset(XMLDataset):

    CLASSES = ['class_{}'.format(i) for i in range(26)]


    def __init__(self, **kwargs):
        super(AinnoDataset, self).__init__(**kwargs)
        self.uuids = {'455f3a9e-4419-11eb-90ab-0242cb7ccd7c': 'class_0',
               '7a5299f8-48dd-11eb-bdf9-0242cb7ccd7c': 'class_1',
               '344f49e4-9ae3-11e8-a2d1-02422fc40004': 'class_2',
               'f157016c-4419-11eb-9230-0242cb7ccd7c': 'class_3',
               'c475cd3a-80e7-11e8-94ee-34363bd1db02': 'class_4',
               '6962cc64-27eb-11eb-98f3-0242cb7ccd7c': 'class_5',
               '7fb77bba-83e6-11e8-ad2e-34363bd1db02': 'class_6',
               '2bafb1b8-fa17-11e8-bd57-0242cb7ccd7c': 'class_7',
               '7fb2cc8c-83e6-11e8-90f6-34363bd1db02': 'class_8',
               '7fb763ca-83e6-11e8-9abf-34363bd1db02': 'class_9',
               '7fb76be8-83e6-11e8-aa77-34363bd1db02': 'class_10',
               'cd3de592-d54c-11ea-9621-0242cb7ccd7c': 'class_11',
               '3834b2ca-27cc-11ea-b61d-0242cb7ccd7c': 'class_12',
               'f0336e64-d6bc-11e8-8187-0242cb7ccd7c': 'class_13',
               'ce2a8ae8-d737-11e8-95c0-0242cb7ccd7c': 'class_14',
               '5db4475a-c23b-11e9-8b9e-0242cb7ccd7c': 'class_15',
               'c475eef4-80e7-11e8-adf3-34363bd1db02': 'class_16',
               '0725b0fa-5049-11e9-a657-0242cb7ccd7c': 'class_17',
               '29b2a092-602a-11e9-a238-0242cb7ccd7c': 'class_18',
               'e89c0728-0816-11e9-bf48-0242cb7ccd7c': 'class_19',
               '90bd0686-2fa9-11eb-9e8c-0242cb7ccd7c': 'class_20',
               '27a2fc54-d6be-11e8-8e39-0242cb7ccd7c': 'class_21',
               '332d3fb4-3617-11eb-ac21-0242cb7ccd7c': 'class_22',
               '435f47e8-d736-11e8-9e3c-0242cb7ccd7c': 'class_23',
               '2ebeaf4a-e04b-11ea-aa6c-0242cb7ccd7c': 'class_24',
               '1df8dd62-3617-11eb-b552-0242cb7ccd7c': 'class_25'}


    def load_annotations(self, ann_file):
        """Load annotation from XML style ann_file.

        Args:
            ann_file (str): Path of XML file.

        Returns:
            list[dict]: Annotation info from XML file.
        """

        data_infos = []
        # img_ids = mmcv.list_from_file(ann_file)
        # for img_id in img_ids:
        img_dir = osp.join(self.data_root, 'JPEGImages')
        for img_name in os.listdir(img_dir):
            img_id = img_name[:-4]
            filename = f'JPEGImages/{img_id}.jpg'
            xml_path = osp.join(self.data_root, 'Annotations',
                                f'{img_id}.xml')
            tree = ET.parse(xml_path)
            root = tree.getroot()
            size = root.find('size')
            width = 0
            height = 0
            if size is not None:
                width = int(size.find('width').text)
                height = int(size.find('height').text)
            else:
                img_path = osp.join(self.data_root, 'JPEGImages',
                                    '{}.jpg'.format(img_id))
                img = Image.open(img_path)
                width, height = img.size
            data_infos.append(
                dict(id=img_id, filename=filename, width=width, height=height))
        print('set length: {}'.format(len(data_infos)))
        return data_infos

    def get_ann_info(self, idx):
        """Get annotation from XML file by index.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Annotation info of specified index.
        """

        img_id = self.data_infos[idx]['id']
        xml_path = osp.join(self.data_root, 'Annotations', f'{img_id}.xml')
        tree = ET.parse(xml_path)
        root = tree.getroot()
        bboxes = []
        labels = []
        bboxes_ignore = []
        labels_ignore = []
        uuids = {}
        max_id = 0
        for obj in root.findall('object'):
            name = obj.find('name').text
            if name not in self.uuids:
                continue
            label = self.cat2label[self.uuids[name]]
            # difficult = int(obj.find('difficult').text)
            bnd_box = obj.find('bndbox')
            # TODO: check whether it is necessary to use int
            # Coordinates may be float type
            bbox = [
                int(float(bnd_box.find('xmin').text)),
                int(float(bnd_box.find('ymin').text)),
                int(float(bnd_box.find('xmax').text)),
                int(float(bnd_box.find('ymax').text))
            ]
            ignore = False
            if self.min_size:
                assert not self.test_mode
                w = bbox[2] - bbox[0]
                h = bbox[3] - bbox[1]
                if w < self.min_size or h < self.min_size:
                    ignore = True
            # if ignore or difficult:
            if ignore:
                bboxes_ignore.append(bbox)
                labels_ignore.append(label)
            else:
                bboxes.append(bbox)
                labels.append(label)

        if not bboxes:
            bboxes = np.zeros((0, 4))
            labels = np.zeros((0, ))
        else:
            bboxes = np.array(bboxes, ndmin=2) - 1
            labels = np.array(labels)
        if not bboxes_ignore:
            bboxes_ignore = np.zeros((0, 4))
            labels_ignore = np.zeros((0, ))
        else:
            bboxes_ignore = np.array(bboxes_ignore, ndmin=2) - 1
            labels_ignore = np.array(labels_ignore)
        ann = dict(
            bboxes=bboxes.astype(np.float32),
            labels=labels.astype(np.int64),
            bboxes_ignore=bboxes_ignore.astype(np.float32),
            labels_ignore=labels_ignore.astype(np.int64))
        return ann

    def evaluate(self,
                 results,
                 metric='mAP',
                 logger=None,
                 proposal_nums=(100, 300, 1000),
                 iou_thr=0.5,
                 scale_ranges=None):
        """Evaluate in VOC protocol.

        Args:
            results (list[list | tuple]): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated. Options are
                'mAP', 'recall'.
            logger (logging.Logger | str, optional): Logger used for printing
                related information during evaluation. Default: None.
            proposal_nums (Sequence[int]): Proposal number used for evaluating
                recalls, such as recall@100, recall@1000.
                Default: (100, 300, 1000).
            iou_thr (float | list[float]): IoU threshold. Default: 0.5.
            scale_ranges (list[tuple], optional): Scale ranges for evaluating
                mAP. If not specified, all bounding boxes would be included in
                evaluation. Default: None.

        Returns:
            dict[str, float]: AP/recall metrics.
        """

        if not isinstance(metric, str):
            assert len(metric) == 1
            metric = metric[0]
        allowed_metrics = ['mAP', 'recall']
        if metric not in allowed_metrics:
            raise KeyError(f'metric {metric} is not supported')
        annotations = [self.get_ann_info(i) for i in range(len(self))]
        eval_results = OrderedDict()
        iou_thrs = [iou_thr] if isinstance(iou_thr, float) else iou_thr
        if metric == 'mAP':
            assert isinstance(iou_thrs, list)
            ds_name = self.CLASSES
            mean_aps = []
            for iou_thr in iou_thrs:
                print_log(f'\n{"-" * 15}iou_thr: {iou_thr}{"-" * 15}')
                mean_ap, _ = eval_map_single_proc(
                # mean_ap, _ = eval_map(
                    results,
                    annotations,
                    scale_ranges=None,
                    iou_thr=iou_thr,
                    dataset=ds_name,
                    logger=logger)
                # mean_ap = 0.5
                mean_aps.append(mean_ap)
                eval_results[f'AP{int(iou_thr * 100):02d}'] = round(mean_ap, 3)
            eval_results['mAP'] = sum(mean_aps) / len(mean_aps)
        elif metric == 'recall':
            gt_bboxes = [ann['bboxes'] for ann in annotations]
            recalls = eval_recalls(
                gt_bboxes, results, proposal_nums, iou_thr, logger=logger)
            for i, num in enumerate(proposal_nums):
                for j, iou in enumerate(iou_thr):
                    eval_results[f'recall@{num}@{iou}'] = recalls[i, j]
            if recalls.shape[1] > 1:
                ar = recalls.mean(axis=1)
                for i, num in enumerate(proposal_nums):
                    eval_results[f'AR@{num}'] = ar[i]
        return eval_results
