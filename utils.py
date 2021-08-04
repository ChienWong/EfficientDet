import numpy as np
from tensorflow import keras
import tensorflow as tf
import cv2

class AnchorParameters:
    """
    The parameters that define how anchors are generated.
    Args
        sizes : List of sizes to use. Each size corresponds to one feature level.
        strides : List of strides to use. Each stride correspond to one feature level.
        ratios : List of ratios to use per location in a feature map.
        scales : List of scales/area to use per location in a feature map.
    """

    def __init__(self, sizes=(32, 64, 128, 256, 512),
                 strides=(8, 16, 32, 64, 128),
                 ratios=(1, 0.5, 2),
                 scales=(2 ** 0, 2 ** (1. / 3.), 2 ** (2. / 3.))):
        self.sizes = sizes
        self.strides = strides
        self.ratios = np.array(ratios, dtype=keras.backend.floatx())
        self.scales = np.array(scales, dtype=keras.backend.floatx())

    def num_anchors(self):
        return len(self.ratios) * len(self.scales)

"""
The default anchor parameters.
"""
AnchorParameters.default = AnchorParameters(
    sizes=[32, 64, 128, 256, 512],
    strides=[8, 16, 32, 64, 128],
    # ratio=h/w
    ratios=np.array([1, 0.5, 2], keras.backend.floatx()),
    scales=np.array([2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)], keras.backend.floatx()),
)

def guess_shapes(image_shape, pyramid_levels):
    """
    Guess shapes based on pyramid levels.
    Args
         image_shape: The shape of the image.
         pyramid_levels: A list of what pyramid levels are used.
    Returns
        A list of image shapes at each pyramid level.
    """
    image_shape = np.array(image_shape[:2])
    image_shapes = [(image_shape + 2 ** x - 1) // (2 ** x) for x in pyramid_levels]
    return image_shapes

def generate_anchors(base_size=16, ratios=None, scales=None):
    """
    Generate anchor (reference) windows by enumerating aspect ratios X scales w.r.t. a reference window.
    Args:
        base_size:
        ratios:
        scales:
    Returns:
    """
    if ratios is None:
        ratios = AnchorParameters.default.ratios

    if scales is None:
        scales = AnchorParameters.default.scales

    num_anchors = len(ratios) * len(scales)

    # initialize output anchors
    anchors = np.zeros((num_anchors, 4))

    anchors[:, 2:] = base_size * np.tile(np.repeat(scales, len(ratios))[None], (2, 1)).T

    areas = anchors[:, 2] * anchors[:, 3]

    # correct for ratios
    anchors[:, 2] = np.sqrt(areas / np.tile(ratios, len(scales)))
    anchors[:, 3] = anchors[:, 2] * np.tile(ratios, len(scales))

    anchors[:, 0::2] -= np.tile(anchors[:, 2] * 0.5, (2, 1)).T
    anchors[:, 1::2] -= np.tile(anchors[:, 3] * 0.5, (2, 1)).T

    return anchors

def shift(feature_map_shape, stride, anchors):
    """
    Produce shifted anchors based on shape of the map and stride size.
    Args
        feature_map_shape : Shape to shift the anchors over.
        stride : Stride to shift the anchors with over the shape.
        anchors: The anchors to apply at each location.
    """

    # create a grid starting from half stride from the top left corner
    shift_x = (np.arange(0, feature_map_shape[1]) + 0.5) * stride
    shift_y = (np.arange(0, feature_map_shape[0]) + 0.5) * stride

    shift_x, shift_y = np.meshgrid(shift_x, shift_y)

    shifts = np.vstack((
        shift_x.ravel(), shift_y.ravel(),
        shift_x.ravel(), shift_y.ravel()
    )).transpose()

    A = anchors.shape[0]
    K = shifts.shape[0]
    all_anchors = (anchors.reshape((1, A, 4)) + shifts.reshape((1, K, 4)).transpose((1, 0, 2)))
    all_anchors = all_anchors.reshape((K * A, 4))

    return all_anchors

def anchors_for_shape(
        image_shape,
        pyramid_levels=None,
        anchor_params=None,
        shapes_callback=None,
):
    """
    Generators anchors for a given shape.
    Args
        image_shape: The shape of the image.
        pyramid_levels: List of ints representing which pyramids to use (defaults to [3, 4, 5, 6, 7]).
        anchor_params: Struct containing anchor parameters. If None, default values are used.
        shapes_callback: Function to call for getting the shape of the image at different pyramid levels.
    Returns
        np.array of shape (N, 4) containing the (x1, y1, x2, y2) coordinates for the anchors.
    """

    if pyramid_levels is None:
        pyramid_levels = [3, 4, 5, 6, 7]

    if anchor_params is None:
        anchor_params = AnchorParameters.default

    if shapes_callback is None:
        shapes_callback = guess_shapes
    feature_map_shapes = shapes_callback(image_shape, pyramid_levels)

    # compute anchors over all pyramid levels
    all_anchors = np.zeros((0, 4), dtype=np.float32)
    for idx, p in enumerate(pyramid_levels):
        anchors = generate_anchors(
            base_size=anchor_params.sizes[idx],
            ratios=anchor_params.ratios,
            scales=anchor_params.scales
        )
        shifted_anchors = shift(feature_map_shapes[idx], anchor_params.strides[idx], anchors)
        all_anchors = np.append(all_anchors, shifted_anchors, axis=0)

    return all_anchors.astype(np.float32)

# for all anchor box apply predict offset
def bbox_transform_inv(boxes, deltas, image_shape, scale_factors=None):
    cxa = (boxes[..., 0] + boxes[..., 2]) / 2
    cya = (boxes[..., 1] + boxes[..., 3]) / 2
    wa = boxes[..., 2] - boxes[..., 0]
    ha = boxes[..., 3] - boxes[..., 1]
    ty, tx, th, tw = deltas[..., 0], deltas[..., 1], deltas[..., 2], deltas[..., 3]
    if scale_factors:
        ty *= scale_factors[0]
        tx *= scale_factors[1]
        th *= scale_factors[2]
        tw *= scale_factors[3]
    w = tf.exp(tw) * wa
    h = tf.exp(th) * ha
    cy = ty * ha + cya
    cx = tx * wa + cxa
    ymin = cy - h / 2.
    xmin = cx - w / 2.
    ymax = cy + h / 2.
    xmax = cx + w / 2.
    height = image_shape[0]
    width = image_shape[1]
    tf.clip_by_value(xmin, 0, width-1)
    tf.clip_by_value(ymin, 0, height-1)
    tf.clip_by_value(xmax, 0, width-1)
    tf.clip_by_value(ymax, 0, height-1)
    return tf.stack([xmin, ymin, xmax, ymax], axis=-1)

def filterDetections(boxes,classification,detect_quadrangle=False,alphas=None,ratios=None,nms=True,
                    score_threshold=0.01,max_detections=100,nms_threshold=0.5,class_specific_filter=True):

    def _filterDetections(boxes,classification,alphas=None,ratios=None,nms=True,
                    score_threshold=0.01,max_detections=100,nms_threshold=0.5,class_specific_filter=True):
        """
        Filter detections using the boxes and classification values.
        Args
            boxes: Tensor of shape (num_boxes, 4) containing the boxes in (x1, y1, x2, y2) format.
            classification: Tensor of shape (num_boxes, num_classes) containing the classification scores.
            other: List of tensors of shape (num_boxes, ...) to filter along with the boxes and classification scores.
            class_specific_filter: Whether to perform filtering per class, or take the best scoring class and filter those.
            nms: Flag to enable/disable non maximum suppression.
            score_threshold: Threshold used to prefilter the boxes with.
            max_detections: Maximum number of detections to keep.
            nms_threshold: Threshold for the IoU value to determine when a box should be suppressed.
        Returns
            A list of [boxes, scores, labels, other[0], other[1], ...].
            boxes is shaped (max_detections, 4) and contains the (x1, y1, x2, y2) of the non-suppressed boxes.
            scores is shaped (max_detections,) and contains the scores of the predicted class.
            labels is shaped (max_detections,) and contains the predicted label.
            other[i] is shaped (max_detections, ...) and contains the filtered other[i] data.
            In case there are less than max_detections detections, the tensors are padded with -1's.
        """

        def _filter_detections(scores_, labels_):
            # threshold based on score
            # (num_score_keeps, 1)
            indices_ = tf.where(keras.backend.greater(scores_, score_threshold))

            if nms:
                # (num_score_keeps, 4)
                filtered_boxes = tf.gather_nd(boxes, indices_)
                # In [4]: scores = np.array([0.1, 0.5, 0.4, 0.2, 0.7, 0.2])
                # In [5]: tf.greater(scores, 0.4)
                # Out[5]: <tf.Tensor: id=2, shape=(6,), dtype=bool, numpy=array([False,  True, False, False,  True, False])>
                # In [6]: tf.where(tf.greater(scores, 0.4))
                # Out[6]:
                # <tf.Tensor: id=7, shape=(2, 1), dtype=int64, numpy=
                # array([[1],
                #        [4]])>
                #
                # In [7]: tf.gather(scores, tf.where(tf.greater(scores, 0.4)))
                # Out[7]:
                # <tf.Tensor: id=15, shape=(2, 1), dtype=float64, numpy=
                # array([[0.5],
                #        [0.7]])>
                filtered_scores = keras.backend.gather(scores_, indices_)[:, 0]

                # perform NMS
                # filtered_boxes = tf.concat([filtered_boxes[..., 1:2], filtered_boxes[..., 0:1],
                #                             filtered_boxes[..., 3:4], filtered_boxes[..., 2:3]], axis=-1)
                nms_indices = tf.image.non_max_suppression(filtered_boxes, filtered_scores, max_output_size=max_detections,
                                                        iou_threshold=nms_threshold)

                # filter indices based on NMS
                # (num_score_nms_keeps, 1)
                indices_ = keras.backend.gather(indices_, nms_indices)

            # add indices to list of all indices
            # (num_score_nms_keeps, )
            labels_ = tf.gather_nd(labels_, indices_)
            # (num_score_nms_keeps, 2)
            indices_ = keras.backend.stack([indices_[:, 0], labels_], axis=1)

            return indices_

        if class_specific_filter:
            all_indices = []
            # perform per class filtering
            for c in range(int(classification.shape[1])):
                scores = classification[:, c]
                labels = c * tf.ones((keras.backend.shape(scores)[0],), dtype='int64')
                all_indices.append(_filter_detections(scores, labels))

            # concatenate indices to single tensor
            # (concatenated_num_score_nms_keeps, 2)
            indices = keras.backend.concatenate(all_indices, axis=0)
        else:
            scores = keras.backend.max(classification, axis=1)
            labels = keras.backend.argmax(classification, axis=1)
            indices = _filter_detections(scores, labels)

        # select top k
        scores = tf.gather_nd(classification, indices)
        labels = indices[:, 1]
        scores, top_indices = tf.nn.top_k(scores, k=keras.backend.minimum(max_detections, keras.backend.shape(scores)[0]))

        # filter input using the final set of indices
        indices = keras.backend.gather(indices[:, 0], top_indices)
        boxes = keras.backend.gather(boxes, indices)
        labels = keras.backend.gather(labels, top_indices)

        # zero pad the outputs
        pad_size = keras.backend.maximum(0, max_detections - keras.backend.shape(scores)[0])
        boxes = tf.pad(boxes, [[0, pad_size], [0, 0]], constant_values=-1)
        scores = tf.pad(scores, [[0, pad_size]], constant_values=-1)
        labels = tf.pad(labels, [[0, pad_size]], constant_values=-1)
        labels = keras.backend.cast(labels, 'int32')

        # set shapes, since we know what they are
        boxes.set_shape([max_detections, 4])
        scores.set_shape([max_detections])
        labels.set_shape([max_detections])

        if detect_quadrangle:
            alphas = keras.backend.gather(alphas, indices)
            ratios = keras.backend.gather(ratios, indices)
            alphas = tf.pad(alphas, [[0, pad_size], [0, 0]], constant_values=-1)
            ratios = tf.pad(ratios, [[0, pad_size]], constant_values=-1)
            alphas.set_shape([max_detections, 4])
            ratios.set_shape([max_detections])
            return [boxes, scores, alphas, ratios, labels]
        else:
            return [boxes, scores, labels]

    if detect_quadrangle:
        outputs = tf.map_fn(
            _filterDetections,
            elems=[boxes, classification, alphas, ratios],
            dtype=['float32', 'float32', 'float32', 'float32', 'int32'],
            parallel_iterations=16)
    else:
        outputs = tf.map_fn(
            _filterDetections,
            elems=[boxes, classification],
            dtype=['float32', 'float32', 'int32'],
            parallel_iterations=16)
    return outputs

def compute_gt_annotations(anchors,annotations,negative_overlap=0.4,positive_overlap=0.5):
    """
    Obtain indices of gt annotations with the greatest overlap.
    Args
        anchors: np.array of annotations of shape (N, 4) for (x1, y1, x2, y2).
        annotations: np.array of shape (K, 5) for (x1, y1, x2, y2, label).
        negative_overlap: IoU overlap for negative anchors (all anchors with overlap < negative_overlap are negative).
        positive_overlap: IoU overlap or positive anchors (all anchors with overlap > positive_overlap are positive).
    Returns
        positive_indices: indices of positive anchors, (N, )
        ignore_indices: indices of ignored anchors, (N, )
        argmax_overlaps_inds: ordered overlaps indices, (N, )
    """
    # (N, K)
    overlaps = compute_overlap(anchors.astype(np.float64), annotations.astype(np.float64))
    # (N, )
    argmax_overlaps_inds = np.argmax(overlaps, axis=1)
    # (N, )
    max_overlaps = overlaps[np.arange(overlaps.shape[0]), argmax_overlaps_inds]

    # assign "dont care" labels
    # (N, )
    positive_indices = max_overlaps >= positive_overlap

    # adam: in case of there are gt boxes has no matched positive anchors
    # nonzero_inds = np.nonzero(overlaps == np.max(overlaps, axis=0))
    # positive_indices[nonzero_inds[0]] = 1

    # (N, )
    ignore_indices = (max_overlaps > negative_overlap) & ~positive_indices

    return positive_indices, ignore_indices, argmax_overlaps_inds

def anchor_targets_bbox(anchors,image,annotations,num_classes,
        negative_overlap=0.4,positive_overlap=0.5,detect_quadrangle=False):
    """
    Generate anchor targets for bbox detection.
    Args
        anchors: np.array of annotations of shape (N, 4) for (x1, y1, x2, y2).
        image:images.
        annotations: annotations (np.array of shape (N, 5) for (x1, y1, x2, y2, label)).
        num_classes: Number of classes to predict.
        negative_overlap: IoU overlap for negative anchors (all anchors with overlap < negative_overlap are negative).
        positive_overlap: IoU overlap or positive anchors (all anchors with overlap > positive_overlap are positive).
    Returns
        labels: contains labels & anchor states (np.array of shape (N, num_classes + 1),
                      where N is the number of anchors for an image and the last column defines the anchor state
                      (-1 for ignore, 0 for bg, 1 for fg).
        regression: batch that contains bounding-box regression targets for an image & anchor states
                      (np.array of shape (N, 4 + 1), where N is the number of anchors for an image,
                      the first 4 columns define regression targets for (x1, y1, x2, y2) and the last column defines
                      anchor states (-1 for ignore, 0 for bg, 1 for fg).
    """
    if detect_quadrangle:
        regression = np.zeros((anchors.shape[0], 9 + 1), dtype=np.float32)
    else:
        regression = np.zeros((anchors.shape[0], 4 + 1), dtype=np.float32)
    labels = np.zeros((anchors.shape[0], num_classes + 1), dtype=np.float32)

    # compute labels and regression targets
    if annotations.shape[0]:
        # obtain indices of gt annotations with the greatest overlap
        # argmax_overlaps_inds: id of ground truth box has greatest overlap with anchor
        # (N, ), (N, ), (N, ) N is num_anchors
        positive_indices, ignore_indices, argmax_overlaps_inds = compute_gt_annotations(anchors,
                                                                    annotations[:,0:4],negative_overlap,positive_overlap)
        labels[ignore_indices, -1] = -1
        labels[positive_indices, -1] = 1

        regression[ignore_indices, -1] = -1
        regression[positive_indices, -1] = 1

        # compute target class labels
        labels[positive_indices, annotations[:,4][argmax_overlaps_inds[positive_indices]].astype(int)] = 1

        regression[:, :4] = bbox_transform(anchors, annotations[:,0:4][argmax_overlaps_inds, :])
        # if detect_quadrangle:
            # regression[:, 4:8] = annotations['alphas'][argmax_overlaps_inds, :]
            # regression[:, 8] = annotations['ratios'][argmax_overlaps_inds]

    # ignore anchors outside of image
    if image.shape:
        anchors_centers = np.vstack([(anchors[:, 0] + anchors[:, 2]) / 2, (anchors[:, 1] + anchors[:, 3]) / 2]).T
        indices = np.logical_or(anchors_centers[:, 0] >= image.shape[1], anchors_centers[:, 1] >= image.shape[0])

        labels[indices, -1] = -1
        regression[indices, -1] = -1

    return labels, regression

def compute_overlap(boxes,query_boxes):
    """
    Args
        a: (N, 4) ndarray of float
        b: (K, 4) ndarray of float
    Returns
        overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
    N = boxes.shape[0]
    K = query_boxes.shape[0]
    overlaps = np.zeros((N, K), dtype=np.float64)
    for k in range(K):
        box_area = (
            (query_boxes[k, 2] - query_boxes[k, 0] + 1) *
            (query_boxes[k, 3] - query_boxes[k, 1] + 1)
        )
        for n in range(N):
            iw = (
                min(boxes[n, 2], query_boxes[k, 2]) -
                max(boxes[n, 0], query_boxes[k, 0]) + 1
            )
            if iw > 0:
                ih = (
                    min(boxes[n, 3], query_boxes[k, 3]) -
                    max(boxes[n, 1], query_boxes[k, 1]) + 1
                )
                if ih > 0:
                    ua = np.float64(
                        (boxes[n, 2] - boxes[n, 0] + 1) *
                        (boxes[n, 3] - boxes[n, 1] + 1) +
                        box_area - iw * ih
                    )
                    overlaps[n, k] = iw * ih / ua
    return overlaps

def bbox_transform(anchors, gt_boxes, scale_factors=None):
    wa = anchors[:, 2] - anchors[:, 0]
    ha = anchors[:, 3] - anchors[:, 1]
    cxa = anchors[:, 0] + wa / 2.
    cya = anchors[:, 1] + ha / 2.

    w = gt_boxes[:, 2] - gt_boxes[:, 0]
    h = gt_boxes[:, 3] - gt_boxes[:, 1]
    cx = gt_boxes[:, 0] + w / 2.
    cy = gt_boxes[:, 1] + h / 2.
    # Avoid NaN in division and log below.
    ha += 1e-7
    wa += 1e-7
    h += 1e-7
    w += 1e-7
    tx = (cx - cxa) / wa
    ty = (cy - cya) / ha
    tw = np.log(w / wa)
    th = np.log(h / ha)
    if scale_factors:
        ty /= scale_factors[0]
        tx /= scale_factors[1]
        th /= scale_factors[2]
        tw /= scale_factors[3]
    targets = np.stack([ty, tx, th, tw], axis=1)
    return targets

def transformImageSize(image,size,bboxs):
    ratio=size[0]/size[1]
    if image.shape[0]/image.shape[1] == ratio:
        bboxs=bboxs*(size[0]/image.shape[0])
        image=cv2.resize(image,size)
        return image,bboxs
    if image.shape[0]/image.shape[1] > ratio:
        pad=int(image.shape[0]/ratio/2)
        image=np.pad(image,((0,0),(pad,pad),(0,0)))
        bboxs[:,1]=bboxs[:,1]+pad
        bboxs=bboxs*(size[0]/image.shape[0])
        image=cv2.resize(image,size)
        return image,bboxs
    if image.shape[0]/image.shape[1] < ratio:
        pad=int(image.shape[1]*ratio/2)
        image=np.pad(image,((pad,pad),(0,0),(0,0)))
        bboxs[:,0]=bboxs[:,0]+pad
        bboxs=bboxs*(size[0]/image.shape[0])
        image=cv2.resize(image,size)
        return image,bboxs