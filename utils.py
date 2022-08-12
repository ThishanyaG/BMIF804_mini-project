import SimpleITK as sitk
import numpy as np


def prostate_segmenter(mri, upper_lim, lower_lim):
    thresh_filter = sitk.BinaryThresholdImageFilter()
    thresh_filter.SetLowerThreshold(lower_lim)
    thresh_filter.SetUpperThreshold(upper_lim)
    thresh_filter.SetInsideValue(1)
    thresh_filter.SetOutsideValue(0)
    filtered_img = thresh_filter.Execute(mri)
    return filtered_img


def seg_eval_dice(ground_truth, segment):
    seg = sitk.Cast(segment, sitk.sitkInt8)
    dsc = sitk.LabelOverlapMeasuresImageFilter()
    dsc.Execute(ground_truth, seg)
    return dsc.GetDiceCoefficient()


def get_target_loc(segment_mask):
    pixel = 0
    depth = 0
    for i in range(segment_mask.GetDepth()):
        if np.count_nonzero(segment_mask[:, :, i]) > pixel:
            pixel = np.count_nonzero(segment_mask[:, :, i])
            depth = i
    array = sitk.GetArrayFromImage(segment_mask[:, :, depth])
    loc = np.where(array != 0)
    centroid = (np.mean(loc[0]), np.mean(loc[1]), depth)
    return centroid


def pixel_extract(img, point, width):
    length = [int(point[0] - (width / img.GetSpacing()[0])), int(point[0] + (width / img.GetSpacing()[0]))]
    height = [int(point[1] - (width / img.GetSpacing()[1])), int(point[1] + (width / img.GetSpacing()[1]))]
    depth = [int(point[2] - (width / img.GetSpacing()[2])), int(point[2] + (width / img.GetSpacing()[2]))]
    array_img = sitk.GetArrayFromImage(img)
    new_img = array_img[depth[0]:depth[1], height[0]:height[1], length[0]:length[1]]
    final_img = sitk.GetImageFromArray(new_img)
    return final_img
