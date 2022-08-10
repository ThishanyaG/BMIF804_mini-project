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
    print("The LP plane with the largest prostate cross section is at depth", depth)


