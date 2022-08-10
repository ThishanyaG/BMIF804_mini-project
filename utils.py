import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt


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
    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(segment_mask)
    print(stats.GetNumberOfLabels())
    centroid = stats.GetCentroid(1)
    return depth, centroid


def pixel_extract(img, point, width):
    length = [int(point[0] - (width * img.GetSpacing()[0])), int(point[0] + (width * img.GetSpacing()[0]))]
    height = [int(point[1] - (width * img.GetSpacing()[1])), int(point[1] + (width * img.GetSpacing()[1]))]
    depth = [int(point[2] - (width * img.GetSpacing()[2])), int(point[2] + (width * img.GetSpacing()[2]))]
    new_img = img[length[0]:length[1], height[0]:height[1], depth[0]:depth[1]]
    plt.imshow(sitk.GetArrayFromImage(img[:,:,point[2]]), cmap='gray')
    plt.show()
    plt.imshow(sitk.GetArrayFromImage(new_img[:, :, point[2]]), cmap='gray')
    plt.show()
