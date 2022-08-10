import SimpleITK as sitk
import utils
import numpy as np
import matplotlib.pyplot as plt

prostate_img = sitk.ReadImage("C:/Users/thish/Documents/804/mini-project/case23_resampled.nii")
prostate_seg = sitk.ReadImage("C:/Users/thish/Documents/804/mini-project/case23_resampled_segmentation.nii")
segment = utils.prostate_segmenter(prostate_img, 550, 280)

writer = sitk.ImageFileWriter()
writer.SetFileName("C:/Users/thish/Documents/804/mini-project/my_segmentation.nrrd")
writer.Execute(segment)

img_overlay = sitk.LabelOverlay(prostate_img, segment)

plt.imshow(sitk.GetArrayFromImage(img_overlay[:,:,35]), cmap='gray')
plt.show()

prostate_dsc = utils.seg_eval_dice(prostate_seg, segment)
print("The DSC is:", prostate_dsc)

print(np.count_nonzero(prostate_seg[:, :, 30]))
utils.get_target_loc(prostate_seg)

