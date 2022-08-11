import SimpleITK as sitk
import utils
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

centroid = utils.get_target_loc(prostate_seg)

plt.figure()
plt.imshow(sitk.GetArrayFromImage(prostate_img[:,:,centroid[2]]), cmap='gray')
plt.plot([centroid[0]],[centroid[1]],'rx')
plt.show()

cube_intensities = utils.pixel_extract(prostate_img, centroid, 6)

plt.boxplot(cube_intensities)
plt.show()

