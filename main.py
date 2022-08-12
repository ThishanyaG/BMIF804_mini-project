import SimpleITK as sitk
import utils
import matplotlib.pyplot as plt

"""
BMIF 804 Mini Project
Thishanya Gunasekera (20367785)

GitHub: https://github.com/ThishanyaG/BMIF804_mini-project

This code will use the functions of util.py to create a segment mask of the given prostate image, calculate the dice
similarity coefficient of the created mask and a given gold standard segment mask, visualize both masks overlaying the
prostate image, determine the centroid of the prostate, extract the pixel intensities 6mm around the centroid, and plot 
the intensities in a boxplot. 
"""
def main():
    prostate_img = sitk.ReadImage("C:/Users/thish/Documents/804/mini-project/case23_resampled.nii")
    prostate_seg = sitk.ReadImage("C:/Users/thish/Documents/804/mini-project/case23_resampled_segmentation.nii")

    segment = utils.prostate_segmenter(prostate_img, 400, 200)

    writer = sitk.ImageFileWriter()
    writer.SetFileName("C:/Users/thish/Documents/804/mini-project/my_segmentation.nrrd")
    writer.Execute(segment)

    img_overlay_made = sitk.LabelOverlay(prostate_img, segment)
    img_overlay_ideal = sitk.LabelOverlay(prostate_img, prostate_seg)

    plt.imshow(sitk.GetArrayFromImage(img_overlay_made[:, :, 35]), cmap='gray')
    plt.title("Created Segmentation Mask Overlay")
    plt.axis('off')
    plt.show()
    plt.imshow(sitk.GetArrayFromImage(img_overlay_ideal[:, :, 35]), cmap='gray')
    plt.title("Gold Standard Segmentation Mask Overlay")
    plt.axis('off')
    plt.show()

    prostate_dsc = utils.seg_eval_dice(prostate_seg, segment)
    print("The DSC of these segments is:", prostate_dsc)

    centroid = utils.get_target_loc(prostate_seg)

    plt.figure()
    plt.imshow(sitk.GetArrayFromImage(prostate_img[:, :, centroid[2]]), cmap='gray')
    plt.plot([centroid[0]], [centroid[1]], 'rx')
    plt.show()

    cube_intensities = utils.pixel_extract(prostate_img, centroid, 6)

    plt.boxplot(cube_intensities)
    plt.xlabel(centroid)
    plt.ylabel("Pixel Intensity")
    plt.show()


main()
