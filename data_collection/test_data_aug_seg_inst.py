import cv2
import numpy as np
import os
import utils

# def empty(x):
#     pass

def getCountours(img, imgContour):
    countours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for cnt in countours:
        area = cv2.contourArea(cnt)
        if area > 100:
            cv2.drawContours(imgContour, cnt, -1, (0, 255, 0), 2)
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            x, y, w, h = cv2.boundingRect(approx)
            cv2.rectangle(imgContour, (x, y), (x + w, y + h), (0, 255, 0), 2)


dataset_files = list(filter(lambda x: "npz" in x, os.listdir("./data_collection/dataset")))

classes_list = ["background", "duckie", "cone", "truck", "bus"]

label_map = {"duckie": {"color": [100, 117, 226], "label": 1},
             "cone": {"color": [226, 111, 101], "label": 2},
             "truck": {"color": [116, 114, 117], "label": 3},
             "bus": {"color": [216, 171, 15], "label": 4}}

for df in dataset_files:
    print(df)
    data = np.load(f"./data_collection/dataset/{df}")
    seg_img = data[f"arr_{0}"]
    imgContour = seg_img.copy()
    for obj, label in label_map.items():
        cu_mask = np.all(seg_img == label["color"], axis=-1)
        non_cu_mask = ~cu_mask
        mask_scaled_copy = seg_img.copy()
        mask_scaled_copy[cu_mask] = [255, 255, 255]
        mask_scaled_copy[non_cu_mask] = [0, 0, 0]
        # mask_scaled_copy = cv2.GaussianBlur(mask_scaled_copy, (3, 3), 1)
        mask_scaled_copy = cv2.medianBlur(mask_scaled_copy, 7)
        cv2.imshow("IMG1", mask_scaled_copy)
        cv2.waitKey(0)
        med_val = np.median(mask_scaled_copy)
        threshold1 = int(max(0, 0.7 * med_val))
        threshold2 = int(min(255, 1.3 * med_val))
        imgCanny = cv2.Canny(mask_scaled_copy, threshold1, threshold2)
        kernel = np.ones((5, 5))
        imgDil = cv2.dilate(imgCanny, kernel, iterations=1)
        cv2.imshow("IMG2", mask_scaled_copy)
        cv2.waitKey(0)
        getCountours(imgDil, imgContour)

    cv2.imshow("IMG", imgContour)
    cv2.waitKey(0)

# img = cv2.imread("/home/dejan/PennFudanPed/PNGImages/PennPed00001.png")
# print("display img and mask")
# utils.display_seg_mask(img[::4, ::4, :], mask_scaled[::10, ::10, :])


# TODO boxes, classes = clean_segmented_image(segmented_obs)
        # TODO save_npz(obs, boxes, classes)
        # mask_scaled_copy = segmented_obs.copy()
        # for obj, label in label_map.items():
        #     cu_mask = np.all(segmented_obs == label["color"], axis=-1)
        #     mask_scaled_copy[cu_mask] = [0, 0, 0]
        #
        # cu_mask = np.all(segmented_obs == [255,255,255], axis=-1)
        # mask_scaled_copy[cu_mask] = [0, 0, 0]
        # cu_mask = np.all(segmented_obs == [255, 0, 255], axis=-1)
        # mask_scaled_copy[cu_mask] = [0, 0, 0]
        # cu_mask = np.all(segmented_obs == [254, 254, 254], axis=-1)
        # mask_scaled_copy[cu_mask] = [0, 0, 0]
        # cu_mask = np.all(segmented_obs == [240, 239, 237], axis=-1)
        # mask_scaled_copy[cu_mask] = [0, 0, 0]
        # cu_mask = np.all(segmented_obs == [238, 237, 233], axis=-1)
        # mask_scaled_copy[cu_mask] = [0, 0, 0]
        # cu_mask = np.all(segmented_obs == [237, 224, 56], axis=-1)
        # mask_scaled_copy[cu_mask] = [0, 0, 0]
        # cv2.imshow("IMG", mask_scaled_copy)
        # cv2.waitKey(0)
        # n_splits = 5
        # n_pixels_vertical = np.size(mask_scaled_copy, axis=0)
        # n_pixels_sector = int(n_pixels_vertical / n_splits)
        # print(f'The total number of secors is: {n_splits}')
        # print('The most present color in each sector is: ')
        # count_fla = False
        # for i in range(0, n_splits):
        #     p_0 = i * n_pixels_sector
        #     p_1 = (i + 1) * n_pixels_sector
        #     sector_i = mask_scaled_copy[p_0:p_1, :, :]
        #     color_unique, counts = np.unique(sector_i.reshape(-1, sector_i.shape[2]), return_counts=True, axis=0)
        #     if len(color_unique) == 1:
        #         count_fla = False
        #     else:
        #         cu_mask = np.all(color_unique == [0, 0, 0], axis=-1)
        #         color_unique = np.delete(color_unique, cu_mask, 0)
        #         counts = np.delete(counts, cu_mask, 0)
        #         count_fla = True
