import cv2
import numpy as np
import utils

# def empty(x):
#     pass

def getCountours(img, imgContour):
    countours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for cnt in countours:
        area = cv2.contourArea(cnt)
        if area > 1000:
            cv2.drawContours(imgContour, cnt, -1, (255, 0, 255), 2)
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            x, y, w, h = cv2.boundingRect(approx)
            cv2.rectangle(imgContour, (x, y), (x + w, y + h), (0, 255, 0), 2)

for i in range(1, 100):
    if i > 9:
        n = f"000{i}"
    else:
        n = f"0000{i}"
    mask = cv2.imread(f"/home/dejan/PennFudanPed/PedMasks/PennPed{n}_mask.png")
    img = cv2.imread(f"/home/dejan/PennFudanPed/PNGImages/PennPed{n}.png")
    mask_scaled = np.floor(mask / np.max(mask) * 255).astype(np.uint8)
    imgContour = mask_scaled.copy()
    color_unique, counts = np.unique(mask_scaled.reshape(-1, mask_scaled.shape[2]), return_counts=True, axis=0)
    black = np.where(np.all(color_unique == [0, 0, 0], axis=1))
    color_unique = np.delete(color_unique, black[0], 0)
    print(color_unique)
    # mask_scaled = cv2.GaussianBlur(mask_scaled, (5,5), 1)
    # cv2.namedWindow("Parameters")
    # cv2.resizeWindow("Parameters", 640, 240)
    # cv2.createTrackbar("Treshold1", "Parameters", 1, 255, empty)
    # cv2.createTrackbar("Treshold2", "Parameters", 1, 255, empty)
    # threshold1 = cv2.getTrackbarPos("Treshold1", "Parameters")
    # threshold2 = cv2.getTrackbarPos("Treshold2", "Parameters")
    for cu in color_unique:
        print(cu)
        cu_mask = np.all(mask_scaled == cu, axis=-1)
        non_cu_mask = ~cu_mask
        mask_scaled_copy = mask_scaled.copy()
        mask_scaled_copy[cu_mask] = cu
        mask_scaled_copy[non_cu_mask] = [0, 0, 0]
        mask_scaled_copy = cv2.GaussianBlur(mask_scaled_copy, (5, 5), 1)
        # cv2.imshow("filterd", mask_scaled_copy)
        # cv2.waitKey(0)
        med_val = np.median(mask_scaled_copy)
        threshold1 = int(max(0, 0.7 * med_val))
        threshold2 = int(min(255, 1.3 * med_val))
        imgCanny = cv2.Canny(mask_scaled_copy, threshold1, threshold2)
        kernel = np.ones((5, 5))
        imgDil = cv2.dilate(imgCanny, kernel, iterations=1)
        getCountours(imgDil, imgContour)

    cv2.imshow("IMG", imgContour)
    cv2.waitKey(0)

# img = cv2.imread("/home/dejan/PennFudanPed/PNGImages/PennPed00001.png")
# print("display img and mask")
# utils.display_seg_mask(img[::4, ::4, :], mask_scaled[::10, ::10, :])
