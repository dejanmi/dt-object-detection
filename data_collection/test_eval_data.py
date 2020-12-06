import cv2
import numpy as np
import os

dataset_files = list(filter(lambda x: "npz" in x, os.listdir("./data_collection/dataset")))

classes_list = ["background", "duckie", "cone", "truck", "bus"]

for df in dataset_files:
    print(df)
    data = np.load(f"./data_collection/dataset/{df}")
    img = data[f"arr_{0}"]
    boxes = data[f"arr_{1}"]
    classes = data[f"arr_{2}"]

    for i in range(len(classes)):
        label = classes_list[classes[i]]
        cv2.rectangle(img, (boxes[i][0], boxes[i][1]), (boxes[i][2], boxes[i][3]), (0, 255, 0), 1)
        # cv2.putText(img, label, (boxes[i][0], boxes[i][1] - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)


    img = cv2.resize(img, (224*8, 224*8))
    print(np.shape(img))
    cv2.imshow("IMG", img)
    cv2.waitKey(0)
