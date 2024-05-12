import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from paddleocr import PaddleOCR
import os

def intersection(box_1, box_2):
    return [box_2[0], box_1[1],box_2[2], box_1[3]]
def iou(box_1, box_2):

    x_1 = max(box_1[0], box_2[0])
    y_1 = max(box_1[1], box_2[1])
    x_2 = min(box_1[2], box_2[2])
    y_2 = min(box_1[3], box_2[3])

    inter = abs(max((x_2 - x_1, 0)) * max((y_2 - y_1), 0))
    if inter == 0:
        return 0

    box_1_area = abs((box_1[2] - box_1[0]) * (box_1[3] - box_1[1]))
    box_2_area = abs((box_2[2] - box_2[0]) * (box_2[3] - box_2[1]))

    return inter / float(box_1_area + box_2_area - inter)

def load_ocr_model(lang='fr'):
    return PaddleOCR(lang=lang)

def extract_text_from_image(image_path, ocr_model):
    image_cv = cv2.imread(image_path)
    if image_cv is None:
        raise ValueError(f"Failed to read image file: {image_path}")

    output = ocr_model.ocr(image_path)[0]

    boxes = [line[0] for line in output]
    texts = [line[1][0] for line in output]
    probabilities = [line[1][1] for line in output]

    return image_cv, boxes, texts, probabilities

def save_ocr_detections(image_cv, boxes, texts, output_path='detections.jpg'):
    image_boxes = image_cv.copy()
    for box, text in zip(boxes, texts):
        cv2.rectangle(image_boxes, (int(box[0][0]), int(box[0][1])), (int(box[2][0]), int(box[2][1])), (0, 0, 255), 1)
        cv2.putText(image_boxes, text, (int(box[0][0]), int(box[0][1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (222, 0, 0), 1)

    cv2.imwrite(output_path, image_boxes)



def main(image_paths):
    ocr = load_ocr_model()
    for image_path in image_paths:
        image_cv, boxes, texts, probabilities = extract_text_from_image(image_path, ocr)
        image_width = image_cv.shape[1]
        image_height = image_cv.shape[0]
        im = image_cv.copy()
        horiz_boxes = []
        vert_boxes = []

        for box in boxes:
            x_h, x_v = 0,int(box[0][0])
            y_h, y_v = int(box[0][1]),0
            width_h,width_v = image_width, int(box[2][0]-box[0][0])
            height_h,height_v = int(box[2][1]-box[0][1]),image_height

            horiz_boxes.append([x_h,y_h,x_h+width_h,y_h+height_h])
            vert_boxes.append([x_v,y_v,x_v+width_v,y_v+height_v])

            cv2.rectangle(im,(x_h,y_h), (x_h+width_h,y_h+height_h),(0,0,255),1)
            cv2.rectangle(im,(x_v,y_v), (x_v+width_v,y_v+height_v),(0,255,0),1)
        
        horiz_out = tf.image.non_max_suppression(
            horiz_boxes,
            probabilities,
            max_output_size = 1000,
            iou_threshold=0.1,
            score_threshold=float('-inf'),
            name=None
        )
        horiz_lines = np.sort(np.array(horiz_out))
        im_nms = image_cv.copy()
        for val in horiz_lines:
            cv2.rectangle(im_nms, (int(horiz_boxes[val][0]),int(horiz_boxes[val][1])), (int(horiz_boxes[val][2]),int(horiz_boxes[val][3])),(0,0,255),1)
        
        cv2.imwrite('im_nms.jpg',im_nms)
        vert_out = tf.image.non_max_suppression(
            vert_boxes,
            probabilities,
            max_output_size = 1000,
            iou_threshold=0.1,
            score_threshold=float('-inf'),
            name=None
        )
        vert_lines = np.sort(np.array(vert_out))
        for val in vert_lines:
            cv2.rectangle(im_nms, (int(vert_boxes[val][0]),int(vert_boxes[val][1])), (int(vert_boxes[val][2]),int(vert_boxes[val][3])),(255,0,0),1)
        
        out_array = [["" for i in range(len(vert_lines))] for j in range(len(horiz_lines))]
        unordered_boxes = []

        for i in vert_lines:
            print(vert_boxes[i])
            unordered_boxes.append(vert_boxes[i][0])
            
        ordered_boxes = np.argsort(unordered_boxes)
        for i in range(len(horiz_lines)):
            for j in range(len(vert_lines)):
                resultant = intersection(horiz_boxes[horiz_lines[i]], vert_boxes[vert_lines[ordered_boxes[j]]] )

                for b in range(len(boxes)):
                    the_box = [boxes[b][0][0],boxes[b][0][1],boxes[b][2][0],boxes[b][2][1]]
                    if(iou(resultant,the_box)>0.1):
                        out_array[i][j] = texts[b]
        out_array=np.array(out_array)
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        #pd.DataFrame(out_array).to_csv(f'{image_name}_output.csv')
        writer = pd.ExcelWriter(f'{image_name}_output.xlsx')
        pd.DataFrame(out_array).to_excel(writer)
        writer.close()
        #save_ocr_detections(image_cv, boxes, texts)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="OCR on images")
    parser.add_argument("image_paths", nargs='+', type=str, help="Paths to the images")
    args = parser.parse_args()

    main(args.image_paths)
