import cv2
import numpy as np
import pandas as pd
from paddleocr import PaddleOCR
import tensorflow as tf
def load_ocr_model(lang='fr'):
    return PaddleOCR(lang=lang)

def intersection(box_1, box_2):
    return [box_2[0], box_1[1], box_2[2], box_1[3]]

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
            
def process_image(image_name, image_data, ocr_model):
    # Convert image data to numpy array
    nparr = np.frombuffer(image_data, np.uint8)
    
    # Decode image
    image_cv = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Perform OCR on the image
    output = ocr_model.ocr(image_cv)[0]

    # Extract text, boxes, and probabilities
    boxes = [line[0] for line in output]
    texts = [line[1][0] for line in output]
    probabilities = [line[1][1] for line in output]

    # Extract information and put it into Excel file
    image_width = image_cv.shape[1]
    image_height = image_cv.shape[0]
    horiz_boxes = []
    vert_boxes = []
    
    im = image_cv.copy()

    for box in boxes:
        x_h, x_v = 0, int(box[0][0])
        y_h, y_v = int(box[0][1]), 0
        width_h, width_v = image_width, int(box[2][0] - box[0][0])
        height_h, height_v = int(box[2][1] - box[0][1]), image_height

        horiz_boxes.append([x_h, y_h, x_h + width_h, y_h + height_h])
        vert_boxes.append([x_v, y_v, x_v + width_v, y_v + height_v])
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

    # Construct output CSV and Excel file paths
    #output_csv_path = f'uploads/{image_name}_output.csv'
    output_excel_path = f'C:/Users/fatha/OneDrive/Desktop/PaddleOCR/{image_name}_output.xlsx'

    # Write output to CSV
    out_array = np.array(out_array)
    #pd.DataFrame(out_array).to_csv(output_csv_path, header=False, index=False)

    # Write output to Excel
    writer = pd.ExcelWriter(output_excel_path)
    pd.DataFrame(out_array).to_excel(writer, header=False, index=False, sheet_name=image_name)  # Use image_name as sheet name
    writer.close()

    return output_excel_path

