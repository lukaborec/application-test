import pytesseract
import cv2
from PIL import  Image
import tesserocr
import argparse
import imutils
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, help='path to image')
args = parser.parse_args()

def is_equal(image1, image2):
    if image1.shape != image2.shape:
        return "Shapes don't match"
    return not np.any(cv2.subtract(image1, image2))

def overrun(box1, box2):
    return box1[0] == box2[0] or box1[1] == box2[1] or box1[2] == box2[2] or box1[3] == box2[3]

def load_img(path):
    # Load the image, resize it and turn it into grayscale
    image = cv2.imread(path)
    image = imutils.resize(image, width=900)
    ratio = image.shape[0] / float(image.shape[0])
    return image, cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), ratio

def remove_text(image):
    # Extracts text from the img
    thresh = cv2.threshold(image, 90, 255, cv2.THRESH_BINARY_INV)[1] #Binary threshold
    inp_mask = cv2.dilate(thresh, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10,10)))
    dst = cv2.inpaint(image, inp_mask, 7, cv2.INPAINT_NS)
    return dst

def detect_shapes(image):
    # Detects shapes on the image without text, creates contours and fills them in
    _, threshold = cv2.threshold(image, 240, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return cv2.fillPoly(np.zeros(image.shape), contours, 255)

if __name__ == '__main__':
    path = args.path

    original, img, ratio = load_img(path)
    removed_text = remove_text(img)
    filled_img = detect_shapes(removed_text)

    # Binary hreshold to detect text
    thresh = cv2.threshold(img, 90, 255, cv2.THRESH_BINARY)[1]

    # Use tesserocr to detect text bounding boxes
    with tesserocr.PyTessBaseAPI() as api:
        api.SetImage(Image.fromarray(thresh))
        boxes = api.GetComponentImages(tesserocr.RIL.TEXTLINE, True)
        text = api.GetUTF8Text()

    # Takes the image with extracted shapes and draws the bounding boxes on top of the image using the same color for the
    # text box as for the shapes
    # If the resulting image looks the same as the image only containing shapes, that means that text is contained within
    # the shape and we accept it; otherwise, if the boxes of the text and the shape overlap, the text is outside of the shape
    # and we reject it
    rejected = []
    accepted = []
    for box in boxes:
        if overrun((box[1]['x'], box[1]['y'], box[1]['x'] + box[1]['w'], box[1]['y'] + box[1]['h']),
                   (0, 0, img.shape[1], img.shape[0])):
            rejected.append(box)

    for i, box in enumerate(boxes):
        if box not in rejected:
            temp = filled_img.copy()
            temp = cv2.rectangle(temp, (box[1]['x'], box[1]['y']),
                                 (box[1]['x'] + box[1]['w'], box[1]['y'] + box[1]['h']), 255, 2)
            if is_equal(temp, filled_img):
                accepted.append(box)
            else:
                rejected.append(box)

    # Draws the result on the original image
    for box in rejected:
        cv2.rectangle(original, (box[1]['x'], box[1]['y']), (box[1]['x'] + box[1]['w'], box[1]['y'] + box[1]['h']),
                              (0, 0, 255), 1)
    for box in accepted:
        cv2.rectangle(original, (box[1]['x'], box[1]['y']), (box[1]['x'] + box[1]['w'], box[1]['y'] + box[1]['h']),
                              (0, 255, 0), 1)

    cv2.imshow('image', original)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
