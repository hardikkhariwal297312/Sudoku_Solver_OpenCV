# DataFlair Sudoku solver

import cv2
import numpy as np
from tensorflow.keras.models import load_model
import imutils
from solver import *
from imageprocessing import *
from extractnumber import *

classes = np.arange(0, 10)

model = load_model('model-OCR.h5')
print(model.summary())
input_size = 48

# Read image
img = cv2.imread('image_data/image8.jpg')


# extract board from input image
board, location = find_board(img)


gray = cv2.cvtColor(board, cv2.COLOR_BGR2GRAY)
print(gray.shape)
rois = split_boxes(gray)
rois = np.array(rois).reshape(-1, input_size, input_size, 1)

# get prediction
prediction = model.predict(rois)
# print(prediction)

predicted_numbers = []
# get classes from prediction
for i in prediction: 
    index = (np.argmax(i)) # returns the index of the maximum number of the array
    predicted_number = classes[index]
    predicted_numbers.append(predicted_number)

# print(predicted_numbers)

# reshape the list 
board_num = np.array(predicted_numbers).astype('uint8').reshape(9, 9)



# solve the board
try:
    solved_board_nums = get_board(board_num)

    # create a binary array of the predicted numbers. 0 means unsolved numbers of sudoku and 1 means given number.
    binArr = np.where(np.array(predicted_numbers)>0, 0, 1)
    print(binArr)
    # get only solved numbers for the solved board
    flat_solved_board_nums = solved_board_nums.flatten()*binArr
    # create a mask
    mask = np.zeros_like(board)
    # displays solved numbers in the mask in the same position where board numbers are empty
    solved_board_mask = displayNumbers(mask, flat_solved_board_nums)
    cv2.imshow("Solved Mask", solved_board_mask)
    inv = get_InvPerspective(img, solved_board_mask, location)
    cv2.imshow("Inverse Perspective", inv)
    combined = cv2.addWeighted(img, 0.7, inv, 1, 0)
    cv2.imshow("Final result", combined)
    cv2.waitKey(0)
    

except:
    print("Solution doesn't exist. Model misread digits.")

cv2.imshow("Input image", img)
cv2.imshow("Board", board)
cv2.waitKey(0)
cv2.destroyAllWindows()