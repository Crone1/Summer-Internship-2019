#!/usr/bin/env python

"""
This program needs the path to a YAML/JSON Configuration file
It returns a file containing a list of dictionaries containing the details of the coordinates of the areas drawn when this program is run.
"""

# imports
import numpy as np
import json
import yaml
import cv2
import sys

from DrawingInputs import DrawingInputs
from utils.maths_utils import get_distance


# read the drawing customisation configuration variables
with open("configs/drawing_configs.yaml", "r") as drawing_config_file:
    drawing_configs = yaml.load(drawing_config_file, Loader=yaml.FullLoader)
with open("configs/default_configs.yaml", "r") as default_config_file:
    default_configs = yaml.load(default_config_file, Loader=yaml.FullLoader)
with open("configs/video_resolutions.yaml", "r") as resolution_file:
    resolution_configs = yaml.load(resolution_file, Loader=yaml.FullLoader)


# variables needed to set up drawing
drawing_mode = drawing_configs["drawing_mode"]
drawing = False
drawing_poly = False
start_x, start_y = -1, -1
area_details = []
poly_points = []
img_hist = []


def reshape_background_image(img, size):
    """
    Function Goal : reshape the image so that it still maintains the same proportion but that its width is the base_width

    img : 3D numpy array of integers - the array representing the image you want to reshape
    size : tuple of integers - the height and width you want the image to be

    return : 3D numpy array of integers - this array represents the image in its reshaped form in a way that python can deal with.
    """

    # get image orientation in line with size
    height, width, _ = img.shape
    desired_width, desired_height = size
    if (desired_height <= desired_width) and not (height <= width):
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    elif (desired_width <= desired_height) and not (width <= height):
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

    # scale to correct resolution
    return cv2.resize(img, size)


def print_how_to_use_image_drawer():
    """
    Function Goal : This function contains a series of print statements that explain to the user how to use the program and draw the areas the image

    return : None

    keyboard shortcuts to control the program:
        "c": circle
        "p": polygon
        "r": rectangle
        "Escape": Exit program
        "Enter" (in polygon drawing mode): Finish polygon
        "Backspace": Undo
    """

    print(" To:\t\t\t\t Press:")
    print(" Draw rectangle:\t\t 'r'\n Draw multi cornered polygon:\t 'p'\n Draw circle:\t\t\t 'c'")
    print(" Finish drawing polygon:\t 'Enter'")
    print(" Undo the last drawn shape:\t 'Backspace'\n Finished drawing:\t\t 'Esc'\n")
    print("The default shape is:", drawing_mode)


def keyboard_callbacks(key):

    global drawing_mode, drawing, poly_points, tmp_img

    # Modes
    if key == ord("c"):
        drawing_mode = "circle"

    elif key == ord("r"):
        drawing_mode = "rectangle"

    elif key == ord("p"):
        drawing_mode = "polygon"

    elif key == 13:  # Enter
        # Save polygon
        if drawing_mode == "polygon":
            drawing = False

            # Update image
            tmp_img = np.copy(img_hist[-1])
            line_thickness = int(drawing_configs["proportion_for_line_thickness"] * img.shape[1])
            cv2.polylines(tmp_img, np.int32([poly_points]), True, drawing_configs["drawing_colour"], line_thickness)
            img_hist.append(np.copy(tmp_img))

            # Save parameters
            area_details.append({
                "type": "polygon",
                "points": poly_points,
            })

            poly_points = []

    elif key == 8:  # Backspace
        # Undo
        if drawing_mode == "polygon" and drawing:
            if len(poly_points) > 0:
                poly_points.pop()

        elif len(img_hist) > 1:
            img_hist.pop()
            area_details.pop()
            tmp_img = np.copy(img_hist[-1])


def mouse_callbacks(event, x, y):

    global start_x, start_y, drawing, drawing_mode, img, tmp_img, poly_points

    line_thickness = int(drawing_configs["proportion_for_line_thickness"] * img.shape[1])

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        start_x, start_y = x, y

        if drawing_mode == "polygon":
            poly_points.append((x, y))

    elif event == cv2.EVENT_MOUSEMOVE:

        if drawing:
            tmp_img = np.copy(img_hist[-1])

            # Draw temporary shape that follows the cursor
            if drawing_mode == "rectangle":
                cv2.rectangle(tmp_img, (start_x, start_y),
                              (x, y), drawing_configs["drawing_colour"], line_thickness)

            elif drawing_mode == "circle":
                cv2.circle(tmp_img, (start_x, start_y), int(
                    get_distance((start_x, start_y), (x, y))), drawing_configs["drawing_colour"], line_thickness)

            elif drawing_mode == "polygon":
                cv2.polylines(tmp_img, np.int32(
                    [poly_points + [(x, y)]]), True, drawing_configs["drawing_colour"], line_thickness)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False

        # Finish drawing shape
        if drawing_mode == "rectangle" and (start_x != x or start_y != y):
            cv2.rectangle(tmp_img, (start_x, start_y), (x, y), drawing_configs["drawing_colour"], line_thickness)
            # save shape as json
            area_details.append({
                "type": "rectangle",
                "start": (start_x, start_y),
                "end": (x, y),
            })

        elif drawing_mode == "circle":
            radius = int(get_distance((start_x, start_y), (x, y)))
            cv2.circle(tmp_img, (start_x, start_y), radius, drawing_configs["drawing_colour"], line_thickness)
            area_details.append({
                "type": "circle",
                "centre": (start_x, start_y),
                "radius": radius,
            })

        elif drawing_mode == "polygon":
            # don't cancel drawing
            drawing = True
            return

        img_hist.append(np.copy(tmp_img))


def draw_on_image(image):

    global img, tmp_img

    img = image
    img_hist.append(img)
    tmp_img = np.copy(img)
    cv2.namedWindow('image')
    cv2.setMouseCallback('image', lambda event, x, y, flags, params: mouse_callbacks(event, x, y))
    while True:
        cv2.imshow('image', tmp_img if drawing else img_hist[-1])

        # key press handling
        key = cv2.waitKey(20) & 0xFF
        if key == 27:  # Escape
            break

        else:
            keyboard_callbacks(key)

    cv2.destroyAllWindows()

    return area_details


def handle_inputs():
    """
    Function Goal : call the functions that input the variables that are needed to make the program work and pass these variables to the main() function

    return : tuple of strings - the path to the background image, the path to the output file
    """

    input_handler = DrawingInputs()

    if len(sys.argv) > 1:
        input_handler.get_variables_from_command_line()
    else:
        input_handler.get_variables_from_user()

    return input_handler.background_path, input_handler.output_path


def main():

    # get inputs
    background_image_path, output_path = handle_inputs()

    # read image
    raw_img = cv2.imread(background_image_path)
    video_width, video_height = resolution_configs[default_configs["video"]["resolution"]]
    desired_size = (
        int(default_configs["video"]["proportions"]["width"]["background"] * video_width),
        int(default_configs["video"]["proportions"]["height"]["background"] * video_height)
    )
    background = reshape_background_image(raw_img, desired_size)

    # draw areas
    print_how_to_use_image_drawer()
    area_details = draw_on_image(background)

    # output to file
    with open(output_path, 'w') as file_of_area_details:
        file_of_area_details.write(json.dumps(area_details))
        print("\nThe area details were written to the file under the name '" + str(output_path) + "'.")

    # return area details
    return area_details


if __name__ == '__main__':
    main()
