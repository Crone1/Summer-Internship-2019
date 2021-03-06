#!/usr/bin/env python

# import libraries
import matplotlib.pyplot as plt
import drawing_images_program
import pandas as pd
import numpy as np
import matplotlib
import threading
import argparse
import queue
import json
import yaml
import time
import sys
import cv2
import os


print(time.time())

'''
# read in JSON configuration files
with open("Configuration files/Configurate default read in variables.json", "r") as defaults:
    default_config_variables = json.load(defaults)

with open("Configuration files/Configurate variables.json", "r") as variables:
    config_variables = json.load(variables)
'''

# read in YAML configuration files
with open("Configuration files/Yaml/Configurate default read in variables.yaml", "r") as defaults:
    default_config_variables = yaml.load(defaults, Loader=yaml.FullLoader)

with open("Configuration files/Yaml/Configurate variables.yaml", "r") as variables:
    config_variables = yaml.load(variables, Loader=yaml.FullLoader)


# dictionary to use built in openCV variables
cv2_dic = {

    "BORDER_REPLICATE": cv2.BORDER_REPLICATE,
    "BORDER_CONSTANT": cv2.BORDER_CONSTANT,

    "FONT_HERSHEY_SIMPLEX": cv2.FONT_HERSHEY_SIMPLEX,
    "FONT_HERSHEY_PLAIN": cv2.FONT_HERSHEY_PLAIN,
    "FONT_HERSHEY_DUPLEX": cv2.FONT_HERSHEY_DUPLEX,
    "FONT_HERSHEY_COMPLEX": cv2.FONT_HERSHEY_COMPLEX,
    "FONT_HERSHEY_TRIPLEX": cv2.FONT_HERSHEY_TRIPLEX,
    "FONT_HERSHEY_COMPLEX_SMALL": cv2.FONT_HERSHEY_COMPLEX_SMALL,
    "FONT_HERSHEY_SCRIPT_SIMPLEX": cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
    "FONT_HERSHEY_SCRIPT_COMPLEX": cv2.FONT_HERSHEY_SCRIPT_COMPLEX,
    "FONT_ITALIC": cv2.FONT_ITALIC,

    "FILLED": cv2.FILLED,
    "LINE_4": cv2.LINE_4,
    "LINE_8": cv2.LINE_8,
    "LINE_AA": cv2.LINE_AA,
}

# default variables - most only used for testing
default_colourmap_name = default_config_variables["default_colourmap_name"]
default_background_image_name = default_config_variables["default_background_image_name"]
default_basewidth = default_config_variables["default_basewidth"]
default_length_of_video = default_config_variables["default_length_of_video"]
default_output_filename = default_config_variables["default_output_filename"]
default_folder_of_videos = default_config_variables["default_folder_of_videos"]
defualt_shapes_filename = default_config_variables["defualt_shapes_filename"]
default_file_of_events = default_config_variables["default_file_of_events"]
default_folder_of_csvs = default_config_variables["default_folder_of_csvs"]
default_csv = default_config_variables["default_csv"]

sensor_value_name = config_variables["sensor_value_name"]

# other global variables
alpha_for_transperancy_of_shapes_on_background = config_variables["alpha_for_transperancy_of_shapes_on_background"]
gamma_for_merging_shapes_and_background = config_variables["gamma_for_merging_shapes_and_background"]

max_sensor_value = config_variables["max_sensor_value"]
min_sensor_value = config_variables["min_sensor_value"]

colour_of_Nan_values = config_variables["colour_of_Nan_values"]

event_duration_sec = config_variables["event_duration_sec"]

lineType_of_arrow_between_camera_nd_area = config_variables["lineType_of_arrow_between_camera_nd_area"]
lineThickness_of_arrow_between_camera_nd_area = config_variables["lineThickness_of_arrow_between_camera_nd_area"]
colour_of_arrow_between_camera_nd_area = config_variables["colour_of_arrow_between_camera_nd_area"]

angle_to_change_line_for_arrow_head = config_variables["angle_to_change_line_for_arrow_head"]
length_of_arrow_head_lines = config_variables["length_of_arrow_head_lines"]
num_points_on_line_for_outside_of_shape = config_variables["num_points_on_line_for_outside_of_shape"]

proportion_of_image_height_to_make_text_box = config_variables["proportion_of_image_height_to_make_text_box"]
proportion_of_basewidth_to_make_second_image = config_variables["proportion_of_basewidth_to_make_second_image"]
proportion_of_image_height_to_make_colourmap_and_second_image = config_variables["proportion_of_image_height_to_make_colourmap_and_second_image"]
proportion_of_basewidth_to_make_the_left_and_right_images = config_variables["proportion_of_basewidth_to_make_the_left_and_right_images"]


inner_colourmap_image_border_type = cv2_dic[config_variables["inner_colourmap_image_border_type"]]
inner_colourmap_image_border_colour = config_variables["inner_colourmap_image_border_colour"]
proportion_of_width_to_make_inner_colourmap_horizontal_border = config_variables["proportion_of_width_to_make_inner_colourmap_horizontal_border"]

proportion_of_height_to_make_inner_colourmap_bottom_border = config_variables["proportion_of_height_to_make_inner_colourmap_bottom_border"]
proportion_of_height_to_make_inner_colourmap_top_border = config_variables["proportion_of_height_to_make_inner_colourmap_top_border"]

colour_of_lines_for_scale_on_colourmap = config_variables["colour_of_lines_for_scale_on_colourmap"]
colour_of_line_seperating_colmap_header_and_colmap = config_variables["colour_of_line_seperating_colmap_header_and_colmap"]

proportion_of_colourmap_image_to_make_indexes = config_variables["proportion_of_colourmap_image_to_make_indexes"]
proportion_of_colourmap_image_to_make_heading = config_variables["proportion_of_colourmap_image_to_make_heading"]
proportion_of_height_to_make_space_between_line_and_text = config_variables["proportion_of_height_to_make_space_between_line_and_text"]

font_type_of_heading_on_colourmap = cv2_dic[config_variables["font_type_of_heading_on_colourmap"]]
font_size_of_heading_on_colourmap = config_variables["font_size_of_heading_on_colourmap"]
font_thickness_of_heading_on_colourmap = config_variables["font_thickness_of_heading_on_colourmap"]
colour_of_heading_on_colourmap = config_variables["colour_of_heading_on_colourmap"]
line_type_of_heading_on_colourmap = config_variables["line_type_of_heading_on_colourmap"]

font_type_of_index_on_colourmap = cv2_dic[config_variables["font_type_of_index_on_colourmap"]]
font_size_of_index_on_colourmap = config_variables["font_size_of_index_on_colourmap"]
font_thickness_of_index_on_colourmap = config_variables["font_thickness_of_index_on_colourmap"]
colour_of_index_on_colourmap = config_variables["colour_of_index_on_colourmap"]
line_type_of_index_on_colourmap = config_variables["line_type_of_index_on_colourmap"]


# borders---------------------------------
colour_of_outline_of_areas = config_variables["colour_of_outline_of_areas"]
thickness_of_outline_on_areas = config_variables["thickness_of_outline_on_areas"]

colmap_border_width = config_variables["colmap_border_width"]
colmap_border_colour = config_variables["colmap_border_colour"]
colmap_border_type = cv2_dic[config_variables["colmap_border_type"]]

second_image_border_width = config_variables["second_image_border_width"]
second_image_border_colour = config_variables["second_image_border_colour"]
second_image_border_type = cv2_dic[config_variables["second_image_border_type"]]

merged_image_border_width = config_variables["merged_image_border_width"]
merged_image_border_colour = config_variables["merged_image_border_colour"]
merged_image_border_type = cv2_dic[config_variables["merged_image_border_type"]]

bar_plot_border_width = config_variables["bar_plot_border_width"]
bar_plot_border_colour = config_variables["bar_plot_border_colour"]
bar_plot_border_type = cv2_dic[config_variables["bar_plot_border_type"]]

text_box_image_border_width = config_variables["text_box_image_border_width"]
text_box_image_border_colour = config_variables["text_box_image_border_colour"]
text_box_image_border_type = cv2_dic[config_variables["text_box_image_border_type"]]

video_footage_border_width = config_variables["video_footage_border_width"]
video_footage_border_colour = config_variables["video_footage_border_colour"]
video_footage_border_type = cv2_dic[config_variables["video_footage_border_type"]]

# Text of images----------------------
font_size_of_second_image = config_variables["font_size_of_second_image"]
font_type_of_second_image = cv2_dic[config_variables["font_type_of_second_image"]]
line_type_of_second_image = cv2_dic[config_variables["line_type_of_second_image"]]
font_thickness_of_second_image = config_variables["font_thickness_of_second_image"]
colour_of_second_image_text = config_variables["colour_of_second_image_text"]

font_size_of_top_text = config_variables["font_size_of_top_text"]
font_type_of_top_text = cv2_dic[config_variables["font_type_of_top_text"]]
line_type_of_top_text = cv2_dic[config_variables["line_type_of_top_text"]]
font_thickness_of_top_text = config_variables["font_thickness_of_top_text"]
colour_of_top_text = config_variables["colour_of_top_text"]

font_size_of_shape_name_text = config_variables["font_size_of_shape_name_text"]
font_type_of_shape_name_text = cv2_dic[config_variables["font_type_of_shape_name_text"]]
line_type_of_shape_name_text = cv2_dic[config_variables["line_type_of_shape_name_text"]]
font_thickness_of_shape_name_text = config_variables["font_thickness_of_shape_name_text"]
colour_of_shape_name_text = config_variables["colour_of_shape_name_text"]

font_size_of_camera_footage_text = config_variables["font_size_of_camera_footage_text"]
font_type_of_camera_footage_text = cv2_dic[config_variables["font_type_of_camera_footage_text"]]
line_type_of_camera_footage_text = cv2_dic[config_variables["line_type_of_camera_footage_text"]]
font_thickness_of_camera_footage_text = config_variables["font_thickness_of_camera_footage_text"]
colour_of_camera_footage_text = config_variables["colour_of_camera_footage_text"]

# lists for timing everything
loop_through_sensor_value_columns_and_return_list_of_coloured_shape_images_list = []
turn_all_the_different_images_into_one_image_list = []
video_list = []
shapes_list = []
resize_list = []
concat_outside_list = []
bar_list = []
video_2_frames_list = []
merge_list = []
slice_list = []
concat_inside_list = []
border_list = []
read_in_list = []
merge_heatmap = []
colmap_creation = []


class VideoReader:

    """
    This class is used along with the VideoReaderQueue class to read the frames of the videos into a queue to speed up this process 
    """

    def __init__(self, filename):
        self.filename = filename

        self.nframes = None
        self.framerate = None
        self.img_shape = None
        self._set_video_info()

    def load_video(self, interval=1):
        """
        Function Goal: Loads the images of the video

        interval: integer - Interval between frames returned. eg. 1 = every frame, 20 = every 20th frame.

        Return: a generator with the images, and the corresponding frame numbers.
        """

        def read_frames():
            vid = cv2.VideoCapture(self.filename)

            for i in frame_nums:
                vid.set(cv2.CAP_PROP_POS_FRAMES, i)

                has_frame, img = vid.read()

                if has_frame:
                    yield img
                else:
                    break

            vid.release()

        frame_nums = range(0, self.nframes, interval)

        return read_frames(), frame_nums

    def get_frame(self, n):
        """
        Return: the image at a specific frame number
        """

        vid = cv2.VideoCapture(self.filename)
        vid.set(cv2.CAP_PROP_POS_FRAMES, n)

        _, img = vid.read()

        vid.release()

        return img

    def _set_video_info(self):

        vid = cv2.VideoCapture(self.filename)
        _, img = vid.read()

        self.nframes = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
        self.framerate = vid.get(cv2.CAP_PROP_FPS)
        self.img_shape = img.shape

        vid.release()


class VideoReaderQueue(VideoReader):

    """
    This class is used along with the VideoReader class to read the frames of the videos into a queue to speed up this process 
    """

    def __init__(self, filename, queue_size=32):

        super().__init__(filename)
        self.queue_size = queue_size

    def load_video(self, interval=1):
        """
        Function Goal: Loads the images of the video - Loads the video on a separate thread lazily.

        interval: integer - Interval between frames returned. eg. 1 = every frame, 20 = every 20th frame.

        Return: a generator with the images, and the corresponding frame numbers.
        """

        def read_frames():

            vid = cv2.VideoCapture(self.filename)

            i = 0
            while vid.isOpened():
                grabbed = vid.grab()

                if grabbed:
                    if i in frame_nums:
                        _, img = vid.retrieve()
                        q.put(img)
                else:
                    break

                i += 1

            q.put(None)  # Mark end of video
            vid.release()

        def yield_frames():

            while True:
                if q.qsize() > 0:
                    item = q.get()

                    if item is None:
                        break
                    else:

                        yield item
                else:
                    time.sleep(0.01)

        frame_nums = range(0, self.nframes, interval)

        q = queue.Queue(maxsize=self.queue_size)
        thread = threading.Thread(target=read_frames)
        thread.daemon = True
        thread.start()

        return yield_frames(), frame_nums


def set_variables_using_folder():

    """
    This function is only used to speed up the process of running the program when testing
    """

    folder = default_folder_of_csvs
    csv_files = [f for f in sorted(os.listdir(folder)) if os.path.isfile(os.path.join(folder, f))]
    output_filename = default_output_filename
    length_of_video = default_length_of_video
    background_image_name = default_background_image_name
    basewidth = default_basewidth
    colourmap_name = default_colourmap_name
    shapes_filename = defualt_shapes_filename
    events_file = default_file_of_events

    folder_of_videos = default_folder_of_videos
    list_of_videos = [os.path.join(folder_of_videos, f) for f in os.listdir(folder_of_videos) if os.path.isfile(os.path.join(folder_of_videos, f))]

    return folder, csv_files, output_filename, length_of_video, background_image_name, basewidth, colourmap_name, shapes_filename, events_file, folder_of_videos, list_of_videos


def set_variables_using_file():

    """
    This function is only used to speed up the process of running the program when testing
    """

    folder = ""
    csv_files = [default_csv]
    output_filename = default_output_filename
    length_of_video = default_length_of_video
    background_image_name = default_background_image_name
    basewidth = default_basewidth
    colourmap_name = default_colourmap_name
    shapes_filename = defualt_shapes_filename
    events_file = default_file_of_events

    folder_of_videos = default_folder_of_videos
    list_of_videos = [os.path.join(folder_of_videos, f) for f in os.listdir(folder_of_videos) if os.path.isfile(os.path.join(folder_of_videos, f))]

    return folder, csv_files, output_filename, length_of_video, background_image_name, basewidth, colourmap_name, shapes_filename, events_file, folder_of_videos, list_of_videos


def touch_up_video_name(name):
    """
    Function Goal : add ".mp4" to the end of the video name if it isn't already there

    name : string - name of the output video

    return : string - name of the output video ending in ".mp4"
    """

    if name[-4:] != ".mp4":
        return name + ".mp4"

    else:
        return name


def get_list_of_csv_files(path_input):
    """
    Function Goal : Take the input string and extract the csvs from it

    path_input : string - it is either the name of a folder full of csvs or the name of a single csv file

    return : string, list of strings - folder name and list of the names of the csv files in the folder
    """

    csv_files = []
    folder = ""

    # check if the input is csv file
    if os.path.isfile(path_input) and path_input[-4:] == ".csv":
        csv_files.append(path_input)

    # check if the input is a folder and put the names of the csv files within this folder into a list
    elif os.path.isdir(path_input):
        folder = path_input
        csv_files = [f for f in sorted(os.listdir(path_input)) if os.path.isfile(os.path.join(path_input, f)) and f[-4:] == ".csv"]
        if csv_files == []:
            print("The folder you gave does not contain any csv files. Please run the program again with a valid folder path.")
            exit(0)

    else:
        print("The location you gave isn't a csv file or a folder of csv files, please run this program again with a valid file name.")
        exit(0)

    return folder, csv_files


def get_variables_from_command_line():
    """
    Function Goal: This function is used to read all of the variables in from the command line arguements
    """

    parser = argparse.ArgumentParser(description="The variables that make this program work")

    # csv path
    parser.add_argument(dest="path", nargs="?", type=str, help="The path to the csv files or the folder of csv files needed in this program.")

    # background image name
    parser.add_argument(dest="background_image_name", nargs="?", type=str, help="The background image that will contain the different areas of the cameras.")

    # output filename
    parser.add_argument('-of', dest="output_filename", default=default_output_filename, nargs="?", type=str, required=False,
                        help="The name of the file that you want the video to be made under.")

    # length of video
    parser.add_argument('-vl', dest="length_of_video", default=default_length_of_video, nargs="?", type=int, required=False,
                        help="How long you want the output video to be.")

    # basewidth
    parser.add_argument('-bw', dest="basewidth", default=default_basewidth, nargs="?", type=int, required=False,
                        help="The width of the base of the background image when scaled.")

    # shapes file
    parser.add_argument('-sf', dest="shapes_filename", default="", nargs="?", type=str, required=False,
                        help="The file containing the a list of the shapes and coordinates to use for the areas drawn on the image.")

    # colourmap_name
    parser.add_argument('-cm', dest="colourmap_name", default=default_colourmap_name, nargs="?", type=str, required=False,
                        help="The name of the colourmap used when changing the colours of the areas.")

    # file of events
    parser.add_argument('-ef', dest="events_file", default="", nargs="?", type=str, required=False,
                        help="The name of the file containing the events that happen during the duration of the video. It features a second and the text describing the event seperated by a space and a new line for each event.")

    # folder of videos
    parser.add_argument('-vf', dest="folder_of_videos", default="", nargs="?", type=str, required=False,
                        help="The name of the folder containing the video footage taken from the cameras that the data in the csvs was taken from.")

    args = parser.parse_args()

    # take the path and turn it into a list of the names of the csv files needed
    folder, csv_files = get_list_of_csv_files(args.path)

    # background
    if args.background_image_name:
        # if a value is input
        try:
            if cv2.imread(args.background_image_name).all() == None:
                pass

        except AttributeError:
            print("\nThis background image does not exist, please run this program again with a valid path to an image.")
            exit(0)

    else:
        # if  a value is not input
        print("\nYou did not enter a path to a background image. Please re-run this program and ensure to enter a valid path to a background image.")
        exit(0)

    # output filename
    if args.output_filename:
        # if a value is input
        output_filename = touch_up_video_name(args.output_filename)

    else:
        # if a value is not input
        print("\nYou did not enter a valid output filename so the default output filename '{}' will be used instead.".format(default_output_filename))
        # set to a default variable
        output_filename = default_output_filename

    # length of video
    if not args.length_of_video:
        # if a value is not input
        print("\nYou did not enter an integer of how long you want the video to be. Please re-run the program entering a valid video length.")
        exit(0)

    # basewidth
    if not args.basewidth:
        # if a value is not input
        print("\nYou did not enter an integer for the baswidth. Please re-run the program entering a valid basewidth.")
        exit(0)

    # shapes filename
    if args.shapes_filename:
        # if value is input
        if os.path.isfile(args.shapes_filename):
            # if the input is a valid file
            try:
                with open(args.shapes_filename, "r") as file:
                    json.load(file)

            except (AttributeError, ValueError):
                print("\nCould not load in the contents of the file containing the list of the shapes and coordinates to use for the areas drawn on the image.")
                print("Please run this program again ensuring the format of the file is standard json format.")
                exit(0)

        else:
            # if the input is not a valid file
            print("\nThe file containing the list of the shapes and coordinates to use for the areas drawn on the image given does not exist.")
            print("Please run this program again with a valid filename.")
            exit(0)

    else:
        # if a value is not input
        print("\nYou did not enter anything for the name of the file containing the list of the shapes and coordinates to use for the areas drawn on the image.")
        print("You will have to re-draw these areas or press 'CTR + C' and re-run the program with a valid filename.")

    # coulourmap name
    if args.colourmap_name:
        # if a value is input
        try:
            # check if the value is valid
            mapper = matplotlib.cm.ScalarMappable(norm=matplotlib.colors.Normalize(vmin=min_sensor_value, vmax=max_sensor_value), cmap=args.colourmap_name)
            colourmap_name = args.colourmap_name

        except ValueError:
            print("\nThe colourmap name entered is not a valid colourmap. Please run this program again, entering a valid colourmap name.")
            exit(0)

    else:
        # if a value is not input
        pass
        

    # file of events
    if args.events_file:
        try:
            with open(args.events_file, "r") as file:
                file.read()

        except FileNotFoundError:
            print("\nThe file containing the events that happen during the duration of the video that you entered is not a valid file name.")
            print(" Please run the program again with a valid file name")
            exit(0)

        except PermissionError:
            print("\nThe file path you entered can not be accessed. Please check that you entered the correct path and re-run the program.")
            exit(0)

        try:
            with open(args.events_file, "r") as file:
                event = file.readline().strip()
                tokens = event.split(" ")
                int(tokens[0])

        except ValueError:
            print("\nThe file containing the events that happen during the duration of the video that you entered is not in the correct format.")
            print("Please run the program again with a valid text file containing 2 tokens per line, a second and then text for an event seperated by a space.")
            exit(0)

    # folder of videos
    if args.folder_of_videos:
        # if value is input
        try:
            list_of_videos = [os.path.join(args.folder_of_videos, f) for f in os.listdir(args.folder_of_videos) if os.path.isfile(os.path.join(args.folder_of_videos, f))]

        except:
            print("\nThe folder_of_videos that you entered is not a valid folder name. Please run the program again with a valid folder name")
            exit(0)

    else:
        list_of_videos = []

    return folder, csv_files, output_filename, args.length_of_video, args.background_image_name, args.basewidth, args.colourmap_name, args.shapes_filename, args.events_file, args.folder_of_videos, list_of_videos


def obtain_variables_from_user():

    # read in data
    print("\nHello, Please enter a path to the csv/folder of csv files of minutes and sensor values?")
    path = input()

    if path:
        # if value is input
        folder, csv_files = get_list_of_csv_files(path)

    else:
        print("You did not enter a path. Please re-run this program and ensure to enter a valid path to a csv/folder of csvs.")
        exit(0)

    # background image
    print("\nPlease enter the path to the image you want to use as the background to the heatmap.")
    background_image_name = input()
    if background_image_name:
        # if value is input
        try:
            if cv2.imread(background_image_name).all() == None:
                pass

        except AttributeError:
            print("\nThe path you supplied is invalid. Please run this program again and enter a valid path to an image.")
            exit(0)

    else:
        print("\nYou did not enter a path to a background image. Please re-run this program and enter a valid path to a background image.")
        exit(0)

    print("\nTo set the following variables to default values: press 'Enter'.")

    # video name
    print("\nPlease enter the name/path you would like the output video to be labeled as")
    x = input()
    if x:
        # if value is input
        output_filename = touch_up_video_name(x)

    else:
        output_filename = default_output_filename

    # length of video
    print("\nThanks again, how long do you want the video to be? (in seconds)")
    read_in = input()
    if read_in:
        # if value is input
        try:
            length_of_video = int(read_in)

        except ValueError:
            print("\nThis value must be an integer, please run this program again with a valid video length.")
            exit(0)

    else:
        length_of_video = default_length_of_video

    # basewidth
    print("\nPlease enter the number that you would like the width of the base of the background image to be when scaled.")
    read_in = input()
    if read_in:
        # if value is input
        try:
            basewidth = int(read_in)

        except ValueError:
            print("\nThis value must be an integer, please run this program again with a valid width.")
            exit(0)

    else:
        basewidth = default_basewidth

    # shapes filename
    print("\nPlease enter the name of the file containing the list of the shapes and coordinates to use for the areas drawn on the image")
    read_in = input()
    if read_in:
        # if value is input
        if os.path.isfile(read_in):

            try:
                with open(args.shapes_filename, "r") as file:
                    json.load(file)

            except AttributeError:
                print("\nCould not load in the contents of the file containing the list of the shapes and coordinates to use for the areas drawn on the image.\nPlease run this program again ensuring the format of the file is standard json format.")
                exit(0)

        else:
            print("\nThe file containing the a list of the shapes and coordinates to use for the areas drawn on the image given does not exist. Please run this program again with a valid filename.")
            exit(0)

    shapes_filename = read_in

    # colour map
    print("\nPlease enter the name of the colourmap you would like to be used.")
    read_in = input()
    if read_in:
        # if value is input
        try:
            matplotlib.cm.ScalarMappable(cmap=read_in)
            colourmap_name = read_in

        except ValueError:
            print("\nThis value must be a valid name of a colour map, please run this program again fixing this error.")
            exit(0)

    else:
        colourmap_name = default_colourmap_name

    # file of events
    print("\nPlease enter the name of the file containing the events that happen during the duration of the video. It should feature a second and the text describing the event seperated by a space and a new line for each event.")
    events_file = input()
    if events_file:
        try:
            with open(events_file, "r") as file:
                file.read()

        except FileNotFoundError:
            print("\nThe file containing the events that happen during the duration of the video that you entered is not a valid file name.")
            print(" Please run the program again with a valid file name")
            exit(0)

        try:
            with open(events_file, "r") as file:
                event = events.readline().strip()
                while event:
                    tokens = event.split(" ")
                    time = int(tokens[0])

        except ValueError:
            print("\nThe file containing the events that happen during the duration of the video that you entered is not in the correct format.")
            print("Please run the program again with a valid text file containing 2 tokens per line, a second and then text for an event seperated by a space.")
            exit(0)

    # folder of videos
    print("\nPlease enter the nam of the folder containing the video footage taken from the cameras that the data in the csvs was taken from.")
    folder_of_videos = input()
    if folder_of_videos:
        try:
            list_of_videos = [os.path.join(folder_of_videos, f) for f in os.listdir(args.folder_of_videos) if os.path.isfile(os.path.join(args.folder_of_videos, f))]

        except FileNotFoundError:
            print("\nThe folder containing the videos that you entered is not a valid folder name. Please run the program again with a valid folder name")
            exit(0)

    print("\nOK, Making the video now, please wait.")

    return folder, csv_files, output_filename, length_of_video, background_image_name, basewidth, colourmap_name, shapes_filename, events_file, folder_of_videos, list_of_videos


def reshape_background_image(img_name, basewidth):
    """
    Function Goal : reshape the image so that it still maintains the same proportion but that its width is the basewidth

    img_name : string - the name of the file containing the image you want to reshape
    basewidth : integer - the width that you want the image to be

    return : 3D numpy array of integers - this array represents the image in its reshaped form in a way that python can deal with.
    """

    img = cv2.imread(img_name)

    height, width = img.shape[:2]

    # calculate the desired hight of the image based on the proportion of the original image and the desired width
    width_percent = (basewidth / float(img.shape[1]))
    height_size = int(img.shape[0] * width_percent)

    reshaped_img = cv2.resize(img, (basewidth, height_size))

    return reshaped_img


def create_array_of_shape(shape_details, img_size):
    """
    Function Goal : Create an array with the same dimensions as the image that the shapes are being drawn on with 1's where the shape should be and 0's everywhere else
                    This function deals with one shape and returns an array containing only 1 shapes mask

    shape_details : dictionary - the dictionary containing the details needed to identifying the shape and its coordinates on the image
    img_size : tuple (int, int, int) - the size of the image that the shapes are being drawn on

    return : the array containing the 1's and 0's, the center point of the 1's
    """

    value_in_mask = [0.5, 0.5, 0.5]

    # sort colours for making areas and outlines
    if colour_of_outline_of_areas == [1, 1, 1]:
        outline_canvas = np.zeros(img_size)
        canvas = np.zeros(img_size)
        background_colour_of_areas = [0, 0, 0]

    elif colour_of_outline_of_areas == [0, 0, 0]:
        outline_canvas = np.ones(img_size)
        canvas = np.ones(img_size)
        background_colour_of_areas = [1, 1, 1]

    else:
        outline_canvas = np.zeros(img_size)
        canvas = np.zeros(img_size)
        background_colour_of_areas = [0, 0, 0]

    if shape_details["type"] == "rectangle":
        start = tuple(shape_details["start"])
        end = tuple(shape_details["end"])
        center = ((start[0] + end[0])/2, (start[1] + end[1])/2)

        # draw 1's on array
        cv2.rectangle(canvas, start, end, color=value_in_mask, thickness=cv2.FILLED)

        # create an array of the outline of the area
        cv2.rectangle(outline_canvas, start, end, color=colour_of_outline_of_areas, thickness=thickness_of_outline_on_areas)

    elif shape_details["type"] == "circle":
        center = tuple(shape_details["centre"])
        radius = shape_details["radius"]

        # draw 1's on array
        cv2.circle(canvas, center, radius, color=value_in_mask, thickness=cv2.FILLED)

        # create an array of the outline of the area
        cv2.circle(outline_canvas, center, radius, color=colour_of_outline_of_areas, thickness=thickness_of_outline_on_areas)

    elif shape_details["type"] == "poly":
        points = shape_details["points"]
        dataframe_means = np.mean(pd.DataFrame(points))
        center = (dataframe_means[0], dataframe_means[1])

        # draw 1's on array
        cv2.fillPoly(canvas, pts=np.int32([points]), color=value_in_mask)

        # create an array of the outline of the area
        cv2.polylines(outline_canvas, pts=np.int32([points]), isClosed=True, color=colour_of_outline_of_areas, thickness=thickness_of_outline_on_areas)

    return canvas, center, outline_canvas, background_colour_of_areas


def turn_the_drawings_into_arrays(list_of_shapes_details, img_size):
    """
    Function Goal : Iterate over the dictionaries, call the function "create_array_of_shapes" and put the created arrays and their centers in a list

    list_of_shapes_details : list of dictionaries - a list of dictionaries containing the details needed to identify the shapes and their coordinates on the image
    img_size : tuple (int, int, int) - the size of the background image that the masks are to be drawn on

    return : list of arrays, list of tuples (int, int)
    """

    list_of_shape_arrays = []
    list_of_outlines_of_shape_arrays = []
    list_of_centers = []
    for shape_details in list_of_shapes_details:
        shape_array, center, outline_of_shape_array, background_colour_of_areas = create_array_of_shape(shape_details, img_size)
        list_of_outlines_of_shape_arrays.append(outline_of_shape_array)
        list_of_shape_arrays.append(shape_array)
        list_of_centers.append(center)

    return list_of_shape_arrays, list_of_centers, list_of_outlines_of_shape_arrays, background_colour_of_areas


def create_a_list_of_dataframes(folder, csv_files):
    """
    Function Goal : Read each csv into a DataFrame with 2 columns, Second and Sensor value, and add the DataFrame to a list

    folder : string - either the name of the folder containing the csv files or an empty string
    csv_files : list of strings - a list of the names of the csvs input

    return : list of DataFrames
    """

    list_of_dfs = []

    for csv in csv_files:

        if folder:
            # read in 2 column data of Minute and sensor value
            df = pd.read_csv(os.path.join(folder, csv), names=["Minute", "Sensor_value"])

        else:
            df = pd.read_csv(csv, names=["Minute", "Sensor_value"])

        # change the minute column to second
        df.Minute = df.Minute * 60
        df = df.rename(columns={"Minute": "Second"})

        list_of_dfs.append(df)

    return list_of_dfs


def turn_dataframe_into_1_second_per_frame(df):
    """
    Function Goal : Change the joined dataframe so that each row is 1 second by averaging out the values where there is more than 1 row for a particular second

    df : DataFrame - a DataFrame containing a Second column and then with columns of the the sensor values from each csv at that second
                     (Second, df1 sensor value, df2 Crowd desnity, .... ect.)

    return : DataFrame - a DataFrame containing a Second column with one Second per row and then with columns of the the sensor values 
                         from each csv at that second
                         (Second, df1 sensor value, df2 Crowd desnity, .... ect.) but with 1 second per row
    """

    new_df = pd.DataFrame(columns=(df.columns))

    min_second_in_df = int(round(min(df.Second-0.5)))
    max_second_in_df = int(round(max(df.Second+0.5)))

    for i in range(max(1, min_second_in_df), max_second_in_df):

        # get the rows between second "i" and second "i + 1"
        rows = df.loc[(df.Second > i - 0.00000001) & (df.Second < i + 0.99999999)]
        rows = rows.reset_index(drop=True)

        if len(rows) == 0:
            # add empty row
            new_df.loc[i] = [i] + np.repeat(float("Nan"), len(df.columns))
            new_df.loc[i].Second = i

        elif len(rows) > 0:
            # average the values in these rows to get just 1 row with the values for each column
            list_of_averages = []
            for mean in rows.loc[:, rows.columns.difference(["Second"])].mean():
                list_of_averages.append(mean)

            # add this row to the new DataFrame
            new_df.loc[i] = [i] + list_of_averages

        # fill the "Nan" values with the value encountered before the "Nan" up untill you hit the point where the rest of that column is only "Nan" values
        filled_new_df = new_df.fillna(method="ffill")

    return filled_new_df


def join_dataframes(list_of_dfs):
    """
    Function Goal : Take the list of DataFrames and join them together column wise on their second column
                    Then populate the values where a particular column doesn't contain any value for sensor value for this Second
                    Populate these values by just using the value for the previous Second

    list_of_dfs : list of DataFrames - a list containing DataFrames with 2 columns, Second and sensor value

    return : DataFrame - a DataFrame containing a Second column with one Second per row and then with columns of the the sensor values 
                         from each csv at that second
                         (Second, df1 sensor value, df2 Crowd desnity, .... ect.)
    """

    join = pd.merge(list_of_dfs[0], list_of_dfs[1], left_on="Second", right_on="Second", how="outer")
    join.set_axis(["Second"] + list(range(2)), axis=1, inplace=True)

    for i in range(2, len(list_of_dfs)):
        join = pd.merge(join, list_of_dfs[i], left_on="Second", right_on="Second", how="outer")
        join.set_axis(join.columns.tolist()[:-1] + [i], axis=1, inplace=True)

    # sort df by second column
    join["Second"] = join["Second"].astype("float")
    join = join.sort_values("Second")
    join = join.reset_index(drop=True)

    for col in join.loc[:, join.columns.difference(["Second"])]:

        # iterate backwards over rows and turn "Nan" boxes to "-1" and find where the "Nan" values stop
        i = len(join[col]) - 1
        while np.isnan(join[col][i]):
            join[col][i] = -1
            i = i - 1

        # iterate forwards over rows and turn "Nan" boxes to "-1" and find where real values start
        j = 0
        while np.isnan(join[col][j]):
            join[col][j] = -1
            j = j + 1

    # fill the "Nan" values with the value encountered before the "Nan" up untill you hit the point where the rest of that column is only "Nan" values
    join = join.fillna(method="ffill")

    # change the joined dataframe so that each row is 1 second by averaging out the values where there is more than 1 row for a particular second
    df_with_one_sec_per_row = turn_dataframe_into_1_second_per_frame(join.iloc[18000:18030, :])

    # replace the "-1" values with "Nan" values
    df_with_one_sec_per_row = df_with_one_sec_per_row.replace(-1.0, float("Nan"))

    return df_with_one_sec_per_row


def calculate_frames_per_sec(number_of_frames_in_the_video, length_of_video):
    """
    Function Goal : To calcuate the amount of frames per second that would have to be shown in order to create a video of the desired length

    number_of_frames_in_the_video : integer - the number of frames worth of data in the input csvs
    length_of_video : integer - the desired length in seconds that you want the video to be

    return : float - the amount of frames per second that would have to be shown in order to create a video of this length
    """

    return number_of_frames_in_the_video / length_of_video


def create_coloured_image_of_shape(sensor_value, mapper, mask_of_shapes_position_on_image, outline_array, background_colour_of_areas):
    """
    Function Goal : turn the mask of 0's and 1's of the shape into a coloured mask of 0's and the sensor value

    sensor_value : integer - the sensor value number for that particular frame
    mapper : the heatmap mapper that maps a number to a colour
    mask_of_shapes_position_on_image : array of integer - the array of 0's and 1's, the mask, of the shape

    return : 3D numpy array of integers, list of 1x3 arrays of integers - an array of a coloured mask of 0's and the sensor values of the shape and a list of the colours
                                                                          of the coloured masks
    """

    array_mask = np.copy(mask_of_shapes_position_on_image)
    outline = np.copy(outline_array)

    if np.isnan(sensor_value):
        # turn to grey
        colour = np.reshape(colour_of_Nan_values, (1, 1, 3))/255

    else:
        # get the colour we want the image to be
        colour = mapper.to_rgba(sensor_value)[:3][::-1]

    coloured_array_mask = np.where(array_mask != background_colour_of_areas, colour, array_mask)

    outlined_coloured_array_mask = np.where(outline == colour_of_outline_of_areas, colour_of_outline_of_areas, coloured_array_mask)

    return outlined_coloured_array_mask, colour


def loop_through_sensor_value_columns_and_return_list_of_coloured_shape_images(df_row_of_sensor_values, list_of_masks_of_shapes, mapper, list_of_outlines_of_shape_arrays, background_colour_of_areas):
    """
    Function Goal : Be given a row from the DataFrame which is a row of the sensor values for 1 frame worth of video for each differnt shape in the list of shapes
                    and to turn the mask for each shape the colour output when the sensor value for its respective csv is put into the mapper

    df_row_of_sensor_values : DataFrame - a row from the DataFrame which is a row of the sensor values for 1 frame worth of video for each differnt csv input
    list_of_masks_of_shapes : list of arrays of integers - a list containing the masks for each shape
    mapper : the heatmap mapper that maps a number to a colour

    return : list of arrays of integers, list of 1x3 arrays of integers - a list containing the coloured masks for each shape and the sensor values of the shape and a list
                                                                          of the colours of the coloured masks
    """

    list_of_coloured_masks_of_shapes = []
    list_of_colours = []
    for col in df_row_of_sensor_values:

        coloured_array_mask, colour = create_coloured_image_of_shape(df_row_of_sensor_values.iloc[0, col], mapper, list_of_masks_of_shapes[col],
                                                                     list_of_outlines_of_shape_arrays[col], background_colour_of_areas)

        list_of_colours.append(colour[::-1])
        list_of_coloured_masks_of_shapes.append(coloured_array_mask)

    return list_of_coloured_masks_of_shapes, list_of_colours


def create_image_of_text_box_at_top(second, dictionary_of_events, x_width, y_height, event_duration_frame):
    """
    Function Goal : Create the event text box for the top of the visualisation

    second : integer - the second that the particular frame is produced at
    dictionary_of_events : dictionary of integer to string {integer : string, integer : string, ..etc.} - This is a dictionary of different integers representing particular
                                                                                                          seconds in the video mapped to an event that happend at that 
                                                                                                          second. The string contains the text to be displayed in the text
                                                                                                          box at the top of the image.
    x_width : integer - the width of the text box along the x-axis
    y_height : integer - the height of the text box on the y-axis
    event_duration_frame : integer - the number of frames either side of the event to display the text for that event

    return : 3D numpy array of integers - an array corrosponding to the text box containing the text about the event
    """

    img = np.ones((y_height, x_width, 3))

    second = int(second)

    seconds = list(range(second - event_duration_frame, second + event_duration_frame + 1))

    for sec in seconds:
        if sec in dictionary_of_events:

            text = dictionary_of_events[sec]

            text_width, text_height = cv2.getTextSize(text, font_type_of_top_text, font_size_of_top_text, font_thickness_of_top_text)[0]

            if text_width > x_width:
                print("The event name '" + text + "'' is too long for the text bar at the top, please input a shorter description of the event and re-run the program.")
                # exit(0)

                """
                # this was written to try to sort the problem when the text of the event is too long for the bar
                #print(text_width)
                #print(x_width)
                num_times_to_add_newline = int(text_width/x_width)
                print(num_times_to_add_newline)
                length_per_line = int(text_width/(num_times_to_add_newline + 1))
                #print(length_per_line)
                new_text = ""
                for i in range(1, num_times_to_add_newline + 1):
                    print(new_text)
                    print(text[(length_per_line * (i-1)):(length_per_line * i)])
                    new_text = new_text + text[(length_per_line * (i-1)):(length_per_line * i)] + '\n'
                    print(new_text)

                text_width, text_height = cv2.getTextSize(new_text, font_type_of_top_text, font_size_of_top_text, font_thickness_of_top_text)[0]
                """

            # this scale from bottom to top goes in descending order
            start_y = int(y_height/2 + text_height/2)
            start_x = int(x_width/2 - text_width/2)

            cv2.putText(img, text, (start_x, start_y), font_type_of_top_text, font_size_of_top_text, colour_of_top_text, lineType=line_type_of_top_text,
                        thickness=font_thickness_of_top_text)

    bordered_image = cv2.copyMakeBorder(img, top=text_box_image_border_width, bottom=text_box_image_border_width, left=text_box_image_border_width,
                                        right=text_box_image_border_width, borderType=text_box_image_border_type, value=text_box_image_border_colour)

    return bordered_image


def fig_to_img(fig):
    """
    Function Goal : Turn a matplotlib figure into a BGRA image

    adapted from: http://www.icare.univ-lille1.fr/tutorials/convert_a_matplotlib_figure

    fig : matplotlib figure

    return : 3D numpy array of integers - the image drawn from the matplotlib figure
    """

    # draw the renderer
    fig.canvas.draw()

    # Get the RGBA buffer from the figure
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8).reshape(h, w, 4)

    # ARGB -> BGR
    return buf[:, :, ::-1][:, :, :3]/255


def create_bar_plot_image(row, x_width, y_height, names, list_of_colours_of_bars):
    """
    Function Goal : take a row of sensor values and make a bar plot from these integers

    row : DataFrame of integers - this is a single row from a DataFrame that contains the sensor values corrosponding to a particular second in the larger DataFrame
    x_width : integer - the width on the x-axis to make the bar plot
    y_height : integer - the height on the y-axis to make the bar plot
    names : list of strings [str, str, ...etc.] - a list containing the names of the cameras to put on the bar plot

    return : an 3D numpy array of integers - an array corrosponding to an image of the bar plot
    """

    fig = plt.figure()
    plt.subplot(title="The {} in the different areas.".format(sensor_value_name), xlabel="Areas", ylim=(min_sensor_value, max_sensor_value), ylabel=sensor_value_name)
    plt.bar(names, list(row.iloc[0]), color=list_of_colours_of_bars)

    img = fig_to_img(fig)
    plt.close()

    image = cv2.resize(img, (x_width, y_height))

    bordered_image = cv2.copyMakeBorder(image, top=bar_plot_border_width, bottom=bar_plot_border_width, left=bar_plot_border_width, right=bar_plot_border_width,
                                        borderType=bar_plot_border_type, value=bar_plot_border_colour)

    return bordered_image


def cartesian_coordinates_to_polar_coordinates(point):
    """
    Function Goal : Turn a point into polar coordinates based on its location with respect to the origin (0,0)

    point : a tuple of integers (int, int) - a point on the x-y plane

    return : float, float - the distance between the point and the origin and the angle between a line from the point to the origin and the x axis
    """

    rho = np.sqrt(point[0]**2 + point[1]**2)
    phi = np.arctan2(point[1], point[0])

    return rho, phi


def polar_coordinates_to_cartesian_coordinates(rho, phi):
    """
    Function Goal : Turn the polar coordinates into cartesian coordinates

    rho : float - The distance from the output point to the origin (0,0)
    phi : float - The angle between the a line from the output point to the origin and a line along the x-axis

    return : a tuple of integers (int, int) - a point on the x-y plane
    """

    x = rho * np.cos(phi)
    y = rho * np.sin(phi)

    return (int(x), int(y))


def get_point_that_divides_2_points_into_interval(p1, p2, a, b):
    """
    Function Goal : find the point that devides 2 points into the ratio a:b where dist(p1, point):dist(point, p2) is proportional to a:b

    p1 : tuple of integers (int, int) - a point
    p2 : tuple of integers (int, int) - a point
    a : integer - ratio a
    b : integer - ratio b

    return : tuple of integers (int, int) - the point that devides the 2 points into the ration a:b
    """

    x = ((b * p1[0]) + (a * p2[0]))/(a + b)
    y = ((b * p1[1]) + (a * p2[1]))/(a + b)

    return (int(x), int(y))


def dist_between_2_points(p1, p2):
    """
    Function Goal : get the distance between 2 points

    p1 : tuple of integers (int, int) - a point
    p2 : tuple of integers (int, int) - a point

    return : intager - th distance between p1 and p2
    """

    return np.hypot(p1[0] - p2[0], p1[1] - p2[1])


def get_slope(p1, p2):
    """
    Function Goal : Get the slope of a line going through 2 points

    p1 : tuple of integers (int, int) - a point
    p2 : tuple of integers (int, int) - a point

    return : integer - the slope of the line going through the 2 points
    """

    return (p2[1] - p1[1])/(p2[0] - p1[0])


def generate_all_points_on_outside_of_shape(corners):
    """
    Function Goal : To take the corners of the shape drawn and to use these to generate all the points that are on the outside permimeter of the shape drawn

    corners : list of tuples of integers [(int, int), (int, int), ...etc.] - a list of the points on the corners of the shape that was drawn

    return : list of tuples of integers [(int, int), (int, int), ...etc.] - a list of points that surround the whole area
    """

    points = []
    start_x, start_y = corners[0]
    for x, y in corners[1:] + corners[:1]:

        if start_x == x:
            # line goes along x axis (horizonal)

            for new_y in range(min(start_y, y), max(start_y, y)):
                points.append((x, new_y))

        elif start_y == y:
            # line goes along the y axis (verticle)

            for new_x in range(min(start_x, x), max(start_x, x)):
                points.append((new_x, y))

        else:
            # line is slanted diagonally either upwards from left to right or downwards from left to right
            slope = get_slope((start_x, start_y), (x, y))

            length_on_x_axis = max(start_x, x) - min(start_x, x)

            for i in range(num_points_on_line_for_outside_of_shape):

                x_val = min(start_x, x) + (i * length_on_x_axis) / num_points_on_line_for_outside_of_shape

                new_y = (slope * x_val) - (slope * start_x) + start_y

                points.append((int(x_val), int(new_y)))

        start_x = x
        start_y = y

    return points


def get_equation_of_line(point1, point2):
    """
    Functon Goal : This function gets the coefficients and the constants of the equation of a line passing through 2 points

    point1 : a tuple of integers (int, int) - a point on the cartesion plane
    point2 : a tuple of integers (int, int) - a point on the cartesion plane

    return : a tuple of a list of integers and an integer ([int, int], int) - a list containing the coefficients for the x and y values in the equation of the line and the constant in the equation of the line
    """

    slope = get_slope(point1, point2)

    x1, y1 = point1
    x2, y2 = point2

    coefficients_of_x_nd_y = [-slope, 1]
    constant = y1 - (slope * x1)

    return coefficients_of_x_nd_y, constant


def get_closest_point(corners, camera_point, center):
    """
    Function Goal : get a list of all the points on the outside of shape and find the closest point on the outside to the point given

    corners : list of tuples of intrest [(int, int), (int, int), ...etc.] - a list of the corner points of the rectangle/polygon
    point : tuple of integers (int, int) - a point

    return : the closest point on the outside of the shape to the point given
    """

    closest = ""
    closest_dist_to_line = np.inf
    list_of_points_on_outside = generate_all_points_on_outside_of_shape(corners)

    cam_point = int(camera_point[0]), int(camera_point[1])
    center_point = int(center[0]), int(center[1])

    coefficients_of_x_nd_y, constant = get_equation_of_line(center_point, cam_point)

    for point in list_of_points_on_outside:

        if (center[0] < camera_point[0] and center[0] < point[0] < camera_point[0]) or (center[0] > camera_point[0] and center[0] > point[0] > camera_point[0]):

            intercept_of_points = np.array(coefficients_of_x_nd_y).dot(np.array([point[0], point[1]]))

            if np.abs(constant - intercept_of_points) < closest_dist_to_line:
                closest = point
                closest_dist_to_line = np.abs(constant - intercept_of_points)

    return closest


def draw_arrows_from_cameras_to_shapes(image, list_of_shapes_details, list_of_camera_image_midpoints, width_to_move, height_to_move):
    """
    Function Goal : Take an array corrosponding to the image, change some values so that when it is turned to an image, arrows are drawn between the shapes on the image
                    and the boxes containing the camera footage

    image : 3D numpy array of integers - the array that corrosponds to one frame of the video
    list_of_shapes_details : list of dictionaries - a list of dictionaries containing the details needed to identify the shapes and their coordinates on the image
    list_of_camera_image_midpoints : list of tuples of integers [(int, int), (int, int), ...etc.] - a list of points. These points are the coordinates of the midpoints of the edges of the boxes 
                                     containing camera footage
    width_to_move : integer - the value to add to the x value of the points on the shapes to account for the fact I added in the camera footage on the LHS and RHS
    height_to_move : integer - the value to add to the y value of the points on the shapes to account for the fact I added in the colourmap and text box on the 
                               top and bottom of the image

    return : 3D numpy array of integers - the array that corrosponds to one frame of the video that includes the arrows draw on the image
    """

    for i in range(len(list_of_camera_image_midpoints)):
        shape_details = list_of_shapes_details[i]

        if shape_details["type"] == "rectangle":
            start = tuple(shape_details["start"])
            end = tuple(shape_details["end"])

            moved_start = (start[0] + width_to_move, start[1] + height_to_move)
            moved_end = (end[0] + width_to_move, end[1] + height_to_move)

            moved_center = ((moved_start[0] + moved_end[0])/2, (moved_start[1] + moved_end[1])/2)

            start_x_nd_end_y = (moved_start[0], moved_end[1])
            end_x_nd_start_y = (moved_end[0], moved_start[1])

            moved_corners = [moved_start, start_x_nd_end_y, moved_end, end_x_nd_start_y]

            closest = get_closest_point(moved_corners, list_of_camera_image_midpoints[i], moved_center)

        elif shape_details["type"] == "poly":
            corners = shape_details["points"]

            moved_corners = []
            for corner in corners:
                moved_corners.append((corner[0] + width_to_move, corner[1] + height_to_move))

            dataframe_means = np.mean(pd.DataFrame(moved_corners))
            moved_center = (int(dataframe_means[0]), int(dataframe_means[1]))

            closest = get_closest_point(moved_corners, list_of_camera_image_midpoints[i], moved_center)

        elif shape_details["type"] == "circle":
            center = tuple(shape_details["centre"])
            radius = shape_details["radius"]

            moved_center = (center[0] + width_to_move, center[1] + height_to_move)

            distance = dist_between_2_points(moved_center, list_of_camera_image_midpoints[i])

            closest = get_point_that_divides_2_points_into_interval(moved_center, list_of_camera_image_midpoints[i], radius, distance-radius)

        # draw line for the arrow between the edge of the area to the camera footage frame
        cv2.line(image, list_of_camera_image_midpoints[i], closest, colour_of_arrow_between_camera_nd_area, thickness=lineThickness_of_arrow_between_camera_nd_area,
                 lineType=lineType_of_arrow_between_camera_nd_area)

        # calculate the angle that the arrow head needs to be
        point_relative_to_point_on_shape = (list_of_camera_image_midpoints[i][0] - closest[0], list_of_camera_image_midpoints[i][1] - closest[1])

        rho, pi = cartesian_coordinates_to_polar_coordinates(point_relative_to_point_on_shape)

        new_angles = [pi - angle_to_change_line_for_arrow_head, pi + angle_to_change_line_for_arrow_head]

        # draw the lines for the arrow head
        for angle in new_angles:
            x, y = polar_coordinates_to_cartesian_coordinates(length_of_arrow_head_lines, angle)

            cv2.line(image, (closest[0] + x, closest[1] + y), closest, colour_of_arrow_between_camera_nd_area,
                     thickness=lineThickness_of_arrow_between_camera_nd_area, lineType=lineType_of_arrow_between_camera_nd_area)

    return image


def get_list_of_camera_image_midpoints(first_x, distance_between_first_nd_second, num_of_images_on_lhs, num_of_images_on_rhs, total_height):
    """
    Function goal : create a list of points which corrospond to the coordinates of the midpoints of the edges of the boxes containing camera footage

    first_x : integer - the distance along the x axis between the most left point and the edge of the first set of camera footage videos
    distance_between_first_nd_second : integer - the distance along the x axis between the edge of the first set of camera footage videos and the start of the second
                                                 set of camera footage video
    num_of_images_on_rhs : integer - the number of camera footage videos on the right hand side of the middle heatmap image
    total_height : integer - the total height along the y axis of the whole video

    return : list of tuples of integers [(int, int), (int, int), ...etc.] - a list of points. These points are the coordinates of the midpoints of the edges of the boxes 
                                                                            containing camera footage
    """

    left_token_height = int(total_height/(2 * (num_of_images_on_lhs + 1)))
    right_token_height = int(total_height/(2 * num_of_images_on_rhs))

    list_of_camera_image_midpoints = []
    for i in range(num_of_images_on_lhs + num_of_images_on_rhs):
        if i < num_of_images_on_lhs:
            # videos on the left hand side
            x = first_x
            y = ((2 * i) + 1) * left_token_height

        elif i >= num_of_images_on_lhs and i < num_of_images_on_lhs + num_of_images_on_rhs:
            #  videos on the right hand side
            x = first_x + distance_between_first_nd_second
            y = ((2 * (i - num_of_images_on_lhs)) + 1) * right_token_height

        # add the point to a list
        list_of_camera_image_midpoints.append((x, y))

    return list_of_camera_image_midpoints


def merge_lhs_and_rhs_frames(frames, num_of_images_on_lhs, bar_plot_image):
    """
    Function goal : Take the list of the arrays corrosponding to a frame from the different videos and seperate these into the images that will go on the LHS and the RHS, 
                    then merge all the LHS and RHS images, put a border around them and then add the bar plot to the LHS image.

    frames : A list of 3D numpy arrays [Array, Array, etc...] - A list of arrays of the different images from the different videos that corrosponds to 1 frame of video
                                                                The length of this list is the number of videos that will feature on the frame
    num_of_images_on_lhs : integer - the number of images on the left hand side of the main heatmap image
    bar_plot_image : 3D numpy array of integers - this array represents the image of the bar plot

    return : 3D numpy arrays of integers => Array, Array - the left array is an array corrosponding to the image that will go to the left of the main heatmap image
                                                           the right array is an array corrosponding to the image that will go to the right of the main heatmap image
    """

    import time
    start_slice = time.time()
    lhs_list = frames[:num_of_images_on_lhs]
    rhs_list = frames[num_of_images_on_lhs:]
    slice_list.append(time.time() - start_slice)

    start_concat_inside = time.time()
    rhs_img = np.concatenate(rhs_list)
    lhs_img = np.concatenate(lhs_list)
    concat_inside_list.append(time.time() - start_concat_inside)

    start_border = time.time()
    bordered_rhs_image = cv2.copyMakeBorder(rhs_img, top=video_footage_border_width, bottom=video_footage_border_width, left=video_footage_border_width,
                                            right=video_footage_border_width, borderType=video_footage_border_type, value=video_footage_border_colour)
    bordered_lhs_image = cv2.copyMakeBorder(lhs_img, top=bar_plot_border_width, bottom=bar_plot_border_width, left=bar_plot_border_width,
                                            right=bar_plot_border_width, borderType=bar_plot_border_type, value=bar_plot_border_colour)
    border_list.append(time.time() - start_border)

    try:
        bordered_lhs_image = np.concatenate((bordered_lhs_image, bar_plot_image))

    except ValueError:
        print("Error, line 1393")

    return bordered_lhs_image, bordered_rhs_image


def video_to_frames(read_videos, width, total_height):
    """
    Function Goal : read in one frame from the videos and resize them to a specific width and all to the same height.

    read_videos : list of video read ins - a list of the variables for each different video to read in the next frame from
    width : integer - the width to resize the images to

    return : a List of 3D numpy arrays of images and an integer => [Array, array, etc..], integer - a list of the resized frames and the height of each frame
    """

    height_of_frame = int(total_height/int((len(read_videos)/2)))

    # read in the images and resize them
    resized_frames = []
    for i in range(len(read_videos)):
        # for each video

        import time
        #start_read = time.time()

        try:
            start_read = time.time()
            success, image = read_videos[i].read()

            if success == False:
                raise StopIteration

            #image = next(read_videos[i])

            read_in_list.append(time.time() - start_read)

        except StopIteration:
            image = np.zeros((1, 1, 3))

        img = cv2.resize(image, (width, height_of_frame))/255

        if np.array_equal(image, np.zeros((1, 1, 3))):

            text_when_video_finishes = "No Video"

            text_width, text_height = cv2.getTextSize(text_when_video_finishes, font_type_of_camera_footage_text, font_size_of_camera_footage_text,
                                                      font_thickness_of_camera_footage_text)[0]

            height = int(total_height/2 - text_height/2)
            start_x = int(width/2 - text_width/2)

            cv2.putText(img, text_when_video_finishes, (start_x, height), font_type_of_camera_footage_text, font_size_of_camera_footage_text,
                        colour_of_camera_footage_text, lineType=line_type_of_camera_footage_text, thickness=font_thickness_of_camera_footage_text)

        resized_frames.append(img)

    return resized_frames, height_of_frame


def turn_numbers_into_abreviations(num):
    """
    Function Goal : take a number and turn it into a string of that number with only 3 digits and using abbrevations such a K,M,B,T to shorten its length

    num : integer - a number

    return : string - the number in its abbreviated form
    """

    num_digits = len(str(num))

    if 3 < num_digits < 7:
        new_num = "{:.3g}".format(num/1000) + "K"

    elif 6 < num_digits < 10:
        new_num = "{:.3g}".format(num/1000000) + "M"

    elif 9 < num_digits < 13:
        new_num = "{:.3g}".format(num/1000000000) + "B"

    elif 12 < num_digits < 16:
        new_num = "{:.3g}".format(num/1000000000000) + "T"

    else:
        new_num = "{:.3g}".format(num)

    return new_num


def create_colourmap(size, mapper, num_dividers):
    """
    Function Goal : create the colourmap 

    size : tuple of integers (int, int, int) - the size (height & width & colour) of the colourmap that we are going to create
    mapper : the heatmap mapper that maps a number to a colour
    num_dividers : integer - the number of values on the scale of the colourmap

    return : a 3D numpy array of integers - an array that corrosponds to the colourmap image that wwas created
    """

    inner_colourmap_horizontal_border_width = int(size[1] * proportion_of_width_to_make_inner_colourmap_horizontal_border)
    space_between_line_and_text = int(size[0] * proportion_of_height_to_make_space_between_line_and_text)
    inner_colourmap_bottom_border_height = int(size[0] * proportion_of_height_to_make_inner_colourmap_bottom_border)
    inner_colourmap_top_border_height = int(size[0] * proportion_of_height_to_make_inner_colourmap_top_border)
    height_of_box_for_indexes = int(size[0] * proportion_of_colourmap_image_to_make_indexes)
    heading_box_height = int(size[0] * proportion_of_colourmap_image_to_make_heading)

    height_of_inner_colmap = size[0] - inner_colourmap_bottom_border_height - inner_colourmap_top_border_height - height_of_box_for_indexes - heading_box_height

    width_of_inner_colmap = size[1] - (2 * inner_colourmap_horizontal_border_width)

    starting_y_coord_of_inner_colmap = size[0] - height_of_inner_colmap - inner_colourmap_bottom_border_height

    colmap_image = np.ones(size)

    # put spectrum of colours on colourmap colours map
    sensor_value_range = max_sensor_value - min_sensor_value
    step = sensor_value_range/(width_of_inner_colmap-1)
    i = inner_colourmap_horizontal_border_width
    sensor_value = min_sensor_value
    while sensor_value < max_sensor_value:
        colour = mapper.to_rgba(sensor_value)[:3][::-1]

        colmap_image[starting_y_coord_of_inner_colmap:starting_y_coord_of_inner_colmap + height_of_inner_colmap, i, :] = colour

        sensor_value += step
        i = i + 1

    # add lines for the scale on the map
    width_of_inner_colourmap = size[1] - (2 * inner_colourmap_horizontal_border_width)
    list_of_coordinates_to_draw_lines = []
    colourmap_divider = width_of_inner_colourmap/num_dividers

    # get a list of the x-coordinates to draw the lines at
    for i in range(0, num_dividers + 1):
        coord = int(i * colourmap_divider) + inner_colourmap_horizontal_border_width
        list_of_coordinates_to_draw_lines.append(coord)

    start_y_coord_of_line_for_scale = heading_box_height + height_of_box_for_indexes + space_between_line_and_text

    # draw the lines along these x-coordinate lines
    for coord in list_of_coordinates_to_draw_lines:
        colmap_image[start_y_coord_of_line_for_scale:size[0] - inner_colourmap_bottom_border_height, coord-1:coord+1, :] = colour_of_lines_for_scale_on_colourmap

    # add the scale for the colourmap

    list_of_index_values = []
    sensor_value_divider = sensor_value_range/num_dividers

    for i in range(0, num_dividers + 1):
        abreviated_num = turn_numbers_into_abreviations(int(i * sensor_value_divider) + min_sensor_value)
        list_of_index_values.append(abreviated_num)

    for i in range(len(list_of_index_values)):

        index = list_of_index_values[i]
        index_width, index_height = cv2.getTextSize(index, font_type_of_index_on_colourmap, font_size_of_index_on_colourmap, font_thickness_of_index_on_colourmap)[0]

        start_y_of_index = height_of_box_for_indexes + heading_box_height
        start_x_of_index = int(list_of_coordinates_to_draw_lines[i] - index_width/2)

        cv2.putText(colmap_image, index, (start_x_of_index, start_y_of_index), font_type_of_index_on_colourmap, font_size_of_index_on_colourmap,
                    colour_of_index_on_colourmap, lineType=line_type_of_index_on_colourmap, thickness=font_thickness_of_index_on_colourmap)

    # draw black line
    colmap_image[heading_box_height+1, :, :] = colour_of_line_seperating_colmap_header_and_colmap

    heading = "{} in each area".format(sensor_value_name)

    heading_width, heading_height = cv2.getTextSize(heading, font_type_of_heading_on_colourmap, font_size_of_heading_on_colourmap,
                                                    font_thickness_of_heading_on_colourmap)[0]

    start_x_of_heading = int(size[1]/2 - heading_width/2)
    start_y_of_heading = int(heading_box_height/2 + heading_height/2)

    cv2.putText(colmap_image, heading, (start_x_of_heading, start_y_of_heading), font_type_of_heading_on_colourmap, font_size_of_heading_on_colourmap,
                colour_of_heading_on_colourmap, lineType=line_type_of_heading_on_colourmap, thickness=font_thickness_of_heading_on_colourmap)

    # border colourmap
    border_colmap = cv2.copyMakeBorder(colmap_image, top=colmap_border_width, bottom=colmap_border_width, left=colmap_border_width, right=colmap_border_width,
                                       borderType=colmap_border_type, value=colmap_border_colour)

    return border_colmap


def create_image_of_second(second, x_width, y_height):
    """
    Function Goal : Take a integer second and create an array corrosponding to an image that is a particular width and height that contains the second fed in

    second : integer - the second that the particular frame is produced at
    x_width : integer - the width along the x-axis to make the array
    y_height : integer - the height along the y-axis to make the array

    return : a 3D numpy array of integers - this array corosponds to the image of a particular width and height that contains the integer second given
    """

    img = np.ones((y_height, x_width, 3))

    text = '%05d' % second

    text_width, text_height = cv2.getTextSize(text, font_type_of_second_image, font_size_of_second_image, font_thickness_of_second_image)[0]

    start_y = int(y_height/2 + text_height/2)
    start_x = int(x_width/2 - text_width/2)

    cv2.putText(img, text, (start_x, start_y), font_type_of_second_image, font_size_of_second_image, colour_of_second_image_text,
                lineType=line_type_of_second_image, thickness=font_thickness_of_second_image)

    bordered_image = cv2.copyMakeBorder(img, top=second_image_border_width, bottom=second_image_border_width, left=second_image_border_width,
                                        right=second_image_border_width, borderType=second_image_border_type, value=second_image_border_colour)

    return bordered_image


def merge_to_one_image(list_of_images, background, list_of_centers, names, background_colour_of_areas):
    """
    Function Goal : Take a list of arrays corrosponding to the images of the areas on the background and merge these arrays so that this array corrosponds to one image 
                    of all the different areas and then add this image to the background to create one image

    list_of_images : list of arrays of integers - la ist of arrays corrosponding to the images of the areas and their locations on the background 
    background : 3D numpy array of integers - an array that corrosponds to the background image of the shapes
    list_of_centers : a list of tuples of integers [(int, int), (int, int), ...etc.] -  a list containing the center points of each of the shapes
    names : list of strings [str, str, ...etc.] - a list containing the names of the cameras to put on the bar plot

    return : 3D numpy array of integers - this array corrosponds to the image of the differnt areas all overlayed on th background image
    """

    background_image = np.copy(background).astype(np.float64, copy=False)/255

    if background_colour_of_areas == [1, 1, 1]:
        canvas_of_just_the_shapes = np.ones((background.shape))

    elif background_colour_of_areas == [0, 0, 0]:
        canvas_of_just_the_shapes = np.zeros((background.shape))

    # create blank image with just the areas on it
    for img in list_of_images:

        # if the element of the blank canvas is already changed get the mean of the 2
        canvas_of_just_the_shapes = np.where((img != background_colour_of_areas) & (canvas_of_just_the_shapes != background_colour_of_areas),
                                             np.average(np.array([canvas_of_just_the_shapes, img]), axis=0), canvas_of_just_the_shapes)

        # if the element of the blank canvas is still blank
        canvas_of_just_the_shapes = np.where((img != background_colour_of_areas) & (canvas_of_just_the_shapes == background_colour_of_areas), img,
                                             canvas_of_just_the_shapes)

    # make a new array with the shapes on top of the background image
    non_transparent_background_nd_shapes = np.where(canvas_of_just_the_shapes != background_colour_of_areas, canvas_of_just_the_shapes, background_image)

    transparent_background_nd_shapes = cv2.addWeighted(non_transparent_background_nd_shapes, alpha_for_transperancy_of_shapes_on_background, background_image,
                                                       1-alpha_for_transperancy_of_shapes_on_background, gamma_for_merging_shapes_and_background)

    for i in range(len(list_of_centers)):

        text = names[i]

        text_width, text_height = cv2.getTextSize(text, font_type_of_shape_name_text, font_size_of_shape_name_text, font_thickness_of_shape_name_text)[0]

        start_x_of_shape_name_text = int(list_of_centers[i][0] - text_width/2)
        height_of_shape_name_text = int(list_of_centers[i][1] + text_height/2)

        cv2.putText(transparent_background_nd_shapes, text, (start_x_of_shape_name_text, height_of_shape_name_text), font_type_of_shape_name_text,
                    font_size_of_shape_name_text, colour_of_shape_name_text, lineType=line_type_of_shape_name_text, thickness=font_thickness_of_shape_name_text)

    bordered_transparent_background_nd_shapes = cv2.copyMakeBorder(transparent_background_nd_shapes, top=merged_image_border_width, bottom=merged_image_border_width,
                                                                   left=merged_image_border_width, right=merged_image_border_width,
                                                                   borderType=merged_image_border_type, value=merged_image_border_colour)

    return bordered_transparent_background_nd_shapes


def turn_all_the_different_images_into_one_image(list_of_coloured_shapes, background, colourmap_image, df_of_row, x_width_of_second_image, y_height_of_second_image,
                                                 width_of_left_and_right_images, dictionary_of_events, event_duration_frame, names, list_of_centers, read_videos,
                                                 num_of_images_on_lhs, list_of_shapes_details, height_of_text_box, list_of_colours, background_colour_of_areas):
    """
    Function Goal : Get the images of the backround, the shapes, the colourmap, the frame number, the camera footage videos and the bar plot image and merge these images 
                    together to form one singular image

    list_of_coloured_shapes : a list 3D numpy arrays of integers - the list containing the coloured masks of the shapes drawn on the background
    background : 3D numpy Arrays of integers - an array that corrosponds to the background image of the shapes
    colourmap_image : 3D numpy array of integers - an array that corrosponds to the image of the colourmap
    df_of_row : DataFrame of integers - this is a single row from a DataFrame that contains a second and the sensor values corrosponding to that particular second 
                                        in the larger DataFrame
    x_width_of_second_image : the width along the x-axis to make the array
    y_height_of_second_image : the height along the y-axis to make the array
    width_of_left_and_right_images : the width along the x-axis to make the camera footage videos on either side of the main heatmap image
    dictionary_of_events : dictionary of integer to string {integer : string, integer : string, ..etc.} - This is a dictionary of different integers representing particular
                                                                                                          seconds in the video mapped to an event that happend at that 
                                                                                                          second. The string contains the text to be displayed in the text
                                                                                                          box at the top of the image.
    event_duration_frame : integer - the number of frames either side of the event to display the text for that event
    names : list of strings [str, str, ...etc.] - a list containing the names of the cameras to put on the bar plot
    list_of_centers : a list of tuples of integers [(int, int), (int, int), ...etc.] -  a list containing the center points of each of the shapes
    read_videos : list of the camera footage videos read ins - a list of the variables for each different video to read in the next frame from
    num_of_images_on_lhs : integer - the number of camera footage videos on the left hand side of the middle heatmap image
    list_of-shapes_details : list of dictionaries - a list of dictionaries containing the details needed to identify the shapes and their coordinates on the image
    height_of_text_box : the height of the text box on the y-axis

    return : 3D numpy array of integers - an array corrosponding to the image which is made up of all the the different images to be featured in the video merged together
    """

    import time
    start_other = time.time()

    merge_heatmap_start = time.time()
    # merge background and coloured shape images
    shapes_nd_background_image = merge_to_one_image(list_of_coloured_shapes, background, list_of_centers, names, background_colour_of_areas)

    merge_heatmap.append(time.time() - merge_heatmap_start)

    if dictionary_of_events:#

        # create an image for the text box at the top
        text_box = create_image_of_text_box_at_top(df_of_row.Second.iloc[0], dictionary_of_events, background.shape[1], height_of_text_box, event_duration_frame)

        shapes_nd_background_image = np.concatenate((text_box, shapes_nd_background_image))

    # create the image of the second
    second_image = create_image_of_second(df_of_row.Second.iloc[0], x_width_of_second_image, y_height_of_second_image)

    # merge the colourmap and the second images
    colmap_nd_second = np.concatenate((colourmap_image, second_image), axis=1)

    # merge the colormap and second image with the background image
    main_heatmap_image = np.concatenate((shapes_nd_background_image, colmap_nd_second))

    import time
    shapes_list.append(time.time() - start_other)

    start_video = time.time()

    # if there are camera footage videos to put on the sides of the heatmap video
    if read_videos:

        import time
        start_video_2_frames = time.time()

        # get a list of 1 frame from each video and reshape them
        frames, height_of_frame = video_to_frames(read_videos, width_of_left_and_right_images, main_heatmap_image.shape[0])

        video_2_frames_list.append(time.time() - start_video_2_frames)

        import time
        start_bar = time.time()

        # create bar plot
        bar_plot_image = create_bar_plot_image(df_of_row.iloc[:, 1:], width_of_left_and_right_images, height_of_frame, names, list_of_colours)

        bar_list.append(time.time() - start_bar)

        import time
        start_merge = time.time()

        # crate the left and right images from the frames of the videos
        lhs_img, rhs_img = merge_lhs_and_rhs_frames(frames, num_of_images_on_lhs, bar_plot_image)

        merge_list.append(time.time() - start_merge)

        import time
        start_resize = time.time()

        # resize the left and right images
        lhs_img = cv2.resize(lhs_img, (lhs_img.shape[1], main_heatmap_image.shape[0]))
        rhs_img = cv2.resize(rhs_img, (rhs_img.shape[1], main_heatmap_image.shape[0]))

        resize_list.append(time.time() - start_resize)

        start_concat = time.time()

        # merge the lhs images, the main heatmap image and the rhs images
        fully_merged_image = np.concatenate((lhs_img, main_heatmap_image, rhs_img), axis=1)

        # draw arrows on the images joining the camera footage videos with their respective area on the heatmap
        list_of_camera_image_midpoints = get_list_of_camera_image_midpoints(width_of_left_and_right_images, shapes_nd_background_image.shape[1],
                                                                            num_of_images_on_lhs, len(frames) - num_of_images_on_lhs, fully_merged_image.shape[0])

        final_image = draw_arrows_from_cameras_to_shapes(fully_merged_image, list_of_shapes_details, list_of_camera_image_midpoints, width_of_left_and_right_images,
                                                         height_of_text_box)

        concat_outside_list.append(time.time() - start_concat)

    else:
        # create bar plot
        bar_plot_image = create_bar_plot_image(df_of_row.iloc[:, 1:], width_of_left_and_right_images, main_heatmap_image.shape[0] - (2 * bar_plot_border_width),
                                               names, list_of_colours)

        # merge the main heatmap image and the bar plot
        final_image = np.concatenate((bar_plot_image, main_heatmap_image), axis=1)

    import time
    video_list.append(time.time() - start_video)

    return final_image


def write_an_image_to_a_video(writer, img):
    """
    Function Goal : take an array corrosponding to an image and write this image to a video so that it is one frame of the video

    writer : all the details to do with writing the images to the video
    img : 3D numpy array of integers - the array that corrosponds to the image representing one frame of the video

    return : None
    """

    writer.write(np.uint8(img * 255))


def write_an_image_to_a_folder(output_foldername, img):
    """
    Function Goal : take an array corrosponding to an image and write this image to a folder

    output_foldername : string - the name of the folder that you want to write the images to
    img : 3D numpy array of integers - the array that corrosponds to the image representing one frame of the video

    return : None
    """

    cv2.imwrite(output_foldername, img*255)


def main():
    import time
    start_time = time.time()
    print(start_time)

    # read in the variables
    if len(sys.argv) > 1:
        folder, csv_files, output_filename, length_of_video, background_image_name, basewidth, colourmap_name, shapes_filename, events_file, folder_of_videos, list_of_videos = get_variables_from_command_line()
        full_read_time = 0

    else:
        start_read_time = time.time()
        folder, csv_files, output_filename, length_of_video, background_image_name, basewidth, colourmap_name, shapes_filename, events_file, folder_of_videos, list_of_videos = obtain_variables_from_user()

        # folder, csv_files, output_filename, length_of_video, background_image_name, basewidth, colourmap_name, shapes_filename, events_file, folder_of_videos, list_of_videos = set_variables_using_folder()
        # folder, csv_files, output_filename, length_of_video, background_image_name, basewidth, colourmap_name, shapes_filename, events_file, folder_of_videos, list_of_videos = set_variables_using_file()

        full_read_time = time.time() - start_read_time

    names = []
    for csv in csv_files:
        names.append(csv[:-10])

    # draw the areas on the image
    if shapes_filename:
        full_draw_time = 0
        with open(shapes_filename, "r") as file_of_shapes:
            list_of_shapes_details = json.load(file_of_shapes)
            print("\nUsing the areas contained in the file '{}'.".format(shapes_filename))

    else:
        start_draw_time = time.time()
        list_of_shapes_details = drawing_images_program.main(len(csv_files), background_image_name, basewidth)
        full_draw_time = time.time() - start_draw_time

    # reshape the background
    background = reshape_background_image(background_image_name, basewidth)

    list_of_shape_arrays, list_of_centers, list_of_outlines_of_shape_arrays, background_colour_of_areas = turn_the_drawings_into_arrays(list_of_shapes_details, background.shape)

    """
    # rotate shape_arrays in list
    new_list_of_shape_arrays = []
    for shape_array in list_of_shape_arrays:
        new_shape_array = np.zeros((shape_array.shape[1], shape_array.shape[0], 3), np.uint8)
        cv2.transpose(shape_array, new_shape_array)
        cv2.flip(new_shape_array, 1, new_shape_array)

        new_list_of_shape_arrays.append(new_shape_array)
        plt.imshow(new_shape_array)
        plt.show()

    list_of_shape_arrays = new_list_of_shape_arrays
    """

    if len(csv_files) != len(list_of_shape_arrays):
        print("\nThe number of areas you drew and the number of csvs you supplied do not match. Please re-run the program drawing the same amount of areas on the image as csvs you supplied.")
        exit(0)

    if len(csv_files) < len(list_of_videos):
        print("\nThe number of videos in the folder supplied is greater than the number of csvs you supplied. Please re-run the program inputting the same amount or less videos than csvs you supply.")
        exit(0)

    # turn the data into a dataframe
    list_of_dfs = create_a_list_of_dataframes(folder, csv_files)

    if len(list_of_dfs) > 1:
        joined_df = join_dataframes(list_of_dfs)

    else:
        joined_df = list_of_dfs[0]

    frames_per_second = calculate_frames_per_sec(len(joined_df.Second), length_of_video)

    # create dictionary of events
    dictionary_of_events = {}
    if events_file:
        with open(events_file, 'r') as events:
            event = events.readline().strip()
            while event:
                tokens = event.split(" ")
                time = int(tokens[0])
                text = " ".join(tokens[1:])
                dictionary_of_events[time] = text
                event = events.readline().strip()

    event_duration_frame = int(frames_per_second * event_duration_sec)

    if folder_of_videos:
        # find out how many images will be on the lhs
        num_of_images_on_lhs = len(list_of_videos) // 2

        # create video caps
        read_videos = []
        for vid in list_of_videos:
            #v = VideoReaderQueue(vid, queue_size=32)
            #img, num_of_frames = v.load_video()
            img = cv2.VideoCapture(vid)
            read_videos.append(img)

    else:
        read_videos = []
        num_of_images_on_lhs = 0

    # create mapper
    mapper = matplotlib.cm.ScalarMappable(norm=matplotlib.colors.Normalize(vmin=min_sensor_value, vmax=max_sensor_value), cmap=colourmap_name)

    # get width of second image
    x_width_of_second_image = int(basewidth*proportion_of_basewidth_to_make_second_image) - (2 * second_image_border_width)

    # get the size of the colourmap image
    colourmap_width = background.shape[1] - (x_width_of_second_image + (2 * second_image_border_width))
    colourmap_height = int(background.shape[0] * proportion_of_image_height_to_make_colourmap_and_second_image) - (2 * colmap_border_width)

    import time
    start_colmap = time.time()
    # create the colourmap image
    border_colmap = create_colourmap((colourmap_height, colourmap_width, 3), mapper, 4)
    colmap_creation.append(time.time() - start_colmap)

    # height of second image
    y_height_of_second_image = colourmap_height

    # width of camera images beside heatmap
    width_of_left_and_right_images = int(basewidth * proportion_of_basewidth_to_make_the_left_and_right_images)

    # height of event text box
    height_of_text_box = int(proportion_of_image_height_to_make_text_box * (border_colmap.shape[0] + background.shape[0]))

    # get the height of the video
    if dictionary_of_events:
        height_of_video = (background.shape[0] + (2 * merged_image_border_width)) + border_colmap.shape[0] + (height_of_text_box + (2 * text_box_image_border_width))

    else:
        height_of_video = (background.shape[0] + (2 * merged_image_border_width)) + border_colmap.shape[0]

    # get the width of the video
    if read_videos:
        width_of_video = basewidth + (2 * merged_image_border_width) + (2 * width_of_left_and_right_images) + (2 * video_footage_border_width) + (2 * bar_plot_border_width)

    else:
        width_of_video = basewidth + (2 * merged_image_border_width) + width_of_left_and_right_images  + (2 * bar_plot_border_width)

    #print(height_of_video)
    #print(width_of_video)

    # create the writer to write the image to the video
    writer = cv2.VideoWriter(filename=output_filename, fourcc=cv2.VideoWriter_fourcc(*'mp4v'), fps=frames_per_second, frameSize=(width_of_video, height_of_video),
                             isColor=True)

    import time
    print("time before iteration through rows = {}".format(time.time() - start_time))
    new_start_time = time.time()

    # iterate through each row in the dataframe
    i = 0
    total = len(joined_df)
    for index, r in joined_df.iterrows():
        print("Row {}, {:.3g}% Done".format(str(i), (100 * i/total)))

        df_row = pd.DataFrame(r).T

        import time
        start_loop_through_sensor_value_columns_and_return_list_of_coloured_shape_images = time.time()

        # create the images of the shapes from the sensor values and areas
        list_of_coloured_shape_images, list_of_colours = loop_through_sensor_value_columns_and_return_list_of_coloured_shape_images(
            df_row.loc[:, df_row.columns.difference(["Second"])], list_of_shape_arrays, mapper, list_of_outlines_of_shape_arrays, background_colour_of_areas)

        loop_through_sensor_value_columns_and_return_list_of_coloured_shape_images_list.append(
            time.time() - start_loop_through_sensor_value_columns_and_return_list_of_coloured_shape_images)

        import time
        start_turn_all_the_different_images_into_one_image = time.time()

        # merge these images with the image of the colourmap and of the second
        merged_image = turn_all_the_different_images_into_one_image(list_of_coloured_shape_images, background, border_colmap, df_row, x_width_of_second_image, 
                                                                    y_height_of_second_image, width_of_left_and_right_images, dictionary_of_events, 
                                                                    event_duration_frame, names, list_of_centers, read_videos, num_of_images_on_lhs, 
                                                                    list_of_shapes_details, height_of_text_box, list_of_colours, background_colour_of_areas)

        turn_all_the_different_images_into_one_image_list.append(time.time() - start_turn_all_the_different_images_into_one_image)

        # write the images to the video
        # print(merged_image.shape)
        write_an_image_to_a_video(writer, merged_image)

        import time
        print(time.time() - new_start_time)
        new_start_time = time.time()

        #write_an_image_to_a_folder("folder of images/img" + str(i) + ".png", merged_image)
        i = i + 1

    writer.release()
    print("The video was written to the file with the name '" + output_filename + "'.")

    print("colourmap = {}".format(sum(colmap_creation)))
    print("loop_through_sensor_value_columns_and_return_list_of_coloured_shape_images = {}".format(
                                                                    sum(loop_through_sensor_value_columns_and_return_list_of_coloured_shape_images_list)))
    print("turn_all_the_different_images_into_one_image = {}".format(sum(turn_all_the_different_images_into_one_image_list)))
    print("other stuff = {}".format(sum(shapes_list)))
    print("merge_heatmap = {}".format(sum(merge_heatmap)))
    print("video stuff = {}".format(sum(video_list)))
    print("video_2_frames = {}".format(sum(video_2_frames_list)))
    print("read in videos = {}".format(sum(read_in_list)))
    print("merge = {}".format(sum(merge_list)))
    print("concat outside = {}".format(sum(concat_outside_list)))
    print("resize = {}".format(sum(resize_list)))
    print("bar = {}".format(sum(bar_list)))
    print("concat inside = {}".format(sum(concat_inside_list)))
    print("slice = {}".format(sum(slice_list)))
    print("border = {}".format(sum(border_list)))

    import time
    print("Time taken = {}".format(time.time() - start_time - full_draw_time - full_read_time))


if __name__ == '__main__':
    main()
