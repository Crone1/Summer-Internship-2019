
import argparse
import os.path
import sys

import cv2
import yaml

from image import Image

from utils.input_utils import exit_if_false, exit_if_try_fails


# read the default configuration variables
with open("configs/default_configs.yaml", "r") as config_file:
    default_configs = yaml.load(config_file, Loader=yaml.FullLoader)
default_output_file_name = default_configs["drawing"]["output_file_name"]


class DrawingInputHandler:

    def __init__(self):
        if len(sys.argv) > 1:
            self._get_variables_from_command_line()
        else:
            self._get_variables_from_user()

    @staticmethod
    def _process_background_image_path(image_path):

        universal_criteria = "the path to the background image points to a valid image file."
        # check it's not empty
        exit_if_empty(image_path, error="You did not enter a valid path to an image.", criteria=universal_criteria)
        # ensure it has correct extension
        supported_extensions = [".bmp", ".dib", ".jpeg", ".jpg", ".jpe", ".jp2", ".png", ".webp", ".pbm", ".pgm", ".ppm", ".pxm", ".pnm", ".sr", ".ras", ".tiff", ".tif", ".exr", ".hdr", ".pic"]
        exit_if_empty(os.path.splitext(image_path)[-1] in supported_extensions, error="The file path entered does not have a supported file extension.", criteria=universal_criteria)
        # check the path exists
        exit_if_empty(os.path.exists(image_path), error="The file path entered does not point to an existing file.", criteria=universal_criteria)
        # check the file is an image file
        exit_if_try_fails(
            cv2.imread,
            args=[image_path],
            exception=AttributeError,
            error="The file path entered does not point to a valid image file.",
            criteria=universal_criteria
        )
        # read the image
        return Image.from_path(image_path)

    @staticmethod
    def _process_output_file_path(file_path):

        universal_criteria = "the path entered points to an existing folder."
        # check it's not empty
        exit_if_empty(file_path, error="You did not enter a valid path to a folder.", criteria=universal_criteria)
        # check the directories in the path exists
        folder_path = os.path.dirname(file_path)
        exit_if_empty(os.path.exists(folder_path), error="The folder structure in the file path entered does not exist.", criteria=universal_criteria)
        # check the path is a directory
        exit_if_empty(os.path.isdir(folder_path), error="The folder structure in the file path entered does not point to a folder.", criteria=universal_criteria)
        # check if something will be overwritten
        if os.path.isfile(file_path):
            print("WARNING: file contents at '{}' will be overwritten.".format(file_path))

        return file_path

    def _get_variables_from_command_line(self):

        parser = argparse.ArgumentParser(description="The variables that make this programme work")

        # background image name
        parser.add_argument(
            dest="background_image_path",
            nargs="?",
            type=str,
            help="The path to the background image.",
        )
        # output folder
        parser.add_argument(
            '-of',
            dest="output_file_path",
            default=default_output_file_name,
            nargs="?",
            type=str,
            required=False,
            help="The path to the file where the area details will be output to.",
        )

        args = parser.parse_args()

        # process data
        self.background = self._process_background_image_path(args.background_image_path)
        self.output_path = self._process_output_file_path(args.output_file_path)

    def _get_variables_from_user(self):

        # background image
        background_path = input("Please enter the path to the background image: ")
        self.background = self._process_background_image_path(background_path)

        # output file
        supplied_output_file_path = input(
            "Please enter the path where the area details will be output (Press 'Enter' for default): "
        )
        if supplied_output_file_path == "":
            supplied_output_file_path = default_output_file_name
            print("The output file has been set to the default - '{}'.".format(default_output_file_name))
        self.output_path = self._process_output_file_path(supplied_output_file_path)
