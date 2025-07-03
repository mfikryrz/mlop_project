import sys
import traceback
import logging


class CustomException(Exception):
    def __init__(self, error_message, error_detail: sys):
        super().__init__(error_message)
        self.error_message = self.get_detailed_error_message(error_message, error_detail)

    def get_detailed_error_message(self, error_message, error_detail: sys):
        """
        Membuat pesan error yang menjelaskan file, line, dan pesan asli.
        """
        _, _, exc_tb = error_detail.exc_info()
        file_name = exc_tb.tb_frame.f_code.co_filename if exc_tb else "Unknown"
        line_number = exc_tb.tb_lineno if exc_tb else "Unknown"
        return f"""
        Error occurred in script: [{file_name}]
        At line number: [{line_number}]
        Error message: [{error_message}]
        """

    def __str__(self):
        return self.error_message

