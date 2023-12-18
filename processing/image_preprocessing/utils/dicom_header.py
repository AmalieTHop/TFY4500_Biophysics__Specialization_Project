from typing import Union, List

from processing.image_preprocessing.utils.header_entry import HeaderEntry
#from utils.header_entry import HeaderEntry


class DicomHeader:
    def __init__(self):
        # used to identify sub images
        self.acquisition_time = HeaderEntry('0008|0032',  self._str_to_time)
        self.echo_time = HeaderEntry('0018|0081', self._str_to_float)
        self.b_value = HeaderEntry('0019|0100c', self._str_to_float)

        # used to find position in image stack
        self.image_position_patient = HeaderEntry('0020|0032', self._str_to_float_list)
        self.image_orientation_patient = HeaderEntry('0020|0037', self._str_to_float_list)

    @staticmethod
    def _str_to_float(s: str) -> Union[float, List[float]]:
        """ Convert the string to float or array of floats """
        f = float(s)
        return f

    @staticmethod
    def _str_to_float_list(s: str) -> List[float]:
        float_list = [float(i) for i in s.split('\\')]
        return float_list

    @staticmethod
    def _str_to_time(s: str) -> float:
        """ Convert the string hhmmss.ss to number of seconds as float """
        hour = float(s[0:2]) * 60**2
        minute = float(s[2:4]) * 60
        second = float(s[4:])
        return hour + minute + second
