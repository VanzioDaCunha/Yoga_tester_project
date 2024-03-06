"""
# define all constants FOR tree pose
MODEL_INPUT = 99
CLASS_OUTPUT = 9
SEQUENCE_LENGTH = 5
LABELS = ['tp_end', 'tp_final', 'tp_inter1', 'tp_inter2', 'tp_inter3', 'tp_inter4',
          'tp_inter5', 'tp_inter6', 'treepose_start']

CSV_FILE_PATH = 'Dataset/Yoga/Tadasana/Annotations/'
CSV_FILE = '13.csv'

VIDEO_FILE_PATH = 'Dataset/Yoga/Tadasana/Videos/'
VIDEO_FILE = '13.mp4'

OUTPUT_FILE_PATH = 'Dataset/Yoga/Tadasana/Output/'
OUTPUT_FILE = 'output13.csv'
"""


# defined all the constants for Trikonasana
MODEL_INPUT = 99
CLASS_OUTPUT = 5
SEQUENCE_LENGTH = 5
LABELS = ['inter1', 'inter2', 'inter3', 'trikonasana_final', 'trikonasana_start']

CSV_FILE_PATH = 'Dataset/Yoga/Trikonasana/Annotations/'
CSV_FILE = '10.csv'

VIDEO_FILE_PATH = 'Dataset/Yoga/Trikonasana/Videos/'
VIDEO_FILE = '10.mp4'

OUTPUT_FILE_PATH = 'Dataset/Yoga/Trikonasana/Output/'
OUTPUT_FILE = 'output10.csv'
