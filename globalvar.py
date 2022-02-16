global SET_SAMPLE_RATE
SET_SAMPLE_RATE = 8000

global SET_CUT
SET_CUT = 2

global pickle_filee

global dir_dataset
global define_labels
global no_classes

global EXT
EXT = 'wav'

global SYSTEM
SYSTEM = 'system/'

global FILE_SAVE
FILE_SAVE = 'system/saved.dat'

global FILE_H5
FILE_H5 = 'system/cnn_model.h5'

global FILE_TEMP
FILE_TEMP = 'system/temp_output.wav'


global START_RECORD
START_RECORD = 'system/start_record.wav'

global PROCESS
PROCESS = 'system/process.wav'

global RESULT
RESULT = 'system/result.wav'

global RESULT_FAILED
RESULT_FAILED = 'system/failed.wav'

global SHUTDOWN
SHUTDOWN = 'system/shutdown.wav'

global TEMP_RECOGNIZE
TEMP_RECOGNIZE = 'system/temp_recognize.mp3'

global EMPTY
EMPTY = ''

global DOT
DOT = '.'

global FAILED
FAILED = 'failed'

global LIMIT_RECOGNIZE
# ganti nilai LIMIT RECOGNIZE, jika selalu tidak teridentifikasi, maka besarkan nilainya, jika saat identifikasi suara
# kemudian suara dianggap selalu berhasil teridentifikasi, padahal tidak ada rekaman yang memadai, silakan kecilkan nilainya
# range perubahan angka ini mulai dari 0 - 100000
# kenapa harus diganti nilai LIMIT RECOGNIZE ini, karena saat merekam suara menggunakan microphone, kualitas microphone berbeda-beda,
# belum lagi kualitas chipset yang digunakan, belum lagi kondisi ruangan tempat perekaman suara, dll, makanya nilai LIMIT RECOGNIZE
# perlu disesuaikan apakah lebih sensitif atau sebaliknya
LIMIT_RECOGNIZE = 7000

global OUTPUT
OUTPUT = 'Not Recognized'