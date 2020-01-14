import itertools
import math
import os
import random
import sys
import numpy as np
import cv2
import string

from img_utils import *
from jittering_methods import *

class FakePlateGenerator():
    def __init__(self, fake_resource_dir, plate_size):
        chinese_dir = fake_resource_dir + "/chinese/"
        number_dir = fake_resource_dir + "/numbers/" 
        letter_dir = fake_resource_dir + "/letters/" 
        plate_dir = fake_resource_dir + "/plate_background_use/"

        character_y_size = 113
        plate_y_size = 164

        self.dst_size = plate_size

        self.chinese = self.load_image(chinese_dir, character_y_size)
        self.numbers = self.load_image(number_dir, character_y_size)
        self.letters = self.load_image(letter_dir, character_y_size)

        self.numbers_and_letters = dict(self.numbers, **self.letters)

        #we only use blue plate here
        self.plates = self.load_image(plate_dir, plate_y_size)
    
        for i in self.plates.keys():
            self.plates[i] = cv2.cvtColor(self.plates[i], cv2.COLOR_BGR2BGRA)

        #take "苏A xxxxx" for example

        #position for "苏A"
        self.character_position_x_list_part_1 = [43, 111]  
        #position for "xxxxx"              
        self.character_position_x_list_part_2 = [205, 269, 330, 395, 464]

        self.chinese_index = {'京': '00', '津': '01', '冀': '02', '晋': '03', '蒙': '04',
                                '辽': '05', '吉': '06', '黑': '07', '沪': '08', '苏': '09',
                                '浙': '10', '皖': '11', '闽': '12', '赣': '13', '鲁': '14',
                                '豫': '15', '鄂': '16', '湘': '17', '粤': '18', '桂': '19',
                                '琼': '20', '渝': '21', '川': '22', '贵': '23', '云': '24',
                                '藏': '25', '陕': '26', '甘': '27', '青': '28', '宁': '29',
                                '新': '30', '港': '31', '澳': '32'}
    
    def get_radom_sample(self, data):
        keys = list(data.keys())
        i = random.randint(0, len(data) - 1)
        key = keys[i]
        value = data[key]

        #注意对矩阵的深拷贝
        return key, value.copy()

    def load_image(self, path, dst_y_size):
        img_list = {}
        current_path = sys.path[0]

        listfile = os.listdir(path)     

        for filename in listfile:
            img = cv2.imread(path + filename, -1)
            
            height, width = img.shape[:2]
            x_size = int(width*(dst_y_size/height))
            img_scaled = cv2.resize(img, (x_size, dst_y_size), interpolation = cv2.INTER_CUBIC)
            
            img_list[filename[:-4]] = img_scaled

        return img_list

    def add_character_to_plate(self, character, plate, x):
        h_plate, w_plate = plate.shape[:2]
        h_character, w_character = character.shape[:2]

        start_x = x - int(w_character/2)
        start_y = int((h_plate - h_character)/2)

        a_channel = cv2.split(character)[3]
        ret, mask = cv2.threshold(a_channel, 100, 255, cv2.THRESH_BINARY)

        overlay_img(character, plate, mask, start_x, start_y)

    def generate_one_plate(self):
        _, plate_img = self.get_radom_sample(self.plates)
        plate_name = ""
    
        character, img = self.get_radom_sample(self.chinese)
        self.add_character_to_plate(img, plate_img, self.character_position_x_list_part_1[0])
        plate_name += "%s"%(character,)

        character, img = self.get_radom_sample(self.letters)
        self.add_character_to_plate(img, plate_img, self.character_position_x_list_part_1[1])
        plate_name += "%s"%(character,)

        for i in range(5):
            character, img =  self.get_radom_sample(self.numbers_and_letters)
            self.add_character_to_plate(img, plate_img, self.character_position_x_list_part_2[i])
            plate_name += character

        #转换为RBG三通道
        plate_img = cv2.cvtColor(plate_img, cv2.COLOR_BGRA2BGR)
  
        #转换到目标大小
        plate_img = cv2.resize(plate_img, self.dst_size, interpolation = cv2.INTER_AREA)

        return plate_img, plate_name

    def generate_specific_plate(self, plate):
        _, plate_img = self.get_radom_sample(self.plates)
        plate_name = ''

        # 添加省份
        character = plate[0]
        img = self.chinese[self.chinese_index[character]]
        self.add_character_to_plate(img, plate_img, self.character_position_x_list_part_1[0])
        plate_name += '%s'%(character,)
        # 添加地区字母
        letter = plate[1]
        img = self.letters[letter.lower()]
        self.add_character_to_plate(img, plate_img, self.character_position_x_list_part_1[1])
        plate_name += '%s'%(letter,)
        # 添加后面字母数字组合
        for idx in range(2, len(plate)):
            character = plate[idx]
            img = self.numbers_and_letters[character.lower()]
            self.add_character_to_plate(img, plate_img, self.character_position_x_list_part_2[idx-2]) # 0 based
            plate_name += character
        
        # RGBA to RGB
        plate_img = cv2.cvtColor(plate_img, cv2.COLOR_BGRA2BGR)
        # resize
        plate_img = cv2.resize(plate_img, self.dst_size, interpolation=cv2.INTER_AREA)

        return plate_img, plate_name

chinese_letters = '京津冀晋蒙辽吉黑沪苏浙皖闽赣鲁豫鄂湘粤桂琼渝川贵云藏陕甘青宁新港澳'
alphabet = [l for l in string.ascii_uppercase if l not in 'IO']
def same_character_in_a_row():
    plate_name = ''
    # 添加省份
    plate_name += '%s'%(np.random.choice(list(chinese_letters), 1)[0],)
    # 添加地区字母
    plate_name += '%s'%(np.random.choice(alphabet, 1)[0],)
    for i in range(5):
        if np.random.random() <= 0.2:
            plate_name += plate_name[-1]
        else:
            plate_name += '%s'%np.random.choice(alphabet+list(range(10)), 1)[0]
    return plate_name
    

if __name__ == "__main__":
    fake_resource_dir  = sys.path[0] + "/fake_resource/" 
    output_dir = sys.path[0] + "/test_plate/"
    img_size = (100, 30)

    fake_plate_generator = FakePlateGenerator(fake_resource_dir, img_size)
    reset_folder(output_dir)

    for i in range(0, 100000):
        plate, plate_name = fake_plate_generator.generate_specific_plate(same_character_in_a_row())
        plate = jittering_color(plate)
        plate = add_noise(plate)
        plate = jittering_blur(plate)
        plate = jittering_scale(plate)

        #save_random_img(output_dir, plate)
        cv2.imwrite(os.path.join(output_dir, plate_name+'.jpg'), plate)