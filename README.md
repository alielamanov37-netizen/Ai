# Ai
ai bot for diagnostic for tit and pigeon
import telebot
import os
import numpy as np
from keras.models import load_model
from PIL import Image, ImageOps

my_model = load_model("keras_model.h5", compile=False)
txt_labels = open("labels.txt", "r", encoding="utf-8").readlines()

bot = telebot.TeleBot("your token")

def get_class(model_path, labels_path, img_path):
    img = Image.open(img_path).convert("RGB")
    img = ImageOps.fit(img, (224, 224), Image.Resampling.LANCZOS)
    
    img_array = np.asarray(img)
    img_norm = (img_array.astype(np.float32) / 127.5) - 1
    
    final_data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    final_data[0] = img_norm
    
    res = my_model.predict(final_data)
    idx = np.argmax(res)
    
    name = txt_labels[idx].strip()
    conf = res[0][idx]
    
    return name, conf

@bot.message_handler(commands=['start'])
def welcome(message):
    bot.reply_to(message, "Привет, кидай фото, я попробую угадать что это!")

@bot.message_handler(content_types=['photo'])
def check_photo(message):
    bot.send_message(message.chat.id, "Ща гляну...")
    
    info = bot.get_file(message.photo[-1].file_id)
    file = bot.download_file(info.file_path)
    
    path = "test.jpg"
    with open(path, 'wb') as f:
        f.write(file)
    
    obj_name, percent = get_class("keras_model.h5", "labels.txt", path)
    
    ans = f"Похоже на: {obj_name}\nУверен на {round(percent * 100, 1)}%"
    bot.reply_to(message, ans)
    
    if os.path.exists(path):
        os.remove(path)

if __name__ == "__main__":
    bot.polling(none_stop=True)
