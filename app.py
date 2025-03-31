import gradio as gr
import tensorflow as tf
import cv2 as cv
import os
import shutil
from natsort import natsorted
from zipfile import ZipFile
def generate(model, images, labels, img_height, img_width, num_classes, progress=gr.Progress()):
    mdl = tf.keras.models.load_model(model)
    mdl.trainable = False
    if os.path.exists('images'):
        shutil.rmtree('images')
    if os.path.exists('adv_imgs'):
        shutil.rmtree('adv_imgs')
    if os.path.exists('adv_imgs.zip'):
        os.remove('adv_imgs.zip')
    with ZipFile(images, 'r') as zip:
        zip.extractall('images')
    images_folder = natsorted(os.listdir('images/images'))
    progress_step = 0
    total_steps = len(images_folder)
    progress((progress_step, total_steps))
    label_file = open(labels, 'r')
    shape = (img_height, img_width)
    if not os.path.exists('adv_imgs'):
        os.makedirs('adv_imgs')
    for image in images_folder:
        progress_step += 1
        progress((progress_step, total_steps))
        img = cv.imread(f'images/images/{image}')
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        img = tf.cast(img, tf.float32)
        img = tf.image.resize(img, shape)
        img = img / 255
        img = tf.reshape(img, (1, shape[0], shape[1], 3))
        label = int(label_file.readline())
        lbl = tf.one_hot(int(label), num_classes)
        lbl = tf.reshape(lbl, (1, num_classes))
        if num_classes == 1:
            loss_object = tf.keras.losses.BinaryCrossentropy()
        else:
            loss_object = tf.keras.losses.CategoricalCrossentropy()
        with tf.GradientTape() as tape:
            tape.watch(img)
            prediction = mdl(img)
            loss = loss_object(lbl, prediction)
        gradient = tape.gradient(loss, img)
        temp_img = img
        for eps in [0.05 * i for i in range(20)]:
            temp_img = img + eps * gradient
            temp_img = tf.clip_by_value(temp_img, 0, 1)
            pred = mdl.predict(temp_img, verbose=0).argmax()
            if pred != label:
                break
        temp_img = tf.clip_by_value(temp_img, 0, 1)
        cv.imwrite(f'adv_imgs/{image}', cv.cvtColor(temp_img.numpy()[0] * 255, cv.COLOR_BGR2RGB))
    label_file.close()
    with ZipFile('adv_imgs.zip', 'w') as zip:
        for file in os.listdir('adv_imgs'):
            zip.write(f'adv_imgs/{file}')
    return 'adv_imgs.zip'
upload_button_model = gr.File(label='Upload the Keras CNN Model')
upload_button_image = gr.File(label='Upload the Images Zip File')
upload_button_label = gr.File(label='Upload the Labels Text File')
image_height = gr.Number(label='Enter the Image Height')
image_width = gr.Number(label='Enter the Image Width')
num_classes = gr.Number(label='Enter the Number of Classes')
download_button = gr.File(label='Download the Adversarial Images')
long_desc = '''1. The Model should be a Keras Model of any format (keras or SavedModel or HDF5)
2. The Images should be in a Zip File 'images.zip' ordered in the same way as the Labels
3. The Labels should be in the same order as those of images and are written in a seperate line
'''
examples = [
    ['set1/model.h5', 'set1/images.zip', 'set1/labels.txt', 128, 128, 1],
    ['set2/model.keras', 'set2/images.zip', 'set2/labels.txt', 224, 224, 1000]
]
interface = gr.Interface(generate,
                         [upload_button_model, upload_button_image, upload_button_label, image_height, image_width, num_classes],
                         download_button,
                         title='Adversarial Images Generator for Keras CNN Classifiers',
                         description=long_desc,
                         allow_flagging='never',
                         examples=examples)
interface.launch(debug=True)