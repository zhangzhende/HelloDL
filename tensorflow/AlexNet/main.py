import myProject.tensorflowDemo.CatVsDog.Cat_vs_dog as catvsdog
import myProject.tensorflowDemo.CatVsDog.Cat_vs_dog2_with_batch_norm as carvsdog2
import os

"""
训练数据
"""


def training(times):
    catvsdog.train(times)
    """
    """


def training2(times):
    carvsdog2.train(times)


"""
测试数据
"""


def predict():
    imagefile = "./datatest/"
    cat = dog = 0
    for root, sub_folders, files in os.walk(imagefile):
        for name in files:
            imagefile = os.path.join(root, name)
            print(imagefile)

            if catvsdog.predict_class(imageFile=imagefile) == "cat":
                cat += 1
            else:
                dog += 1
            print("cat is:", cat, "    dog is :", dog)


def predict2():
    imagefile = "./datatest/"
    cat = dog = 0
    for root, sub_folders, files in os.walk(imagefile):
        for name in files:
            imagefile = os.path.join(root, name)
            print(imagefile)

            if catvsdog.predict_class(imageFile=imagefile) == "cat":
                cat += 1
            else:
                dog += 1
            print("cat is:", cat, "    dog is :", dog)


# training(10)
predict()
training2(10)
