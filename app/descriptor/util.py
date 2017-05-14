import os

def getTestImages():
    test_images=[]
    for file in os.listdir("app/static/upload_img/"):
        test_images.append(file)
    return test_images;

if __name__ == '__main__':
    getTestImages()