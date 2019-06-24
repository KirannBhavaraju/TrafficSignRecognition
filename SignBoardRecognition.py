import cv2
import numpy as np
import glob
from PIL import Image
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import pickle
import os

con_results = []
flag = 0
path = ""
image_points = []
max = []
all_images = []
all_labels = []
#test_files = glob.glob(r'C:\Users\Kirann Bhavaraju\Desktop\external_sources\GTSRB\Final_Test\Images\*.ppm')
test_files = ''
all_results = []
test_images = []

def kneighbors_classifier():
    global all_images,all_labels,test_files,all_results,con_results,path
    #xtr , xtes , ytr, ytes = train_test_split(all_images,all_labels, test_size=0.01)
    #kneighbors = KNeighborsClassifier(5)
    #kneighbors.fit(xtr,ytr)
    print('training complete')
    print(str(os.getcwd()))
    test_files = glob.glob(path)
    #with open('knei.pickle','wb') as file1:
    #    pickle.dump(kneighbors,file1)
    #model_score = kneighbors.score(xtes,ytes)

    #print(model_score)

    kneighbors = pickle.load(open('knei91.pickle','rb'))
    for file in test_files:
        image = Image.open(file)
        image = image.resize((32, 32), resample=Image.ANTIALIAS)
        test_images.append(np.array(image).flatten())
    for file in test_images:
        all_results.append(kneighbors.predict(file.reshape(1,-1)))

    i=0
    for result in all_results:
        if(result == '00003' or result == '00011' or result == '00012' or result == '00013'):
            print(test_files[i],'|','Relevant Sign Identified : ',result)
            #print('Sign Identified : ',result)
            i+=1
        else:
            con_results.append(test_files[i])
            print(test_files[i],'|','Not Relevant')
            #print('')
            i+=1

    print('To run another stride press 1')

def stride2():
    print('stride2')
    template_matching()
    feature_Extractor()


def support_vector_machine():

    global test_images,all_labels,all_images,all_results
    xtr, xtes, ytr, ytes = train_test_split(all_images, all_labels, test_size=0.4)
    classifier = svm.SVC(gamma=0.001, C=100)
    #images , labels = all_images[] , all_labels[]
    classifier.fit(xtr,ytr)

    for file in test_files:
        image = Image.open(file)
        image = image.resize((32, 32), resample=Image.ANTIALIAS)
        test_images.append(np.array(image).flatten())
    for file in test_images:
        all_results.append(classifier.predict(file.reshape(1,-1)))
    print(all_results)



def template_matching():
    '''
    img = cv2.imread("2.3.ppm")
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    template = cv2.imread("2.ppm", cv2.IMREAD_GRAYSCALE)
    #template = cv2.resize(template ,(32,32))
    #template = cv2.resize(template ,(0,0),fx=0.5,fy=0.5)

    # template dimensions
    print(template.shape)
    cv2.imshow("template", template)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # image dimensions
    print(img.shape)
    cv2.imshow("img", gray_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    '''
    template = cv2.imread("2.ppm", cv2.IMREAD_GRAYSCALE)
    template = np.array(cv2.resize(template ,(32,32))).flatten()
    for i in con_results:
        image = cv2.imread(str(i))
        result = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
        max = np.amax(result)
        print(max)
        print(result)
        cv2.imshow("result", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    # potential_location = np.where(result >= 0.7)
        index = 1.0
        while index > 0.7:
            index -= 0.1
        for point in zip(*np.where(result >= index)[::-1]):
            image_points = list(point)

    # drawing rectangles
        w, h = template.shape
        print(image_points)
        if len(image_points) == 0: print('image not found')
           # cv2.rectangle(img, point, (point[0] + w, point[1] + h), (0, 255, 0), 3)
           # cv2.imshow("image", i)
           # cv2.waitKey(0)

def feature_Extractor():
    img1 = cv2.imread("3.ppm",cv2.COLOR_BGR2GRAY)
    img2 = cv2.imread("3.6.ppm",cv2.COLOR_BGR2GRAY)
    #sift = cv2.xfeatures2d.SIFT_create()
    #key1 , desc1 = sift.detectAndCompute(img1, None)
    #key2 , desc2 = sift.detectAndCompute(img2, None)
    surf = cv2.xfeatures2d.SURF_create()
    key1, desc1 = surf.detectAndCompute(img1, None)
    key2, desc2 = surf.detectAndCompute(img2, None)
    for iter in desc1:
        print(iter)
    for iter in desc2:
        print(iter)
    print(np.size(desc1))
    print(np.size(desc2))
    bruteforce_matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck = True)
    matches = bruteforce_matcher.match(desc1,desc2)
    sorted_matches = sorted(matches,key = lambda x:x.distance)
    for m in sorted_matches:
        print(m.distance)
    matched_result = cv2.drawMatches(img1,key1,img2,key2,sorted_matches[:10],None)
    cv2.imshow("Matches",matched_result)
    cv2.imshow("img1",img1)
    cv2.imshow("img2",img2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    global all_images,all_labels,path
    #feature_Extractor()
    #template_matching()
    print('Enter the test file path : ')
    o = input()
    path = str(str(o) + "\*.ppm")
    all_files = glob.glob(r'C:\Users\Kirann Bhavaraju\Desktop\Computer Vision\Slides\Git\Final_Training/*/*.ppm')
    #for file in all_files:
    #    image = Image.open(file)
    #    image = image.resize((32,32), resample = Image.ANTIALIAS)
    #    all_images.append(np.array(image).flatten())
    #    all_labels.append(file.split('\\')[-2])
    kneighbors_classifier()
    #support_vector_machine()
    #print(len(all_images))
    #print(len(all_labels))



if __name__ == "__main__":
    main()

