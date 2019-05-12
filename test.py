import cv2
import csv
import os

training_directory = os.path.normpath(str(os.getcwd())+"\\"+"Final_Training")
new_list = []


def csv_file_extr(class_id):
    global new_list
    print("csv version used : " + csv.__version__)
    file_path = os.path.normpath(str(training_directory) + "\\" + str(class_id) + "\\" + "GT" + "-" + str(class_id) + ".csv")
    with open(file_path) as ground_truth:
       reader = csv.reader(ground_truth,delimiter=';')
       new_list = list(reader)

def main():
    global new_list
    print("CV2 version used : " + cv2.__version__)
    print("Enter Class Id")
    class_id = str(input())
    csv_file_extr(class_id)
    for iterator in new_list[1:len(new_list)]:
        print(iterator)
        image_path = str(training_directory) + "\\" + str(class_id) + "\\" + str(iterator[0])
        width = int(iterator[1])
        height = int(iterator[2])
        print("[height width]",[height,width])
        pt1 = (int(iterator[3]), int(iterator[4]))
        pt2 = (int(iterator[5]), int(iterator[6]))
        class_id1 = iterator[7]
        image_opener = cv2.imread(image_path)
        cv2.putText(image_opener, "c:"+str(class_id1) ,pt1 ,thickness=1 ,color=(0,255,0),
                    fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL,lineType=cv2.LINE_AA,fontScale=0.4,bottomLeftOrigin=False)
        cv2.rectangle(image_opener,pt1,pt2,(255,0,0))
        cv2.imshow('NewWindow',image_opener)
        cv2.resizeWindow('NewWindow', 250, 250)
        cv2.waitKey(0)
        cv2.destroyWindow('NewWindow')


if __name__ == "__main__":
    main()