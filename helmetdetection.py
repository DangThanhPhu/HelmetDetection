from time import sleep
from darkflow.net.build import TFNet
import datetime
import cv2 as cv
import numpy as np

options = {
    'model': 'cfg/yolov2-tiny.cfg',
    'load': 'bin/yolov2-tiny_3000.weights',
    'threshold': 0.4,
}

# Cung cấp các tệp cấu hình và trọng lượng cho mô hình và tải mạng bằng chúng.
modelConfiguration = "cfg/yolov3-obj.cfg";
modelWeights = "bin/yolov3-obj_2400.weights";

datetime_object = datetime.datetime.now()
d = datetime_object.strftime("%m-%d-%Y, %H-%M")

tfnet = TFNet(options)
frame_count = 0             # được sử dụng trong vòng lặp chính nơi chúng tôi đang trích xuất hình ảnh, và sau đó để dự đoán thô (được gọi là quá trình đăng)
frame_count_out=0           # được sử dụng trong vòng lặp quá trình đăng, để nhận giá trị lớp không xác định.
# Khởi tạo các tham số
confThreshold = 0.5  # Ngưỡng tin cậy
nmsThreshold = 0.5  # Ngưỡng ngăn chặn không tối đa
inpWidth = 416       # Chiều rộng của hình ảnh đầu vào của mạng
inpHeight = 416      #Height của hình ảnh đầu vào của mạng


# Tải tên các lớp
classesFile = "bin/obj.names"

with open(classesFile, 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')



net = cv.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)
layersNames = net.getLayerNames()
# Lấy tên của các lớp đầu ra, tức là các lớp có đầu ra không được kết nối
output_layer = [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]

#gắp tham số mở camera
cap = cv.VideoCapture(0)

def draw_bounding_box(classId, conf, left, top, right, bottom, frame, classes):
    frame_count = 0
    cv.rectangle(frame, (left, top), (right, bottom), (255, 178, 50), 3)
    label = '%.2f' % conf
    if classes:
        assert(classId < len(classes))
        label = '%s:%s' % (classes[classId], label)

    label_size, base_line = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    top = max(top, label_size[1])

    label_name,label_conf = label.split(':')
    if label_name == 'Helmet':
        cv.rectangle(frame, (left, top - round(1.5*label_size[1])), (left + round(1.5*label_size[0]), top + base_line),
                     (255, 255, 255), cv.FILLED)
        cv.putText(frame, label, (left, top), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,0), 1)
        frame_count+=1


# Xóa các hộp giới hạn với độ tin cậy thấp bằng cách sử dụng phương pháp triệt tiêu không cực đại
def postprocess(frame, outs, conf_threshold, nms_threshold, classes,sampleNum):
    frameHeight = frame.shape[0]
    frame_width = frame.shape[1]
    classIds = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > conf_threshold:
                center_x = int(detection[0] * frame_width)
                center_y = int(detection[1] * frameHeight)
                width = int(detection[2] * frame_width)
                height = int(detection[3] * frameHeight)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                classIds.append(classId)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])
    # Thực hiện ngăn chặn không tối đa để loại bỏ các hộp chồng chéo dư thừa với xác xuất thấp hơn.
    indices = cv.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
    count_person=0 # để đếm các lớp trong vòng lặp này.
    for i in indices:
        i = i[0]
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]
        #draw_bounding_box(classIds[i], confidences[i], left, top, left + width, top + height, frame, classes)
        my_class='Helmet'
        unknown_class = classes[classId]

        if my_class == unknown_class:
            count_person += 1
    results = tfnet.return_predict(frame)
    if(count_person>0):
            label="Helmet"
    else:
            label="NoHelmet"
    colors = (0, 255, 0) if label == "Helmet" else (0, 0, 255)
    for color, result in zip(colors, results):
        tl = (result['topleft']['x'], result['topleft']['y'])
        br = (result['bottomright']['x'], result['bottomright']['y'])
        confidence = result['confidence']        
        text = '{}: {:.0f}%'.format(label, confidence * 100)
        frame = cv.rectangle(frame, tl, br, color, 5)
        frame = cv.putText(frame, text, tl, cv.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)        
        if(label=="NoHelmet"):
            sampleNum = sampleNum+1
            cv.imwrite("Nohelmet/NoHelmet-"'.' + str(d)+ "-" + str(sampleNum) + ".jpg",frame[result['topleft']['y']:result['bottomright']['y'],result['topleft']['x']:result['bottomright']['x']])
while True:
    ret, frame = cap.read()
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

    # Tạo một đốm màu 4D từ một khung hình.
    blob = cv.dnn.blobFromImage(frame, 1/255, (inpWidth, inpHeight), [0,0,0], 1, crop=False)

    # Đặt đầu vào cho mạng
    net.setInput(blob)

    # Chạy chuyển tiếp để nhận đầu ra của các lớp đầu ra
    outs = net.forward(output_layer)

    # Xóa các hộp giới hạn với độ tin cậy thấp
    postprocess(frame, outs, confThreshold, nmsThreshold, classes,0)
    cv.imshow('Helmet Detetection', frame)
    t, _ = net.getPerfProfile()
    
    label = 'Inference time: %.2f ms' % (t * 1000.0 / cv.getTickFrequency())
    
    cv.putText(frame, label, (0, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))
    
