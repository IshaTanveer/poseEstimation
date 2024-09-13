import cv2
import mediapipe as mp
import time


class poseDetector():
    def __init__(self, mode = False, modelComplexity=1, smooth = True,
                 detectionConf = 0.5, trackingConf = 0.5):
        self.mode = mode
        self.modelComplexity = modelComplexity
        self.smooth = smooth
        self.detectionConf = detectionConf
        self.trackingConf = trackingConf

        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(
            static_image_mode=self.mode,
            model_complexity=self.modelComplexity,
            smooth_landmarks=self.smooth,
            min_detection_confidence=self.detectionConf,
            min_tracking_confidence=self.trackingConf
        )

    def findPose(self, img, draw = True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # the image is in BGR and the mediapipe uses RGB, so....
        self.results = self.pose.process(imgRGB)  # send the image tp model
        #print(self.results.pose_landmarks)
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks,
                                self.mpPose.POSE_CONNECTIONS)
        return img



    def findPosition(self, img, draw = True):
        landmarkList = []
        if self.results.pose_landmarks:
            for id, land_mark in enumerate(self.results.pose_landmarks.landmark):
                h, w, c,  =img.shape
                #print(id, land_mark)
                cx, cy = int(land_mark.x*w), int(land_mark.y*h)
                landmarkList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 10 , (255,0,0), cv2.FILLED)
        return landmarkList


#cap.release()  # Release the video capture object
#cv2.destroyAllWindows()  # Close all OpenCV windows


def main():
    cap = cv2.VideoCapture('videos/squat_test1.avi')  # capturing our video
    previousTime = 0
    detector = poseDetector()
    while True:
        success, img = cap.read()  # Read the next frame

        if not success:
           print("End of video")
           break  # Break the loop if the frame could not be read

        img = detector.findPose(img)
        landmarkList = detector.findPosition(img)
        print(landmarkList)
        
        # gonna change the frame-rate
        currentTime = time.time()
        fps = 1 / (currentTime - previousTime)
        previousTime = currentTime
        cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN,
                    3, (255, 0, 0), 3)

        cv2.imshow('image', img)  # Display the frame

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break  # Optionally, press 'q' to quit the video early

if __name__ == "__main__":
    main()