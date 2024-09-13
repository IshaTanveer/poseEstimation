import cv2
import mediapipe as mp
import time

mpDraw = mp.solutions.drawing_utils
mpPose = mp.solutions.pose
pose = mpPose.Pose()

cap = cv2.VideoCapture('videos/forward.mp4') #capturing our video
previousTime = 0
while True:
    success, img = cap.read()  # Read the next frame

    if not success:
        print("End of video")
        break  # Break the loop if the frame could not be read

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # the image is in BGR and the mediapipe uses RGB, so....
    results = pose.process(imgRGB)  # send the image tp model
    print(results.pose_landmarks)
    if results.pose_landmarks:
        mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
        for id, land_mark in enumerate(results.pose_landmarks.landmark):
            h, w, c,  =img.shape
            print(id, land_mark)
            cx, cy = int(land_mark.x*w), int(land_mark.y*h)
            cv2.circle(img, (cx, cy), 10 , (255,0,0), cv2.FILLED)



    #gonna change the frame-rate
    currentTime = time.time()
    fps = 1/(currentTime - previousTime)
    previousTime = currentTime
    cv2.putText(img, str(int(fps)), (70,50), cv2.FONT_HERSHEY_PLAIN,
                3, (255,0,0), 3)



    cv2.imshow('image', img)  # Display the frame

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break  # Optionally, press 'q' to quit the video early

cap.release()  # Release the video capture object
cv2.destroyAllWindows()  # Close all OpenCV windows
