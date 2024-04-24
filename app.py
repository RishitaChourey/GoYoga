from flask import Flask, render_template, Response
import cv2
import pyautogui
import mediapipe as mp
import matplotlib as plt
import math

app = Flask(__name__)

camera_video = cv2.VideoCapture(0)
camera_video.set(3, 1280)
camera_video.set(4, 960)

# Initializing mediapipe pose class.
mp_pose = mp.solutions.pose

# Setting up the Pose function.
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.3, model_complexity=2)

# Initializing mediapipe drawing class, useful for annotation.
mp_drawing = mp.solutions.drawing_utils 

# Initialize Pose function for videos
pose_video = mp_pose.Pose(static_image_mode=False, model_complexity=1, min_detection_confidence=0.5)

def detectPose(image, pose,draw=False, display=True):
    
    # Create a copy of the input image.
    output_image = image.copy()
    
    # Convert the image from BGR into RGB format.
    imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Perform the Pose Detection.
    results = pose.process(imageRGB)
    
    # Retrieve the height and width of the input image.
    height, width, _ = image.shape
    
    # Initialize a list to store the detected landmarks.
    landmarks = []
    
    # Check if any landmarks are detected.
    if results.pose_landmarks:
    
        # Draw Pose landmarks on the output image.
        mp_drawing.draw_landmarks(image=output_image, landmark_list=results.pose_landmarks,connections=mp_pose.POSE_CONNECTIONS, landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255,255,255),thickness=3, circle_radius=3), connection_drawing_spec=mp_drawing.DrawingSpec(color=(49,125,237),thickness=2, circle_radius=2))
        
        # Iterate over the detected landmarks.
        for landmark in results.pose_landmarks.landmark:
            
            # Append the landmark into the list.
            landmarks.append((int(landmark.x * width), int(landmark.y * height),
                                  (landmark.z * width)))
    
    # Check if the original input image and the resultant image are specified to be displayed.
    if display:
    
        # Display the original input image and the resultant image.
        plt.figure(figsize=[22,22])
        plt.subplot(121);plt.imshow(image[:,:,::-1]);plt.title("Original Image");plt.axis('off');
        plt.subplot(122);plt.imshow(output_image[:,:,::-1]);plt.title("Output Image");plt.axis('off');
        
        # Also Plot the Pose landmarks in 3D.
        mp_drawing.plot_landmarks(results.pose_world_landmarks, mp_pose.POSE_CONNECTIONS)
        
    # Otherwise
    else:
        
        # Return the output image and the found landmarks.
        return output_image, landmarks, results

def checkLeftRight(image, results, draw=False, display=False):
    
    # Declare a variable to store the horizontal position (left, center, right) of the person.
    horizontal_position = None
    
    # Get the height and width of the image.
    height, width, _ = image.shape
    
    # Create a copy of the input image to write the horizontal position on.
    output_image = image.copy()
    
    left_x = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].x * width)
    right_x = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].x * width)

    left_y = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].y * height)
    right_y = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].y * height)
    
    # Check if the person is at left that is when both shoulder landmarks x-corrdinates
    # are less than or equal to the x-corrdinate of the center of the image.
    if ((left_x <= width//8 and left_y<=height//8) or (right_x <=width//8 and right_y<=height//8)):
        
        # Set the person's position to left.
        horizontal_position = 'Left button'

    # Check if the person is at right that is when both shoulder landmarks x-corrdinates
    # are greater than or equal to the x-corrdinate of the center of the image.
    elif ((right_x >= 7*width//8 and right_y<=height//8) or (left_x>=7*width//8 and left_y<=height//8)):
        
        # Set the person's position to right.
        horizontal_position = 'Right button'
        
    # Check if the person's horizontal position and a line at the center of the image is specified to be drawn.
    if draw:

        # Write the horizontal position of the person on the image. 
        cv2.putText(output_image, horizontal_position, (5, height - 10), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 3)
        
        # Draw a line at the center of the image.
        cv2.line(output_image, (width//8, 0), (width//8, height//8), (255, 255, 255), 2)
        cv2.line(output_image, (0, height//8), (width//8, height//8), (255, 255, 255), 2)
        cv2.line(output_image, (7*width//8, 0), (7*width//8, height//8), (255, 255, 255), 2)        
        cv2.line(output_image, (7*width//8, height//8), (width,height//8), (255, 255, 255), 2)

        cv2.rectangle(output_image, (7*width//8, 0) , (width, height//8),  (66, 148, 45), -1)
        cv2.putText(output_image, "NEXT", (7*width//8 + 10, height//16), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
    # Check if the output image is specified to be displayed.
    if display:

        # Display the output image.
        plt.figure(figsize=[10,10])
        plt.imshow(output_image[:,:,::-1]);plt.title("Output Image");plt.axis('off');
    
    # Otherwise
    else:
    
        # Return the output image and the person's horizontal position.
        return output_image, horizontal_position
    
def calculateAngle(landmark1, landmark2, landmark3):

    # Get the required landmarks coordinates.
    x1, y1, _ = landmark1
    x2, y2, _ = landmark2
    x3, y3, _ = landmark3

    # Calculate the angle between the three points
    angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))
    
    # Check if the angle is less than zero.
    if angle < 0:

        # Add 360 to the found angle.
        angle += 360
    
    # Return the calculated angle.
    return angle

def classifyPose(landmarks, output_image, display=False):
    
    # Initialize the label of the pose. It is not known at this stage.
    label = 'Correct your posture'

    # Specify the color (Red) with which the label will be written on the image.
    color = (0, 0, 255)
    
    # Calculate the required angles.
    #----------------------------------------------------------------------------------------------------------------
    
    # Get the angle between the right shoulder, elbow and wrist points. 
    right_elbow_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                                       landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value],
                                       landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value])   
    
    # Get the angle between the right hip, shoulder and elbow points. 
    right_shoulder_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value],
                                          landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                                          landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value])

    # Get the angle between the right hip, knee and ankle points 
    right_knee_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                      landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value],
                                      landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value])
    left_knee_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
                                      landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value],
                                      landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value])
    
    #right index, right wrist, right elbow
    right_wrist_angle=calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value],
                                      landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value],
                                      landmarks[mp_pose.PoseLandmark.RIGHT_INDEX.value])

    #right knee, hip, shoulder
    right_hip_angle=calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value],
                                      landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                      landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value])
    #pose 1
    if right_elbow_angle>50 and right_elbow_angle<90 and right_shoulder_angle>20 and right_shoulder_angle <50 and right_wrist_angle>120 and right_wrist_angle<160:
        label='1, 12. Pranamasana'
        
    #pose 2
    if right_elbow_angle>140 and right_elbow_angle< 170 and right_shoulder_angle>170 and right_shoulder_angle<210 and right_wrist_angle>130 and right_wrist_angle<170 :
        label='2, 11. Hasta Uttanasana'
    
    #pose 3
    if right_hip_angle>280 and right_elbow_angle>160 and right_elbow_angle< 210: 
        label='3, 10. Pada Hastasana'
    
    #pose 4
    if (right_elbow_angle>160 and right_elbow_angle<190 and right_shoulder_angle>30 and right_shoulder_angle<60 and right_hip_angle>160 and right_hip_angle<190 and right_knee_angle>210) or (right_elbow_angle>160 and right_elbow_angle<190 and right_shoulder_angle>30 and right_shoulder_angle<60 and right_hip_angle>160 and right_hip_angle<190 and left_knee_angle>210):
        label='4, 9. Ashwa Sanchalanasana'
    
    #pose 5
    if right_elbow_angle>160 and right_elbow_angle< 190 and  right_shoulder_angle>50 and  right_shoulder_angle<80 and right_knee_angle>160 and right_knee_angle<190 and right_wrist_angle>80 and right_wrist_angle<120 and right_hip_angle>160 and right_hip_angle<190:
        label='5. Dandasana'
        
    #pose 6
    if right_elbow_angle>30 and right_elbow_angle<60 and right_shoulder_angle>320 and right_knee_angle>190 and right_knee_angle<240 and right_hip_angle>210 and right_hip_angle<250:
        label= '6. Ashtanga Namaskara'
        
    #pose 7
    if right_elbow_angle>150 and right_elbow_angle<185 and right_shoulder_angle>10 and right_shoulder_angle<50 and right_knee_angle>175 and right_knee_angle< 200 and right_hip_angle>100 and right_hip_angle<140:
        label= '7. Bhujang Asana'
        
    #pose 8
    if right_elbow_angle>160 and right_elbow_angle<190 and right_shoulder_angle>150 and right_shoulder_angle<190 and right_knee_angle>150 and right_knee_angle< 190  and right_hip_angle>250 and right_hip_angle<310:
        label= '8. Adho mukha savasana'
        
    #pose 9 is the same as pose 4
    #pose 10 is the same as pose 3
    #pose 11 is the same as pose 2
    #pose 12 is the same as pose 1
    # Check if the pose is classified successfully
    if label != 'Unknown Pose':
        
        # Update the color (to green) with which the label will be written on the image.
        color = (0, 255, 0)  
    
    # Write the label on the output image. 
    cv2.putText(output_image, label, (10, 180),cv2.FONT_HERSHEY_PLAIN, 2, color, 2)
    
    # Check if the resultant image is specified to be displayed.
    if display:
    
        # Display the resultant image.
        plt.figure(figsize=[10,10])
        plt.imshow(output_image[:,:,::-1]);plt.title("Output Image");plt.axis('off');
        
    else:
        
        # Return the output image and the classified label.
        return output_image
    


def generate_frames():
    while True:
        success, frame = camera_video.read()
        if not success:
            continue
        
        # Flip the frame horizontally for natural (selfie-view) visualization.
        frame = cv2.flip(frame, 1)
        
        # Get the width and height of the frame
        frame_height, frame_width, _ =  frame.shape
        
        # Resize the frame while keeping the aspect ratio.
        frame = cv2.resize(frame, (int(frame_width * (640 / frame_height)), 640))
        print(frame.shape())
        # Perform Pose landmark detection.
        frame, landmarks, results = detectPose(frame, pose_video, display=False, draw=True)
        if landmarks:
            
            # Perform the Pose Classification.
            frame= classifyPose(landmarks, frame, display=False)
        
        if results.pose_landmarks:
            
            # Check the horizontal position of the person in the frame.
            frame, pressbutton = checkLeftRight(frame, results, draw=True)

        if(pressbutton=='Left button'):
            pyautogui.press('left')
        if(pressbutton=='Right button'):
            pyautogui.press('right')


        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)
