import cv2
import face_recognition

# Get default webcam video stream
webcam_video_stream = cv2.VideoCapture(0)

# Array to store face locations
all_face_locations = []

# Load authorized face encodings
authorized_face_encodings = []

# Load and encode authorized faces
authorized_face_1 = face_recognition.load_image_file("C:\\Users\\ybbha\\Downloads\\Real_Time_Face_Recognition\\Data\\Person_1.jpg")
authorized_face_1_encoding = face_recognition.face_encodings(authorized_face_1)[0]
authorized_face_encodings.append(authorized_face_1_encoding)

authorized_face_2 = face_recognition.load_image_file("C:\\Users\\ybbha\\Downloads\\Real_Time_Face_Recognition\\Data\\Person_2.jpg")
authorized_face_2_encoding = face_recognition.face_encodings(authorized_face_2)[0]
authorized_face_encodings.append(authorized_face_2_encoding)

authorized_face_3 = face_recognition.load_image_file("C:\\Users\\ybbha\\Downloads\\Real_Time_Face_Recognition\\Data\\Person_3.jpg")
authorized_face_3_encoding = face_recognition.face_encodings(authorized_face_3)[0]
authorized_face_encodings.append(authorized_face_3_encoding)

# Define a function for the chat functionality
def chat_unauthorized_person_detected():

    from twilio.rest import Client

    account_sid = ' ' # enter twilio account sid
    auth_token = ' '  # enter twilio auth token
    client = Client(account_sid, auth_token)

    message = client.messages.create(
    from_='whatsapp:+14155238886',
    body='Unauthorized person detected at location',
    to='whatsapp:+ ' # enter whatsapp number
    )

    print(message.sid)

# Loop through each video frame until user exits
while True:
    ret, current_frame = webcam_video_stream.read()

    # Lets use a smaller version (0.25x) of the image for faster processing
    scale_factor = 4
    current_frame_small = cv2.resize(
        current_frame, (0, 0), fx=1/scale_factor, fy=1/scale_factor)

    # Find total number of faces
    all_face_locations = face_recognition.face_locations(
        current_frame_small, number_of_times_to_upsample=1, model='hog')

    # Let's print the location of each of the detected faces
    for index, current_face_location in enumerate(all_face_locations):
        # Splitting up tuple of face location
        top_pos, right_pos, bottom_pos, left_pos = current_face_location
        
        # Correct positions based on scale factor
        top_pos *= scale_factor
        right_pos *= scale_factor
        bottom_pos *= scale_factor
        left_pos *= scale_factor
        
        # Encode the current face
        current_face_encoding = face_recognition.face_encodings(current_frame, [(top_pos, right_pos, bottom_pos, left_pos)], num_jitters=1)
        
        # Check if the face is authorized
        authorized = False
        for face_encoding in authorized_face_encodings:
            if len(current_face_encoding) > 0:
                face_distance = face_recognition.face_distance([face_encoding], current_face_encoding[0])
                if face_distance < 0.6:  # You may need to adjust this threshold
                    authorized = True
                    break

        if authorized:
            print(f'Authorized person detected at location Top: {top_pos}, Left: {left_pos}, Bottom: {bottom_pos}, Right: {right_pos}')
            cv2.rectangle(current_frame, (left_pos, top_pos), (right_pos, bottom_pos), (0, 255, 0), 2)
        else:
            print(f'Unauthorized person detected at location Top: {top_pos}, Left: {left_pos}, Bottom: {bottom_pos}, Right: {right_pos}')
            cv2.rectangle(current_frame, (left_pos, top_pos), (right_pos, bottom_pos), (0, 0, 255), 2)
            # Call the chat function for unauthorized person detected
            chat_unauthorized_person_detected()

    # Show current face with rectangle
    cv2.imshow('Webcam Video', current_frame)
        
    # Press 'enter' key to exit loop
    if cv2.waitKey(1) == 13:
        break
        
webcam_video_stream.release()
cv2.destroyAllWindows()
