import cv2
from hik_camera.hik_camera.hik_camera import HikCamera

ips = HikCamera.get_all_ips()
ip = ips[0]

cam = HikCamera(ip)

with cam:
    # Set the camera's exposure settings to manual mode (off) and set the exposure time to 50 milliseconds
    cam["ExposureAuto"] = "Off"
    cam["ExposureTime"] = 20000

    # Create a VideoWriter object to save the video
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (640, 480))

    while True:
        # Capture a frame from the camera
        bgr = cam.robust_get_frame()

        # Convert the frame to grayscale
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        # Display the grayscale frame
        #cv2.imshow('Grayscale Frame', gray)

        cv2.imshow('color Frame', rgb)

        # Write the grayscale frame to the video file
        out.write(rgb)

        # Exit the loop if the 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release the VideoWriter object and close all windows
out.release()
cv2.destroyAllWindows()