import cv2
import numpy as np

def generate_ar_marker(size, text, border_size=20):
    # Create a blank white image
    marker = np.ones((size, size), dtype=np.uint8) * 0

    # Draw the thick black border
    marker[border_size:-border_size, border_size:-border_size] = 255

    # Define the text properties
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.5
    font_thickness = 2
    text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]

    # Calculate the position to center the text
    text_position = ((size - text_size[0]) // 2, (size + text_size[1]) // 2)

    # Put the text on the marker
    cv2.putText(marker, text, text_position, font, font_scale, 0, font_thickness)

    return marker

# Define the size of the marker and border size
marker_size = 200
border_size = 20

# Generate the AR marker with the word "AMRIT", a white background, and a thick black border
ar_marker = generate_ar_marker(marker_size, "AMRIT", border_size)

# Display the generated AR marker
cv2.imshow("AR Marker", ar_marker)
cv2.waitKey(0)
cv2.destroyAllWindows()
