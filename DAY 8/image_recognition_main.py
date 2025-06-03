import cv2

# Load the large (superset) and small (subset) images
superset_img = cv2.imread('walpaper.png')
subset_img = cv2.imread('image1.png')

# Convert both images to grayscale
superset_gray = cv2.cvtColor(superset_img, cv2.COLOR_BGR2GRAY)
subset_gray = cv2.cvtColor(subset_img, cv2.COLOR_BGR2GRAY)

# Perform template matching
result = cv2.matchTemplate(superset_gray, subset_gray, cv2.TM_CCOEFF_NORMED)

# Set a threshold for detection
threshold = 0.8
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

if max_val >= threshold:
    print("Subset image found in superset image.")
    top_left = max_loc
    h, w = subset_gray.shape

    # Draw rectangle around the match
    bottom_right = (top_left[0] + w, top_left[1] + h)
    cv2.rectangle(superset_img, top_left, bottom_right, (0, 255, 0), 2)

    # Show the result window
    cv2.imshow("Detected Image", superset_img)

    # Wait in a loop for ESC key to exit
    while True:
        if cv2.waitKey(1) & 0xFF == 27:  # ESC key code = 27
            break

    cv2.destroyAllWindows()
else:
    print("Subset image not found.")
