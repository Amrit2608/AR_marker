import numpy as np
from numpy.linalg import inv, norm
from scipy.linalg import sqrtm
import cv2



class RectangleDetection:
    def __init__(self):
        self.min_length = 100  # Example value, adjust as needed
        self.gaussian_kernel_size = 5  # Example value, adjust as needed
        self.gaussian_sigma = 1.0  # Example value, adjust as needed
        self.canny_param = [50, 150]  # Example value, adjust as needed
        self.axis_ratio = 0.7  # Example value, adjust as needed
        self.axis_length = 50  # Example value, adjust as needed
        self.error_threshold = 5.0  # Example value, adjust as needed
        self.draw_ellipse_center = True

    def detect(self, image):
        # Convert image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Gaussian blur
        blurred = cv2.GaussianBlur(gray, (self.gaussian_kernel_size, self.gaussian_kernel_size), self.gaussian_sigma)

        # Canny edge detection
        edges = cv2.Canny(blurred, self.canny_param[0], self.canny_param[1])

        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

        
        rectangles = []
        for contour in contours:
            epsilon = 0.03* cv2.arcLength(contour,True) 
            temp = cv2.approxPolyDP(contour,epsilon, closed = True)
            if len(temp) == 4 and cv2.isContourConvex(temp):
                rectangles.append(temp)

        

        # rectangle_vertices = []
        # for rectangle in rectangles:
            
        #     for vertex in rectangle:
                
        #         x, y = vertex[0]
        #         rectangle_vertices.append((x, y))
        #         # Draw circles on the vertices for visualization
        #         cv2.circle(image, (x, y), 5, (0, 0, 255), -1)

        #     # Draw the rectangle
        #     cv2.drawContours(image, [rectangle], -1, (0, 255, 0), 2)
        # print(rectangle_vertices)

        filtered_rectangles = []
        rectangle_vertices = []
        for rectangle in rectangles:
            # Calculate the length of each arm (assuming known scale)
            arm_lengths = [np.linalg.norm(rectangle[i] - rectangle[(i + 1) % 4]) for i in range(4)]
            
            # Filter rectangles based on arm length criterion (greater than 3 cm)
            if all(length * 0.1 > 3 for length in arm_lengths):
                filtered_rectangles.append(rectangle)
                for vertex in rectangle:
                    x, y = vertex[0]
                    rectangle_vertices.append((x, y))
                    # Draw circles on the vertices for visualization
                    cv2.circle(image, (x, y), 5, (0, 0, 255), -1)

                # Draw the rectangle
                cv2.drawContours(image, [rectangle], -1, (0, 255, 0), 2)

        print("Number of filtered rectangles:", len(filtered_rectangles))
        return rectangle_vertices, image

class PoseEstimation:
    def __init__(self, f, u0, v0):
        self.A = np.array([[f, 0.0, u0], [0.0, f, v0], [0.0, 0.0, 1.0]], dtype = "double")
        
        

    def compute_homography_matrix(self, dst_points, image):


        rect_detection = RectangleDetection()
        rectangle_vertices, image_n = rect_detection.detect(image)
        # print(rectangle_vertices)
        if rectangle_vertices is None:
            return None, image_n
        
        if len(rectangle_vertices) < 1:
            print("Error: Insufficient rectangles detected.")
            return None, image_n
        
        # print(rectangle_vertices.shape)
        src_points = rectangle_vertices
        # consiidering only one rectangle out of the detected rectangles 
        assert len(src_points[0:4]) == len(dst_points) == 4, "Number of points should be 4"

        X = np.zeros((8, 9))

        for i in range(4):
            u, v = src_points[0:4][i]
            x, y = dst_points[i]

            X[2*i] = [-u, -v, -1, 0, 0, 0, x*u, y*u, x]
            X[2*i + 1] = [0, 0, 0, -x, -y, -1, x*v, y*v, y]

        # Compute the eigenvalues and eigenvectors of A^T Atr
        val, V = np.linalg.eig(np.dot(X.T, X))

        # Find the index of the smallest eigenvalue
        min_eigenvalue_index = np.argmin(np.abs(val))

        # Extract the corresponding unit eigenvector
        h = V[:, min_eigenvalue_index]
        h /= h[-1]  # Normalize by the last element

        # Reshape h to a 3x3 matrix
        H = h.reshape((3, 3))

        return H, image_n



    def compute_camera_pose(self, H ):


        # H = np.array([[h11, h12, h13],
        #       [h21, h22, h23],
        #       [h31, h32, h33]])

        # Extract h1, h2, h3 from H
        if H is None:
            success = False
            return success, None, None, None
        
        h1 = H[:, 0]
        h2 = H[:, 1]
        h3 = H[:, 2]

        # Calculate lambda
        A_inv = np.linalg.inv(self.A)
        lambda_val = np.sqrt(np.dot(np.dot(h1.T, A_inv.T), A_inv @ h1))

        # Calculate rotation and translation
        
        r1 = (1 / lambda_val) * np.dot(A_inv, h1)
        r2 = (1 / lambda_val) * np.dot(A_inv, h2)
        t =  (1 / lambda_val) * np.dot(A_inv, h3)


        success = True
        return success, r1, r2, t
        


            

        



                 

        # # Draw ellipse centers
        # if self.draw_rectangle_center:
        #     for ellipse in ellipse_list:
        #         center = (int(ellipse.cx), int(ellipse.cy))
        #         cv2.circle(image, center, 3, (0, 255, 0), 1, 1)

        # return ellipse_list, image