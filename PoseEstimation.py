import numpy as np
from numpy.linalg import inv, norm
from scipy.linalg import sqrtm
import cv2



class EllipseDetection:
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

        # Candidate ellipses
        candidate_list = []

        for contour in contours:
            if len(contour) < self.min_length:
                continue

            # Ellipse fitting
            ellipse_fitting = EllipseFitting()
            result = ellipse_fitting.fit(contour)

            if result and ellipse_fitting.error < self.error_threshold:
                ellipse = Ellips(ellipse_fitting.u)
                ellipse.set_points(contour)
                ellipse.compute_attributes()

                if ellipse.minor_length / ellipse.major_length >= self.axis_ratio and ellipse.major_length > self.axis_length:
                    candidate_list.append(ellipse)

        if len(candidate_list) <= 1:
            return None, image

        # Sort ellipses by major length
        candidate_list.sort(key=lambda x: x.major_length, reverse=True)

        # Merge ellipses with close centers
        use_index = np.ones(len(candidate_list), dtype=int)

        for i in range(len(candidate_list) - 1):
            if use_index[i] == 0:
                continue

            target = candidate_list[i]

            for j in range(i + 1, len(candidate_list)):
                if use_index[j] == 0:
                    continue

                reff = candidate_list[j]
                dx = target.cx - reff.cx
                dy = target.cy - reff.cy

                if dx ** 2 + dy ** 2 < 4:
                    use_index[j] = 0

        ellipse_list = [ellipse for ellipse, use in zip(candidate_list, use_index) if use == 1]

        # Draw ellipse centers
        if self.draw_ellipse_center:
            for ellipse in ellipse_list:
                center = (int(ellipse.cx), int(ellipse.cy))
                cv2.circle(image, center, 3, (0, 255, 0), 1, 1)

        return ellipse_list, image



# Assuming you have the EllipseDetection and CircularMarkerDetection classes from previous examples

class Ellips:
    def __init__(self, u):
        self.u = u
        self.cx = 0
        self.cy = 0
        self.major_length = 0.0
        self.minor_length = 0.0
        self.point_list = []
    
    def set_points(self, points):
        self.point_list = points.copy()
    
    def get_points(self):
        return self.point_list
    
    def compute_attributes(self):
    
        A = self.u[0]
        B = self.u[1]
        C = self.u[2]
        D = self.u[3]
        E = self.u[4]
        F = self.u[5]

        self.cx = -(C * D - B * E) / (A * C - B * B)
        self.cy = -(A * E - B * D) / (A * C - B * B)

        c = A * self.cx * self.cx + 2 * B * self.cx * self.cy + C * self.cy * self.cy - F

        A /= c
        B /= c
        C /= c

        part = np.sqrt((A + C) * (A + C) - 4 * (A * C - B * B))
        lambda1 = 0.5 * ((A + C) - part)
        lambda2 = 0.5 * ((A + C) + part)

        self.major_length = 1.0 / np.sqrt(np.abs(lambda1))
        self.minor_length = 1.0 / np.sqrt(np.abs(lambda2))
    

class EllipseFitting:
    def __init__(self):
        self.u = np.zeros(6)
        self.F0 = 1.0
        self.compute_error = True
        self.error = 0.0

    def fit(self, points):
        npoints = len(points)
        XI = np.zeros((6, npoints))

        for n in range(npoints):
            x, y = points[n][0]
            XI[0, n] = x * x
            XI[1, n] = 2.0 * x * y
            XI[2, n] = y * y
            XI[3, n] = 2.0 * x * self.F0
            XI[4, n] = 2.0 * y * self.F0
            XI[5, n] = self.F0 * self.F0

        M = np.zeros((6, 6))
        for n in range(npoints):
            M += np.outer(XI[:, n], XI[:, n])
        M /= npoints

        _, eigenvectors = np.linalg.eigh(M)
        self.u = eigenvectors[:, 0]
        self.u /= norm(self.u)

        result = (self.u[0] * self.u[2] - self.u[1] * self.u[1] > 0)

        self.error = 0.0
        if self.compute_error:
            for n in range(npoints):
                V0 = self.compute_v0(points[n])
                uxi = np.dot(self.u, XI[:, n])
                uV0u = np.dot(self.u, np.dot(V0, self.u))
                self.error += 0.5 * np.sqrt(uxi * uxi / uV0u)
            self.error /= npoints

        return result

    def compute_v0(self, point):
        x, y = point[0]
        xx = x * x
        xy = x * y
        yy = y * y
        xf0 = x * self.F0
        yf0 = y * self.F0
        f0f0 = self.F0 * self.F0

        V0 = np.array([
            [xx, xy, 0.0, xf0, 0.0, 0.0],
            [xy, xx + yy, xy, yf0, xf0, 0.0],
            [0.0, xy, yy, 0.0, yf0, 0.0],
            [xf0, yf0, 0.0, f0f0, 0.0, 0.0],
            [0.0, xf0, yf0, 0.0, f0f0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        ])

        return V0
    
class PoseEstimation:
    def __init__(self, f, u0, v0):
        self.A = np.array([[f, 0.0, u0], [0.0, f, v0], [0.0, 0.0, 1.0]], dtype = "double")
        self.dist_coeff = np.zeros((4, 1))

    def set_param(self, u):
        Q = np.array([[u[0], u[1], u[3]],
                    [u[1], u[2], u[4]],
                    [u[3], u[4], u[5]]])
        return Q

    # Function to calculate ellipse parameters
    def calculate_ellipse_parameters(self, image):
        

        # Ellipse detection
        ellipse_detection = EllipseDetection()
        detected_ellipses, image_n = ellipse_detection.detect(image)
        if detected_ellipses is None:
            return None, None, image_n
        if len(detected_ellipses) < 2:
            print("Error: Insufficient ellipses detected.")
            return None, None, image_n

        # Ellipse fitting for outer ellipse
        outer_ellipse_fitting = EllipseFitting()
        outer_points = detected_ellipses[0].get_points()
        success_outer = outer_ellipse_fitting.fit(outer_points)

        if not success_outer:
            print("Error: Outer ellipse fitting failed.")
            return None, None, image_n

        u_outer = outer_ellipse_fitting.u
        Q_outer = self.set_param(u_outer)
        # Ellipse fitting for inner ellipse
        inner_ellipse_fitting = EllipseFitting()
        inner_points = detected_ellipses[1].get_points()
        success_inner = inner_ellipse_fitting.fit(inner_points)

        if not success_inner:
            print("Error: Inner ellipse fitting failed.")
            return None, None, image_n

        u_inner = inner_ellipse_fitting.u
        Q_inner = self.set_param(u_inner)
        

        return Q_outer, Q_inner, image_n



    def ellipse_value(self, u, x, y):
        A, B, C, D, E, F = u
        return A * x * x + 2 * B * x * y + C * y * y + 2 * D * x + 2 * E * y + F

    def cubic_root(self, y):
        pre_x, x = -1, -1
        eps = 1.0e-8
        dx = 1.0

        while np.abs(dx) > eps:
            pre_x = x
            dx = (pre_x * pre_x * pre_x - y) / (3.0 * pre_x * pre_x)
            x = pre_x - dx

        return x

    def compute_normal(self, Q, radius, position):
        param = self.cubic_root(-np.linalg.det(Q))
        Q /= param

        eigval, eigvec = np.linalg.eigh(Q)
        u0 = eigvec[:, 0]
        u2 = eigvec[:, 2]

        v1 = np.sqrt((eigval[2] - eigval[1]) / (eigval[2] - eigval[0])) * u2
        v2 = np.sqrt((eigval[1] - eigval[0]) / (eigval[2] - eigval[0])) * u0

        v = (v1 + v2) / norm(v1 + v2)

        if (v[0] * v[2] > 0 and position == 0) or (v[0] * v[2] <= 0 and position == 1):
            v = (v1 - v2) / norm(v1 - v2)

        if v[2] > 0:
            v = -v

        return v, np.sqrt(eigval[1] * eigval[1] * eigval[1]) * radius

    def compute_camera_pose(self, ellipse_outer, radius_outer, ellipse_inner, position):
        normal_outer, dist_outer = self.compute_normal(ellipse_outer, radius_outer , position)
        Y = normal_outer

        dist = dist_outer

        Xc_outer = np.dot(inv(ellipse_outer), Y)
        Xc_outer /= Xc_outer[2]
        Rc_outer = -dist * Xc_outer / np.dot(Y, Xc_outer)

        Xc_inner = np.dot(inv(ellipse_inner), Y)
        Xc_inner /= Xc_inner[2]
        Rc_inner = -dist * Xc_inner / np.dot(Y, Xc_inner)

        T = Rc_outer

        I = np.eye(3)
        tmp = ((I - np.outer(Y, Y)) * (Rc_inner - Rc_outer))
        Z = tmp / norm(tmp) 
        # Z = (((I - np.outer(Y, Y)) * (Rc_inner - Rc_outer)).normalized())

        tmp = np.cross(Y, Z)
        X = tmp / norm(tmp)
        # X = np.cross(Y, Z).normalized()

        R = np.column_stack((X, Y, Z))

        R_ = np.zeros((3, 3))
        R_[0, 1] = 1.0
        R_[1, 0] = 1.0
        R_[2, 2] = -1.0

        R = np.dot(R_, R)
        t = np.dot(R_, T)
        Success = True
        return Success, R, t

    

# # Example usage
# image_path = 'your_image_path.jpg'
# ellipse_outer, ellipse_inner = calculate_ellipse_parameters(image_path)

# if ellipse_outer is not None and ellipse_inner is not None:


#     position = 1  # Update with the desired position (0 or 1)
#     R, t = compute_camera_param(ellipse_outer, 27.5, ellipse_inner, position)

#     print("Camera Parameters:")
#     print("R:", R)
#     print("t:", t)




