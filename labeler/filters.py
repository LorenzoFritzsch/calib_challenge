import cv2

class filters:


    def bilateral_filter(old_gray, gray):
        return cv2.bilateralFilter(old_gray, 9, 75, 75), cv2.bilateralFilter(gray, 9, 75, 75)


    def gaussian_filter(old_gray, gray):
        old_gray_blurred = cv2.GaussianBlur(old_gray, (5, 5), 0)
        gray_blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        return old_gray_blurred, gray_blurred
