cv2.line(copy, (left_bottom_x, left_bottom_y), (left_top_x, left_top_y), color, width)
cv2.line(copy, (left_top_x, left_top_y), (right_top_x, right_top_y), color, width)
cv2.line(copy, (right_top_x, right_top_y), (right_bottom_x, right_bottom_y), color, width)
cv2.line(copy, (right_bottom_x, right_bottom_y), (left_bottom_x, left_bottom_y), color, width)