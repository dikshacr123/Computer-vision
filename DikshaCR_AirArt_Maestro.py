import cv2
import numpy as np

x, y, k = 200, 200, -1
stp = 0
cap = cv2.VideoCapture(0)

def take_inp(event, x1, y1,f,p):
    global x, y, k
    if event == cv2.EVENT_LBUTTONDOWN:
        x, y, k = x1, y1, 1

cv2.namedWindow('enter_point')
cv2.setMouseCallback('enter_point', take_inp)

while cv2.waitKey(1) != 27:  #'Esc' key is pressed
    _, inp_img = cap.read()
    inp_img = cv2.flip(inp_img, 1)
    gray_inp_img = cv2.cvtColor(inp_img, cv2.COLOR_BGR2GRAY)
    
    cv2.putText(inp_img,'select the point',(10,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
    cv2.imshow('enter_point', inp_img)
    

    if k == 1:
        cv2.destroyAllWindows()
        break

old_pts = np.array([[x, y]], dtype=np.float32).reshape(1, 1, 2)
mask = np.zeros_like(inp_img)

while cv2.waitKey(30) & 0xff != 27:  #'Esc' key
    _, new_inp_img = cap.read()
    new_inp_img = cv2.flip(new_inp_img, 1)
    new_gray = cv2.cvtColor(new_inp_img, cv2.COLOR_BGR2GRAY)
    new_pts, _, _ = cv2.calcOpticalFlowPyrLK(gray_inp_img, new_gray, old_pts, None, maxLevel=1, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 15, 0.08))

    x, y = new_pts.ravel()

    key = cv2.waitKey(1) & 0xff
    if key == ord('q'):
        stp = 1
    elif key == ord('n'):
        mask = np.zeros_like(new_inp_img)

    if stp == 0:
        mask = cv2.line(mask, (int(old_pts[0, 0, 0]), int(old_pts[0, 0, 1])), (int(x), int(y)), (0, 0, 255), 6)

    cv2.circle(new_inp_img, (int(x), int(y)), 6, (0, 255, 0), -1)

    new_inp_img = cv2.addWeighted(mask, 0.3, new_inp_img, 0.7, 0)
    cv2.putText(mask, "'q' to stop 'n' to clear", (10, 50), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255))
    cv2.imshow('output', new_inp_img)
    cv2.imshow('result', mask)

    gray_inp_img = new_gray.copy()
    old_pts = new_pts.reshape(-1, 1, 2)

    if stp == 1:
        break

cv2.destroyAllWindows()
cap.release()
