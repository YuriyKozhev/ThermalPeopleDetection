import cv2

colors = {
    'red': (0, 0, 255),
    'green': (0, 255, 0),
    'blue': (255, 0, 0),
    'white': (255, 255, 255)
}

def showRects(frame, rects, color = 'green', thickness = 2):
    for rect in rects:
        x,y,w,h = rect
        cv2.rectangle(frame, (x, y), (x+w, y+h), colors[color], thickness)
        
def showContours(frame, contours, color = 'green', thickness = 2):
    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)
        cv2.rectangle(frame, (x, y), (x+w, y+h), colors[color], thickness)
        
def showRect(frame, rect, color = 'green', thickness = 2):
    x,y,w,h = rect
    cv2.rectangle(frame, (x, y), (x+w, y+h), colors[color], thickness)