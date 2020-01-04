def intersecting_detect(fir, sec, near = False, alt = False):

    a = False
    
    if alt == False:
        ax1 = fir[0]
        ay1 = fir[1]
        ax2 = fir[2]
        ay2 = fir[3]
        
    else:
        ax1 = fir[0]
        ay1 = fir[1]
        ax2 = ax1 + fir[2]
        ay2 = ay1 + fir[3]

    bx1 = sec[0]
    by1 = sec[1]
    bx2 = sec[2]
    by2 = sec[3]

    s1 = ( ax1>=bx1 and ax1<=bx2 ) or ( ax2>=bx1 and ax2<=bx2 )
    s2 = ( ay1>=by1 and ay1<=by2 ) or ( ay2>=by1 and ay2<=by2 )
    s3 = ( bx1>=ax1 and bx1<=ax2 ) or ( bx2>=ax1 and bx2<=ax2 )
    s4 = ( by1>=ay1 and by1<=ay2 ) or ( by2>=ay1 and by2<=ay2 )

    if ((s1 and s2) or (s3 and s4)) or ((s1 and s4) or (s3 and s2)):
        a = True
        return a
        
    elif near:
        if (s1 or s3) and (abs(ay1 - by2) < 35 or abs(by1 - ay2) < 35):
            a = True
            return a
        
        if (abs(ax1 - bx2) < 12 or abs(bx1 - ax2) < 12) and (abs(ay1 - by2) < 22 or abs(by1 - ay2) < 22): 
            a = True
            return a
    else:
        return False

def getIoU(gr, ct):
    
#   gr: X,Y,W,H
#   ct: X,Y,W,H
    
    
    x,y,w,h = ct
    ct = x,y,x+w,y+h
    #   ct: X,Y,X+W,Y+H
    
    if intersecting_detect(gr, ct, alt = True):

#               Пересечение

        left = max(gr[0], ct[0])
        top = min(gr[3] + gr[1], ct[3]);
        right = min(gr[2] + gr[0], ct[2]);
        bottom = max(gr[1], ct[1]);

        s0 = (right - left) * (top - bottom)

#                Объединение

        s1 = gr[2] * gr[3]
        s2 = (ct[2] - ct[0]) * (ct[3]- ct[1])

        s_12_union = s1 + s2 - s0

        return s0 / s_12_union
    return 0.0
             



