class Poi:
    def __init__(self,file_name,width,height,classname,x1,y1,x2,y2):
        self.file_name = file_name
        self.width = width
        self.height = height
        self.classname = classname
        if (x1 > x2):
            self.xmin = x2
            self.xmax = x1
        else:
            self.xmin = x1
            self.xmax = x2
        if (y1 > y2):
            self.ymax = y1
            self.ymin = y2
        else:
            self.ymax = y2
            self.ymin = y1

    def __str__(self):
        return "{},{},{},{},{},{},{},{}".format(self.file_name,self.width,self.height,self.classname,
                                                self.xmin,self.ymin,self.xmax,self.ymax)

#test = Poi('asdasd',123,123,'asda',12,220,2,440)
#print(test)