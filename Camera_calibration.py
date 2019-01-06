#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 11 11:23:02 2018

@author: byakuya
"""

import cv2
import numpy as np  
from scipy import optimize

def supressor(bad_lines=[]):
 
    bad_lines.sort(key=lambda x:x[0])
    lines=[]
    for x in range(len(bad_lines)-1):
        if (bad_lines[x][0] not in [i for i in range(int(bad_lines[x+1][0]-9),int(bad_lines[x+1][0])+9)]):
            lines.append(bad_lines[x])
    lines.append(bad_lines[len(bad_lines)-1])
    return lines


def Homography(correspondences,world):
    #loop through correspondences and create assemble matrix
    aList = []
    corsp=np.asarray(correspondences)
    
    for corr in range(len(corsp)):
        if len(world[corr])==3:
            p1 = world[corr]
            p2 = np.append(corsp[corr], 1)
        else:
            a=np.asarray(world[corr])
            p1 = np.append(a, 1)
            p2 = np.append(corsp[corr], 1)

        a2 = [0, 0, 0, -p2.item(2) * p1.item(0), -p2.item(2) * p1.item(1), -p2.item(2) * p1.item(2),
              p2.item(1) * p1.item(0), p2.item(1) * p1.item(1), p2.item(1) * p1.item(2)]
        a1 = [-p2.item(2) * p1.item(0), -p2.item(2) * p1.item(1), -p2.item(2) * p1.item(2), 0, 0, 0,
              p2.item(0) * p1.item(0), p2.item(0) * p1.item(1), p2.item(0) * p1.item(2)]
        aList.append(a1)
        aList.append(a2)

    A = np.matrix(aList)

    #svd composition
    u, s, v = np.linalg.svd(A)
    hf=v[8]
    
    #reshape the min singular value into a 3 by 3 matrix
    h = np.reshape(hf, (3, 3))
    
    #normalize and now we have h
    h = (h/h.item(8))
    return h


def v_val(i, j, H):

    return np.array([
        H[0, i] * H[0, j],
        H[0, i] * H[1, j] + H[1, i] * H[0, j],
        H[1, i] * H[1, j],
        H[2, i] * H[0, j] + H[0, i] * H[2, j],
        H[2, i] * H[1, j] + H[1, i] * H[2, j],
        H[2, i] * H[2, j]
    ])



def corner(readpath='',writepath='',num=40):
    imagelist=[]
    c=0
    corner_list=[]
    for img in range(1,num+1):
        image=cv2.imread(readpath+str(img)+'.jpg')
        #image=cv2.imread('/Users/byakuya/work/computer vision/homework8/Files/Dataset1/Pic_34.jpg')
        image1=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        edge=cv2.Canny(image1,390,410)
        #cv2.imwrite('/Users/byakuya/work/computer vision/homework8/Files/canny.jpg',edge)
        line = cv2.HoughLines(edge,1,np.pi/180,55)
        
        hl=[]
        hv=[]
        for rho,theta in line[0]:
            
            if (abs(np.tan(theta))>1):
                hl.append([rho,theta])
            else:
                hv.append([rho,theta])
        horizontal=supressor(hl)
        horizontal=sorted(horizontal,key=lambda x:x[0]*np.sin(x[1]))
        vertical=supressor(hv)
        vertical=sorted(vertical,key=lambda x:x[0]*np.cos(x[1]))
        lines=horizontal+vertical
        line_hc=[]
        if (len(lines)==18):
            for rho,theta in lines:
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a*rho
                y0 = b*rho
                x1 = (x0 + 1000*(-b))
                y1 = (y0 + 1000*(a))
                x2 = (x0 - 1000*(-b))
                y2 = (y0 - 1000*(a))
                hc1=np.array([x1,y1,1])
                hc2=np.array([x2,y2,1])
                line_hc.append(np.cross(hc1,hc2))
                cv2.line(image,(int(x1),int(y1)),(int(x2),int(y2)),(0,0,255),1)
            line_horizontal=line_hc[:10]
            
            line_vertical=line_hc[10:]
            corner_points=[]
            test=[]
            count=0
            for x in line_horizontal:
                for y in line_vertical:
                    point=np.cross(x,y)
                    test.append(point)
                    point=point/point[2]
                    x_cord,y_cord=point[0],point[1]
                    corner_points.append([x_cord,y_cord])
                    cv2.circle(image,(int(x_cord),int(y_cord)),1,(200,0,0),2)  
                    cv2.putText(image,str(count),(int(x_cord),int(y_cord)),cv2.FONT_HERSHEY_SIMPLEX,.8,(255,0,0))
                    count+=1
                
            cv2.imwrite(writepath+str(c)+'.jpg',image)
            
            c+=1
            corner_list.append(corner_points)
            imagelist.append(img)
    return [corner_list,imagelist]


def find_intrinsic(corner_point,world_coordinates):
    v_list=[]

    for points  in corner_point:
        h=Homography(points,world_coordinates)
        #h=np.transpose(h)
        v1=v_val(0,1,h)
        v2=v_val(0,0,h)-v_val(1,1,h)
        v_list.append(v1)
        v_list.append(v2)
    vmat=np.matrix(v_list)
    u, s, v = np.linalg.svd(vmat)
    hf=v[5]
    #hf=hf/hf.item(5)
    x0=(hf.item(1)*hf.item(3)-hf.item(0)*hf.item(4))/(hf.item(0)*hf.item(2)-hf.item(1)**2)
    lam= hf.item(5)-(hf.item(3)**2+x0*(hf.item(1)*hf.item(3)-hf.item(0)*hf.item(4)))/hf.item(0)
    alphax=np.sqrt(lam/hf.item(0))
    alphay=np.sqrt((lam*hf.item(0))/(hf.item(0)*hf.item(2)-hf.item(1)**2))
    s=-(hf.item(1)*(alphax**2)*alphay)/lam
    y0=(s*x0/alphay)-(hf.item(3)*(alphax**2)/lam)
    k= np.array([[alphax, s, y0],[0,alphay,  x0],[0,0,1]])
    print 'intrinsic -' 
    print k
    return k


def find_extrinsic(intrinsic,corner_point,world_coordinates,world=0):
    if world==1:
        h=Homography(corner_point,world_coordinates)
    else:
        h=Homography(corner_point,world_coordinates)
        h=np.linalg.inv(h)
    inv_intrinsics = np.linalg.inv(intrinsic)
    
    h1=h[:,0]
    h2=h[:,1]
    h3=h[:,2]
    ld1 = 1 / np.linalg.norm(np.dot(inv_intrinsics, h1))
    ld2 = 1 / np.linalg.norm(np.dot(inv_intrinsics, h2))
    ld3 = (ld1 + ld2) / 2
    
    r0 = ld1 * np.dot(inv_intrinsics, h1)
    r1 = ld2 * np.dot(inv_intrinsics, h2)
    r2=np.cross(r0,r1,axis=0)
    
    t = np.array(ld3 * np.dot(inv_intrinsics, h3)).transpose()
    
    rt= np.append(r0,r1,axis=1)
    r=np.append(rt,r2,axis=1)
    rt=np.append(rt,np.transpose(t),axis=1)
    print 'extrinsic -'
    print rt
    #print r
    return rt


def cost(pflat,corr,world_coordinates):
    #pflat=np.array([p.item(0),p.item(1),p.item(2),p.item(3),p.item(4),p.item(5),p.item(6),p.item(7),p.item(8)])
    p=np.reshape(pflat,(3,3))
    transform=[]
    for poi in world_coordinates:
        if len(poi)!=3:
            poi=np.asarray(poi)
            poi=np.array([poi[0],poi[1],1])
        tran=np.dot(p,poi)
        tran=tran/tran[2]
        transform.append([tran[0],tran[1]])
    dis=[]
    for i in range(len(corr)):
        a=np.array(corr[i])
        b=np.array(transform[i])
        dis.append(np.linalg.norm(a-b))
    #error=sum(dis)
    return dis


def error(pflat,corr,world_coordinates):
    #pflat=np.array([p.item(0),p.item(1),p.item(2),p.item(3),p.item(4),p.item(5),p.item(6),p.item(7),p.item(8)])
    p=np.reshape(pflat,(3,3))
    
    transform=[]
    for poi in world_coordinates:
        if len(poi)!=3:
            poi=np.asarray(poi)
            poi=np.array([poi[0],poi[1],1])
        tran=np.dot(p,poi)
        tran=tran/tran[2]
        transform.append([tran[0],tran[1]])
    dis=[]
    for i in range(len(corr)):
        a=np.array(corr[i])
        b=np.array(transform[i])
        dis.append(np.linalg.norm(a-b))
    dis=np.asarray(dis)
    mean_error=np.mean(dis)
    variance=np.var(dis)
    return [mean_error, variance]
def reproj(pflat,pnew_3,readpath,image_number,corr,write):
    p=np.reshape(pflat,(3,3))
    transform=[]
    for poi in corr:
        if len(poi)!=3:
            poi=np.asarray(poi)
            poi=np.array([poi[0],poi[1],1])
        tran=np.dot(p,poi)
        tran=tran/tran[2]
        transform.append([tran[0],tran[1]])
    image=cv2.imread(readpath+str(image_number)+'.jpg')
    
    for x_cord,y_cord in transform:
        cv2.circle(image,(int(x_cord),int(y_cord)),1,(200,0,0),2)
    p=np.reshape(pnew_3.x,(3,3))
    transform=[]
    for poi in corr:
        if len(poi)!=3:
            poi=np.asarray(poi)
            poi=np.array([poi[0],poi[1],1])
        tran=np.dot(p,poi)
        tran=tran/tran[2]
        transform.append([tran[0],tran[1]])
    #image=cv2.imread('/Users/byakuya/work/computer vision/homework8/Files/Dataset2/'+str(2)+'.jpg')
    for x_cord,y_cord in transform:
        cv2.circle(image,(int(x_cord),int(y_cord)),1,(0,200,0),2)
    cv2.imwrite(write,image)




##################################################################################################################

world_coordinates=[]
for x in range(10):
    for y in range(8):
        world_coordinates.append([x,y,1])

#p=projection matrix
world_coordinates=np.asarray(world_coordinates)
corners,image_list=corner('Camera pattern source path','path for storing images with houghlines')
intrinsic=find_intrinsic(corners,world_coordinates)
extrinsic=find_extrinsic(intrinsic,corners[0],world_coordinates,1)
p=np.dot(intrinsic,extrinsic)
pflat=np.array([p.item(0),p.item(1),p.item(2),p.item(3),p.item(4),p.item(5),p.item(6),p.item(7),p.item(8)])
errorblm=error(pflat,corners[0],world_coordinates)
pnew=optimize.root(cost, pflat,args=(corners[0],world_coordinates), method='lm')
erroralm=error(pnew.x,corners[0],world_coordinates)
reproj(pflat,pnew,'Camera pattern source path',1,world_coordinates,'output path to show the reprojection')
#reproj contains projection from both the optimised and first calculated Projection matrix
print 'p before lm',np.reshape(pflat,(3,3))
print 'The mean error and variance before lm is=', errorblm
print 'P after Lm', np.reshape(pnew.x,(3,3))
print 'The mean error and variance after lm is=', erroralm
