#-----Header-----#
#This file tests the methods in Methods.py by applying matrices
#to a generated observation.
#The observation consists of a star emitting unpolarized light 
#and a planet reflecting Q+ polarized light
#In the first test the Q component of the stokes vector 
#is measured using the single difference method.
#In the second test IP is introduced and corrected using 
#the double difference method.
#
#--/--header--/--#

#-----Imports-----#
import numpy as np
import matplotlib.pyplot as plt
import Methods as Mt
#--/--Imports--/--#

#-----Parameters-----#

Width = 300 #Width(and height) of field in pixels
StarI0 = 100
StarR0 = 90
PlanetX = 30 #Position of planet compared to the middle
PlanetI0 = 6
PlanetR0 = 6
PL = 1 #Degree of polarization used in butterfly test
#--/--Parameters--/--#


#-----Methods-----#
#Calculates intensity of star as function of position
#Star intensity falls off exponentially 
#Star is placed at (0,0)
def StarIntensity(X,Y,I0,R0):
    R = np.sqrt(X**2+Y**2)
    return I0*np.exp(-R/R0)

#Calculates intensity of planet as function of position
#Planet intensity falls off exponentially
#Planet is placed at (dx,0)
#In this case Q = I
def PlanetIntensity(X,Y,I0,R0,dx):
    R = np.sqrt((X-dx)**2+Y**2) 
    return I0*np.exp(-R/R0)
    
#Simulates the observation field
#Returns Stokes field
def SimulateObservation(Width,StarI0,StarR0,PlanetI0,PlanetR0,PlanetX):
    SField = np.empty((4),dtype=np.ndarray) #Creates empty stokes field
    SField[0] = np.zeros((Width,Width))
    SField[1] = np.zeros((Width,Width))
    SField[2] = np.zeros((Width,Width))
    SField[3] = np.zeros((Width,Width))
    for x in range(Width):
        for y in range(Width):
            X = x-Width/2 #Position relative to the middle
            Y = y-Width/2
            #Sets the intensity of the field
            SField[0][x,y] = StarIntensity(X,Y,StarI0,StarR0)+PlanetIntensity(X,Y,PlanetI0,PlanetR0,PlanetX)
            #Sets the Q parameter of the field
            SField[1][x,y] = PlanetIntensity(X,Y,PlanetI0,PlanetR0,PlanetX)
    
    return SField

#Simulates different observation field
#Q and U change with the angle around the star
#This creates a butterfly pattertn when looking at Q or U
def SimulateButterfly(Width,StarI0,PL):
    SField = np.empty((4),dtype=np.ndarray) #Creates empty stokes field
    SField[0] = np.zeros((Width,Width))
    SField[1] = np.zeros((Width,Width))
    SField[2] = np.zeros((Width,Width))
    SField[3] = np.zeros((Width,Width))
    
    for x in range(Width):
        for y in range(Width):
            X = x-Width/2 #Position relative to the middle
            Y = y-Width/2
            
            if(X==0): #Finds the angle = arctan(Y/X)
                if(Y>=0):
                    Theta = np.pi
                if(Y<0):
                    Theta = -np.pi
            else:
                Theta = np.arctan(Y/X)                   
            
            #The formula needs this because we only get the absolute value
            if((Theta <= np.pi/4 and Theta >= -np.pi/4)):
                Sign = 1
            else:
                Sign = -1
            
            #See Boer et al. p3, we set Theta = AoLP
            I = StarIntensity(X,Y,StarI0,StarR0)
            Q = I*PL / np.sqrt(1+np.tan(2*Theta)**2)*Sign
            U = Q*np.tan(2*Theta)

            SField[0][x,y] = I
            SField[1][x,y] = Q
            SField[2][x,y] = U
            SField[3][x,y] = 0
            
    return SField
            
        
#--/--Methods--/--#

#-----Main-----#
SField = SimulateObservation(Width,StarI0,StarR0,PlanetI0,PlanetR0,PlanetX)

#-----ShowIntensity-----#
#Shows the intensity of the Stokes field
if(False):
    plt.figure()
    plt.title("Total intensity")
    plt.imshow(SField[0],cmap="gray")
    plt.colorbar()
#--/--ShowIntensity-----#

#-----SingleDifference-----#
#Calculates Q using single difference method
if(False):    
    QDifference = Mt.MeasureQ(SField) #This uses single difference method

    plt.figure() #Q found directly from stokes vector.
    plt.title("Actual Q image")
    plt.imshow(SField[1],cmap="gray")
    plt.colorbar()
    
    plt.figure() #Q found using single difference method, it's the same thing.
    plt.title("Q measured using double difference method")
    plt.imshow(QDifference,cmap="gray")
    plt.colorbar()

#--/--SingleDifference--/--#

#-----DoubleDifference-----#
#Measures Q again but with double difference method
#Now Ip is introduced
if(False):
    Ip = 0.1 #Amount of IP introduced, Ip < 1
    IpMatrix = Mt.IdentityMatrix() #The matrix that introduces IP
    IpMatrix[1,0] = Ip #Technically I should decrease so IpMatrix[0,0] < 1
    
    #Matrix that turns +Q into -Q
    RetarderMatrix = Mt.ApplyRotation(Mt.CreateRetarder(np.pi),0.25*np.pi)
       
    SFieldRot = np.dot(RetarderMatrix,SField)#Field rotated by +Pi
    SFieldIp = np.dot(IpMatrix,SField) #Field with IP
    SFieldRotIp = np.dot(IpMatrix,SFieldRot) #Rotated field with IP
    QCorrected = (Mt.MeasureQ(SFieldIp)-Mt.MeasureQ(SFieldRotIp))/2 #QField corrected for IP
    
    plt.figure() #Shows Q field if we had not corrected for IP(Single difference)
    plt.title("Q not corrected for Ip")
    plt.imshow(Mt.MeasureQ(SFieldIp),cmap="gray")
    plt.colorbar()
    
    plt.figure() #Shows Q field corrected for IP
    plt.title("Q corrected for Ip")
    plt.imshow(QCorrected,cmap="gray")
    plt.colorbar()

#--/--DoubleDifference--/--#

#-----ButterflyPattern-----#
#Simulates the butterfly pattern seen in the paper Boer et al. p7
if(True):
    
    SFieldButterfly = SimulateButterfly(Width,StarI0,PL)
    plt.figure()
    plt.title("QField of butterfly pattern")
    plt.imshow(SFieldButterfly[1],cmap="gray")
    plt.colorbar()
    
    plt.figure()
    plt.title("UField of butterfly pattern")
    plt.imshow(SFieldButterfly[2],cmap="gray")
    plt.colorbar()


#--/--ButterflyPattern--/--#

plt.show()
#--/--Main--/--#