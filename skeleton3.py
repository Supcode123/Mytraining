
from PIL import Image
import numpy as np
np.random.seed(seed=2022)


# convert a RGB image to grayscale
# input (rgb): numpy array of shape (H, W, 3)
# output (gray): numpy array of shape (H, W)
def rgb2gray(rgb):
    ##############################################################################################
	#										IMPLEMENT											 #
	##############################################################################################
    r,g,b = rgb[:,:,0],rgb[:,:,1],rgb[:,:,2] #separate and get the values of 3 channels of each pixel
    gray=r*0.2126 + g*0.7152 + b*0.0722
    return gray


#load the data
# input (i0_path): path to the first image
# input (i1_path): path to the second image
# input (gt_path): path to the disparity image
# output (i_0): numpy array of shape (H, W, 3)
# output (i_1): numpy array of shape (H, W, 3)
# output (g_t): numpy array of shape (H, W)
def load_data(i0_path, i1_path, gt_path):

    ##############################################################################################
	#										IMPLEMENT											 #
	##############################################################################################

    i_0 = np.array(Image.open(i0_path), dtype=np.float64)
    i_0 /= (np.max(i_0) - np.min(i_0))     #normalization
    i_1 = np.array(Image.open(i1_path), dtype=np.float64)
    i_1 /= (np.max(i_1) - np.min(i_1))
    g_t = np.array(Image.open(gt_path), dtype=np.float64)
    assert np.amax(i_0) <=1    #make sure that image values are in the range [0, 1]
    assert np.amax(i_1) <=1
    assert np.amax(g_t) <=16    #make sure that image values are in the range [0, 16]
    return i_0, i_1, g_t


# image to the size of the non-zero elements of disparity map
# input (img): numpy array of shape (H, W)
# input (d): numpy array of shape (H, W)
# output (img_crop): numpy array of shape (H', W')
def crop_image(img, d):

    ##############################################################################################
	#										IMPLEMENT											 #
	##############################################################################################
    h,w = np.shape(d)
    for i in range(int(h/2)):     #retrieve rows in range from start to the central to find the upper boundry
        if np.array_equal(d[i,:],np.zeros(w)) and not (np.array_equal(d[i+1,:],np.zeros(w))):
            up=i+1
    for i in range(int(h/2),h):   #from central to the end to find the under boundry     
        if not (np.array_equal(d[i,:],np.zeros(w))) and np.array_equal(d[i+1,:],np.zeros(w)):
            down=i
    for j in range(int(w/2)):     #find the left boundry
        if  np.array_equal(d[:,j],np.zeros(h)) and not (np.array_equal(d[:,j+1],np.zeros(h))):
            left=j+1
    for j in range(int(w/2),w):   #find the right boundry     
        if not (np.array_equal(d[:,j],np.zeros(h))) and np.array_equal(d[:,j+1],np.zeros(h)):
            right=j
    img_crop=img[up:(down+1),left:(right+1)]        
    return img_crop


# shift all pixels of i1 by the value of the disparity map
# input (i_1): numpy array of shape (H, W)
# input (d): numpy array of shape (H, W)
# output (i_d): numpy array of shape (H, W)
def shift_disparity(i_1,d):

    ##############################################################################################
	#										IMPLEMENT											 #
	##############################################################################################
    i_d=i_1.copy()
    h,w=np.shape(i_1) # get the size of the picture
    for i in range(h):
        for j in range(w):
            if j-int(d[i,j])>=0:
                i_d[i,int(j-d[i,j])]=i_d[i,j]
    return i_d


# compute the negative log of the Gaussian likelihood
# input (i_0): numpy array of shape (H, W)
# input (i_1_d): numpy array of shape (H, W)
# input (mu): float
# input (sigma): float
# output (nll): numpy scalar of shape ()
def gaussian_nllh(i_0, i_1_d, mu, sigma):

    ##############################################################################################
	#										IMPLEMENT											 #
	##############################################################################################
    nll=0.0
    h,w=i_0.shape
    for i in range(h):
        for j in range(w):
            nll+=(i_0[i,j]-i_1_d[i,j]-mu)**2
    nll=nll/(2*sigma**2)+h*w*np.log((2*np.pi)**0.5*sigma)
    #print(nll)
    return nll

# compute the negative log of the Laplacian likelihood
# input (i_0): numpy array of shape (H, W)
# input (i_1_d): numpy array of shape (H, W)
# input (mu): float
# input (s): float
# output (nll): numpy scalar of shape ()
def laplacian_nllh(i_0, i_1_d, mu,s):

    ##############################################################################################
	#										IMPLEMENT											 #
	##############################################################################################
    nll=0.0
    h,w=i_0.shape
    for i in range(h):
        for j in range(w):
            nll+=np.abs(i_0[i,j]-i_1_d[i,j]-mu)
    nll=nll/s+h*w*np.log(2*s)     
    #print(nll)
    return nll


# replace p% of the image pixels with values from a normal distribution
# input (img): numpy array of shape (H, W)
# input (p): float
# output (img_noise): numpy array of shape (H, W)
def make_noise(img, p):
    ##############################################################################################
	#										IMPLEMENT											 #
	##############################################################################################
    h,w=img.shape
    img_noise=img.copy()
    num=int(np.round(h*w*p/100))
    x_index=np.random.choice(w,num)
    y_index=np.random.choice(h,num)
    Normal=np.random.normal(0.45,0.14,num)
    for i in range(num):
        img_noise[y_index[i],x_index[i]]=Normal[i]    #replace the image pixels
    return img_noise


# apply noise to i1_sh and return the values of the negative lok-likelihood for both likelihood models with mu, sigma, and s
# input (i0): numpy array of shape (H, W)
# input (i1_sh): numpy array of shape (H, W)
# input (noise): float
# input (mu): float
# input (sigma): float
# input (s): float
# output (gnllh) - gaussian negative log-likelihood: numpy scalar of shape ()
# output (lnllh) - laplacian negative log-likelihood: numpy scalar of shape ()
def get_nllh_for_corrupted(i_0, i_1_d, noise, mu, sigma, s):

    ##############################################################################################
	#										IMPLEMENT											 #
	##############################################################################################
    img=make_noise(i_1_d, noise)
    gnllh=gaussian_nllh(i_0, img, mu, sigma)
    lnllh=laplacian_nllh(i_0, img, mu,s)
    return gnllh, lnllh

# DO NOT CHANGE
def main():
     # load images
    i0, i1, gt = load_data('./data/i0.png', './data/i1.png', './data/gt.png')
    i0, i1 = rgb2gray(i0), rgb2gray(i1)

    # shift i1
    i1_sh = shift_disparity(i1, gt)
    
    # crop images
    i0 = crop_image(i0, gt)
    i1_sh = crop_image(i1_sh, gt)
    
    mu = 0.0
    sigma = 1.4
    s = 1.4
    for noise in [0.0, 14.0, 27.0]:

        gnllh, lnllh = get_nllh_for_corrupted(i0, i1_sh, noise, mu, sigma, s)

        #print(gnllh, lnllh)
if __name__ == "__main__":
    main()




