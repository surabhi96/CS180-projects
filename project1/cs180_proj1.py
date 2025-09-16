# CS180 (CS280A): Project 1 starter Python code

# these are just some suggested libraries
# instead of scikit-image you could use matplotlib and opencv to read, write, and display images

import numpy as np
import skimage as sk
import skimage.io as skio
from skimage.filters import sobel  
from tifffile import imread
import matplotlib.pyplot as plt

def shift(ref_im, ip_im, dx, dy):
    """Return overlapping crops when ip_im is shifted by (dx,dy)."""
    h, w = ref_im.shape[:2]
    min_x = max(0, dx)  
    max_x = min(w, w + dx)
    min_y = max(0, dy)
    max_y = min(h, h + dy)
    if max_x <= min_x or max_y <= min_y:
        return None, None
    ref_im_new = ref_im[min_y:max_y, min_x:max_x].copy()
    mov_y0 = min_y - dy
    mov_y1 = max_y - dy
    mov_x0 = min_x - dx
    mov_x1 = max_x - dx
    ip_im_new = ip_im[mov_y0:mov_y1, mov_x0:mov_x1].copy()
    return ref_im_new, ip_im_new

def final_crop(del_x, del_y, src, canvas_like):
    """Paste src into a canvas_like image at integer shift (del_x, del_y)."""
    out = np.array(canvas_like, copy=True)
    H, W = out.shape[:2]
    h, w = src.shape[:2]
    y_dst = max(0, del_y)
    x_dst = max(0, del_x)
    y_src = max(0, -del_y)
    x_src = max(0, -del_x)
    h_ov = min(H - y_dst, h - y_src)
    w_ov = min(W - x_dst, w - x_src)
    if h_ov > 0 and w_ov > 0:
        out[y_dst:y_dst+h_ov, x_dst:x_dst+w_ov, ...] = \
            src[y_src:y_src+h_ov, x_src:x_src+w_ov, ...]
    return out

def inner_crop(a, frac=0.08):
    """Drop a margin on all sides to ignore plate borders."""
    h, w = a.shape[:2]
    dy = int(h*frac)
    dx = int(w*frac)
    return a[dy:h-dy, dx:w-dx]

def ncc(A, B):
    """Normalized cross-correlation (higher is better)."""
    A = A.astype(np.float32)
    B = B.astype(np.float32)
    A = A - A.mean()
    B = B - B.mean()
    denom = (np.linalg.norm(A) * np.linalg.norm(B)) 
    if denom == 0:
        return 0
    return float((A*B).sum() / denom)

def ssd(A,B):
    diff = A.astype(np.float32) - B.astype(np.float32)
    return float(np.mean(diff * diff))

def estimate_translation(b,g,r,centroid_b=[0,0],centroid_r=[0,0], \
                         window_xy=[40,40],method='ssd'):
    # Use only edges 
    # g_edges = sobel(g)
    # b_edges = sobel(b)
    # r_edges = sobel(r)
    # Use pixel information
    g_edges = g
    b_edges = b
    r_edges = r

    max_x = window_xy[0]   
    max_y = window_xy[1] 
    # SSD: (dx, dy, score)
    best_b_ssd = (0, 0, np.inf)  
    best_r_ssd = (0, 0, np.inf)
    # NCC: (dx, dy, score)
    best_b_ncc = (0, 0, -1.0)  
    best_r_ncc = (0, 0, -1.0)
    best_b = best_b_ssd
    best_r = best_r_ssd

    for dx in range(centroid_b[0]-max_x, centroid_b[0]+max_x+1):
        for dy in range(centroid_b[1]-max_y, centroid_b[1]+max_y+1):
            G, B = shift(g_edges, b_edges, dx, dy)   
            if G is not None:
                s = ncc(inner_crop(G), inner_crop(B))
                s_ssd = ssd(inner_crop(G), inner_crop(B))
                if s > best_b_ncc[2]:
                    best_b_ncc = (dx, dy, s)
                if s_ssd < best_b_ssd[2]:
                    best_b_ssd = (dx, dy, s_ssd)

    for dx in range(centroid_r[0]-max_x, centroid_r[0]+max_x+1):
        for dy in range(centroid_r[1]-max_y, centroid_r[1]+max_y+1):   
            G, R = shift(g_edges, r_edges, dx, dy)
            if G is not None:
                s = ncc(inner_crop(G), inner_crop(R))
                s_ssd = ssd(inner_crop(G), inner_crop(R))
                if s > best_r_ncc[2]:
                    best_r_ncc = (dx, dy, s)
                if s_ssd < best_r_ssd[2]:
                    best_r_ssd = (dx, dy, s_ssd)

    if method == 'ncc':
        best_b = best_b_ncc
        best_r = best_r_ncc
    else:
        best_b = best_b_ssd
        best_r = best_r_ssd

    return best_b, best_r

def single_scale_align(b,g,r,method='ssd'):   
    best_b, best_r = estimate_translation(b,g,r)

    dx_b, dy_b, _ = best_b
    dx_r, dy_r, _ = best_r

    b_new = final_crop(dx_b, dy_b, b, np.full_like(b, 0.0))
    r_new = final_crop(dx_r, dy_r, r, np.full_like(r, 0.0))

    return np.dstack([r_new, g, b_new])

def get_gaussian_kernel(sigma, sz):
    x = np.arange(-sz, sz+1, dtype=np.float32)
    k = np.exp(-0.5*(x/sigma)**2)
    k /= k.sum()
    return k

def pyr_down(im, sigma, radius):
    img = im.astype(np.float32, copy=False)
    gaussian_kernel = get_gaussian_kernel(sigma, radius)
    r = len(gaussian_kernel) // 2
    H, W = img.shape
    # convolution in x direction 
    ph = np.pad(img, ((0, 0), (r, r)), mode='reflect')
    tmp = np.empty_like(img, dtype=np.float32)
    for x in range(W):
        window = ph[:, x:x+len(gaussian_kernel)]           
        tmp[:, x] = np.tensordot(window, gaussian_kernel, axes=([1], [0])) 
    # convolution in y direction 
    pv = np.pad(tmp, ((r, r), (0, 0)), mode='reflect')
    out = np.empty_like(tmp, dtype=np.float32)
    for y in range(H):
        window = pv[y:y+len(gaussian_kernel), :]            
        out[y, :] = np.tensordot(gaussian_kernel, window, axes=([0], [0]))  
    # remove every other row and column to return the downsized image
    return out[::2, ::2]

def multi_scale_align(b, g, r, pyr_levels=4, sigma=1.0, gaussian_sz=3, \
                      base_window=40, method='ncc'):
    # building pyramids
    b_pyr = [b.astype(np.float32, copy=False)]
    g_pyr = [g.astype(np.float32, copy=False)]
    r_pyr = [r.astype(np.float32, copy=False)]
    for _ in range(1, pyr_levels):
        b_pyr.append(pyr_down(b_pyr[-1], sigma, gaussian_sz))
        g_pyr.append(pyr_down(g_pyr[-1], sigma, gaussian_sz))
        r_pyr.append(pyr_down(r_pyr[-1], sigma, gaussian_sz))

    centroid_b = (0, 0)
    centroid_r = (0, 0)
    best_b = (0, 0, -1.0)  
    best_r = (0, 0, -1.0)

    for level in reversed(range(pyr_levels)):
        print('pyr level ', level)
        # shrinking window at finer levels 
        win = max(5, base_window // (2 ** (pyr_levels - 1 - level)))
        window_xy = [win, win]

        cx_b, cy_b = map(int, centroid_b)
        cx_r, cy_r = map(int, centroid_r)

        best_b, best_r = estimate_translation(
            b_pyr[level], g_pyr[level], r_pyr[level],
            centroid_b=[cx_b, cy_b], centroid_r=[cx_r, cy_r],
            window_xy=window_xy, method=method
        )

        # scaling dx, dy by 2 when moving to the next finer image
        centroid_b = (int(best_b[0]) * 2, int(best_b[1]) * 2)
        centroid_r = (int(best_r[0]) * 2, int(best_r[1]) * 2)

    dx_b, dy_b = int(best_b[0]), int(best_b[1])
    dx_r, dy_r = int(best_r[0]), int(best_r[1])
    # print(dx_b, dy_b)
    # print(dx_r,dy_r)
    b_new = final_crop(dx_b, dy_b, b.astype(np.float32, copy=False), np.zeros_like(b, dtype=np.float32))
    r_new = final_crop(dx_r, dy_r, r.astype(np.float32, copy=False), np.zeros_like(r, dtype=np.float32))

    return np.dstack([r_new, g, b_new])

if __name__ == '__main__':
    # image_path = 'cs180_proj1_data/cathedral.jpg'
    # Single scale metric for alignment (ssd/ncc)
    method = 'ncc'
    image_path = 'cs180_proj1_data/harvesters.tif'
    # Multi scale pyramid alignment 
    if image_path.endswith('.tif'):
        im = imread(image_path)
        im.astype(np.float32)
    # Single scale alignment 
    else:
        im = skio.imread(image_path)
        im = sk.img_as_float(im)
        
    # compute the height of each part (just 1/3 of total)
    height = np.floor(im.shape[0] / 3.0).astype(np.int32)

    b = im[:height, :]
    g = im[height:2*height, :]
    r = im[2*height:3*height, :]
    
    if image_path.endswith('.tif'):
        print('Performing multi-scale alignment')
        rgb_new = multi_scale_align(b,g,r)
        disp = rgb_new.astype(np.float32) / 65535.0   # normalise 16-bit to 0–1
        disp = np.clip(disp, 0, 1)
        plt.imshow(disp)
        plt.show()
    else:
        print('Performing single scale alignment')
        rgb_new = single_scale_align(b,g,r,method=method)
        skio.imshow(rgb_new)
        skio.show()
