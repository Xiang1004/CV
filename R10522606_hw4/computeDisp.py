import numpy as np
import cv2.ximgproc as xip
import cv2


def computeDisp(Il, Ir, max_disp):
    h, w, ch = Il.shape
    labels = np.zeros((h, w), dtype=np.float32)
    Il = Il.astype(np.float32)
    Ir = Ir.astype(np.float32)

    Il_gray = cv2.cvtColor(Il, cv2.COLOR_BGR2GRAY)
    Ir_gray = cv2.cvtColor(Ir, cv2.COLOR_BGR2GRAY)

    # >>> Cost Computation
    # TODO: Compute matching cost
    # [Tips] Census cost = Local binary pattern -> Hamming distance
    # [Tips] Set costs of out-of-bound pixels = cost of closest valid pixel  
    # [Tips] Compute cost both "Il to Ir" and "Ir to Il" for later left-right consistency
    cost_l = np.zeros((max_disp, h, w), dtype=np.uint8)
    cost_r = np.zeros((max_disp, h, w), dtype=np.uint8)

    for d in range(max_disp):
        dist_l = np.zeros((h, w))
        dist_r = np.zeros((h, w))
        for y in range(h):
            for x in range(w):
                if x - d >= 0:
                    dist_l[y, x] = (Il_gray[y, x] - Ir_gray[y, x - d]) ** 2
                else:
                    dist_l[y, x] = (Il_gray[y, x] - Ir_gray[y, x]) ** 2
                if x + d < w:
                    dist_r[y, x] = (Ir_gray[y, x] - Il_gray[y, x + d]) ** 2
                else:
                    dist_r[y, x] = (Ir_gray[y, x] - Il_gray[y, x]) ** 2
        cost_l[d, ...] = dist_l
        cost_r[d, ...] = dist_r

    # >>> Cost Aggregation
    # TODO: Refine the cost according to nearby costs
    # [Tips] Joint bilateral filter (for the cost of each disparity)
        cost_l[d, ...] = xip.guidedFilter(Il, cost_l[d, ...], 8, 0.001 ** 2)
        cost_r[d, ...] = xip.guidedFilter(Ir, cost_r[d, ...], 8, 0.001 ** 2)
    # >>> Disparity Optimization
    # TODO: Determine disparity based on estimated cost.
    # [Tips] Winner-take-all
    winnerL = np.argmin(cost_l, axis=0)
    winnerR = np.argmin(cost_r, axis=0)
    
    # >>> Disparity Refinement
    # TODO: Do whatever to enhance the disparity map
    # [Tips] Left-right consistency check -> Hole filling -> Weighted median filtering
    # Left-right consistency check
    for y in range(h):
        for x in range(w):
            if x - winnerL[y, x] >= 0 and winnerL[y, x] == winnerR[y, x - winnerL[y, x]]:
                continue
            else:
                winnerL[y, x] = 0
    # Hole filling
    for y in range(h):
        for x in range(w):
            if winnerL[y, x] == 0:
                dx = 0
                while x - dx >= 0 and winnerL[y, x - dx] == 0:
                    dx += 1
                if x - dx < 0:
                    FL = max_disp
                else:
                    FL = winnerL[y, x - dx]
                dx = 0
                while x + dx <= w - 1 and winnerL[y, x + dx] == 0:
                    dx += 1
                if x + dx >= w:
                    FR = max_disp
                else:
                    FR = winnerL[y, x + dx]
                winnerL[y, x] = min(FL, FR)
    # Weighted median filtering
    labels = xip.weightedMedianFilter(Il.astype(np.uint8), winnerL.astype(np.uint8), 12)
    return labels.astype(np.uint8)
