# Functions to calculate orientation/direction selectivity metrics, and significance, from ephys neurons spiking data.

def get_DSI_OSI(fir_matrix, set_ORI, set_SF):

    thetas = np.deg2rad(set_ORI)

    # Collapse across SF and/or TF if needed
    if len(set_SF) > 1:
        # Collapse across first axis (SF), then keep only the first 3 TFs
        ori_bars = np.sum(np.sum(fir_matrix[1:], axis=1)[:3], axis=0)
    else:
        ori_bars = np.sum(fir_matrix[1:], axis=0)

    # Normalize baseline
    ori_bars = ori_bars - np.nanmin(ori_bars)

    # Identify preferred, opposite, orthogonal orientations
    pref_ori_index = np.argmax(ori_bars)
    n_oris = len(set_ORI)

    opposite_ori_index = (pref_ori_index + n_oris // 2) % n_oris
    ortho_ori_index1 = (pref_ori_index + n_oris // 4) % n_oris
    ortho_ori_index2 = (pref_ori_index + 3 * n_oris // 4) % n_oris

    # Extract relevant responses
    R_pref = ori_bars[pref_ori_index]
    R_opposite = ori_bars[opposite_ori_index]
    R_ortho1 = ori_bars[ortho_ori_index1]
    R_ortho2 = ori_bars[ortho_ori_index2]

    # Handle division by zero safely
    DSI = (R_pref - R_opposite) / R_pref if R_pref != 0 else np.nan
    OSI = ((R_pref + R_opposite) - (R_ortho1 + R_ortho2)) / (R_pref + R_opposite) if (R_pref + R_opposite) != 0 else np.nan

    return ori_bars, DSI, OSI


def get_gDSI_gOSI(fir_matrix,set_ORI,set_SF):
    
    
    thetas = np.deg2rad(set_ORI)
    
    if len(set_SF)>1:
        ori_bars = np.sum(np.sum(fir_matrix[1:], axis=1)[:3], axis=0)
    else:
        ori_bars = np.sum(fir_matrix[1:], axis=0)
    ori_bars=ori_bars-np.nanmin(ori_bars)
    vec_sum = np.sum(ori_bars * np.exp(1j * thetas))
    gDSI = np.abs(vec_sum) / np.sum(ori_bars)
    pref_ori = np.angle(vec_sum, deg=True) % 360  # Preferred direction


    vec_sum_osi = np.sum(ori_bars * np.exp(1j * 2 * thetas))
    gOSI = np.abs(vec_sum_osi) / np.sum(ori_bars)
    pref_ori = (0.5 * np.angle(vec_sum_osi, deg=True)) % 180

    return ori_bars,gDSI,gOSI

def shuffle_responses(arr):
    shuf_arr=np.copy(arr)
    np.random.shuffle(shuf_arr)
    return shuf_arr
