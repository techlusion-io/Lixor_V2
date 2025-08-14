import numpy as np

def calculate_liquid_percentage(liquid_mask, vessel_mask, cork_mask=None):
    """
    Calculate liquid fill percentage as:
    (liquid area) / (vessel area - cork area) * 100

    Args:
        liquid_mask (np.ndarray): binary mask (uint8, 0/1 or 0/255) for liquid
        vessel_mask (np.ndarray): binary mask (same shape) for vessel
        cork_mask (np.ndarray or None): binary mask for cork, or None

    Returns:
        percent (float): liquid percentage (0-100, or 0 if any mask missing/invalid)
        details (dict): debug info (areas)
    """
    if liquid_mask is None or vessel_mask is None:
        return 0.0, {
            "liquid_area": 0,
            "vessel_area": 0,
            "cork_area": 0,
            "denominator": 0,
        }

    lmask = (liquid_mask > 0).astype(np.uint8)
    vmask = (vessel_mask > 0).astype(np.uint8)
    cmask = (cork_mask > 0).astype(np.uint8) if cork_mask is not None else None

    vessel_area = vmask.sum()
    liquid_area = lmask.sum()
    cork_area = cmask.sum() if cmask is not None else 0

    denominator = max(vessel_area - cork_area, 1) 
    percent = (liquid_area / denominator) * 100 if denominator > 0 else 0.0

    return percent, {
        "liquid_area": int(liquid_area),
        "vessel_area": int(vessel_area),
        "cork_area": int(cork_area),
        "denominator": int(denominator),
    }
