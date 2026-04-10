import numpy as np
from scipy import ndimage
import time

# ==========================================
# CONFIGURATION
# ==========================================
INPUT_FILE = "LA3_d0_v1_uint8_CLEANED_169_169_142.raw"
OUTPUT_FILE = "LA3_d0_v1_uint8_FRACTURED_WIDE_169_169_142.raw"

# Dimensions: (Z, Y, X)
SHAPE = (142, 169, 169)

# THESE MUST MATCH THE C++ RAWType ENUMERATION!
VAL_AIR = 1
VAL_PORE = 2
VAL_ROCK = 3
VAL_SULPHIDE = 4
VAL_PRECIPITATE = 5

def main():
    print(f"Loading cleaned geometry: {INPUT_FILE}...")
    t0 = time.time()
    
    # 1. Load the RAW file
    data = np.fromfile(INPUT_FILE, dtype=np.uint8).reshape(SHAPE)
    
    # 2. Extract Masks
    pore_mask = (data == VAL_PORE)
    sulphide_mask = (data == VAL_SULPHIDE)
    
    # 3. Identify Encapsulated Sulphides
    struct_6_conn = ndimage.generate_binary_structure(3, 1)
    exposed_pores = ndimage.binary_dilation(pore_mask, structure=struct_6_conn)
    
    encapsulated_sulphide_mask = sulphide_mask & (~exposed_pores)
    num_encapsulated = np.sum(encapsulated_sulphide_mask)
    
    print(f"Total Sulphide voxels: {np.sum(sulphide_mask)}")
    print(f"Encapsulated Sulphide voxels to liberate: {num_encapsulated}")
    
    if num_encapsulated == 0:
        print("All sulphides are already exposed. No fracturing needed.")
        return

    # 4. Euclidean Distance Transform (EDT)
    print("Calculating shortest paths through the solid rock matrix...")
    distances, indices = ndimage.distance_transform_edt(~pore_mask, return_indices=True)
    
    # 5. Extract coordinates of the trapped sulphides
    z_sulph, y_sulph, x_sulph = np.where(encapsulated_sulphide_mask)
    
    # 6. Drill the Micro-Fractures (Centerlines)
    print("Tracing micro-fracture centerlines...")
    
    # NEW: Create a blank boolean mask to hold JUST the new fractures
    fracture_mask = np.zeros(SHAPE, dtype=bool)
    
    for i in range(len(z_sulph)):
        sz, sy, sx = z_sulph[i], y_sulph[i], x_sulph[i]
        
        pz, py, px = indices[:, sz, sy, sx]
        dist = distances[sz, sy, sx]
        num_points = int(np.ceil(dist)) + 1
        
        if num_points > 1:
            lz = np.round(np.linspace(sz, pz, num_points)).astype(int)
            ly = np.round(np.linspace(sy, py, num_points)).astype(int)
            lx = np.round(np.linspace(sx, px, num_points)).astype(int)
            
            # Map the 1-voxel centerline
            for z, y, x in zip(lz, ly, lx):
                if data[z, y, x] == VAL_ROCK:
                    fracture_mask[z, y, x] = True

    # 7. Widen the Fractures (Dilation)
    print("Dilating fractures to prevent capillary shocks...")
    # Use a 3x3x3 full connectivity block to create robust, wide flow channels
    struct_26_conn = ndimage.generate_binary_structure(3, 3) 
    wide_fractures = ndimage.binary_dilation(fracture_mask, structure=struct_26_conn)
    
    # 8. Safely apply the wide fractures to the domain
    # We only overwrite ROCK voxels. This protects Air boundaries and Sulphide grains.
    valid_new_pores = wide_fractures & (data == VAL_ROCK)
    data[valid_new_pores] = VAL_PORE
    
    voxels_drilled = np.sum(valid_new_pores)
    print(f"Successfully converted {voxels_drilled} rock voxels into wide micro-fractures.")
    
    # 9. Save the fractured geometry
    print(f"Saving fractured geometry to {OUTPUT_FILE}...")
    data.tofile(OUTPUT_FILE)
    
    t1 = time.time()
    print(f"Micro-fracturing complete in {t1 - t0:.2f} seconds.")

if __name__ == "__main__":
    main()