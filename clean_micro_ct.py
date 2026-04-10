import numpy as np
from scipy import ndimage
import time

# ==========================================
# CONFIGURATION
# ==========================================
# For cleaning the original (refined) image
#INPUT_FILE = "LA3_d0_v1_uint8_unnormalised_338_338_283.raw"
#OUTPUT_FILE = "LA3_d0_v1_uint8_CLEANED_338_338_283.raw"

# Dimensions: (Z, Y, X) matching the C++ memory layout
#SHAPE = (283, 338, 338) 

# For cleaning the coarsened image
# Coarsening must be applied before cleaning (if using coarsening)
INPUT_FILE = "LA3_coarse_169_169_142.raw"
OUTPUT_FILE = "LA3_d0_v1_uint8_CLEANED_169_169_142.raw"

# Dimensions: (Z, Y, X) matching the C++ memory layout
SHAPE = (142, 169, 169) 

# THESE MUST MATCH THE C++ RAWType ENUMERATION!
VAL_AIR = 1
VAL_PORE = 2
VAL_ROCK = 3
VAL_SULPHIDE = 4
VAL_PRECIPITATE = 5

# FILTER MODE: 'agglomerate' (Surface showered) or 'core_sample' (Top boundary only)
MODE = 'agglomerate' 

def main():
    print(f"Loading {INPUT_FILE}...")
    t0 = time.time()
    
    # 1. Load the RAW file into a 3D NumPy array
    data = np.fromfile(INPUT_FILE, dtype=np.uint8).reshape(SHAPE)
    
    # 2. Define 6-connectivity (faces only, no diagonals) to match C++ Darcy flux
    struct_6_conn = ndimage.generate_binary_structure(3, 1)
    
    # 3. Extract the Pore mask
    print("Extracting Pore network and executing Morphological Opening...")
    pore_mask = (data == VAL_PORE)
    
    # 4. Destroy 1-voxel micro-throats (Erosion followed by Dilation)
    opened_pores = ndimage.binary_opening(pore_mask, structure=struct_6_conn)
    
    # 5. Connected Component Labeling on the opened pores
    print("Labeling independent pore islands...")
    labeled_pores, num_pore_islands = ndimage.label(opened_pores, structure=struct_6_conn)
    print(f"Found {num_pore_islands} independent pore islands.")

    valid_labels = set()

    # ==========================================
    # MODE A: AGGLOMERATE (Showered Surface)
    # ==========================================
    if MODE == 'agglomerate':
        print("Running Agglomerate Mode: Identifying Ambient Air Envelope...")
        air_mask = (data == VAL_AIR)
        labeled_air, num_air_islands = ndimage.label(air_mask, structure=struct_6_conn)
        
        # Identify the largest continuous block of air (Ambient Air)
        # bincount is highly optimized for finding the most frequent label
        counts = np.bincount(labeled_air.ravel())
        counts[0] = 0 # Ignore the background (0)
        ambient_air_label = counts.argmax()
        
        ambient_air_mask = (labeled_air == ambient_air_label)
        
        print("Dilating Ambient Air to find surface intersections...")
        # Dilate the ambient air by 1 voxel to create an intersection envelope
        surface_envelope = ndimage.binary_dilation(ambient_air_mask, structure=struct_6_conn)
        
        # Find which pore islands intersect with this surface envelope
        intersecting_voxels = surface_envelope & opened_pores
        valid_labels = np.unique(labeled_pores[intersecting_voxels])
        valid_labels = valid_labels[valid_labels > 0] # Remove background 0

    # ==========================================
    # MODE B: CORE SAMPLE (Top Boundary Only)
    # ==========================================
    elif MODE == 'core_sample':
        print("Running Core Sample Mode: Checking Top Boundary Intersections...")
        # Assuming X is the flow axis (based on C++ i < global.extent.i / 2 logic)
        # If your top boundary is actually Z=0, change this to labeled_pores[0, :, :]
        top_boundary_slice = labeled_pores[:, :, 0] 
        
        valid_labels = np.unique(top_boundary_slice)
        valid_labels = valid_labels[valid_labels > 0]
    
    else:
        raise ValueError("Invalid MODE specified.")

    print(f"Identified {len(valid_labels)} viable, surface-connected pore islands.")

    # 6. Reconstruct the final valid pore mask
    print("Executing Geometry Overwrite...")
    final_pore_mask = np.isin(labeled_pores, valid_labels)
    
    # 7. Overwrite invalid pores (isolated, dead-ends, micro-throats) with Rock
    pores_to_rock = pore_mask & (~final_pore_mask)
    voxels_changed = np.sum(pores_to_rock)
    
    data[pores_to_rock] = VAL_ROCK
    
    print(f"Successfully converted {voxels_changed} problematic pore voxels into Rock.")
    
    # 8. Save the cleaned binary file
    print(f"Saving optimized geometry to {OUTPUT_FILE}...")
    data.tofile(OUTPUT_FILE)
    
    t1 = time.time()
    print(f"Pre-processing complete in {t1 - t0:.2f} seconds.")

if __name__ == "__main__":
    main()